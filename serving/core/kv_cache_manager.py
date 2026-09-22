"""Tiered KV cache manager over per-tier :class:`BlockPool`s.

Port of vLLM v0.19.0's ``vllm/v1/core/kv_cache_manager.py`` plus the parts of
``single_type_kv_cache_manager.py`` that a full-attention, single-group model
needs, extended with a memory hierarchy.

The tiers share **one key space**. Block hashes are chained once at the NPU
block size (``hash(parent_hash, block_tokens)``, as vLLM does), and a lower tier
whose blocks are ``factor`` times larger keys on every ``factor``-th hash -- the
last fine hash of each coarse block. Because the chain is cumulative that single
hash identifies the whole prefix up to that point, so no concatenation and no
second hash function is needed. This is exactly
``offloading/scheduler.py::_get_block_hashes``.

Consequences of that shared key space, all deliberate:

* one walk over a request's hashes yields both the NPU hit and the lower-tier
  hit, instead of comparing two independent tree traversals
* the recoverable prefix is prefix-shaped: the walk stops at the first miss, so
  freeing a request's blocks in reverse order (tail first) keeps the head --
  and therefore the recoverable prefix -- alive longest
* only full blocks are indexed, so a resumed request always recomputes the
  remainder past its last complete block boundary: up to ``block_size`` tokens
  from the NPU, up to the coarse chunk from a lower tier

Space is not contents. A layer's HBM slot carries two independent facts --
whether space is reserved for it (``hbm_block``) and whether the KV sitting in
that space may be used (``valid``). Reserving a slot for a transfer that has
not landed yet must not read back as a cache hit, so ``valid`` is the single
gate on entering the prefix index: an invalid slot is never indexed, and what
is never indexed can never be hit. The bulk path (``allocate_slots``) fills its
slots in the same iteration and marks them valid immediately; only the
layer-granular path (``allocate_layer``) leaves them invalid until
``mark_layer_valid``.

Traffic model. Recall (lower tier -> NPU) is on the critical path and is
charged. The write-through (NPU -> lower tier) is not: vLLM's own
``OffloadingConnector`` defers it to the next engine step and runs it on a
dedicated stream precisely so it cannot delay token generation, so its bytes are
reported for energy accounting only. Eviction from the NPU costs nothing --
either the data is a finished request's cache, or a copy already exists below.
"""

from .block_pool import NONE_HASH, BlockPool, Device
from .utils import LOCAL_ATTENTION

###################################################################
class LayerKVState:
    """Storage reference for one layer of a logical token block.

    ``hbm_block`` is the reserved HBM slot, ``valid`` says whether the KV in it
    is usable. They are separate because a slot can be reserved before the data
    that belongs in it has arrived -- see the module docstring.
    """

    __slots__ = ("hbm_block", "valid")

    def __init__(self):
        self.hbm_block = None
        self.valid = False


class LogicalKVBlock:
    """A token range with independently tracked layer KV storage."""

    def __init__(self, token_block_index, num_layers):
        self.token_block_index = token_block_index
        self.layers = [LayerKVState() for _ in range(num_layers)]

    def __getitem__(self, layer_id):
        return self.layers[layer_id]

###################################################################


def cdiv(a, b):
    return -(-a // b)


def request_block_hashes(req, block_size):
    """Chained block hashes over ``input_hash_ids + output_hash_ids``.

    Computed once and cached on the request. The simulator knows the whole token
    sequence up front, so unlike vLLM -- which extends the chain each time an
    output token is appended -- the entire chain can be built in one pass. That
    is not future information leaking into scheduling: indexing into the chain
    is gated by ``num_computed_tokens`` at every call site, so a block only
    becomes insertable after its tokens have actually been computed.

    Returns an empty list for workloads that carry no token ids, which disables
    prefix caching for those requests without disabling allocation.
    """
    cached = req.block_hashes
    if cached is not None:
        return cached

    tokens = req.input_hash_ids
    if not tokens:
        req.block_hashes = []
        return req.block_hashes
    if req.output_hash_ids:
        tokens = tokens + req.output_hash_ids

    hashes = []
    parent = NONE_HASH
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = hash((parent, tuple(tokens[start:start + block_size])))
        hashes.append(parent)
    req.block_hashes = hashes
    return hashes


class TieredKVCacheManager:
    """Owns the per-request NPU block tables and the tier hierarchy.

    Args:
        block_size: NPU block size, and the granularity of the hash chain.
        npu_pool: the tier requests actually compute against.
        lower_pools: victim tiers in order (CPU, then CXL), each with a
            ``block_size`` that is a multiple of ``block_size``. Empty when
            ``--prefix-storage`` is not set, which is the default and matches
            plain vLLM: there is no host KV tier, so a preempted request
            recovers only what is still resident and recomputes the rest.
        enable_caching: mirrors ``--enable-prefix-caching``.
        num_layers: HBM allocations per logical token block. Each allocation
            occupies ``npu_pool.bytes_per_block``. Lower tiers still store
            complete, all-layer copies at their existing chunk size.
    """

    def __init__(self, block_size, npu_pool, lower_pools=(), enable_caching=True,
                 num_layers=1, layer_types=None, sliding_window=None):
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        self.num_layers = num_layers
        self.block_size = block_size
        self.npu_pool = npu_pool
        self.lower_pools = list(lower_pools)
        self.enable_caching = enable_caching and npu_pool.enable_caching

        for pool in self.lower_pools:
            if pool.block_size % block_size != 0:
                raise ValueError(
                    f"[TieredKVCacheManager] {pool.tier}: block_size "
                    f"{pool.block_size} must be a multiple of the NPU block_size "
                    f"{block_size} -- the lower tier keys on every Nth hash of the "
                    f"same chain, so a non-integer factor has no key to use"
                )

        # Attention type per layer, and the sliding window a local layer sees.
        # A local layer only ever reads the last ``sliding_window`` tokens, and
        # the window only moves forward, so KV that falls out of it is dead: it
        # is never read again and its slot can go back to the pool. That is what
        # makes a local layer's footprint constant in sequence length instead of
        # linear -- vLLM's own SlidingWindowManager, at block granularity.
        if layer_types is not None and len(layer_types) != num_layers:
            raise ValueError(
                f"[TieredKVCacheManager] layer_types has {len(layer_types)} "
                f"entries but the manager tracks {num_layers} layers"
            )
        if layer_types is not None and sliding_window is None:
            raise ValueError(
                "[TieredKVCacheManager] layer_types given without a "
                "sliding_window: a local layer needs to know how far back it "
                "may look"
            )
        self.sliding_window = sliding_window
        self.layer_is_local = [
            layer_types is not None and layer_types[i] == LOCAL_ATTENTION
            for i in range(num_layers)
        ]
        self._local_layers = [i for i, local in enumerate(self.layer_is_local) if local]
        self._has_local = bool(self._local_layers) and bool(sliding_window)

        # Request id -> logical token blocks, each holding per-layer HBM slots.
        self.req_to_blocks = {}
        # request id -> per-layer index of the first block this layer still
        # holds. The bulk path keeps each layer's slots a *suffix* of the block
        # list, so this one number is the whole held range and the accounting
        # below needs no scan over blocks x layers -- which at 62 layers and a
        # 32k sequence would be six figures of work per request per step.
        self.req_layer_start = {}
        # request id -> per-layer count of how many of those blocks are
        # already indexed. Per layer because free_layer/allocate_layer let one
        # layer's caching progress fall behind the others.
        self.num_cached_block = {}

        # Drained once per scheduling step by take_traffic().
        self._recall_bytes = 0
        self._write_through_bytes = 0

    # -------------------- helpers --------------------

    @staticmethod
    def _hbm_blocks(logical_blocks):
        """Flatten logical blocks into the pool's independently pinned slots."""
        return [state.hbm_block for block in logical_blocks
                for state in block.layers if state.hbm_block is not None]

    def _factor(self, pool):
        return pool.block_size // self.block_size

    def _first_live_block(self, num_tokens):
        """First block a local layer still needs when the sequence is
        ``num_tokens`` long. Everything below it fell out of the window.

        Block-granular, as vLLM's ``SlidingWindowManager.remove_skipped_blocks``
        is: a block survives while it holds even one token inside the window.
        """
        if not self._has_local:
            return 0
        return max(0, num_tokens - self.sliding_window) // self.block_size

    def _layer_first_needed(self, layer_id, first_live_block):
        return first_live_block if self.layer_is_local[layer_id] else 0

    def _trim_local_layers(self, req, num_tokens):
        """Release local layers' KV that has fallen out of the sliding window.

        Always safe and always profitable: those tokens cannot be read again by
        those layers, now or at any later step, because the window only moves
        forward. Called before the capacity check in :meth:`allocate_slots`, so
        the space it frees counts towards the allocation that follows.

        Returns the number of slots released.
        """
        if not self._has_local:
            return 0
        blocks = self.req_to_blocks.get(req.id)
        if not blocks:
            return 0
        first_live = self._first_live_block(num_tokens)
        if first_live <= 0:
            return 0
        starts = self.req_layer_start[req.id]
        counts = self.num_cached_block.get(req.id)
        freed = []
        limit = min(first_live, len(blocks))
        for layer_id in self._local_layers:
            start = starts[layer_id]
            if limit <= start:
                continue
            for index in range(start, limit):
                state = blocks[index].layers[layer_id]
                if state.hbm_block is not None:
                    freed.append(state.hbm_block)
                    state.hbm_block = None
                    state.valid = False
            starts[layer_id] = limit
            if counts is not None:
                # Those blocks are gone for good, so the layer's caching
                # progress steps over them rather than stalling on the gap.
                counts[layer_id] = max(counts[layer_id], limit)
        if freed:
            self.npu_pool.free_blocks(freed)
        return len(freed)

    def _coarse_hash(self, hashes, pool, coarse_idx):
        """Key for coarse block ``coarse_idx`` of ``pool``: the last fine hash
        it covers. ``offloading/scheduler.py::_get_block_hashes``."""
        factor = self._factor(pool)
        fine_idx = factor * coarse_idx + factor - 1
        if fine_idx >= len(hashes):
            return None
        return hashes[fine_idx]

    # -------------------- lookup --------------------

    def get_computed_blocks(self, req):
        """Longest recoverable prefix for ``req``, split by tier.

        Returns ``(npu_blocks, num_npu_hit_tokens, num_lower_hit_tokens)``.
        ``num_lower_hit_tokens`` are tokens whose KV has to be recalled into
        fresh NPU blocks; the recall bytes are charged in
        :meth:`allocate_slots`, once the allocation is known to succeed.
        """
        if not self.enable_caching:
            return [], 0, 0

        hashes = request_block_hashes(req, self.block_size)
        if not hashes:
            return [], 0, 0

        # vLLM caps the hit at num_tokens - 1: at least one token must be
        # recomputed to produce logits. Combined with block alignment this is
        # why a fully resident resume still costs up to one block.
        max_hit_len = req.num_tokens_reached - 1
        max_blocks = max_hit_len // self.block_size

        npu_blocks = []
        for index, block_hash in enumerate(hashes[:max_blocks]):
            slots = [self.npu_pool.get_cached_block((layer_id, block_hash))
                     for layer_id in range(self.num_layers)]
            if any(slot is None for slot in slots):
                # A token prefix is computed only when every layer is resident.
                break
            block = LogicalKVBlock(index, self.num_layers)
            for state, slot in zip(block.layers, slots):
                state.hbm_block = slot
                # An index hit is data that was computed and written: valid.
                state.valid = True
            npu_blocks.append(block)
        num_npu_hit = len(npu_blocks) * self.block_size

        num_lower_hit = self._lookup_lower(req, hashes, num_npu_hit)
        return npu_blocks, num_npu_hit, num_lower_hit

    def _lookup_lower(self, req, hashes, num_computed_tokens):
        """Extra tokens recoverable from a victim tier beyond the NPU hit.

        Mirrors ``get_num_new_matched_tokens``: round the NPU hit down to a
        coarse boundary, count consecutive coarse hits from there, and bail
        unless at least one whole coarse block is gained. The coarse block
        straddling the boundary is re-fetched -- transfers are block-granular.
        """
        for pool in self.lower_pools:
            coarse = pool.block_size
            num_coarse_blocks = req.num_tokens_reached // coarse
            full_block_tokens = coarse * num_coarse_blocks
            if full_block_tokens - num_computed_tokens < coarse:
                continue

            start_idx = num_computed_tokens // coarse
            hits = 0
            for idx in range(start_idx, num_coarse_blocks):
                block_hash = self._coarse_hash(hashes, pool, idx)
                if block_hash is None or pool.get_cached_block(block_hash) is None:
                    break
                hits += 1
            if hits == 0:
                continue

            gained = coarse * (start_idx + hits) - num_computed_tokens
            if gained < coarse:
                continue
            req.storage_hit_pool = pool
            req.storage_hit_blocks = hits
            return gained
        req.storage_hit_pool = None
        req.storage_hit_blocks = 0
        return 0

    # -------------------- allocation --------------------

    def _num_slots_to_add(self, num_known_blocks, num_required_blocks,
                          first_live_block):
        """Slots the pool must hand over so every layer covers its own range.

        A full layer needs blocks ``[0, required)``; a local one only
        ``[first_live, required)``, which is what caps its footprint at the
        window. ``num_known_blocks`` is what the request already holds as a
        suffix per layer, so the work is O(layers), not O(blocks x layers).
        """
        if not self._has_local:
            return max(num_required_blocks - num_known_blocks, 0) * self.num_layers
        total = 0
        for layer_id in range(self.num_layers):
            first = self._layer_first_needed(layer_id, first_live_block)
            total += max(0, num_required_blocks - max(num_known_blocks, first))
        return total

    def _num_blocks_to_allocate(self, req, num_tokens, new_computed_blocks):
        """NPU blocks the pool would have to hand over to give ``req`` slots for
        ``num_tokens`` in total. Port of
        ``single_type_kv_cache_manager.get_num_blocks_to_allocate``, extended so
        a local layer is only charged for its window.

        Net of what :meth:`_trim_local_layers` would release first, because
        ``allocate_slots`` trims before it checks -- dropping dead KV is always
        safe, so that space is genuinely available to this allocation.

        Slots that the explicit layer API (``free_layer``) punched out of the
        middle are *not* counted: the bulk path only ever fills the suffix it
        owns, and ``allocate_layer`` does its own all-or-nothing check for the
        rest.
        """
        req_blocks = self.req_to_blocks.get(req.id, ())
        num_required_blocks = cdiv(num_tokens, self.block_size)
        first_live = self._first_live_block(num_tokens)

        if req.id in self.num_cached_block:
            # Fast path: a running request has no new prefix hits to account for.
            add = self._num_slots_to_add(len(req_blocks), num_required_blocks,
                                         first_live)
            return add - self._num_slots_trim_would_free(req, first_live)
        num_known_blocks = len(new_computed_blocks) + len(req_blocks)
        add = self._num_slots_to_add(num_known_blocks, num_required_blocks,
                                     first_live)
        # A hit block sitting in the free list leaves it when touched, so it
        # consumes free capacity too and must be counted here.
        num_evictable = sum(1 for b in self._hbm_blocks(new_computed_blocks)
                            if b.ref_cnt == 0)
        return add + num_evictable

    def _num_slots_trim_would_free(self, req, first_live_block):
        """Slots :meth:`_trim_local_layers` is about to release, without doing it."""
        if not self._has_local or first_live_block <= 0:
            return 0
        blocks = self.req_to_blocks.get(req.id)
        starts = self.req_layer_start.get(req.id)
        if not blocks or starts is None:
            return 0
        limit = min(first_live_block, len(blocks))
        return sum(max(0, limit - starts[layer_id])
                   for layer_id in self._local_layers)

    def can_fit_full_sequence(self, req, new_computed_blocks=None,
                              num_new_computed_tokens=0, num_lower_tier_tokens=0):
        """Would ``req``'s whole sequence fit, not just this step's chunk?

        vLLM's admission gate (``kv_cache_manager.can_fit_full_sequence``, called
        from ``schedule()`` under ``scheduler_reserve_full_isl``, which defaults
        to True). Without it, chunked prefill admits a request on the strength of
        its first chunk, the request then grows past what the pool can hold, and
        something has to be preempted -- the "over-admission and KV cache
        thrashing" the vLLM config docstring names.

        Measured on the RTX 4090 replay, the absence of this gate was worth
        ~230 preemptions and ~145k recomputed tokens at *every* utilization from
        0.8 to 1.0, i.e. it is not a capacity effect.
        """
        new_computed_blocks = new_computed_blocks or []
        total_computed = (req.num_computed_tokens + num_new_computed_tokens
                          + num_lower_tier_tokens)
        # The length reached so far, which for a request being admitted for the
        # first time is its whole prompt. num_tokens_reached rather than
        # num_computed_tokens: preemption resets the latter to 0.
        full_num_tokens = max(req.num_tokens_reached, total_computed)
        return self._num_blocks_to_allocate(
            req, full_num_tokens, new_computed_blocks
        ) <= self.npu_pool.get_num_free_blocks()

    def allocate_slots(self, req, num_new_tokens, new_computed_blocks=None,
                       num_new_computed_tokens=0, num_lower_tier_tokens=0):
        """Give ``req`` slots for ``num_new_tokens`` more tokens.

        Returns the newly allocated blocks, or **None** when the pool cannot
        satisfy the request -- and in that case nothing has been mutated. That
        all-or-nothing property in the same call that can fail is the point of
        the block pool: the previous allocator decided whether to preempt
        against one estimate and charged memory later against another.
        """
        new_computed_blocks = new_computed_blocks or []

        num_local_computed = req.num_computed_tokens + num_new_computed_tokens
        total_computed = num_local_computed + num_lower_tier_tokens
        num_tokens_need_slot = total_computed + num_new_tokens

        if req.id in self.num_cached_block and (new_computed_blocks or num_lower_tier_tokens):
            # Running request: it cannot pick up new prefix hits mid-flight.
            raise RuntimeError(
                f"[TieredKVCacheManager] request {req.id} is running but was "
                f"given new computed blocks"
            )

        # Dropping KV that fell out of a local layer's window is safe whatever
        # happens next -- it can never be read again -- so it runs before the
        # check, and the space it returns counts towards this allocation.
        self._trim_local_layers(req, num_tokens_need_slot)

        if self._num_blocks_to_allocate(
                req, num_tokens_need_slot, new_computed_blocks
        ) > self.npu_pool.get_num_free_blocks():
            return None

        # ---- past this point the allocation is guaranteed to succeed ----
        req_blocks = self.req_to_blocks.setdefault(req.id, [])
        self.req_layer_start.setdefault(req.id, [0] * self.num_layers)

        if new_computed_blocks or num_lower_tier_tokens:
            self.npu_pool.touch(self._hbm_blocks(new_computed_blocks))
            req_blocks.extend(new_computed_blocks)
            self.num_cached_block[req.id] = [len(req_blocks)] * self.num_layers
            if num_lower_tier_tokens:
                self._charge_recall(req)

        new_blocks = self._allocate_new_blocks(req, num_tokens_need_slot)

        if self.enable_caching:
            # Cap at the reached length: only finalised tokens may be indexed.
            self.cache_blocks(
                req, min(total_computed + num_new_tokens, req.num_tokens_reached))
        return new_blocks

    def _allocate_new_blocks(self, req, num_tokens):
        """Extend ``req``'s block list, giving each layer only its own range.

        A local layer skips any new block that is already behind the window --
        which happens whenever a long prompt is admitted in one go: the head of
        it is out of window before the first token is ever generated.
        """
        req_blocks = self.req_to_blocks[req.id]
        num_known = len(req_blocks)
        num_required = cdiv(num_tokens, self.block_size)
        if num_required <= num_known:
            return []
        first_live = self._first_live_block(num_tokens)
        starts = self.req_layer_start[req.id]

        needed = self._num_slots_to_add(num_known, num_required, first_live)
        slots = iter(self.npu_pool.get_new_blocks(needed))
        new_blocks = []
        for index in range(num_known, num_required):
            block = LogicalKVBlock(index, self.num_layers)
            for layer_id, state in enumerate(block.layers):
                if index < self._layer_first_needed(layer_id, first_live):
                    continue                # already behind this layer's window
                state.hbm_block = next(slots)
                # The bulk path allocates for tokens this iteration computes,
                # so the slot is filled by the time anything can look at it.
                # allocate_layer, which reserves ahead of a transfer, does not
                # do this.
                state.valid = True
            new_blocks.append(block)
        req_blocks.extend(new_blocks)
        if self._has_local and first_live > 0:
            # New blocks below the window were skipped, not trimmed, so nothing
            # advanced the start index for them. Without this the suffix
            # invariant breaks -- the block list would carry a gap the
            # accounting believes is filled.
            for layer_id in self._local_layers:
                starts[layer_id] = max(starts[layer_id],
                                       min(first_live, num_required))
        return new_blocks

    def _charge_recall(self, req):
        """Bytes moved from a victim tier into the NPU for a resume or a hit.

        Whole coarse blocks move, including the one straddling the NPU-hit
        boundary, so the charge is ``hits x bytes_per_block`` of that tier.
        """
        pool = req.storage_hit_pool
        if pool is None or req.storage_hit_blocks <= 0:
            return
        self._recall_bytes += req.storage_hit_blocks * pool.bytes_per_block
        req.storage_hit_pool = None
        req.storage_hit_blocks = 0

    # -------------------- caching --------------------

    def cache_blocks(self, req, num_tokens):
        """Index every block of ``req`` that is now full, then write down.

        Called from ``allocate_slots`` and again when a chunk completes, so it
        must be idempotent -- ``num_cached_block`` is what makes it so.

        Progress is tracked per layer. Only a slot that is present *and* valid
        may be indexed, so a layer awaiting a transfer stops there while the
        others carry on; the write-down follows the slowest layer, because a
        coarse block is only complete once every layer of it is indexed. With
        no layer-granular calls every layer advances together and this is the
        scalar behaviour it replaces.
        """
        if not self.enable_caching:
            return
        hashes = request_block_hashes(req, self.block_size)
        if not hashes:
            return

        counts = self.num_cached_block.get(req.id)
        if counts is None:
            counts = [0] * self.num_layers
        num_full = min(num_tokens // self.block_size, len(hashes))
        if min(counts) >= num_full:
            return

        blocks = self.req_to_blocks[req.id]
        if len(blocks) < num_full:
            raise RuntimeError(
                f"[TieredKVCacheManager] request {req.id}: {num_full} full blocks "
                f"to cache but only {len(blocks)} allocated"
            )
        num_written = min(counts)
        for layer_id in range(self.num_layers):
            start = counts[layer_id]
            end = start
            while end < num_full:
                state = blocks[end].layers[layer_id]
                if state.hbm_block is None or not state.valid:
                    break
                end += 1
            if end == start:
                continue
            layer_hashes = [(layer_id, h) for h in hashes[:end]]
            layer_blocks = [b.layers[layer_id].hbm_block for b in blocks[:end]]
            self.npu_pool.cache_full_blocks(layer_hashes, layer_blocks, start, end)
            counts[layer_id] = end
        self.num_cached_block[req.id] = counts
        self._write_down(hashes, num_written, min(counts))

    def _write_down(self, hashes, num_cached, num_full):
        """Inclusive write-through into every victim tier.

        A coarse block is written once it is fully covered by indexed NPU
        blocks. The bytes are recorded for energy accounting only -- see the
        module docstring on why the latency is assumed overlapped.
        """
        for pool in self.lower_pools:
            factor = self._factor(pool)
            first = num_cached // factor
            last = num_full // factor          # exclusive; only complete coarse blocks
            for idx in range(first, last):
                block_hash = self._coarse_hash(hashes, pool, idx)
                if block_hash is None:
                    break
                if pool.cache_copy(block_hash):
                    self._write_through_bytes += pool.bytes_per_block

    # -------------------- layer-granular slots --------------------

    def _layer_states(self, req, layer_id, block_indices=None):
        """``(block_index, LayerKVState)`` pairs for one layer of ``req``.

        ``block_indices`` selects a subset in the caller's order; None means
        every block the request holds, in order. A sliding-window layer drops
        the blocks that fell out of its window, which is what the subset is for.
        """
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(
                f"[TieredKVCacheManager] layer_id {layer_id} out of range for "
                f"{self.num_layers} layers"
            )
        blocks = self.req_to_blocks.get(req.id, [])
        if block_indices is None:
            return list(enumerate(blocks))
        pairs = []
        for index in block_indices:
            if not 0 <= index < len(blocks):
                raise IndexError(
                    f"[TieredKVCacheManager] request {req.id}: block index "
                    f"{index} out of range for {len(blocks)} blocks"
                )
            pairs.append((index, blocks[index]))
        return pairs

    def allocate_layer(self, req, layer_id, block_indices=None):
        """Reserve HBM for one layer of ``req``'s blocks, leaving the rest alone.

        All-or-nothing, like :meth:`allocate_slots`: returns False without
        mutating anything when the pool cannot cover every missing slot. The
        slots come back **invalid** -- reserving space says nothing about
        whether the KV that belongs there has arrived, and only
        :meth:`mark_layer_valid` may claim that it has.

        Returns True when the layer holds every requested slot afterwards,
        including the no-op case where it already did.
        """
        missing = [block.layers[layer_id]
                   for _, block in self._layer_states(req, layer_id, block_indices)
                   if block.layers[layer_id].hbm_block is None]
        if not missing:
            return True
        if len(missing) > self.npu_pool.get_num_free_blocks():
            return False
        slots = self.npu_pool.get_new_blocks(len(missing))
        for state, slot in zip(missing, slots):
            state.hbm_block = slot
            state.valid = False
        self._resync_layer_start(req, layer_id)
        return True

    def free_layer(self, req, layer_id, block_indices=None):
        """Release one layer's HBM, keeping every other layer pinned.

        Reverse order, as :meth:`free` does, so the tail is reused first and
        the head -- the recoverable prefix -- survives longest. The blocks keep
        their hashes, so this is a release of *space*, not of data: whatever is
        still resident stays findable until the pool actually reuses it.

        The layer's caching progress rewinds to the first block released, so a
        later re-allocation can index those blocks again once they are valid.

        Returns the number of slots released.
        """
        pairs = self._layer_states(req, layer_id, block_indices)
        held = [(index, block.layers[layer_id]) for index, block in pairs
                if block.layers[layer_id].hbm_block is not None]
        if not held:
            return 0
        self.npu_pool.free_blocks(
            [state.hbm_block for _, state in reversed(held)])
        for _, state in held:
            state.hbm_block = None
            state.valid = False
        counts = self.num_cached_block.get(req.id)
        if counts is not None:
            counts[layer_id] = min(counts[layer_id], min(i for i, _ in held))
        self._resync_layer_start(req, layer_id)
        return len(held)

    def _resync_layer_start(self, req, layer_id):
        """Recompute where one layer's held suffix begins.

        The explicit layer API may free or reserve any subset, so the suffix
        invariant the bulk path relies on has to be re-derived. An interior gap
        pushes the start past it, which under-reports what is held and so
        over-reports what an allocation would need: the safe direction.
        """
        starts = self.req_layer_start.get(req.id)
        if starts is None:
            return
        blocks = self.req_to_blocks.get(req.id, [])
        start = len(blocks)
        for index in range(len(blocks) - 1, -1, -1):
            if blocks[index].layers[layer_id].hbm_block is None:
                break
            start = index
        starts[layer_id] = start

    def mark_layer_valid(self, req, layer_id, block_indices=None):
        """Declare that a compute or a completed recall filled these slots.

        Until this is called the slots hold space and nothing else, so
        :meth:`cache_blocks` will not index them and no other request can hit
        them.
        """
        for index, block in self._layer_states(req, layer_id, block_indices):
            state = block.layers[layer_id]
            if state.hbm_block is None:
                raise RuntimeError(
                    f"[TieredKVCacheManager] request {req.id}: block {index} "
                    f"layer {layer_id} has no HBM slot to validate"
                )
            state.valid = True

    def layer_resident(self, req, layer_id, block_indices=None):
        """True when every requested slot of this layer is present and valid."""
        pairs = self._layer_states(req, layer_id, block_indices)
        if not pairs:
            return False
        return all(block.layers[layer_id].hbm_block is not None
                   and block.layers[layer_id].valid
                   for _, block in pairs)

    # -------------------- release --------------------

    def free(self, req):
        """Release a finished request's blocks, keeping their hashes.

        Reverse order so the tail is reused first and the head -- the
        recoverable prefix -- survives longest.
        """
        blocks = self.req_to_blocks.pop(req.id, [])
        self.num_cached_block.pop(req.id, None)
        self.req_layer_start.pop(req.id, None)
        if blocks:
            self.npu_pool.free_blocks(self._hbm_blocks(reversed(blocks)))

    def preempt(self, req):
        """Release a *running* request's blocks so someone else can use them.

        Identical to :meth:`free`; kept as a separate call site because the
        request lives on and the scheduler counts these. Nothing here touches
        ``num_computed_tokens``: the request is re-admitted through the normal
        path, and :meth:`get_computed_blocks` re-derives where it got to. That
        is vLLM's behaviour, not a deviation from it -- with a victim tier the
        recovery is a recall, and without one it is a recompute.
        """
        self.free(req)

    # -------------------- accounting --------------------

    def take_traffic(self):
        """``(recall_bytes, write_through_bytes)`` since the last call."""
        recall, write_through = self._recall_bytes, self._write_through_bytes
        self._recall_bytes = 0
        self._write_through_bytes = 0
        return recall, write_through

    def npu_used_bytes(self):
        return self.npu_pool.used_bytes()

    def usage(self):
        return self.npu_pool.usage()

    def is_free(self):
        """Every tier back to a fully free list. End-of-run leak check."""
        return all(p.is_free() for p in [self.npu_pool] + self.lower_pools)


def _selftest(num_layers=1):
    """Runs with a plain ``python3 -m serving.core.kv_cache_manager``."""
    B, KB = 16, 1024
    BYTES = B * 128 * KB * num_layers

    class Req:
        _next = 0

        def __init__(self, prompt, output=()):
            Req._next += 1
            self.id = Req._next
            self.input_hash_ids = list(prompt)
            self.output_hash_ids = list(output)
            self.original_input = len(prompt)
            self.num_computed_tokens = 0
            self.num_tokens_reached = len(prompt)
            self.block_hashes = None
            self.storage_hit_pool = None
            self.storage_hit_blocks = 0

    def new_mgr(npu_blocks=64, lower=None, caching=True):  # noqa: D401
        npu = BlockPool(Device.NPU, npu_blocks * num_layers, B,
                        BYTES // num_layers, enable_caching=caching)
        pools = []
        if lower:
            pools.append(BlockPool(Device.CPU, lower, 256, 16 * BYTES,
                                   enable_caching=caching))
        return TieredKVCacheManager(B, npu, pools, enable_caching=caching,
                                   num_layers=num_layers)

    # chained hashes: same tail tokens, different prefix -> different hash
    a, b = Req(list(range(0, 16)) + list(range(100, 116))), Req(list(range(50, 66)) + list(range(100, 116)))
    ha, hb = request_block_hashes(a, B), request_block_hashes(b, B)
    assert len(ha) == 2 and ha[1] != hb[1], "unchained hashes would collide here"

    # cold miss allocates everything; the hit path allocates nothing for the prefix
    m = new_mgr()
    r1 = Req(list(range(64)))
    blocks, npu_hit, low_hit = m.get_computed_blocks(r1)
    assert (blocks, npu_hit, low_hit) == ([], 0, 0)
    assert m.allocate_slots(r1, 64) is not None
    assert len(m.req_to_blocks[r1.id]) == 4
    r1.num_computed_tokens = 64

    r2 = Req(list(range(64)))                       # same prompt
    blocks, npu_hit, low_hit = m.get_computed_blocks(r2)
    assert npu_hit == 48 and low_hit == 0, (npu_hit, low_hit)   # capped at 64-1 -> 3 blocks
    free_before = m.npu_pool.get_num_free_blocks()
    assert m.allocate_slots(r2, 64 - npu_hit, blocks, npu_hit) is not None
    assert m.npu_pool.get_num_free_blocks() == free_before - num_layers, "only the tail block is new"

    # allocation failure returns None and mutates nothing
    m = new_mgr(npu_blocks=4)
    r = Req(list(range(64)))
    assert m.allocate_slots(r, 64) is not None
    r2 = Req(list(range(100, 164)))
    free_before = m.npu_pool.get_num_free_blocks()
    assert m.allocate_slots(r2, 64) is None
    assert m.npu_pool.get_num_free_blocks() == free_before
    assert r2.id not in m.req_to_blocks

    # preempt then resume with everything resident: no recall, and the recompute
    # is only the block-aligned remainder
    m = new_mgr(npu_blocks=64)
    r = Req(list(range(64)))
    m.allocate_slots(r, 64)
    r.num_computed_tokens = 64
    m.preempt(r)
    r.num_computed_tokens = 0
    blocks, npu_hit, low_hit = m.get_computed_blocks(r)
    assert low_hit == 0 and npu_hit == 48, (npu_hit, low_hit)
    assert m.take_traffic() == (0, 0), "resident resume must be free"
    m.allocate_slots(r, 64 - npu_hit, blocks, npu_hit)
    assert len(m.req_to_blocks[r.id]) == 4

    # preempt, force reuse, no lower tier -> full recompute, still no transfer
    m = new_mgr(npu_blocks=8)
    r = Req(list(range(128)))
    m.allocate_slots(r, 128)
    r.num_computed_tokens = 128
    m.preempt(r)
    r.num_computed_tokens = 0
    other = Req(list(range(500, 628)))
    m.allocate_slots(other, 128)                    # evicts every one of r's blocks
    m.req_to_blocks[r.id] = []
    m.num_cached_block.pop(r.id, None)
    _, npu_hit, low_hit = m.get_computed_blocks(r)
    assert (npu_hit, low_hit) == (0, 0), (npu_hit, low_hit)
    assert m.take_traffic() == (0, 0)

    # with a lower tier the same resume is a recall, not a recompute
    m = new_mgr(npu_blocks=64, lower=8)
    r = Req(list(range(512)))
    m.allocate_slots(r, 512)
    r.num_computed_tokens = 512
    _, wt = m.take_traffic()
    assert wt == 2 * 16 * BYTES, wt                 # two 256-token coarse blocks
    m.preempt(r)
    r.num_computed_tokens = 0
    m.req_to_blocks[r.id] = []
    m.num_cached_block.pop(r.id, None)
    fillers = []                                    # push every NPU block out...
    for i in range(16):
        f = Req(list(range(9000 + 100 * i, 9064 + 100 * i)))
        if m.allocate_slots(f, 64) is not None:
            fillers.append(f)
    _, npu_hit, low_hit = m.get_computed_blocks(r)
    assert npu_hit == 0 and low_hit == 512, (npu_hit, low_hit)
    for f in fillers:                               # ...then make room to resume
        m.free(f)
    m.take_traffic()
    assert m.allocate_slots(r, 1, [], 0, low_hit) is not None
    recall, _ = m.take_traffic()
    assert recall == 2 * 16 * BYTES, recall

    # coarse tier only hits on its own boundary
    m = new_mgr(npu_blocks=64, lower=8)
    r = Req(list(range(200)))                       # < one 256-token chunk
    m.allocate_slots(r, 200)
    r.num_computed_tokens = 200
    _, wt = m.take_traffic()
    assert wt == 0, "no complete coarse block yet"

    # caching disabled: no index, no hits, resume is a full recompute
    m = new_mgr(caching=False)
    r = Req(list(range(64)))
    m.allocate_slots(r, 64)
    r.num_computed_tokens = 64
    m.preempt(r)
    r.num_computed_tokens = 0
    assert m.get_computed_blocks(r) == ([], 0, 0)
    assert m.npu_pool.is_free()

    # can_fit_full_sequence: the gate refuses a request whose first chunk fits
    # but whose whole prompt does not
    m = new_mgr(npu_blocks=8)                       # 8 blocks = 128 tokens
    r = Req(list(range(2048)))                      # 128 full blocks wanted
    _, npu_hit, low_hit = m.get_computed_blocks(r)
    assert m.can_fit_full_sequence(r, [], npu_hit, low_hit) is False
    assert m.allocate_slots(r, 64) is not None, "a single chunk still fits"
    m.free(r)
    r2 = Req(list(range(100)))                      # 6 full blocks + tail
    assert m.can_fit_full_sequence(r2, [], 0, 0) is True
    # a hit already on the NPU does not have to be re-reserved
    m = new_mgr(npu_blocks=8)
    a = Req(list(range(64)))
    m.allocate_slots(a, 64); a.num_computed_tokens = 64
    b = Req(list(range(64)))
    blocks, npu_hit, low_hit = m.get_computed_blocks(b)
    assert npu_hit == 48
    assert m.can_fit_full_sequence(b, blocks, npu_hit, low_hit) is True

    # ---- layer-granular slots ----
    # freeing one layer returns exactly that layer's slots; the rest stay pinned
    m = new_mgr(npu_blocks=64)
    r = Req(list(range(64)))
    assert m.allocate_slots(r, 64) is not None
    r.num_computed_tokens = 64
    free_before = m.npu_pool.get_num_free_blocks()
    target = num_layers - 1
    assert m.free_layer(r, target) == 4
    assert m.npu_pool.get_num_free_blocks() == free_before + 4, m.npu_pool
    assert not m.layer_resident(r, target)
    for other in range(num_layers):
        if other != target:
            assert m.layer_resident(r, other), f"layer {other} must be untouched"

    # re-allocating hands back space, not contents
    assert m.allocate_layer(r, target) is True
    assert m.npu_pool.get_num_free_blocks() == free_before
    assert not m.layer_resident(r, target), "space is not contents"
    m.mark_layer_valid(r, target)
    assert m.layer_resident(r, target)

    # a subset of blocks, which is what a sliding window drops
    assert m.free_layer(r, target, block_indices=[2, 3]) == 2
    assert m.npu_pool.get_num_free_blocks() == free_before + 2
    assert m.layer_resident(r, target, block_indices=[0, 1])
    assert not m.layer_resident(r, target, block_indices=[2])
    assert m.allocate_layer(r, target, block_indices=[2, 3]) is True
    m.mark_layer_valid(r, target, block_indices=[2, 3])
    assert m.layer_resident(r, target)
    m.free(r)
    assert m.is_free()

    # all-or-nothing: a layer allocation that cannot fit mutates nothing
    m = new_mgr(npu_blocks=4)
    r = Req(list(range(64)))
    assert m.allocate_slots(r, 64) is not None       # exactly fills the pool
    m.free_layer(r, 0)
    filler = m.npu_pool.get_new_blocks(m.npu_pool.get_num_free_blocks())
    before = m.npu_pool.get_num_free_blocks()
    assert m.allocate_layer(r, 0) is False
    assert m.npu_pool.get_num_free_blocks() == before
    assert all(b.layers[0].hbm_block is None for b in m.req_to_blocks[r.id])
    m.npu_pool.free_blocks(list(reversed(filler)))

    # a reserved-but-invalid slot never enters the prefix index
    m = new_mgr(npu_blocks=64)
    r = Req(list(range(64)))
    m.allocate_slots(r, 64)
    r.num_computed_tokens = 64
    hashes = request_block_hashes(r, B)
    m.free_layer(r, target)
    assert m.allocate_layer(r, target) is True
    m.cache_blocks(r, 64)
    fresh = m.req_to_blocks[r.id][0].layers[target].hbm_block
    assert fresh.block_hash is None, "an invalid slot must not be indexed"
    m.mark_layer_valid(r, target)
    m.cache_blocks(r, 64)
    assert fresh.block_hash == (target, hashes[0])
    assert m.npu_pool.get_cached_block((target, hashes[0])) is fresh
    m.free(r)
    assert m.is_free()

    # ---- sliding-window (hybrid attention) layers ----
    # 4 layers, the last one global, a 64-token window = 4 blocks
    def hybrid_mgr(npu_blocks=64, window=4 * B):
        from .utils import FULL_ATTENTION
        types = [LOCAL_ATTENTION] * 3 + [FULL_ATTENTION]
        npu = BlockPool(Device.NPU, npu_blocks * 4, B, BYTES // 4,
                        enable_caching=True)
        return TieredKVCacheManager(B, npu, [], enable_caching=True,
                                    num_layers=4, layer_types=types,
                                    sliding_window=window)

    # below the window nothing is saved: every layer covers the whole sequence
    m = hybrid_mgr()
    r = Req(list(range(4 * B)))                     # 64 tokens = 4 blocks
    free_all = m.npu_pool.get_num_free_blocks()
    assert m.allocate_slots(r, 4 * B) is not None
    assert free_all - m.npu_pool.get_num_free_blocks() == 4 * 4, "4 blocks x 4 layers"
    r.num_computed_tokens = 4 * B

    # past the window the local layers stop growing and give back their head
    r.num_tokens_reached = 10 * B                   # 160 tokens
    assert m.allocate_slots(r, 6 * B) is not None
    used = free_all - m.npu_pool.get_num_free_blocks()
    # full layer: 10 blocks. local layers: the last 4 blocks only.
    assert used == 10 + 3 * 4, used
    assert used < 10 * 4, "a uniform allocator would have taken 40"
    for layer_id in range(3):
        assert m.req_layer_start[r.id][layer_id] == 6, m.req_layer_start[r.id]
        assert all(b.layers[layer_id].hbm_block is None
                   for b in m.req_to_blocks[r.id][:6])
        assert all(b.layers[layer_id].hbm_block is not None
                   for b in m.req_to_blocks[r.id][6:])
    assert all(b.layers[3].hbm_block is not None for b in m.req_to_blocks[r.id])
    m.free(r)
    assert m.is_free(), "trimmed slots must not leak"

    # a long prompt admitted in one go never allocates the head of a local layer
    m = hybrid_mgr()
    r = Req(list(range(40 * B)))                    # 640 tokens = 40 blocks
    free_all = m.npu_pool.get_num_free_blocks()
    assert m.allocate_slots(r, 40 * B) is not None
    used = free_all - m.npu_pool.get_num_free_blocks()
    assert used == 40 + 3 * 4, used                 # full 40, each local 4
    assert m.req_layer_start[r.id][:3] == [36, 36, 36]
    m.free(r)
    assert m.is_free()

    # the footprint is flat in sequence length, which is the whole point
    for total in (40, 80, 160):
        m = hybrid_mgr(npu_blocks=256)
        r = Req(list(range(total * B)))
        free_all = m.npu_pool.get_num_free_blocks()
        m.allocate_slots(r, total * B)
        used = free_all - m.npu_pool.get_num_free_blocks()
        assert used == total + 3 * 4, (total, used)

    # admission accounts for the saving: a sequence that would not fit if every
    # layer took the whole range does fit when three of them take a window
    m = hybrid_mgr(npu_blocks=16)                   # 64 storage blocks
    r = Req(list(range(40 * B)))                    # needs 40 + 12 = 52
    assert m.can_fit_full_sequence(r) is True
    assert m.allocate_slots(r, 40 * B) is not None
    m.free(r)
    m_flat = TieredKVCacheManager(
        B, BlockPool(Device.NPU, 64, B, BYTES // 4, enable_caching=True),
        [], enable_caching=True, num_layers=4)      # no layer types: all full
    r2 = Req(list(range(40 * B)))                   # would need 40 x 4 = 160
    assert m_flat.can_fit_full_sequence(r2) is False

    # no leaks after a normal finish
    m = new_mgr()
    r = Req(list(range(64)))
    m.allocate_slots(r, 64)
    m.free(r)
    assert m.is_free(), m.npu_pool

    # Storage identities are independent across layers; total bytes stay equal
    # to the all-layer model, and an unfinished tail is not indexed.
    m = new_mgr(caching=False)
    r = Req(list(range(B + 1)))
    assert m.allocate_slots(r, B + 1) is not None
    logical = m.req_to_blocks[r.id]
    assert [b.token_block_index for b in logical] == [0, 1]
    slots = m._hbm_blocks(logical)
    assert len({b.block_id for b in slots}) == 2 * num_layers
    assert m.npu_used_bytes() == 2 * BYTES
    r.num_computed_tokens = B + 1
    assert m.allocate_slots(r, 1) == []
    assert m.npu_used_bytes() == 2 * BYTES
    m.free(r)
    assert m.npu_used_bytes() == 0

    # A missing layer invalidates the whole logical prefix. This also checks
    # that layer-qualified keys never alias the same token hash.
    m = new_mgr()
    r = Req(list(range(2 * B)))
    m.allocate_slots(r, 2 * B)
    hashes = request_block_hashes(r, B)
    first = m.req_to_blocks[r.id][0]
    for layer_id in range(num_layers):
        assert m.npu_pool.get_cached_block((layer_id, hashes[0])) is first[layer_id].hbm_block
    m.free(r)
    m.npu_pool._maybe_evict_cached_block(first[0].hbm_block)
    assert m.get_computed_blocks(r) == ([], 0, 0)

    # There must be space for every layer before any slot is claimed.
    m = new_mgr(npu_blocks=2)
    r = Req(list(range(2 * B)))
    occupied = m.npu_pool.get_new_blocks(1)
    before = m.npu_pool.get_num_free_blocks()
    assert m.allocate_slots(r, 2 * B) is None
    assert m.npu_pool.get_num_free_blocks() == before
    assert r.id not in m.req_to_blocks
    m.npu_pool.free_blocks(occupied)

    print(f"kv_cache_manager self-test: {num_layers} layers, all checks passed")


if __name__ == "__main__":
    _selftest()
    _selftest(num_layers=3)
