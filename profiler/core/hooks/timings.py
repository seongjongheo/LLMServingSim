"""Extract canonical-layer timings from vLLM's layerwise-profile tree.

vLLM's ``layerwise_profile()`` context manager records CUDA kernel
events per nn.Module invocation and returns a nested tree where each
node has:
    entry["entry"]["name"]       str like "QKVParallelLinear(...)"
    entry["entry"]["cuda_time_us"]  total CUDA time across invocations
    entry["entry"]["invocations"]   how many times the module was called
    entry["children"]             list of child entries

We walk that tree in DFS order, strip the ``(...)`` argument suffix
from each class name, and try to match every node against the
catalog slice the host passed in. A match produces a ``TimingSample``
(layer_name, per-invocation microseconds).

Matching rule: a catalog entry matches a node iff
    node_class == entry.vllm
AND (entry.within is None OR some ancestor_class == entry.within)

The DFS path carries the list of ancestor class names, so the
``within`` check is just a membership test.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TimingSample:
    """One layer-level CUDA timing extracted from a shot.

    ``microseconds`` is already divided by the invocation count, so if
    a layer was called multiple times inside a single forward pass
    (which can happen e.g. if a decoder has more than one layer and we
    forgot to set hf_overrides.num_hidden_layers=1) each sample
    represents the *per-call* cost.
    """

    layer: str
    microseconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _strip_class_name(raw: str) -> str:
    """Turn ``"QKVParallelLinear(in_features=4096, ...)"`` → ``"QKVParallelLinear"``.

    vLLM's profiler stringifies nn.Module instances as ``ClassName(repr)``.
    We only need the class name for matching.
    """
    paren = raw.find("(")
    return raw if paren < 0 else raw[:paren]


def _match_slice(
    node_class: str,
    ancestors: list[str],
    slice_: dict[str, dict[str, Any]],
) -> str | None:
    """Return the canonical layer name that matches, or None.

    ``slice_`` is the host-to-worker serialized form of a ``Catalog``
    group — ``{canonical_name: {"vllm": cls, "within": parent_cls_or_None,
    "tp_stable": ...}}``.

    Ambiguity rule: when several catalog entries match the same node
    (same ``vllm`` class, several ``within`` candidates all present in
    the ancestor chain), the one whose ``within`` is **deepest** in the
    ancestor chain wins. That disambiguates cases like Qwen3's two
    RMSNorms — one inside ``Qwen3DecoderLayer`` (input/post layernorm)
    and one inside ``Qwen3Attention`` (qk_norm) — so the inner match
    (``Qwen3Attention``) doesn't get swallowed by the outer catalog
    entry purely because of YAML ordering. Entries without ``within``
    are treated as the lowest-specificity fallback.
    """
    best_name: str | None = None
    best_depth = -2  # within=None → depth -1; any match wins over no match
    for canonical, spec in slice_.items():
        if spec["vllm"] != node_class:
            continue
        within = spec.get("within")
        if within is None:
            depth = -1
        else:
            try:
                depth = ancestors.index(within)
            except ValueError:
                continue
        if depth > best_depth:
            best_depth = depth
            best_name = canonical
    return best_name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_samples(
    tree: list[dict[str, Any]],
    slice_: dict[str, dict[str, Any]],
) -> list[TimingSample]:
    """Walk the profiler tree; emit samples for nodes matching the slice.

    The slice is typically category-scoped (only the layers relevant
    to the category being profiled right now — passing the full
    catalog works too but produces samples the caller will discard).
    """
    samples: list[TimingSample] = []

    def walk(nodes: list[dict[str, Any]], ancestors: list[str]) -> None:
        for node in nodes:
            raw_name = str(node["entry"]["name"])
            cls = _strip_class_name(raw_name)

            # Try to match this node against the requested slice.
            canonical = _match_slice(cls, ancestors, slice_)
            if canonical is not None:
                cuda_us = float(node["entry"]["cuda_time_us"])
                invocations = max(1, int(node["entry"]["invocations"]))
                samples.append(
                    TimingSample(
                        layer=canonical,
                        microseconds=cuda_us / invocations,
                    )
                )

            # Always recurse, even after a match. Some catalog entries
            # are defined by parent-class; their actual kernel time is
            # in a leaf that we want to reach independently.
            children = node.get("children") or []
            walk(children, ancestors + [cls])

    walk(tree, ancestors=[])
    return samples


def select_attention(
    samples: list[TimingSample],
    keep: list[int] | None,
) -> list[TimingSample]:
    """Drop the attention samples of layers this run is not measuring.

    A hybrid model is profiled as one block of its layer pattern, and its
    shots are fired with ``per_layer_attention`` so they return one
    ``attention`` sample per layer, in layer order (``attention_per_layer``).
    ``keep`` lists the positions to retain; the writer averages what is
    left on its row key. Every other layer's samples pass through.
    None keeps everything.
    """
    if keep is None:
        return samples
    out: list[TimingSample] = []
    seen = 0
    for s in samples:
        if s.layer == "attention":
            if seen in keep:
                out.append(s)
            seen += 1
        else:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Per-layer attention from raw kernels (hybrid models)
# ---------------------------------------------------------------------------
#
# vLLM's ``summary_stats`` merges modules that share a name, so the six
# ``Attention`` modules of a hybrid block come back as one node: the mean of
# five local layers and one global one. The kernels themselves keep their
# order, and every attention brackets the same way -- the KV write, the
# attention kernel, and (split-kv only) its combine pass -- so cutting the
# kernel stream there gives each layer's attention on its own.

_ATTN_MAIN_RE = re.compile(r"flash_fwd(?!_splitkv_combine)")
_ATTN_AFTER_RE = re.compile(r"flash_fwd_splitkv_combine")
_ATTN_BEFORE_RE = re.compile(r"reshape_and_cache")


def attention_per_layer(names_durs: list[tuple[str, float]]) -> list[float]:
    """Each layer's attention time, in order, from one forward's kernels.

    ``names_durs`` is ``(kernel_name, duration_us)`` in launch order. A
    layer's attention is its attention kernel, the KV-write kernel before it
    and any split-kv combine after it. Raises if no attention kernel is
    recognised -- a backend this does not know yet -- rather than return
    nothing and let the caller write an empty table.
    """
    attn: list[float] = []
    pending = 0.0  # a KV write waiting for the attention it belongs to
    for name, dur in names_durs:
        if _ATTN_MAIN_RE.search(name):
            attn.append(pending + dur)
            pending = 0.0
        elif _ATTN_AFTER_RE.search(name) and attn:
            attn[-1] += dur
        elif _ATTN_BEFORE_RE.search(name):
            pending += dur
    if not attn:
        raise ValueError(
            "no attention kernel recognised in the forward's kernels; "
            "attention_per_layer knows FlashAttention's flash_fwd* names only"
        )
    return attn
