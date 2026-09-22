import os
from functools import lru_cache
from time import time
import json

from .run_paths import input_path


# Formatting string for a trace file's per-layer row. Kept in this
# module because trace writers live across the codebase and import it
# as the canonical row template.
# The trailing space on every field but the last is load-bearing, not cosmetic.
# ``{:<15}`` pads a *shorter* value but emits nothing extra for one that is
# already 15 characters, so the next field butts straight against it and the
# readers -- which split on whitespace -- see the two merged into one. A
# 3-dimensional involved_dim does exactly that: ``ALLREDUCE:1,0,0`` is 15
# characters on the nose, and the row comes back one field short. The widths
# below are therefore a minimum column, and the separator is explicit.
_FMT = (
    "{:<30} "  # Layername
    "{:<15} "  # comp_time
    "{:<15} "  # input_loc
    "{:<15} "  # input_size
    "{:<15} "  # weight_loc
    "{:<15} "  # weight_size
    "{:<15} "  # output_loc
    "{:<15} "  # output_size
    "{:<15} "  # comm_type
    "{:<15} "  # comm_size
    "{:<15}"   # misc
    "\n"
)


def get_workload(batch, hardware, instance_id=0, event=False, workload_name=None, inputs_root=None):
    if event:
        file_name = 'event_handler'
    elif workload_name:
        file_name = workload_name
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    if inputs_root is None:
        inputs_root = os.path.join(os.getcwd(), "inputs")
    return input_path(inputs_root, "workload", file_name, "llm")


def header():
    """The column-name row, formatted through the same template as the data rows.

    Built with ``_FMT`` rather than its own width list so the whole file has one
    layout and the guaranteed-separator invariant holds for every line. These
    particular labels can never overflow -- the longest is 11 characters against
    a 15-character column -- but a header in a different format than the rows
    below it invites exactly the kind of off-by-one-field reading that the
    separator is there to prevent.
    """
    return _FMT.format(
        "Layername", "comp_time", "input_loc", "input_size",
        "weight_loc", "weight_size", "output_loc", "output_size",
        "comm_type", "comm_size", "misc",
    )


def formatter(layername, comp_time, input_loc, input_size, weight_loc, weight_size, output_loc, output_size, comm_type, comm_size, misc):
    return _FMT.format(
        layername, comp_time, input_loc, input_size, weight_loc,
        weight_size, output_loc, output_size, comm_type, comm_size, misc,
    )


@lru_cache(maxsize=None)
def get_config(model_name):
    """Load a model architecture config, cached for the process lifetime.

    Cached because the hot path re-read and re-parsed the JSON on every
    iteration -- generate_trace, the memory model's size helpers and the
    DP-group scheduling block each call this per scheduled batch.

    The cache hands every caller the *same* dict, so callers must treat it
    as read-only. They do today: every use is a subscript, a ``.get`` or an
    ``in`` test, and nothing assigns into it or mutates it in place.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    serving_dir = os.path.dirname(base_dir)
    repo_root = os.path.dirname(serving_dir)
    candidate_paths = [
        os.path.join(repo_root, "configs", "model", model_name + ".json"),
        os.path.join(serving_dir, "configs", "model", model_name + ".json"),
    ]

    config = None
    for config_path in candidate_paths:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            break
        except FileNotFoundError:
            continue

    if config is None:
        raise FileNotFoundError(
            f"Config file for model '{model_name}' not found. Checked: "
            f"{', '.join(candidate_paths)}. Please add the corresponding config file."
        )

    return config


# Attention type of one transformer block. A local (sliding-window) layer only
# ever attends to the last ``window`` tokens, so its KV past that point is dead:
# it can be dropped, and the attention kernel reads less. A full layer attends to
# the whole sequence, as every layer of a non-hybrid model does.
FULL_ATTENTION = "full"
LOCAL_ATTENTION = "local"


def resolve_layer_types(config, sliding_window=None, sliding_window_pattern=None):
    """Per-layer attention type for a model, as ``(layer_types, window)``.

    ``layer_types[i]`` is :data:`FULL_ATTENTION` or :data:`LOCAL_ATTENTION` and
    ``window`` is the sliding-window size in tokens, or None when every layer is
    full attention (the non-hybrid case, and the default for every model already
    in ``configs/model/``).

    Sources, in precedence order:

    * the explicit arguments -- the CLI's ``--sliding-window`` /
      ``--sliding-window-pattern``. These impose a pattern on *any* model, which
      is what lets an already-profiled model stand in for a hybrid one while the
      hybrid's own profile does not exist yet. The latency then comes from real
      measurements of a real model and the layer pattern is the one deliberate
      experimental variable, rather than every number being borrowed.
    * ``layer_types``: HF's explicit per-layer list, e.g.
      ``["sliding_attention", "full_attention", ...]``.
    * ``sliding_window`` + ``sliding_window_pattern``: the form Gemma-3 ships.
    * ``sliding_window`` alone: every layer is local, which is what a uniform
      sliding-window model (Mistral, Phi-mini-MoE) means.

    The pattern rule is HF's own (``Gemma3Attention.is_sliding``): layer ``i`` is
    full attention when ``(i + 1) % pattern == 0``, so the last layer of every
    group of ``pattern`` is global and the rest are local -- Gemma-3's 5:1.
    """
    num_layers = config['num_hidden_layers']

    if sliding_window_pattern is not None and sliding_window is None:
        raise ValueError(
            "sliding_window_pattern given without sliding_window: a pattern "
            "says which layers are local, not how far back they may look"
        )
    if sliding_window is not None:
        if sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {sliding_window}")
        window, pattern = sliding_window, sliding_window_pattern
    else:
        # Fall back to the model config. A null in the JSON means "not used",
        # which is how Qwen3 and Mixtral spell "every layer is full attention".
        explicit = config.get('layer_types')
        window = config.get('sliding_window')
        pattern = config.get('sliding_window_pattern')
        if explicit:
            if len(explicit) != num_layers:
                raise ValueError(
                    f"layer_types has {len(explicit)} entries but the model has "
                    f"{num_layers} layers"
                )
            if window is None:
                raise ValueError(
                    "layer_types names sliding layers but the config carries no "
                    "sliding_window"
                )
            types = [LOCAL_ATTENTION if 'sliding' in str(t) else FULL_ATTENTION
                     for t in explicit]
            return types, (window if LOCAL_ATTENTION in types else None)

    if not window:
        return [FULL_ATTENTION] * num_layers, None
    if pattern is not None and pattern <= 0:
        raise ValueError(f"sliding_window_pattern must be positive, got {pattern}")
    if pattern is None:
        # Uniform sliding window: no layer is global. A null in the JSON reaches
        # here as None, which is the same thing.
        return [LOCAL_ATTENTION] * num_layers, window
    types = [FULL_ATTENTION if (i + 1) % pattern == 0 else LOCAL_ATTENTION
             for i in range(num_layers)]
    return types, (window if LOCAL_ATTENTION in types else None)


def _selftest():
    """Runs with a plain ``python3 -m serving.core.utils``."""
    # every model already in configs/model/ must stay non-hybrid, or the
    # recorded baselines would move for a reason nobody asked for
    for name in ("meta-llama/Llama-3.1-8B", "Qwen/Qwen3-32B",
                 "Qwen/Qwen3-30B-A3B-Instruct-2507",
                 "mistralai/Mixtral-8x7B-v0.1"):
        types, window = resolve_layer_types(get_config(name))
        assert window is None, (name, window)
        assert set(types) == {FULL_ATTENTION}, name
        assert len(types) == get_config(name)['num_hidden_layers'], name

    # Gemma-3 27B: 62 layers, every 6th is global -> 10 full, 52 local
    cfg = get_config("google/gemma-3-27b-it")
    types, window = resolve_layer_types(cfg)
    assert window == 1024, window
    assert len(types) == 62
    assert types.count(FULL_ATTENTION) == 10, types.count(FULL_ATTENTION)
    assert types.count(LOCAL_ATTENTION) == 52
    assert [i for i, t in enumerate(types) if t == FULL_ATTENTION][:3] == [5, 11, 17]
    assert types[61] == LOCAL_ATTENTION, "62 layers do not divide into groups of 6"

    # CLI override imposes the pattern on a model that has none
    llama = get_config("meta-llama/Llama-3.1-8B")
    types, window = resolve_layer_types(llama, sliding_window=1024,
                                        sliding_window_pattern=6)
    assert window == 1024 and len(types) == 32
    assert types.count(FULL_ATTENTION) == 5, types.count(FULL_ATTENTION)
    assert [i for i, t in enumerate(types) if t == FULL_ATTENTION] == [5, 11, 17, 23, 29]

    # window with no pattern: every layer local, as a uniform sliding-window
    # model means (Mistral, Phi-mini-MoE)
    types, window = resolve_layer_types(llama, sliding_window=4096)
    assert window == 4096 and set(types) == {LOCAL_ATTENTION}

    # pattern 1 makes every layer global, so there is no window to speak of
    types, window = resolve_layer_types(llama, sliding_window=1024,
                                        sliding_window_pattern=1)
    assert set(types) == {FULL_ATTENTION} and window is None

    for bad in (dict(sliding_window_pattern=6),
                dict(sliding_window=0),
                dict(sliding_window=1024, sliding_window_pattern=0)):
        try:
            resolve_layer_types(llama, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    print("utils self-test: all checks passed")


if __name__ == "__main__":
    _selftest()
