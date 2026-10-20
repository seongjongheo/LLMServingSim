"""Step 2a of the per-layer check: the simulator's layer times for the smoke shots.

    python -m bench.layercheck.sim_layers

Feeds each shot in ``smoke.SHOTS`` to the trace generator as a one-off batch
and writes every emitted layer's ``comp_time`` to a CSV next to this file.
Only the profiled-latency lookup runs -- no ASTRA-Sim -- so this works on a
host without the simulator binaries. ``compare.py`` reads the CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from bench.layercheck.smoke import SHOTS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(prog="bench.layercheck.sim_layers")
    ap.add_argument("--hardware", default="RTX4090")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        HERE / f"sim_{args.hardware}_{args.model.split('/')[-1]}.csv")
    out = out.resolve()

    # The trace generator resolves profiler/ relative to astra-sim/, as the
    # simulator does after its own chdir.
    os.chdir(REPO / "astra-sim")
    from serving.core.request import Batch
    from serving.core.trace_generator import _synthesize_trace, resolve_variant
    from serving.core.utils import LOCAL_ATTENTION, get_config, resolve_layer_types

    config = get_config(args.model)
    variant = resolve_variant("bfloat16", "auto", config)
    layer_types, _ = resolve_layer_types(config)
    placement = {"default": {"weights": "LOCAL", "kv_loc": "LOCAL",
                             "kv_evict_loc": "REMOTE:0"},
                 "block": [], "layer": {}}
    num_layers = config["num_hidden_layers"]

    rows_out = []
    for batch_id, (tag, shot) in enumerate(SHOTS.items()):
        q_list = [new for new, _ in shot.requests]
        k_list = [hist for _, hist in shot.requests]
        prefill = [(q, k) for q, k in shot.requests if q > 1]
        decode_k = [k for q, k in shot.requests if q == 1]
        batch = Batch(batch_id, args.model, sum(q_list), sum(decode_k),
                      q_list, k_list, len(prefill), len(decode_k),
                      [q for q, _ in prefill], [k for _, k in prefill],
                      decode_k, 0, 0)
        rows, block_starts = _synthesize_trace(
            args.hardware, args.model, config, 1, 1, 1, 1, None, 0, 0,
            batch, 0, placement, False, None, False, None, None, 2, variant)

        # Map each row to its transformer block: -1 before the first block,
        # num_layers after the last.
        for idx, row in enumerate(rows):
            layer_num = -1
            for b, start in enumerate(block_starts):
                if idx >= start:
                    layer_num = b
            name = row[0]
            if layer_num == num_layers - 1 and name in ("final_layernorm", "lm_head", "sampler"):
                layer_num = num_layers
            is_local = 0 <= layer_num < num_layers and layer_types[layer_num] == LOCAL_ATTENTION
            rows_out.append({"shot": tag, "idx": idx, "layer_num": layer_num,
                             "layer": name, "is_local": int(is_local),
                             "time_us": int(row[1]) / 1000})

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
