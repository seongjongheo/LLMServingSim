"""One-minute check before a multi-hour hybrid attention profile.

    python -m bench.layercheck.precheck

Boots the engine exactly as ``python -m profiler profile`` does for
gemma-3-4b (one 5:1 block), fires three attention shots through the
profiler's own ``fire`` with ``per_layer_attention``, and prints each
layer's time. Expect six numbers per shot, the last (the global layer)
pulling away from the other five as the KV grows.
"""

from __future__ import annotations

from pathlib import Path

from profiler.core.categories import AttentionCategory
from profiler.core.config import ProfileArgs, load_architecture, read_model_config
from profiler.core.engine import attention_layer_indices, spin_down, spin_up
from profiler.core.hooks.batch import Shot


def main() -> int:
    model_config = read_model_config(Path("configs/model/google/gemma-3-4b-it.json"))
    args = ProfileArgs(
        architecture="gemma3_text", model="google/gemma-3-4b-it",
        hardware="RTX4090", model_config=model_config,
        max_num_batched_tokens=2048, max_num_seqs=64,
        attention_layer_type="full", measurement_iterations=3,
    )
    arch = load_architecture(Path("profiler/models/gemma3_text.yaml"))
    slice_ = AttentionCategory("full").catalog_slice(arch)

    llm, _, tmpdir = spin_up(args, 1)
    ok = True
    try:
        # Decode-only shots, then the two mixed ones the first block profile
        # recorded wrong (global ~3414 us and ~119 us when right).
        for pc, kp, n, kv in ((0, 0, 8, 512), (0, 0, 8, 4096), (0, 0, 1, 8192),
                              (256, 4096, 32, 16384), (1024, 0, 16, 256)):
            raw = llm.collective_rpc(
                "fire", args=(Shot.attention(pc, kp, n, kv).as_dict(), slice_,
                              "attention", 3, True))
            times = [d["microseconds"] for d in raw[0] if d["layer"] == "attention"]
            print(f"pc={pc:<4} kp={kp:<5} n={n:<2} kv={kv:<5}  per layer us: "
                  + "  ".join(f"{t:.1f}" for t in times))
            ok &= len(times) == 6
    finally:
        spin_down(llm, tmpdir)

    keep = attention_layer_indices(args)
    print(f"\nglobal run keeps layer {keep}; "
          + ("PRECHECK OK -- start the profile" if ok
             else "PRECHECK FAILED -- do not start the profile"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
