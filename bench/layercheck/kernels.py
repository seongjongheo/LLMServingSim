"""Print the kernel stream of single attention shots in the profiler's block.

    python -m bench.layercheck.kernels

For the shots whose per-layer attention came out wrong in the first block
profile (mixed prefill + decode), lists every kernel of one forward in
launch order, so the attention kernels' names and count per layer can be
read off directly.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from profiler.core.config import ProfileArgs, read_model_config
from profiler.core.engine import fuse_engine_kwargs
from profiler.core.hooks.batch import Shot
from profiler.core.hooks.timings import attention_per_layer

SHOTS = [
    Shot.attention(256, 4096, 32, 16384),  # new 253 us vs old 3725 us
    Shot.attention(1024, 0, 16, 256),      # new 0.0
    Shot.attention(0, 0, 8, 4096),         # decode only: known good
]


def main() -> int:
    model_config = read_model_config(Path("configs/model/google/gemma-3-4b-it.json"))
    args = ProfileArgs(
        architecture="gemma3_text", model="google/gemma-3-4b-it",
        hardware="RTX4090", model_config=model_config,
        max_num_batched_tokens=2048, max_num_seqs=64,
        attention_layer_type="full",
    )
    from vllm import LLM

    # The profiler's own engine shape, but with the layercheck worker
    # extension, whose run_shot returns the raw kernel list.
    kwargs = fuse_engine_kwargs(args, 1)
    tmpdir = Path(tempfile.mkdtemp(prefix="layercheck_model_"))
    (tmpdir / "config.json").write_text(json.dumps(model_config, indent=2))
    kwargs["model"] = str(tmpdir)
    kwargs["worker_extension_cls"] = "bench.layercheck.extension.Extension"
    try:
        llm = LLM(**kwargs)
        for shot in SHOTS:
            raw = llm.collective_rpc("run_shot", args=(shot.as_dict(), "k", 1))
            kernels = raw[0]["kernels"]
            names_durs = [(k["name"], k["dur_us"]) for k in kernels]
            print(f"\n=== shot {shot.requests[:2]}... ({len(shot.requests)} reqs): "
                  f"{len(kernels)} kernels")
            for name, dur in names_durs:
                if any(s in name for s in ("flash", "attn", "attention", "cache", "fmha")):
                    print(f"  {dur:>9.1f}  {name[:110]}")
            try:
                per = attention_per_layer(names_durs)
                print(f"  -> attention_per_layer: {len(per)} entries: "
                      + " ".join(f"{t:.0f}" for t in per))
            except ValueError as e:
                print(f"  -> attention_per_layer: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
