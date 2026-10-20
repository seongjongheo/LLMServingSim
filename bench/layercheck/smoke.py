"""Step 1 of the per-layer check: can we see inside a CUDA-graph run at all?

    python -m bench.layercheck.smoke --mode eager
    python -m bench.layercheck.smoke --mode piecewise

Boots the full model (every layer, dummy weights) in the given mode,
runs a few fixed batches through the profiler's scheduler-bypass path,
and prints, per batch, how many GPU kernels and attention kernels one
forward produced. The check passes when piecewise mode still shows one
attention kernel per layer -- that means torch.profiler records the
kernels inside a graph replay, and the attention cuts can be found.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from profiler.core.config import read_model_config
from profiler.core.hooks.batch import Shot

# Kernel names that read the KV cache for attention. The KV *write*
# (reshape_and_cache*) also says "flash" on some backends, so drop it.
ATTN_RE = re.compile(r"flash_fwd|fmha|attention|attn", re.IGNORECASE)
NOT_ATTN_RE = re.compile(r"reshape_and_cache|cache_flash", re.IGNORECASE)

SHOTS = {
    # Window is 1024: below it local and global layers read the same KV.
    "decode_n8_kv512": Shot.attention(0, 0, 8, 512),
    # Above it a local layer stays at 1024 while a global one reads 4096.
    "decode_n8_kv4096": Shot.attention(0, 0, 8, 4096),
    "decode_n1_kv8192": Shot.attention(0, 0, 1, 8192),
    # A prefill chunk mixed with decodes; 64 + 8 = 72 is a capture size.
    "mixed_pc64_n8_kv2048": Shot.attention(64, 0, 8, 2048),
}


def _is_attention(name: str) -> bool:
    return bool(ATTN_RE.search(name)) and not NOT_ATTN_RE.search(name)


def main() -> int:
    ap = argparse.ArgumentParser(prog="bench.layercheck.smoke")
    ap.add_argument("--mode", choices=["eager", "piecewise"], required=True)
    ap.add_argument("--model-config",
                    default="configs/model/google/gemma-3-4b-it.json")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--out-dir", default="bench/results/layercheck")
    args = ap.parse_args()

    from vllm import LLM

    model_config = read_model_config(Path(args.model_config))
    num_layers = int(model_config["num_hidden_layers"])

    tmpdir = Path(tempfile.mkdtemp(prefix="layercheck_model_"))
    (tmpdir / "config.json").write_text(json.dumps(model_config, indent=2))

    # Same engine shape as the bench run (bench/examples/RTX4090/
    # gemma-3-4b-it/vllm/meta.json), minus what a dummy-weight,
    # no-tokenizer run cannot use.
    kwargs = dict(
        model=str(tmpdir),
        load_format="dummy",
        skip_tokenizer_init=True,
        generation_config="vllm",
        enable_prefix_caching=False,
        dtype="bfloat16",
        max_num_seqs=64,
        max_num_batched_tokens=2048,
        max_model_len=16384,
        seed=42,
        worker_extension_cls="bench.layercheck.extension.Extension",
    )
    if args.mode == "eager":
        kwargs["enforce_eager"] = True
    else:
        # vLLM's default is FULL_AND_PIECEWISE, which runs a pure-decode
        # batch as one uncut graph. Force the cut everywhere.
        kwargs["compilation_config"] = {"cudagraph_mode": "PIECEWISE"}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    try:
        llm = LLM(**kwargs)
        for tag, shot in SHOTS.items():
            raw = llm.collective_rpc(
                "run_shot", args=(shot.as_dict(), tag, args.iterations))
            results[tag] = raw[0]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    out_path = out_dir / f"smoke_{args.mode}.json"
    out_path.write_text(json.dumps(results, indent=1))

    print(f"\n=== mode={args.mode}  layers={num_layers}  -> {out_path}")
    print(f"{'batch':<24}{'fwd ms':>9}{'kernel ms':>11}"
          f"{'kernels/fwd':>13}{'attn/fwd':>10}")
    for tag, res in results.items():
        it = len(res["forward_ms"])
        kernels = res["kernels"]
        n_attn = sum(_is_attention(k["name"]) for k in kernels)
        busy_ms = sum(k["dur_us"] for k in kernels) / 1000 / it
        print(f"{tag:<24}{statistics.mean(res['forward_ms']):>9.3f}"
              f"{busy_ms:>11.3f}{len(kernels) / it:>13.1f}{n_attn / it:>10.1f}")

    first = next(iter(results.values()))
    names = Counter(k["name"] for k in first["kernels"])
    print(f"\nattention kernels seen ({next(iter(results))}):")
    for name, n in names.items():
        if _is_attention(name):
            print(f"  {n:>5}x  {name[:110]}")
    print("\nmost frequent kernels:")
    for name, n in names.most_common(12):
        print(f"  {n:>5}x  {name[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
