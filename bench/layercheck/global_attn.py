"""Why does the profiled global attention miss the full model's by 8-16%?

    python -m bench.layercheck.global_attn --config profile     # one boot each
    python -m bench.layercheck.global_attn --config profile16k
    python -m bench.layercheck.global_attn --config allglobal16k
    python -m bench.layercheck.global_attn --config hybrid16k
    python -m bench.layercheck.global_attn --config hybrid
    python -m bench.layercheck.global_attn --report

The profiler measures Gemma-3's global attention on a one-layer model with
``sliding_window`` deleted and vLLM's default ``max_model_len``; the full
model has 34 layers, a window on 29 of them, and (in smoke.py) a 16k
``max_model_len``. Each config below changes one of those at a time, all in
eager mode -- the gap already shows in eager, so compilation is not it --
and ``--report`` lines the global attention up against the profile CSV:

    profile       the profiler's own engine shape, reproduced
    profile16k    same, max_model_len 16384           (does max_model_len matter?)
    allglobal16k  34 layers, every one global         (does the layer count?)
    hybrid16k     the real 5:1 model                  (does the window on the others?)
    hybrid        the real model, default max_model_len (= the bench run's shape)

The first run showed the layer count is what moves it. Two candidate fixes
for the profiler, each measured against hybrid16k:

    pattern6      6 layers, one real 5:1 repeat     (profile a block, not a layer)
    profile_flush one layer, L2 flushed every forward (profile a layer, cold cache)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import tempfile
from pathlib import Path

from bench.layercheck.compare import COMBINE_RE, KV_WRITE_RE, MAIN_RE, _split
from profiler.core.config import read_model_config
from profiler.core.hooks.batch import Shot

GRID = [(n, kv) for n in (1, 8, 32) for kv in (512, 2048, 8192)
        if n * kv <= 65536]  # n=32 x 8192 does not fit 34 global layers' KV

NO_WINDOW = {"sliding_window": None, "sliding_window_pattern": None}

CONFIGS = {
    "profile": dict(layers=1, all_global=True, max_model_len=None),
    "profile16k": dict(layers=1, all_global=True, max_model_len=16384),
    "allglobal16k": dict(layers=None, all_global=True, max_model_len=16384),
    "hybrid16k": dict(layers=None, all_global=False, max_model_len=16384),
    "hybrid": dict(layers=None, all_global=False, max_model_len=None),
    "pattern6": dict(layers=6, all_global=False, max_model_len=16384),
    "profile_flush": dict(layers=1, all_global=True, max_model_len=16384, flush=True),
}

# Bigger than any GPU's L2 we run on (RTX 4090: 72 MB).
FLUSH_MB = 256
TARGET = "hybrid16k"

OUT_DIR = Path("bench/results/layercheck/global_attn")
PROFILE_CSV = Path("profiler/perf/RTX4090/google/gemma-3-4b-it/bf16/tp1/attention.csv")


def _layer_types(model_config, cfg):
    n = cfg["layers"] or int(model_config["num_hidden_layers"])
    if cfg["all_global"]:
        return ["full_attention"] * n
    return list(model_config["layer_types"][:n])


def run(name, model_config_path, iterations):
    from vllm import LLM

    cfg = CONFIGS[name]
    model_config = read_model_config(Path(model_config_path))
    types = _layer_types(model_config, cfg)

    overrides = {"num_hidden_layers": len(types), "layer_types": types}
    if cfg["all_global"]:
        overrides.update(NO_WINDOW)

    tmpdir = Path(tempfile.mkdtemp(prefix="layercheck_model_"))
    (tmpdir / "config.json").write_text(json.dumps(model_config, indent=2))
    # The profiler's engine (profiler/core/config.py HOST_ENGINE_DEFAULTS and
    # meta.yaml engine_effective), including its +max_num_seqs MNBT bump.
    kwargs = dict(
        model=str(tmpdir), load_format="dummy", skip_tokenizer_init=True,
        generation_config="vllm", enable_prefix_caching=False,
        enforce_eager=True, dtype="bfloat16", seed=42,
        max_num_seqs=64, max_num_batched_tokens=2048 + 64,
        hf_overrides=overrides,
        worker_extension_cls="bench.layercheck.extension.Extension",
    )
    if cfg["max_model_len"]:
        kwargs["max_model_len"] = cfg["max_model_len"]

    results = {}
    try:
        llm = LLM(**kwargs)
        for n, kv in GRID:
            shot = Shot.attention(0, 0, n, kv)
            raw = llm.collective_rpc(
                "run_shot", args=(shot.as_dict(), f"n{n}_kv{kv}", iterations,
                                  FLUSH_MB if cfg.get("flush") else 0))
            results[f"{n},{kv}"] = _global_attention(raw[0], types)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps({"layer_types": types, "results": results}, indent=1))
    print(f"wrote {path}")


def _global_attention(res, types):
    """Mean global-layer attention per forward, and its split-kv share."""
    kernels = res["kernels"]
    it = len(res["forward_ms"])
    per = len(kernels) // it
    total, combine = [], []
    for i in range(it):
        chunk = [(k["name"], k["dur_us"]) for k in kernels[i * per:(i + 1) * per]]
        attn, _ = _split(chunk, MAIN_RE.search, COMBINE_RE.search, KV_WRITE_RE.search)
        comb, _ = _split([(n, d if COMBINE_RE.search(n) else 0.0) for n, d in chunk],
                         MAIN_RE.search, COMBINE_RE.search, KV_WRITE_RE.search)
        if len(attn) != len(types):
            raise ValueError(f"saw {len(attn)} attentions for {len(types)} layers")
        g = [j for j, t in enumerate(types) if t == "full_attention"]
        total.append(statistics.mean(attn[j] for j in g))
        combine.append(statistics.mean(comb[j] for j in g))
    return {"global_us": statistics.mean(total), "combine_us": statistics.mean(combine)}


def report():
    prof = {}
    for r in csv.DictReader(open(PROFILE_CSV)):
        if r["prefill_chunk"] == "0" and r["attn_type"] == "full":
            prof[(int(r["n_decode"]), int(r["kv_decode"]))] = float(r["time_us"])
    runs = {}
    for name in CONFIGS:
        path = OUT_DIR / f"{name}.json"
        if path.exists():
            runs[name] = {k: v["global_us"] for k, v in
                          json.loads(path.read_text())["results"].items()}
    if TARGET not in runs:
        raise SystemExit(f"run --config {TARGET} first: it is what the rest is measured against")

    cols = ["csv"] + [n for n in runs if n != TARGET]
    print(f"global attention, us per layer (eager); % = off from {TARGET}, the real model")
    print(f"{'n,kv':<9}{TARGET:>11}" + "".join(f"{c:>21}" for c in cols))
    worst = dict.fromkeys(cols, 0.0)
    for n, kv in GRID:
        key = f"{n},{kv}"
        target = runs[TARGET][key]
        cells = [f"{key:<9}{target:>11.1f}"]
        for c in cols:
            val = prof.get((n, kv)) if c == "csv" else runs[c].get(key)
            if val is None:
                cells.append(f"{'-':>21}")
                continue
            err = (val - target) / target * 100
            worst[c] = max(worst[c], abs(err))
            cells.append(f"{val:>13.1f} {err:>+6.1f}%")
        print("".join(cells))
    print(f"{'worst':<20}" + "".join(f"{worst[c]:>20.1f}%" for c in cols))


def main() -> int:
    ap = argparse.ArgumentParser(prog="bench.layercheck.global_attn")
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--model-config", default="configs/model/google/gemma-3-4b-it.json")
    ap.add_argument("--iterations", type=int, default=5)
    args = ap.parse_args()
    if args.report:
        report()
    elif args.config:
        run(args.config, args.model_config, args.iterations)
    else:
        ap.error("pass --config or --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
