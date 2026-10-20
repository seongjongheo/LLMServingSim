"""Step 2b of the per-layer check: vLLM's kernels against the simulator's layers.

    python -m bench.layercheck.compare

Reads ``smoke_{eager,piecewise}.json`` (from ``smoke.py``) and the simulator
CSV (from ``sim_layers.py``) and lines them up on the boundary both sides
share: attention. Piecewise CUDA graphs are cut there, and compilation fuses
the smaller layers into one another, so per layer only attention exists on
the vLLM side. Everything between two attentions is one *piece* -- the tail
of block i-1 plus the head of block i -- and the simulator's rows are summed
over exactly the same span.

Standard library only, so it runs wherever the JSON is.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# One attention = KV write + the kernel + (split-kv only) its combine pass.
MAIN_RE = re.compile(r"flash_fwd(?!_splitkv_combine)")
COMBINE_RE = re.compile(r"flash_fwd_splitkv_combine")
KV_WRITE_RE = re.compile(r"reshape_and_cache")


def _split(names_durs, is_main, is_after, is_before):
    """Assign each (name, dur) to attention i or piece i, in order.

    Returns ``(attn, pieces)``: per-attention and per-piece summed duration.
    ``is_before`` kernels belong to the attention that follows them,
    ``is_after`` ones to the attention that precedes them.
    """
    attn, pieces = defaultdict(float), defaultdict(float)
    p = 0
    for name, dur in names_durs:
        if is_main(name):
            attn[p] += dur
            p += 1
        elif is_after(name) and p > 0:
            attn[p - 1] += dur
        elif is_before(name):
            attn[p] += dur
        else:
            pieces[p] += dur
    return [attn[i] for i in range(p)], [pieces[i] for i in range(p + 1)]


def vllm_side(res):
    kernels = res["kernels"]
    it = len(res["forward_ms"])
    per = len(kernels) // it
    if per * it != len(kernels):
        raise ValueError(f"{len(kernels)} kernels do not split into {it} forwards")
    runs = []
    for i in range(it):
        chunk = [(k["name"], k["dur_us"]) for k in kernels[i * per:(i + 1) * per]]
        runs.append(_split(chunk, MAIN_RE.search, COMBINE_RE.search, KV_WRITE_RE.search))
    n_attn = {len(a) for a, _ in runs}
    if len(n_attn) != 1:
        raise ValueError(f"forwards disagree on attention count: {n_attn}")
    attn = [statistics.mean(r[0][j] for r in runs) for j in range(len(runs[0][0]))]
    pieces = [statistics.mean(r[1][j] for r in runs) for j in range(len(runs[0][1]))]
    return attn, pieces


def sim_side(rows):
    seq = [(r["layer"], float(r["time_us"])) for r in rows]
    return _split(seq, lambda n: n == "attention", lambda n: False, lambda n: False)


def _err(sim, ref):
    return f"{(sim - ref) / ref * 100:+6.1f}%" if ref else "    n/a"


def main() -> int:
    ap = argparse.ArgumentParser(prog="bench.layercheck.compare")
    ap.add_argument("--vllm-dir", default="bench/results/layercheck")
    ap.add_argument("--sim-csv", default=str(HERE / "sim_RTX4090_gemma-3-4b-it.csv"))
    args = ap.parse_args()

    sim_rows = defaultdict(list)
    local = {}
    for r in csv.DictReader(open(args.sim_csv)):
        sim_rows[r["shot"]].append(r)
        if r["layer"] == "attention":
            local.setdefault(r["shot"], []).append(r["is_local"] == "1")

    modes = {}
    for m in ("eager", "piecewise"):
        path = Path(args.vllm_dir) / f"smoke_{m}.json"
        if path.exists():
            modes[m] = json.loads(path.read_text())

    print("us per forward; err = (sim - vLLM) / vLLM")
    for shot, rows in sim_rows.items():
        s_attn, s_pieces = sim_side(rows)
        is_local = local[shot]
        v = {m: vllm_side(d[shot]) for m, d in modes.items() if shot in d}
        bad = [m for m, (a, _) in v.items() if len(a) != len(s_attn)]
        if bad:
            print(f"\n== {shot}: attention count differs in {bad} -- skipped")
            continue

        def line(label, pick):
            sim_val = pick(s_attn, s_pieces)
            cells = [f"{label:<16}{sim_val:>9.1f}"]
            for m in modes:
                ref = pick(*v[m])
                cells.append(f"{ref:>9.1f} {_err(sim_val, ref)}")
            print("  ".join(cells))

        def mean_of(xs, mask):
            sel = [x for x, k in zip(xs, mask) if k]
            return statistics.mean(sel) if sel else 0.0

        n = len(s_attn)
        print(f"\n== {shot}")
        print(f"{'':<16}{'sim':>9}  " + "  ".join(f"{m:>17}" for m in modes))
        line("attn local", lambda a, p: mean_of(a, is_local))
        line("attn global", lambda a, p: mean_of(a, [not x for x in is_local]))
        line("piece (middle)", lambda a, p: statistics.mean(p[1:n]))
        line("piece first", lambda a, p: p[0])
        line("piece last", lambda a, p: p[n])
        line("total", lambda a, p: sum(a) + sum(p))
        for m in modes:
            a = v[m][0]
            worst = max(range(n), key=lambda j: abs(s_attn[j] - a[j]) / a[j] if a[j] else 0)
            print(f"  worst attn layer ({m}): #{worst} "
                  f"{'local' if is_local[worst] else 'global'} "
                  f"sim {s_attn[worst]:.1f} vs {a[worst]:.1f} {_err(s_attn[worst], a[worst])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
