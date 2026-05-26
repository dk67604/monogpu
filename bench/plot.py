#!/usr/bin/env python3
"""Offline plotting of monogpu benchmark results -> docs/img/*.png.

This is an ANALYSIS/repro tool, NOT part of the serving stack (router + workers +
launcher are shell + reused binaries, no Python). It is the one place Python is
used, deliberately, to render publication charts.

- QPS-sweep panel reads the saved JSON (bench/out/*qps*.json) dynamically.
- A/B + routing panels use the measured values recorded in docs/benchmarking.md
  (those runs were not --save-result; values are from the verified task logs).

Run:  python bench/plot.py     (needs matplotlib; numpy ships with torch)
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
IMG = os.path.join(HERE, "..", "docs", "img")
os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})


def save(name):
    p = os.path.join(IMG, name)
    plt.tight_layout(); plt.savefig(p); plt.close(); print("wrote", p)


def qps_sweep():
    data = {}
    for f in glob.glob(os.path.join(OUT, "*qps*.json")):
        d = json.load(open(f))
        m = d["model_id"].split("/")[-1]
        data.setdefault(m, []).append((float(d["request_rate"]), d["output_throughput"]))
    if not data:
        print("no sweep JSON found; skipping qps_sweep"); return
    plt.figure(figsize=(7, 4.2))
    for m, pts in sorted(data.items()):
        pts.sort()
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=m)
    plt.xlabel("request rate (req/s)"); plt.ylabel("output throughput (tok/s)")
    plt.title("QPS sweep — each model ALONE on the RTX 5080\n(others resident but idle; CUDA graphs OFF)")
    plt.legend(); save("qps_sweep.png")


def memory_split():
    # Controlled: same model, same concurrency (48), CUDA graphs ON; only the
    # GPU-memory-utilization differs. Verified A=12766, B=12838 -> no effect.
    labels = ["dedicated\n(gmu 0.85)", "memory-split\n(gmu 0.30)"]
    vals = [12766, 12838]
    plt.figure(figsize=(5.5, 4))
    bars = plt.bar(labels, vals, color=["#5eb0ef", "#34d399"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 120, f"{v:,}", ha="center")
    plt.ylim(0, 15000); plt.ylabel("output throughput (tok/s)")
    plt.title("Memory split has NO effect on throughput\n(0.5B, conc 48, CUDA graphs ON — controlled)")
    save("memory_split.png")


def prefix_cache():
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.8))
    for a, vals, ttl, col in [
        (ax[0], [1153, 2316], "Output throughput (tok/s, higher=better)", "#34d399"),
        (ax[1], [314, 113], "P99 TTFT (ms, lower=better)", "#5eb0ef"),
    ]:
        b = a.bar(["random", "shared\nprefix"], vals, color=["#94a3b8", col])
        for bb, v in zip(b, vals):
            a.text(bb.get_x() + bb.get_width() / 2, v * 1.02, str(v), ha="center")
        a.set_title(ttl)
    fig.suptitle("Prefix caching (APC), prefill-dominated: ~2x throughput, −64% P99 TTFT")
    save("prefix_cache.png")


def routing_regimes():
    regimes = ["small\n(fits everywhere)", "sweet spot", "oversized"]
    ca = [62.4, 55.2, 38.5]; rr = [60.6, 53.4, 46.3]
    x = np.arange(len(regimes)); w = 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, ca, w, label="cache_aware", color="#5eb0ef")
    plt.bar(x + w / 2, rr, w, label="round_robin", color="#f59e0b")
    plt.xticks(x, regimes); plt.ylabel("request throughput (req/s)")
    plt.title("Cache-aware routing is regime-dependent\n(tie · +3.5% within noise · −20% when overshot)")
    plt.legend(); save("routing_regimes.png")


if __name__ == "__main__":
    qps_sweep(); memory_split(); prefix_cache(); routing_regimes()
    print("done -> docs/img/")
