#!/usr/bin/env python3
"""Clean decode-throughput chart — fair end-to-end (slope) numbers only.

Run: uv run --with matplotlib python scripts/make_bench_chart.py
Out: docs/bench_results/lattice_benchmark.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

OUT = Path(__file__).resolve().parent.parent / "docs/bench_results/lattice_benchmark.png"

# Fair end-to-end decode throughput (slope method: (N2-N1)/(T(N2)-T(N1)),
# prefill / model-load / per-call overhead cancel). M2 Max, Qwen3.5-0.8B.
LATTICE = 157
OLLAMA = 84
RATIO = LATTICE / OLLAMA

INK = "#0B132B"
LAT = "#2563EB"   # lattice blue
OLL = "#94A3B8"   # ollama grey
BG = "#FFFFFF"

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=220)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

bars = ax.bar(
    ["Lattice\n(pure Rust + Metal)", "Ollama\n(llama.cpp Metal)"],
    [LATTICE, OLLAMA],
    width=0.52,
    color=[LAT, OLL],
    edgecolor="none",
    zorder=3,
)
for b, v in zip(bars, [LATTICE, OLLAMA]):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}",
            ha="center", va="bottom", fontsize=26, fontweight="bold", color=INK)

ax.set_ylim(0, LATTICE * 1.30)
ax.set_ylabel("Decode tokens / sec  (higher is better)", fontsize=12, color=INK)
ax.set_title("Qwen3.5-0.8B decode throughput — Apple M2 Max",
             fontsize=17, fontweight="bold", color=INK, pad=26)
ax.text(0.5, 1.045,
        "Fair end-to-end measurement (slope method, prefill excluded — identical for both engines)",
        transform=ax.transAxes, ha="center", fontsize=10.5, color="#475569", style="italic")

# Speedup callout
ax.annotate(
    f"{RATIO:.1f}× faster",
    xy=(0, LATTICE), xytext=(0.5, LATTICE * 1.16),
    ha="center", fontsize=15, fontweight="bold", color=LAT,
)

for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="both", length=0, labelsize=12, colors=INK)
ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.text(0.5, 0.015,
         "Qwen3.5-0.8B (873M, hybrid GatedDeltaNet + GQA) · greedy · median of 5 runs · 2026-05-16\n"
         "Reproduce: ./scripts/bench_apples_to_apples.sh   |   Note: Apple's MLX (Metal-native) "
         "decodes faster; see README for the full table & methodology.",
         ha="center", fontsize=8.2, color="#64748B")

fig.subplots_adjust(top=0.80, bottom=0.20, left=0.11, right=0.95)
fig.savefig(OUT, facecolor=BG)
print(f"wrote {OUT}")
