#!/usr/bin/env python3
"""Generate a context-scaling bar chart from the profiled harness TSV output."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <data.tsv> <output.png>")
        sys.exit(1)

    data_path, chart_path = Path(sys.argv[1]), Path(sys.argv[2])

    # Parse TSV: engine, context_tokens, slope_tok_s, ...
    engines: dict[str, dict[int, float]] = {}
    with open(data_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            eng, ctx, slope = parts[0], int(parts[1]), float(parts[2])
            engines.setdefault(eng, {})[ctx] = slope

    if not engines:
        print("No data found in TSV")
        sys.exit(1)

    # Collect all context lengths (sorted)
    all_ctx = sorted(set(c for d in engines.values() for c in d))

    # Engine display config
    config = {
        "mlx": ("MLX (Q8 g64, private AMX)", "#4A90D9"),
        "lattice": ("Lattice (Q8, f16 lm_head)", "#E85D3A"),
        "ollama": ("Ollama (Q8_0, llama.cpp)", "#7BC67E"),
    }

    # Order: mlx first (tallest), then lattice, then ollama
    display_order = [e for e in ["mlx", "lattice", "ollama"] if e in engines]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_ctx))
    n_engines = len(display_order)
    width = 0.7 / n_engines

    bar_groups = {}
    for i, eng in enumerate(display_order):
        label, color = config.get(eng, (eng, "#999999"))
        vals = [engines[eng].get(c, 0) for c in all_ctx]
        offset = (i - (n_engines - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        bar_groups[eng] = (bars, vals)

        # Value labels
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.annotate(
                    f"{v:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # Lattice vs Ollama ratio annotations
    if "lattice" in engines and "ollama" in engines:
        for i, c in enumerate(all_ctx):
            lat = engines["lattice"].get(c)
            oll = engines["ollama"].get(c)
            if lat and oll:
                ratio = lat / oll
                offset = (
                    display_order.index("lattice") - (n_engines - 1) / 2
                ) * width
                ax.annotate(
                    f"{ratio:.1f}× Ollama",
                    xy=(x[i] + offset, lat + 12),
                    ha="center",
                    fontsize=8,
                    color="#E85D3A",
                    fontweight="bold",
                )

    ax.set_xlabel("Generated Tokens (context length)", fontsize=12)
    ax.set_ylabel("Decode Throughput (tok/s)", fontsize=12)
    ax.set_title(
        "Qwen3.5-0.8B Decode Throughput vs Context Length\n"
        "Apple M2 Max · Slope Method (N1=8) · Greedy · Median",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(all_ctx)
    ax.legend(loc="upper right", fontsize=11)
    max_val = max(v for d in engines.values() for v in d.values())
    ax.set_ylim(0, max_val * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {chart_path}")


if __name__ == "__main__":
    main()
