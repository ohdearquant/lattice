#!/usr/bin/env python3
"""Generate a context-scaling bar chart from the profiled harness TSV output."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROVENANCE_TAG = "# context-scaling-provenance"
PROVENANCE_PREFIX = PROVENANCE_TAG + ": "
PROVENANCE_FIELDS = {
    "schema_version", "harness", "observation_schema_versions", "git_shas",
    "hardware_ids", "observation_timestamps",
}


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid observation timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"observation timestamp needs a timezone: {value!r}")
    return parsed


def validate_provenance(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != PROVENANCE_FIELDS:
        raise ValueError("provenance must contain exactly the versioned metadata fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported context-scaling provenance schema version")
    if value["harness"] != "bench_decode_harness.py":
        raise ValueError("unrecognized provenance harness")
    versions = value["observation_schema_versions"]
    if (not isinstance(versions, list) or not versions
            or any(type(version) is not int or version <= 0 for version in versions)
            or len(set(versions)) != len(versions)):
        raise ValueError("observation schema versions must be distinct positive integers")
    for field in ("git_shas", "hardware_ids", "observation_timestamps"):
        values = value[field]
        if (not isinstance(values, list) or not values
                or any(not isinstance(item, str) or not item.strip() for item in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{field} must contain distinct nonempty recorded strings")
    for timestamp in value["observation_timestamps"]:
        _timestamp(timestamp)
    return value


def provenance_comment(value: dict) -> str:
    return PROVENANCE_PREFIX + json.dumps(
        validate_provenance(value), ensure_ascii=True, allow_nan=False, sort_keys=True
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate provenance field: {key}")
        result[key] = value
    return result


def parse_tsv(text: str) -> tuple[dict[str, dict[int, float]], dict | None]:
    lines = text.splitlines()
    if not lines or lines[0].split("\t")[:3] != ["engine", "context_tokens", "slope_tok_s"]:
        raise ValueError("context-scaling TSV header is missing")
    engines: dict[str, dict[int, float]] = {}
    provenance = None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith(PROVENANCE_TAG):
            if provenance is not None:
                raise ValueError("duplicate or conflicting context-scaling provenance records")
            if not stripped.startswith(PROVENANCE_PREFIX):
                raise ValueError("malformed context-scaling provenance record")
            provenance = validate_provenance(json.loads(
                stripped[len(PROVENANCE_PREFIX):], object_pairs_hook=_unique_object
            ))
        elif stripped.startswith("#") or not stripped:
            continue
        else:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            eng, ctx, slope = parts[0], int(parts[1]), float(parts[2])
            engines.setdefault(eng, {})[ctx] = slope
    if not engines:
        raise ValueError("no data found in TSV")
    return engines, provenance


def provenance_caption(provenance: dict | None) -> str:
    if provenance is None:
        return "Legacy TSV: measurement timestamps, hardware and harness revision are unknown."
    validate_provenance(provenance)
    timestamps = provenance["observation_timestamps"]
    first, last = min(timestamps, key=_timestamp), max(timestamps, key=_timestamp)
    recorded = first if first == last else f"{first} to {last}"
    hardware = "; ".join(json.dumps(item, ensure_ascii=True) for item in provenance["hardware_ids"])
    revisions = "; ".join(json.dumps(item, ensure_ascii=True) for item in provenance["git_shas"])
    return (
        f"Recorded observations: {recorded}\n"
        f"Recorded hardware IDs: {hardware}\n"
        f"Harness: {provenance['harness']} at recorded revisions: {revisions}\n"
        f"Observation schemas: {provenance['observation_schema_versions']} · Provenance schema: 1"
    )


def render_chart(engines: dict[str, dict[int, float]], provenance: dict | None, chart_path: Path) -> None:
    caption = provenance_caption(provenance)
    import matplotlib.pyplot as plt
    import numpy as np

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
        "Slope Method (N1=8) · Greedy · Median",
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

    fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=8, parse_math=False, usetex=False)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {chart_path}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(f"usage: {sys.argv[0]} <data.tsv> <output.png>", file=sys.stderr)
        return 1
    data_path, chart_path = map(Path, args)
    try:
        engines, provenance = parse_tsv(data_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Cannot render context-scaling chart: {exc}", file=sys.stderr)
        return 1
    render_chart(engines, provenance, chart_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
