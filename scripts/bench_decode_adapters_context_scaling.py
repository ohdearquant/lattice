#!/usr/bin/env python3
"""Engine adapters + wrapper CLI for the `context_scaling` profile (issue
#813 step 2: migrate `bench_context_scaling.sh`, keeping its chart
renderer, `scripts/bench_context_scaling_chart.py`, unchanged).

This module does more than exec `bench_decode_harness.main()` (unlike the
apples_to_apples/apples_precise/q4_apples adapters) because the legacy
script's output contract is not just a printed report: it also writes a
`docs/bench_results/context_scaling.tsv` file in a specific
engine/context_tokens/slope_tok_s/... shape that
`bench_context_scaling_chart.py` consumes, and it supports `CONTEXTS`/
`RUNS`/`CHART_ONLY` environment-variable overrides. All warmup/repeat/
aggregation POLICY still lives in the harness/profile; this module only
adds the TSV-rendering + chart-invocation step the harness core does not
own (`render_report`/`aggregate` produce the harness's own report and raw
JSONL, not a chart-compatible TSV).

Run with (from repo root): `uv run --quiet --with mlx-lm python3
scripts/bench_decode_adapters_context_scaling.py [--contexts 64,128,256]
[--runs 5] [--chart-only] [--allow-missing-engine]` -- or via the thin shell
wrapper `scripts/bench_context_scaling.sh`, which forwards
`CONTEXTS`/`RUNS`/`CHART_ONLY` exactly as the legacy script did.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_decode_harness as harness  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_NAME = "context_scaling"
OUT_DIR = REPO_ROOT / "docs" / "bench_results"
DATA_TSV = OUT_DIR / "context_scaling.tsv"
CHART_PNG = OUT_DIR / "context_scaling_benchmark.png"
CHART_SCRIPT = REPO_ROOT / "scripts" / "bench_context_scaling_chart.py"

LAT_BIN = REPO_ROOT / "target" / "release" / "bench_decode_ab"
MODEL_DIR = Path.home() / ".lattice" / "models" / "qwen3.5-0.8b"

_RESULT_RE = re.compile(r"^RESULT n_req=(\d+) completion=(\d+) total_ms=([\d.]+)$")


def parse_lattice_result_line(line: str) -> tuple[int, int, float] | None:
    m = _RESULT_RE.match(line)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


class LatticeUnavailableError(RuntimeError):
    """`bench_decode_ab` is not built, or the model dir is missing."""


class LatticeResultError(RuntimeError):
    """See the apples_to_apples adapter's identically-named exception."""


def extract_single_result(stdout: str, *, n_tokens: int) -> tuple[int, int, float]:
    matches = [
        parsed
        for parsed in (parse_lattice_result_line(line) for line in stdout.splitlines())
        if parsed is not None
    ]
    if len(matches) == 0:
        raise LatticeResultError(f"lattice: no full-line RESULT record found in stdout: {stdout[-500:]!r}")
    if len(matches) > 1:
        raise LatticeResultError(
            f"lattice: expected exactly one RESULT record for BENCH_RUNS=1, found {len(matches)}: {matches!r}"
        )
    n_req, completion, total_ms = matches[0]
    if n_req != n_tokens:
        raise LatticeResultError(
            f"lattice: RESULT n_req={n_req} does not match the requested window n_tokens={n_tokens}"
        )
    if completion < 0:
        raise LatticeResultError(f"lattice: RESULT completion={completion} is negative")
    if not math.isfinite(total_ms) or total_ms < 0:
        raise LatticeResultError(f"lattice: RESULT total_ms={total_ms} is not finite and non-negative")
    return n_req, completion, total_ms


class LatticeAdapter:
    def __init__(self, bin_path: Path = LAT_BIN, model_dir: Path = MODEL_DIR):
        self.bin_path = bin_path
        self.model_dir = model_dir

    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> harness.AdapterRunResult:
        if not self.model_dir.is_dir():
            raise LatticeUnavailableError(f"lattice: model dir missing ({self.model_dir})")
        env = os.environ.copy()
        env["BENCH_N"] = str(n_tokens)
        env["BENCH_RUNS"] = "1"
        env["LATTICE_MODEL_DIR"] = str(self.model_dir)
        proc = subprocess.run(
            [str(self.bin_path)], env=env, capture_output=True, text=True, timeout=600, check=False
        )
        if proc.returncode != 0:
            raise LatticeUnavailableError(
                f"lattice: bench_decode_ab exited {proc.returncode}: {proc.stderr.strip()[-2000:]}"
            )
        try:
            _n_req, completion, total_ms = extract_single_result(proc.stdout, n_tokens=n_tokens)
        except LatticeResultError as exc:
            raise LatticeResultError(f"{exc} (stderr: {proc.stderr.strip()[-500:]})") from exc
        return harness.AdapterRunResult(
            actual_completion_tokens=completion,
            native_ns=round(total_ms * 1e6),
            engine_version="lattice-bench_decode_ab",
        )


def lattice_available(bin_path: Path = LAT_BIN, model_dir: Path = MODEL_DIR) -> bool:
    return bin_path.is_file() and os.access(bin_path, os.X_OK) and model_dir.is_dir()


def build_lattice_binary_if_missing() -> None:
    """Mirrors the legacy script's own `if [[ ! -x "$LAT_BIN" ]]; then cargo
    build ...`. Best-effort: failures are swallowed here exactly like the
    legacy script's own `2>/dev/null` did, and surface later as a normal
    missing-adapter skip."""
    if LAT_BIN.is_file() and os.access(LAT_BIN, os.X_OK):
        return
    print("Building bench_decode_ab (release)...")
    subprocess.run(
        [
            "cargo", "build", "--release", "-p", "lattice-inference",
            "--bin", "bench_decode_ab", "--features", "f16,metal-gpu",
        ],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )


# --------------------------------------------------------------------------
# ollama
# --------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("LATTICE_BENCH_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL_TAG = "qwen3.5:0.8b"


class OllamaResponseError(RuntimeError):
    pass


_REQUIRED_OLLAMA_FIELDS = ("eval_count", "eval_duration", "total_duration")


def ollama_response_to_result(data: dict) -> harness.AdapterRunResult:
    """Mirrors `bench_context_scaling.sh`'s `ollama_median()`: its median is
    over `total_duration/1e6` (ms) values -- the SAME primary field as the
    apples_to_apples/apples_precise adapters, just pre-divided by run count
    in the legacy shell function. `eval_count`/`eval_duration` are not used
    by the legacy script's context-scaling ollama path, but are validated
    here anyway for response-shape integrity (same contract as the other
    two ollama adapters in this consolidation)."""
    if "error" in data:
        raise OllamaResponseError(f"ollama: response is an error body: {data['error']!r}")
    for field in _REQUIRED_OLLAMA_FIELDS:
        if field not in data:
            raise OllamaResponseError(f"ollama: response missing required field {field!r}: {data!r}")
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise OllamaResponseError(
                f"ollama: response field {field!r} must be a non-boolean int, got {type(value).__name__}: {data!r}"
            )
    eval_count = data["eval_count"]
    eval_duration = data["eval_duration"]
    total_duration = data["total_duration"]
    if eval_count < 0:
        raise OllamaResponseError(f"ollama: eval_count={eval_count!r} is negative: {data!r}")
    if eval_duration < 0:
        raise OllamaResponseError(f"ollama: eval_duration={eval_duration!r} is negative: {data!r}")
    if total_duration <= 0:
        raise OllamaResponseError(f"ollama: total_duration={total_duration!r} must be positive: {data!r}")
    return harness.AdapterRunResult(
        actual_completion_tokens=int(eval_count),
        native_ns=int(total_duration),
        engine_version="ollama",
    )


class OllamaAdapter:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url

    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> harness.AdapterRunResult:
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": n_tokens, "temperature": 0, "seed": 42},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return ollama_response_to_result(data)


def ollama_available(base_url: str = OLLAMA_BASE_URL, *, model_tag: str = OLLAMA_MODEL_TAG) -> bool:
    """Mirrors the legacy script's ollama preflight: binary + model pulled +
    server reachable. Unlike the legacy script's `command -v ollama &&
    curl ... | python3 -c "..."` one-liner, this does not attempt to start a
    down server (the legacy context-scaling script's own preflight is a pure
    check, not a start-if-down like bench_apples_to_apples.sh's)."""
    if shutil.which("ollama") is None:
        return False
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3).read()
    except (urllib.error.URLError, OSError):
        return False
    try:
        listed = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return model_tag in listed.stdout


# --------------------------------------------------------------------------
# mlx
# --------------------------------------------------------------------------


class MlxAdapter:
    """Invokes `mlx_lm`, mirroring `bench_context_scaling.sh`'s MLX heredoc:
    load `Qwen/Qwen3.5-0.8B` from the Hub directly (the legacy heredoc never
    tries the local dir first, unlike the other three scripts), quantize to
    8 bits, one 4-token warmup (the profile's `warmup_repeats=1`/
    `warmup_tokens=4`), then time only `generate()` per measured call."""

    def __init__(self, hub_model_id: str = "Qwen/Qwen3.5-0.8B"):
        self.hub_model_id = hub_model_id
        self._cache: tuple[object, object, object] | None = None

    def _load(self):
        if self._cache is not None:
            return self._cache
        import mlx.core as mx
        import mlx.nn as nn
        import mlx_lm

        mlx_model, tok = mlx_lm.load(self.hub_model_id, tokenizer_config={"eos_token": "<|endoftext|>"})
        nn.quantize(mlx_model, bits=8, group_size=64)
        mx.eval(mlx_model.parameters())
        self._cache = (mlx_model, tok, mx)
        return self._cache

    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> harness.AdapterRunResult:
        import mlx_lm

        mlx_model, tok, mx = self._load()
        greedy = lambda logits: mx.argmax(logits, axis=-1)  # noqa: E731 -- matches legacy heredoc's own lambda
        mlx_lm.generate(mlx_model, tok, prompt=prompt, max_tokens=n_tokens, sampler=greedy, verbose=False)
        mx.eval(mx.zeros(1))
        return harness.AdapterRunResult(actual_completion_tokens=n_tokens, engine_version="mlx_lm")


def mlx_available() -> bool:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def register_available_adapters() -> None:
    if lattice_available():
        harness.register_adapter("lattice", LatticeAdapter())
        print("  lattice: adapter registered")
    else:
        print(f"  lattice: BIN MISSING or MODEL MISSING ({LAT_BIN}, {MODEL_DIR}) — skipping")

    if ollama_available():
        harness.register_adapter("ollama", OllamaAdapter())
        print("  ollama: adapter registered")
    else:
        print("  ollama: not running or qwen3.5:0.8b not pulled — skipping")

    if mlx_available():
        harness.register_adapter("mlx", MlxAdapter())
        print("  mlx: adapter registered")
    else:
        print("  mlx: mlx_lm not importable — run via `uv run --with mlx-lm ...` — skipping")


# --------------------------------------------------------------------------
# TSV rendering (the legacy output contract bench_context_scaling_chart.py
# consumes) + chart invocation
# --------------------------------------------------------------------------


def render_legacy_tsv(run_result: harness.HarnessRunResult, slopes: list[harness.SlopeResult]) -> str:
    """`engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns` -- the exact
    header/column shape `bench_context_scaling_chart.py` parses (it only
    reads columns 0/1/2 positionally, but the full shape is reproduced for
    anyone reading the raw TSV directly, matching the legacy script)."""
    lines = ["engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns"]
    baseline_window = run_result.profile.windows[0]
    for s in slopes:
        baseline_vals = harness._measured_elapsed_seconds(run_result.observations, s.engine, baseline_window)
        window_vals = harness._measured_elapsed_seconds(run_result.observations, s.engine, s.window)
        t1_ms = (sum(baseline_vals) / len(baseline_vals) * 1000) if baseline_vals else float("nan")
        t2_ms = (sum(window_vals) / len(window_vals) * 1000) if window_vals else float("nan")
        lines.append(f"{s.engine}\t{s.window}\t{s.slope_tok_per_s:.1f}\t{t1_ms:.3f}\t{t2_ms:.3f}\t{s.window_n}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="context_scaling profile wrapper (issue #813)")
    parser.add_argument("--contexts", default=None, help="Comma-separated context lengths, e.g. 64,128,256")
    parser.add_argument("--runs", type=int, default=None, help="Measured repeats per window (default: profile's)")
    parser.add_argument("--chart-only", action="store_true", help="Regenerate the chart from the existing TSV only")
    parser.add_argument("--allow-missing-engine", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="Raw-observation JSONL path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.chart_only:
        print(f"=== Regenerating chart from {DATA_TSV} ===")
        proc = subprocess.run(
            [sys.executable, str(CHART_SCRIPT), str(DATA_TSV), str(CHART_PNG)], cwd=REPO_ROOT, check=False
        )
        return proc.returncode

    _schema_version, profiles = harness.load_profiles_file(harness.DEFAULT_PROFILES_FILE)
    profile = profiles[PROFILE_NAME]

    # CONTEXTS/RUNS overrides (legacy env-var knobs, exposed here as flags so
    # the shell wrapper can forward CONTEXTS/RUNS verbatim): rebuild an
    # equivalent ProfileConfig with N1=8 (unchanged) followed by the
    # requested context lengths, and/or the requested measured_repeats.
    if args.contexts is not None:
        contexts = tuple(sorted(int(c) for c in args.contexts.split(",") if c.strip()))
        profile = dataclasses.replace(profile, windows=(8, *contexts))
    if args.runs is not None:
        profile = dataclasses.replace(profile, measured_repeats=args.runs)

    build_lattice_binary_if_missing()
    register_available_adapters()

    try:
        result = harness.run_profile(profile, harness.ADAPTER_REGISTRY, allow_missing_engine=args.allow_missing_engine)
    except harness.MissingEngineError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    slopes = harness.aggregate(result)
    print(harness.render_report(result, slopes))

    if args.out is not None:
        harness.write_jsonl(list(result.observations), args.out)
        print(f"\nRaw observations: {args.out}")

    DATA_TSV.write_text(render_legacy_tsv(result, slopes))
    print(f"Raw data (chart-compatible TSV): {DATA_TSV}")

    print("\n--- Generating chart ---")
    subprocess.run([sys.executable, str(CHART_SCRIPT), str(DATA_TSV), str(CHART_PNG)], cwd=REPO_ROOT, check=False)
    print(f"Chart: {CHART_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
