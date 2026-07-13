#!/usr/bin/env python3
"""Engine adapters for the `apples_precise` profile (issue #813 step 2:
migrate `bench_apples_precise.sh`).

Registers three `EngineAdapter`s (lattice, ollama, mlx) that invoke exactly
what `bench_apples_precise.sh`'s inline lattice-loop/ollama-curl/mlx-`uv run`
sections invoked, then delegates argument parsing and execution to
`bench_decode_harness.main()`. All repeat-count, warmup, and
aggregation/CI decisions stay in the harness/profile
(`bench_decode_profiles.toml`); this module owns invocation and parsing
only, per `EngineAdapter`'s contract.

Disclosed methodology note (lattice only, same class as
`bench_decode_adapters_apples_to_apples.py`'s note): `bench_decode_ab`
amortizes one model load across `BENCH_RUNS` internal repeats within a
single process, timing only AFTER load. This adapter invokes it with
`BENCH_RUNS=1` per harness call (one call per measured/warmup repeat, each
independently wall-clock-timed by the harness), so each `elapsed_ns` here
additionally includes one subprocess spawn + model load that the legacy
script's own `BENCH_RUNS=$RUNS` single-process loop never paid per
observation. The engine's own decode-only `total_ms` is preserved
losslessly via `AdapterRunResult.native_ns` (the harness's "native tok/s"
diagnostic column).

Run with (from repo root): `uv run --quiet --with mlx-lm python3
scripts/bench_decode_adapters_apples_precise.py run --profile apples_precise
--allow-missing-engine`.
"""

from __future__ import annotations

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

# -- lattice: exactly bench_apples_precise.sh's LAT_BIN/MODEL_DIR --
LAT_BIN = REPO_ROOT / "target" / "release" / "bench_decode_ab"
MODEL_DIR = Path.home() / ".lattice" / "models" / "qwen3.5-0.8b"

_RESULT_RE = re.compile(r"^RESULT n_req=(\d+) completion=(\d+) total_ms=([\d.]+)$")


def parse_lattice_result_line(line: str) -> tuple[int, int, float] | None:
    """Same full-line RESULT contract as
    `bench_decode_adapters_apples_to_apples.py.parse_lattice_result_line`."""
    m = _RESULT_RE.match(line)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


class LatticeUnavailableError(RuntimeError):
    """`bench_decode_ab` is not built, or the model dir is missing."""


class LatticeResultError(RuntimeError):
    """`bench_decode_ab` ran and exited 0, but its RESULT output failed
    validation -- see `bench_decode_adapters_apples_to_apples.py`'s
    identically-named exception for the full rationale."""


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
    """Invokes `target/release/bench_decode_ab` against the Q8 safetensors
    dir, mirroring `bench_apples_precise.sh`'s lattice section (including its
    own untimed pre-loop warmup, now represented as the profile's
    `warmup_repeats=2`/`warmup_tokens=512` instead of the legacy script's
    inline `for _ in $(seq 1 $WARMUP)` loop)."""

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


# --------------------------------------------------------------------------
# ollama
# --------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("LATTICE_BENCH_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL_TAG = "qwen3.5:0.8b"


class OllamaResponseError(RuntimeError):
    """See `bench_decode_adapters_apples_to_apples.py`'s identically-named
    exception."""


_REQUIRED_OLLAMA_FIELDS = ("eval_count", "eval_duration", "total_duration")


def ollama_response_to_result(data: dict) -> harness.AdapterRunResult:
    """Mirrors `bench_apples_precise.sh`'s ollama Python inline block
    exactly: `tot = d.get('total_duration',0)/1e9` is the legacy script's
    PRIMARY metric (fed straight into `rows[eng][N]`), and
    `ec/ed = eval_count/eval_duration` was its separate native-rate column
    -- preserved here as `native_ns`/`AdapterRunResult` respectively, same
    split as the apples_to_apples adapter's identically-named function."""
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
    """Invokes ollama's `/api/generate`, one HTTP call per measured/warmup
    repeat -- matching `bench_apples_precise.sh`'s per-run curl loop
    (including its own inline 2-run warmup, represented here via the
    profile's `warmup_repeats=2`)."""

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
                "options": {"num_predict": n_tokens, "temperature": 0},
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
    if shutil.which("ollama") is None:
        return False
    try:
        listed = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if model_tag not in listed.stdout:
        pulled = subprocess.run(["ollama", "pull", model_tag], capture_output=True, text=True, timeout=600, check=False)
        if pulled.returncode != 0:
            return False
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3).read()
        return True
    except (urllib.error.URLError, OSError):
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return False
        time.sleep(3)
        try:
            urllib.request.urlopen(f"{base_url}/api/tags", timeout=3).read()
            return True
        except (urllib.error.URLError, OSError):
            return False


# --------------------------------------------------------------------------
# mlx
# --------------------------------------------------------------------------


class MlxAdapter:
    """Invokes `mlx_lm` in-process, mirroring `bench_apples_precise.sh`'s
    MLX heredoc: load the Q8 safetensors model once (falling back to the Hub
    id on failure, exactly as the legacy heredoc's bare `try/except`), quantize
    to 8 bits once, then time only `generate()` per call -- the profile's
    `warmup_repeats=2`/`warmup_tokens=512` reproduce the heredoc's own
    pre-loop `for _ in range(warmup): generate(...)`."""

    def __init__(self, source_model_dir: Path = MODEL_DIR):
        self.source_model_dir = source_model_dir
        self._cache: tuple[object, object, object, str | None] | None = None

    def _load(self):
        if self._cache is not None:
            return self._cache
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        fallback_model_id: str | None = None
        try:
            mlx_model, tok = load(str(self.source_model_dir))
        except Exception:  # noqa: BLE001 -- legacy heredoc's own bare except fallback
            mlx_model, tok = load("Qwen/Qwen3.5-0.8B")
            fallback_model_id = "Qwen/Qwen3.5-0.8B"
        nn.quantize(mlx_model, bits=8, group_size=64)
        mx.eval(mlx_model.parameters())
        sampler = make_sampler(temp=0.0)
        self._cache = (mlx_model, tok, sampler, fallback_model_id)
        return self._cache

    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> harness.AdapterRunResult:
        from mlx_lm import generate

        mlx_model, tok, sampler, fallback_model_id = self._load()
        generate(mlx_model, tok, prompt=prompt, max_tokens=n_tokens, sampler=sampler, verbose=False)
        return harness.AdapterRunResult(
            actual_completion_tokens=n_tokens,
            engine_version="mlx_lm",
            actual_model=fallback_model_id,
        )


def mlx_available() -> bool:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------
# registration + CLI delegation
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
        print("  ollama: not installed, unreachable, or model pull failed — skipping")

    if mlx_available():
        harness.register_adapter("mlx", MlxAdapter())
        print("  mlx: adapter registered")
    else:
        print("  mlx: mlx_lm not importable — run via `uv run --with mlx-lm ...` — skipping")


if __name__ == "__main__":
    register_available_adapters()
    sys.exit(harness.main())
