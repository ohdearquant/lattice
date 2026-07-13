#!/usr/bin/env python3
"""Engine adapters for the `apples_to_apples_q8`/`apples_to_apples_q4`
profiles (issue #813 step 2: migrate `bench_apples_to_apples.sh`).

This is the "thin wrapper that execs the harness" the issue calls for: it
registers three `EngineAdapter`s (lattice, ollama, mlx) that invoke exactly
what `bench_apples_to_apples.sh`'s `bench_lattice`/`bench_ollama`/`bench_mlx`
shell functions invoked, then delegates argument parsing and execution to
`bench_decode_harness.main()`. All repeat-count, warmup, and verdict
decisions stay in the harness/profile (`bench_decode_profiles.toml`); this
module owns invocation and parsing only, per `EngineAdapter`'s contract.

Known, disclosed methodology note (lattice only): `bench_decode_ab` amortizes
one model load across `BENCH_RUNS` internal repeats within a single process
(its own per-run timer starts AFTER load, so load never enters any of its
`total_ms` values). The harness's contract is one `EngineAdapter.run()` call
per measured repeat, each independently wall-clock-timed by the harness
itself around the call -- there is no persistent-server mode for
`bench_decode_ab` to reuse a loaded model across separate harness-driven
calls, so this adapter invokes it with `BENCH_RUNS=1` per call. Each of the
five `elapsed_ns` measurements per window therefore includes one subprocess
spawn + model load + `bench_decode_ab`'s own internal untimed warmup, which
the legacy script's `total_ms` excluded entirely. The engine's own
decode-only `total_ms` (methodology-identical to the legacy metric) is
preserved losslessly via `AdapterRunResult.native_ns`, reported as the
harness's separate "native tok/s" diagnostic column -- this is exactly the
distinction `Observation.elapsed_ns` vs `Observation.engine_native_ns` exists
for. Closing this gap fully would need a persistent/server mode for
`bench_decode_ab`, which is out of scope for this migration (no Rust source
under `crates/inference/` is touched here). Ollama and MLX do not have this
issue: ollama's harness call is a single HTTP round trip (methodology-
identical to the legacy script's `curl` call), and MLX's model is loaded and
quantized once, in-process, lazily on this module's first call for a given
(model, quantization) pair -- reproducing the legacy script's one-load-per-
tier shape exactly, since `generate()` alone is timed per call, same as the
legacy script's own `t0 = time.time(); generate(...); dt = ...`.

Run with (from repo root): `uv run --quiet --with mlx-lm python3
scripts/bench_decode_adapters_apples_to_apples.py run --profile
apples_to_apples_q8 --allow-missing-engine` (mlx-lm must be available in the
invoking environment since the mlx engine is present in both profiles; the
`--with mlx-lm` flag is how `scripts/bench_apples_to_apples.sh` provides it,
matching the legacy script's own `uv run --with mlx-lm` invocation).
"""

from __future__ import annotations

import json
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

# -- lattice: exactly bench_apples_to_apples.sh's Q8_DIR/Q4_DIR/LAT_BIN --
LAT_BIN = REPO_ROOT / "target" / "release" / "bench_decode_ab"
Q8_MODEL_DIR = Path.home() / ".lattice" / "models" / "qwen3.5-0.8b"
Q4_MODEL_DIR = Path.home() / ".lattice" / "models" / "qwen3.5-0.8b-q4-quarot"
# The legacy script's Q4 lattice invocation keeps LATTICE_TOKENIZER_DIR
# pointed at Q8_DIR even for the Q4 tier (`bench_lattice "q4" "$Q4_DIR"
# "$Q8_DIR" "LATTICE_QUANT_FORMAT=Q4"`); preserved verbatim here.
_LATTICE_MODEL_DIRS: dict[str, Path] = {
    "qwen3.5-0.8b": Q8_MODEL_DIR,
    "qwen3.5-0.8b-q4-quarot": Q4_MODEL_DIR,
}

# -- ollama: exactly bench_apples_to_apples.sh's model_tag / API shape --
OLLAMA_BASE_URL = os.environ.get("LATTICE_BENCH_OLLAMA_URL", "http://localhost:11434")

_RESULT_RE = re.compile(r"RESULT n_req=(\d+) completion=(\d+) total_ms=([\d.]+)")


def parse_lattice_result_line(line: str) -> tuple[int, int, float] | None:
    """Parse one `bench_decode_ab` stdout line. Returns `(n_req, completion,
    total_ms)` or `None` if the line is not a RESULT line (the binary also
    prints `[bench] ...` progress lines to stderr, and may print other
    stdout noise -- callers scan all stdout lines and use the last match,
    mirroring the legacy awk filter `/^RESULT/`).
    """
    m = _RESULT_RE.search(line)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


class LatticeUnavailableError(RuntimeError):
    """`bench_decode_ab` is not built, or the requested model dir is missing."""


class LatticeAdapter:
    """Invokes `target/release/bench_decode_ab`, mirroring
    `bench_apples_to_apples.sh`'s `bench_lattice()` exactly, one measured
    repeat per call (`BENCH_RUNS=1`) -- see module docstring for the
    disclosed model-load-per-call methodology note.
    """

    def __init__(self, bin_path: Path = LAT_BIN, tokenizer_dir: Path = Q8_MODEL_DIR):
        self.bin_path = bin_path
        self.tokenizer_dir = tokenizer_dir

    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> harness.AdapterRunResult:
        model_dir = _LATTICE_MODEL_DIRS.get(model)
        if model_dir is None:
            raise LatticeUnavailableError(f"lattice: no known model dir for model={model!r}")
        if not model_dir.is_dir():
            raise LatticeUnavailableError(f"lattice: model dir missing ({model_dir})")

        env = os.environ.copy()
        env["BENCH_N"] = str(n_tokens)
        env["BENCH_RUNS"] = "1"
        env["LATTICE_MODEL_DIR"] = str(model_dir)
        env["LATTICE_TOKENIZER_DIR"] = str(self.tokenizer_dir)
        if quantization == "q4":
            env["LATTICE_QUANT_FORMAT"] = "Q4"

        proc = subprocess.run(
            [str(self.bin_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if proc.returncode != 0:
            raise LatticeUnavailableError(
                f"lattice: bench_decode_ab exited {proc.returncode}: {proc.stderr.strip()[-2000:]}"
            )
        parsed = None
        for out_line in proc.stdout.splitlines():
            result = parse_lattice_result_line(out_line)
            if result is not None:
                parsed = result
        if parsed is None:
            raise LatticeUnavailableError(
                "lattice: no RESULT line in bench_decode_ab stdout "
                f"(stderr: {proc.stderr.strip()[-500:]})"
            )
        _n_req, completion, total_ms = parsed
        return harness.AdapterRunResult(
            actual_completion_tokens=completion,
            native_ns=round(total_ms * 1e6),
            engine_version="lattice-bench_decode_ab",
        )


def lattice_available(bin_path: Path = LAT_BIN) -> bool:
    return bin_path.is_file() and os.access(bin_path, os.X_OK)


# --------------------------------------------------------------------------
# ollama
# --------------------------------------------------------------------------


def ollama_response_to_result(data: dict, *, requested_tokens: int) -> harness.AdapterRunResult:
    """Pure translation of one `/api/generate` JSON body into an
    `AdapterRunResult`, mirroring the legacy script's
    `ec=d.get('eval_count',0); ed=d.get('eval_duration',1)/1e9; ec/ed` native
    rate. `eval_duration` is already nanoseconds in ollama's response, so it
    is used directly as `native_ns` (no unit conversion needed, unlike the
    legacy script's own `/1e9` which converts to seconds only for its
    printed ratio).
    """
    eval_count = data.get("eval_count", 0) or 0
    eval_duration = data.get("eval_duration")
    native_ns = eval_duration if eval_duration else None
    return harness.AdapterRunResult(
        actual_completion_tokens=eval_count or requested_tokens,
        native_ns=native_ns,
        engine_version="ollama",
    )


class OllamaAdapter:
    """Invokes ollama's `/api/generate`, mirroring
    `bench_apples_to_apples.sh`'s `bench_ollama()` exactly: one HTTP call per
    measured repeat (the shell script also issues no warmup for ollama, and
    neither does this profile)."""

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
        return ollama_response_to_result(data, requested_tokens=n_tokens)


def ollama_available(
    base_url: str = OLLAMA_BASE_URL, *, model_tag: str, start_if_down: bool = True
) -> bool:
    """Mirrors `bench_apples_to_apples.sh`'s `bench_ollama()` preflight:
    binary installed, model pulled (best-effort pull if missing), server
    reachable (best-effort `ollama serve &` + 3s settle if not)."""
    if shutil.which("ollama") is None:
        return False
    try:
        listed = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if model_tag not in listed.stdout:
        pulled = subprocess.run(
            ["ollama", "pull", model_tag], capture_output=True, text=True, timeout=600, check=False
        )
        if pulled.returncode != 0:
            return False
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3).read()
        return True
    except (urllib.error.URLError, OSError):
        if not start_if_down:
            return False
        try:
            subprocess.Popen(
                ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
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
    """Invokes `mlx_lm` in-process, mirroring `bench_apples_to_apples.sh`'s
    `bench_mlx()`: load the Q8_DIR safetensors model once, `nn.quantize` it
    to the requested bit width once, run ONE 8-token warmup, then time only
    `generate()` per measured call -- model load/quantize is cached per
    `(model, quantization)` pair across calls within this process, exactly
    reproducing the legacy script's one-load-per-tier shape (a fresh
    process per profile run keeps Q8 and Q4 tiers independent, matching the
    legacy script's separate `uv run` invocation per tier).
    """

    def __init__(self, source_model_dir: Path = Q8_MODEL_DIR):
        self.source_model_dir = source_model_dir
        self._cache: dict[tuple[str, str], tuple[object, object, object]] = {}

    def _load(self, model: str, quantization: str):
        key = (model, quantization)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        bits = 4 if quantization == "q4" else 8
        try:
            mlx_model, tok = load(str(self.source_model_dir))
        except Exception:  # noqa: BLE001 -- legacy script's own bare except fallback
            mlx_model, tok = load("Qwen/Qwen3.5-0.8B")
        nn.quantize(mlx_model, bits=bits, group_size=64)
        mx.eval(mlx_model.parameters())
        sampler = make_sampler(temp=0.0)
        self._cache[key] = (mlx_model, tok, sampler)
        return mlx_model, tok, sampler

    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> harness.AdapterRunResult:
        from mlx_lm import generate

        mlx_model, tok, sampler = self._load(model, quantization)
        result = generate(
            mlx_model, tok, prompt=prompt, max_tokens=n_tokens, sampler=sampler, verbose=False
        )
        actual_tokens = len(tok.encode(result)) if isinstance(result, str) else n_tokens
        return harness.AdapterRunResult(
            actual_completion_tokens=actual_tokens, engine_version="mlx_lm"
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
    """Registers whichever of lattice/ollama/mlx are actually available,
    printing a legacy-style message for each -- mirroring
    `bench_apples_to_apples.sh`'s per-engine graceful skip (missing binary /
    missing model dir / ollama not installed all print-and-continue rather
    than aborting the whole run)."""
    if lattice_available():
        harness.register_adapter("lattice", LatticeAdapter())
        print("  lattice: adapter registered")
    else:
        print(
            f"  lattice: BIN MISSING ({LAT_BIN}) — build first "
            "(cargo build --release --bin bench_decode_ab)"
        )

    if ollama_available(model_tag="qwen3.5:0.8b"):
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
