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
under `crates/inference/` is touched here).

Duration-boundary parity for ollama and MLX (declared, post-hardening):
the harness's PRIMARY reported slope always uses its own wall-clock
`elapsed_ns` around the whole `adapter.run()` call -- this is a fixed,
intentional harness invariant (`AdapterRunResult.native_ns` is a diagnostic,
"never a replacement" for it), not something an adapter can override, so the
primary slope column is NOT claimed to be byte-for-byte identical to the
legacy script's per-engine self-timed metric. MLX's primary metric IS an
exact match regardless: this adapter times nothing beyond `generate()`
itself (no post-generation retokenization happens inside `run()` at all, see
`MlxAdapter.run()`), so the harness's wall-clock window equals the legacy
script's own `t0 = time.time(); generate(...); dt = ...` window exactly,
model load amortized identically via the same first-call warmup shape.
Ollama's primary metric is NOT byte-identical to the legacy script's
`total_duration`-based figure -- the harness's wall clock additionally
includes the local HTTP round trip's connection/JSON marshalling overhead,
which the legacy script's own duration source did not -- but the
legacy-equivalent number is not lost: ollama's `total_duration` (exactly the
field the legacy script used for its own primary slope, NOT the decode-only
`eval_duration`) is preserved losslessly via `AdapterRunResult.native_ns`,
reported as the harness's "native tok/s" diagnostic column, same mechanism
as lattice's entry above.

Run with (from repo root): `uv run --quiet --with mlx-lm python3
scripts/bench_decode_adapters_apples_to_apples.py run --profile
apples_to_apples_q8 --allow-missing-engine` (mlx-lm must be available in the
invoking environment since the mlx engine is present in both profiles; the
`--with mlx-lm` flag is how `scripts/bench_apples_to_apples.sh` provides it,
matching the legacy script's own `uv run --with mlx-lm` invocation).
"""

from __future__ import annotations

import argparse
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

_RESULT_RE = re.compile(r"^RESULT n_req=(\d+) completion=(\d+) total_ms=([\d.]+)$")


def parse_lattice_result_line(line: str) -> tuple[int, int, float] | None:
    """Parse one `bench_decode_ab` stdout line as a full-line RESULT record.
    Returns `(n_req, completion, total_ms)`, or `None` if the line is not an
    EXACT `RESULT n_req=<int> completion=<int> total_ms=<float>` match -- no
    prefix or suffix noise tolerated. The legacy script's awk filter
    (`/^RESULT/`) matched a RESULT-bearing SUBSTRING anywhere in the line;
    this adapter invokes with `BENCH_RUNS=1` (one clean line expected per
    call), so a stricter full-line anchor rejects a stale/prefixed/noisy
    line outright instead of scavenging a match out of it.
    """
    m = _RESULT_RE.match(line)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


class LatticeUnavailableError(RuntimeError):
    """`bench_decode_ab` is not built, or the requested model dir is missing."""


class LatticeResultError(RuntimeError):
    """`bench_decode_ab` ran and exited 0, but its RESULT output failed
    validation: not exactly one full-line RESULT record, a `n_req` that
    does not match the window actually requested, or a non-finite/negative
    measurement. Distinct from `LatticeUnavailableError` (binary/model not
    present, a registration-time condition) -- this is a data-integrity
    failure from an engine that DID run, and must crash loud rather than
    silently become a fabricated or mislabeled observation.
    """


def extract_single_result(stdout: str, *, n_tokens: int) -> tuple[int, int, float]:
    """Scan `stdout` for full-line RESULT records (see
    `parse_lattice_result_line`) and return the single one expected for a
    `BENCH_RUNS=1` invocation, as `(n_req, completion, total_ms)`.

    Raises `LatticeResultError` if there is not EXACTLY one RESULT line (zero
    -- no output; more than one -- a stale/duplicated line, since one
    `BENCH_RUNS=1` call must print exactly one), if the line's `n_req` does
    not equal the `n_tokens` window that was actually requested (a stale or
    wrong binary emitting a RESULT line for a different window must not
    silently become an observation labeled as the requested window), or if
    `completion`/`total_ms` are not finite and non-negative.
    """
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
            f"lattice: RESULT n_req={n_req} does not match the requested window "
            f"n_tokens={n_tokens} (stale or wrong binary?)"
        )
    if completion < 0:
        raise LatticeResultError(f"lattice: RESULT completion={completion} is negative")
    if not math.isfinite(total_ms) or total_ms < 0:
        raise LatticeResultError(f"lattice: RESULT total_ms={total_ms} is not finite and non-negative")
    return n_req, completion, total_ms


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
        try:
            _n_req, completion, total_ms = extract_single_result(proc.stdout, n_tokens=n_tokens)
        except LatticeResultError as exc:
            raise LatticeResultError(f"{exc} (stderr: {proc.stderr.strip()[-500:]})") from exc
        return harness.AdapterRunResult(
            actual_completion_tokens=completion,
            native_ns=round(total_ms * 1e6),
            engine_version="lattice-bench_decode_ab",
        )


def lattice_available(bin_path: Path = LAT_BIN) -> bool:
    return bin_path.is_file() and os.access(bin_path, os.X_OK)


def _lattice_required_model_dirs(profile: harness.ProfileConfig) -> tuple[Path, ...]:
    """Every local model directory the profile's `lattice` engine group(s)
    will need, resolved via `_LATTICE_MODEL_DIRS`. Empty if the profile has
    no `lattice` engine group.
    """
    dirs: list[Path] = []
    for group in profile.engine_groups:
        if group.name != "lattice":
            continue
        model_dir = _LATTICE_MODEL_DIRS.get(group.model)
        if model_dir is not None:
            dirs.append(model_dir)
    return tuple(dirs)


def lattice_registration_status(
    profile: harness.ProfileConfig | None, *, bin_path: Path = LAT_BIN
) -> tuple[bool, tuple[Path, ...]]:
    """Whether the lattice adapter should be registered for `profile`, and
    which of the profile's required model directories (if any) are missing.

    Binary presence alone is NOT sufficient: a binary present with a
    profile-required model directory absent must reproduce the legacy
    script's per-engine graceful skip (`bench_lattice`'s own
    `[[ ! -d "$model_dir" ]] && echo MODEL MISSING && return`) at
    REGISTRATION time, not surface as a `LatticeUnavailableError` raised
    mid-run that `--allow-missing-engine` cannot catch (the harness only
    treats an engine as "missing" when its name is absent from the adapter
    registry entirely -- see `bench_decode_harness.run_profile`).

    `profile=None` (the requested profile could not be determined, e.g. no
    `--profile` argument was found while peeking argv) falls back to
    binary-only availability -- the pre-fix behavior, so an unrecognized
    invocation shape still registers lattice and lets the harness's
    `MissingEngineError` / this module's `LatticeUnavailableError` surface
    any problem, rather than silently never registering lattice at all.
    """
    bin_ok = bin_path.is_file() and os.access(bin_path, os.X_OK)
    if not bin_ok:
        return False, ()
    if profile is None:
        return True, ()
    required = _lattice_required_model_dirs(profile)
    missing = tuple(d for d in required if not d.is_dir())
    return (len(missing) == 0), missing


def _peek_requested_profile(argv: list[str]) -> harness.ProfileConfig | None:
    """Best-effort peek at the `--profile`/`--profiles-file` arguments
    `bench_decode_harness.main()` will parse, so lattice's model-dir
    availability can be checked for the SPECIFIC profile about to run.
    Registration happens at module-import time (`register_available_adapters`
    runs before `harness.main()` parses `argv`), so this module has to do its
    own lightweight peek rather than wait for the harness's own parse.

    Returns `None` (never raises) on anything unexpected: no `--profile`
    found, an unknown profile name, or a profiles file that fails to load --
    `lattice_registration_status` treats `None` as "fall back to binary-only
    availability", matching this module's pre-fix behavior.
    """
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--profile", default=None)
    peek.add_argument("--profiles-file", type=Path, default=harness.DEFAULT_PROFILES_FILE)
    try:
        known, _ignored = peek.parse_known_args(argv)
    except SystemExit:
        return None
    if known.profile is None:
        return None
    try:
        _schema_version, profiles = harness.load_profiles_file(known.profiles_file)
    except harness.ProfileConfigError:
        return None
    return profiles.get(known.profile)


# --------------------------------------------------------------------------
# ollama
# --------------------------------------------------------------------------


class OllamaResponseError(RuntimeError):
    """An ollama `/api/generate` response is an error body, or is missing a
    required field, or has a required field of the wrong type. A
    structurally-valid-JSON error response (e.g. a proxy or an overloaded
    server returning `{"error": "..."}` with HTTP 200) must never be
    silently translated into a fabricated measurement -- it must crash the
    call loud instead.
    """


_REQUIRED_OLLAMA_FIELDS = ("eval_count", "eval_duration", "total_duration")


def ollama_response_to_result(data: dict) -> harness.AdapterRunResult:
    """Pure translation of one `/api/generate` JSON body into an
    `AdapterRunResult`.

    `native_ns` mirrors the legacy script's PRIMARY slope metric exactly:
    `tot = d.get('total_duration',0)/1e9`, the field the legacy script fed
    directly into its own slope calculation -- NOT the decode-only
    `eval_duration` (that was the legacy script's separate `ec/ed`
    reference-only "native tok/s" column, a different number). Requires and
    type-checks `eval_count`/`eval_duration`/`total_duration`; rejects an
    error body (a top-level `"error"` key) or any missing/non-numeric
    required field by raising `OllamaResponseError` rather than substituting
    a default -- a `0` or an absent field must never quietly become a fake
    "successful" observation. `eval_duration` is required and type-checked
    for response-shape integrity even though it is not the field this
    function extracts into `native_ns`.
    """
    if "error" in data:
        raise OllamaResponseError(f"ollama: response is an error body: {data['error']!r}")
    for field in _REQUIRED_OLLAMA_FIELDS:
        if field not in data:
            raise OllamaResponseError(f"ollama: response missing required field {field!r}: {data!r}")
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OllamaResponseError(
                f"ollama: response field {field!r} has non-numeric type "
                f"{type(value).__name__}: {data!r}"
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
        return ollama_response_to_result(data)


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

    `run()` does no work after `generate()` returns (no retokenization, no
    length verification): the legacy script never verified actual token
    count either (`generate(..., max_tokens=N, ...)` under `temp=0.0`, its
    printed row always used the REQUESTED `N`, never a verified count), and
    the harness's wall-clock `elapsed_ns` wraps the entire `run()` call --
    any work added after `generate()` returns would silently inflate the
    harness's primary timing beyond decode alone. Trusting `n_tokens`
    verbatim, exactly like the legacy script did, keeps that window equal to
    `generate()`'s own duration.
    """

    def __init__(self, source_model_dir: Path = Q8_MODEL_DIR):
        self.source_model_dir = source_model_dir
        # (mlx_model, tok, sampler, fallback_model_id) -- fallback_model_id
        # is None when the requested source_model_dir loaded successfully,
        # else the Hub identifier that was actually loaded instead (see
        # `run()`'s `actual_model` below).
        self._cache: dict[tuple[str, str], tuple[object, object, object, str | None]] = {}

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
        fallback_model_id: str | None = None
        try:
            mlx_model, tok = load(str(self.source_model_dir))
        except Exception:  # noqa: BLE001 -- legacy script's own bare except fallback
            mlx_model, tok = load("Qwen/Qwen3.5-0.8B")
            fallback_model_id = "Qwen/Qwen3.5-0.8B"
        nn.quantize(mlx_model, bits=bits, group_size=64)
        mx.eval(mlx_model.parameters())
        sampler = make_sampler(temp=0.0)
        entry = (mlx_model, tok, sampler, fallback_model_id)
        self._cache[key] = entry
        return entry

    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> harness.AdapterRunResult:
        from mlx_lm import generate

        mlx_model, tok, sampler, fallback_model_id = self._load(model, quantization)
        generate(mlx_model, tok, prompt=prompt, max_tokens=n_tokens, sampler=sampler, verbose=False)
        return harness.AdapterRunResult(
            actual_completion_tokens=n_tokens,
            engine_version="mlx_lm",
            # Provenance: a fallback load means the observation actually
            # measured Qwen/Qwen3.5-0.8B from the Hub, not the requested
            # local artifact -- record what really ran (see
            # AdapterRunResult.actual_model's contract) instead of letting
            # the raw observation silently mislabel it as the local model.
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


def register_available_adapters(argv: list[str] | None = None) -> None:
    """Registers whichever of lattice/ollama/mlx are actually available,
    printing a legacy-style message for each -- mirroring
    `bench_apples_to_apples.sh`'s per-engine graceful skip (missing binary /
    missing model dir / ollama not installed all print-and-continue rather
    than aborting the whole run).

    Lattice's registration is MODEL-aware, not just binary-aware: the binary
    can be present while the specific profile about to run needs a model
    directory that is not (e.g. Q8 weights present, Q4 weights absent). A
    `LatticeUnavailableError` raised mid-run for that case is not caught
    anywhere `--allow-missing-engine` can act on, so it must not be
    registered in the first place -- see `lattice_registration_status` and
    `_peek_requested_profile`.
    """
    argv = sys.argv[1:] if argv is None else argv
    profile = _peek_requested_profile(argv)
    should_register, missing_model_dirs = lattice_registration_status(profile)
    if should_register:
        harness.register_adapter("lattice", LatticeAdapter())
        print("  lattice: adapter registered")
    elif missing_model_dirs:
        joined = ", ".join(str(d) for d in missing_model_dirs)
        print(f"  lattice: MODEL MISSING ({joined}) — build/download first — skipping")
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
