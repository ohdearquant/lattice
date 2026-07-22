#!/usr/bin/env python3
"""Engine adapters and report compatibility for the ``agentic`` profile.

Collection policy, call ordering, repeat counts, raw observations, and
aggregation belong to ``bench_decode_harness``. This module only invokes the
three engines, parses their native timing data, supplies runtime context/runs
overrides, and renders the established agentic JSON/table output.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_decode_harness as harness  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES_FILE = REPO_ROOT / "scripts" / "bench_decode_profiles.toml"
OUT_DIR = REPO_ROOT / "docs" / "bench_results"
LAT_BIN = REPO_ROOT / "target" / "release" / "bench_decode_ab"
MODEL_DIR = Path.home() / ".lattice" / "models" / "qwen3.5-0.8b"
OLLAMA_URL = os.environ.get("LATTICE_BENCH_OLLAMA_URL", "http://localhost:11434")
SWEEP_CONTEXTS = (1000, 2000, 4000)
BASE = (
    "The quick brown fox jumps over the lazy dog. "
    "Once upon a time in a land far away, there lived a wise old owl "
    "who knew many secrets. Every morning the sun rose over the "
    "mountains and cast long shadows across the quiet valley. "
)

_RESULT_RE = re.compile(r"^RESULT n_req=(\d+) completion=(\d+) total_ms=([\d.]+)$")
_PROMPT_RE = re.compile(r"\[bench\] prompt_tokens=(\d+)")


class AdapterOutputError(RuntimeError):
    """An engine returned output that cannot satisfy the harness contract."""


@functools.lru_cache(maxsize=1)
def _default_profile() -> harness.ProfileConfig:
    """The unconfigured ``agentic`` profile, for deriving CLI/report defaults
    (ctx, runs, report windows) so they cannot drift from
    ``bench_decode_profiles.toml``.
    """
    _, profiles = harness.load_profiles_file(PROFILES_FILE)
    return profiles["agentic"]


def make_ollama_prompt(ctx: int) -> str:
    """Build the former Ollama character-count heuristic prompt."""
    repetitions = max(1, ctx * 4 // (len(BASE.split()) * 3))
    return BASE * repetitions


def make_fallback_padded_prompt(ctx: int) -> str:
    """Build the former no-tokenizer fallback prompt."""
    repetitions = max(1, (ctx * 4) // (len(BASE.split()) * 3) + 1)
    return BASE * repetitions


def configure_profile(
    profile: harness.ProfileConfig, *, ctx: int, runs: int, padded_prompt: str
) -> harness.ProfileConfig:
    """Apply runtime flags without moving schedule policy into adapters."""
    if ctx < 1:
        raise ValueError("--ctx must be positive")
    if runs < 1:
        raise ValueError("--runs must be positive")
    groups = tuple(
        dataclasses.replace(group, measured_prompt=make_ollama_prompt(ctx))
        if group.name == "ollama"
        else group
        for group in profile.engine_groups
    )
    return dataclasses.replace(
        profile,
        measured_repeats=runs,
        requested_prompt_tokens=ctx,
        prompt=padded_prompt,
        engine_groups=groups,
    )


def parse_lattice_output(stdout: str, stderr: str, *, n_tokens: int) -> harness.AdapterRunResult:
    matches = [match for line in stdout.splitlines() if (match := _RESULT_RE.fullmatch(line))]
    if len(matches) != 1:
        raise AdapterOutputError(f"lattice: expected one RESULT record, found {len(matches)}")
    requested, completion, total_ms = matches[0].groups()
    if int(requested) != n_tokens:
        raise AdapterOutputError(f"lattice: RESULT requested {requested}, expected {n_tokens}")
    elapsed_ms = float(total_ms)
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise AdapterOutputError(f"lattice: invalid total_ms={total_ms}")
    prompt_match = _PROMPT_RE.search(stderr)
    if prompt_match is None:
        raise AdapterOutputError("lattice: missing prompt_tokens diagnostic")
    return harness.AdapterRunResult(
        actual_completion_tokens=int(completion),
        actual_prompt_tokens=int(prompt_match.group(1)),
        native_ns=round(elapsed_ms * 1e6),
        engine_version="lattice-bench_decode_ab",
    )


class LatticeAdapter:
    def __init__(self, ctx: int, bin_path: Path = LAT_BIN, model_dir: Path = MODEL_DIR):
        self.ctx = ctx
        self.bin_path = bin_path
        self.model_dir = model_dir

    def run(self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str):
        env = os.environ.copy()
        env.update(
            BENCH_N=str(n_tokens),
            BENCH_RUNS="1",
            BENCH_PROMPT_TOKENS=str(self.ctx),
            LATTICE_MODEL_DIR=str(self.model_dir),
        )
        proc = subprocess.run(
            [str(self.bin_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if proc.returncode != 0:
            raise AdapterOutputError(f"lattice exited {proc.returncode}: {proc.stderr[-1000:]}")
        return parse_lattice_output(proc.stdout, proc.stderr, n_tokens=n_tokens)


def ollama_response_to_result(data: dict) -> harness.AdapterRunResult:
    if "error" in data:
        raise AdapterOutputError(f"ollama: {data['error']}")
    required = (
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
        "eval_count",
        "prompt_eval_count",
    )
    for key in required:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AdapterOutputError(f"ollama: invalid {key}={value!r}")
    ttft_ns = data["load_duration"] + data["prompt_eval_duration"]
    total_ns = ttft_ns + data["eval_duration"]
    return harness.AdapterRunResult(
        actual_completion_tokens=data["eval_count"],
        actual_prompt_tokens=data["prompt_eval_count"],
        engine_version="ollama",
        component_ns={1: ttft_ns, 100: total_ns},
    )


class OllamaAdapter:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url
        self.breakdowns: list[dict[str, int]] = []

    def run(self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str):
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": n_tokens, "temperature": 0, "seed": 42},
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read())
        result = ollama_response_to_result(data)
        if not warmup:
            self.breakdowns.append(
                {
                    "load_ms": data["load_duration"] / 1e6,
                    "prompt_eval_ms": data["prompt_eval_duration"] / 1e6,
                    "eval_ms": data["eval_duration"] / 1e6,
                    "prompt_eval_count": data["prompt_eval_count"],
                }
            )
        return result


class MlxAdapter:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-0.8B"):
        self.model_id = model_id
        self._loaded: tuple[object, object] | None = None
        self._tokenizer: object | None = None
        self._prompt_token_counts: dict[str, int] = {}

    def _load_tokenizer(self):
        """Fetch only the tokenizer files (no model weights), so building the
        shared tokenizer-padded prompt does not bring an MLX model into
        residency before Lattice/Ollama are measured.
        """
        if self._tokenizer is None:
            import mlx_lm.utils

            self._tokenizer = mlx_lm.utils.load_tokenizer(self.model_id)
        return self._tokenizer

    def _load(self):
        if self._loaded is None:
            import mlx_lm

            self._loaded = mlx_lm.load(self.model_id)
        return self._loaded

    def padded_prompt(self, ctx: int) -> str:
        tokenizer = self._load_tokenizer()
        prompt = ""
        token_count = 0
        while token_count < ctx:
            prompt += BASE
            token_count = len(tokenizer.encode(prompt))
        self._prompt_token_counts[prompt] = token_count
        return prompt

    def run(self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str):
        import mlx.core as mx
        import mlx_lm

        mlx_model, tokenizer = self._load()
        mlx_lm.generate(mlx_model, tokenizer, prompt=prompt, max_tokens=n_tokens, verbose=False)
        mx.eval(mx.array([0]))
        actual_prompt_tokens = self._prompt_token_counts.get(prompt)
        if actual_prompt_tokens is None:
            # Warmup calls (and any prompt not built by padded_prompt) are not
            # part of the measured region, so an on-demand encode here is safe.
            actual_prompt_tokens = len(tokenizer.encode(prompt))
        return harness.AdapterRunResult(
            actual_completion_tokens=n_tokens,
            actual_prompt_tokens=actual_prompt_tokens,
            engine_version="mlx_lm",
        )


def _ollama_available(base_url: str = OLLAMA_URL) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            data = json.loads(response.read())
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return any(model.get("name") == "qwen3.5:0.8b" for model in data.get("models", []))


def _mlx_available() -> bool:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


def register_available_adapters(ctx: int) -> tuple[dict[str, object], dict[str, str], str]:
    adapters: dict[str, object] = {}
    missing: dict[str, str] = {}
    mlx_adapter: MlxAdapter | None = None
    if LAT_BIN.is_file() and os.access(LAT_BIN, os.X_OK) and MODEL_DIR.is_dir():
        adapters["lattice"] = LatticeAdapter(ctx)
    else:
        missing["lattice"] = "bench_decode_ab binary or model directory is missing"
    if _ollama_available():
        adapters["ollama"] = OllamaAdapter()
    else:
        missing["ollama"] = "ollama server or qwen3.5:0.8b model is unavailable"
    if _mlx_available():
        mlx_adapter = MlxAdapter()
        adapters["mlx"] = mlx_adapter
    else:
        missing["mlx"] = "mlx_lm is unavailable"
    if mlx_adapter is not None:
        try:
            padded_prompt = mlx_adapter.padded_prompt(ctx)
        except Exception as exc:  # noqa: BLE001 -- an unavailable optional engine is reported, never substituted
            adapters.pop("mlx", None)
            adapters.pop("lattice", None)
            missing["mlx"] = f"mlx_lm model/tokenizer load failed: {exc}"
            missing["lattice"] = "the shared tokenizer-padded prompt could not be constructed"
            padded_prompt = make_fallback_padded_prompt(ctx)
    else:
        adapters.pop("lattice", None)
        missing["lattice"] = "mlx_lm is required to construct the exact tokenizer-padded Lattice prompt"
        padded_prompt = make_fallback_padded_prompt(ctx)
    return adapters, missing, padded_prompt


def _median_ms(result: harness.HarnessRunResult, engine: str, window: int) -> float | None:
    values = [
        harness._primary_elapsed_seconds(obs) * 1000
        for obs in result.observations
        if obs.engine == engine and not obs.warmup and obs.requested_completion_tokens == window
    ]
    return statistics.median(values) if values else None


def result_rows(
    result: harness.HarnessRunResult,
    missing_reasons: dict[str, str],
    ollama_adapter: OllamaAdapter | None,
) -> list[dict]:
    ttft_window, total_window = result.profile.windows[0], result.profile.windows[-1]
    rows: list[dict] = []
    for engine in result.profile.engines:
        ttft_ms = _median_ms(result, engine, ttft_window)
        total_ms = _median_ms(result, engine, total_window)
        prompt_counts = [
            obs.actual_prompt_tokens
            for obs in result.observations
            if obs.engine == engine and not obs.warmup and obs.actual_prompt_tokens is not None
        ]
        if ttft_ms is None or total_ms is None:
            rows.append(
                {
                    "engine": engine,
                    "context": result.profile.requested_prompt_tokens,
                    "response": total_window,
                    "runs": 0,
                    "ttft_ms": None,
                    "decode_ms": None,
                    "total_ms": None,
                    "prefill_tok_s": None,
                    "decode_tok_s": None,
                    "source": "unavailable",
                    "unavailable_reason": missing_reasons.get(engine, "engine was not run"),
                }
            )
            continue
        context = round(statistics.median(prompt_counts)) if prompt_counts else result.profile.requested_prompt_tokens
        decode_ms = total_ms - ttft_ms
        row = {
            "engine": engine,
            "context": context,
            "response": total_window,
            "runs": result.profile.measured_repeats,
            "ttft_ms": round(ttft_ms, 1),
            "decode_ms": round(decode_ms, 1),
            "total_ms": round(total_ms, 1),
            "prefill_tok_s": round(context / (ttft_ms / 1000)) if ttft_ms > 0 else 0,
            "decode_tok_s": round(total_window / (decode_ms / 1000)) if decode_ms > 0 else 0,
            "source": "live",
        }
        if engine == "ollama" and ollama_adapter is not None and ollama_adapter.breakdowns:
            row["ollama_breakdown"] = {
                key: round(statistics.median(item[key] for item in ollama_adapter.breakdowns), 1)
                for key in ("load_ms", "prompt_eval_ms", "eval_ms")
            }
            row["ollama_breakdown"]["prompt_eval_count"] = round(
                statistics.median(item["prompt_eval_count"] for item in ollama_adapter.breakdowns)
            )
        rows.append(row)
    return rows


def render_table(rows: list[dict], ctx: int, response_window: int) -> str:
    lines = [
        f"\n## Agentic Workload: ~{ctx}-token context, {response_window}-token response\n\n",
        "| Engine | Precision | Ctx (tok) | TTFT (ms) | Decode (ms) | Total (ms) | Prefill t/s | Decode t/s | Runs | Source |\n",
        "|--------|-----------|-----------|-----------|-------------|-----------|-------------|------------|------|--------|\n",
    ]
    precision = {"lattice": "Q8 sf", "ollama": "Q8_0", "mlx": "bf16"}
    for row in rows:
        value = lambda key: "N/A" if row[key] is None else str(round(row[key]))
        prefill = value("prefill_tok_s") + ("*" if row["engine"] == "ollama" and row["prefill_tok_s"] else "")
        lines.append(
            f"| {row['engine']:<10} | {precision[row['engine']]:<9} | {value('context'):>9} "
            f"| {value('ttft_ms'):>9} | {value('decode_ms'):>11} | {value('total_ms'):>9} "
            f"| {prefill:>11} | {value('decode_tok_s'):>10} | {row['runs']:>4} | {row['source']:<6} |\n"
        )
    lines.append(
        "\n*ollama prefill tok/s reflects prefix-cache lookup for repeated filler text, not fresh-token prefill.\n"
        "Precision note: MLX runs bf16 (~2x memory bandwidth vs lattice Q8). "
        "The lattice decode gap is conservative — bandwidth-adjusted, the gap is larger.\n"
    )
    return "".join(lines)


def run_context(ctx: int, runs: int, allow_missing: bool, out: Path | None) -> list[dict]:
    default_profile = _default_profile()
    adapters, missing, padded_prompt = register_available_adapters(ctx)
    profile = configure_profile(default_profile, ctx=ctx, runs=runs, padded_prompt=padded_prompt)
    result = harness.run_profile(profile, adapters, allow_missing_engine=allow_missing)
    raw_path = out if out is not None else OUT_DIR / f"agentic_{ctx}tok_raw.jsonl"
    harness.write_jsonl(list(result.observations), raw_path)
    ollama = adapters.get("ollama")
    rows = result_rows(result, missing, ollama if isinstance(ollama, OllamaAdapter) else None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"agentic_{ctx}tok.json").write_text(json.dumps(rows, indent=2) + "\n")
    if abs(ctx - default_profile.requested_prompt_tokens) <= 50:
        (OUT_DIR / "agentic_1k_compare.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(render_table(rows, ctx, default_profile.windows[-1]))
    print(f"Raw observations: {raw_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    profile = _default_profile()
    parser = argparse.ArgumentParser(description="Profiled agentic decode benchmark")
    parser.add_argument("--ctx", type=int, default=profile.requested_prompt_tokens)
    parser.add_argument("--runs", type=int, default=profile.measured_repeats)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--allow-missing-engine", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sweep and args.out is not None:
        print("FAIL: --out cannot be combined with --sweep", file=sys.stderr)
        return 1
    contexts = SWEEP_CONTEXTS if args.sweep else (args.ctx,)
    all_rows: list[dict] = []
    try:
        for ctx in contexts:
            all_rows.extend(run_context(ctx, args.runs, args.allow_missing_engine, args.out))
    except (ValueError, harness.MissingEngineError, AdapterOutputError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.sweep:
        path = OUT_DIR / "agentic_sweep.json"
        path.write_text(json.dumps(all_rows, indent=2) + "\n")
        print(f"Sweep JSON: {path}")
    notices = [row for row in all_rows if row.get("source") != "live"]
    if notices:
        print("\n*** FRAMEWORK STATUS NOTICES ***")
        for row in notices:
            print(f"  {row['engine']}: UNAVAILABLE — {row.get('unavailable_reason', 'unknown reason')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
