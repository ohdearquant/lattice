#!/usr/bin/env python3
"""Q4 perplexity regression gate (issue #616).

Runs `eval_perplexity` on the *unrotated* Metal Q4 tier (the actually-shipping
quantization) over a fixed corpus slice and compares the result against a golden
PPL frozen in `crates/inference/tests/fixtures/ppl_gate_v1/golden.json`.

Why a frozen-golden regression gate and not the QuaRot delta gate
-----------------------------------------------------------------
The original ADR-044 dual-Q4 gate asserted `quarot - unrotated < 0.5`. On the
current forward path offline QuaRot v0 is a *net-negative* (+1.9 PPL worse than
unrotated asymmetric Q4) by design — Hadamard rotation forces symmetric Q4,
which has a worse fidelity floor, and offline v0 has no online R3/R4 rotation to
recover it (see ADR-044 "Conclusion superseded (2026-07-06)" and issue #703).
A `quarot must win` gate would therefore red main on a known, accepted
limitation. So #616 repoints the gate onto the shipping tier: catch a
*regression* in unrotated Q4 PPL against a known-good baseline.

Why an absolute golden captured on the CI runner (not a relative delta)
-----------------------------------------------------------------------
GitHub's `macos-latest` exposes a PARAVIRTUAL Metal device. The Q4 forward path
is numerically correct there (proven by the `q4-composed` token-ID gate), but
absolute PPL differs from a real Apple-GPU. A CPU-BF16-vs-Metal-Q4 relative gate
would NOT cancel that quirk (the two numbers come from different compute paths,
only one of which is paravirtual). The honest, env-robust design is an absolute
golden *captured on the same paravirtual runner*: run-to-run it is reproducible,
so it catches regressions while the env-specific offset is baked into the golden.

Modes
-----
  RECORD  (entered iff golden `ppl` is null): measure and print the PPL, DO NOT
          fail on value. Used to bootstrap the golden — the first CI run on the
          runner prints the paravirtual PPL, a maintainer commits it into
          golden.json, and the gate goes live. Loud banner marks this UNARMED
          (fail-open) state. LATTICE_PPL_GATE_RECORD=1 is ignored once the golden
          is numeric — an env flag can never suppress enforcement.
  ENFORCE (whenever golden `ppl` is numeric): fail-closed if
          |measured - golden| > tolerance.

Fail-closed contract: any provisioning/execution failure (binary missing, no PPL
line parsed, corpus missing, golden missing while enforcing) is a hard FAIL, never
a skip-as-green — a broken run must not masquerade as a passing gate.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "crates/inference/tests/fixtures/ppl_gate_v1/golden.json"


def fail(msg: str) -> NoReturn:
    print(f"::error::[ppl-gate] {msg}", file=sys.stderr)
    print(f"\nFAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not GOLDEN.exists():
        fail(f"golden not found: {GOLDEN}")
    spec = json.loads(GOLDEN.read_text())

    eval_bin = REPO / "target/release/eval_perplexity"
    if not eval_bin.exists():
        fail(f"eval_perplexity not built at {eval_bin}")

    def resolve_dir(var: str) -> str:
        # GitHub Actions `env:` values are NOT shell-expanded, so a `~` or
        # `$HOME` reaches us literally. Expand both here so the caller can use
        # either form; a still-unresolved path fails closed below.
        raw = os.environ.get(var)
        if not raw:
            fail(f"{var} unset")
        resolved = os.path.expanduser(os.path.expandvars(raw))
        if not Path(resolved).is_dir():
            fail(f"{var} is not a directory: {raw!r} (resolved: {resolved!r})")
        return resolved

    q4_dir = resolve_dir("LATTICE_PPL_Q4_DIR")
    tok_dir = resolve_dir("LATTICE_PPL_TOKENIZER_DIR")

    corpus = REPO / spec["corpus"]
    if not corpus.exists():
        fail(f"corpus not found: {corpus}")

    cmd = [
        str(eval_bin),
        "--q4-dir",
        q4_dir,
        "--tokenizer-dir",
        tok_dir,
        "--corpus-file",
        str(corpus),
        "--window",
        str(spec["window"]),
        "--stride",
        str(spec["stride"]),
        "--max-tokens",
        str(spec["max_tokens"]),
        "--json",
        "--label",
        "q4",
    ]
    print(f"[ppl-gate] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        fail(f"eval_perplexity exited {proc.returncode}")

    # Parse the machine-readable event line: @@lattice {"ev":"perplexity",...}
    measured = None
    tokens = None
    for line in proc.stdout.splitlines():
        if line.startswith("@@lattice "):
            obj = json.loads(line[len("@@lattice ") :])
            if obj.get("ev") == "perplexity" and obj.get("label") == "q4":
                measured = float(obj["ppl"])
                tokens = int(obj["tokens"])
    if measured is None:
        sys.stderr.write(proc.stdout)
        fail("no @@lattice perplexity event parsed from eval_perplexity output")

    # Validate golden numerics: a non-finite tolerance (e.g. "1e999" -> inf) or
    # golden ppl would make enforcement a fail-open no-op. Reject, fail closed.
    tol = float(spec["tolerance"])
    if not (math.isfinite(tol) and tol > 0):
        fail(f"golden tolerance must be a finite positive number, got {spec['tolerance']!r}")
    golden_ppl = spec.get("ppl")
    if golden_ppl is not None:
        golden_ppl = float(golden_ppl)
        if not (math.isfinite(golden_ppl) and golden_ppl > 0):
            fail(f"golden ppl must be null or a finite positive number, got {spec.get('ppl')!r}")

    # RECORD mode (measure-only, no value enforcement) is entered ONLY when the
    # golden is unarmed (ppl is null). A numeric golden ALWAYS enforces — an env
    # flag cannot suppress enforcement, otherwise a regression could be forced to
    # exit 0. The env flag only makes the *unarmed* state explicit locally.
    record = golden_ppl is None
    if os.environ.get("LATTICE_PPL_GATE_RECORD") == "1" and not record:
        print("::warning::LATTICE_PPL_GATE_RECORD ignored: golden is armed (numeric); enforcing.")

    # When this gate is wired as a REQUIRED status check, an unarmed golden must
    # NOT pass green: reverting `ppl` to null would otherwise silently de-arm the
    # gate while branch protection still sees a success. The required CI job sets
    # LATTICE_PPL_GATE_REQUIRE_ARMED=1 so a null/missing golden fails closed here.
    # RECORD mode stays available for the deliberate re-capture bootstrap (run
    # without this flag), and the sandbox self-test still exercises the null path.
    if record and os.environ.get("LATTICE_PPL_GATE_REQUIRE_ARMED") == "1":
        fail(
            "golden ppl is null/missing but LATTICE_PPL_GATE_REQUIRE_ARMED=1: this "
            "is a required gate and must be armed. Commit a numeric `ppl` into the "
            "golden fixture, or run without the flag to re-capture in RECORD mode."
        )

    report = REPO / os.environ.get("PPL_GATE_REPORT", "ppl-gate-report.md")
    lines = [
        "## Q4 perplexity regression gate (#616)",
        "",
        "- tier: unrotated Metal Q4 (shipping)",
        f"- corpus: `{spec['corpus']}`, first {spec['max_tokens']} tokens",
        f"- window/stride: {spec['window']}/{spec['stride']}",
        f"- tokens scored: {tokens}",
        f"- **measured PPL: {measured:.6f}**",
    ]

    if record:
        lines += [
            f"- golden PPL: {'null (UNARMED)' if golden_ppl is None else f'{golden_ppl:.6f}'}",
            "",
            "> **GATE UNARMED (fail-open).** This run is in RECORD mode. To arm the",
            "> gate: commit the measured PPL above into",
            "> `crates/inference/tests/fixtures/ppl_gate_v1/golden.json` (`ppl` field),",
            "> then add `q4-ppl-quality` to `parity-gate.needs` and to the repository's",
            "> required status checks. Until then this gate reports but does not block.",
        ]
        report.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"\n[ppl-gate] RECORD mode — measured PPL {measured:.6f} (gate not enforced)")
        return

    delta = abs(measured - float(golden_ppl))
    passed = delta <= tol
    lines += [
        f"- golden PPL: {float(golden_ppl):.6f}",
        f"- tolerance: {tol:.6f}",
        f"- |measured - golden|: {delta:.6f}",
        f"- verdict: **{'PASS' if passed else 'FAIL'}**",
    ]
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not passed:
        fail(
            f"Q4 PPL regressed: measured {measured:.6f} vs golden {float(golden_ppl):.6f} "
            f"(|Δ|={delta:.6f} > tol={tol:.6f})"
        )
    print(
        f"\n[ppl-gate] PASS — Q4 PPL {measured:.6f} within {tol:.6f} "
        f"of golden {float(golden_ppl):.6f}"
    )


if __name__ == "__main__":
    main()
