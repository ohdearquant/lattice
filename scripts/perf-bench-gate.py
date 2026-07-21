#!/usr/bin/env python3
"""Parse Criterion change reports and apply the ADR-058 regression gate.

For every Criterion bench under target/criterion/, read change/estimates.json
(produced when running with --baseline <name>). Apply the rule:

  CI-lower of change in (-inf, +3%]    : pass silently
  CI-lower of change in (+3%, +7%]     : warn (PR-comment only, no fail)
  CI-lower of change in (+7%, +inf)    : FAIL
  Point estimate < -3% AND CI-upper<0% : celebrate

Usage:
  perf-bench-gate.py <criterion_root> <arch_label> [--out report.md]
  perf-bench-gate.py <criterion_root> <arch_label> --informational-groups-file <path>

Exit codes:
  0 — pass (no gated FAILs)
  1 — at least one gated FAIL (regression > 7%, using the LOWER bound of
      Criterion's two-sided 95% CI as a one-sided cutoff — see the
      WARN_PCT/FAIL_PCT note below for the actual one-sided confidence
      level this implies, which is tighter than "95%")
  2 — parse error / bad input, or (with --require-measurements) the gate
      refusing to certify a run it could not judge: no comparison data, or
      no gating comparison among the parsed results. An automated lane must
      not read "nothing was measured" as "nothing regressed".

--informational-groups-file (lattice#714): quick-mode Criterion runs on
sub-microsecond micro-benches (lattice-embed's `simd` bench target) are
dominated by scheduler/thermal jitter rather than code changes — confirmed
by two same-toolchain quick-mode A/A runs on identical refs flipping FAIL/
WARN sign across dozens of entries (lattice#714). Groups listed in this file
(one Criterion top-level group name per line, e.g. from `cargo bench ... --
--list`) are still measured and reported, but excluded from the FAIL/WARN
gate and the exit code — they render in a separate "informational" section
labeled below quick-mode resolution. This file should only be passed for
quick-mode runs; full-mode (tight-CI) runs gate every group normally so a
real embed SIMD regression is still caught.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Thresholds — ADR-058 §D3. Edit here; the workflow imports nothing else.
#
# Precision note (bench-gate math audit, finding #4): `ci_low`/`ci_high` are
# Criterion's own TWO-SIDED 95% CI endpoints. Using `ci_low` as a one-sided
# FAIL cutoff is directionally sound (a slowdown is a one-sided hypothesis,
# and gating on the lower endpoint of a two-sided interval can only be MORE
# conservative than a properly-computed one-sided 95% bound, never less —
# assuming Criterion's CI is symmetric two-sided at 0.95, `ci_low` sits at
# roughly a 97.5%-one-sided-confidence level, not 95%). This raises the bar
# for FAIL (fewer true positives caught), never lowers it, so it cannot by
# itself produce a false FAIL — but "regression >7% confirmed by 95% CI" in
# this module's docstring/comments overstates precision and should be read
# as "confirmed at approximately a 97.5% one-sided level via the two-sided
# 95% CI's lower bound," not a calibrated one-sided-95% test. Verify against
# the `criterion` crate's own CI-construction source before relying on the
# exact number if it ever matters (not independently checked here).
WARN_PCT = 3.0   # CI-lower above this => warning
FAIL_PCT = 7.0   # CI-lower above this => FAIL
CELEBRATE_PCT = -3.0  # point estimate below this AND CI-upper<0 => celebrate


@dataclass
class BenchResult:
    name: str            # e.g. "rms_norm/4096"
    point: float         # change.estimates.mean.point_estimate (fraction, not %)
    ci_low: float
    ci_high: float
    new_ns: float        # new median time, nanoseconds
    old_ns: float        # baseline median time, nanoseconds

    @property
    def point_pct(self) -> float: return self.point * 100.0
    @property
    def ci_low_pct(self) -> float: return self.ci_low * 100.0
    @property
    def ci_high_pct(self) -> float: return self.ci_high * 100.0
    @property
    def group(self) -> str:
        """Top-level Criterion group name (name is 'group/function/param' or 'group/param')."""
        return self.name.split("/", 1)[0]

    def is_informational(self, informational_groups: frozenset[str]) -> bool:
        return self.group in informational_groups

    def verdict(self) -> str:
        if self.ci_low_pct > FAIL_PCT:
            return "FAIL"
        if self.ci_low_pct > WARN_PCT:
            return "WARN"
        if self.point_pct < CELEBRATE_PCT and self.ci_high_pct < 0:
            return "WIN"
        return "PASS"


def load_informational_groups(path: Path | None) -> frozenset[str]:
    """Load top-level group names to exclude from gating (lattice#714).

    One group name per line; blank lines and '#'-prefixed comments ignored.
    """
    if path is None:
        return frozenset()
    if not path.exists():
        print(f"warn: --informational-groups-file {path} does not exist — gating "
              f"every group normally", file=sys.stderr)
        return frozenset()
    groups = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        groups.add(line)
    return frozenset(groups)


def find_change_files(root: Path) -> list[Path]:
    """Find every change/estimates.json under root (Criterion's per-bench output)."""
    return sorted(root.rglob("change/estimates.json"))


def find_baseline_estimates(bench_dir: Path, baseline_name: str) -> Path | None:
    """Locate the baseline estimates.json for a bench directory.

    Criterion writes the pre-run comparison snapshot under a directory named
    after the baseline: the default (unnamed) rotation uses `base/`, while a
    named baseline (`--save-baseline <name>` / `--baseline <name>`, as used by
    bench-compare.sh's `compare-base` leg) writes under `<name>/` instead —
    `base/` is never created in that flow. Prefer the caller-supplied baseline
    name FIRST: Criterion computed change/ against that baseline, so a stale
    `base/` left in a dirty local tree must not shadow it (review of
    PR #548 reproduced exactly that wrong-baseline report). Then try the
    default `base/` (covers CI's default-rotation runs). As a last resort,
    accept a sibling directory holding an estimates.json that isn't
    `new`/`change` — but only when it is unambiguous: Criterion supports
    multiple named baselines side by side, and guessing among several would
    silently gate against the wrong one.
    """
    candidates = [baseline_name, "base"]
    for candidate in candidates:
        p = bench_dir / candidate / "estimates.json"
        if p.exists():
            return p

    fallbacks = [
        child for child in sorted(bench_dir.iterdir())
        if child.is_dir()
        and child.name not in ("new", "change")
        and (child / "estimates.json").exists()
    ]
    if len(fallbacks) == 1:
        print(f"note: {bench_dir.name}: using sole sibling baseline dir "
              f"'{fallbacks[0].name}/' (neither '{baseline_name}/' nor 'base/' found)",
              file=sys.stderr)
        return fallbacks[0] / "estimates.json"
    if len(fallbacks) > 1:
        names = ", ".join(f.name for f in fallbacks)
        print(f"warn: {bench_dir.name}: multiple candidate baseline dirs ({names}) "
              f"and none match '{baseline_name}/' or 'base/' — refusing to guess",
              file=sys.stderr)
    return None


def parse_bench(change_file: Path, root: Path, baseline_name: str) -> BenchResult | None:
    """Parse one change/estimates.json + sibling new/estimates.json + baseline estimates.json.

    Returns None if files are malformed (bench skipped, not failed).
    """
    bench_dir = change_file.parent.parent  # .../<bench>/<test>/
    rel = bench_dir.relative_to(root)
    name = str(rel)

    try:
        change = json.loads(change_file.read_text())
        mean = change["mean"]
        point = mean["point_estimate"]
        ci_low = mean["confidence_interval"]["lower_bound"]
        ci_high = mean["confidence_interval"]["upper_bound"]

        new_path = bench_dir / "new" / "estimates.json"
        new_ns = json.loads(new_path.read_text())["mean"]["point_estimate"]

        base_path = find_baseline_estimates(bench_dir, baseline_name)
        if base_path is None:
            print(f"warn: {name}: change/estimates.json present but no resolvable "
                  f"baseline dir (tried base/, {baseline_name}/, and other siblings) "
                  f"— skipping", file=sys.stderr)
            return None
        old_ns = json.loads(base_path.read_text())["mean"]["point_estimate"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"warn: skipping {name}: {e}", file=sys.stderr)
        return None

    return BenchResult(name=name, point=point, ci_low=ci_low, ci_high=ci_high,
                       new_ns=new_ns, old_ns=old_ns)


def render_report(results: list[BenchResult], arch: str,
                   informational_groups: frozenset[str] = frozenset()) -> str:
    gated = [r for r in results if not r.is_informational(informational_groups)]
    info = [r for r in results if r.is_informational(informational_groups)]

    fails = [r for r in gated if r.verdict() == "FAIL"]
    warns = [r for r in gated if r.verdict() == "WARN"]
    wins = [r for r in gated if r.verdict() == "WIN"]

    info_fails = [r for r in info if r.verdict() == "FAIL"]
    info_warns = [r for r in info if r.verdict() == "WARN"]
    info_wins = [r for r in info if r.verdict() == "WIN"]

    lines = [f"### `{arch}` — perf regression report\n"]
    if fails:
        lines.append(f"**❌ {len(fails)} FAIL** (regression >{FAIL_PCT}% confirmed by 95% CI)")
    if warns:
        lines.append(f"**⚠ {len(warns)} WARN** (regression {WARN_PCT}-{FAIL_PCT}% confirmed)")
    if wins:
        lines.append(f"**🚀 {len(wins)} confirmed improvement**")
    if not (fails or warns or wins):
        lines.append(f"✅ All {len(gated)} gated benches within noise band (±{WARN_PCT}%)")
    lines.append("")

    if fails or warns or wins:
        lines.append("| Bench | Δ point | 95% CI | new ns | base ns | verdict |")
        lines.append("|---|---:|---|---:|---:|---|")
        for r in sorted(fails + warns + wins, key=lambda r: -r.ci_low_pct):
            icon = {"FAIL": "❌", "WARN": "⚠", "WIN": "🚀"}[r.verdict()]
            lines.append(
                f"| `{r.name}` | {r.point_pct:+.2f}% | [{r.ci_low_pct:+.2f}%, {r.ci_high_pct:+.2f}%] "
                f"| {r.new_ns:.1f} | {r.old_ns:.1f} | {icon} {r.verdict()} |"
            )
        lines.append("")

    if info:
        lines.append(
            f"**ℹ️ {len(info)} informational** (below quick-mode resolution — "
            f"lattice-embed SIMD micro-benches, tracked in #714; not gated here, "
            f"re-run `--full` for a gated verdict)"
        )
        if info_fails or info_warns or info_wins:
            lines.append("| Bench | Δ point | 95% CI | new ns | base ns | (would-be verdict) |")
            lines.append("|---|---:|---|---:|---:|---|")
            for r in sorted(info_fails + info_warns + info_wins, key=lambda r: -r.ci_low_pct):
                icon = {"FAIL": "❌", "WARN": "⚠", "WIN": "🚀"}[r.verdict()]
                lines.append(
                    f"| `{r.name}` | {r.point_pct:+.2f}% | [{r.ci_low_pct:+.2f}%, {r.ci_high_pct:+.2f}%] "
                    f"| {r.new_ns:.1f} | {r.old_ns:.1f} | {icon} {r.verdict()} (informational) |"
                )
        lines.append("")

    lines.append(
        f"<details><summary>All {len(results)} measurements</summary>\n\n"
        "| Bench | Δ point | CI-lower | CI-upper |\n|---|---:|---:|---:|"
    )
    for r in sorted(results, key=lambda r: r.name):
        lines.append(
            f"| `{r.name}` | {r.point_pct:+.2f}% | {r.ci_low_pct:+.2f}% | {r.ci_high_pct:+.2f}% |"
        )
    lines.append("\n</details>\n")
    lines.append(
        f"_Rule: CI-lower of change ≤{WARN_PCT}% passes silently; "
        f"({WARN_PCT}%, {FAIL_PCT}%] warns; >{FAIL_PCT}% fails. Override via PR label `bench-allow-regression`._"
    )
    if informational_groups:
        lines.append(
            f"_{len(informational_groups)} group(s) excluded from gating as quick-mode "
            f"informational-only (lattice#714): {', '.join(sorted(informational_groups))}._\n"
        )
    else:
        lines.append("")
    return "\n".join(lines)


def _fabricate_bench(bench_dir: Path, baseline_dirname: str,
                     point: float = 0.10, ci_low: float = 0.05, ci_high: float = 0.15,
                     new_ns: float = 100.0, base_ns: float = 90.0) -> None:
    """Write a fake Criterion bench dir (new/, <baseline_dirname>/, change/) for --selftest."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "new").mkdir(exist_ok=True)
    (bench_dir / "new" / "estimates.json").write_text(
        json.dumps({"mean": {"point_estimate": new_ns}}))
    (bench_dir / baseline_dirname).mkdir(exist_ok=True)
    (bench_dir / baseline_dirname / "estimates.json").write_text(
        json.dumps({"mean": {"point_estimate": base_ns}}))
    (bench_dir / "change").mkdir(exist_ok=True)
    (bench_dir / "change" / "estimates.json").write_text(json.dumps({
        "mean": {
            "point_estimate": point,
            "confidence_interval": {"lower_bound": ci_low, "upper_bound": ci_high},
        }
    }))


def run_selftest() -> int:
    """Fabricate both baseline layouts + an orphan case; assert the parser handles each.

    Regression coverage for #545: a default-rotation `base/` layout, a named-baseline
    `compare-base/` layout (what bench-compare.sh actually produces), and a `change/`
    dir with no resolvable baseline at all (must WARN by bench name, not silently skip).
    Plus the findings on PR #548: when BOTH base/ and the named baseline exist,
    the named baseline must win (dirty local tree with stale base/); when multiple
    unrelated sibling baselines exist and none match, the gate must refuse to guess.
    """
    import contextlib
    import io
    import subprocess
    import tempfile

    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        default_dir = root / "grp_a" / "bench_default"
        _fabricate_bench(default_dir, "base")

        named_dir = root / "grp_b" / "bench_named"
        _fabricate_bench(named_dir, "compare-base")

        orphan_dir = root / "grp_c" / "bench_orphan"
        orphan_dir.mkdir(parents=True)
        (orphan_dir / "new").mkdir()
        (orphan_dir / "new" / "estimates.json").write_text(
            json.dumps({"mean": {"point_estimate": 100.0}}))
        (orphan_dir / "change").mkdir()
        (orphan_dir / "change" / "estimates.json").write_text(json.dumps({
            "mean": {"point_estimate": 0.1,
                     "confidence_interval": {"lower_bound": 0.05, "upper_bound": 0.15}}
        }))

        # Finding 1: both base/ and compare-base/ present with different
        # values — the named baseline must win over stale base/.
        both_dir = root / "grp_d" / "bench_both"
        _fabricate_bench(both_dir, "compare-base", base_ns=100.0)
        (both_dir / "base").mkdir()
        (both_dir / "base" / "estimates.json").write_text(
            json.dumps({"mean": {"point_estimate": 1.0}}))  # stale decoy

        # Finding 2: multiple unrelated sibling baselines, none matching
        # the requested name — must skip loudly, not guess.
        multi_dir = root / "grp_e" / "bench_multi"
        _fabricate_bench(multi_dir, "old-run-1")
        (multi_dir / "old-run-2").mkdir()
        (multi_dir / "old-run-2" / "estimates.json").write_text(
            json.dumps({"mean": {"point_estimate": 42.0}}))

        change_files = find_change_files(root)
        if len(change_files) != 5:
            failures.append(f"expected 5 change/estimates.json, found {len(change_files)}")

        stderr_buf = io.StringIO()
        results: dict[str, BenchResult] = {}
        with contextlib.redirect_stderr(stderr_buf):
            for cf in change_files:
                r = parse_bench(cf, root, baseline_name="compare-base")
                if r is not None:
                    results[r.name] = r
        stderr_text = stderr_buf.getvalue()

        if "grp_a/bench_default" not in results:
            failures.append("default base/ layout: bench not parsed")
        if "grp_b/bench_named" not in results:
            failures.append("named compare-base/ layout: bench not parsed")
        if "grp_c/bench_orphan" in results:
            failures.append("orphan bench (no resolvable baseline) was parsed instead of skipped")
        if "grp_c/bench_orphan" not in stderr_text:
            failures.append("orphan bench did not emit a warning naming the bench")

        both = results.get("grp_d/bench_both")
        if both is None:
            failures.append("both-dirs layout: bench not parsed")
        elif both.old_ns != 100.0:
            failures.append(f"both-dirs layout: expected named-baseline old_ns=100.0 "
                            f"(compare-base/), got {both.old_ns} (stale base/ shadowed it)")

        if "grp_e/bench_multi" in results:
            failures.append("multi-sibling layout: gate guessed a baseline instead of refusing")
        if "bench_multi" not in stderr_text or "refusing to guess" not in stderr_text:
            failures.append("multi-sibling layout: no loud refusal warning emitted")

        # lattice#714: informational-groups exclusion. Two confirmed FAILs, one in a
        # group named as informational (quick-mode embed-SIMD noise floor), one not —
        # the exit-code fail count and the gated report section must only count the
        # real one; the informational one must still be measured and reported.
        noisy_dir = root / "grp_f" / "noisy_fail"
        _fabricate_bench(noisy_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
        real_dir = root / "grp_g" / "real_fail"
        _fabricate_bench(real_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)

        for cf in find_change_files(root):
            r = parse_bench(cf, root, baseline_name="compare-base")
            if r is not None:
                results[r.name] = r

        informational = frozenset({"grp_f"})
        all_results = list(results.values())
        gated_fails = [
            r for r in all_results
            if r.verdict() == "FAIL" and not r.is_informational(informational)
        ]
        if "grp_f/noisy_fail" not in results or "grp_g/real_fail" not in results:
            failures.append("informational-groups fixture: benches not parsed")
        elif not results["grp_f/noisy_fail"].is_informational(informational):
            failures.append("informational-groups: grp_f/noisy_fail not classified informational")
        elif results["grp_g/real_fail"].is_informational(informational):
            failures.append("informational-groups: grp_g/real_fail wrongly classified informational")
        elif any(r.name == "grp_f/noisy_fail" for r in gated_fails):
            failures.append("informational-groups: noisy FAIL leaked into gated fail count")
        elif not any(r.name == "grp_g/real_fail" for r in gated_fails):
            failures.append("informational-groups: real FAIL missing from gated fail count")

        report = render_report(all_results, "selftest-arch", informational)
        if "grp_g/real_fail" not in report:
            failures.append("informational-groups: real FAIL missing from rendered report")
        if "ℹ️" not in report or "grp_f/noisy_fail" not in report:
            failures.append("informational-groups: noisy FAIL not shown in informational section")

        # lattice#714 / lattice#1060: the shell-side manifest handoff,
        # exercised end-to-end against the real helper and the real
        # manifest (scripts/lib/bench-quick-informational-targets.txt) —
        # the same files bench-compare.sh uses in production. Three
        # probes: (1) --print-targets must equal the reviewed expectation
        # set below, so a manifest-only or expectation-only edit fails
        # the selftest; (2) a demoted target key must emit every group of
        # a controlled listing (target-level semantics — including groups
        # the old per-group allowlist never contained); (3) a non-demoted
        # target key against the same listing must emit nothing — the
        # cross-target guarantee that keeps inference gating intact.
        # Probe 2's output then drives the Python classifier to prove
        # embed FAILs land informational while an inference FAIL gates.
        helper = Path(__file__).resolve().parent / "lib" / "bench-informational-groups.sh"
        # The reviewed demoted-target set, duplicated here on purpose:
        # the selftest compares this against the manifest itself (via
        # --print-targets), so a target added to only ONE side —
        # manifest or this expectation — fails the selftest.
        approved_targets = frozenset({"lattice-embed:simd"})
        if not helper.exists():
            failures.append(f"manifest-handoff: shell helper missing at {helper}")
        else:
            raw_proc = subprocess.run(
                ["bash", str(helper), "--print-targets"],
                capture_output=True, text=True, timeout=30,
            )
            raw_targets = frozenset(
                ln.strip() for ln in raw_proc.stdout.splitlines() if ln.strip()
            )
            if raw_proc.returncode != 0:
                failures.append(
                    f"manifest-handoff: --print-targets exited "
                    f"{raw_proc.returncode}: {raw_proc.stderr}"
                )
            elif raw_targets != approved_targets:
                failures.append(
                    "manifest-handoff: manifest and selftest expectation "
                    f"disagree — manifest-only: {sorted(raw_targets - approved_targets)}, "
                    f"expectation-only: {sorted(approved_targets - raw_targets)}. "
                    "Every demotion change must update both sides in one PR."
                )
            listing_dir = root / "manifest-listing"
            listing_dir.mkdir(parents=True, exist_ok=True)
            listing_file = listing_dir / "list.txt"
            listing_file.write_text(
                "simd_dot_product/scalar/384: benchmark\n"
                "simd_dot_product/simd/384: benchmark\n"
                "simd_cosine_similarity/scalar/384: benchmark\n"
                "simd_normalize/scalar/384: benchmark\n"
                "simd_dot_product_extra/scalar/384: benchmark\n"
                "int8_raw_dot_product/dot_product_i8_raw/128: benchmark\n"
            )
            expected_groups = frozenset({
                "simd_dot_product", "simd_cosine_similarity", "simd_normalize",
                "simd_dot_product_extra", "int8_raw_dot_product",
            })
            demoted_proc = subprocess.run(
                ["bash", str(helper), "lattice-embed:simd", str(listing_file)],
                capture_output=True, text=True, timeout=30,
            )
            shell_emitted = frozenset(
                ln.strip() for ln in demoted_proc.stdout.splitlines() if ln.strip()
            )
            gated_proc = subprocess.run(
                ["bash", str(helper), "lattice-inference:elementwise_cpu_bench",
                 str(listing_file)],
                capture_output=True, text=True, timeout=30,
            )
            gated_emitted = [ln for ln in gated_proc.stdout.splitlines() if ln.strip()]
            if demoted_proc.returncode != 0:
                failures.append(
                    f"manifest-handoff: demoted-target probe exited "
                    f"{demoted_proc.returncode}: {demoted_proc.stderr}"
                )
            elif shell_emitted != expected_groups:
                failures.append(
                    "manifest-handoff: demoted target emitted "
                    f"{sorted(shell_emitted)}, expected every listing group "
                    f"{sorted(expected_groups)}"
                )
            elif gated_proc.returncode != 0:
                failures.append(
                    f"manifest-handoff: non-demoted-target probe exited "
                    f"{gated_proc.returncode}: {gated_proc.stderr}"
                )
            elif gated_emitted:
                failures.append(
                    "manifest-handoff: non-demoted target emitted "
                    f"{gated_emitted} — cross-target exemption leak"
                )
            else:
                manifest_dir = root / "manifest"
                emb_a = manifest_dir / "simd_dot_product" / "384"
                _fabricate_bench(emb_a, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
                emb_b = manifest_dir / "simd_normalize" / "384"
                _fabricate_bench(emb_b, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
                inf_c = manifest_dir / "rms_norm" / "4096"
                _fabricate_bench(inf_c, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)

                manifest_results: dict[str, BenchResult] = {}
                for cf in find_change_files(manifest_dir):
                    r = parse_bench(cf, manifest_dir, baseline_name="compare-base")
                    if r is not None:
                        manifest_results[r.name] = r

                needed = {"simd_dot_product/384", "simd_normalize/384", "rms_norm/4096"}
                if not needed.issubset(manifest_results):
                    failures.append("manifest-handoff fixture: not all benches parsed")
                else:
                    manifest_gated_fails = {
                        r.name for r in manifest_results.values()
                        if r.verdict() == "FAIL" and not r.is_informational(shell_emitted)
                    }
                    if "simd_dot_product/384" in manifest_gated_fails:
                        failures.append(
                            "manifest-handoff: demoted simd_dot_product "
                            "leaked into gated fails"
                        )
                    if "simd_normalize/384" in manifest_gated_fails:
                        failures.append(
                            "manifest-handoff: simd_normalize gated despite "
                            "target-level demotion (listing-derivation broken)"
                        )
                    if "rms_norm/4096" not in manifest_gated_fails:
                        failures.append(
                            "manifest-handoff: inference group rms_norm did not gate"
                        )

        # Composed-path collision guard. The two probes above show the
        # helper is target-aware in isolation, but bench-compare.sh used to
        # concatenate every target's helper output into one flat file with
        # no target attribution — a group name demoted for one target
        # silently exempted an identically-named group produced by a
        # different, gated target. This drives the resolver
        # (scripts/lib/resolve-informational-groups.sh) with a fabricated
        # listing where the demoted target's groups include `rms_norm`,
        # which also appears in the gated target's listing, and asserts
        # the collision gates instead of staying informational.
        resolver = Path(__file__).resolve().parent / "lib" / "resolve-informational-groups.sh"
        if not resolver.exists():
            failures.append(f"collision-guard: resolver missing at {resolver}")
        else:
            collision_dir = root / "collision-listing"
            collision_dir.mkdir(parents=True, exist_ok=True)
            embed_listing = collision_dir / "embed-list.txt"
            embed_listing.write_text(
                "simd_dot_product/scalar/384: benchmark\n"
                "simd_normalize/scalar/384: benchmark\n"
                "rms_norm/scalar/384: benchmark\n"  # fabricated bare-name collision
            )
            inference_listing = collision_dir / "inference-list.txt"
            inference_listing.write_text(
                "rms_norm/4096: benchmark\n"
                "gelu/4096: benchmark\n"
            )

            def list_groups(listing_path: Path) -> str:
                proc = subprocess.run(
                    ["bash", str(helper), "--list-groups", str(listing_path)],
                    capture_output=True, text=True, timeout=30,
                )
                return proc.stdout

            demoted_groups_file = collision_dir / "demoted.txt"
            demoted_groups_file.write_text(list_groups(embed_listing))
            gated_groups_file = collision_dir / "gated.txt"
            gated_groups_file.write_text(list_groups(inference_listing))

            resolve_proc = subprocess.run(
                ["bash", str(resolver),
                 str(demoted_groups_file), "lattice-embed:simd",
                 str(gated_groups_file), "lattice-inference:elementwise_cpu_bench"],
                capture_output=True, text=True, timeout=30,
            )
            resolved = frozenset(
                ln.strip() for ln in resolve_proc.stdout.splitlines() if ln.strip()
            )
            if resolve_proc.returncode != 0:
                failures.append(
                    f"collision-guard: resolver exited {resolve_proc.returncode}: "
                    f"{resolve_proc.stderr}"
                )
            if "rms_norm" in resolved:
                failures.append(
                    "collision-guard: rms_norm (demoted+gated collision) leaked "
                    "into the informational set instead of gating"
                )
            if "rms_norm" not in resolve_proc.stderr or "lattice-embed:simd" not in resolve_proc.stderr:
                failures.append(
                    "collision-guard: no stderr warning naming the colliding "
                    "group and both targets"
                )
            if not {"simd_dot_product", "simd_normalize"}.issubset(resolved):
                failures.append(
                    "collision-guard: non-colliding demoted groups lost "
                    "informational status — resolver over-suppressed"
                )

    # --require-measurements: the lane must not read "nothing measured" as
    # "nothing regressed" (#1105 review). bench-compare.sh creates the criterion
    # directory itself before benching, and the cargo pipelines swallow bench
    # failures, so an EMPTY-but-present root is the realistic failure shape.
    with tempfile.TemporaryDirectory() as td:
        gate = Path(__file__).resolve()
        empty_root = Path(td) / "empty" / "criterion"
        empty_root.mkdir(parents=True)

        def _run(root: Path, *extra: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, str(gate), str(root), "selftest-arch", *extra],
                capture_output=True, text=True, timeout=60,
            )

        # Without the flag, an absent baseline stays a pass (first-run semantics).
        if _run(empty_root).returncode != 0:
            failures.append("require-measurements: empty root without the flag "
                            "must still exit 0 (first-run semantics changed)")
        # With it, the same empty root must refuse to certify.
        if _run(empty_root, "--require-measurements").returncode != 2:
            failures.append("require-measurements: empty criterion root exited "
                            "0 with the flag set — the lane can go green having "
                            "measured nothing (the #1105 fail-open)")

        # A real gating comparison must still pass, or the flag is just a brake.
        ok_root = Path(td) / "ok" / "criterion"
        _fabricate_bench(ok_root / "grp_ok" / "bench_ok", "compare-base")
        if _run(ok_root, "--require-measurements").returncode != 0:
            failures.append("require-measurements: a parsed gating comparison "
                            "was rejected — the flag over-fails")

        # All-informational is measured-but-unjudgeable: nothing could FAIL.
        info_file = Path(td) / "informational.txt"
        info_file.write_text("grp_info\n")
        info_root = Path(td) / "info" / "criterion"
        _fabricate_bench(info_root / "grp_info" / "bench_info", "compare-base")
        if _run(info_root, "--require-measurements",
                "--informational-groups-file", str(info_file)).returncode != 2:
            failures.append("require-measurements: an all-informational run was "
                            "certified — no gating comparison was judged")

    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    if failures:
        print(f"SELFTEST: FAIL ({len(failures)} failure(s))")
        return 1
    print("SELFTEST: PASS — base/, compare-base/, orphan-warn, named-wins-over-stale-base, "
          "multi-sibling-refusal, manifest-handoff (demoted target informational, "
          "non-demoted target gated), and composed-path collision-guard, and require-measurements (empty root, gating pass, all-informational refusal) all correct")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("criterion_root", type=Path, nargs="?",
                    help="Path to target/criterion (or per-bench root)")
    ap.add_argument("arch", nargs="?",
                    help="Arch label for the report header (e.g. aarch64-linux)")
    ap.add_argument("--out", type=Path, help="Write markdown report to this path")
    ap.add_argument("--baseline-name", default="compare-base",
                    help="Named-baseline dir to look for when base/ is absent "
                         "(default: compare-base, matching bench-compare.sh)")
    ap.add_argument("--informational-groups-file", type=Path, default=None,
                    help="Path to a file listing Criterion top-level group names (one per "
                         "line) to measure+report but exclude from gating/exit-code — quick-"
                         "mode noise-floor groups (lattice#714). Omit for full-mode runs.")
    ap.add_argument("--require-measurements", action="store_true",
                    help="Fail (exit 2) instead of passing when the run produced no gating "
                         "comparison to judge. Without this, an absent baseline exits 0, which "
                         "is right for a first run but wrong for an automated lane: it cannot "
                         "tell 'nothing regressed' from 'nothing was measured'.")
    ap.add_argument("--selftest", action="store_true",
                    help="Run the fixture self-test (no criterion_root/arch needed) and exit")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()

    if args.criterion_root is None or args.arch is None:
        ap.error("criterion_root and arch are required unless --selftest is passed")

    if not args.criterion_root.exists():
        print(f"error: {args.criterion_root} does not exist", file=sys.stderr)
        return 2

    change_files = find_change_files(args.criterion_root)
    if not change_files:
        print(f"warn: no change/estimates.json under {args.criterion_root}; baseline missing?",
              file=sys.stderr)
        if args.require_measurements:
            print(f"error: --require-measurements set but no change/estimates.json under "
                  f"{args.criterion_root}: the run produced no comparison, so it is not "
                  f"evidence that nothing regressed.", file=sys.stderr)
            return 2
        # Treat missing baseline as pass — first run on a bench has no comparison.
        if args.out:
            args.out.write_text(f"### `{args.arch}` — no baseline to compare\n\n"
                                f"No `change/estimates.json` found; this is expected on the "
                                f"first run for a bench. Future runs will gate against this run.\n")
        return 0

    results = []
    for cf in change_files:
        r = parse_bench(cf, args.criterion_root, args.baseline_name)
        if r is not None:
            results.append(r)

    if not results:
        print("error: change files found but all failed to parse", file=sys.stderr)
        return 2

    informational_groups = load_informational_groups(args.informational_groups_file)
    report = render_report(results, args.arch, informational_groups)
    print(report)
    if args.out:
        args.out.write_text(report)

    gating = [r for r in results if not r.is_informational(informational_groups)]

    # Parsed results are not automatically judgeable results. If every parsed
    # result is informational, nothing in this run could have produced a FAIL,
    # so a zero exit says only that no gating comparison existed.
    if args.require_measurements and not gating:
        print(f"error: --require-measurements set but all {len(results)} parsed result(s) are "
              f"informational: no gating comparison was judged.", file=sys.stderr)
        return 2

    fails = sum(1 for r in gating if r.verdict() == "FAIL")
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
