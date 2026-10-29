"""Fixture self-test for scripts/perf-bench-gate.py, run by `perf-bench-gate.py --selftest`.

The gate hands its module namespace to `run`, which binds the gate names the
fixtures use and then runs the self-test. The fixtures call the gate's parsing
and rendering functions in process and run the gate script as a subprocess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GATE_NAMES = (
    "AMBIENT_SAMPLE_SCHEMA",
    "BenchResult",
    "EXIT_ERROR",
    "EXIT_NOT_MEASURABLE",
    "EXIT_REGRESSION",
    "FAIL_PCT",
    "MACHINE_STATE_SCHEMA",
    "PROVENANCE_FIELDS",
    "REQUIRED_AMBIENT_PHASES",
    "find_change_files",
    "order_balance_pair",
    "parse_bench",
    "render_report",
)


def run(gate: dict) -> int:
    """Run the self-test against the gate whose module namespace is `gate`.

    The fixtures read the gate names bound here, so `run_selftest` is only callable through `run`.
    """
    globals().update({name: gate[name] for name in GATE_NAMES})
    globals()["GATE_FILE"] = gate["__file__"]
    return run_selftest()


def _fabricate_baseline(bench_dir: Path, baseline_dirname: str,
                        base_ns: float = 90.0) -> None:
    """Write one fake Criterion named-baseline estimate for --selftest."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / baseline_dirname).mkdir(parents=True, exist_ok=True)
    (bench_dir / baseline_dirname / "estimates.json").write_text(
        json.dumps({"mean": {"point_estimate": base_ns}}))


def _fabricate_bench(bench_dir: Path, baseline_dirname: str,
                     point: float = 0.10, ci_low: float = 0.05, ci_high: float = 0.15,
                     new_ns: float = 100.0, base_ns: float = 90.0) -> None:
    """Write a fake Criterion bench dir (new/, <baseline_dirname>/, change/) for --selftest."""
    _fabricate_baseline(bench_dir, baseline_dirname, base_ns)
    (bench_dir / "new").mkdir(exist_ok=True)
    (bench_dir / "new" / "estimates.json").write_text(
        json.dumps({"mean": {"point_estimate": new_ns}}))
    (bench_dir / "change").mkdir(exist_ok=True)
    (bench_dir / "change" / "estimates.json").write_text(json.dumps({
        "mean": {
            "point_estimate": point,
            "confidence_interval": {"lower_bound": ci_low, "upper_bound": ci_high},
        }
    }))


def _fabricate_sample(artifact_dir: Path, count: int, mode: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "sample.json").write_text(json.dumps({
        "sampling_mode": mode,
        "iters": [float(index + 1) for index in range(count)],
        "times": [float(index + 1) for index in range(count)],
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
    import os
    import subprocess
    import tempfile

    failures: list[str] = []

    # lattice#1137: a fixed-order single Criterion pair can report a narrow,
    # gate-crossing regression on byte-identical source when both arms drift in
    # the same direction. ABBA supplies the mirrored B→A ratio. The balanced
    # source effect must collapse to zero while retaining the observed order
    # term as an explicit uncertainty bound.
    directional_drift = BenchResult(
        name="softmax_attention/512",
        point=0.10,
        ci_low=0.095,
        ci_high=0.105,
        new_ns=110.0,
        old_ns=100.0,
        head_sample_count=20,
        head_sampling_mode="Flat",
        base_sample_count=20,
        base_sampling_mode="Flat",
    )
    reverse_directional_drift = BenchResult(
        name="softmax_attention/512",
        point=0.10,
        ci_low=0.095,
        ci_high=0.105,
        new_ns=133.1,
        old_ns=121.0,
        head_sample_count=20,
        head_sampling_mode="Flat",
        base_sample_count=20,
        base_sampling_mode="Flat",
    )
    if directional_drift.verdict() != "FAIL":
        failures.append(
            "order-balance fixture no longer reproduces the old fixed-order false FAIL"
        )
    balanced_drift = order_balance_pair(
        directional_drift, reverse_directional_drift
    )
    if balanced_drift.verdict() == "FAIL" or abs(balanced_drift.point) > 1e-12:
        failures.append(
            "order-balance: identical-source directional drift still produced "
            f"{balanced_drift.verdict()} at {balanced_drift.point_pct:+.6f}%"
        )
    if (
        balanced_drift.order_bias_bound_pct is None
        or balanced_drift.order_bias_bound_pct <= FAIL_PCT
    ):
        failures.append(
            "order-balance: gate-sized directional drift was not retained as "
            "a fail-closed order-bias bound"
        )

    # A real 20% source slowdown under a 2% directional drift remains
    # distinguishable after the same correction. This prevents the control
    # envelope from becoming an unconditional regression exemption.
    true_regression_forward = BenchResult(
        name="softmax_attention/512",
        point=0.224,
        ci_low=0.222,
        ci_high=0.226,
        new_ns=122.4,
        old_ns=100.0,
        head_sample_count=20,
        head_sampling_mode="Flat",
        base_sample_count=20,
        base_sampling_mode="Flat",
    )
    true_regression_reverse = BenchResult(
        name="softmax_attention/512",
        point=-0.15,
        ci_low=-0.152,
        ci_high=-0.148,
        new_ns=106.1208,
        old_ns=124.848,
        head_sample_count=20,
        head_sampling_mode="Flat",
        base_sample_count=20,
        base_sampling_mode="Flat",
    )
    balanced_regression = order_balance_pair(
        true_regression_forward, true_regression_reverse
    )
    if balanced_regression.verdict() != "FAIL":
        failures.append(
            "order-balance: a true 20% regression under 2% drift did not remain "
            f"a FAIL: point={balanced_regression.point_pct:+.2f}%, "
            f"bound=[{balanced_regression.ci_low_pct:+.2f}%, "
            f"{balanced_regression.ci_high_pct:+.2f}%]"
        )

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

        # lattice#714 / lattice#1090: classification follows the target-qualified
        # Criterion root, never a slash-derived group prefix. Criterion 0.5 permits
        # `/` in group names and also uses `/` between group/function/parameter, so
        # splitting the rendered ID cannot recover the group boundary.
        noisy_dir = root / "grp_f" / "noisy_fail"
        _fabricate_bench(noisy_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
        real_dir = root / "grp_g" / "real_fail"
        _fabricate_bench(real_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)
        slash_dir = root / "slash" / "is" / "legal" / "noisy_fail"
        _fabricate_bench(slash_dir, "compare-base", point=0.10, ci_low=0.10, ci_high=0.20)

        for cf in find_change_files(root):
            r = parse_bench(cf, root, baseline_name="compare-base")
            if r is not None:
                results[r.name] = r

        embed_target = "lattice-embed:simd"
        inference_target = "lattice-inference:elementwise_cpu_bench"
        needed = {"grp_f/noisy_fail", "grp_g/real_fail", "slash/is/legal/noisy_fail"}
        if not needed.issubset(results):
            failures.append("target-qualified fixture: benches not parsed")
        else:
            noisy = results["grp_f/noisy_fail"]
            real = results["grp_g/real_fail"]
            slash = results["slash/is/legal/noisy_fail"]
            if not noisy.is_informational(embed_target, embed_target):
                failures.append("target-qualified: demoted target did not classify informational")
            if real.is_informational(inference_target, embed_target):
                failures.append("target-qualified: gated target crossed into demotion")
            if not slash.is_informational(embed_target, embed_target):
                failures.append("target-qualified: slash-containing group was misclassified")

            embed_report = render_report(
                [noisy, slash], "selftest-embed", embed_target, embed_target)
            inference_report = render_report(
                [real], "selftest-inference", inference_target, embed_target)
            if "ℹ️" not in embed_report or "slash/is/legal/noisy_fail" not in embed_report:
                failures.append("target-qualified: informational report lost slash-group bench")
            if "❌" not in inference_report or "grp_g/real_fail" not in inference_report:
                failures.append("target-qualified: gated FAIL missing from inference report")
            if "bench-allow-regression" in embed_report + inference_report:
                failures.append("rendered report advertises an unsupported label override")

        # The shell owns manifest membership. Python receives only the exact
        # target identity selected by that shell path, so there is no second
        # parser whose group-name semantics can drift. The controlled manifest
        # probes cover CRLF, surrounding whitespace, comments, duplicate
        # entries and C-locale sorting; malformed or whitespace-bearing keys
        # must refuse rather than demote.
        helper = Path(GATE_FILE).resolve().parent / "lib" / "bench-informational-targets.sh"
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
            demoted_proc = subprocess.run(
                ["bash", str(helper), "--is-informational", embed_target],
                capture_output=True, text=True, timeout=30,
            )
            gated_proc = subprocess.run(
                ["bash", str(helper), "--is-informational", inference_target],
                capture_output=True, text=True, timeout=30,
            )
            if demoted_proc.returncode != 0:
                failures.append(
                    f"manifest-handoff: demoted-target probe exited "
                    f"{demoted_proc.returncode}: {demoted_proc.stderr}"
                )
            elif gated_proc.returncode != 1:
                failures.append(
                    "manifest-handoff: non-demoted target did not return the "
                    f"expected predicate miss (1): {gated_proc.returncode}"
                )

            policy_dir = root / "manifest-policy"
            policy_dir.mkdir(parents=True, exist_ok=True)
            normalized_manifest = policy_dir / "normalized.txt"
            normalized_manifest.write_bytes(
                b"  z-target  # comment\r\n"
                b"\r\n"
                b"a-target \r\n"
                b"z-target\r\n"
            )
            policy_env = {
                **os.environ,
                "INFO_TARGETS_MANIFEST": str(normalized_manifest),
            }
            normalized = subprocess.run(
                ["bash", str(helper), "--print-targets"],
                capture_output=True, text=True, timeout=30, env=policy_env,
            )
            if normalized.returncode != 0 or normalized.stdout != "a-target\nz-target\n":
                failures.append(
                    "manifest-normalization: CRLF/whitespace/dedup/C-sort drifted: "
                    f"rc={normalized.returncode}, stdout={normalized.stdout!r}, "
                    f"stderr={normalized.stderr!r}"
                )

            malformed_manifest = policy_dir / "malformed.txt"
            malformed_manifest.write_text("one-target two-targets\n")
            malformed_env = {
                **os.environ,
                "INFO_TARGETS_MANIFEST": str(malformed_manifest),
            }
            malformed = subprocess.run(
                ["bash", str(helper), "--print-targets"],
                capture_output=True, text=True, timeout=30, env=malformed_env,
            )
            if malformed.returncode != 2:
                failures.append(
                    "manifest-normalization: malformed policy did not fail closed "
                    f"(rc={malformed.returncode})"
                )
            bad_key = subprocess.run(
                ["bash", str(helper), "--is-informational", "bad target"],
                capture_output=True, text=True, timeout=30,
            )
            if bad_key.returncode != 2:
                failures.append(
                    "manifest-normalization: whitespace-bearing target key did not refuse"
                )

    # --require-measurements: the lane must not read "nothing measured" as
    # "nothing regressed" (#1105). bench-compare.sh creates the criterion
    # directory itself before benching, and the cargo pipelines swallow bench
    # failures, so an EMPTY-but-present root is the realistic failure shape.
    with tempfile.TemporaryDirectory() as td:
        gate = Path(GATE_FILE).resolve()
        empty_root = Path(td) / "empty" / "criterion"
        empty_root.mkdir(parents=True)

        def _run(root: Path, *extra: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, str(gate), str(root), "selftest-arch", *extra],
                capture_output=True, text=True, timeout=60,
            )

        def _prepare(root: Path, mode: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [
                    sys.executable,
                    str(gate),
                    str(root),
                    "--baseline-name",
                    "compare-base",
                    mode,
                ],
                capture_output=True,
                text=True,
                timeout=60,
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
        empty_informational = _run(
            empty_root,
            "--require-measurements",
            "--target",
            "lattice-embed:simd",
            "--informational-target",
            "lattice-embed:simd",
        )
        if empty_informational.returncode != 2:
            failures.append(
                "require-measurements: an empty informational target exited "
                "0 with enforcement enabled"
            )
        if "for target lattice-embed:simd" not in empty_informational.stderr:
            failures.append(
                "require-measurements: empty informational target was not "
                "reported with its target identity"
            )

        # A stale comparison against a DIFFERENT baseline is not a substitute
        # for the exact selected base set. Mutation-sensitive: condition
        # selected-set filtering on a non-empty set (or omit the empty-set
        # refusal) and this valid base/new/change tree exits 0.
        wrong_baseline_root = Path(td) / "wrong-baseline" / "criterion"
        _fabricate_bench(
            wrong_baseline_root / "rms_norm" / "4096", "base"
        )
        wrong_baseline = _run(
            wrong_baseline_root, "--require-measurements"
        )
        if wrong_baseline.returncode != 2:
            failures.append(
                "require-measurements: stale default-base comparison certified "
                "an empty selected compare-base set"
            )
        if "selected baseline 'compare-base' contains no benchmark estimates" not in (
            wrong_baseline.stderr
        ):
            failures.append(
                "require-measurements: empty selected-set refusal did not name "
                "the selected baseline"
            )

        # A real gating comparison must still pass, or the flag is just a brake.
        ok_root = Path(td) / "ok" / "criterion"
        _fabricate_bench(ok_root / "grp_ok" / "bench_ok", "compare-base")
        if _run(ok_root, "--require-measurements").returncode != 0:
            failures.append("require-measurements: a parsed gating comparison "
                            "was rejected — the flag over-fails")
        missing_order_control = _run(
            ok_root,
            "--require-measurements",
            "--require-order-balance",
        )
        if (
            missing_order_control.returncode != EXIT_ERROR
            or "needs --order-control-root" not in missing_order_control.stderr
        ):
            failures.append(
                "order-balance: enforcing request without reverse evidence did "
                "not fail closed"
            )

        drift_root = Path(td) / "directional-drift" / "criterion"
        drift_control = Path(td) / "directional-drift-control" / "criterion"
        _fabricate_bench(
            drift_root / "softmax_attention" / "512",
            "compare-base",
            point=0.10,
            ci_low=0.095,
            ci_high=0.105,
            new_ns=110.0,
            base_ns=100.0,
        )
        _fabricate_bench(
            drift_control / "softmax_attention" / "512",
            "compare-head",
            point=0.10,
            ci_low=0.095,
            ci_high=0.105,
            new_ns=133.1,
            base_ns=121.0,
        )
        drift_gate = _run(
            drift_root,
            "--require-measurements",
            "--require-order-balance",
            "--order-control-root",
            str(drift_control),
        )
        if drift_gate.returncode != EXIT_NOT_MEASURABLE:
            failures.append(
                "order-balance: identical-source directional drift exited "
                f"{drift_gate.returncode}, expected NOT_MEASURABLE (3)"
            )
        if "order-bias bound above" not in drift_gate.stderr:
            failures.append(
                "order-balance: directional-drift refusal did not identify the "
                "order-bias evidence"
            )

        regression_root = Path(td) / "true-regression" / "criterion"
        regression_control = (
            Path(td) / "true-regression-control" / "criterion"
        )
        _fabricate_bench(
            regression_root / "softmax_attention" / "512",
            "compare-base",
            point=0.224,
            ci_low=0.222,
            ci_high=0.226,
            new_ns=122.4,
            base_ns=100.0,
        )
        _fabricate_bench(
            regression_control / "softmax_attention" / "512",
            "compare-head",
            point=-0.15,
            ci_low=-0.152,
            ci_high=-0.148,
            new_ns=106.1208,
            base_ns=124.848,
        )
        regression_gate = _run(
            regression_root,
            "--require-measurements",
            "--require-order-balance",
            "--order-control-root",
            str(regression_control),
        )
        if regression_gate.returncode != EXIT_REGRESSION:
            failures.append(
                "order-balance: true 20% regression under 2% drift exited "
                f"{regression_gate.returncode}, expected regression (1)"
            )

        incomplete_control = (
            Path(td) / "incomplete-order-control" / "criterion"
        )
        _fabricate_bench(
            incomplete_control / "different_bench",
            "compare-head",
        )
        incomplete_order = _run(
            drift_root,
            "--require-measurements",
            "--require-order-balance",
            "--order-control-root",
            str(incomplete_control),
        )
        if incomplete_order.returncode != EXIT_ERROR:
            failures.append(
                "order-balance: mismatched reverse benchmark set did not fail "
                "closed with exit 2"
            )

        for malformed_name, malformed_value, malformed_field in (
            ("string-point", "corrupt", "point_estimate"),
            ("boolean-ci", True, "lower_bound"),
        ):
            malformed_control = (
                Path(td) / f"malformed-order-control-{malformed_name}" / "criterion"
            )
            malformed_bench = malformed_control / "softmax_attention" / "512"
            _fabricate_bench(
                malformed_bench,
                "compare-head",
                point=0.10,
                ci_low=0.095,
                ci_high=0.105,
                new_ns=133.1,
                base_ns=121.0,
            )
            malformed_change = malformed_bench / "change" / "estimates.json"
            malformed_payload = json.loads(malformed_change.read_text())
            if malformed_field == "point_estimate":
                malformed_payload["mean"]["point_estimate"] = malformed_value
            else:
                malformed_payload["mean"]["confidence_interval"][
                    malformed_field
                ] = malformed_value
            malformed_change.write_text(json.dumps(malformed_payload))
            malformed_order = _run(
                drift_root,
                "--require-measurements",
                "--require-order-balance",
                "--order-control-root",
                str(malformed_control),
            )
            if malformed_order.returncode != EXIT_ERROR:
                failures.append(
                    f"order-balance: malformed reverse {malformed_name} exited "
                    f"{malformed_order.returncode}, expected input error (2)"
                )
            if "Traceback" in malformed_order.stderr:
                failures.append(
                    f"order-balance: malformed reverse {malformed_name} escaped "
                    "as an uncaught exception"
                )

        no_coverage_root = Path(td) / "no-baseline-coverage" / "criterion"
        _fabricate_bench(
            no_coverage_root / "existing_group" / "existing_bench",
            "compare-base",
        )
        head_only = no_coverage_root / "new_group" / "new_bench" / "new"
        head_only.mkdir(parents=True)
        (head_only / "estimates.json").write_text(
            json.dumps({"mean": {"point_estimate": 100.0}})
        )
        no_coverage_ambient = no_coverage_root.parent / "ambient.jsonl"
        no_coverage_ambient.write_text(
            "".join(
                json.dumps(
                    {
                        "schema": AMBIENT_SAMPLE_SCHEMA,
                        "phase": phase,
                        "idle_pct": 95.0,
                    }
                )
                + "\n"
                for phase in REQUIRED_AMBIENT_PHASES
            )
        )
        no_coverage_status = no_coverage_root.parent / "status.json"
        no_coverage = _run(
            no_coverage_root,
            "--require-measurements",
            "--target",
            "lattice-inference:elementwise_cpu_bench",
            "--ambient-samples",
            str(no_coverage_ambient),
            "--status-out",
            str(no_coverage_status),
        )
        if no_coverage.returncode != 2:
            failures.append(
                "head-baseline-completeness: head benchmark without baseline exited "
                f"{no_coverage.returncode} instead of 2"
            )
        if "NO COVERAGE" not in no_coverage.stderr:
            failures.append(
                "head-baseline-completeness: refusal did not label unbaselined "
                "head benchmarks NO COVERAGE"
            )
        no_coverage_payload = json.loads(no_coverage_status.read_text())
        if (
            no_coverage_payload["verdict"] != "error"
            or no_coverage_payload["exit_code"] != 2
            or "NO COVERAGE" not in no_coverage_payload["reason"]
        ):
            failures.append(
                "head-baseline-completeness: status surface rendered unbaselined "
                "head benchmark as a pass"
            )
        if (
            "  - lattice-inference:elementwise_cpu_bench: new_group/new_bench"
            not in no_coverage.stderr
        ):
            failures.append(
                "head-baseline-completeness: refusal did not name exact head-only "
                "bench ID 'new_group/new_bench'"
            )
        if "produced no head comparison" in no_coverage.stderr:
            failures.append(
                "inventory-direction: head-without-baseline was collapsed into "
                "the baseline-without-head refusal"
            )

        # A stale/non-selected change/ dir must not exempt an
        # uncovered head artifact. existing_group/existing_bench compares
        # cleanly against the selected baseline; stale_group/stale_bench has a
        # head new/ artifact and a change/ comparison, but that change/ is
        # against a non-selected "stale-baseline", not "compare-base".
        # Mutation-sensitive: restore the `- all_change_ids` subtraction and
        # this exits 0 instead of 2.
        stale_change_root = Path(td) / "stale-change-coverage" / "criterion"
        _fabricate_bench(
            stale_change_root / "existing_group" / "existing_bench",
            "compare-base",
        )
        _fabricate_bench(
            stale_change_root / "stale_group" / "stale_bench",
            "stale-baseline",
        )
        stale_change = _run(
            stale_change_root,
            "--require-measurements",
            "--target",
            "lattice-inference:elementwise_cpu_bench",
        )
        if stale_change.returncode != 2:
            failures.append(
                "stale-change-coverage: a head artifact whose only companion "
                "is a stale non-selected change/ exited "
                f"{stale_change.returncode} instead of 2"
            )
        if "NO COVERAGE" not in stale_change.stderr:
            failures.append(
                "stale-change-coverage: refusal did not label the "
                "stale-change-only head artifact NO COVERAGE"
            )
        if (
            "  - lattice-inference:elementwise_cpu_bench: stale_group/stale_bench"
            not in stale_change.stderr
        ):
            failures.append(
                "stale-change-coverage: refusal did not name exact bench ID "
                "'stale_group/stale_bench'"
            )

        # A target can be intentionally all-informational only when the exact
        # target key accompanies both the root and the demotion; a mismatched key
        # must refuse. Demotion is target-wide, so it empties the gated set: this
        # invocation has no verdict to render and exits 3 with its rows beside the
        # refusal. It remains valid inside an enforcing multi-target run —
        # bench-compare decides the RUN from the arms that did gate, and refuses
        # only when none did.
        # Mutation-sensitive: drop the empty-gated-set refusal in main() and this
        # exits 0 having printed a tick over nothing.
        info_root = Path(td) / "info" / "criterion"
        _fabricate_bench(info_root / "grp_info" / "bench_info", "compare-base")
        _fabricate_bench(
            info_root / "grp_clean" / "bench_clean",
            "compare-base",
            point=0.001,
            ci_low=-0.005,
            ci_high=0.007,
        )
        all_informational = _run(
            info_root, "--require-measurements",
            "--target", "lattice-embed:simd",
            "--informational-target", "lattice-embed:simd")
        if all_informational.returncode != 3:
            failures.append(
                "empty-gated-set: an all-informational target exited "
                f"{all_informational.returncode} instead of 3")
        if "NO GATED MEASUREMENT" not in all_informational.stdout:
            failures.append(
                "empty-gated-set: the refusal was not labelled NO GATED "
                "MEASUREMENT in the report")
        if "gated benches within noise band" in all_informational.stdout:
            failures.append(
                "empty-gated-set: a green tick was printed over an empty "
                "gated set")
        beside_verdict = all_informational.stdout.partition(
            "<details><summary>All ")[0]
        for demoted_id in ("grp_info/bench_info", "grp_clean/bench_clean"):
            if demoted_id not in beside_verdict:
                failures.append(
                    f"empty-gated-set: demoted row {demoted_id!r} was not "
                    "rendered beside the verdict")
        if _run(info_root, "--require-measurements",
                "--target", "lattice-inference:elementwise_cpu_bench",
                "--informational-target", "lattice-embed:simd").returncode != 2:
            failures.append("target-qualified: mismatched informational target "
                            "did not fail closed")

        # PARTIAL is the shape the first fix missed: asking whether ANY judgeable
        # comparison exists passes a run where one target compared cleanly and the
        # other produced malformed selected-baseline data. The unmeasured target
        # is exactly the one a green exit would be vouching for.
        mixed_root = Path(td) / "mixed" / "criterion"
        _fabricate_bench(mixed_root / "grp_ok" / "bench_ok", "compare-base")
        mixed_orphan = mixed_root / "grp_bad" / "bench_orphan"
        (mixed_orphan / "compare-base").mkdir(parents=True)
        (mixed_orphan / "compare-base" / "estimates.json").write_text("{}")
        (mixed_orphan / "new").mkdir(parents=True)
        (mixed_orphan / "new" / "estimates.json").write_text(
            json.dumps({"mean": {"point_estimate": 100.0}}))
        (mixed_orphan / "change").mkdir(parents=True)
        (mixed_orphan / "change" / "estimates.json").write_text(json.dumps({
            "mean": {"point_estimate": 0.10,
                     "confidence_interval": {"lower_bound": 0.05, "upper_bound": 0.15}}
        }))
        mixed = _run(mixed_root, "--require-measurements")
        if mixed.returncode != 2:
            failures.append("require-measurements: a run with one judgeable and one "
                            "unresolvable comparison exited "
                            f"{mixed.returncode} — a partial A/B was certified")
        if "bench_orphan" not in mixed.stderr:
            failures.append("require-measurements: the partial-run refusal did not "
                            "name the unjudged bench")
        # Without the flag the same mixed root stays a pass: the reporter is
        # allowed to render what it has.
        if _run(mixed_root).returncode != 0:
            failures.append("require-measurements: the mixed root without the flag "
                            "must still exit 0 (reporter behavior changed)")

        # lattice#1204: completeness is selected-base minus head-change, not
        # merely change-found minus change-parsed. These use actual IDs from
        # the shipping inference/embed targets, including Criterion's nested
        # function/parameter path. Mutation-sensitive: remove the set
        # subtraction (or exempt informational groups from it) and this exits
        # 0 even though two base measurements vanished from the head.
        coverage_root = Path(td) / "coverage" / "criterion"
        _fabricate_bench(coverage_root / "rms_norm" / "896", "compare-base")
        _fabricate_baseline(coverage_root / "rms_norm" / "4096", "compare-base")
        _fabricate_baseline(
            coverage_root / "simd_dot_product" / "scalar" / "384",
            "compare-base",
        )
        _fabricate_bench(
            coverage_root / "renamed_rms_norm" / "4096", "compare-base"
        )
        # An estimates file directly under the selected baseline directory has
        # no benchmark ID and is unrelated Criterion-tree debris, not a bench.
        (coverage_root / "compare-base").mkdir(parents=True)
        (coverage_root / "compare-base" / "estimates.json").write_text("{}")

        coverage = _run(
            coverage_root,
            "--require-measurements",
            "--target",
            "lattice-inference:elementwise_cpu_bench",
        )
        if coverage.returncode != 2:
            failures.append(
                "baseline-completeness: missing head benchmarks exited "
                f"{coverage.returncode} instead of 2"
            )
        for expected_id in ("rms_norm/4096", "simd_dot_product/scalar/384"):
            qualified = (
                "  - lattice-inference:elementwise_cpu_bench: "
                f"{expected_id}"
            )
            if qualified not in coverage.stderr:
                failures.append(
                    "baseline-completeness: refusal did not name exact missing "
                    f"bench ID {expected_id!r}"
                )
        if "produced no head comparison" not in coverage.stderr:
            failures.append(
                "inventory-direction: baseline-without-head refusal lost its "
                "direction-specific description"
            )
        if "NO COVERAGE" in coverage.stderr:
            failures.append(
                "inventory-direction: baseline-without-head was collapsed into "
                "the head-without-baseline NO COVERAGE disposition"
            )
        if _run(
            coverage_root,
        ).returncode != 0:
            failures.append(
                "baseline-completeness: reporter mode rejected missing head "
                "benchmarks without --require-measurements"
            )

        informational_missing_root = (
            Path(td) / "coverage-informational" / "criterion"
        )
        _fabricate_baseline(
            informational_missing_root / "simd_dot_product" / "scalar" / "384",
            "compare-base",
        )
        informational_missing = _run(
            informational_missing_root,
            "--require-measurements",
            "--target",
            "lattice-embed:simd",
            "--informational-target",
            "lattice-embed:simd",
        )
        if informational_missing.returncode != 2:
            failures.append(
                "baseline-completeness: a missing informational target "
                "did not fail measurement completeness"
            )
        if (
            "  - lattice-embed:simd: simd_dot_product/scalar/384"
            not in informational_missing.stderr
        ):
            failures.append(
                "baseline-completeness: missing informational benchmark was "
                "not reported with its target identity"
            )

        # The CLI-selected name can itself be a relative path. Only that exact
        # suffix contributes to completeness; a stale compare-base tree must
        # not contaminate a run selecting snapshot/arm.
        custom_root = Path(td) / "custom-baseline" / "criterion"
        _fabricate_bench(custom_root / "rms_norm" / "896", "snapshot/arm")
        _fabricate_baseline(custom_root / "gelu" / "4096", "snapshot/arm")
        _fabricate_baseline(
            custom_root / "stale_compare_base" / "4096", "compare-base"
        )
        custom = _run(
            custom_root,
            "--require-measurements",
            "--baseline-name",
            "snapshot/arm",
        )
        if custom.returncode != 2 or "  - gelu/4096" not in custom.stderr:
            failures.append(
                "baseline-completeness: nested --baseline-name did not select "
                "and report the missing gelu/4096 benchmark"
            )
        if "stale_compare_base/4096" in custom.stderr:
            failures.append(
                "baseline-completeness: stale non-selected compare-base tree "
                "contaminated custom selected baseline"
            )

        baseline_copy_root = Path(td) / "baseline-copy" / "criterion"
        stale_selected = baseline_copy_root / "old_target" / "old_bench"
        _fabricate_bench(stale_selected, "compare-base")
        preserved_other = baseline_copy_root / "other_target" / "other_bench"
        _fabricate_bench(preserved_other, "previous-run")
        prepared_copy = _prepare(
            baseline_copy_root, "--prepare-baseline-copy"
        )
        if prepared_copy.returncode != 0:
            failures.append(
                f"baseline-copy freshness: prepare step failed: {prepared_copy.stderr}"
            )
        if (stale_selected / "compare-base").exists():
            failures.append(
                "baseline-copy freshness: stale exact selected baseline survived"
            )
        if (stale_selected / "new").exists() or (stale_selected / "change").exists():
            failures.append(
                "baseline-copy freshness: stale new/change siblings of a pruned "
                "selected baseline survived to false-fail head coverage"
            )
        if not (preserved_other / "previous-run" / "estimates.json").exists():
            failures.append(
                "baseline-copy freshness: unrelated named baseline was removed"
            )

        atomic_baseline_root = Path(td) / "atomic-baseline-copy" / "criterion"
        atomic_safe_baseline = atomic_baseline_root / "a_safe" / "1"
        _fabricate_baseline(atomic_safe_baseline, "compare-base")
        atomic_unsafe_baseline = atomic_baseline_root / "z_unsafe" / "1"
        (atomic_unsafe_baseline / "compare-base").mkdir(parents=True)
        outside_baseline = Path(td) / "outside-baseline-estimates.json"
        outside_baseline.write_text('{"mean":{"point_estimate":90.0}}')
        (atomic_unsafe_baseline / "compare-base" / "estimates.json").symlink_to(
            outside_baseline
        )
        atomic_baseline_prepare = _prepare(
            atomic_baseline_root, "--prepare-baseline-copy"
        )
        if atomic_baseline_prepare.returncode != 2:
            failures.append(
                "baseline-copy freshness: unsafe later baseline was not refused"
            )
        if not (
            atomic_safe_baseline / "compare-base" / "estimates.json"
        ).exists():
            failures.append(
                "baseline-copy freshness: cleanup partially deleted a safe "
                "baseline before refusing a later unsafe path"
            )

        # The in-place HEAD freshness step removes only new/change siblings of
        # exact selected-baseline benches, preserving that baseline and
        # unrelated Criterion trees. Mutation-sensitive: stop removing the
        # stale same-path change below and the deleted rms_norm/4096 appears
        # judged, defeating the completeness check.
        freshness_root = Path(td) / "freshness" / "criterion"
        removed_bench = freshness_root / "rms_norm" / "4096"
        _fabricate_bench(removed_bench, "compare-base")
        sibling_root = Path(td) / "freshness-sibling" / "criterion"
        sibling_bench = sibling_root / "simd_dot_product" / "scalar" / "384"
        _fabricate_bench(sibling_bench, "compare-base")
        prepared = _prepare(freshness_root, "--prepare-head")
        if prepared.returncode != 0:
            failures.append(
                f"head-freshness: prepare step failed: {prepared.stderr}"
            )
        if not (removed_bench / "compare-base" / "estimates.json").exists():
            failures.append("head-freshness: selected baseline was removed")
        if (removed_bench / "new").exists() or (removed_bench / "change").exists():
            failures.append(
                "head-freshness: stale selected-bench new/change artifacts survived"
            )
        if not (sibling_bench / "change" / "estimates.json").exists():
            failures.append(
                "head-freshness: preparing one target root touched its sibling"
            )
        freshness = _run(freshness_root, "--require-measurements")
        if freshness.returncode != 2 or "  - rms_norm/4096" not in freshness.stderr:
            failures.append(
                "head-freshness: deleted head benchmark was masked by stale "
                "same-path output"
            )

        # Reconciling missing_baseline_ids against baseline_ids instead of any
        # change/ presence means an unrelated named-baseline bench surviving
        # cleanup now correctly reports its own NO COVERAGE gap. Test that
        # survival-of-cleanup property in isolation so it does not also
        # contaminate the single-direction check above.
        freshness_unrelated_root = Path(td) / "freshness-unrelated" / "criterion"
        unrelated_bench = freshness_unrelated_root / "legacy_group" / "legacy_bench"
        _fabricate_bench(unrelated_bench, "previous-run")
        prepared_unrelated = _prepare(freshness_unrelated_root, "--prepare-head")
        if prepared_unrelated.returncode != 0:
            failures.append(
                "head-freshness: unrelated-baseline prepare step failed: "
                f"{prepared_unrelated.stderr}"
            )
        if not (unrelated_bench / "new" / "estimates.json").exists():
            failures.append(
                "head-freshness: unrelated named-baseline benchmark was cleaned"
            )

        unsafe_root = Path(td) / "not-the-criterion-root"
        _fabricate_bench(
            unsafe_root / "rms_norm" / "4096", "compare-base"
        )
        unsafe_prepare = _prepare(unsafe_root, "--prepare-head")
        if unsafe_prepare.returncode != 2:
            failures.append(
                "head-freshness: cleanup outside a criterion-named root was not refused"
            )
        if not (unsafe_root / "rms_norm" / "4096" / "change").exists():
            failures.append(
                "head-freshness: refused cleanup still removed an artifact"
            )

        symlink_root = Path(td) / "symlink-safety" / "criterion"
        symlink_bench = symlink_root / "rms_norm" / "4096"
        _fabricate_baseline(symlink_bench, "compare-base")
        outside = Path(td) / "outside-change"
        outside.mkdir()
        outside_sentinel = outside / "keep.txt"
        outside_sentinel.write_text("must survive")
        (symlink_bench / "change").symlink_to(outside, target_is_directory=True)
        symlink_prepare = _prepare(symlink_root, "--prepare-head")
        if symlink_prepare.returncode != 2:
            failures.append(
                "head-freshness: symlinked cleanup artifact was not refused"
            )
        if not outside_sentinel.exists():
            failures.append(
                "head-freshness: cleanup followed a symlink outside criterion root"
            )

        atomic_head_root = Path(td) / "atomic-head-cleanup" / "criterion"
        atomic_safe_head = atomic_head_root / "a_safe" / "1"
        _fabricate_bench(atomic_safe_head, "compare-base")
        atomic_unsafe_head = atomic_head_root / "z_unsafe" / "1"
        _fabricate_baseline(atomic_unsafe_head, "compare-base")
        (atomic_unsafe_head / "new").mkdir()
        (atomic_unsafe_head / "new" / "estimates.json").write_text("{}")
        (atomic_unsafe_head / "change").symlink_to(
            outside, target_is_directory=True
        )
        atomic_head_prepare = _prepare(atomic_head_root, "--prepare-head")
        if atomic_head_prepare.returncode != 2:
            failures.append(
                "head-freshness: unsafe later head artifact was not refused"
            )
        if not (atomic_safe_head / "new" / "estimates.json").exists():
            failures.append(
                "head-freshness: cleanup partially deleted a safe artifact "
                "before refusing a later unsafe path"
            )

        # #1107: stored Markdown must carry the machine/phase handoff and the
        # actual sample shape Criterion persisted for both A/B arms.
        provenance_path = Path(td) / "provenance.txt"
        provenance_fields = {
            "started_utc": "2026-07-29T12:00:00Z",
            "finished_utc": "2026-07-29T12:00:40Z",
            "host_id": "hostname-sha256:0123456789abcdef",
            "os": "fixture-os",
            "base_ref": "fixture-base",
            "base_sha": "a" * 40,
            "head_ref": "fixture-head",
            "head_sha": "b" * 40,
            "head_mode": "detached-worktree",
            "base_rustc": "rustc fixture-base",
            "head_rustc": "rustc fixture-head",
            "base_cargo": "cargo fixture-base",
            "head_cargo": "cargo fixture-head",
            "base_criterion": "0.5.1",
            "head_criterion": "0.5.1",
            "criterion_mode": "quick",
            "baseline_name": "compare-base",
            "targets": "lattice-inference:fixture",
            "inference_features": "<none>",
            "embed_features": "<none>",
            "filters": "inference='<all>' embed='<all>'",
            "enforcement": "fail-on-regression",
        }
        machine_states = [
            {
                "schema": MACHINE_STATE_SCHEMA,
                "label": "before first arm",
                "captured_at_utc": "2026-07-29T12:00:10Z",
                "power": {"status": "measured", "source": "pmset", "state": "ac"},
                "thermal": {
                    "status": "measured",
                    "source": "pmset",
                    "state": "nominal",
                    "cpu_speed_limit_percent": 100,
                },
                "idle": {
                    "status": "measured",
                    "source": "IOHIDSystem.HIDIdleTime",
                    "seconds": 30.0,
                },
            },
            {
                "schema": MACHINE_STATE_SCHEMA,
                "label": "between order strata",
                "captured_at_utc": "2026-07-29T12:00:20Z",
                "power": {"status": "unavailable", "reason": "fixture unsupported"},
                "thermal": {
                    "status": "unavailable",
                    "reason": "fixture unsupported",
                },
                "idle": {
                    "status": "unavailable",
                    "reason": "fixture unsupported",
                },
            },
            {
                "schema": MACHINE_STATE_SCHEMA,
                "label": "after final arm",
                "captured_at_utc": "2026-07-29T12:00:30Z",
                "power": {
                    "status": "measured",
                    "source": "pmset",
                    "state": "battery",
                },
                "thermal": {
                    "status": "measured",
                    "source": "pmset",
                    "state": "throttled",
                    "cpu_speed_limit_percent": 90,
                },
                "idle": {
                    "status": "measured",
                    "source": "IOHIDSystem.HIDIdleTime",
                    "seconds": 31.0,
                },
            },
        ]
        provenance_lines = [
            "schema=lattice-bench-provenance-v2",
            *[f"{field}={provenance_fields[field]}" for field in PROVENANCE_FIELDS],
            "lock=bench-window: acquired",
            "lock=Metal GPU: acquired",
            "ambient=[quiet] before first arm: idle 99.0% (floor 70.0%) ok | top: none",
            "ambient=[quiet] between order strata: idle 98.0% (floor 70.0%) ok | top: none",
            "ambient=[quiet] after final arm: idle 97.0% (floor 70.0%) ok | top: none",
            "phase_load=[phase-load] base1: samples=6 idle min/mean=95.0%/97.0% "
            "foreign max/mean=3.0%/1.0% self mean=8.0% (ceiling 30.0%) ok | "
            "top foreign at max: none 0.0% @ 2026-07-29T12:00:05Z",
            "phase_load=[phase-load] head1: samples=6 idle min/mean=94.0%/96.0% "
            "foreign max/mean=4.0%/2.0% self mean=9.0% (ceiling 30.0%) ok | "
            "top foreign at max: none 0.0% @ 2026-07-29T12:00:12Z",
            "phase_load=[phase-load] head2: samples=6 idle min/mean=93.0%/95.0% "
            "foreign max/mean=5.0%/2.0% self mean=9.0% (ceiling 30.0%) ok | "
            "top foreign at max: none 0.0% @ 2026-07-29T12:00:18Z",
            "phase_load=[phase-load] base2: samples=6 idle min/mean=92.0%/94.0% "
            "foreign max/mean=6.0%/3.0% self mean=8.0% (ceiling 30.0%) ok | "
            "top foreign at max: none 0.0% @ 2026-07-29T12:00:25Z",
            *[
                f"machine_state={json.dumps(state, separators=(',', ':'), sort_keys=True)}"
                for state in machine_states
            ],
        ]
        provenance_path.write_text("\n".join(provenance_lines) + "\n")

        provenance_root = Path(td) / "provenance" / "criterion"
        provenance_bench = provenance_root / "grp_provenance" / "bench_provenance"
        _fabricate_bench(provenance_bench, "compare-base")
        _fabricate_sample(provenance_bench / "new", count=2, mode="Flat")
        _fabricate_sample(provenance_bench / "compare-base", count=4, mode="Linear")
        provenance_run = _run(
            provenance_root,
            "--require-measurements",
            "--provenance-file",
            str(provenance_path),
            "--require-provenance",
        )
        if provenance_run.returncode != 0:
            failures.append(
                "run-provenance: complete provenance and sample metadata were "
                f"rejected: {provenance_run.stderr}"
            )
        for expected in (
            "<summary>Run provenance</summary>",
            "schema=lattice-bench-provenance-v2",
            "host_id=hostname-sha256:0123456789abcdef",
            "criterion_base_samples=4 Linear (1 benchmark)",
            "criterion_head_samples=2 Flat (1 benchmark)",
            "ambient=[quiet] after final arm",
            "phase_load=[phase-load] base1: samples=6",
            "phase_load=[phase-load] head1: samples=6",
            "phase_load=[phase-load] head2: samples=6",
            "phase_load=[phase-load] base2: samples=6",
            "machine_state[between order strata]=captured",
            "power unavailable (fixture unsupported)",
            "HID idle unavailable (fixture unsupported)",
            "gate not enforced on this platform",
            "| 4 Linear | 2 Flat |",
        ):
            if expected not in provenance_run.stdout:
                failures.append(
                    f"run-provenance: stored report omitted {expected!r}"
                )

        legacy_v1_path = Path(td) / "legacy-v1-provenance.txt"
        legacy_v1_path.write_text(
            "\n".join(
                "schema=lattice-bench-provenance-v1"
                if line.startswith("schema=")
                else line
                for line in provenance_lines
                if not line.startswith("embed_features=")
            )
            + "\n"
        )
        legacy_v1_run = _run(
            provenance_root,
            "--provenance-file",
            str(legacy_v1_path),
            "--require-provenance",
        )
        if (
            legacy_v1_run.returncode != 0
            or "schema=lattice-bench-provenance-v1" not in legacy_v1_run.stdout
            or "embed_features=<none>" not in legacy_v1_run.stdout
        ):
            failures.append(
                "run-provenance: legacy v1 was not accepted with the "
                "deterministic embed_features=<none> default"
            )
        else:
            print(
                "SCHEMA PROOF: legacy provenance v1 accepted and rendered "
                "with embed_features=<none>"
            )

        v1_with_v2_field_path = Path(td) / "v1-with-v2-field-provenance.txt"
        v1_with_v2_field_path.write_text(
            "\n".join(
                "schema=lattice-bench-provenance-v1"
                if line.startswith("schema=")
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        v1_with_v2_field_run = _run(
            provenance_root,
            "--provenance-file",
            str(v1_with_v2_field_path),
            "--require-provenance",
        )
        if (
            v1_with_v2_field_run.returncode != 2
            or "v1 does not allow provenance fields: embed_features"
            not in v1_with_v2_field_run.stderr
        ):
            failures.append(
                "run-provenance: v1 silently accepted the v2 embed_features field"
            )
        else:
            print(
                "SCHEMA PROOF: provenance v1 rejects the v2-only "
                "embed_features field"
            )

        incomplete_v2_path = Path(td) / "incomplete-v2-provenance.txt"
        incomplete_v2_path.write_text(
            "\n".join(
                line
                for line in provenance_lines
                if not line.startswith("embed_features=")
            )
            + "\n"
        )
        incomplete_v2_run = _run(
            provenance_root,
            "--provenance-file",
            str(incomplete_v2_path),
            "--require-provenance",
        )
        if (
            incomplete_v2_run.returncode != 2
            or "missing provenance fields: embed_features"
            not in incomplete_v2_run.stderr
        ):
            failures.append(
                "run-provenance: v2 accepted a missing embed_features field"
            )
        else:
            print(
                "SCHEMA PROOF: provenance v2 requires embed_features"
            )

        darwin_ungated = Path(td) / "darwin-ungated-provenance.txt"
        darwin_ungated.write_text(
            "\n".join(
                "os=Darwin 24.6.0 arm64"
                if line.startswith("os=")
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        darwin_ungated_run = _run(
            provenance_root,
            "--provenance-file",
            str(darwin_ungated),
            "--require-provenance",
        )
        if (
            darwin_ungated_run.returncode != 2
            or "lacks a passed fail-closed gate" not in darwin_ungated_run.stderr
        ):
            failures.append(
                "run-provenance: macOS handoff without machine-state gates "
                "did not fail closed"
            )

        legacy_report = _run(provenance_root)
        if legacy_report.returncode != 0:
            failures.append(
                "run-provenance: existing report-only CLI without provenance changed"
            )
        if "unsuitable as benchmark evidence" not in legacy_report.stdout:
            failures.append(
                "run-provenance: absent handoff was not disclosed in the report"
            )

        if _run(provenance_root, "--require-provenance").returncode != 2:
            failures.append(
                "run-provenance: --require-provenance accepted no provenance file"
            )

        incomplete_provenance = Path(td) / "incomplete-provenance.txt"
        incomplete_provenance.write_text(
            "\n".join(
                line
                for line in provenance_lines
                if not line.startswith("host_id=")
            )
            + "\n"
        )
        incomplete_report = _run(
            provenance_root,
            "--provenance-file",
            str(incomplete_provenance),
        )
        if (
            incomplete_report.returncode != 0
            or "unsuitable as benchmark evidence" not in incomplete_report.stdout
        ):
            failures.append(
                "run-provenance: incomplete report-only handoff did not render unsuitable"
            )
        else:
            print(
                "MUTATION PROOF: missing host_id report-only exit=0; "
                "unsuitable marker rendered"
            )
        incomplete_enforcing = _run(
            provenance_root,
            "--provenance-file",
            str(incomplete_provenance),
            "--require-provenance",
        )
        if (
            incomplete_enforcing.returncode != 2
            or "missing provenance fields: host_id" not in incomplete_enforcing.stderr
        ):
            failures.append(
                "run-provenance: incomplete enforcing handoff did not fail closed"
            )
        else:
            print(
                "MUTATION PROOF: missing host_id enforcing exit=2; "
                "missing provenance fields: host_id"
            )

        reversed_provenance = Path(td) / "reversed-provenance.txt"
        reversed_provenance.write_text(
            "\n".join(
                "finished_utc=2026-07-29T11:59:59Z"
                if line.startswith("finished_utc=")
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        reversed_run = _run(
            provenance_root,
            "--provenance-file",
            str(reversed_provenance),
            "--require-provenance",
        )
        if reversed_run.returncode != 2 or "precedes started_utc" not in reversed_run.stderr:
            failures.append(
                "run-provenance: reversed wall-clock interval did not fail closed"
            )

        out_of_order_state = dict(machine_states[1])
        out_of_order_state["captured_at_utc"] = "2026-07-29T12:00:05Z"
        out_of_order_provenance = Path(td) / "out-of-order-provenance.txt"
        out_of_order_provenance.write_text(
            "\n".join(
                f"machine_state={json.dumps(out_of_order_state, separators=(',', ':'), sort_keys=True)}"
                if '"label":"between order strata"' in line
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        out_of_order_run = _run(
            provenance_root,
            "--provenance-file",
            str(out_of_order_provenance),
            "--require-provenance",
        )
        if (
            out_of_order_run.returncode != 2
            or "out of phase order" not in out_of_order_run.stderr
        ):
            failures.append(
                "run-provenance: out-of-order phase timestamps did not fail closed"
            )

        unknown_state = dict(machine_states[0])
        unknown_state["power"] = {
            "status": "measured",
            "source": "pmset",
            "state": "wall-outlet",
        }
        invalid_state_provenance = Path(td) / "invalid-state-provenance.txt"
        invalid_state_provenance.write_text(
            "\n".join(
                f"machine_state={json.dumps(unknown_state, separators=(',', ':'), sort_keys=True)}"
                if '"label":"before first arm"' in line
                else line
                for line in provenance_lines
            )
            + "\n"
        )
        invalid_state_run = _run(
            provenance_root,
            "--provenance-file",
            str(invalid_state_provenance),
            "--require-provenance",
        )
        if invalid_state_run.returncode != 2 or "ac, battery" not in invalid_state_run.stderr:
            failures.append(
                "run-provenance: unknown measured power state did not fail closed"
            )
        else:
            print(
                "MUTATION PROOF: malformed measured power capability exit=2; "
                "accepted states remain ac,battery"
            )

        missing_samples = _run(
            ok_root,
            "--provenance-file",
            str(provenance_path),
            "--require-provenance",
        )
        if missing_samples.returncode != 2 or "sample metadata" not in missing_samples.stderr:
            failures.append(
                "run-provenance: missing Criterion sample.json metadata was certified"
            )

        invalid_sample_path = provenance_bench / "new" / "sample.json"
        invalid_sample_path.write_text(
            json.dumps(
                {
                    "sampling_mode": "Flat",
                    "iters": [1.0, 2.0],
                    "times": [1.0, 0.0],
                }
            )
        )
        invalid_sample = _run(
            provenance_root,
            "--provenance-file",
            str(provenance_path),
            "--require-provenance",
        )
        if invalid_sample.returncode != 2 or "sample metadata" not in invalid_sample.stderr:
            failures.append(
                "run-provenance: non-positive Criterion sample was certified"
            )

    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    if failures:
        print(f"SELFTEST: FAIL ({len(failures)} failure(s))")
        return 1
    print("SELFTEST: PASS — base/, compare-base/, orphan-warn, named-wins-over-stale-base, "
          "multi-sibling-refusal, target-qualified manifest handoff (CRLF/whitespace/C-sort, "
          "slash-bearing group, demoted target informational, non-demoted target gated), "
          "require-measurements (empty root, gating pass, explicit informational target, "
          "mismatch refusal, partial-run refusal), ABBA directional-drift correction, "
          "malformed reverse-number refusal, selected-baseline completeness, provenance "
          "v1/v2 compatibility, and stale-head freshness all correct")
    return 0
