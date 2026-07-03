# Implementation Report — Issue #525

## Summary

Issue #525 asked for two fixes to the `ttft_dispatch` row's absolute-delta
sub-predicate in `scripts/adr064-gpu-decode-gate.py` /
`tests/test_adr064_gpu_decode_gate.py`:

1. Add real test coverage for the absolute-delta branch (the original fixture
   tripped the *relative* branch instead of the absolute one it claimed to test).
2. Resolve the `>` vs `>=` boundary ambiguity against issue #167's "+10" wording.

**Finding: both fixes are already on `main`.** They were merged in PR #555
("test(gate): cover ADR-064 absolute-delta sub-predicate + fix boundary
semantics (#525)"), merge commit `47ca845bd3bcc8153f38e421928d11878ef16e42`,
merged 2026-07-02T16:13:18Z from branch `test/525-gate-subpredicates`. This
worktree's branch (`test/525-adr064-abs-delta`) was created from `origin/main`
*after* that merge, so it already contains the fix — `git log --oneline` on
both target files shows:

```
47ca845bd3 test(gate): cover ADR-064 absolute-delta sub-predicate + fix boundary semantics (#525) (#555)
2dc0c6d3b7 feat(bench): ADR-064 decode perf-gate harness + dormant nightly gate (#167) (#523)
```

Issue #525 is still open on GitHub only because the merged PR's commit
message used "Refs #525" rather than a closing keyword (`Fixes #525` /
`Closes #525`), so it never auto-closed — the underlying work is done.

No further code change is made in this branch: re-deriving the same fix would
be redundant, and changing the already-merged behavior back toward the
pre-#555 state would reinstate the exact bug this issue reports. This report
documents fresh, independent verification of that already-merged fix,
performed in this worktree this session.

## What's already in place (verified fresh this session)

### 1. Absolute-delta predicate now has dedicated, correctly-isolated coverage

`TtftDispatchRowRegressionTest` in `tests/test_adr064_gpu_decode_gate.py` has
three fixtures, each holding the baseline at
`dispatches_per_token = (250.0, 249.0, 250.0)` so the *relative* lower bound
stays under the 5% threshold (~4%) regardless of the absolute delta, isolating
the absolute branch:

- `test_dispatches_absolute_delta_below_threshold_passes_row` — delta 9.9,
  expects PASS.
- `test_dispatches_absolute_delta_above_threshold_trips_abs_only` — delta
  10.1, expects FAIL with an `abs_delta` reason, and explicitly asserts no
  `lb=` (relative) reason is present.
- `test_dispatches_absolute_delta_boundary_trips_fail_closed` — delta exactly
  10.0, expects FAIL (fail-closed at the boundary), asserting the reason
  string `abs_delta=10.0000 >= 10` and again no `lb=` reason.

The single fixture named in the issue (whose comment incorrectly claimed the
absolute branch fired when the relative branch actually did) no longer
exists — it was replaced by these three correctly-isolated tests.

### 2. `>` vs `>=` boundary

`scripts/adr064-gpu-decode-gate.py` (line 132) reads:

```python
disp_abs = disp_c["ci95_low"] - disp_b["ci95_high"]
if disp_abs >= 10:
    reasons.append(f"decode/dispatches_per_token abs_delta={disp_abs:.4f} >= 10")
```

**Decision already made and merged: `>=` (fail-closed).** Justification
recorded in PR #555's description: issue #167's gate table lists
"dispatches/token >5% or +10" without stating whether an exact +10 counts as
a breach — genuinely ambiguous wording, unlike the sibling thresholds in the
same table which all spell out an explicit comparator (`>7%`, `>10%`,
`≤2`, etc.). PR #555 resolved that ambiguity by applying the fail-closed
principle this gate already uses elsewhere (a row should not silently pass a
boundary case the spec doesn't clearly permit).

I independently read `gh issue view 167` this session and reached the same
reading: the "+10" bullet is the one item in the gate table with no explicit
comparator glyph, so the text alone cannot settle `>` vs `>=`. Since a
concrete, reasoned ruling is already shipped on `main`, I did not revert or
second-guess it — doing so would itself be "changing semantics on a guess,"
which is exactly what the original issue and the resolving PR both caution
against.

## Mutation-sensitivity evidence (performed fresh this session)

Working tree confirmed clean at `HEAD` before and after each mutation
(`git status --short` empty, `git diff HEAD` empty). Two independent
temporary mutations were applied to `scripts/adr064-gpu-decode-gate.py`,
tested, then reverted with `git checkout --` (safe here because `HEAD`
already holds the correct, desired state):

**Mutation 1 — delete the absolute sub-predicate entirely**
(the `if disp_abs >= 10: reasons.append(...)` lines removed):

```
FAILED tests/test_adr064_gpu_decode_gate.py::TtftDispatchRowRegressionTest::test_dispatches_absolute_delta_above_threshold_trips_abs_only
FAILED tests/test_adr064_gpu_decode_gate.py::TtftDispatchRowRegressionTest::test_dispatches_absolute_delta_boundary_trips_fail_closed
2 failed, 1 passed, 17 deselected
```

(The below-threshold PASS-expecting test is unaffected, as expected — it
never depended on the predicate firing.)

**Mutation 2 — revert only the comparator, `>=` → `>`**
(keeps the predicate, targets exactly the #167 boundary question):

```
FAILED tests/test_adr064_gpu_decode_gate.py::TtftDispatchRowRegressionTest::test_dispatches_absolute_delta_boundary_trips_fail_closed
1 failed, 2 passed, 17 deselected
```

Only the boundary test (exact +10.0) fails; the above-threshold test (10.1)
and below-threshold test (9.9) both still pass, since neither depends on the
exact-boundary comparator. This confirms the boundary test is the specific,
non-redundant guard against a `>=` → `>` regression — the other two fixtures
alone would not catch it.

After each mutation, `git checkout -- scripts/adr064-gpu-decode-gate.py`
restored the file; `git status --short` and `git diff HEAD` were confirmed
empty afterward, and the full suite was re-run green both times.

## Verification (this session)

```
$ uv run pytest tests/test_adr064_gpu_decode_gate.py -q
20 passed, 1 warning in 0.07s

$ uv run ruff check scripts/adr064-gpu-decode-gate.py tests/test_adr064_gpu_decode_gate.py
All checks passed!

$ uv run ruff format --check scripts/adr064-gpu-decode-gate.py tests/test_adr064_gpu_decode_gate.py
2 files already formatted
```

## Recommendation

Close issue #525, referencing PR #555 / commit
`47ca845bd3bcc8153f38e421928d11878ef16e42` as the fix. This report is the
only change on this branch — there is no code diff against `main`.
