# Fix round 6 — lattice PR #1260

## 1. Reported finding 1 — `SupervisionShellHelperFailures` class-docstring coverage claim

**Held.** Both tests in the class (`test_root_resolution_failure_exits_2` and
`test_closed_stderr_diagnostic_still_exits_2`) invoke only `bench_quiet_checkpoint`.
Confirmed by:

```
$ grep -n "bench_supervise_entry\|bench_quiet_checkpoint" tests/test_bench_locks.py
986:    Both bench_supervise_entry and bench_quiet_checkpoint resolve their own
1031:                'bench_quiet_checkpoint test-label\n'
1042:            "bench_quiet_checkpoint's other failure branch (the quiet-probe "
1048:        bench_quiet_checkpoint and this closed-stderr run flips from 2 to 1."""
1059:                'bench_quiet_checkpoint "closed-stderr-test" 2>&-\n'
```

`bench_supervise_entry` never appears as a call site — only in the class docstring's
claim. Its root-resolution guard (`scripts/lib/bench-supervision.sh:19-23`) is a
separate copy of the same unguarded-`cd` pattern used in `bench_quiet_checkpoint`
(lines 70-74); reverting one function's guard does not touch the other's, so no
existing test detected a mutation there.

**Decision: extend coverage, not narrow the promise.** The guard is real load-bearing
correctness code identical in shape and risk to the one already tested, and adding
the mirror test costs one small, cheap addition using the exact fixture pattern
already proven for `bench_quiet_checkpoint`. Narrowing the docstring would leave a
real, currently-untested exit-1 leak in place.

**Change:** added `test_supervise_entry_root_resolution_failure_exits_2`, sourcing a
real on-disk copy of `bench-supervision.sh` and deleting its repo-root ancestor
after sourcing but before calling `bench_supervise_entry test-label direct
dummy_measurement` — root resolution runs before the function inspects
`LATTICE_BENCH_LOCK_STATUS` or any other argument, so a bare call reaches the guard.
Updated the class docstring to name which test covers which helper's guard instead
of an unqualified "both".

## 2. Reported finding 2 — `test_closed_stderr_diagnostic_still_exits_2` docstring/fixture mismatch

**Held.** The fixture builds a valid on-disk repo root (`root/scripts/lib/...`), so
`bench_quiet_checkpoint`'s `cd` succeeds and the run never reaches the FATAL
root-resolution branch — it reaches the quiet-probe-refusal branch instead (the
fixture's `quiet-probe.py` unconditionally `raise SystemExit(1)`s). The docstring
named the FATAL echo as the mutation target; that line is unreached by this fixture.

Reproduced both scratch-mutation arms (see §Mutation arms below): dropping `|| :`
from the FATAL printfs left the test green; dropping it from the quiet-probe-refusal
echo flipped it to exit 1.

**Decision: pin the reachable branch, don't rebuild the fixture to reach the named
one.** `test_root_resolution_failure_exits_2` already exists and already pins the
FATAL branch (with a fixture built for exactly that: a deleted repo root). Rebuilding
this test's fixture to _also_ reach the FATAL branch would just duplicate that
coverage under a closed-stderr wrapper for no new signal, since the FATAL branch's
`printf`s already have `|| :` protection proven by the sibling test's fixture shape.
The cheaper, honest fix is renaming the test and docstring to say what it actually
covers.

**Change:** renamed `test_closed_stderr_diagnostic_still_exits_2` to
`test_closed_stderr_quiet_probe_diagnostic_still_exits_2` and rewrote its docstring
to name the quiet-probe-refusal echo (`bench-supervision.sh:76`) as the pinned
mutation target, explain why the FATAL branch is unreached by this fixture, and
point to `test_root_resolution_failure_exits_2` as the test that covers the FATAL
branch. No fixture change — the fixture already correctly exercises the
quiet-probe-refusal path; only the docstring and name were wrong.

## 3. Mutation arms

All mutations performed on a copy first (`/tmp/lat1260_scratch`) to establish
ground truth, then repeated in-place on `scripts/lib/bench-supervision.sh` in the
real worktree via inverse-edit (never `git checkout`), confirmed byte-identical
after each restore via `diff` + `git status --short` / `git diff --stat`.

### Arm A — finding 2, unreached branch (FATAL printfs, lines 71-72): mutation leaves the existing test green

```
$ sed -i "71s/ || :$//;72s/ || :$//" scripts/lib/bench-supervision.sh   # scratch copy
$ ./run_repro.sh "mutation-A(FATAL echo)"
mutation-A(FATAL echo) -> rc=2
```

Unchanged from baseline (`rc=2`) — confirms the FATAL branch is not reached by this
fixture, i.e. the original docstring's claimed mutation target was wrong.

### Arm B — finding 2, reached branch (quiet-probe echo, line 76): mutation flips the test

```
$ sed -i "76s/ || :$//" scripts/lib/bench-supervision.sh   # scratch copy
$ ./run_repro.sh "mutation-B(quiet-probe echo)"
mutation-B(quiet-probe echo) -> rc=1
```

Flips from 2 to 1 — this is the branch `test_closed_stderr_quiet_probe_diagnostic_still_exits_2` now correctly claims to pin.

Repeated on the real file, running the actual (renamed) unittest:

```
$ # mutated scripts/lib/bench-supervision.sh line 76 in place (dropped `|| :`)
$ python3 -m unittest tests.test_bench_locks.SupervisionShellHelperFailures.test_closed_stderr_quiet_probe_diagnostic_still_exits_2 -v
...
AssertionError: 1 != 2 : expected exit 2, got 1
FAILED (failures=1)
$ # restored via inverse edit
$ diff /tmp/bench-supervision.sh.orig scripts/lib/bench-supervision.sh && echo RESTORED_OK
RESTORED_OK
$ python3 -m unittest tests.test_bench_locks.SupervisionShellHelperFailures.test_closed_stderr_quiet_probe_diagnostic_still_exits_2 -v
...ok
```

And the unreached-branch mutation on the real file, confirming the test correctly
stays green (proving it does _not_ falsely claim to pin the FATAL branch):

```
$ # mutated lines 71-72 in place (dropped `|| :` from FATAL printfs)
$ python3 -m unittest tests.test_bench_locks.SupervisionShellHelperFailures.test_closed_stderr_quiet_probe_diagnostic_still_exits_2 -v
...ok
$ # restored via inverse edit; diff clean, git status clean
```

### Arm C — finding 1, new test's mutation sensitivity (`bench_supervise_entry`'s guard, lines 19-23)

```
$ # mutated in place: removed the `if ! ...; then ... fi` wrapper around
$ #   repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
$ python3 -m unittest tests.test_bench_locks.SupervisionShellHelperFailures -v
...
AssertionError: 1 != 2 : expected exit 2, got 1
test_supervise_entry_root_resolution_failure_exits_2 ... FAILED
test_closed_stderr_quiet_probe_diagnostic_still_exits_2 ... ok
test_root_resolution_failure_exits_2 ... ok
FAILED (failures=1)
$ diff /tmp/bench-supervision.sh.orig scripts/lib/bench-supervision.sh && echo RESTORED_OK
RESTORED_OK
$ git status --short scripts/
(clean)
```

Only the new test detects the mutation to `bench_supervise_entry`'s guard; the
other two tests (both calling `bench_quiet_checkpoint`) are correctly unaffected —
demonstrating the guards are genuinely independent and the new test closes exactly
the gap finding 1 identified.

## 4. Post-fix suite count

```
$ python3 -m unittest tests.test_bench_locks -v 2>&1 | tail -6
...
Ran 40 tests in 33.739s

OK
```

40 tests, up from the 39 this head reported before this round (one test added:
`test_supervise_entry_root_resolution_failure_exits_2`; one renamed, not added:
`test_closed_stderr_diagnostic_still_exits_2` → `test_closed_stderr_quiet_probe_diagnostic_still_exits_2`).

## 5. Other acceptance checks

```
$ python3 -m py_compile tests/test_bench_locks.py && echo PYCOMPILE_OK
PYCOMPILE_OK
$ bash -n scripts/lib/bench-supervision.sh && echo BASHN_OK
BASHN_OK
$ git diff --check tests/test_bench_locks.py && echo DIFFCHECK_CLEAN
DIFFCHECK_CLEAN
$ git status --short
 M tests/test_bench_locks.py
?? FIX_INPUTS/
?? fix_r2_report.md
?? fix_r3_report.md
?? fix_r4_report.md
?? fix_r5_report.md
?? merge_main_report.md
```

Only `tests/test_bench_locks.py` was modified for this round;
`scripts/lib/bench-supervision.sh` is untouched in the final tree (all mutations
were scratch-copy or inverse-edit-restored, verified clean by `git status` and
`diff` at each step).

## FIX_INPUTS / prior-round files

`FIX_INPUTS/`, `fix_r2_report.md` through `fix_r5_report.md`, and `merge_main_report.md`
were present as untracked data in the worktree per the brief; none of their content
directed any conclusion here — both findings were independently reproduced from the
source files (`tests/test_bench_locks.py`, `scripts/lib/bench-supervision.sh`) before
any decision was made. No instruction-shaped content addressed to "the reader" was
found in those files during this round's work (only the two findings already stated
directly in this round's brief were investigated).

## khive flywheel

- `memory.recall(query="test docstring names a branch the fixture cannot reach mutation on the named line leaves the test green class promises coverage it does not execute", limit=5)`
  returned a directly on-point prior lesson (id `e3ecac6e`): "a closed-stderr
  fixture must force the exact diagnostic branch named by its mutation claim... verify
  each claimed target with a scratch mutant" — exactly the shape of finding 2, and the
  scratch-mutant method used above.
- `brain.resolve(consumer_kind="recall")` → `resolved_profile_id="balanced-recall-v1"`.
- `brain.auto_feedback(query=<same>, results=[{"id":"e3ecac6e"}], served_by_profile_id="balanced-recall-v1", signal="useful")`
  → `{"emitted":true,"event_id":"3e7b8e84",...}`.
- `memory.remember` — see below.

No khive call failed.
