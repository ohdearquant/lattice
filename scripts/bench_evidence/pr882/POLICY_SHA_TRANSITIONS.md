# Policy-sha transitions affecting the reports in this directory

`report_ctx512.json` and `report_ctx1024.json` pin
`provenance.policy_sha = 17c4f9ef52c647aeb32642c065af135607d03ccc329d6245482ff899abf1de3c`.

`bench_gate_math.policy_sha` hashes the raw bytes of `scripts/perf-policy.toml`, so
any edit to that file moves the sha, including an edit that changes no gating
value. When `validate_run_record` is given `current_policy_sha`, a record whose
pinned sha no longer matches is rejected as a post-run threshold change. That is
the correct default: the check exists so a gate result cannot be re-interpreted
under thresholds that moved after the run.

It cannot, however, tell a threshold change apart from a prose correction, and
this file records the one transition where the difference matters.

| From                | To                  | Semantic delta                        |
| ------------------- | ------------------- | ------------------------------------- |
| `17c4f9ef52c647ae…` | `9d5a3a3776e50610…` | none — three `note` strings rewritten |

## Why that row is a claim you can check rather than one you have to trust

Parse both versions, drop the prose keys (`note`, `other_rule`), and compare
what remains. Everything the gate reads is identical:

```
keys whose VALUE changed: ['.cv_bands[0].note', '.cv_bands[1].note', '.cv_bands[2].note']
gating values identical (notes/prose excluded): True
control (tampered max_cv detected): True
```

The control line matters as much as the result: the comparator was re-run
against a copy with `cv_bands[0].max_cv` set to `0.999` and reported a
difference, so "identical" here is a measurement rather than an instrument that
returns True for everything.

Band values before and after, read back through `parse_cv_bands`:
`[(0.015, 7, 1.0), (0.05, 25, 1.0), (1.0, 25, 2.0)]` in both.

## What the edit corrected

The band notes described each band by the CV the power curves were calibrated
at (`~1%`, `~3%`, `>=8%`) while the lookup tests the registered `max_cv`
(`0.015`, `0.05`, `1.0`). A reader with a measured 2% CV would predict the n=7
band from the notes; the lookup returns n=25. A reader with 6% would predict the
1.0 fail-margin multiplier; the lookup returns 2.0. The notes now state the
boundary first and label the calibration point as such.

## The open design question this exposed

Correcting a misleading comment in the policy file costs exactly what changing a
kill-point costs: every recorded run record pinning the old sha stops
revalidating. That prices honesty at the same rate as a threshold move, which is
part of why the wrong notes survived as long as they did.

A sha computed over the canonicalized _parsed_ values, with prose keys excluded,
would still fail closed on any band, threshold, or noise-class change while
staying quiet for a comment fix. That is a change to what a gate identity means,
so it belongs in its own change with its own review rather than riding along
with a documentation correction. Filed rather than fixed here.
