# Canonical policy identity migration

`report_ctx512.json` and `report_ctx1024.json` pin
`provenance.policy_sha = 17c4f9ef52c647aeb32642c065af135607d03ccc329d6245482ff899abf1de3c`.
That untagged 64-character value is a legacy raw-byte SHA. The reports remain
unchanged because rewriting historical provenance to a newer identity scheme
would make a claim about how they were produced that is not true.

Before `canonical-v1`, `bench_gate_math.policy_sha` hashed the raw bytes of
`scripts/perf-policy.toml`, so any edit moved the SHA, including an edit that
changed no gating value. The canonical scheme hashes deterministic serialized
TOML values, excluding only comments (already absent after parsing) and
`cv_bands[*].note`. Every other parsed value remains identity-bearing, including
`other_rule`.

The scheme tag is part of the identity. `validate_run_record` recognizes legacy
byte SHAs but never aliases one to `canonical-v1`, even when the historical
transition evidence records that a particular transition had no gating delta.
A same-scheme mismatch still fails closed; a cross-scheme comparison fails with
an explicit migration error.

| From                        | To                          | Gating delta                        |
| --------------------------- | --------------------------- | ----------------------------------- |
| legacy-bytes `9d5a3a3776e…` | `canonical-v1:84095f1f4e7…` | none — identity algorithm migration |

## Identity boundary

The canonical identity implementation excludes only `cv_bands[*].note`.
`other_rule` remains hashed because its rule-like text could gain validator
semantics. Tests mutate `max_cv`, required sample count, fail-margin multiplier,
threshold, noise class, policy version, and `other_rule`; every mutation changes
the identity. Separate prose-only tests rewrite both a TOML comment and a band
`note` and require identity stability.

## Migration and versioning

There is deliberately no transition allowlist in the validator. These reports
can be audited against their recorded legacy SHA and the semantic-diff evidence
in `POLICY_SHA_TRANSITIONS.md`, but validating them against the current canonical
identity requires a rerun that emits new provenance. A documented transition is
evidence, not an implicit equivalence rule.

`policy_version` remains 1 because neither the prose correction nor the identity
algorithm changes a gating value. Comments and `cv_bands[*].note` do not require
a version bump. Any threshold, band, noise-class, `other_rule`, or other parsed
policy-value change still changes the canonical identity; gating-policy changes
also require the version/rationale process described in `perf-policy.toml`.

Full identities:

- legacy after the note correction:
  `9d5a3a3776e50610f2fc2bf6156236abe874ea1a4cb4a1b548de43f6c0e27f0b`
- canonical identity for the unchanged gating values:
  `canonical-v1:84095f1f4e7aac7d331b87260ccb9460429527bb88b54dc4e1a6e057e30a98be`
