use super::*;

#[test]
fn effective_scale_zero_rank_is_zero() {
    assert_eq!(effective_scale(0, 4.0), 0.0);
}

#[test]
fn effective_scale_matches_alpha_over_rank() {
    assert_eq!(effective_scale(2, 4.0), 2.0);
}

#[test]
fn effective_scale_non_finite_alpha_collapses_to_zero() {
    for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(effective_scale(8, alpha), 0.0);
    }
}

#[test]
fn validate_alpha_finite_rejects_non_finite() {
    for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = validate_alpha_finite(8, alpha).unwrap_err();
        assert!(err.contains("alpha must be finite"));
    }
}

/// Direct regression for the `rank == 0` short-circuit: without it,
/// `alpha / rank as f32` divides by zero and produces `+inf` for a
/// positive finite `alpha`, which the very next check (`!scale.is_finite()`)
/// would then reject — wrongly failing a legitimate zero-rank adapter
/// (an empty factorization contributing nothing) that this branch exists
/// to accept.
#[test]
fn validate_alpha_finite_accepts_zero_rank_with_finite_alpha() {
    assert!(validate_alpha_finite(0, 1.0).is_ok());
    assert_eq!(effective_scale(0, 1.0), 0.0);
}

/// Direct regression for the effective-scale finite guard itself
/// (distinct from the `rank == 0` branch above): a typical finite
/// `(rank, alpha)` pair must validate. Given `alpha` is already checked
/// finite and `rank == 0` is short-circuited above this line, `alpha /
/// rank as f32` for a nonzero `rank` cannot itself become non-finite —
/// so this guard has no reachable "must fail" input today, and this test
/// instead pins the guard's "must pass" side: inverting the
/// `!scale.is_finite()` condition would make this fail directly, at the
/// fann crate level, instead of only through the many downstream
/// `lattice-tune` callers that route through it.
#[test]
fn validate_alpha_finite_accepts_typical_finite_scale() {
    assert!(validate_alpha_finite(8, 16.0).is_ok());
}

#[test]
fn descriptor_validate_delegates_to_free_function() {
    let d = LoraDescriptor {
        rank: 8,
        alpha: f32::NAN,
        target_modules: vec![],
        dtype: "f32".into(),
    };
    assert!(d.validate().is_err());
    assert_eq!(d.scale(), 0.0);
}

#[test]
fn check_finite_weight_rejects_nan() {
    let err = check_finite_weight("ctx", 3, f32::NAN).unwrap_err();
    assert!(err.contains("weight at index 3 is not finite"));
}

#[test]
fn check_rank_total_cap_rejects_over_budget() {
    let err = check_rank_total_cap(MAX_BLEND_RANK_TOTAL + 1, "ctx").unwrap_err();
    assert!(err.contains("exceeds MAX_BLEND_RANK_TOTAL"));
    assert!(check_rank_total_cap(MAX_BLEND_RANK_TOTAL, "ctx").is_ok());
}

#[test]
fn check_aggregate_elements_cap_rejects_over_budget() {
    let err = check_aggregate_elements_cap(MAX_BLEND_TOTAL_ELEMENTS + 1, "ctx").unwrap_err();
    assert!(err.contains("exceeds MAX_BLEND_TOTAL_ELEMENTS") || err.contains("aggregate"));
    assert!(
        err.contains("4 GiB"),
        "error must report the 4 GiB budget; got: {err}"
    );
    assert!(check_aggregate_elements_cap(MAX_BLEND_TOTAL_ELEMENTS, "ctx").is_ok());
}

/// `MAX_BLEND_TOTAL_ELEMENTS * 4` is a compile-time-constant expression:
/// on a 32-bit `usize` target (`wasm32-unknown-unknown`), evaluating it
/// in `usize` overflows u32::MAX and fails the build under
/// `#[deny(arithmetic_overflow)]`, regardless of `planned_elems` at
/// runtime — `cargo test` on this (64-bit) host cannot reproduce that,
/// so this pins the u64-widened math's result directly as a same-crate
/// regression guard; the 32-bit build itself was verified separately
/// with `rustc --target wasm32-unknown-unknown`.
#[test]
fn max_blend_total_elements_times_four_survives_u64_widening() {
    let widened = (MAX_BLEND_TOTAL_ELEMENTS as u64).checked_mul(4);
    assert!(widened.is_some(), "widened GiB math must not overflow u64");
    assert_eq!(widened.unwrap() / (1024 * 1024 * 1024), 4);
}

#[test]
fn check_buffer_lengths_rejects_short_a() {
    let err = check_buffer_lengths("ctx", 0, 2, 4, 4, 7, 8).unwrap_err();
    assert!(err.contains("A slice length"));
}

#[test]
fn check_buffer_lengths_rejects_short_b() {
    let err = check_buffer_lengths("ctx", 0, 2, 4, 4, 8, 7).unwrap_err();
    assert!(err.contains("B slice length"));
}

/// Direct regression for the `rank.checked_mul(d_in)` overflow guard:
/// replacing it with `unwrap_or(0)` would silently treat an overflowing
/// `rank*d_in` as `0`, so any `a_len` would then satisfy `a_len ==
/// expected_a` only when `a_len == 0` — a real overflow would either
/// false-reject a correctly-sized (impossibly large) buffer or, worse,
/// false-accept an empty one. Neither `checked_group_elements`'s own
/// overflow tests above nor any existing `check_buffer_lengths` test
/// drives `rank*d_in` past `usize::MAX`.
#[test]
fn check_buffer_lengths_rejects_rank_times_d_in_overflow() {
    let err = check_buffer_lengths("ctx", 0, usize::MAX, 2, 1, 0, 0).unwrap_err();
    assert!(
        err.contains("rank*d_in overflowed usize"),
        "expected rank*d_in overflow message; got: {err}"
    );
}

/// Mirror of the above for `d_out.checked_mul(rank)`. Both `expected_a`
/// and `expected_b` are computed before either length is checked, so
/// `rank*d_in` (`2*1=2`) must itself stay within bounds for this to
/// reach and isolate the `d_out*rank` guard rather than failing on the
/// earlier `rank*d_in` guard instead.
#[test]
fn check_buffer_lengths_rejects_d_out_times_rank_overflow() {
    let err = check_buffer_lengths("ctx", 0, 2, 1, usize::MAX, 0, 0).unwrap_err();
    assert!(
        err.contains("d_out*rank overflowed usize"),
        "expected d_out*rank overflow message; got: {err}"
    );
}

#[test]
fn check_dims_match_rejects_mismatch() {
    let err = check_dims_match("ctx", 0, "q_proj", 4, 4, 1, 4, 8).unwrap_err();
    assert!(err.contains("mismatched dimensions"));
}

#[test]
fn accumulate_rank_overflow_errors() {
    let err = accumulate_rank(usize::MAX, 1, "ctx").unwrap_err();
    assert!(err.contains("overflowed usize"));
}

#[test]
fn checked_group_elements_rejects_dims_add_overflow() {
    let err = checked_group_elements("ctx", 0, "q_proj", 1, usize::MAX, 1).unwrap_err();
    assert!(
        err.contains("d_in+d_out overflowed"),
        "expected d_in+d_out overflow message; got: {err}"
    );
}

#[test]
fn checked_group_elements_rejects_rank_dims_mul_overflow() {
    // dims = d_in + d_out = 2 (no overflow); rank_total * dims = usize::MAX * 2
    // overflows the multiplication.
    let err = checked_group_elements("ctx", 0, "q_proj", usize::MAX, 1, 1).unwrap_err();
    assert!(
        err.contains("rank_total*(d_in+d_out) overflowed"),
        "expected rank_total*(d_in+d_out) overflow message; got: {err}"
    );
}

#[test]
fn accumulate_planned_elements_rejects_overflow() {
    let err = accumulate_planned_elements(usize::MAX, 1, "ctx").unwrap_err();
    assert!(
        err.contains("aggregate blend element count overflowed"),
        "expected aggregate overflow message; got: {err}"
    );
}

#[test]
fn validate_target_modules_accepts_known_names() {
    let modules = vec!["q_proj".to_string(), "up_proj".to_string()];
    assert!(validate_target_modules(&modules, KNOWN_LORA_TARGET_MODULES).is_ok());
}

#[test]
fn validate_target_modules_rejects_unknown_name() {
    let modules = vec!["q_proj".to_string(), "not_a_real_module".to_string()];
    let err = validate_target_modules(&modules, KNOWN_LORA_TARGET_MODULES).unwrap_err();
    assert!(err.contains("not_a_real_module"));
}

#[test]
fn validate_target_modules_empty_is_ok() {
    assert!(validate_target_modules(&[], KNOWN_LORA_TARGET_MODULES).is_ok());
}

/// A feasible set: two adapters contributing to the same projection,
/// well under both caps and agreeing on shape.
#[test]
fn plan_blend_accepts_a_feasible_set() {
    let planned = plan_blend(
        "ctx",
        [
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: 8,
                d_in: 4,
                d_out: 4,
            },
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: 4,
                d_in: 4,
                d_out: 4,
            },
        ],
    )
    .expect("a feasible set must plan");
    assert_eq!(planned.len(), 1, "one projection group in, one plan out");
    assert_eq!(planned[0].layer_idx, 0);
    assert_eq!(planned[0].module, "q_proj");
    assert_eq!(planned[0].rank_total, 12);
    assert_eq!((planned[0].d_in, planned[0].d_out), (4, 4));
}

/// Refusal (1): the summed rank for one projection exceeds the cap.
#[test]
fn plan_blend_rejects_rank_over_budget() {
    let err = plan_blend(
        "ctx",
        [
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: MAX_BLEND_RANK_TOTAL,
                d_in: 1,
                d_out: 1,
            },
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: 1,
                d_in: 1,
                d_out: 1,
            },
        ],
    )
    .expect_err("summed rank exceeding the cap must refuse");
    assert!(err.contains("exceeds MAX_BLEND_RANK_TOTAL"), "got: {err}");

    // The must-pass control: exactly at the cap succeeds, so the arm
    // above is refusing the OVER-budget case and not every input.
    assert!(
        plan_blend(
            "ctx",
            [BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: MAX_BLEND_RANK_TOTAL,
                d_in: 1,
                d_out: 1,
            }],
        )
        .is_ok()
    );
}

/// Refusal (2): the aggregate blend size across every projection
/// exceeds the cap, even though each individual projection is within
/// its own per-group rank budget.
#[test]
fn plan_blend_rejects_aggregate_over_budget() {
    let rank = MAX_BLEND_RANK_TOTAL; // exactly at the per-group cap
    let d_in = 2048usize;
    let d_out = 2048usize;
    // 65 distinct (layer_idx, module) groups: 4096*(2048+2048)*65 =
    // 1,090,519,040 > MAX_BLEND_TOTAL_ELEMENTS (1<<30).
    let projections: Vec<BlendProjection<'_>> = (0..65usize)
        .map(|layer_idx| BlendProjection {
            layer_idx,
            module: "q_proj",
            rank,
            d_in,
            d_out,
        })
        .collect();
    let err = plan_blend("ctx", projections).expect_err("aggregate over-budget must refuse");
    assert!(
        err.contains("aggregate") || err.contains("MAX_BLEND_TOTAL_ELEMENTS"),
        "got: {err}"
    );
}

/// Refusal (3): two adapters disagree on a projection's input or output
/// width.
#[test]
fn plan_blend_rejects_dimension_mismatch() {
    let err = plan_blend(
        "ctx",
        [
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: 4,
                d_in: 4,
                d_out: 4,
            },
            BlendProjection {
                layer_idx: 0,
                module: "q_proj",
                rank: 4,
                d_in: 8,
                d_out: 4,
            },
        ],
    )
    .expect_err("a d_in mismatch must refuse");
    assert!(err.contains("mismatched dimensions"), "got: {err}");
}

/// An empty input plans to an empty (trivially feasible) result, rather
/// than refusing -- the "must not be empty" rule belongs to
/// `blend_lora_layer_data`'s own calling contract (a zero-adapter
/// mixture means "base model", handled before it ever reaches a plan),
/// not to plan feasibility itself. This matters for the residency
/// registry, which must report a zero-adapter resident set as
/// blend-feasible rather than refusing.
#[test]
fn plan_blend_of_no_projections_is_trivially_feasible() {
    assert_eq!(plan_blend("ctx", []), Ok(Vec::new()));
}

/// `plan_grouped` over projections already grouped by `(layer_idx,
/// module)` must plan the same set (up to group order) as `plan_blend`
/// grouping the same projections itself, for a feasible multi-group
/// input -- the whole point of splitting `plan_grouped` out is that a
/// caller who already has the grouping gets an identical plan without
/// paying for `plan_blend`'s own re-grouping.
#[test]
fn plan_grouped_matches_plan_blend_on_a_feasible_multi_group_set() {
    let groups: Vec<(usize, &str, Vec<BlendProjection<'_>>)> = vec![
        (
            0,
            "q_proj",
            vec![
                BlendProjection {
                    layer_idx: 0,
                    module: "q_proj",
                    rank: 8,
                    d_in: 4,
                    d_out: 4,
                },
                BlendProjection {
                    layer_idx: 0,
                    module: "q_proj",
                    rank: 4,
                    d_in: 4,
                    d_out: 4,
                },
            ],
        ),
        (
            1,
            "k_proj",
            vec![BlendProjection {
                layer_idx: 1,
                module: "k_proj",
                rank: 2,
                d_in: 6,
                d_out: 6,
            }],
        ),
    ];
    let flattened: Vec<BlendProjection<'_>> = groups
        .iter()
        .flat_map(|(_, _, entries)| entries.iter().copied())
        .collect();

    let mut via_grouped =
        plan_grouped("ctx", groups).expect("a feasible pre-grouped set must plan");
    let mut via_blend = plan_blend("ctx", flattened).expect("the same set flattened must plan");

    via_grouped.sort_by_key(|p| (p.layer_idx, p.module));
    via_blend.sort_by_key(|p| (p.layer_idx, p.module));
    assert_eq!(via_grouped, via_blend);
}

/// `plan_grouped` must refuse with the exact same error string as
/// `plan_blend` on each of the three refusal fixtures above, so a
/// caller that switches from `plan_blend` to `plan_grouped` (as
/// `blend_lora_layer_data` does) sees no change in its own error text.
#[test]
fn plan_grouped_matches_plan_blend_error_strings_on_refusals() {
    // Refusal (1): summed rank over budget.
    let rank_over_budget = vec![
        BlendProjection {
            layer_idx: 0,
            module: "q_proj",
            rank: MAX_BLEND_RANK_TOTAL,
            d_in: 1,
            d_out: 1,
        },
        BlendProjection {
            layer_idx: 0,
            module: "q_proj",
            rank: 1,
            d_in: 1,
            d_out: 1,
        },
    ];
    let err_blend = plan_blend("ctx", rank_over_budget.clone())
        .expect_err("summed rank exceeding the cap must refuse");
    let err_grouped = plan_grouped("ctx", vec![(0usize, "q_proj", rank_over_budget)])
        .expect_err("summed rank exceeding the cap must refuse");
    assert_eq!(err_blend, err_grouped);

    // Refusal (2): aggregate over budget, one entry per group so each
    // group's `plan_grouped` list mirrors `plan_blend`'s own grouping.
    let rank = MAX_BLEND_RANK_TOTAL;
    let d_in = 2048usize;
    let d_out = 2048usize;
    let aggregate_over_budget: Vec<BlendProjection<'_>> = (0..65usize)
        .map(|layer_idx| BlendProjection {
            layer_idx,
            module: "q_proj",
            rank,
            d_in,
            d_out,
        })
        .collect();
    let err_blend = plan_blend("ctx", aggregate_over_budget.clone())
        .expect_err("aggregate over-budget must refuse");
    let aggregate_groups: Vec<(usize, &str, Vec<BlendProjection<'_>>)> = aggregate_over_budget
        .iter()
        .map(|p| (p.layer_idx, p.module, vec![*p]))
        .collect();
    let err_grouped =
        plan_grouped("ctx", aggregate_groups).expect_err("aggregate over-budget must refuse");
    assert_eq!(err_blend, err_grouped);

    // Refusal (3): dimension mismatch within one group.
    let dimension_mismatch = vec![
        BlendProjection {
            layer_idx: 0,
            module: "q_proj",
            rank: 4,
            d_in: 4,
            d_out: 4,
        },
        BlendProjection {
            layer_idx: 0,
            module: "q_proj",
            rank: 4,
            d_in: 8,
            d_out: 4,
        },
    ];
    let err_blend =
        plan_blend("ctx", dimension_mismatch.clone()).expect_err("a d_in mismatch must refuse");
    let err_grouped = plan_grouped("ctx", vec![(0usize, "q_proj", dimension_mismatch)])
        .expect_err("a d_in mismatch must refuse");
    assert_eq!(err_blend, err_grouped);
}

/// An empty group is a caller bug reported through `Err`, never a panic
/// (there is no `entries[0]` to index into).
#[test]
fn plan_grouped_rejects_an_empty_group() {
    let groups: Vec<(usize, &str, Vec<BlendProjection<'_>>)> = vec![(0usize, "q_proj", Vec::new())];
    let err = plan_grouped("ctx", groups).expect_err("an empty group must refuse, not panic");
    assert!(err.contains("empty"), "got: {err}");
}

/// An entry whose own `(layer_idx, module)` disagrees with the group key
/// it was filed under: grouped as `(0, "q_proj")` but the entry itself
/// says `(1, "k_proj")`. Without the group-key check in `plan_grouped`
/// (removing the `check_group_key_match` calls in its first pass would
/// do it), this entry blends silently into the `(0, "q_proj")` plan
/// instead of refusing, and the call succeeds where it must not.
#[test]
fn plan_grouped_rejects_an_entry_whose_own_key_disagrees_with_its_group() {
    let groups: Vec<(usize, &str, Vec<BlendProjection<'_>>)> = vec![(
        0usize,
        "q_proj",
        vec![BlendProjection {
            layer_idx: 1,
            module: "k_proj",
            rank: 4,
            d_in: 4,
            d_out: 4,
        }],
    )];
    let err = plan_grouped("ctx", groups)
        .expect_err("an entry keyed differently from its group must refuse, not blend");
    assert!(err.contains("q_proj"), "got: {err}");
    assert!(err.contains("k_proj"), "got: {err}");
    assert!(err.contains('0') && err.contains('1'), "got: {err}");
}
