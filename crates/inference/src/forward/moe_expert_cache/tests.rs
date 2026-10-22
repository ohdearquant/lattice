use super::*;

fn cfg(num_slots: Option<usize>) -> MoeExpertCacheConfig {
    MoeExpertCacheConfig { num_slots }
}

#[test]
fn env_override_clamped_to_top_k_and_num_experts_when_budget_is_not_binding() {
    // Generous working set (400 GB) so the budget-derived maximum for
    // this shape is >= num_experts (256) and num_experts is the actual
    // binding upper bound — the case the original clamp-only-to-
    // num_experts logic was designed for.
    assert_eq!(
        moe_expert_cache_num_slots(&cfg(Some(1)), 256, 8, 6_291_456, 40, 400_000_000_000, 0, 0)
            .unwrap(),
        8,
        "below top_k clamps up"
    );
    assert_eq!(
        moe_expert_cache_num_slots(
            &cfg(Some(9999)),
            256,
            8,
            6_291_456,
            40,
            400_000_000_000,
            0,
            0,
        )
        .unwrap(),
        256,
        "above num_experts clamps down to num_experts when the budget has headroom to spare"
    );
    assert_eq!(
        moe_expert_cache_num_slots(&cfg(Some(64)), 256, 8, 6_291_456, 40, 400_000_000_000, 0, 0)
            .unwrap(),
        64,
        "in-range passes through unchanged"
    );
}

#[test]
fn env_override_capped_at_working_set_budget_even_when_below_num_experts() {
    // Same 256-expert / top_k=8 / 40-layer / 6 MiB-per-expert shape as
    // above, but on a tight 40 GB device budget: the budget-derived
    // maximum here is 135 slots — well below num_experts (256). This is
    // the exact scenario the Major finding covers: a large override
    // (or, unbounded, a fully-resident cache) must NOT be able to
    // exceed what the device can actually hold, even though 135 and
    // 9999 both clamp to *something* under the old `[top_k,
    // num_experts]`-only bound.
    let budget_max = 135;
    assert_eq!(
        moe_expert_cache_num_slots(
            &cfg(Some(9999)),
            256,
            8,
            6_291_456,
            40,
            40_000_000_000,
            0,
            0
        )
        .unwrap(),
        budget_max,
        "an override far above num_experts must cap at the working-set budget, not num_experts"
    );
    assert_eq!(
        moe_expert_cache_num_slots(&cfg(Some(200)), 256, 8, 6_291_456, 40, 40_000_000_000, 0, 0)
            .unwrap(),
        budget_max,
        "an override below num_experts but above the budget must still be capped at the budget"
    );
    assert_eq!(
        moe_expert_cache_num_slots(&cfg(Some(100)), 256, 8, 6_291_456, 40, 40_000_000_000, 0, 0)
            .unwrap(),
        100,
        "an override under the budget passes through unchanged"
    );
}

#[test]
fn default_zero_eviction_fast_path_when_everything_fits() {
    // Small synthetic-test-sized shapes: everything should fit easily,
    // defaulting to num_experts (zero-eviction fast path).
    let n = moe_expert_cache_num_slots(&cfg(None), 4, 1, 4096, 1, 8_000_000_000, 0, 0).unwrap();
    assert_eq!(n, 4);
}

#[test]
fn auto_shrinks_under_device_budget() {
    // Qwen3.5-35B-A3B-shaped numbers: 256 experts, top_k=8, 40 MoE
    // layers, ~6 MiB/expert (gate_up+down combined, f16), on a 32 GiB
    // device (`recommended_max_working_set_size` conservatively modeled
    // as ~28 GiB here, mirroring real macOS headroom below total RAM).
    let per_expert_bytes = 2 * 512 * 2048 * 2 + 2048 * 512 * 2; // gate_up + down, f16
    let n = moe_expert_cache_num_slots(
        &cfg(None),
        256,
        8,
        per_expert_bytes as u64,
        40,
        28 * 1024 * 1024 * 1024,
        0,
        0,
    )
    .unwrap();
    assert!(
        n < 256,
        "expected auto-shrink below num_experts=256 on a 32 GiB-class device, got {n}"
    );
    assert!(n >= 8, "must never shrink below top_k=8, got {n}");
}

#[test]
fn errors_when_even_top_k_does_not_fit() {
    // Absurdly tiny device budget: even 1 slot (top_k=1) can't fit.
    let err = moe_expert_cache_num_slots(&cfg(None), 4, 1, 10_000_000_000, 1, 1_000_000, 0, 0)
        .unwrap_err();
    assert!(
        err.contains("cannot fit even the lazy dequant-on-demand cache"),
        "unexpected error message: {err}"
    );
}

#[test]
fn rejects_zero_num_experts_or_top_k() {
    assert!(moe_expert_cache_num_slots(&cfg(None), 0, 1, 100, 1, 1_000_000_000, 0, 0).is_err());
    assert!(moe_expert_cache_num_slots(&cfg(None), 4, 0, 100, 1, 1_000_000_000, 0, 0).is_err());
}

#[test]
fn rejects_top_k_exceeding_num_experts() {
    assert!(moe_expert_cache_num_slots(&cfg(None), 4, 5, 100, 1, 1_000_000_000, 0, 0).is_err());
}

// ── Residency budget: `already_allocated_bytes` (issue: the budget was a
// fraction of device capacity, not capacity minus what is already
// resident) ──────────────────────────────────────────────────────────

#[test]
fn already_allocated_bytes_shrinks_the_slot_budget() {
    // Same 256-expert / top_k=8 / 40-layer / 6 MiB-per-expert / 40 GB
    // shape as `env_override_capped_at_working_set_budget_even_when_
    // below_num_experts`, where zero already-allocated bytes gives a
    // budget-derived maximum of 135 slots. With 4 GB already resident
    // on the device, the remaining per-layer budget must shrink
    // accordingly (750_000_000 / 6_291_456 = 119), strictly less than
    // the zero-residency figure — proving the parameter actually moves
    // the result rather than being wired in and ignored.
    let with_zero_resident =
        moe_expert_cache_num_slots(&cfg(None), 256, 8, 6_291_456, 40, 40_000_000_000, 0, 0)
            .unwrap();
    assert_eq!(with_zero_resident, 135);

    let with_4gb_resident = moe_expert_cache_num_slots(
        &cfg(None),
        256,
        8,
        6_291_456,
        40,
        40_000_000_000,
        4_000_000_000,
        0,
    )
    .unwrap();
    assert_eq!(
        with_4gb_resident, 119,
        "4 GB already resident on the device must reduce the remaining per-layer budget, \
             and therefore the slot count, below the zero-residency figure"
    );
    assert!(with_4gb_resident < with_zero_resident);
}

#[test]
fn already_allocated_at_or_above_threshold_errs_instead_of_wrapping() {
    // `already_allocated_bytes` can legitimately reach or exceed 0.85 ×
    // recommendedMaxWorkingSetSize: the working-set size is advisory,
    // not a hard ceiling, so a process under memory pressure can sit
    // above it. This is the underflow case: `threshold -
    // already_allocated_bytes` would be negative in unsigned
    // arithmetic. A bare (non-saturating) subtraction wraps to a value
    // near `u64::MAX`, which turns `per_layer_budget` effectively
    // unbounded and admits every expert (`budget_max_slots` clamps up
    // to `num_experts`) instead of shrinking or erroring — silently
    // reconstructing the fully-resident cache #682 Stage 1 exists to
    // avoid. The safe behavior is for the remaining per-layer budget to
    // saturate at 0, which — with a nonzero per-expert size — is below
    // even the top_k floor, so this must return `Err` (the same "even
    // top_k doesn't fit" outcome `errors_when_even_top_k_does_not_fit`
    // exercises), never `Ok` with a slot count computed from a wrapped
    // budget.
    let recommended_max_working_set_size = 40_000_000_000u64;
    let threshold = (recommended_max_working_set_size as f64 * 0.85) as u64;

    // Exactly at the threshold: remaining budget saturates to exactly 0.
    let err_at = moe_expert_cache_num_slots(
        &cfg(None),
        256,
        8,
        6_291_456,
        40,
        recommended_max_working_set_size,
        threshold,
        0,
    )
    .unwrap_err();
    assert!(
        err_at.contains("cannot fit even the lazy dequant-on-demand cache"),
        "unexpected error message at threshold: {err_at}"
    );

    // Above the threshold: this is exactly the case a bare `threshold -
    // already_allocated_bytes` cannot represent in u64.
    let err_above = moe_expert_cache_num_slots(
        &cfg(None),
        256,
        8,
        6_291_456,
        40,
        recommended_max_working_set_size,
        threshold + 1_000_000_000,
        0,
    )
    .unwrap_err();
    assert!(
        err_above.contains("cannot fit even the lazy dequant-on-demand cache"),
        "unexpected error message above threshold: {err_above}"
    );

    // An explicit `cfg.num_slots` override must not escape the same
    // guard: the working-set budget is derived and enforced
    // unconditionally (see the comment at the override-clamp call
    // site), so a garbage-large override must still error rather than
    // being granted a slot count derived from a wrapped budget.
    let err_override = moe_expert_cache_num_slots(
        &cfg(Some(9999)),
        256,
        8,
        6_291_456,
        40,
        recommended_max_working_set_size,
        threshold + 1_000_000_000,
        0,
    )
    .unwrap_err();
    assert!(
        err_override.contains("cannot fit even the lazy dequant-on-demand cache"),
        "unexpected error message for an override above threshold: {err_override}"
    );
}

#[test]
fn sequential_layer_sizing_gets_equal_shares_not_geometric_shrink() {
    // Simulates what the real loader does: call this pure function once
    // per MoE layer, in order, feeding each layer's ACTUAL consumed
    // bytes (slots * per_expert_bytes — not its budget, which may
    // differ under integer truncation) into the next call's
    // `already_allocated_bytes`, exactly like `ExpertSlotCache::new`
    // committing real Metal buffers before the next layer is sized.
    //
    // Numbers are chosen so every quantity divides evenly and the
    // prediction below is exact, tolerance 0 — not a real-world shape,
    // deliberately, so a geometric-shrink defect cannot hide behind
    // integer-rounding noise. (The production Qwen3.5-35B-A3B shape
    // used elsewhere in this file does not divide evenly, and would
    // show at most a +/-1 slot drift from truncation leftover
    // accumulating across layers — a rounding residual, not the
    // geometric-shrink defect this test targets.)
    //
    // recommended_max_working_set_size = 52_941_177 => threshold
    // (0.85x, truncated) = 45_000_000 exactly. Starting resident
    // baseline (nonzero, as if dense model weights are already loaded)
    // = 5_000_000, so the headroom available to MoE sizing is
    // 40_000_000 over 8 layers = 5_000_000/layer. At 1_000_000
    // bytes/expert that is exactly 5 slots/layer (top_k=4 <= 5, so no
    // auto-shrink-to-top_k and no Err).
    //
    // PREDICTION (stated before running): every one of the 8 layers
    // gets EXACTLY 5 slots (tolerance 0 — see above), and the running
    // total after all 8 layers lands at EXACTLY the threshold
    // (45_000_000), never over it.
    //
    // This must FAIL against the constant-denominator formula (divide the remaining
    // budget by the constant `num_moe_layers` on every call, ignoring
    // how many layers have already been sized): layer 0 still gets 5
    // slots (already_allocated_bytes == baseline for the very first
    // call, so both formulas agree there), but layer 1's
    // already-consumed 5_000_000 bytes are subtracted from the
    // numerator while the denominator stays 8 instead of dropping to
    // 7, so layer 1 gets floor(4_375_000 / 1_000_000) = 4 slots, and by
    // layer 2 the remaining per-layer budget (3_875_000) is below
    // min_required_bytes (4 * 1_000_000 = 4_000_000) and the call
    // returns `Err` outright.
    let num_experts = 256;
    let top_k = 4;
    let per_expert_bytes: u64 = 1_000_000;
    let num_moe_layers = 8;
    let recommended_max_working_set_size: u64 = 52_941_177;
    let threshold = (recommended_max_working_set_size as f64 * 0.85) as u64;
    assert_eq!(
        threshold, 45_000_000,
        "sanity-check the chosen R truncates as predicted"
    );
    let baseline_resident: u64 = 5_000_000;

    let mut already_allocated = baseline_resident;
    for moe_layers_already_sized in 0..num_moe_layers {
        let slots = moe_expert_cache_num_slots(
            &cfg(None),
            num_experts,
            top_k,
            per_expert_bytes,
            num_moe_layers,
            recommended_max_working_set_size,
            already_allocated,
            moe_layers_already_sized,
        )
        .unwrap_or_else(|e| {
            panic!(
                "layer {moe_layers_already_sized}/{num_moe_layers} unexpectedly errored \
                     (already_allocated={already_allocated}): {e}"
            )
        });
        assert_eq!(
            slots, 5,
            "layer {moe_layers_already_sized}/{num_moe_layers} got {slots} slots, expected \
                 exactly 5 (equal share) — already_allocated={already_allocated}"
        );
        already_allocated += slots as u64 * per_expert_bytes;
    }
    assert_eq!(
        already_allocated, threshold,
        "sum of every layer's actual consumption must land exactly at the threshold, never \
             over it"
    );
}

/// Serializes every test below that mutates the real process
/// environment (`LATTICE_MOE_EXPERT_CACHE_SLOTS` is process-global —
/// `cargo test` runs this file's tests on multiple threads within one
/// process, so unguarded concurrent `set_var`/`remove_var` calls would
/// race and flake).
static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Sets (or clears) `LATTICE_MOE_EXPERT_CACHE_SLOTS` for the duration of
/// `f`, holding `ENV_TEST_LOCK` and restoring the prior value
/// (including "was unset") on the way out, even if `f` panics.
fn with_env_var<R>(value: Option<&str>, f: impl FnOnce() -> R) -> R {
    let _guard = ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prior = std::env::var(MOE_EXPERT_CACHE_SLOTS_ENV).ok();
    // SAFETY: serialized by `ENV_TEST_LOCK` above — no concurrent
    // reader/writer of this process-global variable.
    unsafe {
        match value {
            Some(v) => std::env::set_var(MOE_EXPERT_CACHE_SLOTS_ENV, v),
            None => std::env::remove_var(MOE_EXPERT_CACHE_SLOTS_ENV),
        }
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    // SAFETY: see above.
    unsafe {
        match &prior {
            Some(v) => std::env::set_var(MOE_EXPERT_CACHE_SLOTS_ENV, v),
            None => std::env::remove_var(MOE_EXPERT_CACHE_SLOTS_ENV),
        }
    }
    match result {
        Ok(r) => r,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

#[test]
fn from_env_unset_is_none() {
    with_env_var(None, || {
        assert_eq!(MoeExpertCacheConfig::from_env().unwrap().num_slots, None);
    });
}

#[test]
fn from_env_valid_positive_integer_is_some() {
    with_env_var(Some("42"), || {
        assert_eq!(
            MoeExpertCacheConfig::from_env().unwrap().num_slots,
            Some(42)
        );
    });
    with_env_var(Some("  7  "), || {
        assert_eq!(MoeExpertCacheConfig::from_env().unwrap().num_slots, Some(7));
    });
}

#[test]
fn from_env_garbage_errors_loudly_instead_of_silently_falling_back() {
    with_env_var(Some("not-a-number"), || {
        let err = MoeExpertCacheConfig::from_env().unwrap_err();
        assert!(
            err.contains("not a valid positive integer"),
            "unexpected error message: {err}"
        );
    });
    with_env_var(Some("-5"), || {
        assert!(MoeExpertCacheConfig::from_env().is_err());
    });
    with_env_var(Some(""), || {
        assert!(MoeExpertCacheConfig::from_env().is_err());
    });
}

#[test]
fn from_env_zero_errors_loudly() {
    with_env_var(Some("0"), || {
        let err = MoeExpertCacheConfig::from_env().unwrap_err();
        assert!(err.contains("is 0"), "unexpected error message: {err}");
    });
}

/// The per-expert dequantization is one of the Q4 consumers that must honour a
/// negative scale; see `signed_scale_fixture` for the controls.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn dequant_expert_f16_honours_negative_scales() {
    use crate::weights::q4_weights::signed_scale_fixture::{
        abs_scales, assert_sign_honoured, negative_scale_tensor, reference, zero_scales,
    };
    use crate::weights::q4_weights::{Q4Tensor, q4_f16_to_f32, save_q4_file};

    let dir = tempfile::tempdir().expect("tempdir");
    let shape = [2usize, 4, 64];
    let mut t = negative_scale_tensor(8, 64, 0xA1);
    t.shape = shape.to_vec();
    let per_expert = 4 * 64;
    let load = |t: &Q4Tensor, name: &str| -> Result<Vec<f64>, String> {
        let path = dir.path().join(name);
        save_q4_file(&path, t).map_err(|e| e.to_string())?;
        let table = ExpertByteTable::open(&path, &shape)?;
        let mut out = Vec::with_capacity(t.original_len);
        for expert in 0..shape[0] {
            let bits = table.dequant_expert_f16(expert)?;
            assert_eq!(bits.len(), per_expert);
            out.extend(bits.into_iter().map(|b| f64::from(q4_f16_to_f32(b))));
        }
        Ok(out)
    };
    // A zero scale is refused when the expert is read, so the zero control is what
    // a clamping consumer would compute: every weight equal to its bias.
    let zero = zero_scales(&t);
    let err = load(&zero, "zero.q4").expect_err("a zero scale is refused");
    assert!(
        err.contains("finite and nonzero"),
        "unexpected error: {err}"
    );
    assert_sign_honoured(
        "ExpertByteTable::dequant_expert_f16",
        &reference(&t),
        &load(&t, "neg.q4").expect("a negative scale loads"),
        &load(&abs_scales(&t), "abs.q4").expect("abs control loads"),
        &reference(&zero),
        1e-3,
    );
}
