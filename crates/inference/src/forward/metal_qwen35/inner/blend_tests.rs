use super::*;

/// Reference: compute B @ (A @ x) using plain nested loops that match
/// the kernel's row-major indexing: A[r,k]=a[r*d_in+k], B[o,r]=b[o*rank+r].
fn lora_delta(a: &[f32], b: &[f32], x: &[f32], rank: usize, d_in: usize, d_out: usize) -> Vec<f32> {
    let mut ax = vec![0.0f32; rank];
    for r in 0..rank {
        for k in 0..d_in {
            ax[r] += a[r * d_in + k] * x[k];
        }
    }
    let mut y = vec![0.0f32; d_out];
    for o in 0..d_out {
        for r in 0..rank {
            y[o] += b[o * rank + r] * ax[r];
        }
    }
    y
}

#[test]
fn blend_two_adapters_equals_sum_delta() {
    let d_in = 8usize;
    let d_out = 6usize;
    let r1 = 2usize;
    let r2 = 3usize;

    // Deterministic, non-trivial values: A row-major (rank × d_in), B row-major (d_out × rank).
    let a1: Vec<f32> = (0..r1 * d_in).map(|i| i as f32 * 0.1 + 0.01).collect();
    let b1: Vec<f32> = (0..d_out * r1).map(|i| i as f32 * 0.05 - 0.5).collect();
    let a2: Vec<f32> = (0..r2 * d_in).map(|i| i as f32 * 0.07 + 0.03).collect();
    let b2: Vec<f32> = (0..d_out * r2).map(|i| i as f32 * 0.03 - 0.2).collect();

    let l1 = LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: a1.clone(),
        b: b1.clone(),
        rank: r1,
        d_in,
        d_out,
    };
    let l2 = LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: a2.clone(),
        b: b2.clone(),
        rank: r2,
        d_in,
        d_out,
    };

    let w1 = 0.25f32;
    let w2 = 0.75f32;

    let blended = blend_lora_layer_data(&[(&[l1], w1), (&[l2], w2)]).expect("blend must not fail");
    assert_eq!(blended.len(), 1, "one projection in, one out");
    let bl = &blended[0];

    // Structural assertions.
    assert_eq!(bl.rank, r1 + r2);
    assert_eq!(bl.d_in, d_in);
    assert_eq!(bl.d_out, d_out);

    // Delta equality: blended_delta(x) == w1*delta1(x) + w2*delta2(x) for all x.
    // LoraLayerData carries no alpha/rank — w_e IS the effective scale.
    let x: Vec<f32> = (0..d_in).map(|i| i as f32 * 0.3 - 1.0).collect();

    let delta_blend = lora_delta(&bl.a, &bl.b, &x, bl.rank, d_in, d_out);
    let delta1 = lora_delta(&a1, &b1, &x, r1, d_in, d_out);
    let delta2 = lora_delta(&a2, &b2, &x, r2, d_in, d_out);
    let expected: Vec<f32> = delta1
        .iter()
        .zip(&delta2)
        .map(|(d1, d2)| w1 * d1 + w2 * d2)
        .collect();

    let max_err = delta_blend
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-5,
        "blended delta diverges from reference sum: max_abs_diff={max_err}"
    );
}

// -----------------------------------------------------------------
// Summed rank exceeding MAX_BLEND_RANK_TOTAL returns Err.
// -----------------------------------------------------------------
#[test]
fn blend_rank_exceeds_cap_returns_error() {
    let d_in = 1usize;
    let d_out = 1usize;
    // rank = cap + 1; A and B are tiny (~32 KB total).
    let rank = MAX_BLEND_RANK_TOTAL + 1;
    let layer = LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: vec![0.1f32; rank * d_in],
        b: vec![0.1f32; d_out * rank],
        rank,
        d_in,
        d_out,
    };
    let result = blend_lora_layer_data(&[(&[layer], 1.0)]);
    assert!(
        result.is_err(),
        "summed rank exceeding cap must return an error"
    );
}

// -----------------------------------------------------------------
// A slice with wrong length returns Err (fail-closed on
// malformed input before any allocation).
// -----------------------------------------------------------------
#[test]
fn blend_mismatched_a_length_returns_error() {
    let d_in = 4usize;
    let d_out = 4usize;
    let rank = 2usize;
    let layer = LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        // A is one element short of rank * d_in.
        a: vec![0.0f32; rank * d_in - 1],
        b: vec![0.0f32; d_out * rank],
        rank,
        d_in,
        d_out,
    };
    let result = blend_lora_layer_data(&[(&[layer], 1.0)]);
    assert!(
        result.is_err(),
        "mismatched A slice length must return an error"
    );
}

// -----------------------------------------------------------------
// B slice with wrong length returns Err.
// -----------------------------------------------------------------
#[test]
fn blend_mismatched_b_length_returns_error() {
    let d_in = 4usize;
    let d_out = 4usize;
    let rank = 2usize;
    let layer = LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: vec![0.0f32; rank * d_in],
        // B is one element short of d_out * rank.
        b: vec![0.0f32; d_out * rank - 1],
        rank,
        d_in,
        d_out,
    };
    let result = blend_lora_layer_data(&[(&[layer], 1.0)]);
    assert!(
        result.is_err(),
        "mismatched B slice length must return an error"
    );
}

// -----------------------------------------------------------------
// An empty inner layer slice passes the outer `inputs.is_empty()`
// guard (the outer slice has one entry) but must still be rejected,
// not silently produce an empty `Ok(Vec::new())` blend.
// -----------------------------------------------------------------
#[test]
fn blend_lora_layer_data_rejects_empty_inner_layer_slice() {
    let empty: &[LoraLayerData] = &[];
    let result = blend_lora_layer_data(&[(empty, 1.0)]);
    assert!(
        result.is_err(),
        "an empty inner layer slice must return an error, not Ok(Vec::new())"
    );
}

// -----------------------------------------------------------------
// Contract: a malformed huge dimension returns Err, never panics.
//
// rank=2, d_in=usize::MAX/2+1, d_out=1 with empty a and b=[0.0;2].
// blend_lora_layer_data is monolithic here: the aggregate pre-pass
// (rank_total*(d_in+d_out)) strictly dominates the per-entry
// checked_mul, so no input reaches the per-entry guard without
// tripping the pre-pass first. This pins the public Err-not-panic
// contract; the per-entry checked_mul is defense-in-depth, isolated
// on the tune side where blend_layer_entries is a separable helper.
// -----------------------------------------------------------------
#[test]
fn blend_malformed_huge_dim_returns_err_not_panic() {
    let rank = 2usize;
    let d_in = usize::MAX / 2 + 1;
    let d_out = 1usize;
    let layer = LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: vec![],          // empty; pre-pass fails before slice-length check
        b: vec![0.0f32; 2], // rank * d_out = 2 * 1 = 2
        rank,
        d_in,
        d_out,
    };
    let result = blend_lora_layer_data(&[(&[layer], 1.0)]);
    // The pre-pass catches the overflow in rank_total*(d_in+d_out); the
    // per-entry checked_mul is defense-in-depth for the same class.
    assert!(
        result.is_err(),
        "malformed huge dim must return Err, not panic"
    );
}

// -----------------------------------------------------------------
// Contract: an aggregate element budget overrun returns Err before alloc.
//
// 65 layers with rank=4096, d_in=2048, d_out=2048, empty a/b, so
// the test allocates nothing. Per-group: rank_total=4096 == cap
// (passes the per-projection check). Aggregate: 4096*(2048+2048)*65
// = 1,090,519,040 > MAX_BLEND_TOTAL_ELEMENTS (1<<30).
// -----------------------------------------------------------------
#[test]
fn blend_aggregate_budget_exceeded_returns_err() {
    let rank = MAX_BLEND_RANK_TOTAL; // 4096 — exactly at per-group cap
    let d_in = 2048usize;
    let d_out = 2048usize;
    let layers: Vec<LoraLayerData> = (0..65usize)
        .map(|idx| LoraLayerData {
            layer_idx: idx,
            module: "q_proj".into(),
            a: vec![], // empty; pre-pass reads only declared fields
            b: vec![],
            rank,
            d_in,
            d_out,
        })
        .collect();
    let result = blend_lora_layer_data(&[(&layers, 1.0)]);
    assert!(result.is_err(), "aggregate budget exceeded must return Err");
    // Use result.err().expect() rather than unwrap_err() because
    // Vec<LoraLayerData> does not implement Debug (unwrap_err requires T: Debug).
    let msg = result.err().expect("already asserted is_err").to_string();
    assert!(
        msg.contains("aggregate") || msg.contains("MAX_BLEND_TOTAL_ELEMENTS"),
        "error message must mention the aggregate budget; got: {msg}"
    );
}
