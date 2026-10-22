use super::*;
use crate::lora::{LoraAdapter, LoraConfig, LoraLayer};
use std::collections::HashMap;

/// rank=2, d_in=3, d_out=2, scale=1.0 (alpha=2.0, rank=2)
fn make_small_adapter() -> LoraAdapter {
    let config = LoraConfig {
        rank: 2,
        alpha: 2.0,
        target_modules: vec!["q_proj".into()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0, "q_proj".into()),
        LoraLayer {
            a: vec![
                1.0, 0.5, 0.0, // row 0
                0.0, 0.5, 1.0, // row 1
            ],
            b: vec![
                1.0, 0.0, // row 0
                0.0, 1.0, // row 1
            ],
            d_in: 3,
            d_out: 2,
            rank: 2,
        },
    );
    LoraAdapter::new(config, layers).expect("valid adapter config")
}

// ─── gradient computation tests ────────────────────────────────────────

#[test]
fn test_gradient_computation() {
    let adapter = make_small_adapter();
    let input = vec![1.0f32, 2.0, 3.0];
    let target_delta = vec![5.0f32, 5.0];

    let grads = compute_lora_gradients(&adapter, 0, "q_proj", &input, &target_delta)
        .expect("compute_lora_gradients must succeed");

    // grad_b shape: d_out * rank = 2 * 2 = 4
    assert_eq!(
        grads.grad_b.len(),
        4,
        "grad_b must have d_out*rank elements"
    );
    // grad_a shape: rank * d_in = 2 * 3 = 6
    assert_eq!(grads.grad_a.len(), 6, "grad_a must have rank*d_in elements");
    // Loss must be positive because delta != target.
    assert!(grads.loss > 0.0, "loss must be positive");
    // At least one gradient element must be non-zero.
    assert!(
        grads.grad_b.iter().any(|&g| g.abs() > 1e-9)
            || grads.grad_a.iter().any(|&g| g.abs() > 1e-9),
        "gradients must be non-zero"
    );
}

#[test]
fn test_gradient_computation_missing_layer() {
    let adapter = make_small_adapter();
    let err = compute_lora_gradients(&adapter, 99, "missing", &[1.0, 2.0], &[1.0])
        .expect_err("missing layer must return Err");
    assert!(
        matches!(err, TuneError::Training(_)),
        "expected Training variant, got {err:?}"
    );
}

#[test]
fn test_gradient_computation_dimension_mismatch() {
    let adapter = make_small_adapter();
    // input too short (d_in=3, passing 2)
    let err = compute_lora_gradients(&adapter, 0, "q_proj", &[1.0, 2.0], &[1.0, 1.0])
        .expect_err("wrong input length must return Err");
    assert!(
        matches!(err, TuneError::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

// ─── Adam state tests ───────────────────────────────────────────────────

#[test]
fn test_adam_step_updates_params() {
    let mut state = AdamState::new();
    let mut params = vec![1.0f32, 2.0, 3.0];
    let grads = vec![0.1f32, 0.2, 0.3];
    let original = params.clone();

    state.step(
        "test",
        &mut params,
        &grads,
        0.01,
        0.9,
        0.999,
        1e-8,
        0.0,
        false,
    );

    for (p, o) in params.iter().zip(original.iter()) {
        assert_ne!(p, o, "params must change after Adam step");
    }
    assert_eq!(
        state.t["test"], 1,
        "key 'test' step counter must be 1 after first step"
    );
}

/// Verify that each key advances its own independent timestep counter.
///
/// The prior shared-counter bug caused the 2nd..Nth key stepped in a single
/// optimizer round to receive an inflated `t`, defeating Adam's warmup. This
/// test exercises the exact failure mode: step keys a/b/c/d once each, then
/// step only "a" again. Every key must reflect only its own update count.
#[test]
fn test_adam_per_key_timestep_no_cross_advance() {
    let mut state = AdamState::new();
    let grads = vec![0.1f32, 0.2];
    let mut pa = vec![1.0f32, 1.0];
    let mut pb = vec![2.0f32, 2.0];
    let mut pc = vec![3.0f32, 3.0];
    let mut pd = vec![4.0f32, 4.0];

    // Step each key once.
    for (key, params) in [
        ("a", &mut pa),
        ("b", &mut pb),
        ("c", &mut pc),
        ("d", &mut pd),
    ] {
        state.step(key, params, &grads, 0.01, 0.9, 0.999, 1e-8, 0.0, false);
    }

    assert_eq!(state.t["a"], 1, "key 'a' must be at t=1 after one step");
    assert_eq!(state.t["b"], 1, "key 'b' must be at t=1 after one step");
    assert_eq!(state.t["c"], 1, "key 'c' must be at t=1 after one step");
    assert_eq!(state.t["d"], 1, "key 'd' must be at t=1 after one step");

    // Step only "a" a second time.
    state.step("a", &mut pa, &grads, 0.01, 0.9, 0.999, 1e-8, 0.0, false);

    assert_eq!(
        state.t["a"], 2,
        "key 'a' must advance to t=2 on its second step"
    );
    assert_eq!(
        state.t["b"], 1,
        "key 'b' must not advance when only 'a' is stepped"
    );
    assert_eq!(
        state.t["c"], 1,
        "key 'c' must not advance when only 'a' is stepped"
    );
    assert_eq!(
        state.t["d"], 1,
        "key 'd' must not advance when only 'a' is stepped"
    );
}

// Integration tests for AdamW weight decay and Adam-vs-SGD convergence
// require the train_lora loop (issue #88). They will land alongside that.
