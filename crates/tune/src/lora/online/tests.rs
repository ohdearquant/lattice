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

#[test]
fn test_adapt_step_basic() {
    let mut adapter = make_small_adapter();
    let key = (0usize, "q_proj".to_string());
    let a_init = adapter.layers()[&key].a.clone();
    let b_init = adapter.layers()[&key].b.clone();

    let input = [1.0f32, 2.0, 3.0];
    let target_delta = [5.0f32, 5.0];

    let r1 = adapt_step(&mut adapter, 0, "q_proj", &input, &target_delta, 0.01)
        .expect("adapt_step should succeed");

    assert!(r1.loss > 0.0, "initial loss must be positive");
    assert!(r1.grad_norm > 0.0, "initial grad_norm must be positive");
    assert_ne!(adapter.layers()[&key].a, a_init, "A must change after step");
    assert_ne!(adapter.layers()[&key].b, b_init, "B must change after step");

    let r2 = adapt_step(&mut adapter, 0, "q_proj", &input, &target_delta, 0.01)
        .expect("second adapt_step should succeed");

    assert!(
        r2.loss < r1.loss,
        "loss must decrease: {} >= {}",
        r2.loss,
        r1.loss
    );
}

#[test]
fn test_adapt_step_missing_layer() {
    let mut adapter = make_small_adapter();
    let err = adapt_step(
        &mut adapter,
        99,
        "missing",
        &[1.0, 2.0, 3.0],
        &[1.0, 1.0],
        0.01,
    )
    .expect_err("missing layer must return Err");
    assert!(
        matches!(err, crate::error::TuneError::Training(_)),
        "expected Training variant, got {err:?}"
    );
}

#[test]
fn test_adapt_step_dimension_mismatch() {
    let mut adapter = make_small_adapter();

    // input too short (d_in=3, passing 2)
    let err = adapt_step(&mut adapter, 0, "q_proj", &[1.0, 2.0], &[1.0, 1.0], 0.01)
        .expect_err("wrong input length must return Err");
    assert!(
        matches!(err, crate::error::TuneError::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );

    // target_delta wrong length (d_out=2, passing 3)
    let err = adapt_step(
        &mut adapter,
        0,
        "q_proj",
        &[1.0, 2.0, 3.0],
        &[1.0, 1.0, 1.0],
        0.01,
    )
    .expect_err("wrong target_delta length must return Err");
    assert!(
        matches!(err, crate::error::TuneError::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

#[test]
fn test_adapt_step_determinism() {
    let mut adapter1 = make_small_adapter();
    let mut adapter2 = make_small_adapter();

    let input = [1.0f32, 2.0, 3.0];
    let target_delta = [5.0f32, 5.0];

    let r1 = adapt_step(&mut adapter1, 0, "q_proj", &input, &target_delta, 0.01)
        .expect("adapt_step on adapter1 should succeed");
    let r2 = adapt_step(&mut adapter2, 0, "q_proj", &input, &target_delta, 0.01)
        .expect("adapt_step on adapter2 should succeed");

    assert_eq!(
        r1.loss, r2.loss,
        "loss must be identical for identical inputs"
    );
    assert_eq!(
        r1.grad_norm, r2.grad_norm,
        "grad_norm must be identical for identical inputs"
    );

    let key = (0usize, "q_proj".to_string());
    assert_eq!(adapter1.layers()[&key].a, adapter2.layers()[&key].a);
    assert_eq!(adapter1.layers()[&key].b, adapter2.layers()[&key].b);
}
