//! End-to-end LoRA lifecycle integration tests.
//!
//! Exercises the full pipeline: create adapter → train → apply → verify.
//! Each test name describes the lifecycle phase it validates.

use lattice_tune::lora::{LoraAdapter, LoraConfig, LoraLayer, apply_lora};
use lattice_tune::{LoraTrainConfig, TrainSample, train_lora};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// rank=2, d_in=8, d_out=4 with non-zero initial weights so the initial
/// apply_lora output is detectable before training begins.
fn make_test_adapter() -> LoraAdapter {
    let config = LoraConfig {
        rank: 2,
        alpha: 2.0, // scale = alpha / rank = 1.0
        target_modules: vec!["q_proj".into()],
    };

    let mut layers = HashMap::new();
    // A: (rank=2, d_in=8) — row-major, all 0.1
    // B: (d_out=4, rank=2) — row-major, all 0.1
    layers.insert(
        (0usize, "q_proj".to_string()),
        LoraLayer {
            a: vec![0.1; 2 * 8],
            b: vec![0.1; 4 * 2],
            d_in: 8,
            d_out: 4,
            rank: 2,
        },
    );
    LoraAdapter::new(config, layers)
}

/// Build a small set of regression samples for (layer=0, module="q_proj").
/// The target_delta is chosen so a gradient can drive the adapter towards it.
fn make_train_samples(n: usize) -> Vec<TrainSample> {
    (0..n)
        .map(|i| {
            let fi = i as f32;
            TrainSample {
                layer_idx: 0,
                module: "q_proj".to_string(),
                // d_in = 8
                input: (0..8).map(|j| (fi + j as f32) * 0.05 + 0.1).collect(),
                // d_out = 4 — target is reachable by the low-rank adapter
                target_delta: (0..4).map(|j| fi * 0.1 + j as f32 * 0.02).collect(),
            }
        })
        .collect()
}

/// Capture the apply_lora output for layer 0 / q_proj given a fixed input.
fn capture_apply_output(adapter: &LoraAdapter) -> Vec<f32> {
    let layer = adapter
        .layers
        .get(&(0usize, "q_proj".to_string()))
        .expect("layer (0, q_proj) must exist");
    let scale = adapter.config.scale();
    let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.1 + 0.5).collect();
    let mut out = vec![0.0f32; 4];
    apply_lora(layer, scale, &x, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Test 1: train changes adapter weights and shifts apply_lora output
// ---------------------------------------------------------------------------

#[test]
fn test_train_and_apply() {
    let mut adapter = make_test_adapter();

    // Capture apply_lora output BEFORE training.
    let out_before = capture_apply_output(&adapter);

    // Train 20 epochs.
    let samples = make_train_samples(5);
    let config = LoraTrainConfig {
        learning_rate: 0.02,
        num_epochs: 20,
        batch_size: 5,
    };
    let result = train_lora(&mut adapter, &samples, &config).expect("train_lora must succeed");

    // Capture apply_lora output AFTER training.
    let out_after = capture_apply_output(&adapter);

    // Output must have changed — weights moved.
    assert_ne!(
        out_before, out_after,
        "apply_lora output must differ after training"
    );

    // Loss must have decreased over the run.
    let first = result.epoch_losses[0];
    let last = result.epoch_losses[result.epoch_losses.len() - 1];
    assert!(
        last < first,
        "final loss {last:.6} must be less than initial loss {first:.6}"
    );

    // Structural invariants on TrainResult.
    assert_eq!(result.epoch_losses.len(), 20);
    assert_eq!(result.total_steps, 20 * 5);
    assert_eq!(result.final_loss, last);
}

// ---------------------------------------------------------------------------
// Test 2: convergence quality — loss drops below threshold after 50 epochs
// ---------------------------------------------------------------------------

#[test]
fn test_train_convergence_quality() {
    let mut adapter = make_test_adapter();

    // Use a constant-target sample so convergence is tight.
    let samples = vec![TrainSample {
        layer_idx: 0,
        module: "q_proj".to_string(),
        // Fixed input
        input: vec![1.0f32; 8],
        // Fixed target that the adapter CAN reach
        target_delta: vec![0.0f32; 4],
    }];

    let config = LoraTrainConfig {
        learning_rate: 0.05,
        num_epochs: 50,
        batch_size: 1,
    };
    let result = train_lora(&mut adapter, &samples, &config).expect("train_lora must succeed");

    // Loss should fall substantially; exact threshold depends on scale.
    // With lr=0.05, 50 epochs, and zero target the loss converges fast.
    assert!(
        result.final_loss < 0.5,
        "final_loss {:.6} should be < 0.5 after 50 epochs",
        result.final_loss
    );

    // Epoch losses must be non-increasing (or at worst flat).
    for pair in result.epoch_losses.windows(2) {
        assert!(
            pair[1] <= pair[0] + 1e-6,
            "loss increased: epoch N={:.6}, epoch N+1={:.6}",
            pair[0],
            pair[1]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: train → save → load → apply gives identical output
// ---------------------------------------------------------------------------

#[cfg(feature = "safetensors")]
#[test]
fn test_save_load_roundtrip() {
    use lattice_tune::lora::{load_peft_safetensors, save_peft_safetensors};

    let mut adapter = make_test_adapter();

    // Train briefly so weights are non-trivial.
    let samples = make_train_samples(5);
    let config = LoraTrainConfig {
        learning_rate: 0.01,
        num_epochs: 10,
        batch_size: 5,
    };
    train_lora(&mut adapter, &samples, &config).expect("train_lora must succeed");

    // Capture apply_lora output of the trained adapter.
    let out_trained = capture_apply_output(&adapter);

    // Save to a temporary file.
    let dir = tempfile::tempdir().expect("tempdir must succeed");
    let path = dir.path().join("adapter_e2e.safetensors");
    adapter
        .save_safetensors(&path)
        .expect("save_safetensors must succeed");

    // Reload from disk.
    let loaded = LoraAdapter::from_safetensors(&path).expect("from_safetensors must succeed");

    // The loaded adapter must reproduce the same apply_lora output.
    let out_loaded = capture_apply_output(&loaded);

    assert_eq!(
        out_trained.len(),
        out_loaded.len(),
        "output length mismatch after round-trip"
    );
    for (i, (a, b)) in out_trained.iter().zip(out_loaded.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output[{i}] mismatch after round-trip: trained={a:.8} loaded={b:.8}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: error paths — empty samples and dimension mismatches
// ---------------------------------------------------------------------------

#[test]
fn test_train_error_empty_samples() {
    use lattice_tune::TuneError;

    let mut adapter = make_test_adapter();
    let config = LoraTrainConfig {
        learning_rate: 0.01,
        num_epochs: 5,
        batch_size: 1,
    };

    let err = train_lora(&mut adapter, &[], &config).expect_err("empty samples must return Err");
    assert!(
        matches!(err, TuneError::Training(_)),
        "expected Training variant, got {err:?}"
    );
}

#[test]
fn test_train_error_dimension_mismatch() {
    use lattice_tune::TuneError;

    let mut adapter = make_test_adapter();
    // d_in for (0, "q_proj") is 8; pass 3 elements.
    let bad_sample = TrainSample {
        layer_idx: 0,
        module: "q_proj".to_string(),
        input: vec![1.0, 2.0, 3.0], // wrong length
        target_delta: vec![0.0; 4],
    };
    let config = LoraTrainConfig {
        learning_rate: 0.01,
        num_epochs: 1,
        batch_size: 1,
    };

    let err = train_lora(&mut adapter, &[bad_sample], &config)
        .expect_err("dimension mismatch must return Err");
    assert!(
        matches!(err, TuneError::DimensionMismatch { .. }),
        "expected DimensionMismatch variant, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: adapter.apply() delegates to apply_lora correctly post-training
// ---------------------------------------------------------------------------

#[test]
fn test_adapter_apply_method_post_training() {
    let mut adapter = make_test_adapter();

    let samples = make_train_samples(5);
    let config = LoraTrainConfig {
        learning_rate: 0.01,
        num_epochs: 5,
        batch_size: 5,
    };
    train_lora(&mut adapter, &samples, &config).expect("train_lora must succeed");

    // adapter.apply() must agree with apply_lora on the same layer.
    let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.1 + 0.5).collect();

    let mut out_via_apply = vec![0.0f32; 4];
    adapter.apply(0, "q_proj", &x, &mut out_via_apply);

    let direct = capture_apply_output(&adapter);

    for (i, (a, b)) in out_via_apply.iter().zip(direct.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "apply() vs apply_lora mismatch at index {i}: {a:.8} vs {b:.8}"
        );
    }
}
