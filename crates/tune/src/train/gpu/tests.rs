use super::*;
use crate::data::{IntentLabels, TrainingExample};
use lattice_fann::Activation;

fn skip_if_no_gpu() -> bool {
    !lattice_fann::gpu::is_gpu_available()
}

fn make_test_example() -> TrainingExample {
    TrainingExample::new(
        vec![vec![0.1, 0.2, 0.3]],
        vec![0.4, 0.5, 0.6],
        IntentLabels::continuation(0.8),
    )
}

fn make_test_batch(n: usize) -> Batch {
    let examples: Vec<_> = (0..n).map(|_| make_test_example()).collect();
    Batch::from_examples(examples, 0)
}

#[test]
fn test_gpu_trainer_builder() {
    if skip_if_no_gpu() {
        println!("Skipping GPU test - no GPU available");
        return;
    }

    let trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(32, Activation::ReLU)
        .hidden(16, Activation::ReLU)
        .config(TrainingConfig::quick())
        .build();

    assert!(trainer.is_ok());
}

#[test]
fn test_gpu_trainer_forward() {
    if skip_if_no_gpu() {
        return;
    }

    // #797: every GpuOptimizer arm (Adam/AdamW/SGDMomentum/SGD/RMSprop) now
    // fails loudly instead of performing a silent no-op update, so the full
    // `train_batch` pipeline (forward -> backward -> update_weights -> LR
    // step) can never complete today regardless of optimizer choice. This
    // test's actual subject is forward-pass + loss computation, not the
    // optimizer, so it calls `forward_batch`/`compute_loss` directly —
    // bypassing `update_weights` entirely — to keep verifying exactly what
    // it verified before, without depending on optimizer behavior that is
    // honestly unimplemented.
    let mut trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(16, Activation::ReLU)
        .config(TrainingConfig::quick())
        .build()
        .unwrap();

    let example = TrainingExample::new(
        vec![vec![0.1, 0.2, 0.3]],
        vec![0.4, 0.5, 0.6],
        IntentLabels::continuation(0.8),
    );

    let batch = Batch::from_examples(vec![example], 0);

    let (outputs, _activations) = trainer
        .forward_batch(&batch)
        .expect("forward pass should succeed");
    let loss = trainer
        .compute_loss(&outputs, &batch)
        .expect("loss computation should succeed");

    assert!(loss > 0.0);
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_validate_accuracy() {
    if skip_if_no_gpu() {
        return;
    }

    let mut trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(16, Activation::ReLU)
        .config(TrainingConfig::quick())
        .build()
        .unwrap();

    let examples: Vec<_> = (0..5).map(|_| make_test_example()).collect();
    let mut dataset = Dataset::from_examples(examples);

    let result = trainer.validate(&mut dataset);
    assert!(result.is_ok());

    let (loss, accuracy) = result.unwrap();
    assert!(loss >= 0.0, "Loss should be non-negative: {loss}");
    assert!(
        (0.0..=1.0).contains(&accuracy),
        "Accuracy should be in [0,1]: {accuracy}"
    );
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_validate_empty_dataset() {
    if skip_if_no_gpu() {
        return;
    }

    let mut trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(16, Activation::ReLU)
        .config(TrainingConfig::quick())
        .build()
        .unwrap();

    let mut dataset = Dataset::from_examples(vec![]);
    let result = trainer.validate(&mut dataset);
    assert!(result.is_ok());

    let (loss, accuracy) = result.unwrap();
    assert_eq!(loss, 0.0);
    assert_eq!(accuracy, 0.0);
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_update_adam_fails_loud() {
    if skip_if_no_gpu() {
        return;
    }

    let config = TrainingConfig {
        optimizer: OptimizerConfig {
            optimizer: Optimizer::Adam,
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            ..Default::default()
        },
        ..TrainingConfig::quick()
    };

    let mut trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(16, Activation::ReLU)
        .config(config)
        .build()
        .unwrap();

    let batch = make_test_batch(2);
    let result = trainer.train_batch(&batch);

    // #797: the GPU Adam optimizer dispatch has no buffer bindings wired —
    // it must fail loudly rather than silently reporting a successful
    // zero-effect update. If this assertion ever fails because the empty
    // command-buffer no-op was restored, that is the regression this test
    // guards against.
    let err = result.expect_err("Adam GPU optimizer must fail until buffer bindings are wired");
    let msg = err.to_string();
    assert!(
        msg.contains("Adam") && msg.contains("not implemented"),
        "unexpected error message: {msg}"
    );
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_update_adamw_fails_loud() {
    if skip_if_no_gpu() {
        return;
    }

    let config = TrainingConfig {
        optimizer: OptimizerConfig {
            optimizer: Optimizer::AdamW,
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            ..Default::default()
        },
        ..TrainingConfig::quick()
    };

    let mut trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(16, Activation::ReLU)
        .config(config)
        .build()
        .unwrap();

    let batch = make_test_batch(2);
    let result = trainer.train_batch(&batch);

    let err = result.expect_err("AdamW GPU optimizer must fail until buffer bindings are wired");
    let msg = err.to_string();
    assert!(
        msg.contains("AdamW") && msg.contains("not implemented"),
        "unexpected error message: {msg}"
    );
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_update_sgd_momentum_fails_loud() {
    if skip_if_no_gpu() {
        return;
    }

    let config = TrainingConfig {
        optimizer: OptimizerConfig {
            optimizer: Optimizer::SGDMomentum,
            learning_rate: 0.01,
            momentum: 0.9,
            ..Default::default()
        },
        ..TrainingConfig::quick()
    };

    let mut trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(16, Activation::ReLU)
        .config(config)
        .build()
        .unwrap();

    let batch = make_test_batch(2);
    let result = trainer.train_batch(&batch);

    let err =
        result.expect_err("SGD-momentum GPU optimizer must fail until buffer bindings are wired");
    let msg = err.to_string();
    assert!(
        msg.contains("SGD-momentum") && msg.contains("not implemented"),
        "unexpected error message: {msg}"
    );
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_update_rmsprop_fails_loud() {
    if skip_if_no_gpu() {
        return;
    }

    let config = TrainingConfig {
        optimizer: OptimizerConfig {
            optimizer: Optimizer::RMSprop,
            learning_rate: 0.01,
            ..Default::default()
        },
        ..TrainingConfig::quick()
    };

    let mut trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(16, Activation::ReLU)
        .config(config)
        .build()
        .unwrap();

    let batch = make_test_batch(2);
    let result = trainer.train_batch(&batch);

    // #797 adjacent defect: RMSprop used to silently substitute plain SGD
    // instead of running the requested algorithm. It must fail loudly and
    // name the alternative instead.
    let err = result.expect_err("RMSprop GPU optimizer must fail loudly, not silently fall back");
    let msg = err.to_string();
    assert!(
        msg.contains("RMSprop") && msg.contains("not implemented"),
        "unexpected error message: {msg}"
    );
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_update_sgd_fails_loud() {
    if skip_if_no_gpu() {
        return;
    }

    let config = TrainingConfig {
        optimizer: OptimizerConfig {
            optimizer: Optimizer::SGD,
            learning_rate: 0.01,
            ..Default::default()
        },
        ..TrainingConfig::quick()
    };

    let mut trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(8, Activation::ReLU)
        .config(config)
        .build()
        .unwrap();

    // #797: plain SGD is not real SGD either — its previous body used a
    // constant placeholder gradient magnitude (never the actual per-layer
    // gradients) and had no mutable weight write-back path from GpuNetwork,
    // so any computed values were always discarded. It must fail loudly
    // like every other GpuOptimizer arm rather than report success for a
    // step that changed nothing.
    let batch = make_test_batch(2);
    let result = trainer.train_batch(&batch);

    let err = result.expect_err("plain SGD GPU optimizer must fail until it is real SGD");
    let msg = err.to_string();
    assert!(
        msg.contains("SGD") && msg.contains("not implemented"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn test_check_numeric_stability_valid() {
    let outputs = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    assert!(GpuTrainer::check_numeric_stability(&outputs).is_ok());
}

#[test]
fn test_check_numeric_stability_nan() {
    let outputs = vec![vec![0.1, f32::NAN, 0.3]];
    let result = GpuTrainer::check_numeric_stability(&outputs);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("NaN"));
}

#[test]
fn test_check_numeric_stability_inf() {
    let outputs = vec![vec![f32::INFINITY, 0.2, 0.3]];
    let result = GpuTrainer::check_numeric_stability(&outputs);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Inf"));
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_failed_step_preserves_lr_and_global_step() {
    if skip_if_no_gpu() {
        return;
    }

    // #797: every GpuOptimizer arm fails loudly now (none is real yet), so
    // `train_batch` cannot reach the LR-schedule update line (it runs after
    // `update_weights()?` in `train_batch`, and `?` returns early on error
    // regardless of which optimizer is configured). This test's original
    // assertion — that `current_lr()` reflects the schedule after a batch —
    // cannot be verified honestly until a real optimizer arm lands, so
    // instead of dodging via a "harmless" optimizer choice (there isn't
    // one), this test now asserts what *is* true today: a failed optimizer
    // step propagates the error and does not silently advance the learning
    // rate OR the public global-step counter. Both are real,
    // currently-meaningful invariants (no partial/silent state drift on
    // failure), not a weakened stand-in for LR/step tracking.
    let config = TrainingConfig {
        optimizer: OptimizerConfig {
            optimizer: Optimizer::Adam,
            learning_rate: 0.001,
            ..Default::default()
        },
        ..TrainingConfig::quick()
    };

    let mut trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(8, Activation::ReLU)
        .config(config)
        .build()
        .unwrap();

    let initial_lr = trainer.current_lr();
    assert!((initial_lr - 0.001).abs() < 1e-6);
    assert_eq!(trainer.global_step(), 0);

    let batch = make_test_batch(2);
    let result = trainer.train_batch(&batch);
    assert!(
        result.is_err(),
        "train_batch must fail until a real GPU optimizer arm lands (#797)"
    );

    // current_lr must be untouched: the LR-schedule assignment in
    // `train_batch` runs strictly after the optimizer-update `?`, so a
    // failed update must never advance it.
    let lr_after_failed_step = trainer.current_lr();
    assert!(
        (lr_after_failed_step - initial_lr).abs() < 1e-9,
        "learning rate must not change on a failed optimizer step: {initial_lr} -> {lr_after_failed_step}"
    );

    // global_step must also be untouched: it is now incremented after
    // `update_weights()?` succeeds, so a failed optimizer step must not
    // count as a completed training step (it previously did, feeding
    // incorrect values into LR/epoch math on every failed call).
    assert_eq!(
        trainer.global_step(),
        0,
        "a failed optimizer step must not advance global_step"
    );
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_device_info() {
    if skip_if_no_gpu() {
        return;
    }

    let trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(8, Activation::ReLU)
        .config(TrainingConfig::quick())
        .build()
        .unwrap();

    let info = trainer.device_info();
    assert!(!info.is_empty());
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_is_using_gpu() {
    if skip_if_no_gpu() {
        return;
    }

    let trainer = GpuTrainerBuilder::new(6, 6)
        .hidden(8, Activation::ReLU)
        .config(TrainingConfig::quick())
        .build()
        .unwrap();

    assert!(trainer.is_using_gpu());
}
