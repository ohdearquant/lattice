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
    let result = trainer.train_batch(&batch);

    assert!(result.is_ok());
    let loss = result.unwrap();
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
fn test_update_adam() {
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

    for _ in 0..3 {
        let batch = make_test_batch(2);
        let result = trainer.train_batch(&batch);
        assert!(
            result.is_ok(),
            "Adam training step failed: {:?}",
            result.err()
        );
    }

    assert_eq!(trainer.global_step(), 3);
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_update_adamw() {
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

    for _ in 0..3 {
        let batch = make_test_batch(2);
        let result = trainer.train_batch(&batch);
        assert!(
            result.is_ok(),
            "AdamW training step failed: {:?}",
            result.err()
        );
    }

    assert_eq!(trainer.global_step(), 3);
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_update_sgd_momentum() {
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

    for _ in 0..3 {
        let batch = make_test_batch(2);
        let result = trainer.train_batch(&batch);
        assert!(
            result.is_ok(),
            "SGD momentum training step failed: {:?}",
            result.err()
        );
    }

    assert_eq!(trainer.global_step(), 3);
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
fn test_update_sgd_plain() {
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

    for _ in 0..3 {
        let batch = make_test_batch(2);
        let result = trainer.train_batch(&batch);
        assert!(
            result.is_ok(),
            "Plain SGD training step failed: {:?}",
            result.err()
        );
    }
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
fn test_learning_rate_tracking() {
    if skip_if_no_gpu() {
        return;
    }

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

    let batch = make_test_batch(2);
    trainer.train_batch(&batch).unwrap();

    let updated_lr = trainer.current_lr();
    assert!(updated_lr > 0.0);
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
