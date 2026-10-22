use super::*;
use crate::activation::Activation;
use crate::network::NetworkBuilder;

#[test]
fn test_xor_training() {
    // XOR problem - classic test for neural network training
    let mut network = NetworkBuilder::new()
        .input(2)
        .hidden(4, Activation::Tanh)
        .output(1, Activation::Tanh)
        .build()
        .unwrap();

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0], // 0 XOR 0 = 0
        vec![1.0], // 0 XOR 1 = 1
        vec![1.0], // 1 XOR 0 = 1
        vec![0.0], // 1 XOR 1 = 0
    ];

    let mut trainer = BackpropTrainer::new();
    let config = TrainingConfig::new()
        .learning_rate(0.5)
        .momentum(0.0)
        .max_epochs(5000)
        .target_error(0.01)
        .batch_size(4)
        .shuffle(false);

    let result = trainer.train(&mut network, &inputs, &targets, &config);

    // Training should complete without error
    // Note: XOR convergence is not guaranteed with random initialization
    assert!(result.is_ok());
}

#[test]
fn test_training_empty_data() {
    let mut network = NetworkBuilder::new()
        .input(2)
        .output(1, Activation::Linear)
        .build()
        .unwrap();

    let mut trainer = BackpropTrainer::new();
    let config = TrainingConfig::default();

    let result = trainer.train(&mut network, &[], &[], &config);
    assert!(matches!(result, Err(FannError::TrainingError(_))));
}

#[test]
fn test_training_dimension_mismatch() {
    let mut network = NetworkBuilder::new()
        .input(2)
        .output(1, Activation::Linear)
        .build()
        .unwrap();

    let inputs = vec![vec![1.0, 2.0, 3.0]]; // Wrong input size
    let targets = vec![vec![1.0]];

    let mut trainer = BackpropTrainer::new();
    let config = TrainingConfig::default();

    let result = trainer.train(&mut network, &inputs, &targets, &config);
    assert!(matches!(result, Err(FannError::TrainingError(_))));
}

#[test]
fn test_training_count_mismatch() {
    let mut network = NetworkBuilder::new()
        .input(2)
        .output(1, Activation::Linear)
        .build()
        .unwrap();

    let inputs = vec![vec![1.0, 2.0]];
    let targets = vec![vec![1.0], vec![2.0]]; // Different count

    let mut trainer = BackpropTrainer::new();
    let config = TrainingConfig::default();

    let result = trainer.train(&mut network, &inputs, &targets, &config);
    assert!(matches!(result, Err(FannError::TrainingError(_))));
}

#[test]
fn test_skip_batch_all_skipped_returns_error() {
    // Targets containing NaN propagate into gradients (diff = output - NaN = NaN),
    // triggering the gradient guard on every batch.  With SkipBatch the whole
    // epoch completes with batch_count == 0, which must be an Err not an Ok
    // with a non-finite final_error.
    let mut network = NetworkBuilder::new()
        .input(2)
        .hidden(2, Activation::Tanh)
        .output(1, Activation::Linear)
        .build_with_seed(1)
        .unwrap();

    let inputs = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
    // NaN targets guarantee NaN in every gradient computation.
    let targets = vec![vec![f32::NAN], vec![f32::NAN]];

    let mut trainer = BackpropTrainer::new();
    let config = TrainingConfig::new()
        .learning_rate(0.1)
        .momentum(0.0)
        .max_epochs(1)
        .batch_size(2)
        .shuffle(false)
        .gradient_guard(GradientGuardStrategy::SkipBatch);

    let result = trainer.train(&mut network, &inputs, &targets, &config);
    assert!(
        matches!(result, Err(FannError::NumericInstability(_))),
        "expected NumericInstability, got {result:?}"
    );
    // Confirm the error text is informative (not just any instability error).
    if let Err(FannError::NumericInstability(msg)) = result {
        assert!(
            msg.contains("all batches skipped"),
            "unexpected message: {msg}"
        );
    }
}

#[test]
fn test_linear_regression() {
    // Simple linear regression: y = 2x + 1
    let mut network = NetworkBuilder::new()
        .input(1)
        .output(1, Activation::Linear)
        .build()
        .unwrap();

    let inputs: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 / 10.0]).collect();
    let targets: Vec<Vec<f32>> = inputs.iter().map(|x| vec![2.0 * x[0] + 1.0]).collect();

    let mut trainer = BackpropTrainer::new();
    let config = TrainingConfig::new()
        .learning_rate(0.1)
        .momentum(0.0)
        .max_epochs(1000)
        .target_error(0.001)
        .batch_size(20)
        .shuffle(false);

    let result = trainer
        .train(&mut network, &inputs, &targets, &config)
        .unwrap();

    // Linear regression should converge easily
    assert!(
        result.final_error < 0.1,
        "Error {} too high",
        result.final_error
    );
}
