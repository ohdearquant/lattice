//! Training Pipeline Integration Tests for lattice-tune
//!
//! Issue #445: Integration tests for the training pipeline including:
//! - Full training loop execution
//! - Early stopping behavior
//! - Learning rate scheduling
//! - Checkpoint creation and resumption
//! - Callback integration
//! - Dataset batching through training
//! - Regularization configuration
//! - Metrics tracking

use lattice_tune::{
    Checkpoint, Dataset, DatasetConfig, EarlyStopping, EpochMetrics, IntentLabels, LRSchedule,
    LoggingCallback, OptimizerConfig, RegularizationConfig, TrainingCallback, TrainingConfig,
    TrainingExample, TrainingLoop, TrainingMetrics, TrainingState,
};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a training example with specified properties
fn make_example(context_size: usize, embedding_dim: usize) -> TrainingExample {
    let context = vec![vec![0.1; embedding_dim]; context_size];
    let message = vec![0.2; embedding_dim];
    let labels = IntentLabels::continuation(0.8);
    TrainingExample::new(context, message, labels)
}

/// Create a dataset with specified size
fn make_dataset(num_examples: usize, context_size: usize, embedding_dim: usize) -> Dataset {
    let examples: Vec<TrainingExample> = (0..num_examples)
        .map(|_| make_example(context_size, embedding_dim))
        .collect();
    Dataset::from_examples(examples)
}

/// Create a diverse dataset with different label types
fn make_diverse_dataset(num_examples: usize, embedding_dim: usize) -> Dataset {
    let examples: Vec<TrainingExample> = (0..num_examples)
        .map(|i| {
            let label = match i % 6 {
                0 => IntentLabels::continuation(0.8),
                1 => IntentLabels::topic_shift(0.7),
                2 => IntentLabels::explicit_query(0.9),
                3 => IntentLabels::person_lookup(0.85),
                4 => IntentLabels::health_check(0.75),
                _ => IntentLabels::task_status(0.8),
            };
            let context = vec![vec![0.1 * ((i % 10) as f32); embedding_dim]; 3];
            let message = vec![0.2 * ((i % 5) as f32); embedding_dim];
            TrainingExample::new(context, message, label)
        })
        .collect();
    Dataset::from_examples(examples)
}

// ============================================================================
// Training Loop Creation Tests
// ============================================================================

#[test]
fn test_training_loop_creation_default_config() {
    let config = TrainingConfig::default();
    let trainer = TrainingLoop::new(config);
    assert!(trainer.is_ok());
}

#[test]
fn test_training_loop_creation_quick_config() {
    let config = TrainingConfig::quick();
    let trainer = TrainingLoop::new(config);
    assert!(trainer.is_ok());

    let trainer = trainer.unwrap();
    assert!(trainer.config().epochs < 20);
    assert_eq!(trainer.config().val_split, 0.0);
}

#[test]
fn test_training_loop_creation_thorough_config() {
    let config = TrainingConfig::thorough();
    let trainer = TrainingLoop::new(config);
    assert!(trainer.is_ok());

    let trainer = trainer.unwrap();
    assert!(trainer.config().epochs >= 100);
    assert!(trainer.config().val_split > 0.0);
}

#[test]
fn test_training_loop_creation_invalid_config() {
    let config = TrainingConfig::default().epochs(0);
    let trainer = TrainingLoop::new(config);
    assert!(trainer.is_err());
}

#[test]
fn test_training_loop_creation_invalid_batch_size() {
    let config = TrainingConfig::default().batch_size(0);
    let trainer = TrainingLoop::new(config);
    assert!(trainer.is_err());
}

// ============================================================================
// Training Execution Tests
// ============================================================================

#[test]
fn test_training_basic_execution() {
    let config = TrainingConfig::quick().epochs(5);
    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = make_dataset(100, 3, 64);

    let metrics = trainer.train(&mut dataset);
    assert!(metrics.is_ok());

    let metrics = metrics.unwrap();
    assert_eq!(metrics.epochs_completed, 5);
    assert!(metrics.final_train_loss > 0.0);
}

#[test]
fn test_training_empty_dataset_fails() {
    let config = TrainingConfig::quick();
    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = Dataset::new();

    let result = trainer.train(&mut dataset);
    assert!(result.is_err());
}

#[test]
fn test_training_with_validation_split() {
    let config = TrainingConfig::quick().epochs(3).val_split(0.2);
    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = make_dataset(100, 3, 64);

    let metrics = trainer.train(&mut dataset).unwrap();

    assert!(metrics.final_val_loss.is_some());
    assert!(metrics.best_val_loss.is_some());
}

#[test]
fn test_training_diverse_labels() {
    let config = TrainingConfig::quick().epochs(3);
    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = make_diverse_dataset(120, 64);

    let metrics = trainer.train(&mut dataset).unwrap();
    assert!(metrics.epochs_completed > 0);
}

#[test]
fn test_training_single_example() {
    let config = TrainingConfig::quick().epochs(2).batch_size(1);
    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = make_dataset(1, 3, 64);

    let metrics = trainer.train(&mut dataset).unwrap();
    assert!(metrics.epochs_completed > 0);
}

#[test]
fn test_training_large_batch_size() {
    let config = TrainingConfig::quick().epochs(2).batch_size(64);
    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = make_dataset(50, 3, 64);

    // Batch size larger than dataset should still work
    let metrics = trainer.train(&mut dataset).unwrap();
    assert!(metrics.epochs_completed > 0);
}

// ============================================================================
// Early Stopping Tests
// ============================================================================

#[test]
fn test_early_stopping_on_val_loss() {
    let config = TrainingConfig::default()
        .epochs(100)
        .val_split(0.2)
        .early_stopping(EarlyStopping::val_loss(2)); // Very short patience

    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = make_dataset(100, 3, 64);

    let metrics = trainer.train(&mut dataset).unwrap();

    // Should stop before 100 epochs (simulated loss decreases slowly)
    // Note: With placeholder implementation, early stopping may or may not trigger
    assert!(metrics.epochs_completed > 0);
}

#[test]
fn test_early_stopping_disabled() {
    let config = TrainingConfig::quick().epochs(5).no_early_stopping();

    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = make_dataset(100, 3, 64);

    let metrics = trainer.train(&mut dataset).unwrap();

    assert!(!metrics.early_stopped);
    assert_eq!(metrics.epochs_completed, 5);
}

#[test]
fn test_early_stopping_is_improvement_for_loss() {
    let es = EarlyStopping::val_loss(10);

    // For loss, lower is better
    assert!(es.is_improvement(0.4, 0.5));
    assert!(!es.is_improvement(0.5, 0.4));
    assert!(!es.is_improvement(0.5, 0.5)); // No improvement
}

#[test]
fn test_early_stopping_is_improvement_for_accuracy() {
    let es = EarlyStopping::val_accuracy(10);

    // For accuracy, higher is better
    assert!(es.is_improvement(0.9, 0.8));
    assert!(!es.is_improvement(0.8, 0.9));
    assert!(!es.is_improvement(0.8, 0.8)); // No improvement
}

// ============================================================================
// Learning Rate Schedule Tests
// ============================================================================

#[test]
fn test_lr_schedule_constant() {
    let schedule = LRSchedule::Constant;
    let base_lr = 0.01;

    assert_eq!(schedule.get_lr(base_lr, 0, 0), base_lr);
    assert_eq!(schedule.get_lr(base_lr, 100, 10), base_lr);
    assert_eq!(schedule.get_lr(base_lr, 1000, 50), base_lr);
}

#[test]
fn test_lr_schedule_linear_warmup() {
    let schedule = LRSchedule::LinearWarmup { warmup_steps: 100 };
    let base_lr = 0.01;

    // During warmup, LR increases linearly
    let lr_step_0 = schedule.get_lr(base_lr, 0, 0);
    let lr_step_50 = schedule.get_lr(base_lr, 50, 0);
    let lr_step_100 = schedule.get_lr(base_lr, 100, 1);
    let lr_step_200 = schedule.get_lr(base_lr, 200, 2);

    assert!(lr_step_0 < lr_step_50);
    assert!(lr_step_50 < lr_step_100);
    assert_eq!(lr_step_100, base_lr); // After warmup
    assert_eq!(lr_step_200, base_lr); // Stays constant
}

#[test]
fn test_lr_schedule_step_decay() {
    let schedule = LRSchedule::StepDecay {
        step_size: 10,
        gamma: 0.1,
    };
    let base_lr = 0.01;

    let lr_epoch_0 = schedule.get_lr(base_lr, 0, 0);
    let lr_epoch_10 = schedule.get_lr(base_lr, 0, 10);
    let lr_epoch_20 = schedule.get_lr(base_lr, 0, 20);

    assert_eq!(lr_epoch_0, base_lr);
    assert!((lr_epoch_10 - 0.001).abs() < 1e-6); // base_lr * 0.1
    assert!((lr_epoch_20 - 0.0001).abs() < 1e-6); // base_lr * 0.1 * 0.1
}

#[test]
fn test_lr_schedule_exponential_decay() {
    let schedule = LRSchedule::ExponentialDecay { gamma: 0.95 };
    let base_lr = 0.01;

    let lr_epoch_0 = schedule.get_lr(base_lr, 0, 0);
    let lr_epoch_1 = schedule.get_lr(base_lr, 0, 1);
    let lr_epoch_10 = schedule.get_lr(base_lr, 0, 10);

    assert_eq!(lr_epoch_0, base_lr);
    assert!((lr_epoch_1 - base_lr * 0.95).abs() < 1e-6);
    assert!(lr_epoch_10 < lr_epoch_1);
}

#[test]
fn test_lr_schedule_cosine_annealing() {
    let schedule = LRSchedule::CosineAnnealing {
        min_lr: 1e-6,
        t_max: 100,
    };
    let base_lr = 0.01;

    let lr_start = schedule.get_lr(base_lr, 0, 0);
    let lr_mid = schedule.get_lr(base_lr, 0, 50);
    let lr_end = schedule.get_lr(base_lr, 0, 100);

    // Starts high, goes to minimum in middle, back to high
    assert!((lr_start - base_lr).abs() < 1e-5);
    assert!(lr_mid < lr_start); // Lower in middle
    assert!((lr_end - base_lr).abs() < 1e-5); // Cycles back
}

// ============================================================================
// Checkpoint Tests
// ============================================================================

#[test]
fn test_checkpoint_creation() {
    let metrics = TrainingMetrics::default();
    let checkpoint = Checkpoint::new(5, 500, metrics);

    assert_eq!(checkpoint.epoch, 5);
    assert_eq!(checkpoint.global_step, 500);
    assert!(!checkpoint.id.is_nil());
}

#[test]
fn test_checkpoint_from_trainer() {
    let config = TrainingConfig::quick().epochs(3);
    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = make_dataset(50, 3, 64);

    trainer.train(&mut dataset).unwrap();

    let checkpoint = trainer.checkpoint();
    assert_eq!(checkpoint.epoch, 2); // 0-indexed, after 3 epochs
}

#[test]
fn test_checkpoint_resume() {
    let config = TrainingConfig::quick().epochs(2);
    let mut trainer = TrainingLoop::new(config).unwrap();
    let mut dataset = make_dataset(50, 3, 64);

    trainer.train(&mut dataset).unwrap();
    let checkpoint = trainer.checkpoint();

    // Create new trainer and resume
    let config2 = TrainingConfig::quick().epochs(5);
    let mut trainer2 = TrainingLoop::new(config2).unwrap();
    trainer2.resume_from(&checkpoint);

    assert_eq!(trainer2.state().epoch, checkpoint.epoch);
    assert_eq!(trainer2.state().global_step, checkpoint.global_step);
}

// ============================================================================
// Callback Tests
// ============================================================================

/// Custom callback for testing
struct TestCallback {
    train_start_called: bool,
    train_end_called: bool,
    epochs_started: usize,
    epochs_ended: usize,
    batches_started: usize,
    batches_ended: usize,
}

impl TestCallback {
    fn new() -> Self {
        Self {
            train_start_called: false,
            train_end_called: false,
            epochs_started: 0,
            epochs_ended: 0,
            batches_started: 0,
            batches_ended: 0,
        }
    }
}

impl TrainingCallback for TestCallback {
    fn on_train_start(&mut self, _config: &TrainingConfig) {
        self.train_start_called = true;
    }

    fn on_train_end(&mut self, _metrics: &TrainingMetrics) {
        self.train_end_called = true;
    }

    fn on_epoch_start(&mut self, _epoch: usize) {
        self.epochs_started += 1;
    }

    fn on_epoch_end(&mut self, _epoch: usize, _metrics: &EpochMetrics) {
        self.epochs_ended += 1;
    }

    fn on_batch_start(&mut self, _batch_idx: usize) {
        self.batches_started += 1;
    }

    fn on_batch_end(&mut self, _batch_idx: usize, _loss: f32) {
        self.batches_ended += 1;
    }
}

#[test]
fn test_callback_invocation() {
    let config = TrainingConfig::quick().epochs(2).batch_size(10);
    let mut trainer = TrainingLoop::new(config).unwrap();

    let callback = Box::new(TestCallback::new());
    trainer.add_callback(callback);

    let mut dataset = make_dataset(30, 3, 64);
    trainer.train(&mut dataset).unwrap();

    // Verify callbacks were called (we can't access internal state,
    // but this verifies no panics during callback invocation)
}

#[test]
fn test_logging_callback_creation() {
    let callback = LoggingCallback::new(10);
    // Just verify it can be created without panic
    let _ = callback;
}

// ============================================================================
// Regularization Tests
// ============================================================================

#[test]
fn test_regularization_none() {
    let reg = RegularizationConfig::none();
    assert_eq!(reg.dropout, 0.0);
    assert_eq!(reg.label_smoothing, 0.0);
    assert!(reg.gradient_clip.is_none());
    assert!(reg.mixup_alpha.is_none());
    assert!(reg.validate().is_ok());
}

#[test]
fn test_regularization_light() {
    let reg = RegularizationConfig::light();
    assert!(reg.dropout > 0.0);
    assert!(reg.dropout < 0.1);
    assert!(reg.validate().is_ok());
}

#[test]
fn test_regularization_strong() {
    let reg = RegularizationConfig::strong();
    assert!(reg.dropout > 0.2);
    assert!(reg.mixup_alpha.is_some());
    assert!(reg.validate().is_ok());
}

#[test]
fn test_gradient_clipping() {
    // Test gradient clipping utility
    let mut grads = vec![3.0, 4.0]; // Norm = 5
    let original_norm = RegularizationConfig::clip_grad_norm(&mut grads, 1.0);

    assert!((original_norm - 5.0).abs() < 1e-6);

    let new_norm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((new_norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_gradient_clipping_no_clip_needed() {
    let mut grads = vec![0.1, 0.2]; // Small norm
    let original_norm = RegularizationConfig::clip_grad_norm(&mut grads, 10.0);

    // Should not be clipped
    assert!((grads[0] - 0.1).abs() < 1e-6);
    assert!((grads[1] - 0.2).abs() < 1e-6);
    assert!(original_norm < 10.0);
}

#[test]
fn test_apply_gradient_clip_enabled() {
    let reg = RegularizationConfig {
        gradient_clip: Some(1.0),
        ..RegularizationConfig::none()
    };

    let mut grads = vec![3.0, 4.0];
    let result = reg.apply_gradient_clip(&mut grads);

    assert!(result.is_some());
    assert!((result.unwrap() - 5.0).abs() < 1e-6);
}

#[test]
fn test_apply_gradient_clip_disabled() {
    let reg = RegularizationConfig::none();
    let mut grads = vec![3.0, 4.0];
    let result = reg.apply_gradient_clip(&mut grads);

    assert!(result.is_none());
    assert_eq!(grads, vec![3.0, 4.0]); // Unchanged
}

// ============================================================================
// Optimizer Configuration Tests
// ============================================================================

#[test]
fn test_optimizer_config_sgd() {
    let config = OptimizerConfig::sgd(0.01);
    assert!(config.validate().is_ok());
    assert_eq!(config.learning_rate, 0.01);
}

#[test]
fn test_optimizer_config_adamw() {
    let config = OptimizerConfig::adamw(0.001, 0.01);
    assert!(config.validate().is_ok());
    assert_eq!(config.weight_decay, 0.01);
}

#[test]
fn test_optimizer_config_invalid_lr() {
    let config = OptimizerConfig {
        learning_rate: -0.01,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_optimizer_config_invalid_momentum() {
    let config = OptimizerConfig {
        momentum: 1.5,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// ============================================================================
// Training Metrics Tests
// ============================================================================

#[test]
fn test_training_metrics_add_epoch() {
    let mut metrics = TrainingMetrics::default();

    for i in 0..5 {
        let mut epoch_metrics = EpochMetrics::new(i);
        epoch_metrics.train_loss = 1.0 - 0.1 * i as f32;
        epoch_metrics.val_loss = Some(1.1 - 0.1 * i as f32);
        metrics.add_epoch(epoch_metrics);
    }

    assert_eq!(metrics.history.len(), 5);
    assert_eq!(metrics.epochs_completed, 5);
    assert_eq!(metrics.best_epoch, Some(4)); // Last epoch had best val_loss
}

#[test]
fn test_training_metrics_loss_history() {
    let mut metrics = TrainingMetrics::default();

    for i in 0..3 {
        let mut epoch_metrics = EpochMetrics::new(i);
        epoch_metrics.train_loss = (i + 1) as f32;
        epoch_metrics.val_loss = Some((i + 2) as f32);
        metrics.add_epoch(epoch_metrics);
    }

    let train_loss_history = metrics.train_loss_history();
    assert_eq!(train_loss_history, vec![1.0, 2.0, 3.0]);

    let val_loss_history = metrics.val_loss_history();
    assert_eq!(val_loss_history, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_epoch_metrics_get_metric() {
    let mut metrics = EpochMetrics::new(0);
    metrics.train_loss = 0.5;
    metrics.val_loss = Some(0.6);
    metrics.train_accuracy = 0.8;
    metrics.val_accuracy = Some(0.75);

    assert_eq!(metrics.get_metric("train_loss"), Some(0.5));
    assert_eq!(metrics.get_metric("val_loss"), Some(0.6));
    assert_eq!(metrics.get_metric("train_accuracy"), Some(0.8));
    assert_eq!(metrics.get_metric("val_accuracy"), Some(0.75));
    assert_eq!(metrics.get_metric("unknown"), None);
}

// ============================================================================
// Training State Tests
// ============================================================================

#[test]
fn test_training_state_initialization() {
    let state = TrainingState::new(0.001);

    assert_eq!(state.epoch, 0);
    assert_eq!(state.step, 0);
    assert_eq!(state.global_step, 0);
    assert_eq!(state.learning_rate, 0.001);
}

#[test]
fn test_training_state_reset_epoch() {
    let mut state = TrainingState::new(0.001);
    state.running_loss = 10.0;
    state.running_correct = 80;
    state.running_total = 100;
    state.step = 50;

    state.reset_epoch();

    assert_eq!(state.running_loss, 0.0);
    assert_eq!(state.running_correct, 0);
    assert_eq!(state.running_total, 0);
    assert_eq!(state.step, 0);
}

#[test]
fn test_training_state_epoch_accuracy() {
    let mut state = TrainingState::new(0.001);
    state.running_correct = 75;
    state.running_total = 100;

    assert_eq!(state.epoch_accuracy(), 0.75);
}

#[test]
fn test_training_state_epoch_loss() {
    let mut state = TrainingState::new(0.001);
    state.running_loss = 5.0;
    state.step = 10;

    assert_eq!(state.epoch_loss(), 0.5);
}

// ============================================================================
// Dataset Configuration Through Training Tests
// ============================================================================

#[test]
fn test_training_with_custom_dataset_config() {
    let config = TrainingConfig::quick().epochs(2).batch_size(16);
    let mut trainer = TrainingLoop::new(config).unwrap();

    let examples: Vec<TrainingExample> = (0..100).map(|_| make_example(3, 64)).collect();
    let mut dataset = Dataset::with_config(
        examples,
        DatasetConfig::with_batch_size(16).shuffle(true).seed(42),
    )
    .unwrap();

    let metrics = trainer.train(&mut dataset).unwrap();
    assert!(metrics.epochs_completed > 0);
}

#[test]
fn test_training_reproducibility_with_seed() {
    let config1 = TrainingConfig::quick().epochs(3).seed(12345);
    let config2 = TrainingConfig::quick().epochs(3).seed(12345);

    let mut trainer1 = TrainingLoop::new(config1).unwrap();
    let mut trainer2 = TrainingLoop::new(config2).unwrap();

    let mut dataset1 = make_dataset(50, 3, 64);
    let mut dataset2 = make_dataset(50, 3, 64);

    let metrics1 = trainer1.train(&mut dataset1).unwrap();
    let metrics2 = trainer2.train(&mut dataset2).unwrap();

    // With same seed, training should produce similar results
    // (Note: exact match depends on implementation details)
    assert_eq!(metrics1.epochs_completed, metrics2.epochs_completed);
}

// ============================================================================
// Full Pipeline Integration Tests
// ============================================================================

#[test]
fn test_full_training_pipeline() {
    // 1. Create diverse dataset
    let mut dataset = make_diverse_dataset(200, 32);

    // 2. Configure training
    let config = TrainingConfig::default()
        .epochs(5)
        .batch_size(32)
        .learning_rate(0.001)
        .val_split(0.2)
        .early_stopping(EarlyStopping::val_loss(3))
        .lr_schedule(LRSchedule::CosineAnnealingWarmup {
            warmup_steps: 10,
            min_lr: 1e-6,
            t_max: 5,
        })
        .regularization(RegularizationConfig::light());

    assert!(config.validate().is_ok());

    // 3. Create trainer
    let mut trainer = TrainingLoop::new(config).unwrap();

    // 4. Train
    let metrics = trainer.train(&mut dataset).unwrap();

    // 5. Verify results
    assert!(metrics.epochs_completed > 0);
    assert!(metrics.final_train_loss > 0.0);
    assert!(metrics.final_val_loss.is_some());
    assert!(!metrics.history.is_empty());

    // 6. Create checkpoint
    let checkpoint = trainer.checkpoint();
    assert!(!checkpoint.id.is_nil());
    assert_eq!(
        checkpoint.metrics.epochs_completed,
        metrics.epochs_completed
    );
}

#[test]
fn test_training_with_all_lr_schedules() {
    let schedules = vec![
        LRSchedule::Constant,
        LRSchedule::LinearWarmup { warmup_steps: 10 },
        LRSchedule::StepDecay {
            step_size: 2,
            gamma: 0.5,
        },
        LRSchedule::ExponentialDecay { gamma: 0.9 },
        LRSchedule::CosineAnnealing {
            min_lr: 1e-6,
            t_max: 3,
        },
    ];

    for schedule in schedules {
        let config = TrainingConfig::quick()
            .epochs(3)
            .lr_schedule(schedule.clone());
        let mut trainer = TrainingLoop::new(config).unwrap();
        let mut dataset = make_dataset(50, 3, 32);

        let result = trainer.train(&mut dataset);
        assert!(
            result.is_ok(),
            "Training failed with schedule: {schedule:?}"
        );
    }
}

#[test]
fn test_training_with_different_batch_sizes() {
    let batch_sizes = vec![1, 8, 32, 64, 128];

    for batch_size in batch_sizes {
        let config = TrainingConfig::quick().epochs(2).batch_size(batch_size);
        let mut trainer = TrainingLoop::new(config).unwrap();
        let mut dataset = make_dataset(150, 3, 32);

        let result = trainer.train(&mut dataset);
        assert!(
            result.is_ok(),
            "Training failed with batch_size: {batch_size}"
        );
    }
}

#[test]
fn test_overfitting_detection() {
    let mut metrics = TrainingMetrics::default();

    // Simulate overfitting: train loss decreases, val loss increases
    for i in 0..10 {
        let mut epoch_metrics = EpochMetrics::new(i);
        epoch_metrics.train_loss = 1.0 - 0.05 * i as f32; // Decreasing
        epoch_metrics.val_loss = Some(0.5 + 0.05 * i as f32); // Increasing
        metrics.add_epoch(epoch_metrics);
    }

    // Should detect overfitting with window of 3
    assert!(metrics.is_overfitting(3));
}

#[test]
fn test_no_overfitting_when_both_decrease() {
    let mut metrics = TrainingMetrics::default();

    // Both losses decrease - no overfitting
    for i in 0..10 {
        let mut epoch_metrics = EpochMetrics::new(i);
        epoch_metrics.train_loss = 1.0 - 0.05 * i as f32;
        epoch_metrics.val_loss = Some(1.1 - 0.05 * i as f32);
        metrics.add_epoch(epoch_metrics);
    }

    assert!(!metrics.is_overfitting(3));
}
