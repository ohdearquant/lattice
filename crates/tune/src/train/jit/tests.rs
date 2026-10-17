use super::*;
use crate::data::IntentLabels;

fn make_examples(count: usize) -> Vec<TrainingExample> {
    (0..count)
        .map(|_| {
            TrainingExample::new(
                vec![vec![0.1, 0.2, 0.3]],
                vec![0.4, 0.5, 0.6],
                IntentLabels::continuation(0.8),
            )
        })
        .collect()
}

#[test]
fn test_jit_config_defaults() {
    let config = JitConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.steps, 100);
}

#[test]
fn test_jit_config_presets() {
    let fast = JitConfig::fast();
    assert!(fast.validate().is_ok());
    assert!(fast.timeout_secs <= 5.0);

    let thorough = JitConfig::thorough();
    assert!(thorough.validate().is_ok());
    assert!(thorough.steps > fast.steps);
}

#[test]
fn test_jit_adapter_creation() {
    let adapter = JitAdapter::new();
    assert_eq!(adapter.config().steps, 100);

    let adapter = JitAdapter::with_config(JitConfig::fast());
    assert!(adapter.is_ok());
}

#[test]
fn test_jit_low_rank_adapt_cpu_fails_closed() {
    let adapter = JitAdapter::with_config(JitConfig {
        strategy: JitStrategy::LowRank { rank: 4 },
        steps: 10,
        timeout_secs: 1.0,
        ..JitConfig::fast()
    })
    .unwrap();

    let examples = make_examples(20);
    let error = adapter.adapt_cpu(examples).unwrap_err();

    assert!(matches!(
        error,
        TuneError::Training(message) if message.contains("not implemented")
    ));
}

#[test]
fn test_jit_result_metrics() {
    let result = JitResult {
        final_loss: 0.5,
        final_accuracy: Some(0.85),
        steps_completed: 50,
        time_secs: 3.5,
        early_stopped: true,
        timed_out: false,
        loss_history: vec![1.0, 0.8, 0.6, 0.5],
    };

    assert!(result.is_success());
    assert!((result.avg_loss() - 0.725).abs() < 0.01);
}

#[test]
fn test_freeze_last_n_layers() {
    let frozen = freeze::get_frozen_layers(&JitStrategy::LastNLayers(2), 5);
    assert_eq!(frozen, vec![true, true, true, false, false]);
}

#[test]
fn test_freeze_head_only() {
    let frozen = freeze::get_frozen_layers(&JitStrategy::HeadOnly, 4);
    assert_eq!(frozen, vec![true, true, true, false]);
}

#[test]
fn test_freeze_full() {
    let frozen = freeze::get_frozen_layers(&JitStrategy::Full, 3);
    assert_eq!(frozen, vec![false, false, false]);
}

#[test]
fn test_lr_multipliers_gradual() {
    let multipliers = freeze::get_lr_multipliers(&JitStrategy::GradualUnfreeze, 4, 0.1);
    assert_eq!(multipliers.len(), 4);
    assert!((multipliers[0] - 0.1).abs() < 0.01); // First layer
    assert!((multipliers[3] - 1.0).abs() < 0.01); // Last layer
}
