use super::*;

#[test]
fn test_shadow_config_default() {
    let config = ShadowConfig::default();
    assert_eq!(config.sample_rate, 0.1);
    assert_eq!(config.min_samples, 1000);
    assert_eq!(config.min_agreement, 0.95);
    assert_eq!(config.max_latency_increase_ms, 50.0);
    assert!(config.validate().is_ok());
}

#[test]
fn test_shadow_config_quick() {
    let config = ShadowConfig::quick();
    assert_eq!(config.min_samples, 100);
    assert!(config.validate().is_ok());
}

#[test]
fn test_shadow_config_strict() {
    let config = ShadowConfig::strict();
    assert_eq!(config.min_samples, 10000);
    assert_eq!(config.min_agreement, 0.99);
    assert!(config.validate().is_ok());
}

#[test]
fn test_shadow_config_validation() {
    let mut config = ShadowConfig::default();

    config.sample_rate = 1.5;
    assert!(config.validate().is_err());

    config.sample_rate = 0.1;
    config.min_samples = 0;
    assert!(config.validate().is_err());

    config.min_samples = 100;
    config.min_agreement = 1.5;
    assert!(config.validate().is_err());

    config.min_agreement = 0.95;
    config.max_latency_increase_ms = -10.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_shadow_session_creation() {
    let prod_id = Uuid::new_v4();
    let cand_id = Uuid::new_v4();
    let config = ShadowConfig::quick();

    let session = ShadowSession::new(prod_id, cand_id, config);

    assert_eq!(session.production_model_id, prod_id);
    assert_eq!(session.candidate_model_id, cand_id);
    assert!(session.is_running());
    assert!(!session.passed());
    assert!(!session.failed());
    assert_eq!(session.sample_count(), 0);
}

#[test]
fn test_shadow_session_record_sample() {
    let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), ShadowConfig::quick());

    session.record_sample(true, 5.0);
    session.record_sample(false, 10.0);
    session.record_sample(true, -2.0);

    assert_eq!(session.sample_count(), 3);

    let comparison = session.current_comparison();
    assert_eq!(comparison.sample_count, 3);
    // 2 out of 3 agreed
    assert!((comparison.agreement_rate - 2.0 / 3.0).abs() < 0.001);
    // Average latency: (5 + 10 + (-2)) / 3 = 13/3 = 4.333...
    assert!((comparison.latency_diff_ms - 13.0 / 3.0).abs() < 0.001);
}

#[test]
fn test_shadow_session_evaluate_pass() {
    let config = ShadowConfig {
        sample_rate: 0.1,
        min_samples: 10,
        min_agreement: 0.90,
        max_latency_increase_ms: 50.0,
    };

    let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

    // Add 10 samples, all agree, low latency
    for _ in 0..10 {
        session.record_sample(true, 5.0);
    }

    let state = session.evaluate();
    assert!(matches!(state, ShadowState::Passed { .. }));
    assert!(session.passed());
}

#[test]
fn test_shadow_session_evaluate_fail_agreement() {
    let config = ShadowConfig {
        sample_rate: 0.1,
        min_samples: 10,
        min_agreement: 0.90,
        max_latency_increase_ms: 50.0,
    };

    let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

    // Add 10 samples, only 5 agree (50% < 90%)
    for i in 0..10 {
        session.record_sample(i < 5, 5.0);
    }

    session.evaluate();
    assert!(session.failed());

    if let ShadowState::Failed { reason, .. } = session.state() {
        assert!(reason.contains("Agreement rate"));
    } else {
        panic!("Expected Failed state");
    }
}

#[test]
fn test_shadow_session_evaluate_fail_latency() {
    let config = ShadowConfig {
        sample_rate: 0.1,
        min_samples: 10,
        min_agreement: 0.90,
        max_latency_increase_ms: 10.0,
    };

    let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

    // Add 10 samples, all agree but high latency
    for _ in 0..10 {
        session.record_sample(true, 50.0); // 50ms > 10ms threshold
    }

    session.evaluate();
    assert!(session.failed());

    if let ShadowState::Failed { reason, .. } = session.state() {
        assert!(reason.contains("Latency increase"));
    } else {
        panic!("Expected Failed state");
    }
}

#[test]
fn test_shadow_session_evaluate_not_enough_samples() {
    let config = ShadowConfig {
        sample_rate: 0.1,
        min_samples: 100,
        min_agreement: 0.90,
        max_latency_increase_ms: 50.0,
    };

    let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

    // Add only 50 samples
    for _ in 0..50 {
        session.record_sample(true, 5.0);
    }

    session.evaluate();
    // Should still be running
    assert!(session.is_running());
    assert!(!session.passed());
    assert!(!session.failed());
}

#[test]
fn test_shadow_session_cancel() {
    let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), ShadowConfig::quick());

    session.record_sample(true, 5.0);
    session.cancel("Manual cancellation for testing");

    assert!(!session.is_running());
    assert!(session.is_complete());
    assert!(!session.passed());

    if let ShadowState::Cancelled { reason, .. } = session.state() {
        assert_eq!(reason, "Manual cancellation for testing");
    } else {
        panic!("Expected Cancelled state");
    }
}

#[test]
fn test_shadow_session_no_samples_after_cancel() {
    let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), ShadowConfig::quick());

    session.record_sample(true, 5.0);
    assert_eq!(session.sample_count(), 1);

    session.cancel("done");

    // Should not accept more samples
    session.record_sample(true, 5.0);
    assert_eq!(session.sample_count(), 1);
}

#[test]
fn test_shadow_session_progress() {
    let config = ShadowConfig {
        sample_rate: 0.1,
        min_samples: 100,
        min_agreement: 0.90,
        max_latency_increase_ms: 50.0,
    };

    let mut session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), config);

    assert_eq!(session.progress(), 0.0);

    for _ in 0..50 {
        session.record_sample(true, 5.0);
    }
    assert!((session.progress() - 0.5).abs() < 0.001);

    for _ in 0..50 {
        session.record_sample(true, 5.0);
    }
    assert!((session.progress() - 1.0).abs() < 0.001);

    // Progress caps at 1.0
    for _ in 0..50 {
        session.record_sample(true, 5.0);
    }
    assert!((session.progress() - 1.0).abs() < 0.001);
}

#[test]
fn test_shadow_comparison_empty() {
    let session = ShadowSession::new(Uuid::new_v4(), Uuid::new_v4(), ShadowConfig::quick());

    let comparison = session.current_comparison();
    assert_eq!(comparison.sample_count, 0);
    assert_eq!(comparison.agreement_rate, 0.0);
    assert_eq!(comparison.latency_diff_ms, 0.0);
}

#[test]
fn test_shadow_session_with_id() {
    let session_id = Uuid::new_v4();
    let session = ShadowSession::with_id(
        session_id,
        Uuid::new_v4(),
        Uuid::new_v4(),
        ShadowConfig::quick(),
    );

    assert_eq!(session.id, session_id);
}
