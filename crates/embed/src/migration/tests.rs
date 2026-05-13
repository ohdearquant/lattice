//! Tests for the migration module.

use crate::migration::{
    MigrationController, MigrationError, MigrationPlan, MigrationProgress, MigrationState,
    SkipReason,
};
use crate::model::EmbeddingModel;

fn test_plan() -> MigrationPlan {
    MigrationPlan {
        id: "test-migration-001".to_string(),
        source_model: EmbeddingModel::BgeSmallEnV15,
        target_model: EmbeddingModel::BgeBaseEnV15,
        total_embeddings: 1000,
        batch_size: 100,
        created_at: "2026-01-27T00:00:00Z".to_string(),
    }
}

#[test]
fn test_initial_state() {
    let ctrl = MigrationController::new(test_plan());
    assert_eq!(*ctrl.state(), MigrationState::Planned);
    assert_eq!(ctrl.state().progress(), Some(0.0));
    assert!(!ctrl.state().is_terminal());
    assert!(!ctrl.state().is_active());
}

#[test]
fn test_start_transition() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    assert!(ctrl.state().is_active());
    assert_eq!(ctrl.state().processed(), 0);
}

#[test]
fn test_progress_tracking() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(250).unwrap();
    assert_eq!(ctrl.state().processed(), 250);
    assert!((ctrl.state().progress().unwrap() - 0.25).abs() < f64::EPSILON);
}

#[test]
fn test_auto_complete() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(1000).unwrap();
    assert!(ctrl.state().is_terminal());
    assert_eq!(ctrl.state().progress(), Some(1.0));
}

#[test]
fn test_auto_complete_overshoot() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(1500).unwrap();
    assert!(ctrl.state().is_terminal());
    assert_eq!(ctrl.state().processed(), 1500);
}

#[test]
fn test_pause_resume() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(500).unwrap();
    ctrl.pause("user requested").unwrap();
    assert!(ctrl.state().is_resumable());
    assert!(!ctrl.state().is_active());
    ctrl.resume().unwrap();
    assert!(ctrl.state().is_active());
    assert_eq!(ctrl.state().processed(), 500);
}

#[test]
fn test_fail_and_resume() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(300).unwrap();
    ctrl.fail("connection lost").unwrap();
    assert!(ctrl.state().is_resumable());
    ctrl.resume().unwrap();
    assert!(ctrl.state().is_active());
    assert_eq!(ctrl.state().processed(), 300);
}

#[test]
fn test_cancel_from_in_progress() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(200).unwrap();
    ctrl.cancel().unwrap();
    assert!(ctrl.state().is_terminal());
    assert!(!ctrl.state().is_resumable());
}

#[test]
fn test_cancel_from_planned() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.cancel().unwrap();
    assert!(ctrl.state().is_terminal());
    assert_eq!(ctrl.state().processed(), 0);
}

#[test]
fn test_cancel_from_paused() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(100).unwrap();
    ctrl.pause("break").unwrap();
    ctrl.cancel().unwrap();
    assert!(ctrl.state().is_terminal());
    assert_eq!(ctrl.state().processed(), 100);
}

#[test]
fn test_cancel_from_failed() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.fail("boom").unwrap();
    ctrl.cancel().unwrap();
    assert!(ctrl.state().is_terminal());
}

#[test]
fn test_invalid_transitions() {
    let mut ctrl = MigrationController::new(test_plan());

    // Cannot pause before starting
    assert!(ctrl.pause("oops").is_err());
    // Cannot record progress before starting
    assert!(ctrl.record_progress(10).is_err());
    // Cannot resume from Planned
    assert!(ctrl.resume().is_err());

    ctrl.start().unwrap();
    ctrl.record_progress(1000).unwrap(); // auto-completes

    // Cannot cancel a completed migration
    assert!(ctrl.cancel().is_err());
    // Cannot start a completed migration
    assert!(ctrl.start().is_err());
}

#[test]
fn test_cannot_double_start() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    assert!(ctrl.start().is_err());
}

#[test]
fn test_cannot_pause_when_paused() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.pause("first").unwrap();
    assert!(ctrl.pause("second").is_err());
}

#[test]
fn test_cannot_fail_when_not_in_progress() {
    let mut ctrl = MigrationController::new(test_plan());
    assert!(ctrl.fail("not started").is_err());
}

#[test]
fn test_error_count() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_error();
    ctrl.record_error();
    let progress = ctrl.progress();
    assert_eq!(progress.error_count, 2);
}

#[test]
fn test_progress_report() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(500).unwrap();
    let report = ctrl.progress();
    assert_eq!(report.migration_id, "test-migration-001");
    assert!(report.state.is_active());
}

#[test]
fn test_progress_report_when_not_active() {
    let ctrl = MigrationController::new(test_plan());
    let report = ctrl.progress();
    assert_eq!(report.throughput, 0.0);
    assert!(report.eta_secs.is_none());
}

#[test]
fn test_plan_accessor() {
    let plan = test_plan();
    let ctrl = MigrationController::new(plan.clone());
    assert_eq!(ctrl.plan().id, "test-migration-001");
    assert_eq!(ctrl.plan().source_model, EmbeddingModel::BgeSmallEnV15);
    assert_eq!(ctrl.plan().target_model, EmbeddingModel::BgeBaseEnV15);
    assert_eq!(ctrl.plan().total_embeddings, 1000);
    assert_eq!(ctrl.plan().batch_size, 100);
}

#[test]
fn test_zero_total_progress() {
    let plan = MigrationPlan {
        id: "empty".to_string(),
        source_model: EmbeddingModel::BgeSmallEnV15,
        target_model: EmbeddingModel::BgeBaseEnV15,
        total_embeddings: 0,
        batch_size: 100,
        created_at: "2026-01-27T00:00:00Z".to_string(),
    };
    let mut ctrl = MigrationController::new(plan);
    ctrl.start().unwrap();
    // total=0, so any progress completes
    assert!(ctrl.state().is_active());
    // Verify progress shows 1.0 for zero-total when in-progress
    let state = &MigrationState::InProgress {
        processed: 0,
        total: 0,
        skipped: 0,
    };
    assert_eq!(state.progress(), Some(1.0));
}

#[test]
fn test_serialization_roundtrip_state() {
    let state = MigrationState::InProgress {
        processed: 42,
        total: 100,
        skipped: 5,
    };
    let json = serde_json::to_string(&state).unwrap();
    let restored: MigrationState = serde_json::from_str(&json).unwrap();
    assert_eq!(state, restored);
}

#[test]
fn test_serialization_roundtrip_plan() {
    let plan = test_plan();
    let json = serde_json::to_string(&plan).unwrap();
    let restored: MigrationPlan = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.id, plan.id);
    assert_eq!(restored.source_model, plan.source_model);
    assert_eq!(restored.target_model, plan.target_model);
    assert_eq!(restored.total_embeddings, plan.total_embeddings);
}

#[test]
fn test_serialization_roundtrip_progress() {
    let report = MigrationProgress {
        migration_id: "test".to_string(),
        state: MigrationState::Completed {
            processed: 100,
            skipped: 5,
            duration_secs: 5.5,
        },
        skipped: 5,
        effective_total: 95,
        effective_coverage: 1.0,
        throughput: 18.2,
        eta_secs: None,
        error_count: 1,
    };
    let json = serde_json::to_string(&report).unwrap();
    let restored: MigrationProgress = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.migration_id, "test");
    assert_eq!(restored.error_count, 1);
    assert_eq!(restored.skipped, 5);
    assert_eq!(restored.effective_total, 95);
}

#[test]
fn test_migration_error_display() {
    let err = MigrationError::InvalidTransition {
        from: "Planned".to_string(),
        to: "Paused".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "invalid migration transition from Planned to Paused"
    );
}

#[test]
fn test_full_lifecycle() {
    let mut ctrl = MigrationController::new(test_plan());

    assert_eq!(*ctrl.state(), MigrationState::Planned);

    ctrl.start().unwrap();
    assert!(ctrl.state().is_active());

    ctrl.record_progress(200).unwrap();
    ctrl.record_error();
    assert_eq!(ctrl.state().processed(), 200);

    ctrl.pause("maintenance window").unwrap();
    assert!(ctrl.state().is_resumable());

    ctrl.resume().unwrap();
    assert!(ctrl.state().is_active());
    assert_eq!(ctrl.state().processed(), 200);

    ctrl.record_progress(800).unwrap();
    assert!(ctrl.state().is_terminal());
    assert_eq!(ctrl.state().progress(), Some(1.0));

    let report = ctrl.progress();
    assert_eq!(report.error_count, 1);
    assert_eq!(report.migration_id, "test-migration-001");
}

// ==================== Skip Handling Tests ====================

#[test]
fn test_skip_reason_display() {
    let reason = SkipReason::ContentTooLarge {
        size: 50000,
        max: 8192,
    };
    assert_eq!(
        reason.to_string(),
        "content too large: 50000 bytes (max 8192)"
    );

    let reason = SkipReason::InvalidEncoding("UTF-16BE".to_string());
    assert_eq!(reason.to_string(), "invalid encoding: UTF-16BE");

    let reason = SkipReason::ContentDeleted;
    assert_eq!(reason.to_string(), "content deleted");

    let reason = SkipReason::PermanentApiError("rate limit exceeded".to_string());
    assert_eq!(
        reason.to_string(),
        "permanent API error: rate limit exceeded"
    );

    let reason = SkipReason::ManualSkip("deprecated content".to_string());
    assert_eq!(reason.to_string(), "manually skipped: deprecated content");
}

#[test]
fn test_record_skip() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_skip(SkipReason::ContentTooLarge {
        size: 50000,
        max: 8192,
    })
    .unwrap();
    assert_eq!(ctrl.state().skipped(), 1);
    assert_eq!(ctrl.skip_reasons().len(), 1);
}

#[test]
fn test_record_skip_requires_in_progress() {
    let mut ctrl = MigrationController::new(test_plan());
    assert!(ctrl.record_skip(SkipReason::ContentDeleted).is_err());
    ctrl.start().unwrap();
    ctrl.pause("break").unwrap();
    assert!(ctrl.record_skip(SkipReason::ContentDeleted).is_err());
}

#[test]
fn test_effective_coverage_calculation() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    for _ in 0..100 {
        ctrl.record_skip(SkipReason::ContentDeleted).unwrap();
    }
    ctrl.record_progress(450).unwrap();
    assert!((ctrl.effective_coverage() - 0.5).abs() < f64::EPSILON);
    assert_eq!(ctrl.state().effective_total(), 900);
}

#[test]
fn test_auto_complete_with_skipped() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    for _ in 0..200 {
        ctrl.record_skip(SkipReason::ContentDeleted).unwrap();
    }
    ctrl.record_progress(800).unwrap();
    assert!(ctrl.state().is_terminal());
    assert_eq!(ctrl.state().processed(), 800);
    assert_eq!(ctrl.state().skipped(), 200);
}

#[test]
fn test_skipped_preserved_through_pause_resume() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(100).unwrap();
    ctrl.record_skip(SkipReason::ContentTooLarge {
        size: 10000,
        max: 8192,
    })
    .unwrap();
    ctrl.record_skip(SkipReason::InvalidEncoding("binary".to_string()))
        .unwrap();
    assert_eq!(ctrl.state().skipped(), 2);
    ctrl.pause("break").unwrap();
    assert_eq!(ctrl.state().skipped(), 2);
    ctrl.resume().unwrap();
    assert_eq!(ctrl.state().skipped(), 2);
}

#[test]
fn test_skipped_preserved_through_fail_resume() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_progress(100).unwrap();
    ctrl.record_skip(SkipReason::PermanentApiError("503".to_string()))
        .unwrap();
    assert_eq!(ctrl.state().skipped(), 1);
    ctrl.fail("network error").unwrap();
    assert_eq!(ctrl.state().skipped(), 1);
    ctrl.resume().unwrap();
    assert_eq!(ctrl.state().skipped(), 1);
}

#[test]
fn test_skipped_preserved_in_cancel() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    ctrl.record_skip(SkipReason::ManualSkip("test".to_string()))
        .unwrap();
    ctrl.record_skip(SkipReason::ContentDeleted).unwrap();
    ctrl.cancel().unwrap();
    assert_eq!(ctrl.state().skipped(), 2);
}

#[test]
fn test_progress_report_includes_skip_info() {
    let mut ctrl = MigrationController::new(test_plan());
    ctrl.start().unwrap();
    for _ in 0..50 {
        ctrl.record_skip(SkipReason::ContentDeleted).unwrap();
    }
    ctrl.record_progress(200).unwrap();
    let report = ctrl.progress();
    assert_eq!(report.skipped, 50);
    assert_eq!(report.effective_total, 950);
    assert!((report.effective_coverage - (200.0 / 950.0)).abs() < 0.001);
}

#[test]
fn test_effective_coverage_zero_effective_total() {
    let plan = MigrationPlan {
        id: "all-skipped".to_string(),
        source_model: EmbeddingModel::BgeSmallEnV15,
        target_model: EmbeddingModel::BgeBaseEnV15,
        total_embeddings: 10,
        batch_size: 10,
        created_at: "2026-01-27T00:00:00Z".to_string(),
    };
    let mut ctrl = MigrationController::new(plan);
    ctrl.start().unwrap();
    for _ in 0..10 {
        ctrl.record_skip(SkipReason::ContentDeleted).unwrap();
    }
    assert_eq!(ctrl.effective_coverage(), 1.0);
    assert_eq!(ctrl.state().effective_total(), 0);
}

#[test]
fn test_skip_reason_serialization() {
    let reasons = vec![
        SkipReason::ContentTooLarge {
            size: 50000,
            max: 8192,
        },
        SkipReason::InvalidEncoding("UTF-16BE".to_string()),
        SkipReason::ContentDeleted,
        SkipReason::PermanentApiError("quota exceeded".to_string()),
        SkipReason::ManualSkip("deprecated".to_string()),
    ];
    for reason in reasons {
        let json = serde_json::to_string(&reason).unwrap();
        let restored: SkipReason = serde_json::from_str(&json).unwrap();
        assert_eq!(reason, restored);
    }
}

#[test]
fn test_backward_compat_deserialize_without_skipped() {
    let json = r#"{"InProgress":{"processed":42,"total":100}}"#;
    let state: MigrationState = serde_json::from_str(json).unwrap();
    assert_eq!(state.skipped(), 0);
    assert_eq!(state.processed(), 42);
}

#[test]
fn test_lifecycle_with_skips() {
    let mut ctrl = MigrationController::new(test_plan());

    ctrl.start().unwrap();
    assert_eq!(ctrl.state().skipped(), 0);

    ctrl.record_progress(300).unwrap();
    ctrl.record_skip(SkipReason::ContentTooLarge {
        size: 20000,
        max: 8192,
    })
    .unwrap();
    ctrl.record_skip(SkipReason::ContentDeleted).unwrap();
    ctrl.record_progress(200).unwrap();

    assert_eq!(ctrl.state().processed(), 500);
    assert_eq!(ctrl.state().skipped(), 2);
    assert_eq!(ctrl.skip_reasons().len(), 2);

    ctrl.pause("maintenance").unwrap();
    assert_eq!(ctrl.state().skipped(), 2);
    ctrl.resume().unwrap();

    ctrl.record_progress(496).unwrap();
    // 996 < 998 (effective_total = 1000 - 2 = 998), not complete yet
    assert!(!ctrl.state().is_terminal());

    ctrl.record_progress(2).unwrap();
    assert!(ctrl.state().is_terminal());
    assert_eq!(ctrl.state().skipped(), 2);
}
