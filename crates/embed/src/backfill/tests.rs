//! Tests for the backfill module.

use crate::backfill::{
    BackfillConfig, BackfillCoordinator, EmbeddingRoute, EmbeddingRoutingConfig, RoutingPhase,
};
use crate::migration::{MigrationPlan, MigrationState, SkipReason};
use crate::model::EmbeddingModel;

fn test_plan() -> MigrationPlan {
    MigrationPlan {
        id: "backfill-test-001".to_string(),
        source_model: EmbeddingModel::BgeSmallEnV15,
        target_model: EmbeddingModel::BgeBaseEnV15,
        total_embeddings: 1000,
        batch_size: 100,
        created_at: "2026-01-27T00:00:00Z".to_string(),
    }
}

#[test]
fn test_route_request_legacy_when_planned() {
    let coord = BackfillCoordinator::with_defaults(test_plan());
    assert_eq!(coord.route_request(true), EmbeddingRoute::Legacy);
    assert_eq!(coord.route_request(false), EmbeddingRoute::Legacy);
}

#[test]
fn test_route_request_dual_write_for_new_docs_during_migration() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    assert_eq!(coord.route_request(true), EmbeddingRoute::DualWrite);
}

#[test]
fn test_route_request_legacy_for_existing_docs_during_migration() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    assert_eq!(coord.route_request(false), EmbeddingRoute::Legacy);
}

#[test]
fn test_route_request_no_dual_write_when_disabled() {
    let config = BackfillConfig {
        dual_write: false,
        ..BackfillConfig::default()
    };
    let mut coord = BackfillCoordinator::new(test_plan(), config);
    coord.start().unwrap();
    assert_eq!(coord.route_request(true), EmbeddingRoute::Legacy);
}

#[test]
fn test_route_request_target_after_completion() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(1000).unwrap(); // auto-completes
    assert_eq!(coord.route_request(true), EmbeddingRoute::Target);
    assert_eq!(coord.route_request(false), EmbeddingRoute::Target);
}

#[test]
fn test_route_request_legacy_when_paused() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.pause("maintenance").unwrap();
    assert_eq!(coord.route_request(true), EmbeddingRoute::Legacy);
}

#[test]
fn test_route_query_legacy_during_early_migration() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(100).unwrap(); // 10% progress < 80% threshold
    assert_eq!(coord.route_query(), EmbeddingRoute::Legacy);
}

#[test]
fn test_route_query_target_when_above_threshold() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(800).unwrap(); // 80% progress >= 80% threshold
    assert_eq!(coord.route_query(), EmbeddingRoute::Target);
}

#[test]
fn test_route_query_target_after_completion() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(1000).unwrap();
    assert_eq!(coord.route_query(), EmbeddingRoute::Target);
}

#[test]
fn test_route_query_legacy_when_planned() {
    let coord = BackfillCoordinator::with_defaults(test_plan());
    assert_eq!(coord.route_query(), EmbeddingRoute::Legacy);
}

#[test]
fn test_route_query_legacy_when_paused() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(900).unwrap(); // 90% but paused
    coord.pause("stop").unwrap();
    assert_eq!(coord.route_query(), EmbeddingRoute::Legacy);
}

#[test]
fn test_route_query_custom_threshold() {
    let config = BackfillConfig {
        target_query_threshold: 0.5,
        ..BackfillConfig::default()
    };
    let mut coord = BackfillCoordinator::new(test_plan(), config);
    coord.start().unwrap();
    coord.record_batch(500).unwrap(); // 50% >= 50% threshold
    assert_eq!(coord.route_query(), EmbeddingRoute::Target);
}

#[test]
fn test_record_batch_updates_progress() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(250).unwrap();
    assert_eq!(coord.backfilled_count(), 250);
    assert_eq!(coord.state().processed(), 250);

    coord.record_batch(150).unwrap();
    assert_eq!(coord.backfilled_count(), 400);
    assert_eq!(coord.state().processed(), 400);
}

#[test]
fn test_next_batch_size_normal() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    assert_eq!(coord.next_batch_size(), 100);
}

#[test]
fn test_next_batch_size_less_than_batch() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(950).unwrap();
    assert_eq!(coord.next_batch_size(), 50);
}

#[test]
fn test_next_batch_size_zero_when_not_in_progress() {
    let coord = BackfillCoordinator::with_defaults(test_plan());
    assert_eq!(coord.next_batch_size(), 0); // Planned
}

#[test]
fn test_next_batch_size_zero_after_completion() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(1000).unwrap();
    assert_eq!(coord.next_batch_size(), 0); // Completed
}

#[test]
fn test_full_lifecycle() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());

    assert_eq!(*coord.state(), MigrationState::Planned);
    assert_eq!(coord.route_request(true), EmbeddingRoute::Legacy);
    assert_eq!(coord.route_query(), EmbeddingRoute::Legacy);
    assert_eq!(coord.next_batch_size(), 0);

    coord.start().unwrap();
    assert!(coord.state().is_active());
    assert_eq!(coord.route_request(true), EmbeddingRoute::DualWrite);
    assert_eq!(coord.route_query(), EmbeddingRoute::Legacy);
    assert_eq!(coord.next_batch_size(), 100);

    coord.record_batch(100).unwrap();
    coord.record_batch(100).unwrap();
    coord.record_batch(100).unwrap();
    assert_eq!(coord.backfilled_count(), 300);
    assert_eq!(coord.state().processed(), 300);
    assert_eq!(coord.route_query(), EmbeddingRoute::Legacy);

    coord.record_batch(500).unwrap(); // 800/1000 = 80%
    assert_eq!(coord.route_query(), EmbeddingRoute::Target);
    assert_eq!(coord.route_request(true), EmbeddingRoute::DualWrite);
    assert_eq!(coord.next_batch_size(), 100); // 200 remaining

    coord.record_batch(200).unwrap();
    assert!(coord.state().is_terminal());
    assert_eq!(coord.route_request(true), EmbeddingRoute::Target);
    assert_eq!(coord.route_request(false), EmbeddingRoute::Target);
    assert_eq!(coord.route_query(), EmbeddingRoute::Target);
    assert_eq!(coord.backfilled_count(), 1000);
    assert_eq!(coord.next_batch_size(), 0);
}

#[test]
fn test_pause_and_resume_lifecycle() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(500).unwrap();

    coord.pause("resource contention").unwrap();
    assert_eq!(coord.route_request(true), EmbeddingRoute::Legacy);
    assert_eq!(coord.route_query(), EmbeddingRoute::Legacy);
    assert_eq!(coord.next_batch_size(), 0);

    coord.resume().unwrap();
    assert!(coord.state().is_active());
    assert_eq!(coord.route_request(true), EmbeddingRoute::DualWrite);
    assert_eq!(coord.state().processed(), 500);
}

#[test]
fn test_cancel_lifecycle() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_batch(300).unwrap();
    coord.cancel().unwrap();

    assert!(coord.state().is_terminal());
    assert_eq!(coord.route_request(true), EmbeddingRoute::Legacy);
    assert_eq!(coord.route_query(), EmbeddingRoute::Legacy);
    assert_eq!(coord.backfilled_count(), 300);
}

#[test]
fn test_error_tracking() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.record_error();
    coord.record_error();
    coord.record_error();
    let report = coord.progress();
    assert_eq!(report.error_count, 3);
}

#[test]
fn test_model_accessors() {
    let coord = BackfillCoordinator::with_defaults(test_plan());
    assert_eq!(coord.source_model(), EmbeddingModel::BgeSmallEnV15);
    assert_eq!(coord.target_model(), EmbeddingModel::BgeBaseEnV15);
}

#[test]
fn test_backfill_config_defaults() {
    let config = BackfillConfig::default();
    assert_eq!(config.batch_size, 100);
    assert_eq!(config.max_concurrent, 4);
    assert!(config.dual_write);
    assert!((config.target_query_threshold - 0.8).abs() < f64::EPSILON);
    assert_eq!(config.rollback_window_secs, 86400); // 24 hours
}

#[test]
fn test_backfill_config_serialization() {
    let config = BackfillConfig {
        batch_size: 256,
        max_concurrent: 8,
        dual_write: false,
        target_query_threshold: 0.95,
        rollback_window_secs: 3600,
    };
    let json = serde_json::to_string(&config).unwrap();
    let restored: BackfillConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.batch_size, 256);
    assert_eq!(restored.max_concurrent, 8);
    assert!(!restored.dual_write);
    assert!((restored.target_query_threshold - 0.95).abs() < f64::EPSILON);
    assert_eq!(restored.rollback_window_secs, 3600);
}

#[test]
fn test_zero_total_embeddings() {
    let plan = MigrationPlan {
        id: "empty-mig".to_string(),
        source_model: EmbeddingModel::BgeSmallEnV15,
        target_model: EmbeddingModel::BgeBaseEnV15,
        total_embeddings: 0,
        batch_size: 100,
        created_at: "2026-01-27T00:00:00Z".to_string(),
    };
    let mut coord = BackfillCoordinator::with_defaults(plan);
    coord.start().unwrap();
    assert_eq!(coord.route_query(), EmbeddingRoute::Target);
}

#[test]
fn test_embedding_route_traits() {
    let route = EmbeddingRoute::DualWrite;
    let route2 = route; // Copy
    let route3 = route.clone(); // Clone
    assert_eq!(route2, route3); // PartialEq + Eq
    assert_eq!(format!("{route:?}"), "DualWrite"); // Debug

    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(EmbeddingRoute::Legacy);
    set.insert(EmbeddingRoute::Target);
    set.insert(EmbeddingRoute::DualWrite);
    assert_eq!(set.len(), 3);
}

#[test]
fn test_routing_phase_serialization() {
    let phases = [
        RoutingPhase::Stable,
        RoutingPhase::Migrating,
        RoutingPhase::RollbackWindow,
    ];

    for phase in phases {
        let json = serde_json::to_string(&phase).unwrap();
        let restored: RoutingPhase = serde_json::from_str(&json).unwrap();
        assert_eq!(phase, restored);
    }
}

#[test]
fn test_routing_config_when_planned() {
    let coord = BackfillCoordinator::with_defaults(test_plan());
    let config = coord.routing_config();

    assert_eq!(config.query_model, EmbeddingModel::BgeSmallEnV15);
    assert_eq!(config.write_models, vec![EmbeddingModel::BgeSmallEnV15]);
    assert_eq!(config.phase, RoutingPhase::Stable);
    assert!(config.migration_id.is_none());
}

#[test]
fn test_routing_config_during_migration() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    let config = coord.routing_config();

    assert_eq!(config.query_model, EmbeddingModel::BgeSmallEnV15);
    assert_eq!(
        config.write_models,
        vec![EmbeddingModel::BgeSmallEnV15, EmbeddingModel::BgeBaseEnV15]
    );
    assert_eq!(config.phase, RoutingPhase::Migrating);
    assert_eq!(config.migration_id.as_deref(), Some("backfill-test-001"));
}

#[test]
fn test_routing_config_during_migration_no_dual_write() {
    let config = BackfillConfig {
        dual_write: false,
        ..BackfillConfig::default()
    };
    let mut coord = BackfillCoordinator::new(test_plan(), config);
    coord.start().unwrap();
    let routing = coord.routing_config();

    assert_eq!(routing.write_models, vec![EmbeddingModel::BgeSmallEnV15]);
    assert_eq!(routing.phase, RoutingPhase::Migrating);
}

#[test]
fn test_routing_config_after_completion_in_rollback_window() {
    let config = BackfillConfig {
        rollback_window_secs: 86400 * 365, // 1 year
        ..BackfillConfig::default()
    };
    let mut coord = BackfillCoordinator::new(test_plan(), config);
    coord.start().unwrap();
    coord.record_batch(1000).unwrap(); // completes migration

    let routing = coord.routing_config();

    assert_eq!(routing.query_model, EmbeddingModel::BgeBaseEnV15);
    assert_eq!(
        routing.write_models,
        vec![EmbeddingModel::BgeSmallEnV15, EmbeddingModel::BgeBaseEnV15]
    );
    assert_eq!(routing.phase, RoutingPhase::RollbackWindow);
    assert!(routing.migration_id.is_some());
}

#[test]
fn test_routing_config_after_rollback_window_expires() {
    let config = BackfillConfig {
        rollback_window_secs: 0,
        ..BackfillConfig::default()
    };
    let mut coord = BackfillCoordinator::new(test_plan(), config);
    coord.start().unwrap();
    coord.record_batch(1000).unwrap();

    let routing = coord.routing_config();

    assert_eq!(routing.query_model, EmbeddingModel::BgeBaseEnV15);
    assert_eq!(routing.write_models, vec![EmbeddingModel::BgeBaseEnV15]);
    assert_eq!(routing.phase, RoutingPhase::Stable);
    assert!(routing.migration_id.is_none());
}

#[test]
fn test_routing_config_when_paused() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.pause("test").unwrap();

    let routing = coord.routing_config();

    assert_eq!(routing.query_model, EmbeddingModel::BgeSmallEnV15);
    assert_eq!(routing.write_models, vec![EmbeddingModel::BgeSmallEnV15]);
    assert_eq!(routing.phase, RoutingPhase::Stable);
}

#[test]
fn test_routing_config_when_cancelled() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());
    coord.start().unwrap();
    coord.cancel().unwrap();

    let routing = coord.routing_config();

    assert_eq!(routing.query_model, EmbeddingModel::BgeSmallEnV15);
    assert_eq!(routing.write_models, vec![EmbeddingModel::BgeSmallEnV15]);
    assert_eq!(routing.phase, RoutingPhase::Stable);
}

#[test]
fn test_in_rollback_window_false_when_not_completed() {
    let coord = BackfillCoordinator::with_defaults(test_plan());
    assert!(!coord.in_rollback_window()); // Not even started

    let mut coord2 = BackfillCoordinator::with_defaults(test_plan());
    coord2.start().unwrap();
    assert!(!coord2.in_rollback_window()); // In progress, not completed
}

#[test]
fn test_in_rollback_window_true_immediately_after_cutover() {
    let config = BackfillConfig {
        rollback_window_secs: 86400, // 24 hours
        ..BackfillConfig::default()
    };
    let mut coord = BackfillCoordinator::new(test_plan(), config);
    coord.start().unwrap();
    coord.record_batch(1000).unwrap(); // completes migration

    assert!(coord.in_rollback_window());
}

#[test]
fn test_in_rollback_window_false_after_window_expires() {
    let config = BackfillConfig {
        rollback_window_secs: 0, // 0 seconds = no window
        ..BackfillConfig::default()
    };
    let mut coord = BackfillCoordinator::new(test_plan(), config);
    coord.start().unwrap();
    coord.record_batch(1000).unwrap();

    assert!(!coord.in_rollback_window());
}

#[test]
fn test_cutover_at_set_on_completion() {
    let mut coord = BackfillCoordinator::with_defaults(test_plan());

    assert!(coord.cutover_at.is_none());

    coord.start().unwrap();
    coord.record_batch(500).unwrap();
    assert!(coord.cutover_at.is_none());

    coord.record_batch(500).unwrap(); // completes
    assert!(coord.cutover_at.is_some());
}

#[test]
fn test_embedding_routing_config_serialization() {
    let config = EmbeddingRoutingConfig {
        query_model: EmbeddingModel::BgeBaseEnV15,
        write_models: vec![EmbeddingModel::BgeSmallEnV15, EmbeddingModel::BgeBaseEnV15],
        phase: RoutingPhase::RollbackWindow,
        migration_id: Some("mig-001".to_string()),
    };

    let json = serde_json::to_string(&config).unwrap();
    let restored: EmbeddingRoutingConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.query_model, EmbeddingModel::BgeBaseEnV15);
    assert_eq!(restored.write_models.len(), 2);
    assert_eq!(restored.phase, RoutingPhase::RollbackWindow);
    assert_eq!(restored.migration_id.as_deref(), Some("mig-001"));
}

#[test]
fn test_routing_config_lifecycle_full() {
    let config = BackfillConfig {
        rollback_window_secs: 86400 * 365, // Long window for testing
        ..BackfillConfig::default()
    };
    let mut coord = BackfillCoordinator::new(test_plan(), config);

    let routing = coord.routing_config();
    assert_eq!(routing.phase, RoutingPhase::Stable);
    assert_eq!(routing.write_models.len(), 1);
    assert!(routing.migration_id.is_none());

    coord.start().unwrap();
    let routing = coord.routing_config();
    assert_eq!(routing.phase, RoutingPhase::Migrating);
    assert_eq!(routing.write_models.len(), 2);
    assert!(routing.migration_id.is_some());

    coord.record_batch(1000).unwrap();
    let routing = coord.routing_config();
    assert_eq!(routing.phase, RoutingPhase::RollbackWindow);
    assert_eq!(routing.query_model, EmbeddingModel::BgeBaseEnV15);
    assert_eq!(routing.write_models.len(), 2);
    assert!(routing.migration_id.is_some());
}

#[test]
fn test_next_batch_size_accounts_for_skipped() {
    let plan = MigrationPlan {
        id: "batch-skipped".to_string(),
        source_model: EmbeddingModel::BgeSmallEnV15,
        target_model: EmbeddingModel::BgeBaseEnV15,
        total_embeddings: 10,
        batch_size: 100, // larger than total so batch_size is not the limit
        created_at: "2026-01-27T00:00:00Z".to_string(),
    };
    let config = BackfillConfig {
        batch_size: 50,
        ..BackfillConfig::default()
    };
    let mut coord = BackfillCoordinator::new(plan, config);
    coord.start().unwrap();

    for _ in 0..3 {
        coord
            .controller
            .record_skip(SkipReason::ContentDeleted)
            .unwrap();
    }

    assert_eq!(coord.next_batch_size(), 7);

    coord.record_batch(7).unwrap();
    assert!(
        coord.state().is_terminal(),
        "expected Completed, got {:?}",
        coord.state()
    );

    assert_eq!(coord.next_batch_size(), 0);
}
