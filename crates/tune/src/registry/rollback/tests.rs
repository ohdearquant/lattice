use super::*;

#[test]
fn test_rollback_record_creation() {
    let from_id = Uuid::new_v4();
    let to_id = Uuid::new_v4();
    let record = RollbackRecord::new(from_id, to_id, "test reason", Some("tester".to_string()));

    assert_eq!(record.from_model_id, from_id);
    assert_eq!(record.to_model_id, to_id);
    assert_eq!(record.reason, "test reason");
    assert_eq!(record.initiated_by, Some("tester".to_string()));
}

#[test]
fn test_controller_default() {
    let controller = RollbackController::default();
    assert_eq!(controller.max_history(), 100);
    assert!(controller.is_empty());
}

#[test]
fn test_record_rollback() {
    let mut controller = RollbackController::new(10);
    let from_id = Uuid::new_v4();
    let to_id = Uuid::new_v4();

    let record =
        controller.record_rollback(from_id, to_id, "perf regression", Some("ops".to_string()));

    assert_eq!(controller.len(), 1);
    assert_eq!(record.from_model_id, from_id);
    assert_eq!(record.to_model_id, to_id);
    assert_eq!(record.reason, "perf regression");
}

#[test]
fn test_history_trimming() {
    let mut controller = RollbackController::new(3);

    // Add 5 records
    for i in 0..5 {
        let from_id = Uuid::new_v4();
        let to_id = Uuid::new_v4();
        controller.record_rollback(from_id, to_id, format!("reason {i}"), None);
    }

    // Should only have 3 (the most recent)
    assert_eq!(controller.len(), 3);

    // Oldest should be "reason 2" (0 and 1 were trimmed)
    assert_eq!(controller.history()[0].reason, "reason 2");
    assert_eq!(controller.history()[2].reason, "reason 4");
}

#[test]
fn test_last_rollback() {
    let mut controller = RollbackController::new(10);
    assert!(controller.last_rollback().is_none());

    let from_id = Uuid::new_v4();
    let to_id = Uuid::new_v4();
    controller.record_rollback(from_id, to_id, "first", None);

    let from_id2 = Uuid::new_v4();
    let to_id2 = Uuid::new_v4();
    controller.record_rollback(from_id2, to_id2, "second", None);

    assert_eq!(controller.last_rollback().unwrap().reason, "second");
}

#[test]
fn test_rollbacks_involving() {
    let mut controller = RollbackController::new(10);

    let model_a = Uuid::new_v4();
    let model_b = Uuid::new_v4();
    let model_c = Uuid::new_v4();

    // A -> B
    controller.record_rollback(model_a, model_b, "rollback 1", None);
    // B -> C
    controller.record_rollback(model_b, model_c, "rollback 2", None);
    // C -> A
    controller.record_rollback(model_c, model_a, "rollback 3", None);

    // Model A involved in rollback 1 (from) and rollback 3 (to)
    let involving_a = controller.rollbacks_involving(model_a);
    assert_eq!(involving_a.len(), 2);

    // Model B involved in rollback 1 (to) and rollback 2 (from)
    let involving_b = controller.rollbacks_involving(model_b);
    assert_eq!(involving_b.len(), 2);

    // Model C involved in rollback 2 (to) and rollback 3 (from)
    let involving_c = controller.rollbacks_involving(model_c);
    assert_eq!(involving_c.len(), 2);
}

#[test]
fn test_rollbacks_from_and_to() {
    let mut controller = RollbackController::new(10);

    let model_a = Uuid::new_v4();
    let model_b = Uuid::new_v4();
    let model_c = Uuid::new_v4();

    controller.record_rollback(model_a, model_b, "1", None);
    controller.record_rollback(model_a, model_c, "2", None);
    controller.record_rollback(model_b, model_c, "3", None);

    // Model A was source twice
    assert_eq!(controller.rollbacks_from(model_a).len(), 2);
    // Model A was never a target
    assert_eq!(controller.rollbacks_to(model_a).len(), 0);

    // Model C was target twice
    assert_eq!(controller.rollbacks_to(model_c).len(), 2);
    // Model C was never a source
    assert_eq!(controller.rollbacks_from(model_c).len(), 0);
}

#[test]
fn test_clear_history() {
    let mut controller = RollbackController::new(10);

    controller.record_rollback(Uuid::new_v4(), Uuid::new_v4(), "test", None);
    controller.record_rollback(Uuid::new_v4(), Uuid::new_v4(), "test", None);

    assert_eq!(controller.len(), 2);

    controller.clear_history();

    assert!(controller.is_empty());
    assert!(controller.last_rollback().is_none());
}
