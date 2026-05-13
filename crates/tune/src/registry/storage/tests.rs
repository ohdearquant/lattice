use super::backends::InMemoryStorage;
#[cfg(feature = "sqlite")]
use super::backends::SqliteStorage;
use super::*;
use crate::registry::model::{ModelMetadata, ModelStatus, RegisteredModel};

#[test]
fn test_in_memory_storage() {
    let mut storage = InMemoryStorage::new();
    let model = RegisteredModel::new("test", "1.0.0");
    let weights = vec![1u8, 2, 3, 4, 5];

    let path = storage.save(&model, &weights).unwrap();
    assert!(storage.exists(&path));

    let loaded = storage.load(&path).unwrap();
    assert_eq!(loaded, weights);

    storage.delete(&path).unwrap();
    assert!(!storage.exists(&path));
}

#[test]
fn test_registry_register() {
    let registry = ModelRegistry::in_memory();
    let model = RegisteredModel::new("intent_classifier", "1.0.0");
    let weights = vec![0u8; 100];

    let id = registry.register(model, &weights).unwrap();
    assert!(registry.get_by_id(&id).is_some());
    assert!(registry.get("intent_classifier", "1.0.0").is_some());
}

#[test]
fn test_registry_duplicate() {
    let registry = ModelRegistry::in_memory();
    let model1 = RegisteredModel::new("test", "1.0.0");
    let model2 = RegisteredModel::new("test", "1.0.0");
    let weights = vec![0u8; 100];

    registry.register(model1, &weights).unwrap();
    let result = registry.register(model2, &weights);

    assert!(matches!(result, Err(TuneError::DuplicateModel { .. })));
}

#[test]
fn test_registry_versions() {
    let registry = ModelRegistry::in_memory();
    let weights = vec![0u8; 100];

    for version in ["1.0.0", "1.1.0", "2.0.0"] {
        let model = RegisteredModel::new("test", version);
        registry.register(model, &weights).unwrap();
    }

    let versions = registry.list_versions("test");
    assert_eq!(versions.len(), 3);

    let latest = registry.get_latest("test").unwrap();
    assert_eq!(latest.version, "2.0.0");
}

#[test]
fn test_registry_promotion() {
    let registry = ModelRegistry::in_memory();
    let weights = vec![0u8; 100];

    let model1 = RegisteredModel::new("test", "1.0.0");
    let id1 = registry.register(model1, &weights).unwrap();

    let model2 = RegisteredModel::new("test", "2.0.0");
    let id2 = registry.register(model2, &weights).unwrap();

    // Promote first model
    registry.promote_to_production(&id1).unwrap();
    assert_eq!(
        registry.get_by_id(&id1).unwrap().status,
        ModelStatus::Production
    );

    // Promote second model (should demote first)
    registry.promote_to_production(&id2).unwrap();
    assert_eq!(
        registry.get_by_id(&id1).unwrap().status,
        ModelStatus::Staged
    );
    assert_eq!(
        registry.get_by_id(&id2).unwrap().status,
        ModelStatus::Production
    );
}

#[test]
fn test_registry_delete() {
    let registry = ModelRegistry::in_memory();
    let model = RegisteredModel::new("test", "1.0.0");
    let weights = vec![0u8; 100];

    let id = registry.register(model, &weights).unwrap();
    assert_eq!(registry.len(), 1);

    registry.delete(&id).unwrap();
    assert!(registry.is_empty());
}

#[test]
fn test_model_query() {
    let registry = ModelRegistry::in_memory();
    let weights = vec![0u8; 100];

    // Add some models
    let mut model1 = RegisteredModel::new("classifier", "1.0.0");
    model1.metadata = ModelMetadata::default();
    model1.metadata.validation_accuracy = Some(0.9);
    model1.metadata.tags = vec!["production".to_string()];
    let id1 = registry.register(model1, &weights).unwrap();
    registry
        .update_status(&id1, ModelStatus::Production)
        .unwrap();

    let mut model2 = RegisteredModel::new("classifier", "2.0.0");
    model2.metadata = ModelMetadata::default();
    model2.metadata.validation_accuracy = Some(0.95);
    registry.register(model2, &weights).unwrap();

    // Query by status
    let production = ModelQuery::new()
        .status(ModelStatus::Production)
        .execute(&registry);
    assert_eq!(production.len(), 1);

    // Query by min accuracy
    let high_acc = ModelQuery::new().min_accuracy(0.92).execute(&registry);
    assert_eq!(high_acc.len(), 1);

    // Query by tag
    let tagged = ModelQuery::new().tag("production").execute(&registry);
    assert_eq!(tagged.len(), 1);
}

#[test]
fn test_concurrent_read_during_write() {
    use std::sync::Arc;
    use std::thread;

    let registry = Arc::new(ModelRegistry::in_memory());
    let weights = vec![0u8; 100];

    // Register some initial models
    for i in 0..10 {
        let model = RegisteredModel::new("test", &format!("{i}.0.0"));
        registry.register(model, &weights).unwrap();
    }

    let mut handles = Vec::new();

    // Spawn reader threads
    let r1 = Arc::clone(&registry);
    handles.push(thread::spawn(move || {
        for _ in 0..200 {
            let all = r1.list_all();
            // Must always see a consistent snapshot (never empty if we started with 10)
            assert!(!all.is_empty());
            let _ = r1.len();
            let _ = r1.is_empty();
            let _ = r1.list_names();
        }
    }));

    // Spawn writer thread
    let r2 = Arc::clone(&registry);
    handles.push(thread::spawn(move || {
        for i in 10..20 {
            let model = RegisteredModel::new("concurrent", &format!("{i}.0.0"));
            let _ = r2.register(model, &[0u8; 50]);
        }
    }));

    // Spawn another reader
    let r3 = Arc::clone(&registry);
    handles.push(thread::spawn(move || {
        for _ in 0..200 {
            // get_by_id and get should not panic
            let _ = r3.get("test", "0.0.0");
            let _ = r3.get_latest("test");
        }
    }));

    for h in handles {
        h.join().unwrap();
    }

    // Final state: 10 "test" + 10 "concurrent" = 20
    assert_eq!(registry.len(), 20);
}

// SQLite storage tests
#[cfg(feature = "sqlite")]
mod sqlite_tests {
    use super::*;

    #[test]
    fn test_sqlite_storage_save_and_load() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let model = RegisteredModel::new("test_model", "1.0.0");
        let weights = vec![1u8, 2, 3, 4, 5, 6, 7, 8];

        let path = storage.save(&model, &weights).unwrap();
        assert_eq!(path, "test_model/1.0.0/weights.bin");
        assert!(storage.exists(&path));

        let loaded = storage.load(&path).unwrap();
        assert_eq!(loaded, weights);
    }

    #[test]
    fn test_sqlite_storage_delete() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let model = RegisteredModel::new("test_model", "1.0.0");
        let weights = vec![1u8, 2, 3, 4, 5];

        let path = storage.save(&model, &weights).unwrap();
        assert!(storage.exists(&path));

        storage.delete(&path).unwrap();
        assert!(!storage.exists(&path));
    }

    #[test]
    fn test_sqlite_storage_list() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let weights = vec![0u8; 100];

        for version in ["1.0.0", "1.1.0", "2.0.0"] {
            let model = RegisteredModel::new("classifier", version);
            storage.save(&model, &weights).unwrap();
        }

        let paths = storage.list();
        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&"classifier/1.0.0/weights.bin".to_string()));
        assert!(paths.contains(&"classifier/1.1.0/weights.bin".to_string()));
        assert!(paths.contains(&"classifier/2.0.0/weights.bin".to_string()));
    }

    #[test]
    fn test_sqlite_storage_duplicate_rejected() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let model1 = RegisteredModel::new("duplicate_test", "1.0.0");
        let model2 = RegisteredModel::new("duplicate_test", "1.0.0");
        let weights = vec![0u8; 50];

        storage.save(&model1, &weights).unwrap();
        let result = storage.save(&model2, &weights);

        assert!(matches!(result, Err(TuneError::DuplicateModel { .. })));
    }

    #[test]
    fn test_sqlite_storage_exists_false_for_nonexistent() {
        let storage = SqliteStorage::in_memory().unwrap();
        assert!(!storage.exists("nonexistent/1.0.0/weights.bin"));
    }

    #[test]
    fn test_sqlite_storage_load_nonexistent() {
        let storage = SqliteStorage::in_memory().unwrap();
        let result = storage.load("nonexistent/1.0.0/weights.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_sqlite_storage_with_metadata() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let metadata = ModelMetadata::classifier(768, 6, 10000)
            .architecture("MLP(768, 256, 6)")
            .tag("production");
        let model = RegisteredModel::new("intent_classifier", "1.0.0")
            .with_metadata(metadata)
            .with_description("Intent classification model");
        let weights = vec![1u8; 1024];

        let path = storage.save(&model, &weights).unwrap();
        assert!(storage.exists(&path));

        let loaded = storage.load(&path).unwrap();
        assert_eq!(loaded.len(), 1024);
    }

    #[test]
    fn test_sqlite_storage_file_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("models.db");

        // Create storage and save a model
        {
            let mut storage = SqliteStorage::new(&db_path).unwrap();
            let model = RegisteredModel::new("persistent", "1.0.0");
            let weights = vec![42u8; 256];
            storage.save(&model, &weights).unwrap();
        }

        // Reopen storage and verify data persisted
        {
            let storage = SqliteStorage::new(&db_path).unwrap();
            assert!(storage.exists("persistent/1.0.0/weights.bin"));
            let loaded = storage.load("persistent/1.0.0/weights.bin").unwrap();
            assert_eq!(loaded, vec![42u8; 256]);
        }
    }

    #[test]
    fn test_sqlite_storage_path_traversal_rejected() {
        let storage = SqliteStorage::in_memory().unwrap();

        // Path traversal should be rejected
        assert!(!storage.exists("../etc/passwd"));
        assert!(!storage.exists("/etc/passwd"));
        assert!(storage.load("../evil/path").is_err());
    }

    // ========================================================================
    // #1392: Upgrade regression test — models.metadata_json → metadata
    //
    // Verifies that an existing DB whose `models` table was created with the
    // OLD `metadata_json` column name is transparently upgraded on open, so
    // that rows inserted under the old name are readable via the new column.
    // Also verifies idempotency (reopening an already-migrated DB succeeds).
    // ========================================================================

    #[test]
    fn upgrade_from_old_schema_renames_models_metadata_json_to_metadata() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("old_schema.db");

        // Phase 1: Manually create a DB with the OLD column name `metadata_json`.
        // This simulates a production DB created before the column was renamed.
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    registered_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    registered_by TEXT,
                    description TEXT,
                    weights_path TEXT,
                    weights_size INTEGER,
                    weights_hash TEXT,
                    parent_id TEXT,
                    UNIQUE(name, version)
                );",
            )
            .unwrap();

            conn.execute(
                "INSERT INTO models
                 (id, name, version, status, metadata_json, registered_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    "test-id-001",
                    "intent_classifier",
                    "1.0.0",
                    "staged",
                    r#"{"architecture":"MLP","hidden_size":256}"#,
                    1_000_000_i64,
                    1_000_000_i64,
                ],
            )
            .unwrap();
        }

        // Phase 2: Open the DB via SqliteStorage::new() — this must trigger the
        // migrate_models_metadata_json_to_metadata() migration transparently.
        let _storage = SqliteStorage::new(&db_path).unwrap();

        // Phase 3: Verify via a separate raw connection that the column was renamed
        // and data survived.  SqliteStorage wraps the connection privately, so we
        // open a second connection directly for introspection.
        {
            let verify_conn = rusqlite::Connection::open(&db_path).unwrap();

            // New column must be present and hold the original data.
            let metadata: String = verify_conn
                .query_row(
                    "SELECT metadata FROM models WHERE id = 'test-id-001'",
                    [],
                    |r| r.get(0),
                )
                .expect("row must be readable via new `metadata` column after upgrade");
            assert!(
                metadata.contains("MLP"),
                "metadata content must survive upgrade, got: {metadata}"
            );

            // Old column must no longer exist.
            let old_col_result = verify_conn.query_row(
                "SELECT metadata_json FROM models WHERE id = 'test-id-001'",
                [],
                |r| r.get::<_, String>(0),
            );
            assert!(
                old_col_result.is_err(),
                "old `metadata_json` column must not exist after migration"
            );
        }

        // Phase 4: Idempotency — reopening the already-migrated DB must succeed.
        // SqliteStorage::new() will call migrate_models_metadata_json_to_metadata()
        // again, which must be a no-op (not an error).
        let _storage2 =
            SqliteStorage::new(&db_path).expect("reopening already-migrated DB must not fail");
    }

    // ========================================================================
    // #1389: Upgrade regression test — models.registered_at / updated_at
    //        TEXT (RFC 3339) → INTEGER (epoch microseconds)
    //
    // Verifies that an existing DB whose `models` table was created with the
    // OLD TEXT column types for timestamps is transparently upgraded on open,
    // so rows inserted under the old schema remain accessible and the column
    // types are corrected.  Also verifies idempotency (reopening an already-
    // migrated DB succeeds without error).
    // ========================================================================

    #[test]
    fn upgrade_from_old_schema_converts_timestamps_text_to_integer() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("old_timestamps.db");

        // Phase 1: Manually create a DB with the OLD TEXT timestamp columns.
        // This simulates a production DB created before the column types were
        // corrected.
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    registered_by TEXT,
                    description TEXT,
                    weights_path TEXT,
                    weights_size INTEGER,
                    weights_hash TEXT,
                    parent_id TEXT,
                    UNIQUE(name, version)
                );",
            )
            .unwrap();

            // Insert a row with RFC 3339 TEXT timestamps (old format).
            conn.execute(
                "INSERT INTO models
                 (id, name, version, status, metadata, registered_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    "test-ts-001",
                    "ts_model",
                    "1.0.0",
                    "pending",
                    r#"{"architecture":"MLP"}"#,
                    "2024-01-15T10:30:00Z",
                    "2024-01-16T08:00:00Z",
                ],
            )
            .unwrap();
        }

        // Phase 2: Open the DB via SqliteStorage::new() — this must trigger the
        // migrate_models_timestamps_text_to_integer() migration transparently.
        let _storage = SqliteStorage::new(&db_path).unwrap();

        // Phase 3: Verify via a raw connection that:
        //   a) The row is still present.
        //   b) registered_at and updated_at are now INTEGER (epoch microseconds).
        //   c) The values are in the correct ballpark (non-zero, positive).
        {
            let verify_conn = rusqlite::Connection::open(&db_path).unwrap();

            let (reg_at, upd_at): (i64, i64) = verify_conn
                .query_row(
                    "SELECT registered_at, updated_at FROM models WHERE id = 'test-ts-001'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .expect("row must be readable after timestamp migration");

            // 2024-01-15T10:30:00Z in epoch seconds is 1705314600.
            // In epoch microseconds: 1705314600 * 1_000_000 = 1_705_314_600_000_000.
            assert_eq!(
                reg_at, 1_705_314_600_000_000_i64,
                "registered_at must be epoch microseconds, got: {reg_at}"
            );

            // 2024-01-16T08:00:00Z in epoch seconds is 1705392000.
            // In epoch microseconds: 1705392000 * 1_000_000 = 1_705_392_000_000_000.
            assert_eq!(
                upd_at, 1_705_392_000_000_000_i64,
                "updated_at must be epoch microseconds, got: {upd_at}"
            );

            // Verify declared column type is now INTEGER via PRAGMA table_info.
            let col_types: Vec<(String, String)> = verify_conn
                .prepare("PRAGMA table_info(models)")
                .unwrap()
                .query_map([], |row| {
                    Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
                })
                .unwrap()
                .filter_map(|r| r.ok())
                .filter(|(name, _)| name == "registered_at" || name == "updated_at")
                .collect();

            for (col_name, col_type) in &col_types {
                assert_eq!(
                    col_type, "INTEGER",
                    "column {col_name} must declare INTEGER after migration, got: {col_type}"
                );
            }
        }

        // Phase 4: Idempotency — reopening the already-migrated DB must succeed
        // without error.  The migration inspects column type and exits early
        // when it sees INTEGER.
        let _storage2 =
            SqliteStorage::new(&db_path).expect("reopening already-migrated DB must not fail");

        // Phase 5: Writing a new model after migration must succeed (INTEGER path).
        {
            let mut storage3 = SqliteStorage::new(&db_path).unwrap();
            let model = RegisteredModel::new("post_migration_model", "2.0.0");
            let weights = vec![1u8, 2, 3];
            let path = storage3.save(&model, &weights).unwrap();
            assert!(storage3.exists(&path));
        }
    }

    #[test]
    fn upgrade_timestamps_handles_mixed_integer_rows_safely() {
        // Verifies the per-row typeof() guard: a DB can have some rows already
        // storing INTEGER values (if partially migrated or written by new code
        // before migration ran) alongside TEXT rows without corruption.
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("mixed_timestamps.db");

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            // Create table with TEXT column type (old schema).
            conn.execute_batch(
                "CREATE TABLE models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    registered_by TEXT,
                    description TEXT,
                    weights_path TEXT,
                    weights_size INTEGER,
                    weights_hash TEXT,
                    parent_id TEXT,
                    UNIQUE(name, version)
                );",
            )
            .unwrap();

            // Row A: TEXT RFC 3339 timestamps.
            conn.execute(
                "INSERT INTO models
                 (id, name, version, status, metadata, registered_at, updated_at)
                 VALUES ('row-a', 'model_a', '1.0.0', 'pending', '{}',
                         '2024-03-01T00:00:00Z', '2024-03-01T00:00:00Z')",
                [],
            )
            .unwrap();

            // Row B: INTEGER timestamps stored despite TEXT column type
            // (SQLite dynamic typing permits this).
            conn.execute(
                "INSERT INTO models
                 (id, name, version, status, metadata, registered_at, updated_at)
                 VALUES ('row-b', 'model_b', '1.0.0', 'pending', '{}',
                         1709251200000000, 1709251200000000)",
                [],
            )
            .unwrap();
        }

        // Open via SqliteStorage — migration must handle both rows safely.
        let _storage = SqliteStorage::new(&db_path).unwrap();

        {
            let verify_conn = rusqlite::Connection::open(&db_path).unwrap();

            // Row A was TEXT: expect epoch-micro conversion.
            // 2024-03-01T00:00:00Z = 1709251200 seconds → 1_709_251_200_000_000 µs.
            let (ra_reg, ra_upd): (i64, i64) = verify_conn
                .query_row(
                    "SELECT registered_at, updated_at FROM models WHERE id = 'row-a'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert_eq!(ra_reg, 1_709_251_200_000_000_i64, "row-a registered_at");
            assert_eq!(ra_upd, 1_709_251_200_000_000_i64, "row-a updated_at");

            // Row B was already INTEGER: must pass through unchanged.
            let (rb_reg, rb_upd): (i64, i64) = verify_conn
                .query_row(
                    "SELECT registered_at, updated_at FROM models WHERE id = 'row-b'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert_eq!(
                rb_reg, 1_709_251_200_000_000_i64,
                "row-b registered_at unchanged"
            );
            assert_eq!(
                rb_upd, 1_709_251_200_000_000_i64,
                "row-b updated_at unchanged"
            );
        }
    }

    #[test]
    fn test_sqlite_storage_multiple_models() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let weights = vec![0u8; 64];

        // Save multiple different models
        let models = vec![
            ("classifier", "1.0.0"),
            ("classifier", "2.0.0"),
            ("embedder", "1.0.0"),
            ("ranker", "0.1.0"),
        ];

        for (name, version) in &models {
            let model = RegisteredModel::new(*name, *version);
            storage.save(&model, &weights).unwrap();
        }

        let paths = storage.list();
        assert_eq!(paths.len(), 4);

        // Verify all models exist
        for (name, version) in &models {
            let path = format!("{name}/{version}/weights.bin");
            assert!(storage.exists(&path));
        }
    }
}
