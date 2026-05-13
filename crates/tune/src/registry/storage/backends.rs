//! Storage backend implementations.

use super::{StorageBackend, validate_model_identity, validate_path};
use crate::error::{Result, TuneError};
use crate::registry::model::RegisteredModel;
use std::collections::HashMap;
use std::path::PathBuf;

/// In-memory storage backend (for testing)
#[derive(Default)]
pub struct InMemoryStorage {
    storage: HashMap<String, Vec<u8>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }
}

impl StorageBackend for InMemoryStorage {
    fn save(&mut self, model: &RegisteredModel, weights: &[u8]) -> Result<String> {
        let path = format!("{}/{}/weights.bin", model.name, model.version);
        self.storage.insert(path.clone(), weights.to_vec());
        Ok(path)
    }

    fn load(&self, path: &str) -> Result<Vec<u8>> {
        self.storage
            .get(path)
            .cloned()
            .ok_or_else(|| TuneError::Storage(format!("Model not found: {path}")))
    }

    fn delete(&mut self, path: &str) -> Result<()> {
        self.storage.remove(path);
        Ok(())
    }

    fn exists(&self, path: &str) -> bool {
        self.storage.contains_key(path)
    }

    fn list(&self) -> Vec<String> {
        self.storage.keys().cloned().collect()
    }
}

/// Filesystem storage backend
pub struct FileSystemStorage {
    root: PathBuf,
}

/// SQLite-backed storage for model registry.
///
/// Stores model metadata in a SQLite database while weights are stored
/// on the filesystem alongside the database.
#[cfg(feature = "sqlite")]
// Methods are used via the StorageBackend trait impl below; the compiler
// cannot see them as "used" when checking without the sqlite feature.
#[allow(dead_code)]
pub struct SqliteStorage {
    conn: std::sync::Mutex<rusqlite::Connection>,
    weights_dir: PathBuf,
}

/// Rename the `metadata_json` column in the `models` table to `metadata`.
///
/// This is an idempotent upgrade migration (#1392): databases created before
/// the column was renamed will have `metadata_json`; databases created after
/// will already have `metadata`.  We attempt the rename and silently succeed
/// if the old column no longer exists (already migrated or fresh schema).
///
/// Idempotency is implemented by checking the actual column list in
/// `sqlite_master` before attempting the rename, which avoids relying on
/// error-string matching across SQLite versions.
///
/// # Errors
///
/// Returns an error if the column exists but the rename fails for any
/// unexpected reason.  Fail-closed: we never silently swallow unexpected
/// errors.
// Called from SqliteStorage::new() which is cfg(feature = "sqlite");
// the compiler cannot trace the call site when checking without that feature.
#[cfg(feature = "sqlite")]
#[allow(dead_code)]
fn migrate_models_metadata_json_to_metadata(
    conn: &rusqlite::Connection,
) -> std::result::Result<(), rusqlite::Error> {
    // Check whether the old column still exists in the table.
    // We query PRAGMA table_info which lists all columns with their names.
    let has_old_column: bool = conn
        .prepare("PRAGMA table_info(models)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .any(|col| col.as_deref() == Ok("metadata_json"));

    if has_old_column {
        conn.execute_batch("ALTER TABLE models RENAME COLUMN metadata_json TO metadata;")?;
    }
    // If the old column is absent (fresh schema or already migrated), do nothing.
    Ok(())
}

/// Convert `registered_at` and `updated_at` columns from TEXT (RFC 3339) to
/// INTEGER (epoch microseconds) in the `models` table.
///
/// This is an idempotent upgrade migration (#1389).  Databases created before
/// the schema was corrected declare these columns as `TEXT NOT NULL` and store
/// RFC 3339 strings (e.g. `"2024-01-15T10:30:00Z"`).  Databases created after
/// the correction declare them as `INTEGER NOT NULL` and store epoch
/// microseconds (i64).
///
/// SQLite does not support `ALTER TABLE … ALTER COLUMN … TYPE`, so the
/// migration recreates the `models` table with the correct column types and
/// converts existing rows using per-row `typeof()` guards so the migration is
/// safe whether rows contain TEXT RFC 3339 values or INTEGER values already.
///
/// # Idempotency
///
/// The migration inspects `PRAGMA table_info(models)` to detect the declared
/// type of `registered_at`.  If both columns already declare `INTEGER` the
/// migration is a no-op and returns immediately.  Re-running it on an already-
/// migrated database (or a fresh database) is safe.
///
/// # Conversion formula (per-row)
///
/// SQLite coerces all values to TEXT affinity in the old schema, so `typeof()`
/// alone cannot distinguish an RFC 3339 string from a numeric string.  The
/// migration uses a three-way CASE per timestamp column:
///
/// 1. `typeof(val) = 'integer'` → native integer (stored without coercion)
///    → pass through unchanged.
/// 2. `datetime(val) IS NOT NULL` → valid datetime string (RFC 3339 / ISO 8601)
///    → convert: `CAST(strftime('%s', val) AS INTEGER) * 1_000_000`
///    (epoch seconds → epoch microseconds, sub-second component zeroed).
/// 3. Otherwise → numeric string already holding epoch microseconds
///    → `CAST(val AS INTEGER)` to strip TEXT affinity.
///
/// This handles all real-world states: pure TEXT databases (case 2), databases
/// where new code wrote integer micros to old TEXT columns (case 3 — the
/// integer was coerced to a TEXT digit string by SQLite affinity rules), and
/// fresh INTEGER-schema rows that somehow arrived in a migration path (case 1).
///
/// # Errors
///
/// Returns an error if the table recreation or data conversion fails for any
/// unexpected reason.  Fail-closed: we never silently swallow unexpected errors.
// Called from SqliteStorage::new() which is cfg(feature = "sqlite");
// the compiler cannot trace the call site when checking without that feature.
#[cfg(feature = "sqlite")]
#[allow(dead_code)]
fn migrate_models_timestamps_text_to_integer(
    conn: &rusqlite::Connection,
) -> std::result::Result<(), rusqlite::Error> {
    // Check declared type of `registered_at` via PRAGMA table_info.
    // Column indices: 0=cid, 1=name, 2=type, 3=notnull, 4=dflt_value, 5=pk.
    // `|r| r.ok()` cannot be replaced with `Result::ok` here: the iterator
    // yields `rusqlite::Result<_>` but the crate re-exports its own `Result`,
    // so clippy's suggestion produces a type-mismatch compile error.
    #[allow(clippy::redundant_closure_for_method_calls)]
    let registered_at_type: Option<String> = conn
        .prepare("PRAGMA table_info(models)")?
        .query_map([], |row| {
            let name: String = row.get(1)?;
            let col_type: String = row.get(2)?;
            Ok((name, col_type))
        })?
        .filter_map(|r| r.ok())
        .find(|(name, _)| name == "registered_at")
        .map(|(_, col_type)| col_type);

    // If the column already declares INTEGER (or the table doesn't exist yet),
    // this migration is a no-op.
    match registered_at_type.as_deref() {
        Some("INTEGER") | None => return Ok(()),
        _ => {} // TEXT or any other type — proceed with migration
    }

    // Recreate the table with INTEGER columns.  We use a backup + rename
    // pattern which is the only portable way to change column types in SQLite.
    // Wrapped in BEGIN IMMEDIATE so the 4-step recreation (CREATE backup →
    // INSERT SELECT → DROP original → RENAME backup) is atomic.  Any failure
    // in an intermediate step causes an automatic rollback, leaving the
    // original table intact.
    conn.execute_batch(
        "
        BEGIN IMMEDIATE;

        -- Step 1: create backup table with correct INTEGER types
        CREATE TABLE models_backup (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata TEXT NOT NULL,
            registered_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            registered_by TEXT,
            description TEXT,
            weights_path TEXT,
            weights_size INTEGER,
            weights_hash TEXT,
            parent_id TEXT,
            UNIQUE(name, version)
        );

        -- Step 2: copy rows, converting timestamps per-row safely.
        --
        -- We cannot rely on typeof() alone because SQLite coerces inserted
        -- values to the declared column affinity (TEXT in the old schema),
        -- meaning both TEXT RFC 3339 strings and integer numerics stored as
        -- text will report typeof() = 'text'.
        --
        -- Instead we use datetime() as a probe:
        --  * If datetime(val) IS NOT NULL → parseable as date → convert via
        --    strftime('%s', val) * 1_000_000 (epoch seconds → microseconds).
        --  * Otherwise → assume the text holds a numeric string that was
        --    already epoch microseconds → CAST directly to INTEGER.
        --  * If typeof() = 'integer' the value is a native integer (stored
        --    without affinity coercion) → pass through unchanged.
        INSERT INTO models_backup
            SELECT
                id,
                name,
                version,
                status,
                metadata,
                CASE
                    WHEN typeof(registered_at) = 'integer'
                        THEN registered_at
                    WHEN datetime(registered_at) IS NOT NULL
                        THEN CAST(strftime('%s', registered_at) AS INTEGER) * 1000000
                    ELSE CAST(registered_at AS INTEGER)
                END AS registered_at,
                CASE
                    WHEN typeof(updated_at) = 'integer'
                        THEN updated_at
                    WHEN datetime(updated_at) IS NOT NULL
                        THEN CAST(strftime('%s', updated_at) AS INTEGER) * 1000000
                    ELSE CAST(updated_at AS INTEGER)
                END AS updated_at,
                registered_by,
                description,
                weights_path,
                weights_size,
                weights_hash,
                parent_id
            FROM models;

        -- Step 3: drop old table and rename backup into place
        DROP TABLE models;
        ALTER TABLE models_backup RENAME TO models;

        -- Step 4: restore indexes
        CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
        CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);

        COMMIT;
        ",
    )
}

impl FileSystemStorage {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }
}

impl StorageBackend for FileSystemStorage {
    fn save(&mut self, model: &RegisteredModel, weights: &[u8]) -> Result<String> {
        // Validate model name and version to prevent path traversal
        validate_model_identity(&model.name, &model.version)?;

        let model_dir = self.root.join(&model.name).join(&model.version);
        std::fs::create_dir_all(&model_dir)?;

        let weights_path = model_dir.join("weights.bin");
        std::fs::write(&weights_path, weights)?;

        // Also save metadata
        #[cfg(feature = "serde")]
        {
            let metadata_path = model_dir.join("metadata.json");
            let metadata_json = serde_json::to_string_pretty(model)
                .map_err(|e| TuneError::Serialization(e.to_string()))?;
            std::fs::write(metadata_path, metadata_json)?;
        }

        let relative_path = format!("{}/{}/weights.bin", model.name, model.version);
        Ok(relative_path)
    }

    fn load(&self, path: &str) -> Result<Vec<u8>> {
        // Validate path to prevent path traversal
        validate_path(path)?;

        let full_path = self.root.join(path);
        std::fs::read(&full_path)
            .map_err(|e| TuneError::Storage(format!("Failed to load {path}: {e}")))
    }

    fn delete(&mut self, path: &str) -> Result<()> {
        // Validate path to prevent path traversal
        validate_path(path)?;

        let full_path = self.root.join(path);
        if full_path.exists() {
            std::fs::remove_file(&full_path)?;

            // Try to remove parent directories if empty
            if let Some(parent) = full_path.parent() {
                let _ = std::fs::remove_dir(parent);
            }
        }
        Ok(())
    }

    fn exists(&self, path: &str) -> bool {
        // Validate path - return false if path traversal attempted
        if validate_path(path).is_err() {
            return false;
        }
        self.root.join(path).exists()
    }

    fn list(&self) -> Vec<String> {
        let mut paths = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        if let Ok(versions) = std::fs::read_dir(entry.path()) {
                            for version_entry in versions.flatten() {
                                if version_entry.path().is_dir() {
                                    if let Some(version) = version_entry.file_name().to_str() {
                                        let weights_path = format!("{name}/{version}/weights.bin");
                                        if self.root.join(&weights_path).exists() {
                                            paths.push(weights_path);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        paths
    }
}

// Public methods are called externally only when the sqlite feature is enabled;
// without it the compiler sees them as unused.
#[cfg(feature = "sqlite")]
#[allow(dead_code)]
impl SqliteStorage {
    /// Create a new SQLite storage at the given database path.
    ///
    /// Weights are stored in a `weights/` subdirectory next to the database.
    pub fn new(db_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let db_path = db_path.as_ref();
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| TuneError::Storage(format!("Failed to open SQLite database: {e}")))?;

        // Create tables if they don't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata TEXT NOT NULL,
                registered_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                registered_by TEXT,
                description TEXT,
                weights_path TEXT,
                weights_size INTEGER,
                weights_hash TEXT,
                parent_id TEXT,
                UNIQUE(name, version)
            )",
            [],
        )
        .map_err(|e| TuneError::Storage(format!("Failed to create models table: {e}")))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)",
            [],
        )
        .map_err(|e| TuneError::Storage(format!("Failed to create name index: {e}")))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_models_status ON models(status)",
            [],
        )
        .map_err(|e| TuneError::Storage(format!("Failed to create status index: {e}")))?;

        // Upgrade migration (#1392): rename metadata_json → metadata for existing DBs.
        // Idempotent: succeeds silently if the column was already renamed (fresh or
        // previously-migrated DB).  Fail-closed: propagates unexpected errors.
        migrate_models_metadata_json_to_metadata(&conn).map_err(|e| {
            TuneError::Storage(format!("Migration metadata_json→metadata failed: {e}"))
        })?;

        // Upgrade migration (#1389): convert registered_at / updated_at TEXT → INTEGER.
        // Idempotent: no-op if columns already declare INTEGER (fresh or migrated DB).
        // Fail-closed: propagates unexpected errors.
        migrate_models_timestamps_text_to_integer(&conn).map_err(|e| {
            TuneError::Storage(format!(
                "Migration registered_at/updated_at TEXT→INTEGER failed: {e}"
            ))
        })?;

        let weights_dir = db_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("weights");
        std::fs::create_dir_all(&weights_dir)?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
            weights_dir,
        })
    }

    /// Create an in-memory SQLite storage (for testing).
    ///
    /// Uses a temporary directory for weights storage.
    pub fn in_memory() -> Result<Self> {
        let conn = rusqlite::Connection::open_in_memory()
            .map_err(|e| TuneError::Storage(format!("Failed to open in-memory SQLite: {e}")))?;

        conn.execute(
            "CREATE TABLE models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata TEXT NOT NULL,
                registered_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                registered_by TEXT,
                description TEXT,
                weights_path TEXT,
                weights_size INTEGER,
                weights_hash TEXT,
                parent_id TEXT,
                UNIQUE(name, version)
            )",
            [],
        )
        .map_err(|e| TuneError::Storage(format!("Failed to create models table: {e}")))?;

        // Use a unique temp directory for each in-memory instance
        let weights_dir =
            std::env::temp_dir().join(format!("lattice-tune-weights-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&weights_dir)?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
            weights_dir,
        })
    }

    /// Get the weights directory path.
    pub fn weights_dir(&self) -> &std::path::Path {
        &self.weights_dir
    }
}

#[cfg(feature = "sqlite")]
impl StorageBackend for SqliteStorage {
    fn save(&mut self, model: &RegisteredModel, weights: &[u8]) -> Result<String> {
        // Validate model name and version to prevent path traversal
        validate_model_identity(&model.name, &model.version)?;

        // Create weights file path
        let relative_path = format!("{}/{}/weights.bin", model.name, model.version);
        let weights_file = self.weights_dir.join(&model.name).join(&model.version);
        std::fs::create_dir_all(&weights_file)?;
        let weights_path = weights_file.join("weights.bin");

        // Write weights to file
        std::fs::write(&weights_path, weights)?;

        // Serialize metadata to JSON
        let metadata_json = serde_json::to_string(&model.metadata)
            .map_err(|e| TuneError::Serialization(e.to_string()))?;

        // Insert model record into database
        let conn = self
            .conn
            .lock()
            .map_err(|e| TuneError::Storage(format!("Failed to acquire database lock: {e}")))?;

        if let Err(e) = conn.execute(
            "INSERT INTO models (
                id, name, version, status, metadata,
                registered_at, updated_at, registered_by, description,
                weights_path, weights_size, weights_hash, parent_id
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            rusqlite::params![
                model.id.to_string(),
                model.name,
                model.version,
                model.status.to_string(),
                metadata_json,
                model.registered_at.timestamp_micros(),
                model.updated_at.timestamp_micros(),
                model.registered_by,
                model.description,
                relative_path,
                weights.len(),
                model.weights_hash,
                model.parent_id.map(|id| id.to_string()),
            ],
        ) {
            let _ = std::fs::remove_file(&weights_path);
            return Err(if e.to_string().contains("UNIQUE constraint failed") {
                TuneError::DuplicateModel {
                    name: model.name.clone(),
                    version: model.version.clone(),
                }
            } else {
                TuneError::Storage(format!("Failed to insert model record: {e}"))
            });
        }

        Ok(relative_path)
    }

    fn load(&self, path: &str) -> Result<Vec<u8>> {
        // Validate path to prevent path traversal
        validate_path(path)?;

        let full_path = self.weights_dir.join(path);
        std::fs::read(&full_path)
            .map_err(|e| TuneError::Storage(format!("Failed to load weights from {path}: {e}")))
    }

    fn delete(&mut self, path: &str) -> Result<()> {
        // Validate path to prevent path traversal
        validate_path(path)?;

        // Remove the weights file first. If this fails, we do not delete the DB
        // row so the registry remains consistent (no orphaned DB record pointing
        // to a missing file, and no missing DB record for an existing file).
        let full_path = self.weights_dir.join(path);
        if full_path.exists() {
            std::fs::remove_file(&full_path)?;

            // Try to remove parent directories if empty (best-effort)
            if let Some(parent) = full_path.parent() {
                let _ = std::fs::remove_dir(parent);
                if let Some(grandparent) = parent.parent() {
                    let _ = std::fs::remove_dir(grandparent);
                }
            }
        }

        // Only delete the DB row after the file has been successfully removed.
        let conn = self
            .conn
            .lock()
            .map_err(|e| TuneError::Storage(format!("Failed to acquire database lock: {e}")))?;

        conn.execute("DELETE FROM models WHERE weights_path = ?1", [path])
            .map_err(|e| TuneError::Storage(format!("Failed to delete model record: {e}")))?;

        Ok(())
    }

    fn exists(&self, path: &str) -> bool {
        // Validate path - return false if path traversal attempted
        if validate_path(path).is_err() {
            return false;
        }

        // Check both database and filesystem
        let db_exists = self
            .conn
            .lock()
            .ok()
            .map(|conn| {
                conn.query_row(
                    "SELECT 1 FROM models WHERE weights_path = ?1",
                    [path],
                    |_| Ok(()),
                )
                .is_ok()
            })
            .unwrap_or(false);

        db_exists && self.weights_dir.join(path).exists()
    }

    fn list(&self) -> Vec<String> {
        let Ok(conn) = self.conn.lock() else {
            return Vec::new();
        };

        let Ok(mut stmt) =
            conn.prepare("SELECT weights_path FROM models WHERE weights_path IS NOT NULL")
        else {
            return Vec::new();
        };

        let paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .ok()
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default();

        paths
    }
}
