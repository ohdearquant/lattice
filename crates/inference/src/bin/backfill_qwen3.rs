//! Backfill embeddings from BGE-Small-384d to Qwen3-1024d.
//!
//! Reads atoms from lattice.db, encodes with Qwen3-Embedding-0.6B,
//! writes new 1024d embeddings back, replacing the old 384d ones.
//!
//! Usage: cargo run --release --features f16 --bin backfill_qwen3

use lattice_inference::QwenModel;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let model_dir = std::env::var("LATTICE_QWEN_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME not set");
            PathBuf::from(home)
                .join(".lattice")
                .join("models")
                .join("qwen3-embedding-0.6b")
        });

    let db_path = std::env::var("LATTICE_DB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME not set");
            PathBuf::from(home).join(".lattice").join("lattice.db")
        });

    eprintln!("=== Qwen3-Embedding-0.6B Backfill ===");
    eprintln!("Model: {}", model_dir.display());
    eprintln!("DB:    {}", db_path.display());

    // Load model.
    eprintln!("\nLoading model...");
    let t0 = Instant::now();
    let model = QwenModel::from_directory(&model_dir).expect("failed to load Qwen3 model");
    eprintln!(
        "Model loaded in {:.1}s (dim={})",
        t0.elapsed().as_secs_f32(),
        model.dimensions()
    );

    // Open DB.
    let conn = rusqlite::Connection::open(&db_path).expect("failed to open DB");
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .expect("pragma");

    // Count atoms that need re-embedding.
    let total: usize = conn
        .query_row("SELECT COUNT(*) FROM id_mappings", [], |row| row.get(0))
        .expect("count");

    eprintln!("Atoms to re-embed: {total}");
    if total == 0 {
        eprintln!("Nothing to do.");
        return;
    }

    // Read all atom content + mapping info.
    let mut stmt = conn
        .prepare(
            "SELECT m.namespace, m.entry_uuid, m.embedding_id, a.content
             FROM id_mappings m
             JOIN atoms a ON a.namespace = m.namespace AND a.id = m.entry_uuid
             WHERE a.deleted_at IS NULL",
        )
        .expect("prepare");

    let rows: Vec<(String, String, Vec<u8>, String)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .expect("query")
        .filter_map(|r| r.ok())
        .collect();

    let actual = rows.len();
    eprintln!("Loaded {actual} atom texts");

    // Prepare update statement.
    let mut update_stmt = conn
        .prepare(
            "UPDATE embeddings SET vector = ?1, dims = ?2
             WHERE namespace = ?3 AND embedding_id = ?4",
        )
        .expect("prepare update");

    let new_dims: i64 = model.dimensions() as i64;
    let start = Instant::now();
    let mut done = 0usize;
    let mut errors = 0usize;

    for (namespace, _entry_uuid, embedding_id, content) in &rows {
        // Cap at ~600 tokens (~2400 chars). Beyond this, embedding quality
        // plateaus but compute scales quadratically.
        let text = if content.len() > 2400 {
            let mut end = 2400;
            while end < content.len() && !content.is_char_boundary(end) {
                end += 1;
            }
            &content[..end]
        } else {
            content.as_str()
        };

        match model.encode(text) {
            Ok(embedding) => {
                // Convert f32 vec to bytes (little-endian).
                let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

                update_stmt
                    .execute(rusqlite::params![bytes, new_dims, namespace, embedding_id])
                    .expect("update embedding");
            }
            Err(e) => {
                errors += 1;
                if errors <= 5 {
                    eprintln!("  ERROR encoding atom: {e}");
                }
            }
        }

        done += 1;
        if done % 100 == 0 || done == actual {
            let elapsed = start.elapsed().as_secs_f32();
            let rate = done as f32 / elapsed;
            let eta_secs = (actual - done) as f32 / rate;
            let eta_min = eta_secs / 60.0;
            eprint!(
                "\r  [{done}/{actual}] {:.0}/s | elapsed {:.0}m | ETA {:.0}m | errors: {errors}   ",
                rate,
                elapsed / 60.0,
                eta_min,
            );
        }
    }

    eprintln!(
        "\n\nDone! {done} embeddings updated ({errors} errors) in {:.1}m",
        start.elapsed().as_secs_f32() / 60.0
    );

    // Verify.
    let check: (i64, i64) = conn
        .query_row(
            "SELECT dims, COUNT(*) FROM embeddings GROUP BY dims ORDER BY COUNT(*) DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("verify");
    eprintln!(
        "Verification: {}/{actual} embeddings now at {}d",
        check.1, check.0
    );
}
