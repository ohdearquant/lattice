//! Embedding quality comparison: BGE-small-en-v1.5 (384d) vs Qwen3-Embedding-0.6B (1024d).
//!
//! Loads real atoms/memories from the lattice database, embeds a sample with both models,
//! then runs retrieval queries and compares rankings side-by-side.
//!
//! Usage:
//!   cargo run --release --features "backfill,metal-gpu" --bin bench_embed_quality -p lattice-inference

#[cfg(not(feature = "backfill"))]
fn main() {
    eprintln!("This binary requires the `backfill` feature (for rusqlite).");
    eprintln!(
        "Run with: cargo run --features backfill --bin bench_embed_quality -p lattice-inference"
    );
    std::process::exit(1);
}

#[cfg(feature = "backfill")]
use std::path::Path;
#[cfg(feature = "backfill")]
use std::time::Instant;

#[cfg(feature = "backfill")]
fn main() {
    let home = std::env::var("HOME").unwrap();
    let db_path = format!("{home}/.lattice/lattice.db");

    if !Path::new(&db_path).exists() {
        eprintln!("Database not found at {db_path}");
        std::process::exit(1);
    }

    // Load sample documents from the database.
    let conn =
        rusqlite::Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();

    // Sample diverse atoms + memories for the corpus.
    let mut docs: Vec<(String, String)> = Vec::new(); // (id_label, content)

    // Atoms — sample 200 diverse ones
    {
        let mut stmt = conn
            .prepare("SELECT id, content FROM atoms WHERE length(content) > 50 ORDER BY RANDOM() LIMIT 200")
            .unwrap();
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok((id, content))
            })
            .unwrap();
        for row in rows {
            let (id, content) = row.unwrap();
            docs.push((format!("atom:{}", &id[..8]), content));
        }
    }

    // Memories — sample 100
    {
        let mut stmt = conn
            .prepare("SELECT id, content FROM memories WHERE length(content) > 50 ORDER BY RANDOM() LIMIT 100")
            .unwrap();
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok((id, content))
            })
            .unwrap();
        for row in rows {
            let (id, content) = row.unwrap();
            docs.push((format!("mem:{}", &id[..8]), content));
        }
    }

    println!("Loaded {} documents from database", docs.len());
    let avg_len: f64 = docs.iter().map(|(_, c)| c.len() as f64).sum::<f64>() / docs.len() as f64;
    println!("Average content length: {avg_len:.0} chars\n");

    // Test queries — mix of English, Chinese, and domain-specific.
    let queries = vec![
        // English technical queries
        ("HNSW vector search performance", "en-tech"),
        ("Metal GPU inference optimization", "en-tech"),
        ("embedding model comparison quality", "en-tech"),
        ("memory recall semantic search", "en-tech"),
        ("Rust async runtime tokio", "en-tech"),
        // English domain queries
        ("health tracking pills medication", "en-domain"),
        ("task management GTD workflow", "en-domain"),
        ("meeting notes Jason discussion", "en-domain"),
        // Chinese queries
        ("向量搜索性能优化", "zh"),
        ("数据库索引和查询", "zh"),
        ("机器学习模型推理", "zh"),
        // Cross-domain
        ("compliance security authorization pipeline", "en-tech"),
        ("lambda orchestration multi-agent", "en-tech"),
    ];

    // Load BGE-small model.
    println!("Loading BGE-small-en-v1.5...");
    let bge_dir_path = format!("{home}/.cache/lattice/models/bge-small-en-v1.5");
    let bge_dir = Path::new(&bge_dir_path);
    let bge_model = if bge_dir.join("model.safetensors").exists() {
        Some(lattice_inference::BertModel::from_pretrained("bge-small-en-v1.5").unwrap())
    } else {
        eprintln!("  BGE-small not found, trying download...");
        match lattice_inference::BertModel::from_pretrained("bge-small-en-v1.5") {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("  BGE-small unavailable: {e}");
                None
            }
        }
    };

    // Load Qwen3 model.
    println!("Loading Qwen3-Embedding-0.6B...");
    let qwen_dir = format!("{home}/.lattice/models/qwen3-embedding-0.6b");
    let qwen_model = lattice_inference::QwenModel::from_directory(Path::new(&qwen_dir)).unwrap();
    println!("  Qwen3 GPU: {}\n", qwen_model.has_gpu());

    // Embed all documents with both models.
    println!("Embedding {} documents...", docs.len());
    let contents: Vec<&str> = docs.iter().map(|(_, c)| c.as_str()).collect();

    // Qwen3 embeddings
    let t = Instant::now();
    let qwen_embeddings: Vec<Vec<f32>> = contents
        .iter()
        .map(|text| qwen_model.encode(text).unwrap())
        .collect();
    let qwen_ms = t.elapsed().as_millis();
    println!(
        "  Qwen3: {qwen_ms}ms ({:.1}ms/doc)",
        qwen_ms as f64 / docs.len() as f64
    );

    // BGE embeddings (if available)
    let bge_embeddings: Option<Vec<Vec<f32>>> = bge_model.as_ref().map(|model| {
        let t = Instant::now();
        let embs: Vec<Vec<f32>> = contents
            .iter()
            .map(|text| model.encode(text).unwrap())
            .collect();
        let ms = t.elapsed().as_millis();
        println!(
            "  BGE:   {ms}ms ({:.1}ms/doc)",
            ms as f64 / docs.len() as f64
        );
        embs
    });

    let sep = "=".repeat(80);
    println!("\n{sep}");
    println!("=== Retrieval Quality Comparison ===");
    println!("{sep}\n");

    // For each query, embed with both models, rank documents, show top-5.
    for (query, category) in &queries {
        println!("Query [{category}]: \"{query}\"");

        // Qwen3 ranking
        let qwen_query = qwen_model.encode(query).unwrap();
        let mut qwen_scores: Vec<(usize, f64)> = qwen_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, cosine_sim(&qwen_query, emb)))
            .collect();
        qwen_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("  Qwen3 Top-5:");
        for (rank, &(idx, score)) in qwen_scores.iter().take(5).enumerate() {
            let (label, content) = &docs[idx];
            let preview: String = content.chars().take(80).collect();
            println!("    #{}: [{label}] {score:.4} | {preview}...", rank + 1);
        }

        // BGE ranking (if available)
        if let Some(ref bge_embs) = bge_embeddings {
            let bge_query = bge_model.as_ref().unwrap().encode(query).unwrap();
            let mut bge_scores: Vec<(usize, f64)> = bge_embs
                .iter()
                .enumerate()
                .map(|(i, emb)| (i, cosine_sim(&bge_query, emb)))
                .collect();
            bge_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("  BGE Top-5:");
            for (rank, &(idx, score)) in bge_scores.iter().take(5).enumerate() {
                let (label, content) = &docs[idx];
                let preview: String = content.chars().take(80).collect();
                println!("    #{}: [{label}] {score:.4} | {preview}...", rank + 1);
            }

            // Overlap: how many of Qwen3's top-10 appear in BGE's top-10?
            let qwen_top10: Vec<usize> = qwen_scores.iter().take(10).map(|&(i, _)| i).collect();
            let bge_top10: Vec<usize> = bge_scores.iter().take(10).map(|&(i, _)| i).collect();
            let overlap = qwen_top10.iter().filter(|i| bge_top10.contains(i)).count();
            println!("  Top-10 overlap: {overlap}/10");
        }

        println!();
    }

    // Summary statistics
    println!("=== Summary ===");
    println!("Corpus: {} docs (avg {avg_len:.0} chars)", docs.len());
    println!("Qwen3: 1024d, GPU-accelerated, multilingual");
    if bge_embeddings.is_some() {
        println!("BGE:   384d, CPU-only, English-only");
    }
}

#[cfg(feature = "backfill")]
fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let min_len = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..min_len {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}
