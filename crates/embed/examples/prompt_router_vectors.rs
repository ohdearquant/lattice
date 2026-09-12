//! Export BGE-small prompt vectors for a separate adapter-routing experiment.
//! Arguments: labelled JSONL input and output JSON path. Input rows contain
//! `prompt`, `label` (0 or 1), and `split` (`train` or `test`).

#[cfg(feature = "native")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};
    use std::{collections::HashSet, time::Instant};
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected input JSONL and output JSON paths".into());
    }
    let model = EmbeddingModel::BgeSmallEnV15;
    let service = NativeEmbeddingService::with_model(model);
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    for line in std::fs::read_to_string(&args[1])?.lines() {
        let mut row: serde_json::Value = serde_json::from_str(line)?;
        let prompt = row["prompt"].as_str().ok_or("missing prompt")?.to_string();
        if prompt.trim().is_empty() || !seen.insert(prompt.to_lowercase()) {
            return Err("empty or duplicate prompt".into());
        }
        if !matches!(row["label"].as_u64(), Some(0 | 1))
            || !matches!(row["split"].as_str(), Some("train" | "test"))
        {
            return Err("invalid label or split".into());
        }
        let start = Instant::now();
        let vectors = service.embed_query(&[prompt], model).await?;
        let elapsed = start.elapsed().as_secs_f64();
        let vector = vectors.first().ok_or("empty embedding result")?;
        if vector.len() != 384 || vector.iter().any(|v| !v.is_finite()) {
            return Err("invalid embedding".into());
        }
        row["vector"] = serde_json::json!(vector);
        row["embedding_seconds"] = serde_json::json!(elapsed);
        rows.push(row);
    }
    let result = serde_json::json!({"model": "bge-small-en-v1.5", "dimension": 384,
        "mrl": false, "role": "query", "rows": rows});
    std::fs::write(&args[2], serde_json::to_vec(&result)?)?;
    println!(
        "exported {} vectors; model=bge-small-en-v1.5 dimension=384 mrl=false",
        rows.len()
    );
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    panic!("enable native");
}
