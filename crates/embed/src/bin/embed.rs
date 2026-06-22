//! CLI tool for generating text embeddings using lattice-embed.
//!
//! # Usage
//!
//! ```text
//! embed --model bge-small-en-v1.5 --text "hello" --text "world" [--json]
//! ```
//!
//! When `--json` is set, emits a single `@@lattice {"ev":"embed_done",...}` line
//! to stdout in addition to the human-readable summary.

use std::process::ExitCode;
use std::str::FromStr;
use std::time::Instant;

use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};

fn usage(msg: &str) -> ExitCode {
    eprintln!("ERROR: {msg}\n");
    eprintln!("{USAGE}");
    ExitCode::FAILURE
}

const USAGE: &str = "\
usage: embed [--model <NAME>] --text <TEXT> [--text <TEXT> ...] [--json]

Generate embeddings for one or more text strings.

options:
  --model <NAME>   Embedding model to use. Default: bge-small-en-v1.5
                   Accepted: bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5,
                   multilingual-e5-small, multilingual-e5-base, all-minilm-l6-v2,
                   paraphrase-multilingual-minilm-l12-v2
                   Also accepts HuggingFace IDs like BAAI/bge-small-en-v1.5.
  --text <TEXT>    Text to embed. Repeat for multiple texts.
  --json           Emit a structured @@lattice {\"ev\":\"embed_done\",...} line to stdout.
  --download-only  Ensure the model is downloaded and loadable, then exit (no --text needed).
                   Emits @@lattice {\"ev\":\"download_done\",\"ok\":bool} with --json.
  -h, --help       Print this help and exit.
";

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let mut model_name: Option<String> = None;
    let mut texts: Vec<String> = Vec::new();
    let mut emit_json: bool = false;
    let mut download_only: bool = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--model requires an argument");
                };
                model_name = Some(v.clone());
            }
            "--text" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--text requires an argument");
                };
                texts.push(v.clone());
            }
            "--json" => {
                emit_json = true;
            }
            "--download-only" => {
                download_only = true;
            }
            "--help" | "-h" => {
                eprintln!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => return usage(&format!("unknown argument: {other}")),
        }
        i += 1;
    }

    if !download_only && texts.is_empty() {
        return usage("at least one --text argument is required");
    }

    let model = match model_name {
        Some(ref name) => match EmbeddingModel::from_str(name) {
            Ok(m) => m,
            Err(_) => {
                return usage(&format!(
                    "--model '{name}' is not a recognised embedding model"
                ));
            }
        },
        None => EmbeddingModel::default(),
    };

    eprintln!("Model:      {model}");
    eprintln!("Dimensions: {}", model.dimensions());
    eprintln!("Texts:      {}", texts.len());
    eprintln!();
    eprintln!("Generating embeddings (model loads on first call — may download ~130 MB)...");

    let service = NativeEmbeddingService::with_model(model);

    // --download-only: ensure the model is present (downloading + checksum-verifying if
    // needed) and loadable, then exit. A single warmup embedding forces the lazy
    // ensure_model_files + model load without printing the similarity report.
    if download_only {
        match service.embed(&["warmup".to_string()], model).await {
            Ok(_) => {
                eprintln!("Model {model} is downloaded and ready.");
                if emit_json {
                    let obj = serde_json::json!({
                        "ev": "download_done",
                        "model": model.to_string(),
                        "ok": true,
                    });
                    println!("@@lattice {obj}");
                }
                return ExitCode::SUCCESS;
            }
            Err(err) => {
                eprintln!("ERROR: model download/load failed: {err}");
                if emit_json {
                    let obj = serde_json::json!({
                        "ev": "download_done",
                        "model": model.to_string(),
                        "ok": false,
                        "error": err.to_string(),
                    });
                    println!("@@lattice {obj}");
                }
                return ExitCode::FAILURE;
            }
        }
    }

    let t0 = Instant::now();
    let embeddings = match service.embed(&texts, model).await {
        Ok(e) => e,
        Err(err) => {
            eprintln!("ERROR: embedding failed: {err}");
            return ExitCode::FAILURE;
        }
    };
    let elapsed_ms = t0.elapsed().as_millis();

    if embeddings.is_empty() {
        eprintln!("ERROR: service returned zero embeddings");
        return ExitCode::FAILURE;
    }

    let dims = embeddings[0].len();
    let count = embeddings.len();

    // Build NxN pairwise cosine matrix.
    let mut cosine: Vec<Vec<f32>> = Vec::with_capacity(count);
    for i in 0..count {
        let mut row = Vec::with_capacity(count);
        for j in 0..count {
            let sim = lattice_embed::utils::cosine_similarity(&embeddings[i], &embeddings[j]);
            row.push(sim);
        }
        cosine.push(row);
    }

    // Build preview: first 8 dims of each vector.
    let preview_len = dims.min(8);
    let preview: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|e| e[..preview_len].to_vec())
        .collect();

    eprintln!("=== Embedding Results ===");
    eprintln!("Dims:    {dims}");
    eprintln!("Count:   {count}");
    eprintln!("Elapsed: {elapsed_ms}ms");
    eprintln!();
    eprintln!("Pairwise cosine similarity:");
    for (i, row) in cosine.iter().enumerate() {
        let vals: Vec<String> = row.iter().map(|v| format!("{v:.4}")).collect();
        eprintln!("  [{i}] {}", vals.join("  "));
    }

    if emit_json {
        let obj = serde_json::json!({
            "ev": "embed_done",
            "model": model.to_string(),
            "dims": dims,
            "count": count,
            "cosine": cosine,
            "preview": preview,
            "ms": elapsed_ms,
        });
        println!("@@lattice {obj}");
    }

    ExitCode::SUCCESS
}
