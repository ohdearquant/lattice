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
//!
//! This is a native terminal tool built on a multi-threaded async runtime
//! (`#[tokio::main]`), which is categorically unsupported on
//! `wasm32-unknown-unknown` (no OS threads). It has no browser equivalent, so
//! the whole CLI is gated to native targets; wasm32 gets a no-op `main` below
//! so the crate's bin target still links (`required-features` can't exclude
//! a target, only a feature set, and `native` stays on by default).

#[cfg(not(target_arch = "wasm32"))]
mod cli {
    use std::process::ExitCode;
    use std::str::FromStr;
    use std::time::Instant;

    use lattice_embed::vision::{PoolingStrategy, VisionEmbeddingModel};
    use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};

    fn usage(msg: &str) -> ExitCode {
        eprintln!("ERROR: {msg}\n");
        eprintln!("{USAGE}");
        ExitCode::FAILURE
    }

    const USAGE: &str = "\
usage: embed [--model <NAME>] --text <TEXT> [--text <TEXT> ...] [--json]
       embed --image <PATH> [--image <PATH> ...] --vision-model-dir <DIR>
             [--prompt <TEXT>] [--pooling mean_visual|last_token] [--metal] [--json]

Generate embeddings for one or more text strings, or one or more images.
--image and --text are mutually exclusive.

text options:
  --model <NAME>   Embedding model to use. Default: bge-small-en-v1.5
                   Accepted: bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5,
                   multilingual-e5-small, multilingual-e5-base, all-minilm-l6-v2,
                   paraphrase-multilingual-minilm-l12-v2
                   Also accepts HuggingFace IDs like BAAI/bge-small-en-v1.5.
  --text <TEXT>    Text to embed. Repeat for multiple texts.
  --download-only  Ensure the model is downloaded and loadable, then exit (no --text needed).
                   Emits @@lattice {\"ev\":\"download_done\",\"ok\":bool} with --json.

image options:
  --image <PATH>         Path to a PNG or JPEG file to embed. Repeat for multiple images.
  --vision-model-dir <DIR>  Directory of a Qwen3.5 vision-language checkpoint (required
                            with --image).
  --prompt <TEXT>        Text prompt assembled around each image. Default: empty.
  --pooling <STRATEGY>   mean_visual (default) or last_token.
  --metal                Run the ViT forward pass on the Metal GPU instead of the CPU.
                         Fails with a clear error (no silent CPU fallback) if no Metal
                         device is available on this build/machine.

common options:
  --json           Emit a structured @@lattice {\"ev\":\"embed_done\",...} line to stdout.
  -h, --help       Print this help and exit.
";

    #[tokio::main]
    pub(crate) async fn main() -> ExitCode {
        let args: Vec<String> = std::env::args().collect();

        let mut model_name: Option<String> = None;
        let mut texts: Vec<String> = Vec::new();
        let mut images: Vec<String> = Vec::new();
        let mut vision_model_dir: Option<String> = None;
        let mut prompt = String::new();
        let mut pooling_arg: Option<String> = None;
        let mut use_metal = false;
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
                "--image" => {
                    i += 1;
                    let Some(v) = args.get(i) else {
                        return usage("--image requires an argument");
                    };
                    images.push(v.clone());
                }
                "--vision-model-dir" => {
                    i += 1;
                    let Some(v) = args.get(i) else {
                        return usage("--vision-model-dir requires an argument");
                    };
                    vision_model_dir = Some(v.clone());
                }
                "--prompt" => {
                    i += 1;
                    let Some(v) = args.get(i) else {
                        return usage("--prompt requires an argument");
                    };
                    prompt = v.clone();
                }
                "--pooling" => {
                    i += 1;
                    let Some(v) = args.get(i) else {
                        return usage("--pooling requires an argument");
                    };
                    pooling_arg = Some(v.clone());
                }
                "--metal" => {
                    use_metal = true;
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

        if !images.is_empty() && !texts.is_empty() {
            return usage("--image and --text are mutually exclusive");
        }
        if !images.is_empty() && vision_model_dir.is_none() {
            return usage("--vision-model-dir is required with --image");
        }
        if images.is_empty() && vision_model_dir.is_some() {
            return usage("--vision-model-dir requires --image");
        }
        let pooling = match pooling_arg.as_deref() {
            None | Some("mean_visual") => PoolingStrategy::MeanVisualTokens,
            Some("last_token") => PoolingStrategy::LastToken,
            Some(other) => {
                return usage(&format!(
                    "--pooling must be 'mean_visual' or 'last_token', got '{other}'"
                ));
            }
        };

        if !images.is_empty() {
            return run_image_mode(
                &images,
                vision_model_dir.as_deref().unwrap(),
                &prompt,
                pooling,
                use_metal,
                emit_json,
            );
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
        eprintln!("Generating embeddings (model loads on first call, may download ~130 MB)...");

        let service = NativeEmbeddingService::with_model(model);

        // --download-only: ensure the model is present (downloading + checksum-verifying if
        // needed) and loadable, then exit without running any encode pass.
        if download_only {
            match service.ensure_loaded().await {
                Ok(()) => {
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

    /// `--image` mode: pool one embedding per image through a loaded
    /// vision-language checkpoint. Kept synchronous (unlike the text path's
    /// `NativeEmbeddingService`, which downloads over the network) since
    /// checkpoint loading and pooled inference here are local, CPU/GPU-bound
    /// work with nothing to `.await`.
    fn run_image_mode(
        images: &[String],
        vision_model_dir: &str,
        prompt: &str,
        pooling: PoolingStrategy,
        use_metal: bool,
        emit_json: bool,
    ) -> ExitCode {
        eprintln!("Loading vision-language checkpoint from {vision_model_dir}...");
        let model =
            match VisionEmbeddingModel::from_directory(std::path::Path::new(vision_model_dir)) {
                Ok(m) => m,
                Err(err) => {
                    eprintln!("ERROR: failed to load vision-language checkpoint: {err}");
                    return ExitCode::FAILURE;
                }
            };
        let dims = model.dimensions();
        eprintln!("Dimensions: {dims}");
        eprintln!("Images:     {}", images.len());
        eprintln!();

        let t0 = Instant::now();
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(images.len());
        for path in images {
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!("ERROR: failed to read image '{path}': {err}");
                    return ExitCode::FAILURE;
                }
            };
            let result = if use_metal {
                model.embed_image_metal(&bytes, prompt, pooling)
            } else {
                model.embed_image(&bytes, prompt, pooling)
            };
            let embedding = match result {
                Ok(v) => v,
                Err(err) if use_metal => {
                    eprintln!(
                        "ERROR: Metal embedding failed for '{path}': {err}\n\
                         Falling back to the CPU path is not automatic -- rerun without \
                         --metal if that is what you want."
                    );
                    return ExitCode::FAILURE;
                }
                Err(err) => {
                    eprintln!("ERROR: embedding failed for '{path}': {err}");
                    return ExitCode::FAILURE;
                }
            };
            embeddings.push(embedding);
        }
        let elapsed_ms = t0.elapsed().as_millis();
        let count = embeddings.len();

        let mut cosine: Vec<Vec<f32>> = Vec::with_capacity(count);
        for i in 0..count {
            let mut row = Vec::with_capacity(count);
            for j in 0..count {
                let sim = lattice_embed::utils::cosine_similarity(&embeddings[i], &embeddings[j]);
                row.push(sim);
            }
            cosine.push(row);
        }
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
                "model": vision_model_dir,
                "dims": dims,
                "count": count,
                "images": count,
                "cosine": cosine,
                "preview": preview,
                "ms": elapsed_ms,
            });
            println!("@@lattice {obj}");
        }

        ExitCode::SUCCESS
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    cli::main()
}

// wasm32 has no terminal environment for this CLI to run in (see module doc
// comment above); the crate's wasm-facing surface is the `wasm` feature's
// wasm-bindgen bindings instead. This no-op keeps the bin target linkable.
#[cfg(target_arch = "wasm32")]
fn main() {}
