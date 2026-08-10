//! `lattice` CLI - interactive chat, HTTP serve, and preflight subcommands. See [docs/capability-matrix.md](../../../../docs/capability-matrix.md).
//!
//! # Usage
//!
//! ```text
//! lattice chat --model /path/to/model [--max-tokens 256] [--temperature 0.7]
//! lattice serve --model /path/to/model [--host 127.0.0.1] [--port 8080] [--max-tokens 256]
//! lattice doctor --model /path/to/model [--context 4096]
//! lattice prune-score --q4-dir /path/to/model-q4 --tokenizer-dir /path/to/model \
//!   --calibration-corpus calibration.txt --validation-corpus validation.txt \
//!   --prune-layers 4 --output lattice_pruning.json
//! ```

use clap::{Parser, Subcommand};

mod chat;
mod doctor;
mod prune_score;
mod serve;

#[derive(Parser)]
#[command(name = "lattice", about = "Pure-Rust transformer inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive chat with a model
    Chat {
        /// Path to model directory (SafeTensors, or a native Q4 quantized
        /// directory produced by `quantize_q4`)
        #[arg(long)]
        model: String,
        /// Maximum tokens to generate per response
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Directory containing tokenizer.json, when it is not shipped inside
        /// --model (only needed for Q4 directories produced without a
        /// co-located tokenizer; safetensors directories always ship one).
        #[arg(long)]
        tokenizer_dir: Option<String>,
    },
    /// Start HTTP server with OpenAI-compatible API
    Serve {
        /// Path to model directory (SafeTensors, or a native Q4 quantized
        /// directory produced by `quantize_q4`)
        #[arg(long)]
        model: String,
        /// Host address to bind (default: 127.0.0.1; use 0.0.0.0 for LAN)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Maximum tokens to generate per request (default when request omits max_tokens)
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Model identifier echoed in responses (defaults to the model path basename)
        #[arg(long)]
        model_id: Option<String>,
        /// Directory containing tokenizer.json, when it is not shipped inside
        /// --model (only needed for Q4 directories produced without a
        /// co-located tokenizer; safetensors directories always ship one).
        #[arg(long)]
        tokenizer_dir: Option<String>,
        /// Cap on outstanding (queued + in-flight) requests to the Metal GPU
        /// worker (issue #932) before new requests are rejected with HTTP
        /// 503. Only applies to Q4/Metal-backed serving; the CPU backend has
        /// no shared worker queue to bound. Conservative default: this
        /// worker serializes all generation onto one dedicated thread, so a
        /// deep queue just means memory growth with no throughput benefit.
        /// Must be between 1 and `tokio::sync::Semaphore::MAX_PERMITS`
        /// (issue #939): zero would admit nothing (every request fails
        /// admission), and clap rejects anything larger here instead of
        /// deferring to `MetalWorker::spawn`'s own
        /// `Semaphore::new`-precondition panic.
        #[arg(
            long,
            default_value = "32",
            value_parser = clap::builder::RangedU64ValueParser::<usize>::new()
                .range(1..=(tokio::sync::Semaphore::MAX_PERMITS as u64))
        )]
        max_pending: usize,
        /// Eagerly load vision weights at startup instead of on the first
        /// image request (issue #1336). Off by default: lazy loading keeps
        /// text-only startup time and resident memory unchanged from a
        /// text-only checkpoint, since vision weights are never read at all
        /// unless an image request arrives. Pass this flag to trade a
        /// longer, predictable startup (and the vision weights' resident
        /// memory footprint held from the first request onward instead of
        /// only after it) for eliminating the first image request's extra
        /// load latency. Only affects Q4/Metal-backed vision-capable
        /// checkpoints; text-only and non-Metal backends ignore it. If the
        /// eager load fails, startup still succeeds: the server warns on
        /// stderr and falls back to the normal lazy load on the first image
        /// request, exactly as if this flag had not been passed.
        #[arg(long)]
        preload_vision: bool,
    },
    /// Preflight check: memory fit and artifact compatibility, without
    /// loading any model weights (config + tensor index inspection only).
    Doctor {
        /// Path to model directory (SafeTensors, or a native Q4 quantized
        /// directory produced by `quantize_q4`)
        #[arg(long)]
        model: String,
        /// Context length to check feasibility for. When omitted, only the
        /// maximum feasible context length is reported.
        #[arg(long)]
        context: Option<usize>,
        /// Directory containing tokenizer.json, when it is not shipped inside
        /// --model (only needed for Q4 directories produced without a
        /// co-located tokenizer; safetensors directories always ship one).
        #[arg(long)]
        tokenizer_dir: Option<String>,
    },
    /// Score layer importance on a calibration corpus and PPL-gate a pruning plan.
    PruneScore {
        #[command(flatten)]
        args: prune_score::Args,
    },
}

// ---------------------------------------------------------------------------
// backend: model-directory format detection + Q4/Metal loading
//
// `lattice chat`/`lattice serve` originally only understood a safetensors
// directory (`model.safetensors` or a sharded index). Native Q4 quantized
// directories (per-tensor `.q4` files, the output of `quantize_q4`) route to
// the Metal GPU forward pass instead. Safetensors directories are completely
// unaffected: `detect_format` returns `Safetensors` for them exactly as
// before, and the safetensors load path is untouched.
//
// The detector itself (`ModelFormat` + `detect_format` + the two error
// message helpers) now lives in `lattice_inference::model_format` (ADR-080
// amendment, #829): it is shared, unmodified, with `lattice_serve.rs` and
// `chat_metal.rs`, which cannot see a `pub(crate)` item defined in this
// binary's own crate root. `backend` here is a local alias so every existing
// `backend::...` / `crate::backend::...` call site below is unchanged.
// ---------------------------------------------------------------------------

use lattice_inference::model_format as backend;

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Chat {
            model,
            max_tokens,
            temperature,
            tokenizer_dir,
        } => {
            chat::run_chat(&model, max_tokens, temperature, tokenizer_dir.as_deref());
        }
        Command::Serve {
            model,
            host,
            port,
            max_tokens,
            model_id,
            tokenizer_dir,
            max_pending,
            preload_vision,
        } => {
            use std::path::Path;
            use std::sync::Arc;
            use std::sync::atomic::AtomicU64;

            // Derive a model identifier from the path basename when --model-id
            // is not provided.
            let served_model_id = model_id.unwrap_or_else(|| {
                Path::new(&model)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("lattice")
                    .to_string()
            });

            let model_path = Path::new(&model);
            let format = backend::detect_format(model_path);

            eprintln!("Loading model from {model}...");
            let model_backend: serve::ModelBackend = match format {
                backend::ModelFormat::Safetensors => {
                    match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(
                        model_path,
                    ) {
                        Ok(m) => serve::ModelBackend::Cpu(Arc::new(m)),
                        Err(e) => {
                            eprintln!("Error: failed to load model: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                backend::ModelFormat::Q4 => {
                    #[cfg(feature = "metal-gpu")]
                    {
                        let tokenizer_dir_path =
                            tokenizer_dir.as_ref().map(std::path::PathBuf::from);
                        match serve::ModelBackend::spawn_metal(
                            model_path.to_path_buf(),
                            tokenizer_dir_path,
                            max_pending,
                            preload_vision,
                        ) {
                            Ok((backend, _max_context)) => backend,
                            Err(e) => {
                                eprintln!("Error: failed to load Q4 model: {e}");
                                std::process::exit(1);
                            }
                        }
                    }
                    #[cfg(not(feature = "metal-gpu"))]
                    {
                        let _ = &tokenizer_dir;
                        let _ = max_pending;
                        let _ = preload_vision;
                        eprintln!("Error: {}", backend::metal_gpu_required_message(model_path));
                        std::process::exit(1);
                    }
                }
                backend::ModelFormat::Unknown => {
                    eprintln!(
                        "Error: {}",
                        backend::unrecognized_format_message(model_path)
                    );
                    std::process::exit(1);
                }
                // Any format this binary doesn't yet know how to serve is
                // handled the same way as `Unknown`: report it and exit,
                // rather than silently guessing a backend.
                _ => {
                    eprintln!(
                        "Error: {}",
                        backend::unrecognized_format_message(model_path)
                    );
                    std::process::exit(1);
                }
            };
            eprintln!("Model loaded. Serving as '{served_model_id}'.");

            // `/v1/embeddings` needs its own f16-packed vision-language
            // checkpoint load, independent of `model_backend` above (see
            // `lattice_inference::serve::embeddings`'s module doc comment
            // for why the two loaders can't share weights). Best-effort,
            // same policy as `--preload-vision` failing: warn and continue
            // with embeddings disabled rather than aborting startup, since a
            // checkpoint that isn't vision-language-shaped is an expected,
            // common case (most `lattice serve` deployments serve chat
            // only).
            let embedding_model =
                match lattice_inference::serve::embeddings::EmbeddingModel::from_directory(
                    model_path,
                ) {
                    Ok(embedding_model) => {
                        eprintln!(
                            "Embeddings enabled: pooled {}-dim vectors from {model}.",
                            embedding_model.dimensions()
                        );
                        Some(Arc::new(embedding_model))
                    }
                    Err(err) => {
                        eprintln!("Embeddings disabled ({model}): {err}");
                        None
                    }
                };

            let state = serve::AppState {
                model: model_backend,
                default_max_tokens: max_tokens,
                max_tokens_cap: 4096,
                model_id: served_model_id.clone(),
                request_counter: Arc::new(AtomicU64::new(0)),
                embedding_model,
            };

            let app = serve::router(state);

            let addr = format!("{host}:{port}");
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    drop(app);
                    eprintln!("Error: failed to bind to {addr}: {e}");
                    std::process::exit(1);
                }
            };
            eprintln!(
                "Listening on {addr}  (model: {served_model_id}, max_tokens default: {max_tokens})"
            );
            eprintln!("  POST /v1/chat/completions");
            eprintln!("  GET  /health");

            if let Err(e) = lattice_inference::serve::serve_until_shutdown(listener, app).await {
                eprintln!("Server error: {e}");
                std::process::exit(1);
            }
        }
        Command::Doctor {
            model,
            context,
            tokenizer_dir,
        } => {
            use std::path::Path;

            let model_path = Path::new(&model);
            let tokenizer_dir_path = tokenizer_dir.as_deref().map(Path::new);
            match doctor::build_report(model_path, tokenizer_dir_path, context, None) {
                Ok(report) => {
                    println!("{report}");
                    if !report.is_ready() {
                        eprintln!("doctor: model is NOT usable as configured (see reasons above)");
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Command::PruneScore { args } => match prune_score::run(&args) {
            Ok(true) => {}
            Ok(false) => std::process::exit(1),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
    }
}

// ─── #939 CLI boundary tests: `--max-pending` range validation ────────────
//
// clap's own `value_parser!(usize).range(1..=Semaphore::MAX_PERMITS)` on the
// `Serve::max_pending` field (above) is the ONLY validation this binary
// needs for zero / too-large values -- unlike `lattice_serve.rs`'s hand
// rolled argv parser, clap already rejects a malformed string (`abc`,
// `-1`) itself, before this range check ever runs. These tests exercise
// that `value_parser` wiring directly through `Cli::try_parse_from`,
// rather than duplicating the range logic anywhere in this binary.
#[cfg(test)]
mod max_pending_cli_tests {
    use super::*;

    fn parse_max_pending(args: &[&str]) -> Result<usize, clap::Error> {
        let mut full = vec!["lattice", "serve", "--model", "/tmp/model"];
        full.extend_from_slice(args);
        match Cli::try_parse_from(full)?.command {
            Command::Serve { max_pending, .. } => Ok(max_pending),
            _ => panic!("expected Command::Serve, got a different Command variant"),
        }
    }

    #[test]
    fn max_pending_omitted_defaults_to_32() {
        assert_eq!(parse_max_pending(&[]).expect("no --max-pending"), 32);
    }

    #[test]
    fn max_pending_zero_is_rejected() {
        parse_max_pending(&["--max-pending", "0"])
            .expect_err("0 admits nothing and must be rejected, not silently accepted");
    }

    #[test]
    fn max_pending_one_above_max_permits_is_rejected() {
        let too_big = (tokio::sync::Semaphore::MAX_PERMITS as u128 + 1).to_string();
        parse_max_pending(&["--max-pending", &too_big]).expect_err(
            "Semaphore::MAX_PERMITS + 1 must be rejected before it can panic Semaphore::new",
        );
    }

    #[test]
    fn max_pending_at_max_permits_is_accepted() {
        let at_max = tokio::sync::Semaphore::MAX_PERMITS.to_string();
        assert_eq!(
            parse_max_pending(&["--max-pending", &at_max])
                .expect("Semaphore::MAX_PERMITS itself is the inclusive upper bound"),
            tokio::sync::Semaphore::MAX_PERMITS
        );
    }

    #[test]
    fn max_pending_negative_is_rejected() {
        parse_max_pending(&["--max-pending", "-1"])
            .expect_err("a negative value must be rejected, not silently defaulted");
    }

    #[test]
    fn max_pending_malformed_is_rejected() {
        parse_max_pending(&["--max-pending", "not-a-number"])
            .expect_err("a non-numeric value must be rejected, not silently defaulted");
    }

    #[test]
    fn max_pending_valid_override_is_accepted() {
        assert_eq!(
            parse_max_pending(&["--max-pending", "8"]).expect("8 is a valid cap"),
            8
        );
    }
}

// ─── #1336 CLI boundary tests: `--preload-vision` defaults to lazy ────────
#[cfg(test)]
mod preload_vision_cli_tests {
    use super::*;

    fn parse_preload_vision(args: &[&str]) -> bool {
        let mut full = vec!["lattice", "serve", "--model", "/tmp/model"];
        full.extend_from_slice(args);
        match Cli::try_parse_from(full)
            .expect("fixed --model arg always parses")
            .command
        {
            Command::Serve { preload_vision, .. } => preload_vision,
            _ => panic!("expected Command::Serve, got a different Command variant"),
        }
    }

    /// Lazy loading is the default: omitting `--preload-vision` must not
    /// flip it on. This is the CLI-level half of the "lazy stays the
    /// default" contract the issue asks for -- the startup-behavior half is
    /// structural (`ModelBackend::spawn_metal` only calls
    /// `VisionRuntime::preload()` inside `if preload_vision { .. }`).
    #[test]
    fn preload_vision_omitted_defaults_to_false() {
        assert!(!parse_preload_vision(&[]));
    }

    #[test]
    fn preload_vision_flag_present_is_true() {
        assert!(parse_preload_vision(&["--preload-vision"]));
    }
}
