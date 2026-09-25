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
        /// directory produced by `quantize_q4`). If this directory is a
        /// vision-language checkpoint, /v1/embeddings additionally loads its
        /// own independent f16-packed copy of the decoder alongside the chat
        /// backend, costing roughly 2 extra resident bytes per checkpoint
        /// parameter (f16 storage); a checkpoint without vision support
        /// skips that extra load entirely, so only the chat backend stays
        /// resident.
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
        /// Maximum resident adapter identities; reaching the cap rejects new loads.
        #[arg(long, default_value_t = lattice_inference::serve::lora::DEFAULT_MAX_RESIDENT_ADAPTERS,
            value_parser = lattice_inference::serve::lora::parse_resident_limit)]
        max_resident_adapters: usize,
        /// Resident A/B tensor payload budget in bytes, not a process-memory budget.
        #[arg(long, default_value_t = lattice_inference::serve::lora::DEFAULT_MAX_RESIDENT_ADAPTER_BYTES,
            value_parser = lattice_inference::serve::lora::parse_resident_limit)]
        max_resident_adapter_bytes: usize,
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
        /// Directory holding versioned router gate artifacts (ADR-095
        /// decision 3). Omitted, the server has no router: a request that
        /// omits `lora` selects the base model, exactly as it does today.
        /// Given, the highest version present is loaded at startup and a
        /// directory that holds an artifact which will not load FAILS the
        /// startup rather than serving with routing silently off -- those two
        /// states answer requests differently and only one of them was asked
        /// for.
        #[arg(long)]
        router_state: Option<String>,
        /// Serve a specific router gate version instead of the highest one
        /// present (ADR-095 decision 3). This is the rollback arm: when a
        /// refit degrades output, pinning is what an operator reaches for,
        /// and it works when the process itself is the thing misbehaving.
        ///
        /// A pinned version that is absent FAILS the startup. Falling back to
        /// the latest would serve exactly the artifact the operator was
        /// trying to get away from, under a flag that says otherwise. A pin
        /// without --router-state fails too, naming the missing flag.
        #[arg(long)]
        router_pin: Option<u64>,
        /// Directory to load the embedding model from, instead of the served
        /// model's own directory (ADR-094 decision 1, amendment 2).
        ///
        /// Required by `--router-state`, and the reason is a contradiction in
        /// the unamended design: routing applies adapters, adapters need the
        /// Metal backend, only a Q4 directory reaches it -- and a Q4 directory
        /// is exactly what the embeddings loader refuses, because it reads an
        /// f16 decoder. Served model and embedder therefore cannot be the same
        /// directory on any server that can route, and before this flag there
        /// was no second one to name.
        ///
        /// Given, the load is FAIL-CLOSED: an operator who names a directory
        /// gets an error if it cannot serve as an embedder, rather than a
        /// server that starts with embeddings quietly off. Omitted, the
        /// embedder is still loaded best-effort from the served model's
        /// directory, which is the existing behaviour for `/v1/embeddings` and
        /// is unchanged.
        #[arg(long)]
        embedding_model: Option<String>,
        /// Identity recorded for the embedding model, overriding the
        /// directory basename (ADR-094 decision 1, amendment 2).
        ///
        /// Mirrors `--model-id` exactly, including its weakness: this is a
        /// NAME, so it detects a misconfigured server and not a substituted
        /// checkpoint. A different checkpoint of the same family and the same
        /// width, under a directory of the same name, is not detected -- the
        /// config carries only `model_type`, which reads identically for every
        /// member of the family.
        #[arg(long)]
        embedding_model_id: Option<String>,
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
    ///
    /// The score is a last-token variant: one hidden-state cosine per calibration
    /// prompt, not the per-token-averaged metric that
    /// `lattice_inference::pruning::BlockInfluenceAccumulator` implements. The
    /// output artifact's `method` field names the estimator actually used.
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
            max_resident_adapters,
            max_resident_adapter_bytes,
            preload_vision,
            router_state,
            router_pin,
            embedding_model: embedding_model_dir,
            embedding_model_id,
        } => {
            use std::path::Path;
            use std::sync::Arc;
            use std::sync::atomic::AtomicU64;
            use tokio::sync::Semaphore;

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
                        Ok(m) => cpu_serving_backend(m),
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
                            lattice_inference::serve::lora::ResidencyLimits {
                                max_adapters: max_resident_adapters,
                                max_bytes: max_resident_adapter_bytes,
                            },
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
                        let _ = (
                            max_pending,
                            max_resident_adapters,
                            max_resident_adapter_bytes,
                        );
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
            // for why the two loaders can't share weights).
            //
            // Two policies, because the two cases are different asks. An
            // explicitly named directory is an instruction and fails closed:
            // the operator said where the embedder is, so "it did not load"
            // is an error, not a downgrade. The implicit case keeps the
            // best-effort policy described above, since a chat-only
            // checkpoint having no embedder is ordinary and expected.
            let embedding_source = embedding_model_dir.as_deref().unwrap_or(&model);
            let embedding_model =
                match lattice_inference::serve::embeddings::EmbeddingModel::from_directory(
                    Path::new(embedding_source),
                ) {
                    Ok(embedding_model) => {
                        eprintln!(
                            "Embeddings enabled: pooled {}-dim vectors from {embedding_source}.",
                            embedding_model.dimensions()
                        );
                        Some(Arc::new(embedding_model))
                    }
                    Err(err) => {
                        if embedding_model_dir.is_some() {
                            eprintln!(
                                "Error: --embedding-model {embedding_source} cannot serve as an embedding model: {err}"
                            );
                            std::process::exit(1);
                        }
                        // Implicit case: warn and continue, the same policy as
                        // `--preload-vision` failing. A checkpoint that is not
                        // vision-language-shaped is expected and common, since
                        // most `lattice serve` deployments serve chat only.
                        eprintln!("Embeddings disabled ({model}): {err}");
                        None
                    }
                };

            // Fail closed: a configured router that will not load stops the
            // startup. Degrading to no-router would serve base-model output
            // under a configuration that asked for routing, and the operator
            // would learn about it from the responses rather than from here.
            // The decision itself lives in `router_state::resolve_startup` so
            // it is a value a test can produce without launching a server.
            // A build without Metal cannot make an adapter resident, so it
            // cannot apply one either; accepting a router there would load a
            // gate nothing on this build could ever use. Refused for the same
            // reason `adapter_unsupported_build` refuses the adapter routes.
            // Both flags, not just --router-state. A pin is meaningless on a
            // build that cannot route at all, and accepting it silently is the
            // worse half of the pair: the operator pinning a version during an
            // incident would get a server that reports success and routes
            // nothing.
            #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
            if router_state.is_some() || router_pin.is_some() {
                eprintln!(
                    "Error: --router-state and --router-pin require a macOS Metal build; adapter \
                     routing selects resident adapters, which this build cannot load."
                );
                std::process::exit(1);
            }

            // --embedding-model-id names the embedder identity the gate is
            // checked against, and nothing else reads it. On a build that
            // refuses routing outright it would be accepted and ignored, which
            // is the same silent success the refusal above exists to prevent
            // -- with the added cost that an operator who mistypes it here
            // learns nothing, and carries the typo to the build where it
            // decides whether the server starts.
            #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
            if embedding_model_id.is_some() {
                eprintln!(
                    "Error: --embedding-model-id requires a macOS Metal build; it names the \
                     embedder the routing gate is checked against, and this build cannot route."
                );
                std::process::exit(1);
            }

            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            let router_artifact = {
                use lattice_inference::router_state::{StartupDisposition, resolve_startup};
                match resolve_startup(router_state.as_deref().map(Path::new), router_pin) {
                    Ok(StartupDisposition::NoRouter) => None,
                    Ok(StartupDisposition::Loaded(resolved)) => {
                        // Naming the embedder is REQUIRED for routing, rather
                        // than inheriting whatever the served checkpoint
                        // happened to yield. Two reasons, and the first alone
                        // decides it: a server that can route serves Q4, the
                        // embeddings loader refuses Q4, so on a routing server
                        // the inherited embedder is always absent. The second
                        // is that the gate's representation is recorded
                        // against a named embedder, and inheriting the name
                        // from the chat model is what made the old check pass
                        // by co-location rather than by agreement.
                        let Some(embedder_dir) = embedding_model_dir.as_deref() else {
                            eprintln!(
                                "Error: --router-state requires --embedding-model <dir>. Routing needs a context vector, and the served checkpoint cannot supply one: applying adapters needs the Metal backend, which needs a Q4 directory, and the embeddings loader reads an f16 decoder and refuses a Q4 directory. Point --embedding-model at an f16 Qwen3.5 checkpoint."
                            );
                            std::process::exit(1);
                        };
                        // Unreachable rather than merely unlikely: an
                        // explicitly named --embedding-model that fails to
                        // load exits above. It says so instead of unwrapping.
                        let Some(embedder) = embedding_model.as_ref() else {
                            eprintln!(
                                "Error: --embedding-model was given but no embedding model is loaded; this is a bug in the startup ordering."
                            );
                            std::process::exit(1);
                        };
                        // The identity RECORDED for the embedder, and it is
                        // the embedder's, never the chat model's. A NAME, so
                        // it catches a misconfigured server and not a
                        // substituted checkpoint: the config carries only
                        // `model_type`, which reads identically across a whole
                        // family, so a different checkpoint of the same family
                        // and width under a directory of the same name is not
                        // detected. Mirrors `--model-id`, weakness included.
                        let embedder_identity =
                            lattice_inference::serve::routing::EmbedderIdentity::resolve(
                                Path::new(embedder_dir),
                                embedding_model_id.as_deref(),
                            );
                        let version = resolved.artifact.version_label();
                        let names = resolved.artifact.adapter_names.len();
                        let pinned = resolved.pinned;
                        // Checked against THIS server's embedding model, not
                        // only against the gate the artifact ships with: the
                        // two are different pairings and the constructor can
                        // only see one of them.
                        match lattice_inference::serve::routing::ServedRouter::new(
                            *resolved,
                            &embedder_identity,
                            embedder.dimensions(),
                        ) {
                            Ok(served) => {
                                eprintln!(
                                    "Router gate loaded: version {version} over {names} adapter(s){}",
                                    if pinned { " (pinned)" } else { "" }
                                );
                                Some(Arc::new(served))
                            }
                            Err(err) => {
                                eprintln!("Error: {}", err.message());
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {e}");
                        std::process::exit(1);
                    }
                }
            };

            let state = serve::AppState {
                model: model_backend,
                default_max_tokens: max_tokens,
                max_tokens_cap: 4096,
                model_id: served_model_id.clone(),
                request_counter: Arc::new(AtomicU64::new(0)),
                embedding_model,
                embedding_admission: Arc::new(Semaphore::new(serve::EMBEDDING_MAX_CONCURRENT_JOBS)),
                #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
                router_state: router_artifact,
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

fn cpu_serving_backend(
    mut model: lattice_inference::model::qwen35::Qwen35Model,
) -> serve::ModelBackend {
    model.ensure_tokenizer_max_seq_len(model.max_context());
    serve::ModelBackend::Cpu(std::sync::Arc::new(model))
}

#[cfg(test)]
mod cpu_serve_prompt_tests {
    #[cfg(feature = "test-utils")]
    use super::*;
    #[cfg(feature = "test-utils")]
    use lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context;
    #[cfg(feature = "test-utils")]
    use lattice_inference::{GenerateConfig, Tokenizer};

    #[test]
    fn cpu_serve_load_keeps_tokenizer_initialization() {
        // The server's non-returning main cannot be called by a unit test.
        // Keep its load wired to the initialization exercised below.
        let startup = include_str!("main.rs")
            .split("fn cpu_serving_backend(")
            .next()
            .unwrap();
        assert!(startup.contains("Ok(m) => cpu_serving_backend(m),"));
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn serving_model_keeps_long_prompts_after_generation_tokenization() {
        let model = tiny_zero_model_with_context(8192);
        assert_eq!(model.tokenizer().max_seq_len(), 4096);
        let backend = cpu_serving_backend(model);
        let serve::ModelBackend::Cpu(model) = backend else {
            panic!("expected CPU serving backend");
        };
        let mut cfg = GenerateConfig::default();
        cfg.max_new_tokens = 0;
        for n in [4097, model.max_context()] {
            let prompt = "a".repeat(n);
            assert_eq!(model.tokenizer().tokenize(&prompt).real_length, n);
            let output = model.generate(&prompt, &cfg).unwrap();
            assert_eq!(output.prompt_tokens, n);
        }
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

    fn parse_resident_limits(args: &[&str]) -> Result<(usize, usize), clap::Error> {
        let mut full = vec!["lattice", "serve", "--model", "fixture"];
        full.extend_from_slice(args);
        match Cli::try_parse_from(full)?.command {
            Command::Serve {
                max_resident_adapters,
                max_resident_adapter_bytes,
                ..
            } => Ok((max_resident_adapters, max_resident_adapter_bytes)),
            _ => panic!("expected serve"),
        }
    }

    #[test]
    fn resident_limits_defaults_and_overrides() {
        assert_eq!(parse_resident_limits(&[]).unwrap(), (32, 536_870_912));
        assert_eq!(
            parse_resident_limits(&[
                "--max-resident-adapters",
                "2",
                "--max-resident-adapter-bytes",
                "128"
            ])
            .unwrap(),
            (2, 128)
        );
        assert_eq!(
            parse_resident_limits(&[
                "--max-resident-adapters",
                "1",
                "--max-resident-adapter-bytes",
                "1"
            ])
            .unwrap(),
            (1, 1)
        );
    }

    fn assert_resident_flag_rejects_invalid(flag: &str) {
        parse_resident_limits(&[flag, "8"]).unwrap();
        for value in ["not-a-number", "0", "-1", "18446744073709551616"] {
            assert!(
                parse_resident_limits(&[flag, value]).is_err(),
                "{flag} accepted {value}"
            );
        }
        assert!(parse_resident_limits(&[flag]).is_err());
    }

    #[test]
    fn resident_count_malformed_is_rejected() {
        assert_resident_flag_rejects_invalid("--max-resident-adapters");
    }

    #[test]
    fn resident_bytes_malformed_is_rejected() {
        assert_resident_flag_rejects_invalid("--max-resident-adapter-bytes");
    }

    #[test]
    fn resident_limits_valid_flags_are_accepted() {
        parse_max_pending(&[
            "--max-resident-adapters",
            "2",
            "--max-resident-adapter-bytes",
            "128",
        ])
        .unwrap();
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

// ─── ADR-094 D1 amendment 2: the embedder is nameable separately ──────────
#[cfg(test)]
mod embedding_model_cli_tests {
    use super::*;

    fn parse_embedder(args: &[&str]) -> (Option<String>, Option<String>) {
        let mut full = vec!["lattice", "serve", "--model", "/tmp/model"];
        full.extend_from_slice(args);
        match Cli::try_parse_from(full)
            .expect("fixed --model arg always parses")
            .command
        {
            Command::Serve {
                embedding_model,
                embedding_model_id,
                ..
            } => (embedding_model, embedding_model_id),
            _ => panic!("expected Command::Serve, got a different Command variant"),
        }
    }

    /// The CLI half only. The startup behaviour these flags gate -- that
    /// `--router-state` without `--embedding-model` refuses naming the flag,
    /// and that a Q4 directory named as the embedder refuses naming the
    /// format -- exits the process, which a unit test cannot cross. Those two
    /// arms are verified by starting a server.
    #[test]
    fn omitting_both_embedder_flags_leaves_the_existing_behaviour() {
        assert_eq!(parse_embedder(&[]), (None, None));
    }

    /// The flag SPELLINGS are asserted, not just their presence: the
    /// amendment text and the operator runbook both name these strings, and a
    /// rename would leave those documents describing a server that refuses
    /// the command they print.
    #[test]
    fn the_embedder_directory_and_its_identity_are_separate_flags() {
        assert_eq!(
            parse_embedder(&["--embedding-model", "/models/qwen3.5-0.8b"]),
            (Some("/models/qwen3.5-0.8b".to_string()), None)
        );
        assert_eq!(
            parse_embedder(&[
                "--embedding-model",
                "/models/qwen3.5-0.8b",
                "--embedding-model-id",
                "gme-qwen35",
            ]),
            (
                Some("/models/qwen3.5-0.8b".to_string()),
                Some("gme-qwen35".to_string())
            )
        );
    }
}
