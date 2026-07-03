//! `lattice` CLI — interactive chat and HTTP serve subcommands.
//!
//! # Usage
//!
//! ```text
//! lattice chat --model /path/to/model [--max-tokens 256] [--temperature 0.7]
//! lattice serve --model /path/to/model [--host 127.0.0.1] [--port 8080] [--max-tokens 256]
//! ```

use clap::{Parser, Subcommand};

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
    },
}

// ---------------------------------------------------------------------------
// backend: model-directory format detection + Q4/Metal loading
//
// `lattice chat`/`lattice serve` originally only understood a safetensors
// directory (`model.safetensors` or a sharded index). This module adds
// support for native Q4 quantized directories (per-tensor `.q4` files, the
// output of `quantize_q4`) by detecting the format up front and routing to
// the Metal GPU forward pass. Safetensors directories are completely
// unaffected: `detect_format` returns `Safetensors` for them exactly as
// before, and the safetensors load path is untouched.
// ---------------------------------------------------------------------------

mod backend {
    use std::path::Path;

    /// The on-disk format of a model directory, decided before any tensor I/O.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ModelFormat {
        /// `model.safetensors` or `model.safetensors.index.json` present.
        Safetensors,
        /// No safetensors file, but at least one `*.q4` tensor file present
        /// (the output of `quantize_q4`).
        Q4,
        /// Neither a safetensors file nor a `.q4` file was found.
        Unknown,
    }

    /// Detect whether `dir` is a safetensors model directory, a native Q4
    /// quantized directory, or neither.
    ///
    /// Mirrors the detection heuristic already shipped in `chat_metal.rs` and
    /// `lattice_serve.rs`: a directory is Q4 when it has no safetensors file
    /// and contains at least one file whose name ends in `.q4`.
    pub fn detect_format(dir: &Path) -> ModelFormat {
        if dir.join("model.safetensors").exists()
            || dir.join("model.safetensors.index.json").exists()
        {
            return ModelFormat::Safetensors;
        }
        let has_q4_file = std::fs::read_dir(dir)
            .ok()
            .and_then(|mut entries| {
                entries.find(|e| {
                    e.as_ref()
                        .ok()
                        .and_then(|e| e.file_name().to_str().map(|n| n.ends_with(".q4")))
                        .unwrap_or(false)
                })
            })
            .is_some();
        if has_q4_file {
            ModelFormat::Q4
        } else {
            ModelFormat::Unknown
        }
    }

    /// Error message shown when a Q4 directory is passed to a binary that was
    /// built without the `metal-gpu` feature. Q4 inference only runs on the
    /// Metal GPU forward pass; there is no CPU fallback for `.q4` tensors, so
    /// this is a hard, fail-closed error rather than a silent degrade.
    ///
    /// Only reachable from the `#[cfg(not(feature = "metal-gpu"))]` call sites
    /// in `run_chat` / `main`; a `metal-gpu` build never calls this (it loads
    /// Q4 directories instead), so it is legitimately unused in that
    /// configuration rather than by mistake.
    #[cfg_attr(feature = "metal-gpu", allow(dead_code))]
    pub fn metal_gpu_required_message(dir: &Path) -> String {
        format!(
            "model directory '{}' is a native Q4 quantized checkpoint, which requires \
             the Metal GPU forward pass. This binary was built without the `metal-gpu` \
             feature. Rebuild with `--features \"f16 metal-gpu\"` (macOS only), or point \
             --model at a safetensors directory instead.",
            dir.display()
        )
    }

    /// Error message for a directory that is neither safetensors nor Q4.
    pub fn unrecognized_format_message(dir: &Path) -> String {
        format!(
            "'{}' is not a recognized model directory: no model.safetensors, \
             model.safetensors.index.json, or *.q4 tensor files were found",
            dir.display()
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;

        fn tempdir(name: &str) -> std::path::PathBuf {
            let mut dir = std::env::temp_dir();
            dir.push(format!(
                "lattice-backend-test-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            fs::create_dir_all(&dir).expect("create tempdir");
            dir
        }

        #[test]
        fn detect_format_safetensors_file() {
            let dir = tempdir("safetensors-file");
            fs::write(dir.join("model.safetensors"), b"stub").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_safetensors_index_only() {
            let dir = tempdir("safetensors-index");
            fs::write(dir.join("model.safetensors.index.json"), b"{}").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_q4_dir() {
            let dir = tempdir("q4");
            fs::write(dir.join("model_layers_0_weight.q4"), b"stub").unwrap();
            fs::write(dir.join("config.json"), b"{}").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Q4);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_prefers_safetensors_over_q4_files() {
            // A directory that (unusually) has both a safetensors file and a
            // stray .q4 file must resolve as Safetensors — the safetensors
            // loader path is untouched and takes priority.
            let dir = tempdir("mixed");
            fs::write(dir.join("model.safetensors"), b"stub").unwrap();
            fs::write(dir.join("leftover.q4"), b"stub").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_empty_dir_is_unknown() {
            let dir = tempdir("empty");
            assert_eq!(detect_format(&dir), ModelFormat::Unknown);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn detect_format_unrelated_files_is_unknown() {
            let dir = tempdir("unrelated");
            fs::write(dir.join("readme.txt"), b"hello").unwrap();
            fs::write(dir.join("config.json"), b"{}").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Unknown);
            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn metal_gpu_required_message_mentions_rebuild_flags() {
            let msg = metal_gpu_required_message(Path::new("/tmp/some-q4-dir"));
            assert!(msg.contains("metal-gpu"));
            assert!(msg.contains("--features"));
        }

        #[test]
        fn unrecognized_format_message_mentions_expected_files() {
            let msg = unrecognized_format_message(Path::new("/tmp/bogus"));
            assert!(msg.contains("model.safetensors"));
            assert!(msg.contains(".q4"));
        }

        // Fail-closed without metal-gpu: a Q4 directory must never silently
        // fall back to the CPU safetensors loader. This test only compiles
        // (and only means anything) when the binary is built WITHOUT the
        // metal-gpu feature — it asserts that the code path this binary
        // would take for a Q4 directory is the explicit error message above,
        // never `Qwen35Model::from_safetensors`.
        #[cfg(not(feature = "metal-gpu"))]
        #[test]
        fn q4_dir_without_metal_gpu_feature_fails_closed() {
            let dir = tempdir("q4-no-metal");
            fs::write(dir.join("model_layers_0_weight.q4"), b"stub").unwrap();
            assert_eq!(detect_format(&dir), ModelFormat::Q4);
            // Scope: detection + message content only. This proves a Q4 dir
            // classifies as `ModelFormat::Q4` (so the `run_chat`/`main` match
            // arms take the fail-closed branch, never
            // `Qwen35Model::from_safetensors`) and that the error names the
            // rebuild flags. It does not drive `run_chat`/`main` themselves —
            // those exit the process, which a unit test cannot cross.
            let msg = metal_gpu_required_message(&dir);
            assert!(msg.contains("metal-gpu"));
            fs::remove_dir_all(&dir).ok();
        }
    }
}

// ---------------------------------------------------------------------------
// chat subcommand
// ---------------------------------------------------------------------------

/// Load `config.json` for a Q4 directory, falling back to the Qwen3.6-27B
/// default config (matching `chat_metal.rs` / `lattice_serve.rs`) when the
/// directory has none, with a visible warning so a missing config.json is
/// never silently misinterpreted as intentional.
#[cfg(feature = "metal-gpu")]
fn load_q4_config(
    dir: &std::path::Path,
) -> Result<lattice_inference::model::qwen35_config::Qwen35Config, String> {
    let config_path = dir.join("config.json");
    if config_path.exists() {
        lattice_inference::model::qwen35_config::Qwen35Config::from_config_json(&config_path)
            .map_err(|e| format!("config.json parse failed: {e}"))
    } else {
        eprintln!(
            "Warning: {} has no config.json; falling back to the Qwen3.6-27B default config.",
            dir.display()
        );
        Ok(lattice_inference::model::qwen35_config::Qwen35Config::qwen36_27b())
    }
}

/// Metal-GPU chat backend: owns a `MetalQwen35State` plus the tokenizer and
/// context-window cap needed to serve `generate`/`generate_streaming` calls
/// the same way the CPU (`Qwen35Model`) backend does.
///
/// `MetalQwen35State` is `!Send` (it owns raw `metal::*` FFI objects), so
/// this type must never be shared across threads. `run_chat`'s REPL uses it
/// directly on the calling thread; the `serve` module never constructs one
/// on an async task — it lives on a dedicated worker thread instead (see
/// `serve::spawn_metal_worker`).
#[cfg(feature = "metal-gpu")]
struct MetalChatBackend {
    state: lattice_inference::forward::metal_qwen35::MetalQwen35State,
    tokenizer: lattice_inference::tokenizer::bpe::BpeTokenizer,
}

#[cfg(feature = "metal-gpu")]
impl MetalChatBackend {
    /// `max_cache_len` bounds the KV cache (and therefore the usable context
    /// window). 4096 matches the cap used by `chat_metal.rs`.
    const MAX_CACHE_LEN: usize = 4096;

    /// `tokenizer_dir` overrides where `tokenizer.json` is read from, for Q4
    /// directories that were produced without a co-located tokenizer. `None`
    /// resolves it from `dir` itself (the common case: Q4 dirs ship it).
    fn load(
        dir: &std::path::Path,
        tokenizer_dir: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        let tokenizer_path = tokenizer_dir.unwrap_or(dir).join("tokenizer.json");
        let tokenizer =
            lattice_inference::tokenizer::bpe::BpeTokenizer::from_tokenizer_json(&tokenizer_path)
                .map_err(|e| format!("tokenizer load failed ({}): {e}", tokenizer_path.display()))?;
        let cfg = load_q4_config(dir)?;
        let state = lattice_inference::forward::metal_qwen35::MetalQwen35State::from_q4_dir(
            dir,
            &tokenizer_path,
            &cfg,
            Self::MAX_CACHE_LEN,
        )
        .map_err(|e| format!("Q4 model load failed: {e}"))?;
        Ok(Self { state, tokenizer })
    }

    fn generate(
        &mut self,
        prompt: &str,
        gen_cfg: &lattice_inference::model::qwen35_config::GenerateConfig,
    ) -> lattice_inference::model::qwen35_config::GenerateOutput {
        self.state.generate(prompt, &self.tokenizer, gen_cfg)
    }
}

fn run_chat(model_path: &str, max_tokens: usize, temperature: f32, tokenizer_dir: Option<&str>) {
    use std::io::{BufRead, Write};
    use std::path::Path;

    let path = Path::new(model_path);
    let format = backend::detect_format(path);
    #[cfg(feature = "metal-gpu")]
    let tokenizer_dir_path = tokenizer_dir.map(Path::new);
    #[cfg(not(feature = "metal-gpu"))]
    let _ = tokenizer_dir;

    eprintln!("Loading model from {model_path}...");

    enum Backend {
        Cpu(Box<lattice_inference::model::qwen35::Qwen35Model>),
        #[cfg(feature = "metal-gpu")]
        Metal(Box<MetalChatBackend>),
    }

    let mut model = match format {
        backend::ModelFormat::Safetensors => {
            match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(path) {
                Ok(m) => Backend::Cpu(Box::new(m)),
                Err(e) => {
                    eprintln!("Error: failed to load model: {e}");
                    std::process::exit(1);
                }
            }
        }
        backend::ModelFormat::Q4 => {
            #[cfg(feature = "metal-gpu")]
            {
                match MetalChatBackend::load(path, tokenizer_dir_path) {
                    Ok(m) => Backend::Metal(Box::new(m)),
                    Err(e) => {
                        eprintln!("Error: failed to load Q4 model: {e}");
                        std::process::exit(1);
                    }
                }
            }
            #[cfg(not(feature = "metal-gpu"))]
            {
                eprintln!("Error: {}", backend::metal_gpu_required_message(path));
                std::process::exit(1);
            }
        }
        backend::ModelFormat::Unknown => {
            eprintln!("Error: {}", backend::unrecognized_format_message(path));
            std::process::exit(1);
        }
    };
    eprintln!("Model loaded. Type 'exit' or 'quit' to stop.\n");

    let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: max_tokens,
        temperature,
        ..Default::default()
    };

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let prompt = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        };
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            break;
        }

        match &mut model {
            Backend::Cpu(m) => match m.generate(trimmed, &gen_cfg) {
                Ok(output) => {
                    let _ = writeln!(stdout, "{}", output.text);
                    let _ = writeln!(
                        stdout,
                        "[{} prompt tokens, {} generated]",
                        output.prompt_tokens, output.generated_tokens
                    );
                }
                Err(e) => {
                    eprintln!("Generation error: {e}");
                }
            },
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(m) => {
                let output = m.generate(trimmed, &gen_cfg);
                let _ = writeln!(stdout, "{}", output.text);
                let _ = writeln!(
                    stdout,
                    "[{} prompt tokens, {} generated]",
                    output.prompt_tokens, output.generated_tokens
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// serve subcommand: OpenAI-compatible HTTP API
// ---------------------------------------------------------------------------

mod serve {
    use axum::{
        Json, Router,
        extract::{DefaultBodyLimit, State},
        http::StatusCode,
        response::{
            IntoResponse, Response,
            sse::{Event, KeepAlive, Sse},
        },
        routing::{get, post},
    };
    use futures::StreamExt as _;
    use lattice_inference::Tokenizer;
    #[cfg(feature = "metal-gpu")]
    use lattice_inference::model::qwen35_config::GenerateConfig;
    use lattice_inference::model::qwen35_config::GenerateOutput;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Request body cap: 1 MiB.  Requests above this return HTTP 413.
    const REQUEST_BODY_LIMIT_BYTES: usize = 1_048_576;

    // -----------------------------------------------------------------------
    // Model backend: CPU (safetensors) or Metal GPU (native Q4)
    // -----------------------------------------------------------------------

    /// One generation request handed to the Metal GPU worker thread.
    ///
    /// `MetalQwen35State` owns raw `metal::*` FFI objects and is `!Send`, so it
    /// cannot be moved into a `tokio::task::spawn_blocking` closure the way the
    /// CPU model is (`Arc<Qwen35Model>` is `Send + Sync`; `MetalQwen35State` is
    /// neither). Instead the Metal state lives on ONE dedicated OS thread for
    /// the whole process lifetime, and async handlers ship it a `MetalJob` over
    /// an unbounded `tokio::sync::mpsc` channel — the same design already
    /// shipped in `lattice_serve.rs`. `on_token` is called synchronously from
    /// the worker thread for each streamed delta; returning `false` stops
    /// generation early (client disconnected).
    ///
    /// This serializes ALL Metal generation onto one thread: two concurrent
    /// requests to a Q4-backed `lattice serve` run back-to-back, not in
    /// parallel. That is correct for a single-GPU local engine (the same
    /// default ollama uses) and is documented here rather than hidden behind
    /// an innocuous-looking channel send.
    #[cfg(feature = "metal-gpu")]
    struct MetalJob {
        prompt: String,
        gen_cfg: GenerateConfig,
        on_token: Box<dyn FnMut(&str) -> bool + Send>,
        reply: tokio::sync::oneshot::Sender<GenerateOutput>,
    }

    /// Handle to the Metal GPU worker thread. Cheaply `Clone` (an `mpsc`
    /// sender), `Send + Sync`, so it can live in `AppState` like the CPU
    /// `Arc<Qwen35Model>` does — only the underlying `MetalQwen35State` is
    /// confined to the worker thread.
    #[cfg(feature = "metal-gpu")]
    #[derive(Clone)]
    pub struct MetalHandle {
        jobs: tokio::sync::mpsc::UnboundedSender<MetalJob>,
    }

    #[cfg(feature = "metal-gpu")]
    impl MetalHandle {
        /// Load the Q4 model on a new dedicated worker thread and return a
        /// handle once loading succeeds. Loading happens synchronously (the
        /// caller blocks until the model is ready or loading fails) so that
        /// `lattice serve`'s startup sequence keeps its existing "load, then
        /// bind, then listen" ordering and fails closed before ever binding
        /// the socket.
        fn spawn(
            model_dir: std::path::PathBuf,
            tokenizer_path: std::path::PathBuf,
            tokenizer: Arc<lattice_inference::tokenizer::bpe::BpeTokenizer>,
        ) -> Result<Self, String> {
            let (job_tx, mut job_rx) = tokio::sync::mpsc::unbounded_channel::<MetalJob>();
            let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

            std::thread::spawn(move || {
                let cfg = match super::load_q4_config(&model_dir) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let mut state =
                    match lattice_inference::forward::metal_qwen35::MetalQwen35State::from_q4_dir(
                        &model_dir,
                        &tokenizer_path,
                        &cfg,
                        super::MetalChatBackend::MAX_CACHE_LEN,
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = ready_tx.send(Err(format!("Q4 model load failed: {e}")));
                            return;
                        }
                    };
                let _ = ready_tx.send(Ok(()));

                while let Some(job) = job_rx.blocking_recv() {
                    let mut on_token = job.on_token;
                    let output = state.generate_streaming(
                        &job.prompt,
                        &tokenizer,
                        &job.gen_cfg,
                        |delta, _token_id| on_token(delta),
                    );
                    let _ = job.reply.send(output);
                }
            });

            match ready_rx.recv() {
                Ok(Ok(())) => Ok(Self { jobs: job_tx }),
                Ok(Err(e)) => Err(e),
                Err(_) => Err("Metal worker thread exited before loading finished".to_string()),
            }
        }

        /// Run one generation on the worker thread, forwarding each token
        /// delta to `on_token`. Returns the full `GenerateOutput` (including
        /// `stopped`/`stop_reason`) so callers can compute `finish_reason`
        /// with the exact same `finish_reason_for` helper the CPU path uses.
        async fn generate_streaming(
            &self,
            prompt: String,
            gen_cfg: GenerateConfig,
            on_token: impl FnMut(&str) -> bool + Send + 'static,
        ) -> Result<GenerateOutput, ApiError> {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            let job = MetalJob {
                prompt,
                gen_cfg,
                on_token: Box::new(on_token),
                reply: reply_tx,
            };
            self.jobs.send(job).map_err(|_| ApiError::Internal {
                message: "inference worker is not running".to_string(),
            })?;
            reply_rx.await.map_err(|_| ApiError::Internal {
                message: "inference worker dropped the request".to_string(),
            })
        }
    }

    /// The two ways `AppState` can run generation: the original CPU
    /// (safetensors) path via `Arc<Qwen35Model>`, or the Metal GPU (native
    /// Q4) path via a worker-thread handle. Both variants funnel into the
    /// same request handler code below — `chat_completions` branches on this
    /// enum in exactly two places (streaming and non-streaming) rather than
    /// duplicating the handler.
    #[derive(Clone)]
    pub enum ModelBackend {
        Cpu(Arc<lattice_inference::model::qwen35::Qwen35Model>),
        #[cfg(feature = "metal-gpu")]
        Metal {
            handle: MetalHandle,
            tokenizer: Arc<lattice_inference::tokenizer::bpe::BpeTokenizer>,
            max_context: usize,
        },
    }

    impl ModelBackend {
        pub fn tokenize_len(&self, text: &str) -> usize {
            match self {
                ModelBackend::Cpu(m) => m.tokenizer().tokenize(text).real_length,
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { tokenizer, .. } => tokenizer.tokenize(text).real_length,
            }
        }

        pub fn max_context(&self) -> usize {
            match self {
                ModelBackend::Cpu(m) => m.max_context(),
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { max_context, .. } => *max_context,
            }
        }

        /// Load a native Q4 checkpoint on a dedicated Metal worker thread and
        /// return the `ModelBackend::Metal` handle plus the resolved context
        /// window, for `main()`'s `Command::Serve` startup sequence.
        #[cfg(feature = "metal-gpu")]
        pub fn spawn_metal(
            model_dir: std::path::PathBuf,
            tokenizer_dir: Option<std::path::PathBuf>,
        ) -> Result<(Self, usize), String> {
            let tokenizer_path = tokenizer_dir
                .as_deref()
                .unwrap_or(&model_dir)
                .join("tokenizer.json");
            let tokenizer = Arc::new(
                lattice_inference::tokenizer::bpe::BpeTokenizer::from_tokenizer_json(
                    &tokenizer_path,
                )
                .map_err(|e| {
                    format!("tokenizer load failed ({}): {e}", tokenizer_path.display())
                })?,
            );
            let max_context = super::MetalChatBackend::MAX_CACHE_LEN;
            let handle = MetalHandle::spawn(model_dir, tokenizer_path, Arc::clone(&tokenizer))?;
            Ok((
                ModelBackend::Metal {
                    handle,
                    tokenizer,
                    max_context,
                },
                max_context,
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Shared application state
    // -----------------------------------------------------------------------

    /// State shared across all request handlers via axum's `State` extractor.
    #[derive(Clone)]
    pub struct AppState {
        /// The loaded model backend (CPU safetensors or Metal GPU Q4).
        pub model: ModelBackend,
        /// Default `max_tokens` value used when a request omits the field.
        /// Set from the `--max-tokens` CLI flag passed to `lattice serve`.
        pub default_max_tokens: usize,
        /// Hard upper bound on `max_tokens` accepted from any request.
        /// Prevents callers from requesting unbounded generation.
        pub max_tokens_cap: usize,
        /// Canonical model identifier echoed in every response.
        /// Derived from the `--model-id` flag or the model path basename.
        pub model_id: String,
        /// Monotonically increasing counter used to make response IDs unique
        /// across concurrent requests within the same second.
        pub request_counter: Arc<AtomicU64>,
    }

    // -----------------------------------------------------------------------
    // Error type
    // -----------------------------------------------------------------------

    /// Structured HTTP error that serialises to the OpenAI error envelope so
    /// that clients can parse failure responses uniformly.
    #[derive(Debug)]
    pub enum ApiError {
        /// Caller mistake — HTTP 400.
        BadRequest { message: String, code: &'static str },
        /// Request body exceeds size limit — HTTP 413.
        PayloadTooLarge { message: String },
        /// Server-side failure — HTTP 500.
        Internal { message: String },
    }

    #[derive(Serialize)]
    struct ErrorBody {
        error: ErrorDetail,
    }

    #[derive(Serialize)]
    struct ErrorDetail {
        message: String,
        r#type: &'static str,
        code: String,
        param: Option<String>,
    }

    impl IntoResponse for ApiError {
        fn into_response(self) -> Response {
            match self {
                ApiError::BadRequest { message, code } => {
                    let body = Json(ErrorBody {
                        error: ErrorDetail {
                            message,
                            r#type: "invalid_request_error",
                            code: code.to_string(),
                            param: None,
                        },
                    });
                    (StatusCode::BAD_REQUEST, body).into_response()
                }
                ApiError::PayloadTooLarge { message } => {
                    let body = Json(ErrorBody {
                        error: ErrorDetail {
                            message,
                            r#type: "invalid_request_error",
                            code: "request_body_too_large".to_string(),
                            param: None,
                        },
                    });
                    (StatusCode::PAYLOAD_TOO_LARGE, body).into_response()
                }
                ApiError::Internal { message } => {
                    let body = Json(ErrorBody {
                        error: ErrorDetail {
                            message,
                            r#type: "server_error",
                            code: "internal_error".to_string(),
                            param: None,
                        },
                    });
                    (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Request / response types
    // -----------------------------------------------------------------------

    /// OpenAI-compatible chat completions request.
    ///
    /// Known-but-unsupported fields (`stream=true`, `tools`, `tool_choice`,
    /// `logprobs=true`, `n > 1`, `response_format` other than `"text"`) are
    /// parsed and explicitly rejected with HTTP 400 rather than silently dropped.
    /// `stop` is accepted and parsed into string-level stop sequences.
    /// Unknown fields are ignored by default (serde default).
    #[derive(Deserialize)]
    pub struct ChatCompletionRequest {
        /// Required: must match the served model identifier.
        pub model: String,
        pub messages: Vec<Message>,
        /// Generation token budget.  Use at most one of `max_tokens` /
        /// `max_completion_tokens`; if both are present they must agree.
        pub max_tokens: Option<usize>,
        /// Alias for `max_tokens` (current OpenAI naming).
        pub max_completion_tokens: Option<usize>,
        pub temperature: Option<f32>,
        /// Nucleus sampling probability mass.  Mapped into `GenerateConfig`.
        pub top_p: Option<f32>,
        /// SSE streaming — not yet supported; rejected with 400.
        pub stream: Option<bool>,
        /// Stop sequences — a JSON string or array of strings (up to 4, non-empty).
        /// Parsed by `parse_stop_strings`; null/absent → empty vec (no stops).
        pub stop: Option<Value>,
        /// Deterministic sampling seed.  Mapped into `GenerateConfig`.
        pub seed: Option<u64>,
        /// Response format constraint — only `"text"` is accepted.
        pub response_format: Option<ResponseFormat>,
        /// Tool definitions — not supported; rejected with 400.
        pub tools: Option<Value>,
        /// Tool choice — not supported; rejected with 400.
        pub tool_choice: Option<Value>,
        /// Log-probabilities — not supported; rejected with 400.
        pub logprobs: Option<bool>,
        /// Number of completions — only `1` is accepted.
        pub n: Option<usize>,
    }

    #[derive(Deserialize)]
    pub struct ResponseFormat {
        pub r#type: String,
    }

    /// Message content: either a plain string or an array of content parts.
    /// Non-text parts (image, audio, file) are rejected with HTTP 400.
    #[derive(Deserialize)]
    #[serde(untagged)]
    pub enum MessageContent {
        Text(String),
        Parts(Vec<ContentPart>),
    }

    #[derive(Deserialize)]
    pub struct ContentPart {
        #[serde(rename = "type")]
        pub kind: String,
        pub text: Option<String>,
    }

    #[derive(Deserialize)]
    pub struct Message {
        pub role: String,
        pub content: MessageContent,
    }

    #[derive(Serialize)]
    pub struct ChatCompletionResponse {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<Choice>,
        pub usage: Usage,
    }

    #[derive(Serialize)]
    pub struct Choice {
        pub index: usize,
        pub message: ResponseMessage,
        pub finish_reason: String,
    }

    #[derive(Serialize)]
    pub struct ResponseMessage {
        pub role: String,
        pub content: String,
    }

    #[derive(Serialize)]
    pub struct Usage {
        pub prompt_tokens: usize,
        pub completion_tokens: usize,
        pub total_tokens: usize,
    }

    #[derive(Serialize)]
    pub struct HealthResponse {
        pub status: &'static str,
    }

    // -----------------------------------------------------------------------
    // SSE streaming types
    // -----------------------------------------------------------------------

    /// Internal channel message type for the streaming generation path.
    ///
    /// `spawn_blocking` runs the sync `generate_streaming` call on a blocking
    /// thread and sends incremental deltas through an unbounded channel.  The
    /// async SSE handler reads from the other end and maps these messages to
    /// OpenAI `chat.completion.chunk` events.
    pub enum StreamMsg {
        /// One incremental text delta from the model.
        Delta(String),
        /// Generation finished normally; carries the OpenAI finish reason.
        Done { finish_reason: &'static str },
        /// Generation failed (invariant violation or engine error).
        Failed,
    }

    /// Top-level chunk object serialised into each `data:` SSE event.
    #[derive(Serialize)]
    pub struct ChatCompletionChunk {
        pub id: String,
        pub object: &'static str,
        pub created: u64,
        pub model: String,
        pub choices: Vec<ChunkChoice>,
    }

    #[derive(Serialize)]
    pub struct ChunkChoice {
        pub index: usize,
        pub delta: ChunkDelta,
        /// Null while streaming, set on the final choice.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub finish_reason: Option<&'static str>,
    }

    /// The `delta` field of a streaming chunk.
    ///
    /// Exactly one of `role` / `content` is set per chunk:
    /// - First chunk: `role = "assistant"`, no content.
    /// - Subsequent content chunks: `content = <text>`, no role.
    /// - Final finish chunk: both absent (empty delta `{}`).
    #[derive(Serialize)]
    pub struct ChunkDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub role: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<String>,
    }

    // -----------------------------------------------------------------------
    // Validation helpers — pure functions, no model required, easily tested
    // -----------------------------------------------------------------------

    /// Resolve the effective `max_tokens`, rejecting zero, values above the
    /// server cap, and conflicting `max_tokens` / `max_completion_tokens`.
    fn validate_max_tokens(
        req_max: Option<usize>,
        req_max_completion: Option<usize>,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<usize, ApiError> {
        let effective = match (req_max, req_max_completion) {
            (None, None) => default_max_tokens,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (Some(a), Some(b)) if a == b => a,
            (Some(a), Some(b)) => {
                return Err(ApiError::BadRequest {
                    message: format!(
                        "max_tokens ({a}) and max_completion_tokens ({b}) differ; supply only one"
                    ),
                    code: "invalid_request",
                });
            }
        };
        if effective == 0 {
            return Err(ApiError::BadRequest {
                message: "max_tokens must be at least 1".to_string(),
                code: "invalid_max_tokens",
            });
        }
        if effective > max_tokens_cap {
            return Err(ApiError::BadRequest {
                message: format!("max_tokens {effective} exceeds server limit {max_tokens_cap}"),
                code: "max_tokens_exceeds_limit",
            });
        }
        Ok(effective)
    }

    /// Validate `temperature` is in `[0.0, 2.0]`.
    fn validate_temperature(value: Option<f32>) -> Result<f32, ApiError> {
        let temperature = value.unwrap_or(0.7);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(ApiError::BadRequest {
                message: "temperature must be between 0 and 2".to_string(),
                code: "invalid_temperature",
            });
        }
        Ok(temperature)
    }

    /// Validate `top_p` is in `(0.0, 1.0]`.
    fn validate_top_p(value: Option<f32>) -> Result<f32, ApiError> {
        let top_p = value.unwrap_or(0.9);
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(ApiError::BadRequest {
                message: "top_p must be greater than 0 and at most 1".to_string(),
                code: "invalid_top_p",
            });
        }
        Ok(top_p)
    }

    /// Parse the OpenAI `stop` field into a `Vec<String>`.
    ///
    /// Accepted forms:
    /// - `null` / absent → empty vec (no string-level stops)
    /// - a JSON string → `vec![s]`
    /// - a JSON array of 1–4 non-empty strings → that vec
    ///
    /// Returns `Err(BadRequest)` for:
    /// - an empty array
    /// - an array with more than 4 elements
    /// - any array element that is not a string
    /// - any stop string that is empty
    fn parse_stop_strings(stop: &Option<Value>) -> Result<Vec<String>, ApiError> {
        match stop {
            None => Ok(vec![]),
            Some(Value::Null) => Ok(vec![]),
            Some(Value::String(s)) => {
                if s.is_empty() {
                    return Err(ApiError::BadRequest {
                        message: "stop string must not be empty".to_string(),
                        code: "invalid_stop",
                    });
                }
                Ok(vec![s.clone()])
            }
            Some(Value::Array(arr)) => {
                if arr.is_empty() {
                    return Err(ApiError::BadRequest {
                        message: "stop array must not be empty".to_string(),
                        code: "invalid_stop",
                    });
                }
                if arr.len() > 4 {
                    return Err(ApiError::BadRequest {
                        message: format!("stop array has {} elements; maximum is 4", arr.len()),
                        code: "invalid_stop",
                    });
                }
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    match item {
                        Value::String(s) => {
                            if s.is_empty() {
                                return Err(ApiError::BadRequest {
                                    message: "stop string must not be empty".to_string(),
                                    code: "invalid_stop",
                                });
                            }
                            out.push(s.clone());
                        }
                        _ => {
                            return Err(ApiError::BadRequest {
                                message: "each element of stop must be a string".to_string(),
                                code: "invalid_stop",
                            });
                        }
                    }
                }
                Ok(out)
            }
            Some(_) => Err(ApiError::BadRequest {
                message: "stop must be a string or array of strings".to_string(),
                code: "invalid_stop",
            }),
        }
    }

    /// Reject OpenAI fields that are parsed but not yet implemented.
    ///
    /// Note: `stream=true` is now handled by the streaming path in `chat_completions`
    /// and is intentionally NOT rejected here.
    fn reject_unsupported(req: &ChatCompletionRequest) -> Result<(), ApiError> {
        if req.tools.is_some() || req.tool_choice.is_some() {
            return Err(ApiError::BadRequest {
                message: "tools and tool_choice are not supported by this server".to_string(),
                code: "unsupported_feature",
            });
        }
        if req.logprobs.unwrap_or(false) {
            return Err(ApiError::BadRequest {
                message: "logprobs is not supported by this server".to_string(),
                code: "unsupported_feature",
            });
        }
        if req.n.unwrap_or(1) > 1 {
            return Err(ApiError::BadRequest {
                message: "n > 1 is not supported".to_string(),
                code: "unsupported_feature",
            });
        }
        if let Some(fmt) = &req.response_format
            && fmt.r#type != "text"
        {
            return Err(ApiError::BadRequest {
                message: format!(
                    "response_format.type '{}' is not supported; use 'text'",
                    fmt.r#type
                ),
                code: "unsupported_feature",
            });
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Extract a plain text string from a message content value.
    /// Returns `Err` for non-text content parts (image, audio, file).
    fn message_text(content: &MessageContent) -> Result<String, ApiError> {
        match content {
            MessageContent::Text(text) => Ok(text.clone()),
            MessageContent::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    if part.kind != "text" {
                        return Err(ApiError::BadRequest {
                            message: format!(
                                "content part type '{}' is not supported; only 'text' parts are accepted",
                                part.kind
                            ),
                            code: "unsupported_feature",
                        });
                    }
                    out.push_str(part.text.as_deref().unwrap_or(""));
                }
                Ok(out)
            }
        }
    }

    /// Build a single prompt string from the full message list using Qwen ChatML format.
    ///
    /// Format (one block per message, in order):
    /// ```text
    /// <|im_start|>system
    /// {content}<|im_end|>
    /// <|im_start|>user
    /// {content}<|im_end|>
    /// <|im_start|>assistant
    /// {content}<|im_end|>
    /// ```
    /// The final line is the open generation prompt `<|im_start|>assistant\n` — no closing
    /// `<|im_end|>` — so the model generates from there.
    ///
    /// Only the roles `system`, `user`, and `assistant` are supported. Any other role
    /// returns `Err` so the handler can respond with HTTP 400.
    fn render_prompt(messages: &[Message]) -> Result<String, ApiError> {
        let mut buf = String::new();
        for msg in messages {
            let content = message_text(&msg.content)?;
            match msg.role.as_str() {
                "system" | "user" | "assistant" => {
                    buf.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n",
                        msg.role, content
                    ));
                }
                "tool" | "developer" => {
                    return Err(ApiError::BadRequest {
                        message: format!("role '{}' is not supported by this server", msg.role),
                        code: "unsupported_feature",
                    });
                }
                other => {
                    return Err(ApiError::BadRequest {
                        message: format!(
                            "unsupported role '{other}'; must be 'system', 'user', or 'assistant'"
                        ),
                        code: "invalid_role",
                    });
                }
            }
        }
        // Open generation turn — model generates from here.
        buf.push_str("<|im_start|>assistant\n");
        Ok(buf)
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Maps a `GenerateOutput` to the OpenAI `finish_reason` string.
    ///
    /// Returns `"stop"` when the library explicitly ended generation via a stop
    /// condition (EOS token, stop-token-id, or stop-string match); `"length"` when
    /// the token budget was exhausted without a stop condition.
    pub(super) fn finish_reason_for(
        output: &lattice_inference::model::qwen35_config::GenerateOutput,
    ) -> &'static str {
        if output.stopped { "stop" } else { "length" }
    }

    // Handlers
    // -----------------------------------------------------------------------

    pub async fn health() -> Json<HealthResponse> {
        Json(HealthResponse { status: "ok" })
    }

    pub async fn chat_completions(
        State(state): State<AppState>,
        result: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
    ) -> Result<Response, ApiError> {
        // Surface JSON extraction failures as structured 400 responses.
        // Log the raw parser message server-side; never forward it to clients.
        let Json(req) = result.map_err(|rejection| {
            if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                ApiError::PayloadTooLarge {
                    message: "request body exceeds 1 MiB limit".to_string(),
                }
            } else {
                eprintln!("invalid request body: {}", rejection.body_text());
                ApiError::BadRequest {
                    message: "invalid JSON request body".to_string(),
                    code: "invalid_request_body",
                }
            }
        })?;

        // Reject unsupported OpenAI features before any further processing.
        reject_unsupported(&req)?;

        // Validate that the caller targets the served model.
        if req.model != state.model_id {
            return Err(ApiError::BadRequest {
                message: format!(
                    "model '{}' is not loaded; this server serves '{}'",
                    req.model, state.model_id
                ),
                code: "model_not_found",
            });
        }

        if req.messages.is_empty() {
            return Err(ApiError::BadRequest {
                message: "messages must not be empty".to_string(),
                code: "invalid_messages",
            });
        }

        // Require the conversation to end with a user turn (Qwen ChatML constraint).
        let last_role = req.messages.last().map(|m| m.role.as_str()).unwrap_or("");
        if last_role != "user" {
            return Err(ApiError::BadRequest {
                message: "the last message must have role 'user'".to_string(),
                code: "invalid_messages",
            });
        }

        // Validate and resolve sampling parameters.
        let max_tokens = validate_max_tokens(
            req.max_tokens,
            req.max_completion_tokens,
            state.default_max_tokens,
            state.max_tokens_cap,
        )?;
        let temperature = validate_temperature(req.temperature)?;
        let top_p = validate_top_p(req.top_p)?;

        // Render the full conversation into a ChatML prompt.  Returns 400 for
        // any unsupported role or content-part type encountered.
        let prompt = render_prompt(&req.messages)?;

        // Preflight: reject prompts that would overflow the model's context window
        // before entering the blocking generation path.  This converts what would
        // otherwise be a panic inside spawn_blocking into a clean 400 response.
        let prompt_token_count = state.model.tokenize_len(&prompt);
        let max_context = state.model.max_context();
        if prompt_token_count == 0 || prompt_token_count.saturating_add(max_tokens) > max_context {
            return Err(ApiError::BadRequest {
                message: format!(
                    "prompt ({prompt_token_count} tokens) plus max_tokens ({max_tokens}) \
                     exceeds model context window ({max_context})"
                ),
                code: "context_length_exceeded",
            });
        }

        let stop_strings = parse_stop_strings(&req.stop)?;

        let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
            max_new_tokens: max_tokens,
            temperature,
            top_p,
            seed: req.seed,
            stop_strings,
            ..Default::default()
        };

        let model = state.model.clone();

        // Compute shared response metadata before branching on stream flag.
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let seq = state.request_counter.fetch_add(1, Ordering::Relaxed);
        let response_id = format!("chatcmpl-{created}-{seq}");

        if req.stream == Some(true) {
            // --- Streaming path ---
            //
            // `generate_streaming` is a synchronous blocking function.  We run it
            // on the blocking thread pool and feed incremental deltas into an
            // unbounded MPSC channel.  The async SSE handler drains the channel
            // and converts each message to an OpenAI `chat.completion.chunk` event.
            // An unbounded channel is acceptable here because the channel depth is
            // bounded by `max_tokens` (capped at `max_tokens_cap`): the producer
            // sends at most one `Delta` per generated token and generation halts at
            // the cap, so the worst-case buffer is a few thousand short strings —
            // the same order the non-streaming path already holds as one buffered
            // string. There is no true backpressure (an unbounded send never
            // blocks); if the client disconnects mid-stream the producer keeps
            // generating to the cap and the ignored send errors drain harmlessly.
            // Per-token backpressure / disconnect-cancellation is a future refinement.
            let (tx, rx) = futures::channel::mpsc::unbounded::<StreamMsg>();

            let stream_id = response_id.clone();
            let stream_model = state.model_id.clone();

            // Both backends funnel their result through this closure so the
            // "generated_tokens > max_tokens invariant, then finish_reason_for"
            // logic is written exactly once and shared by CPU and Metal.
            let finish_streaming = {
                let tx = tx.clone();
                move |output: GenerateOutput| {
                    if output.generated_tokens > max_tokens {
                        eprintln!(
                            "generation invariant violation: generated_tokens={} max_tokens={}",
                            output.generated_tokens, max_tokens
                        );
                        let _ = tx.unbounded_send(StreamMsg::Failed);
                    } else {
                        let finish_reason = finish_reason_for(&output);
                        let _ = tx.unbounded_send(StreamMsg::Done { finish_reason });
                    }
                }
            };

            match model {
                ModelBackend::Cpu(cpu_model) => {
                    tokio::task::spawn_blocking(move || {
                        let tx_delta = tx.clone();
                        let result = cpu_model.generate_streaming(&prompt, &gen_cfg, |delta| {
                            // Send each incremental text delta; ignore if the receiver
                            // dropped (client disconnected).
                            let _ = tx_delta.unbounded_send(StreamMsg::Delta(delta.to_string()));
                        });
                        match result {
                            Ok(output) => finish_streaming(output),
                            Err(e) => {
                                eprintln!("generation error (streaming): {e}");
                                let _ = tx.unbounded_send(StreamMsg::Failed);
                            }
                        }
                    });
                }
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { handle, .. } => {
                    tokio::spawn(async move {
                        let tx_delta = tx.clone();
                        let result = handle
                            .generate_streaming(prompt, gen_cfg, move |delta| {
                                tx_delta
                                    .unbounded_send(StreamMsg::Delta(delta.to_string()))
                                    .is_ok()
                            })
                            .await;
                        match result {
                            Ok(output) => finish_streaming(output),
                            Err(e) => {
                                eprintln!("generation error (streaming, metal): {e:?}");
                                let _ = tx.unbounded_send(StreamMsg::Failed);
                            }
                        }
                    });
                }
            }

            // Build the SSE stream.
            //
            // Event order (OpenAI spec):
            //   1. Role chunk: `delta: {"role":"assistant"}`, finish_reason absent.
            //   2. One content chunk per Delta: `delta: {"content":"..."}`, finish_reason absent.
            //   3. Finish chunk: `delta: {}`, finish_reason set.
            //   4. Literal `data: [DONE]` sentinel.
            let role_chunk = {
                let chunk = ChatCompletionChunk {
                    id: stream_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: stream_model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: Some("assistant"),
                            content: None,
                        },
                        finish_reason: None,
                    }],
                };
                let data = serde_json::to_string(&chunk).unwrap_or_default();
                Ok::<Event, std::convert::Infallible>(Event::default().data(data))
            };

            // Map each StreamMsg from the channel into one or two SSE events.
            let body_stream = rx.flat_map(move |msg| {
                let id = stream_id.clone();
                let mdl = stream_model.clone();
                match msg {
                    StreamMsg::Delta(text) => {
                        let chunk = ChatCompletionChunk {
                            id,
                            object: "chat.completion.chunk",
                            created,
                            model: mdl,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let events: Vec<Result<Event, std::convert::Infallible>> =
                            vec![Ok(Event::default().data(data))];
                        futures::stream::iter(events)
                    }
                    StreamMsg::Done { finish_reason } => {
                        let chunk = ChatCompletionChunk {
                            id,
                            object: "chat.completion.chunk",
                            created,
                            model: mdl,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some(finish_reason),
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let events: Vec<Result<Event, std::convert::Infallible>> = vec![
                            Ok(Event::default().data(data)),
                            Ok(Event::default().data("[DONE]")),
                        ];
                        futures::stream::iter(events)
                    }
                    StreamMsg::Failed => {
                        // Emit a finish chunk with reason "stop" so the client
                        // receives a well-formed termination, then the [DONE]
                        // sentinel.  The error was already logged in the producer.
                        let chunk = ChatCompletionChunk {
                            id,
                            object: "chat.completion.chunk",
                            created,
                            model: mdl,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some("stop"),
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let events: Vec<Result<Event, std::convert::Infallible>> = vec![
                            Ok(Event::default().data(data)),
                            Ok(Event::default().data("[DONE]")),
                        ];
                        futures::stream::iter(events)
                    }
                }
            });

            let sse_stream = futures::stream::once(async move { role_chunk }).chain(body_stream);

            Ok(Sse::new(sse_stream)
                .keep_alive(KeepAlive::default())
                .into_response())
        } else {
            // --- Non-streaming path (CPU leg byte-identical to the original) ---
            let output = match model {
                ModelBackend::Cpu(cpu_model) => {
                    // `generate` is CPU-bound blocking work; run it on the blocking thread pool.
                    tokio::task::spawn_blocking(move || cpu_model.generate(&prompt, &gen_cfg))
                        .await
                        .map_err(|e| {
                            eprintln!("task join error: {e}");
                            ApiError::Internal {
                                message: "inference failed".to_string(),
                            }
                        })?
                        .map_err(|e| {
                            eprintln!("generation error: {e}");
                            ApiError::Internal {
                                message: "inference failed".to_string(),
                            }
                        })?
                }
                #[cfg(feature = "metal-gpu")]
                ModelBackend::Metal { handle, .. } => handle
                    .generate_streaming(prompt, gen_cfg, |_delta| true)
                    .await
                    .map_err(|e| {
                        eprintln!("generation error (metal): {e:?}");
                        ApiError::Internal {
                            message: "inference failed".to_string(),
                        }
                    })?,
            };

            // Distinguish "hit token cap" from "natural stop" (EOS / stop token / stop string).
            // `GenerateOutput.stopped` carries the explicit stop reason set by the library.
            // Log and return 500 if the invariant is violated.
            if output.generated_tokens > max_tokens {
                eprintln!(
                    "generation invariant violation: generated_tokens={} max_tokens={}",
                    output.generated_tokens, max_tokens
                );
                return Err(ApiError::Internal {
                    message: "inference failed".to_string(),
                });
            }
            let finish_reason = finish_reason_for(&output);

            let response = ChatCompletionResponse {
                id: response_id,
                object: "chat.completion".to_string(),
                created,
                model: state.model_id.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: output.text.clone(),
                    },
                    finish_reason: finish_reason.to_string(),
                }],
                usage: Usage {
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.generated_tokens,
                    total_tokens: output.prompt_tokens + output.generated_tokens,
                },
            };

            Ok(Json(response).into_response())
        }
    }

    // -----------------------------------------------------------------------
    // Router
    // -----------------------------------------------------------------------

    pub fn router(state: AppState) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/chat/completions", post(chat_completions))
            .layer(DefaultBodyLimit::max(REQUEST_BODY_LIMIT_BYTES))
            .with_state(state)
    }

    // -----------------------------------------------------------------------
    // Tests — pure helper functions; no model construction needed
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn validate_max_tokens_rejects_zero() {
            let err = validate_max_tokens(Some(0), None, 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_max_tokens",
                    ..
                }
            ));
        }

        #[test]
        fn validate_max_tokens_rejects_above_cap() {
            let err = validate_max_tokens(Some(9999), None, 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "max_tokens_exceeds_limit",
                    ..
                }
            ));
        }

        #[test]
        fn validate_max_tokens_uses_default_when_absent() {
            assert_eq!(validate_max_tokens(None, None, 128, 4096).unwrap(), 128);
        }

        #[test]
        fn validate_max_tokens_alias_agrees() {
            assert_eq!(
                validate_max_tokens(Some(512), Some(512), 256, 4096).unwrap(),
                512
            );
        }

        #[test]
        fn validate_max_tokens_alias_conflict_rejected() {
            let err = validate_max_tokens(Some(100), Some(200), 256, 4096).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_request",
                    ..
                }
            ));
        }

        #[test]
        fn validate_temperature_rejects_negative() {
            let err = validate_temperature(Some(-0.1)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_temperature",
                    ..
                }
            ));
        }

        #[test]
        fn validate_temperature_rejects_above_two() {
            let err = validate_temperature(Some(2.1)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_temperature",
                    ..
                }
            ));
        }

        #[test]
        fn validate_temperature_accepts_boundary() {
            assert_eq!(validate_temperature(Some(0.0)).unwrap(), 0.0);
            assert_eq!(validate_temperature(Some(2.0)).unwrap(), 2.0);
        }

        #[test]
        fn validate_top_p_rejects_zero() {
            let err = validate_top_p(Some(0.0)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_top_p",
                    ..
                }
            ));
        }

        #[test]
        fn validate_top_p_rejects_above_one() {
            let err = validate_top_p(Some(1.1)).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_top_p",
                    ..
                }
            ));
        }

        #[test]
        fn validate_top_p_accepts_one() {
            assert_eq!(validate_top_p(Some(1.0)).unwrap(), 1.0);
        }

        #[test]
        fn render_prompt_multi_message_chatml() {
            let messages = vec![
                Message {
                    role: "system".to_string(),
                    content: MessageContent::Text("Be helpful.".to_string()),
                },
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("Hello".to_string()),
                },
            ];
            let prompt = render_prompt(&messages).unwrap();
            assert!(prompt.contains("<|im_start|>system\nBe helpful.<|im_end|>"));
            assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
            assert!(prompt.ends_with("<|im_start|>assistant\n"));
        }

        #[test]
        fn render_prompt_rejects_invalid_role() {
            let messages = vec![Message {
                role: "function".to_string(),
                content: MessageContent::Text("data".to_string()),
            }];
            let err = render_prompt(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_role",
                    ..
                }
            ));
        }

        #[test]
        fn render_prompt_rejects_tool_role() {
            let messages = vec![Message {
                role: "tool".to_string(),
                content: MessageContent::Text("result".to_string()),
            }];
            let err = render_prompt(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn render_prompt_rejects_non_text_content_part() {
            let messages = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![ContentPart {
                    kind: "image_url".to_string(),
                    text: None,
                }]),
            }];
            let err = render_prompt(&messages).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        // Exercises finish_reason_for via the real helper function used by the handler.
        // A cap-reached output has stopped=false → "length".
        // A stop-condition output has stopped=true → "stop".
        #[test]
        fn finish_reason_length_only_at_cap() {
            use lattice_inference::model::qwen35_config::GenerateOutput;
            let cap = GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: 10,
                generated_tokens: 64,
                stopped: false,
                stop_reason: Some(lattice_inference::StopReason::Length),
            };
            assert_eq!(super::finish_reason_for(&cap), "length");

            let natural = GenerateOutput {
                text: "hello".into(),
                token_ids: vec![1, 2, 3],
                prompt_tokens: 10,
                generated_tokens: 3,
                stopped: true,
                stop_reason: Some(lattice_inference::StopReason::Eos),
            };
            assert_eq!(super::finish_reason_for(&natural), "stop");
        }

        // M1 regression: a stop-string hit at exactly max_new_tokens must yield "stop",
        // not "length". The old token-count formula (generated == cap → "length") would
        // mislabel this case because the stop-completing token is included in generated_ids
        // before the stop is detected.
        //
        // This test calls the real finish_reason_for helper. It is RED when
        // finish_reason_for reverts to the old `generated_tokens == max_tokens` formula.
        #[test]
        fn finish_reason_stop_string_at_cap_is_stop_not_length() {
            use lattice_inference::model::qwen35_config::GenerateOutput;
            let max_tokens: usize = 4;
            // stop-string hit at exactly the token budget:
            // stopped=true because a stop string matched; generated_tokens==max_tokens
            // because the matching token is included in generated_ids before truncation.
            let output = GenerateOutput {
                text: "hi".into(),
                token_ids: vec![1, 2, 3, 4],
                prompt_tokens: 5,
                generated_tokens: max_tokens,
                stopped: true,
                stop_reason: Some(lattice_inference::StopReason::Eos),
            };
            assert_eq!(
                super::finish_reason_for(&output),
                "stop",
                "stop-string hit at cap must yield finish_reason=stop, not length"
            );
        }

        // Natural length cap (no stop condition) must still yield "length".
        #[test]
        fn finish_reason_natural_length_cap_is_length() {
            use lattice_inference::model::qwen35_config::GenerateOutput;
            let output = GenerateOutput {
                text: "hi".into(),
                token_ids: vec![1, 2, 3, 4],
                prompt_tokens: 5,
                generated_tokens: 4,
                stopped: false,
                stop_reason: Some(lattice_inference::StopReason::Length),
            };
            assert_eq!(super::finish_reason_for(&output), "length");
        }

        #[test]
        fn reject_unsupported_stream_true_ok() {
            // stream=true is now handled by the streaming path and must NOT be
            // rejected by reject_unsupported.
            let req = ChatCompletionRequest {
                model: "m".to_string(),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stream: Some(true),
                stop: None,
                seed: None,
                response_format: None,
                tools: None,
                tool_choice: None,
                logprobs: None,
                n: None,
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        // -----------------------------------------------------------------------
        // ChatCompletionChunk serialization
        // -----------------------------------------------------------------------

        #[test]
        fn chunk_content_delta_serializes_correctly() {
            let chunk = ChatCompletionChunk {
                id: "chatcmpl-1-0".to_string(),
                object: "chat.completion.chunk",
                created: 1_000_000,
                model: "test-model".to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: Some("Hello".to_string()),
                    },
                    finish_reason: None,
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap();
            assert!(
                json.contains("\"object\":\"chat.completion.chunk\""),
                "must contain object field"
            );
            assert!(
                json.contains("\"delta\":{\"content\":\"Hello\"}"),
                "delta must contain only content when role is None"
            );
            // finish_reason must be absent (not null) when None
            assert!(
                !json.contains("finish_reason"),
                "finish_reason must be omitted when None"
            );
        }

        #[test]
        fn chunk_finish_delta_serializes_correctly() {
            let chunk = ChatCompletionChunk {
                id: "chatcmpl-1-0".to_string(),
                object: "chat.completion.chunk",
                created: 1_000_000,
                model: "test-model".to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop"),
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap();
            assert!(
                json.contains("\"finish_reason\":\"stop\""),
                "finish chunk must include finish_reason"
            );
            // delta should be empty object since both role and content are None
            assert!(
                json.contains("\"delta\":{}"),
                "finish chunk delta must be empty object"
            );
        }

        #[test]
        fn reject_unsupported_n_gt_1() {
            let req = ChatCompletionRequest {
                model: "m".to_string(),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stream: None,
                stop: None,
                seed: None,
                response_format: None,
                tools: None,
                tool_choice: None,
                logprobs: None,
                n: Some(3),
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn reject_unsupported_response_format_json() {
            let req = ChatCompletionRequest {
                model: "m".to_string(),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stream: None,
                stop: None,
                seed: None,
                response_format: Some(ResponseFormat {
                    r#type: "json_object".to_string(),
                }),
                tools: None,
                tool_choice: None,
                logprobs: None,
                n: None,
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        // -----------------------------------------------------------------------
        // reject_unsupported — remaining fields
        // -----------------------------------------------------------------------

        fn bare_req() -> ChatCompletionRequest {
            ChatCompletionRequest {
                model: "m".to_string(),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stream: None,
                stop: None,
                seed: None,
                response_format: None,
                tools: None,
                tool_choice: None,
                logprobs: None,
                n: None,
            }
        }

        #[test]
        fn reject_unsupported_tools_rejected() {
            let req = ChatCompletionRequest {
                tools: Some(serde_json::json!([])),
                ..bare_req()
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn reject_unsupported_tool_choice_rejected() {
            let req = ChatCompletionRequest {
                tool_choice: Some(serde_json::json!("auto")),
                ..bare_req()
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn reject_unsupported_logprobs_rejected() {
            let req = ChatCompletionRequest {
                logprobs: Some(true),
                ..bare_req()
            };
            let err = reject_unsupported(&req).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        #[test]
        fn reject_unsupported_stop_now_accepted() {
            // stop is no longer rejected by reject_unsupported; it is parsed separately.
            let req = ChatCompletionRequest {
                stop: Some(serde_json::json!("</s>")),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        // -----------------------------------------------------------------------
        // parse_stop_strings
        // -----------------------------------------------------------------------

        #[test]
        fn parse_stop_strings_null_gives_empty() {
            assert_eq!(parse_stop_strings(&None).unwrap(), Vec::<String>::new());
            assert_eq!(
                parse_stop_strings(&Some(serde_json::Value::Null)).unwrap(),
                Vec::<String>::new()
            );
        }

        #[test]
        fn parse_stop_strings_single_string_gives_vec_of_one() {
            let v = parse_stop_strings(&Some(serde_json::json!("</s>"))).unwrap();
            assert_eq!(v, vec!["</s>".to_string()]);
        }

        #[test]
        fn parse_stop_strings_array_of_two_accepted() {
            let v = parse_stop_strings(&Some(serde_json::json!(["</s>", "\nUser:"]))).unwrap();
            assert_eq!(v, vec!["</s>".to_string(), "\nUser:".to_string()]);
        }

        #[test]
        fn parse_stop_strings_empty_array_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!([]))).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_array_over_four_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!(["a", "b", "c", "d", "e"])))
                .unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_array_with_number_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!(["ok", 42]))).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_empty_string_element_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!(["ok", ""]))).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_empty_string_scalar_rejected() {
            let err = parse_stop_strings(&Some(serde_json::json!(""))).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_stop",
                    ..
                }
            ));
        }

        #[test]
        fn parse_stop_strings_array_exactly_four_accepted() {
            let v = parse_stop_strings(&Some(serde_json::json!(["a", "b", "c", "d"]))).unwrap();
            assert_eq!(v.len(), 4);
        }

        #[test]
        fn reject_unsupported_stream_false_ok() {
            // stream=false must not trigger a rejection.
            let req = ChatCompletionRequest {
                stream: Some(false),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        #[test]
        fn reject_unsupported_n_1_ok() {
            let req = ChatCompletionRequest {
                n: Some(1),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        #[test]
        fn reject_unsupported_response_format_text_ok() {
            let req = ChatCompletionRequest {
                response_format: Some(ResponseFormat {
                    r#type: "text".to_string(),
                }),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        #[test]
        fn reject_unsupported_logprobs_false_ok() {
            let req = ChatCompletionRequest {
                logprobs: Some(false),
                ..bare_req()
            };
            assert!(reject_unsupported(&req).is_ok());
        }

        // -----------------------------------------------------------------------
        // validate_max_tokens — additional edge cases
        // -----------------------------------------------------------------------

        #[test]
        fn validate_max_tokens_at_exactly_cap_ok() {
            assert_eq!(
                validate_max_tokens(Some(4096), None, 256, 4096).unwrap(),
                4096
            );
        }

        #[test]
        fn validate_max_tokens_max_completion_only_ok() {
            assert_eq!(
                validate_max_tokens(None, Some(512), 256, 4096).unwrap(),
                512
            );
        }

        // -----------------------------------------------------------------------
        // validate_temperature — default path
        // -----------------------------------------------------------------------

        #[test]
        fn validate_temperature_none_uses_default() {
            assert_eq!(validate_temperature(None).unwrap(), 0.7);
        }

        // -----------------------------------------------------------------------
        // validate_top_p — default path
        // -----------------------------------------------------------------------

        #[test]
        fn validate_top_p_none_uses_default() {
            assert_eq!(validate_top_p(None).unwrap(), 0.9);
        }

        // -----------------------------------------------------------------------
        // render_prompt — additional cases
        // -----------------------------------------------------------------------

        #[test]
        fn render_prompt_user_only() {
            let msgs = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("hi".to_string()),
            }];
            let prompt = render_prompt(&msgs).unwrap();
            assert_eq!(
                prompt,
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
            );
        }

        #[test]
        fn render_prompt_multi_turn_assistant() {
            let msgs = vec![
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("q1".to_string()),
                },
                Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text("a1".to_string()),
                },
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Text("q2".to_string()),
                },
            ];
            let prompt = render_prompt(&msgs).unwrap();
            assert!(prompt.contains("<|im_start|>user\nq1<|im_end|>"));
            assert!(prompt.contains("<|im_start|>assistant\na1<|im_end|>"));
            assert!(prompt.contains("<|im_start|>user\nq2<|im_end|>"));
            assert!(prompt.ends_with("<|im_start|>assistant\n"));
        }

        #[test]
        fn render_prompt_content_parts_text_ok() {
            let msgs = vec![Message {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![
                    ContentPart {
                        kind: "text".to_string(),
                        text: Some("hello".to_string()),
                    },
                    ContentPart {
                        kind: "text".to_string(),
                        text: Some(" world".to_string()),
                    },
                ]),
            }];
            let prompt = render_prompt(&msgs).unwrap();
            assert!(prompt.contains("<|im_start|>user\nhello world<|im_end|>"));
        }

        #[test]
        fn render_prompt_rejects_developer_role() {
            let msgs = vec![Message {
                role: "developer".to_string(),
                content: MessageContent::Text("system prompt".to_string()),
            }];
            let err = render_prompt(&msgs).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }

        // -----------------------------------------------------------------------
        // Error envelope JSON shape
        // -----------------------------------------------------------------------

        #[test]
        fn error_envelope_bad_request_shape() {
            let err = ApiError::BadRequest {
                message: "test error".to_string(),
                code: "invalid_request",
            };
            // Verify the error serialises to the OpenAI envelope shape:
            // {"error":{"message":"...","type":"invalid_request_error","code":"...","param":null}}
            let body = ErrorBody {
                error: ErrorDetail {
                    message: "test error".to_string(),
                    r#type: "invalid_request_error",
                    code: "invalid_request".to_string(),
                    param: None,
                },
            };
            let json = serde_json::to_string(&body).unwrap();
            assert!(json.contains("\"error\""));
            assert!(json.contains("\"message\":\"test error\""));
            assert!(json.contains("\"type\":\"invalid_request_error\""));
            assert!(json.contains("\"code\":\"invalid_request\""));
            assert!(json.contains("\"param\":null"));
            // Ensure it is NOT a bare message — must be nested under "error".
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(v["error"].is_object(), "top-level key must be 'error'");
            // Variant check kept separate so we know err itself was constructed correctly.
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "invalid_request",
                    ..
                }
            ));
        }

        #[test]
        fn error_envelope_payload_too_large_shape() {
            let body = ErrorBody {
                error: ErrorDetail {
                    message: "request body exceeds 1 MiB limit".to_string(),
                    r#type: "invalid_request_error",
                    code: "request_body_too_large".to_string(),
                    param: None,
                },
            };
            let json = serde_json::to_string(&body).unwrap();
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(v["error"]["code"], "request_body_too_large");
        }

        #[test]
        fn error_envelope_internal_shape() {
            let body = ErrorBody {
                error: ErrorDetail {
                    message: "inference failed".to_string(),
                    r#type: "server_error",
                    code: "internal_error".to_string(),
                    param: None,
                },
            };
            let json = serde_json::to_string(&body).unwrap();
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(v["error"]["type"], "server_error");
            assert_eq!(v["error"]["code"], "internal_error");
        }

        // -----------------------------------------------------------------------
        // message_text helper
        // -----------------------------------------------------------------------

        #[test]
        fn message_text_plain_string() {
            let content = MessageContent::Text("hello".to_string());
            assert_eq!(message_text(&content).unwrap(), "hello");
        }

        #[test]
        fn message_text_parts_concatenates() {
            let content = MessageContent::Parts(vec![
                ContentPart {
                    kind: "text".to_string(),
                    text: Some("foo".to_string()),
                },
                ContentPart {
                    kind: "text".to_string(),
                    text: Some("bar".to_string()),
                },
            ]);
            assert_eq!(message_text(&content).unwrap(), "foobar");
        }

        #[test]
        fn message_text_parts_rejects_image() {
            let content = MessageContent::Parts(vec![ContentPart {
                kind: "image_url".to_string(),
                text: None,
            }]);
            let err = message_text(&content).unwrap_err();
            assert!(matches!(
                err,
                ApiError::BadRequest {
                    code: "unsupported_feature",
                    ..
                }
            ));
        }
    }
}

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
            run_chat(&model, max_tokens, temperature, tokenizer_dir.as_deref());
        }
        Command::Serve {
            model,
            host,
            port,
            max_tokens,
            model_id,
            tokenizer_dir,
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
            };
            eprintln!("Model loaded. Serving as '{served_model_id}'.");

            let state = serve::AppState {
                model: model_backend,
                default_max_tokens: max_tokens,
                max_tokens_cap: 4096,
                model_id: served_model_id.clone(),
                request_counter: Arc::new(AtomicU64::new(0)),
            };

            let app = serve::router(state);

            let addr = format!("{host}:{port}");
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error: failed to bind to {addr}: {e}");
                    std::process::exit(1);
                }
            };
            eprintln!(
                "Listening on {addr}  (model: {served_model_id}, max_tokens default: {max_tokens})"
            );
            eprintln!("  POST /v1/chat/completions");
            eprintln!("  GET  /health");

            let shutdown = async {
                if let Err(e) = tokio::signal::ctrl_c().await {
                    eprintln!("Error waiting for shutdown signal: {e}");
                }
                eprintln!("Shutdown signal received, draining connections...");
            };

            if let Err(e) = axum::serve(listener, app)
                .with_graceful_shutdown(shutdown)
                .await
            {
                eprintln!("Server error: {e}");
                std::process::exit(1);
            }
        }
    }
}
