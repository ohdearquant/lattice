//! lattice_serve — OpenAI-compatible HTTP serving endpoint for Lattice.
//!
//! Exposes the Metal GPU engine over the same `/v1/chat/completions` API that
//! ollama, llama.cpp's server, and most LLM benchmark harnesses already speak,
//! so any OpenAI-compatible client can point at lattice with zero adapter code.
//!
//! # Usage
//!
//! ```text
//! lattice_serve --model qwen3.5-0.8b               # resolves from ~/.lattice/models
//! lattice_serve --model ~/.lattice/models/qwen3.6-27b-q4 --port 11435
//! ```
//!
//! Then point any OpenAI client at `http://127.0.0.1:11435/v1`:
//!
//! ```text
//! curl http://127.0.0.1:11435/v1/chat/completions -H 'content-type: application/json' \
//!   -d '{"model":"lattice","messages":[{"role":"user","content":"hi"}],"stream":true}'
//! ```
//!
//! # Endpoints
//!
//! - `POST /v1/chat/completions` — streaming (SSE) and non-streaming, OpenAI shape
//! - `GET  /v1/models`           — advertises the single loaded model
//! - `GET  /health`              — liveness probe (`ok`)
//!
//! # Design
//!
//! `MetalQwen35State` owns raw `metal::*` objects and is `!Send`, so it lives on
//! one dedicated worker thread for the whole process lifetime. The async axum
//! handlers never touch Metal directly: each request ships a `Job` (messages +
//! sampling config + a reply channel) to the worker over a tokio mpsc, and the
//! worker drives `chat_completion_streaming`, forwarding each token delta back.
//! Generation is therefore serialized — correct for a single-GPU local engine
//! (the same default ollama uses). The ChatML template and `<|im_end|>` stop
//! handling are reused verbatim from the engine; this binary only translates the
//! OpenAI wire format on either side.

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("lattice_serve requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = imp::run() {
            eprintln!("lattice_serve: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod imp {
    use axum::{
        Json, Router,
        extract::State,
        http::StatusCode,
        response::{
            IntoResponse, Response,
            sse::{Event, KeepAlive, Sse},
        },
        routing::{get, post},
    };
    use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{
        GenerateConfig, QWEN_CHAT_IM_END_TOKEN_ID, Qwen35Config,
    };
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;

    // ─── worker protocol ─────────────────────────────────────────────────────

    /// One token-stream event from the worker back to a request handler.
    enum Ev {
        Delta(String),
        Done {
            prompt_tokens: usize,
            completion_tokens: usize,
        },
    }

    /// A generation request handed to the single GPU worker thread.
    struct Job {
        messages: Vec<ChatMessage>,
        cfg: GenerateConfig,
        tx: mpsc::UnboundedSender<Ev>,
    }

    /// Server-side sampling defaults, overridable per-request.
    #[derive(Clone)]
    struct Defaults {
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        reasoning_budget: Option<usize>,
    }

    #[derive(Clone)]
    struct AppState {
        jobs: mpsc::UnboundedSender<Job>,
        model_id: Arc<str>,
        defaults: Defaults,
    }

    // ─── OpenAI request shapes ───────────────────────────────────────────────

    #[derive(Deserialize)]
    struct ChatReq {
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        messages: Vec<InMsg>,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        top_k: Option<usize>,
        #[serde(default)]
        max_tokens: Option<usize>,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        stream: Option<bool>,
        // Lattice extensions (ignored by stock OpenAI clients).
        #[serde(default)]
        repetition_penalty: Option<f32>,
        #[serde(default)]
        reasoning_budget: Option<usize>,
    }

    #[derive(Deserialize)]
    struct InMsg {
        role: String,
        #[serde(default)]
        content: Value,
    }

    /// OpenAI message content is either a string or an array of typed parts
    /// (`[{"type":"text","text":"..."}]`). Flatten both to plain text.
    fn content_text(v: &Value) -> String {
        match v {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }

    fn to_chat_message(m: &InMsg) -> ChatMessage {
        let content = content_text(&m.content);
        match m.role.as_str() {
            "system" => ChatMessage::system(content),
            "assistant" => ChatMessage::assistant(content),
            _ => ChatMessage::user(content),
        }
    }

    fn build_cfg(req: &ChatReq, d: &Defaults) -> GenerateConfig {
        let reasoning_budget = req
            .reasoning_budget
            .filter(|&n| n > 0)
            .or(d.reasoning_budget);
        GenerateConfig {
            max_new_tokens: req.max_tokens.unwrap_or(d.max_tokens),
            temperature: req.temperature.unwrap_or(d.temperature),
            top_k: req.top_k.unwrap_or(d.top_k),
            top_p: req.top_p.unwrap_or(d.top_p),
            repetition_penalty: req.repetition_penalty.unwrap_or(d.repetition_penalty),
            seed: req.seed,
            stop_token_ids: vec![QWEN_CHAT_IM_END_TOKEN_ID],
            enable_thinking: true,
            enable_mtp: None,
            grammar: None,
            stop_strings: vec![],
            reasoning_budget,
        }
    }

    // ─── GPU worker thread ───────────────────────────────────────────────────

    /// Spawn the dedicated thread that owns the `!Send` Metal state. Loads the
    /// model, signals readiness (or a load error) over `ready`, then serves jobs
    /// serially until all `Job` senders drop.
    fn spawn_worker(
        model_dir: std::path::PathBuf,
        tokenizer_path: std::path::PathBuf,
        is_q4: bool,
        ready: std::sync::mpsc::Sender<Result<String, String>>,
    ) -> mpsc::UnboundedSender<Job> {
        let (job_tx, mut job_rx) = mpsc::unbounded_channel::<Job>();
        std::thread::spawn(move || {
            let loaded = load_model(&model_dir, &tokenizer_path, is_q4);
            let (mut metal, tokenizer, fmt) = match loaded {
                Ok(t) => t,
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            };
            let _ = ready.send(Ok(fmt));

            while let Some(job) = job_rx.blocking_recv() {
                metal.reset_state();
                let cb_tx = job.tx.clone();
                let out = metal.chat_completion_streaming(
                    &job.messages,
                    &tokenizer,
                    &job.cfg,
                    |delta, _id| {
                        // `send` fails once the client hangs up; returning false
                        // stops generation early instead of burning GPU on a dead
                        // connection.
                        cb_tx.send(Ev::Delta(delta.to_string())).is_ok()
                    },
                );
                let _ = job.tx.send(Ev::Done {
                    prompt_tokens: out.prompt_tokens,
                    completion_tokens: out.completion_tokens,
                });
            }
        });
        job_tx
    }

    fn load_model(
        model_dir: &std::path::Path,
        tokenizer_path: &std::path::Path,
        is_q4: bool,
    ) -> Result<(MetalQwen35State, BpeTokenizer, String), String> {
        let tokenizer = BpeTokenizer::from_tokenizer_json(tokenizer_path)
            .map_err(|e| format!("tokenizer load failed ({}): {e}", tokenizer_path.display()))?;

        if is_q4 {
            let cfg = if model_dir.join("config.json").exists() {
                Qwen35Config::from_config_json(&model_dir.join("config.json"))
                    .map_err(|e| format!("config.json parse failed: {e}"))?
            } else {
                Qwen35Config::qwen36_27b()
            };
            let metal = MetalQwen35State::from_q4_dir(model_dir, tokenizer_path, &cfg, 4096)
                .map_err(|e| format!("Q4 model load failed: {e}"))?;
            Ok((metal, tokenizer, "q4".to_string()))
        } else {
            let model = Qwen35Model::from_safetensors(model_dir)
                .map_err(|e| format!("safetensors load failed: {e}"))?;
            let cfg = model.config().clone();
            let metal = MetalQwen35State::new(model.weights(), &cfg, 4096)
                .map_err(|e| format!("Metal init failed: {e}"))?;
            Ok((metal, tokenizer, "bf16".to_string()))
        }
    }

    // ─── HTTP handlers ───────────────────────────────────────────────────────

    async fn health() -> &'static str {
        let t = Instant::now();
        emit_serve_event(
            "GET",
            "/health",
            200,
            None,
            t.elapsed().as_secs_f64() * 1000.0,
            false,
        );
        "ok"
    }

    async fn root() -> Json<Value> {
        let t = Instant::now();
        let body = json!({
            "name": "lattice",
            "object": "engine",
            "endpoints": ["/v1/chat/completions", "/v1/models", "/health"],
        });
        emit_serve_event(
            "GET",
            "/",
            200,
            None,
            t.elapsed().as_secs_f64() * 1000.0,
            false,
        );
        Json(body)
    }

    async fn list_models(State(s): State<AppState>) -> Json<Value> {
        let t = Instant::now();
        let body = json!({
            "object": "list",
            "data": [{
                "id": s.model_id.as_ref(),
                "object": "model",
                "created": unix_secs(),
                "owned_by": "lattice",
            }],
        });
        emit_serve_event(
            "GET",
            "/v1/models",
            200,
            None,
            t.elapsed().as_secs_f64() * 1000.0,
            false,
        );
        Json(body)
    }

    /// Phase machine for the SSE token stream.
    /// `Done` and `End` carry the completion token count so the terminal
    /// phase can include it in the telemetry event.
    enum Phase {
        Start,
        Body,
        Done(usize), // holds completion_tokens from worker
        End(usize),  // holds completion_tokens; emits telemetry then stream ends
    }

    async fn chat_completions(State(s): State<AppState>, Json(req): Json<ChatReq>) -> Response {
        let timer = Instant::now();
        if req.messages.is_empty() {
            emit_serve_event(
                "POST",
                "/v1/chat/completions",
                400,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
            );
            return err_response(StatusCode::BAD_REQUEST, "`messages` must not be empty");
        }

        let messages: Vec<ChatMessage> = req.messages.iter().map(to_chat_message).collect();
        let cfg = build_cfg(&req, &s.defaults);
        let model_id = req.model.clone().unwrap_or_else(|| s.model_id.to_string());
        let streaming = req.stream.unwrap_or(false);
        let id = format!("chatcmpl-{}", unix_nanos());
        let created = unix_secs();

        let (tx, mut rx) = mpsc::unbounded_channel::<Ev>();
        if s.jobs.send(Job { messages, cfg, tx }).is_err() {
            emit_serve_event(
                "POST",
                "/v1/chat/completions",
                500,
                None,
                timer.elapsed().as_secs_f64() * 1000.0,
                streaming,
            );
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference worker unavailable",
            );
        }

        if streaming {
            let stream = futures::stream::unfold((rx, Phase::Start), move |(mut rx, phase)| {
                let id = id.clone();
                let model = model_id.clone();
                async move {
                    match phase {
                        Phase::Start => {
                            let chunk = json!({
                                "id": id, "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
                            });
                            Some((
                                Ok::<Event, std::convert::Infallible>(
                                    Event::default().data(chunk.to_string()),
                                ),
                                (rx, Phase::Body),
                            ))
                        }
                        Phase::Body => match rx.recv().await {
                            Some(Ev::Delta(d)) => {
                                let chunk = json!({
                                    "id": id, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": null}],
                                });
                                Some((
                                    Ok(Event::default().data(chunk.to_string())),
                                    (rx, Phase::Body),
                                ))
                            }
                            Some(Ev::Done {
                                completion_tokens: ct,
                                ..
                            }) => {
                                let chunk = json!({
                                    "id": id, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                });
                                Some((
                                    Ok(Event::default().data(chunk.to_string())),
                                    (rx, Phase::Done(ct)),
                                ))
                            }
                            None => {
                                let chunk = json!({
                                    "id": id, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                });
                                Some((
                                    Ok(Event::default().data(chunk.to_string())),
                                    (rx, Phase::Done(0)),
                                ))
                            }
                        },
                        Phase::Done(ct) => {
                            Some((Ok(Event::default().data("[DONE]")), (rx, Phase::End(ct))))
                        }
                        Phase::End(ct) => {
                            emit_serve_event(
                                "POST",
                                "/v1/chat/completions",
                                200,
                                Some(ct),
                                timer.elapsed().as_secs_f64() * 1000.0,
                                true,
                            );
                            None
                        }
                    }
                }
            });
            Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        } else {
            let mut content = String::new();
            let mut prompt_tokens = 0usize;
            let mut completion_tokens = 0usize;
            while let Some(ev) = rx.recv().await {
                match ev {
                    Ev::Delta(d) => content.push_str(&d),
                    Ev::Done {
                        prompt_tokens: pt,
                        completion_tokens: ct,
                    } => {
                        prompt_tokens = pt;
                        completion_tokens = ct;
                    }
                }
            }
            let body = json!({
                "id": id, "object": "chat.completion",
                "created": created, "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            });
            emit_serve_event(
                "POST",
                "/v1/chat/completions",
                200,
                Some(completion_tokens),
                timer.elapsed().as_secs_f64() * 1000.0,
                false,
            );
            Json(body).into_response()
        }
    }

    fn err_response(code: StatusCode, msg: &str) -> Response {
        (
            code,
            Json(json!({"error": {"message": msg, "type": "invalid_request_error"}})),
        )
            .into_response()
    }

    /// Print a structured telemetry line to stdout for the app bridge to parse.
    fn emit_serve_event(
        method: &str,
        route: &str,
        status: u16,
        tokens: Option<usize>,
        dur_ms: f64,
        stream: bool,
    ) {
        println!(
            "@@lattice {}",
            json!({
                "ev": "http_request",
                "method": method,
                "route": route,
                "status": status,
                "tokens": tokens,
                "dur_ms": dur_ms,
                "stream": stream,
            })
        );
    }

    fn unix_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn unix_nanos() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }

    // ─── arg parsing + model resolution ──────────────────────────────────────

    fn parse_arg(args: &[String], flag: &str) -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    }

    fn default_model_cache() -> std::path::PathBuf {
        std::env::var("LATTICE_MODEL_CACHE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                std::path::PathBuf::from(home)
                    .join(".lattice")
                    .join("models")
            })
    }

    fn resolve_model_dir(arg: &str) -> std::path::PathBuf {
        if let Some(rest) = arg.strip_prefix("~/") {
            if let Ok(home) = std::env::var("HOME") {
                return std::path::PathBuf::from(home).join(rest);
            }
        }
        let p = std::path::PathBuf::from(arg);
        if p.is_absolute() {
            p
        } else if p.components().count() == 1 {
            default_model_cache().join(arg)
        } else {
            p
        }
    }

    fn detect_q4(dir: &std::path::Path) -> bool {
        !dir.join("model.safetensors").exists()
            && !dir.join("model.safetensors.index.json").exists()
            && std::fs::read_dir(dir)
                .ok()
                .and_then(|mut entries| {
                    entries.find(|e| {
                        e.as_ref()
                            .ok()
                            .and_then(|e| e.file_name().to_str().map(|n| n.ends_with(".q4")))
                            .unwrap_or(false)
                    })
                })
                .is_some()
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<String> = std::env::args().collect();

        let model_arg = parse_arg(&args, "--model")
            .or_else(|| std::env::var("LATTICE_SERVE_MODEL").ok())
            .ok_or("missing --model <name-or-path> (e.g. --model qwen3.5-0.8b)")?;
        let model_dir = resolve_model_dir(&model_arg);
        if !model_dir.exists() {
            return Err(format!("model directory not found: {}", model_dir.display()).into());
        }
        let is_q4 = detect_q4(&model_dir);
        let tokenizer_path = parse_arg(&args, "--tokenizer-dir")
            .map(|d| std::path::Path::new(&d).join("tokenizer.json"))
            .unwrap_or_else(|| model_dir.join("tokenizer.json"));

        let host = parse_arg(&args, "--host").unwrap_or_else(|| "127.0.0.1".to_string());
        let port: u16 = parse_arg(&args, "--port")
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                std::env::var("LATTICE_SERVE_PORT")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(11435);

        let defaults = Defaults {
            max_tokens: parse_arg(&args, "--max-tokens")
                .and_then(|s| s.parse().ok())
                .unwrap_or(512),
            temperature: parse_arg(&args, "--temperature")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7),
            top_k: parse_arg(&args, "--top-k")
                .and_then(|s| s.parse().ok())
                .unwrap_or(50),
            top_p: parse_arg(&args, "--top-p")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.9),
            repetition_penalty: parse_arg(&args, "--repetition-penalty")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.1),
            reasoning_budget: parse_arg(&args, "--reasoning-budget")
                .and_then(|s| s.parse().ok())
                .filter(|&n| n > 0),
        };

        eprintln!(
            "[lattice_serve] loading model from {} ({}) ...",
            model_dir.display(),
            if is_q4 { "q4" } else { "bf16" }
        );
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let jobs = spawn_worker(model_dir.clone(), tokenizer_path, is_q4, ready_tx);
        let fmt = match ready_rx.recv() {
            Ok(Ok(fmt)) => fmt,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Err("worker thread exited during model load".into()),
        };

        let model_id: Arc<str> = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("lattice")
            .into();
        eprintln!("[lattice_serve] model '{model_id}' ({fmt}) ready");

        let state = AppState {
            jobs,
            model_id,
            defaults,
        };

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(async move {
            let app = Router::new()
                .route("/", get(root))
                .route("/health", get(health))
                .route("/v1/models", get(list_models))
                .route("/v1/chat/completions", post(chat_completions))
                .with_state(state);
            let addr = format!("{host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .map_err(|e| format!("bind {addr} failed: {e}"))?;
            eprintln!("[lattice_serve] OpenAI-compatible API on http://{addr}/v1");
            eprintln!("[lattice_serve]   POST /v1/chat/completions   GET /v1/models   GET /health");
            println!("@@lattice {}", json!({"ev": "ready", "port": port}));
            axum::serve(listener, app)
                .await
                .map_err(|e| format!("serve error: {e}"))?;
            Ok::<(), String>(())
        })?;

        Ok(())
    }
}
