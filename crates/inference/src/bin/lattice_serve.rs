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
    use tokio::sync::{mpsc, watch};

    // ─── worker protocol ─────────────────────────────────────────────────────

    /// One token-stream event from the worker back to a request handler.
    enum Ev {
        Delta(String),
        Done {
            prompt_tokens: usize,
            completion_tokens: usize,
        },
        /// Generation failed closed instead of completing (#611: e.g. a
        /// grammar mask that blocks every candidate token, mirroring the
        /// CPU/`lattice.rs` fail-closed contract). Carries the underlying
        /// error message for server-side logging; the streaming and
        /// non-streaming handlers below decide separately how much (if any)
        /// of it is safe to surface to the HTTP client.
        Failed {
            message: String,
        },
    }

    /// A generation request handed to the single GPU worker thread.
    ///
    /// `cancel` reflects whether the client that submitted this job is still
    /// there. It starts `false` and flips to `true` the moment the matching
    /// [`CancelOnDrop`] guard is dropped -- i.e. the instant axum drops the
    /// response future/stream, which is exactly what happens on client
    /// disconnect (browser tab closed, `curl` killed, request future
    /// cancelled). The worker checks it (a) once at dequeue, before doing any
    /// work, and (b) independently of token emission, via
    /// `chat_completion_streaming_with_cancel`'s `should_cancel` predicate --
    /// before prefill, immediately after prefill returns, and at the top of
    /// every decode iteration -- so an abandoned job is skipped entirely,
    /// stopped before paying for prefill, or stopped within one decode step
    /// of the client leaving.
    struct Job {
        messages: Vec<ChatMessage>,
        cfg: GenerateConfig,
        tx: mpsc::UnboundedSender<Ev>,
        cancel: watch::Receiver<bool>,
    }

    /// Flips the paired `cancel` receiver to `true` when dropped. Held inside
    /// the per-request SSE stream state (streaming) or the handler's local
    /// scope (non-streaming) so it drops exactly when axum stops caring about
    /// the response — on client disconnect, or harmlessly after the request
    /// already finished normally (by then the worker has moved on anyway).
    struct CancelOnDrop(watch::Sender<bool>);

    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            let _ = self.0.send(true);
        }
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

    /// KV-cache length the worker allocates (see `load_model`). A request's
    /// `max_tokens` is clamped to this so an absurd value cannot drive
    /// `Vec::with_capacity(max_new_tokens)` in the Metal decode path to a
    /// capacity-overflow abort (which would kill the GPU worker thread, a
    /// persistent DoS). The prompt+completion-exceeds-window case is handled
    /// fail-closed inside the Metal generate path.
    const MODEL_MAX_CONTEXT: usize = 4096;

    fn build_cfg(req: &ChatReq, d: &Defaults) -> GenerateConfig {
        // Clamp like `max_tokens`: a budget past the KV window is meaningless and
        // would let a future `with_capacity(decode_cap(..))` abort on overflow.
        let reasoning_budget = req
            .reasoning_budget
            .filter(|&n| n > 0)
            .or(d.reasoning_budget)
            .map(|n| n.min(MODEL_MAX_CONTEXT));
        GenerateConfig {
            max_new_tokens: req
                .max_tokens
                .unwrap_or(d.max_tokens)
                .min(MODEL_MAX_CONTEXT),
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
        let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
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

            run_worker_loop(job_rx, move |messages, cfg, on_token, should_cancel| {
                metal.reset_state();
                metal
                    .chat_completion_streaming_with_cancel(
                        messages,
                        &tokenizer,
                        cfg,
                        on_token,
                        should_cancel,
                    )
                    .map(|out| (out.prompt_tokens, out.completion_tokens))
                    .map_err(|e| e.to_string())
            });
        });
        job_tx
    }

    /// Dequeue -> cancel-check -> generate -> reply, serialized on whatever
    /// thread calls this (the dedicated Metal worker thread in production).
    ///
    /// `generate` is injected so tests can swap in a fake, GPU-free generator
    /// while exercising the exact same queue/cancellation logic production
    /// uses. It must call `on_token` for each generated delta and stop as
    /// soon as `on_token` returns `false`; it must also poll `should_cancel`
    /// independently of `on_token` -- including during any phase that never
    /// calls `on_token` at all (a prefill-like section, or a run of
    /// empty-delta steps) -- and stop as soon as `should_cancel` returns
    /// `true`. Either way, return `Ok((prompt_tokens, completion_tokens))` for
    /// whatever was actually produced before stopping (early or at the cap),
    /// or `Err(message)` if generation itself failed closed (#611: e.g. a
    /// grammar mask that blocks every candidate token) rather than
    /// completing or being cancelled.
    fn run_worker_loop(
        mut job_rx: mpsc::UnboundedReceiver<Job>,
        mut generate: impl FnMut(
            &[ChatMessage],
            &GenerateConfig,
            &mut dyn FnMut(&str, u32) -> bool,
            &mut dyn FnMut() -> bool,
        ) -> Result<(usize, usize), String>,
    ) {
        while let Some(job) = job_rx.blocking_recv() {
            if *job.cancel.borrow() {
                // The client was already gone before we ever got to this job:
                // skip it entirely, no prefill, no decode, no reply.
                continue;
            }
            let cb_tx = job.tx.clone();
            let cancel_for_token = job.cancel.clone();
            let mut on_token = move |delta: &str, _id: u32| {
                if *cancel_for_token.borrow() {
                    return false;
                }
                // `send` also fails once the client hangs up; kept as a
                // second, independent check so a job whose cancellation
                // notification is somehow delayed still stops the instant
                // its reply channel is gone.
                cb_tx.send(Ev::Delta(delta.to_string())).is_ok()
            };
            // Separate from `on_token`: this is what reaches the generator's
            // prefill gap and its empty-delta decode iterations, neither of
            // which ever calls `on_token` (see
            // `MetalQwen35State::generate_streaming_with_cancel`).
            let cancel_for_predicate = job.cancel.clone();
            let mut should_cancel = move || *cancel_for_predicate.borrow();
            match generate(&job.messages, &job.cfg, &mut on_token, &mut should_cancel) {
                Ok((prompt_tokens, completion_tokens)) => {
                    let _ = job.tx.send(Ev::Done {
                        prompt_tokens,
                        completion_tokens,
                    });
                }
                Err(message) => {
                    eprintln!("generation error: {message}");
                    let _ = job.tx.send(Ev::Failed { message });
                }
            }
        }
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
            let metal =
                MetalQwen35State::from_q4_dir(model_dir, tokenizer_path, &cfg, MODEL_MAX_CONTEXT)
                    .map_err(|e| format!("Q4 model load failed: {e}"))?;
            Ok((metal, tokenizer, "q4".to_string()))
        } else {
            let model = Qwen35Model::from_safetensors(model_dir)
                .map_err(|e| format!("safetensors load failed: {e}"))?;
            let cfg = model.config().clone();
            let metal = MetalQwen35State::new(model.weights(), &cfg, MODEL_MAX_CONTEXT)
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
        let (cancel_tx, cancel_rx) = watch::channel(false);
        if s.jobs
            .send(Job {
                messages,
                cfg,
                tx,
                cancel: cancel_rx,
            })
            .is_err()
        {
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
        // Dropped when nobody cares about the response anymore: at the end of
        // this SSE stream (moved in below) or at the end of this function for
        // the non-streaming branch. Either way that's the client disconnect
        // signal the worker checks in `run_worker_loop`.
        let cancel_guard = CancelOnDrop(cancel_tx);

        if streaming {
            let stream = futures::stream::unfold(
                (rx, Phase::Start, cancel_guard),
                move |(mut rx, phase, cancel_guard)| {
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
                                    (rx, Phase::Body, cancel_guard),
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
                                        (rx, Phase::Body, cancel_guard),
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
                                        (rx, Phase::Done(ct), cancel_guard),
                                    ))
                                }
                                Some(Ev::Failed { message }) => {
                                    // The HTTP response was already committed as
                                    // 200 + text/event-stream when this SSE stream
                                    // started, so an error mid-stream cannot change
                                    // the status code. Mirror `lattice.rs`'s
                                    // `StreamMsg::Failed` contract: log the real
                                    // cause server-side (#611: e.g. a grammar mask
                                    // that blocks every candidate token) and give
                                    // the client a well-formed termination rather
                                    // than hanging the stream open.
                                    eprintln!("generation error (streaming): {message}");
                                    let chunk = json!({
                                        "id": id, "object": "chat.completion.chunk",
                                        "created": created, "model": model,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                    });
                                    Some((
                                        Ok(Event::default().data(chunk.to_string())),
                                        (rx, Phase::Done(0), cancel_guard),
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
                                        (rx, Phase::Done(0), cancel_guard),
                                    ))
                                }
                            },
                            Phase::Done(ct) => Some((
                                Ok(Event::default().data("[DONE]")),
                                (rx, Phase::End(ct), cancel_guard),
                            )),
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
                },
            );
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
                    Ev::Failed { message } => {
                        // Unlike streaming, the response has not been committed
                        // yet, so a generation failure (#611: e.g. a grammar mask
                        // that blocks every candidate token) can still surface as
                        // a real HTTP error instead of a disguised 200 -- the
                        // same "generic 500, specific detail logged server-side"
                        // contract the CPU/Metal handlers in `lattice.rs` use.
                        eprintln!("generation error: {message}");
                        emit_serve_event(
                            "POST",
                            "/v1/chat/completions",
                            500,
                            None,
                            timer.elapsed().as_secs_f64() * 1000.0,
                            false,
                        );
                        return err_response(StatusCode::INTERNAL_SERVER_ERROR, "inference failed");
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
        if let Some(rest) = arg.strip_prefix("~/")
            && let Ok(home) = std::env::var("HOME")
        {
            return std::path::PathBuf::from(home).join(rest);
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

    // ─── tests ───────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::Duration;

        /// A GPU-free stand-in for `MetalQwen35State::chat_completion_streaming_with_cancel`:
        /// "generates" up to `cap` fake tokens, sleeping briefly between each so
        /// a cancelled job has many opportunities to be observed running past
        /// where it should have stopped. Counts how many times it was entered
        /// (`started`) and how many fake tokens actually ran (`ran_tokens`), so
        /// tests can assert a cancelled queued job's generator was never called
        /// at all. Checks `should_cancel` at the top of each iteration in
        /// addition to `on_token`'s own check, mirroring the production
        /// contract; existing tests here only rely on the `on_token` path, so
        /// this addition does not change their outcomes.
        #[allow(clippy::type_complexity)]
        fn fake_generate(
            cap: usize,
            started: Arc<AtomicUsize>,
            ran_tokens: Arc<AtomicUsize>,
        ) -> impl FnMut(
            &[ChatMessage],
            &GenerateConfig,
            &mut dyn FnMut(&str, u32) -> bool,
            &mut dyn FnMut() -> bool,
        ) -> Result<(usize, usize), String> {
            move |_messages, _cfg, on_token, should_cancel| {
                started.fetch_add(1, Ordering::SeqCst);
                let mut n = 0usize;
                for i in 0..cap {
                    std::thread::sleep(Duration::from_millis(5));
                    if should_cancel() {
                        break;
                    }
                    if !on_token("x", i as u32) {
                        break;
                    }
                    n += 1;
                    ran_tokens.fetch_add(1, Ordering::SeqCst);
                }
                Ok((1, n))
            }
        }

        /// A GPU-free fake with an explicit prefill-like phase *before* any
        /// `on_token` call -- mirroring the real gap this fix closes:
        /// production prefill has no callback point at all, so only
        /// `should_cancel` (never `on_token`) can observe a disconnect that
        /// happens during it. `entered_decode` flips only if the prefill-like
        /// phase runs to completion uncancelled, so tests can assert it never
        /// does.
        #[allow(clippy::type_complexity)]
        fn fake_generate_with_prefill_gap(
            prefill_steps: usize,
            decode_cap: usize,
            entered_decode: Arc<AtomicBool>,
        ) -> impl FnMut(
            &[ChatMessage],
            &GenerateConfig,
            &mut dyn FnMut(&str, u32) -> bool,
            &mut dyn FnMut() -> bool,
        ) -> Result<(usize, usize), String> {
            move |_messages, _cfg, on_token, should_cancel| {
                for _ in 0..prefill_steps {
                    std::thread::sleep(Duration::from_millis(5));
                    if should_cancel() {
                        return Ok((1, 0));
                    }
                }
                entered_decode.store(true, Ordering::SeqCst);
                let mut n = 0usize;
                for i in 0..decode_cap {
                    std::thread::sleep(Duration::from_millis(5));
                    if should_cancel() {
                        break;
                    }
                    if !on_token("x", i as u32) {
                        break;
                    }
                    n += 1;
                }
                Ok((1, n))
            }
        }

        /// Builds a `Job` plus the receiver its worker replies on and the guard
        /// that cancels it when dropped (the same guard `chat_completions`
        /// moves into the SSE stream / keeps local for non-streaming, standing
        /// in here for "the client is still connected").
        fn make_job() -> (Job, mpsc::UnboundedReceiver<Ev>, CancelOnDrop) {
            let (tx, rx) = mpsc::unbounded_channel::<Ev>();
            let (cancel_tx, cancel_rx) = watch::channel(false);
            let job = Job {
                messages: vec![ChatMessage::user("hi")],
                cfg: GenerateConfig::default(),
                tx,
                cancel: cancel_rx,
            };
            (job, rx, CancelOnDrop(cancel_tx))
        }

        #[test]
        fn queued_job_cancelled_before_dequeue_is_skipped_entirely() {
            let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
            let started = Arc::new(AtomicUsize::new(0));
            let ran_tokens = Arc::new(AtomicUsize::new(0));

            // Job 1 occupies the worker (50 fake tokens, 5ms apart = ~250ms)
            // long enough that job 2 is still sitting in the queue, untouched,
            // when we cancel it a few lines down.
            let (job1, rx1, _guard1) = make_job();
            job_tx.send(job1).unwrap();

            // Job 2: cancelled client-side (guard dropped) immediately, while
            // it is still queued behind job 1.
            let (job2, mut rx2, guard2) = make_job();
            job_tx.send(job2).unwrap();
            drop(guard2);

            // Job 3: submitted after the cancelled one, to prove the worker
            // moves on and keeps serving correctly afterward.
            let (job3, rx3, _guard3) = make_job();
            job_tx.send(job3).unwrap();
            drop(job_tx);

            let started2 = started.clone();
            let ran2 = ran_tokens.clone();
            let handle = std::thread::spawn(move || {
                run_worker_loop(job_rx, fake_generate(50, started2, ran2))
            });

            let completion_tokens_of = |mut rx: mpsc::UnboundedReceiver<Ev>| -> Option<usize> {
                let mut ct = None;
                while let Some(ev) = rx.blocking_recv() {
                    if let Ev::Done {
                        completion_tokens, ..
                    } = ev
                    {
                        ct = Some(completion_tokens);
                    }
                }
                ct
            };

            assert_eq!(
                completion_tokens_of(rx1),
                Some(50),
                "job 1 should run to completion undisturbed"
            );

            // Job 2 must produce NOTHING: no Delta, no Done -- the worker
            // `continue`d past it without ever touching `generate`, so its
            // `tx` is simply dropped with the rest of the `Job`.
            assert!(
                rx2.blocking_recv().is_none(),
                "cancelled queued job must be skipped entirely: no events at all"
            );

            assert_eq!(
                completion_tokens_of(rx3),
                Some(50),
                "worker must survive cancelling job 2 and serve job 3 normally afterward"
            );

            handle.join().expect("worker thread must not panic");

            assert_eq!(
                started.load(Ordering::SeqCst),
                2,
                "generate() must run exactly twice (job 1, job 3) -- never for cancelled job 2"
            );
            assert_eq!(
                ran_tokens.load(Ordering::SeqCst),
                100,
                "50 real fake-tokens each for job 1 and job 3, zero for cancelled job 2"
            );
        }

        #[test]
        fn running_job_cancelled_midstream_stops_early_and_worker_survives() {
            let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
            let started = Arc::new(AtomicUsize::new(0));
            let ran_tokens = Arc::new(AtomicUsize::new(0));

            // Job 1: a long fake generation (2000 tokens, 5ms apart) that we
            // cancel partway through -- it must stop well short of the cap.
            let (job1, mut rx1, guard1) = make_job();
            job_tx.send(job1).unwrap();
            // `Option` so it can be moved-out-and-dropped at most once from
            // inside the loop below; the borrow checker cannot see that the
            // `seen == 5` runtime condition only ever holds on one iteration,
            // so a bare `drop(guard1)` there is rejected as a repeated move.
            let mut guard1 = Some(guard1);

            let (job2, mut rx2, _guard2) = make_job();
            job_tx.send(job2).unwrap();
            drop(job_tx);

            let started2 = started.clone();
            let ran2 = ran_tokens.clone();
            let handle = std::thread::spawn(move || {
                run_worker_loop(job_rx, fake_generate(2000, started2, ran2))
            });

            let mut seen = 0;
            loop {
                match rx1.blocking_recv() {
                    Some(Ev::Delta(_)) => {
                        seen += 1;
                        if seen == 5 {
                            // "Client disconnects" mid-stream.
                            guard1.take();
                        }
                    }
                    Some(Ev::Done {
                        completion_tokens, ..
                    }) => {
                        assert!(
                            completion_tokens < 2000,
                            "job 1 must stop well short of its 2000-token cap after \
                             cancellation, got {completion_tokens}"
                        );
                        assert!(
                            completion_tokens < 100,
                            "job 1 must stop within a handful of tokens of the client \
                             disconnecting, not run on regardless; got {completion_tokens}"
                        );
                        break;
                    }
                    Some(Ev::Failed { message }) => {
                        panic!("fake_generate never fails; unexpected Ev::Failed: {message}")
                    }
                    None => panic!("job 1's reply channel closed before a Done event"),
                }
            }

            // Job 2 must still complete in full: the worker thread did not
            // panic or wedge when job 1 was cancelled mid-generation.
            let mut n2 = None;
            while let Some(ev) = rx2.blocking_recv() {
                if let Ev::Done {
                    completion_tokens, ..
                } = ev
                {
                    n2 = Some(completion_tokens);
                }
            }
            assert_eq!(
                n2,
                Some(2000),
                "worker must survive mid-stream cancellation and serve the next job to completion"
            );

            handle.join().expect("worker thread must not panic");
        }

        /// Codex review of PR #606: cancellation was only observed through the
        /// `on_token` callback, so a generator phase that never calls it -- the
        /// real prefill pass has no callback point at all -- could run
        /// unbounded after the client already disconnected. This proves
        /// `run_worker_loop` threads an independent `should_cancel` signal
        /// through to `generate` and that a fake generator honoring only that
        /// signal (never `on_token`) still gets stopped promptly, well short
        /// of its prefill-like phase's natural end.
        #[test]
        fn running_job_cancelled_during_prefill_like_phase_never_calls_on_token() {
            let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
            let entered_decode = Arc::new(AtomicBool::new(false));

            let (job1, mut rx1, guard1) = make_job();
            job_tx.send(job1).unwrap();
            drop(job_tx);

            // 400 * 5ms = up to 2s of "prefill" if never cancelled -- the test
            // cancels at 20ms in, ~100x margin, so reaching Done quickly is
            // only possible if should_cancel actually stopped it early.
            let entered2 = entered_decode.clone();
            let handle = std::thread::spawn(move || {
                run_worker_loop(job_rx, fake_generate_with_prefill_gap(400, 50, entered2))
            });

            std::thread::sleep(Duration::from_millis(20));
            drop(guard1);

            match rx1.blocking_recv() {
                Some(Ev::Delta(_)) => panic!(
                    "on_token must never be called: cancellation happened while the \
                     fake generator was still in its prefill-like phase, which does \
                     not call on_token at all"
                ),
                Some(Ev::Done {
                    completion_tokens, ..
                }) => {
                    assert_eq!(
                        completion_tokens, 0,
                        "job cancelled during the prefill-like phase must produce \
                         zero tokens, got {completion_tokens}"
                    );
                }
                Some(Ev::Failed { message }) => panic!(
                    "fake_generate_with_prefill_gap never fails; unexpected Ev::Failed: {message}"
                ),
                None => panic!("job 1's reply channel closed before a Done event"),
            }

            handle.join().expect("worker thread must not panic");

            assert!(
                !entered_decode.load(Ordering::SeqCst),
                "should_cancel alone (on_token is never called during this phase) \
                 must stop the job before the decode phase is ever reached -- this \
                 is the exact blind spot from the PR #606 review, where production \
                 prefill has no on_token callback point and so could run to \
                 completion after the client already disconnected"
            );
        }

        /// A GPU-free fake that fails closed on its first call (mirroring the
        /// #611 contract: `chat_completion_streaming_with_cancel` now returns
        /// `Err` instead of silently sampling when a grammar mask blocks every
        /// candidate token) and succeeds normally on every call after that, so
        /// a single test can prove both halves of the contract: the failure is
        /// reported honestly, and the worker thread survives it.
        #[allow(clippy::type_complexity)]
        fn fake_generate_fails_once_then_succeeds(
            message: &'static str,
            call_count: Arc<AtomicUsize>,
        ) -> impl FnMut(
            &[ChatMessage],
            &GenerateConfig,
            &mut dyn FnMut(&str, u32) -> bool,
            &mut dyn FnMut() -> bool,
        ) -> Result<(usize, usize), String> {
            move |_messages, _cfg, on_token, _should_cancel| {
                if call_count.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(message.to_string());
                }
                let _ = on_token("x", 0);
                Ok((1, 1))
            }
        }

        /// #611: a generation failure must reach the HTTP layer as
        /// `Ev::Failed` carrying the real error, never as `Ev::Done` with a
        /// fabricated token count -- the latter is exactly the fail-open shape
        /// this issue closes (a blocked grammar silently reported as a normal
        /// zero/short completion instead of an error). The worker thread must
        /// also survive the failure and keep serving subsequent jobs, exactly
        /// as it survives a cancelled job in the tests above.
        #[test]
        fn generation_failure_is_reported_as_ev_failed_not_ev_done() {
            let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();

            let (job1, mut rx1, _guard1) = make_job();
            job_tx.send(job1).unwrap();
            let (job2, mut rx2, _guard2) = make_job();
            job_tx.send(job2).unwrap();
            drop(job_tx);

            let call_count = Arc::new(AtomicUsize::new(0));
            let handle = std::thread::spawn({
                let call_count = call_count.clone();
                move || {
                    run_worker_loop(
                        job_rx,
                        fake_generate_fails_once_then_succeeds(
                            "grammar constraint blocked every token; no legal \
                             continuation exists in the current grammar state",
                            call_count,
                        ),
                    )
                }
            });

            match rx1.blocking_recv() {
                Some(Ev::Failed { message }) => {
                    assert!(
                        message.contains("grammar constraint blocked every token"),
                        "Ev::Failed must carry the underlying error message, got: {message}"
                    );
                }
                Some(Ev::Done { .. }) => panic!(
                    "a failed generation must never be reported as Ev::Done -- that \
                     would silently hand the HTTP layer a fabricated token count for \
                     a request that produced no legal output, which is the #611 \
                     fail-open failure mode this test guards against"
                ),
                Some(Ev::Delta(_)) => {
                    panic!("a generator that fails on its first call must never emit a Delta first")
                }
                None => panic!("job 1's reply channel closed with no event at all"),
            }

            let mut done = None;
            while let Some(ev) = rx2.blocking_recv() {
                if let Ev::Done {
                    completion_tokens, ..
                } = ev
                {
                    done = Some(completion_tokens);
                }
            }
            assert_eq!(
                done,
                Some(1),
                "worker thread must survive a failed generation and serve the next \
                 job normally afterward"
            );

            handle
                .join()
                .expect("worker thread must not panic on a generation error");
        }
    }
}
