//! Chat request preparation and first-token cost, per serving route.
//!
//! Drives the functions the `/v1/chat/completions` handlers call, in process,
//! from the request body to the first generated token, and times the
//! preparation separately from the first token. It renders, templates,
//! tokenizes, validates and admits nothing itself: every one of those steps
//! is a call into `lattice_inference::serve`, so the same source measures both
//! sides of a change to the preparation path.
//!
//! Routes, selected with `BENCH_ROUTE`:
//!   cpu            `lattice serve` on a safetensors checkpoint.
//!                  Handler: `serve::prepare::prepare_chat_request` (render,
//!                  tokenize, context-window check), then the handler's
//!                  `GenerateConfig` mapping. First token:
//!                  `Qwen35Model::generate_streaming_with_cancel`. This route
//!                  has no admission gate.
//!   lattice-metal  `lattice serve` on a Q4 checkpoint (Metal worker).
//!                  Handler: the same `prepare_chat_request` and mapping as
//!                  `cpu`, with the 4096-token context the binary uses.
//!   lattice-serve  `lattice_serve` (Metal worker, Q4 or safetensors).
//!                  Handler: `serve::contract::normalize_request` with the
//!                  `lattice_serve` profile, `serve::prepare::build_cfg`, then
//!                  `serve::into_engine_chat_messages`. No render or tokenize
//!                  happens in this handler.
//!   gemma-cpu      Gemma 4 E2B text on a safetensors checkpoint, CPU.
//!                  Preparation: `serve::prepare::prepare_gemma_chat_request`
//!                  (validate, the Gemma prompt adapter's defaults, render
//!                  with the checkpoint's chat template, tokenize, context
//!                  check) plus the prompt ids. First token:
//!                  `Gemma4Model::generate_streaming_with_cancel`. No serving
//!                  binary routes Gemma yet, so this route measures the
//!                  preparation entry the servers will call.
//!
//! The two Metal routes share one preparation path after the handler: both
//! submit to `MetalWorkerClient::submit_with_lora`, whose admission gate
//! decides whether the job exists, and the worker then renders
//! (`format_chat_template`), tokenizes and applies `check_prompt_fits_window`
//! before prefill. They differ only in the handler-side step above and in the
//! window policy the worker is spawned with, so `lattice-serve` measures that
//! handler difference as its own preparation case.
//!
//! Timed regions, per measured run:
//!   prepare_ms         request JSON -> validated request -> handler output
//!                      (for `cpu`/`lattice-metal`/`gemma-cpu` this includes
//!                      the render, tokenize and context-window check).
//!   worker_prepare_ms  Metal routes: the worker's render, tokenize and
//!                      window check, called here on the same functions.
//!   admit_ms           Metal routes: `submit_with_lora`, the admission
//!                      decision plus enqueue.
//!   first_token_ms     `cpu`/`gemma-cpu`: the generate call up to its first streamed
//!                      delta. Metal routes: from submit to the first
//!                      `WorkerEvent::Delta`, which also contains the worker's
//!                      own render, tokenize and window check.
//! Generation is cancelled after the first delta. The first delta is the
//! first streamed text; a decoder that withholds text (a partial UTF-8
//! sequence, a pending stop string) reports it later than the first sampled
//! id.
//!
//! Cases: `control` (the canonical single-user-turn request), `multi_turn`
//! (system plus a three-turn history), `reasoning` (an explicit
//! `reasoning_budget`), and `unsupported_role` (a `tool` message, which must be
//! refused with the contract's exact error and is not timed). Every positive
//! request streams. Each run prefixes its first message with the run number so
//! the Metal worker's cross-turn prefix cache cannot reuse the previous run's
//! prompt. `gemma-cpu` runs `control` and `multi_turn`; Gemma has no
//! reasoning-budget mode, so its `reasoning` case, like `unsupported_role` and
//! `unsupported_modality` (an image part on a text-only model), must be refused
//! with the contract's error code and is not timed.
//!
//! Not covered: `lattice_serve`'s raw-body pre-checks (duplicate JSON members,
//! content-part limits) and its structured-output admission are private to that
//! binary; both routes here parse with `serde_json::from_slice`, as the
//! `lattice serve` handler does.
//!
//! Env:
//!   BENCH_ROUTE        cpu | lattice-metal | lattice-serve | gemma-cpu (required)
//!   LATTICE_MODEL_DIR  checkpoint directory (default ~/.lattice/models/qwen3.5-0.8b,
//!                      or ~/.lattice/models/gemma-4-e2b-it for gemma-cpu)
//!   BENCH_RUNS         measured runs per case, after one untimed warmup (default 5)
//!
//! Output:
//!   ROUTE route=<r> binary=<lattice|lattice_serve> backend=<cpu|metal>
//!   LOAD route=<r> load_ms=<f>
//!   RESULT route=<r> case=<c> run=<n> prompt_tokens=<n> prepare_ms=<f>
//!     worker_prepare_ms=<f|na> admit_ms=<f|na> first_token_ms=<f>
//!   RESULT route=<r> case=unsupported_role refused=1 code=<code>
//!   RESULT route=gemma-cpu case=<reasoning|unsupported_modality> refused=1 code=<code>
//!   SKIP route=<r> reason=<...>   (exit status 2: nothing was measured)
//!
//! The Metal routes need `--features f16,metal-gpu,serve,bench-internals` and
//! take the machine-wide GPU lock themselves, so run them inside a held bench
//! window rather than under `scripts/bench-command.sh`.
#![allow(clippy::field_reassign_with_default)]

use lattice_inference::GenerateConfig;
use lattice_inference::serve::ApiError;
use lattice_inference::serve::contract::{
    ChatRequest, GenerationDefaults, ServeProfile, normalize_request,
};
use lattice_inference::serve::into_engine_chat_messages;
use lattice_inference::serve::prepare::{
    GemmaPromptAdapter, PreparedChatRequest, PreparedGemmaChatRequest, build_cfg, lattice_gen_cfg,
    prepare_chat_request, prepare_gemma_chat_request,
};
use serde_json::json;

/// Engine messages plus generation config: what a Metal-route handler submits.
type EngineRequest = (
    Vec<lattice_inference::forward::metal_qwen35::ChatMessage>,
    GenerateConfig,
);

/// Items only the Metal routes use are dead code in a build without them.
macro_rules! metal_only {
    ($item:item) => {
        #[cfg_attr(
            not(all(
                target_os = "macos",
                feature = "metal-gpu",
                feature = "bench-internals"
            )),
            allow(dead_code)
        )]
        $item
    };
}

const MODEL_ID: &str = "bench-model";
/// `lattice serve --max-tokens` default.
const LATTICE_DEFAULT_MAX_TOKENS: usize = 256;
/// `lattice serve`'s `max_tokens_cap`.
const LATTICE_MAX_TOKENS_CAP: usize = 4096;
metal_only! {
    /// The Metal context `lattice serve` allocates and checks against.
    const LATTICE_METAL_MAX_CONTEXT: usize = 4096;
}
/// `lattice_serve`'s default generation budget.
const LATTICE_SERVE_DEFAULT_MAX_TOKENS: usize = 512;
const REFUSED_MESSAGE: &str = "role 'tool' is not supported by this server";
const REFUSED_CODE: &str = "unsupported_feature";
const VISION_REFUSED_CODE: &str = "vision_unsupported";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Cpu,
    LatticeMetal,
    LatticeServe,
    GemmaCpu,
}

impl Route {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "cpu" => Ok(Self::Cpu),
            "lattice-metal" => Ok(Self::LatticeMetal),
            "lattice-serve" => Ok(Self::LatticeServe),
            "gemma-cpu" => Ok(Self::GemmaCpu),
            other => Err(format!(
                "BENCH_ROUTE={other:?} must be cpu, lattice-metal, lattice-serve or gemma-cpu"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::LatticeMetal => "lattice-metal",
            Self::LatticeServe => "lattice-serve",
            Self::GemmaCpu => "gemma-cpu",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    Control,
    MultiTurn,
    Reasoning,
}

const POSITIVE_CASES: [Case; 3] = [Case::Control, Case::MultiTurn, Case::Reasoning];
const GEMMA_POSITIVE_CASES: [Case; 2] = [Case::Control, Case::MultiTurn];

impl Case {
    fn name(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::MultiTurn => "multi_turn",
            Self::Reasoning => "reasoning",
        }
    }

    fn body(self, run: usize) -> Vec<u8> {
        let tag = format!("Request {run}. ");
        let body = match self {
            Self::Control => json!({
                "model": MODEL_ID,
                "stream": true,
                "messages": [
                    {"role": "user", "content": format!("{tag}What is the capital of France?")}
                ],
            }),
            Self::MultiTurn => json!({
                "model": MODEL_ID,
                "stream": true,
                "messages": [
                    {"role": "system", "content": format!("{tag}You are a concise assistant.")},
                    {"role": "user", "content": "Name a prime number between 10 and 20."},
                    {"role": "assistant", "content": "13 is a prime number between 10 and 20."},
                    {"role": "user", "content": "Name another one and explain why it is prime."}
                ],
            }),
            Self::Reasoning => json!({
                "model": MODEL_ID,
                "stream": true,
                "reasoning_budget": 64,
                "messages": [
                    {"role": "user", "content": format!("{tag}Is 91 a prime number?")}
                ],
            }),
        };
        body.to_string().into_bytes()
    }
}

fn unsupported_role_body() -> Vec<u8> {
    json!({
        "model": MODEL_ID,
        "stream": true,
        "messages": [
            {"role": "tool", "content": "{\"result\": 42}"},
            {"role": "user", "content": "What was the result?"}
        ],
    })
    .to_string()
    .into_bytes()
}

fn unsupported_modality_body() -> Vec<u8> {
    json!({
        "model": MODEL_ID,
        "stream": true,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}
        ],
    })
    .to_string()
    .into_bytes()
}

fn parse_body(body: &[u8]) -> Result<ChatRequest, String> {
    serde_json::from_slice::<ChatRequest>(body).map_err(|e| format!("request body: {e}"))
}

/// The `lattice serve` handler's preparation, both backends.
fn lattice_prepare(
    body: &[u8],
    tokenize_len: impl FnOnce(&str) -> usize,
    max_context: usize,
) -> Result<Result<PreparedChatRequest, ApiError>, String> {
    let req = parse_body(body)?;
    Ok(prepare_chat_request(
        &req,
        MODEL_ID,
        LATTICE_DEFAULT_MAX_TOKENS,
        LATTICE_MAX_TOKENS_CAP,
        false,
        tokenize_len,
        || max_context,
    ))
}

metal_only! {
    /// The `lattice_serve` handler's preparation: normalize, `build_cfg`, engine
    /// messages.
    fn lattice_serve_prepare(
        body: &[u8],
        model_max_context: usize,
    ) -> Result<Result<EngineRequest, ApiError>, String> {
        let req = parse_body(body)?;
        Ok(normalize_request(
            &req,
            GenerationDefaults::standard(LATTICE_SERVE_DEFAULT_MAX_TOKENS),
            ServeProfile::lattice_serve(MODEL_ID, model_max_context).with_vision_support(false),
        )
        .and_then(|validated| {
            let cfg = build_cfg(&validated);
            into_engine_chat_messages(validated.messages).map(|messages| (messages, cfg))
        }))
    }
}

/// Gemma E2B text preparation, with the `lattice serve` defaults.
fn gemma_prepare(
    adapter: &GemmaPromptAdapter,
    body: &[u8],
    tokenize_len: impl FnOnce(&str) -> usize,
    max_context: usize,
) -> Result<Result<PreparedGemmaChatRequest, ApiError>, String> {
    let req = parse_body(body)?;
    Ok(prepare_gemma_chat_request(
        adapter,
        &req,
        MODEL_ID,
        LATTICE_DEFAULT_MAX_TOKENS,
        LATTICE_MAX_TOKENS_CAP,
        tokenize_len,
        || max_context,
    ))
}

/// Requires a refusal carrying `expected` as its code.
fn check_refusal_code<T>(
    case: &str,
    outcome: Result<T, ApiError>,
    expected: &'static str,
) -> Result<&'static str, String> {
    match outcome {
        Ok(_) => Err(format!("{case}: the request was accepted")),
        Err(ApiError::BadRequest { code, .. }) if code == expected => Ok(code),
        Err(other) => Err(format!(
            "{case}: expected BadRequest {expected:?}, got {other:?}"
        )),
    }
}

/// Requires the exact refusal the contract documents for a `tool` message.
fn check_refusal<T>(outcome: Result<T, ApiError>) -> Result<&'static str, String> {
    match outcome {
        Ok(_) => Err("unsupported_role: the request was accepted".into()),
        Err(ApiError::BadRequest { message, code })
            if message == REFUSED_MESSAGE && code == REFUSED_CODE =>
        {
            Ok(code)
        }
        Err(other) => Err(format!(
            "unsupported_role: expected BadRequest {REFUSED_CODE:?} {REFUSED_MESSAGE:?}, got {other:?}"
        )),
    }
}

fn ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn skip(route: Route, reason: &str) -> ! {
    println!("SKIP route={} reason={reason}", route.name());
    std::process::exit(2);
}

fn main() {
    let route = match std::env::var("BENCH_ROUTE") {
        Ok(raw) => match Route::parse(raw.trim()) {
            Ok(route) => route,
            Err(e) => {
                eprintln!("bench_serve_prepare: {e}");
                std::process::exit(1);
            }
        },
        Err(_) => {
            eprintln!(
                "bench_serve_prepare: set BENCH_ROUTE to cpu, lattice-metal or lattice-serve"
            );
            std::process::exit(1);
        }
    };
    if let Err(e) = run(route) {
        eprintln!("bench_serve_prepare failed: {e}");
        std::process::exit(1);
    }
}

fn run(route: Route) -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::model_format::{ModelFormat, detect_format};

    let runs: usize = match std::env::var("BENCH_RUNS") {
        Ok(raw) => raw
            .trim()
            .parse()
            .map_err(|_| format!("BENCH_RUNS={raw:?} is not a non-negative integer"))?,
        Err(_) => 5,
    };
    let home = std::env::var("HOME")?;
    let default_model = match route {
        Route::GemmaCpu => "gemma-4-e2b-it",
        _ => "qwen3.5-0.8b",
    };
    let model_dir = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/{default_model}"));
    let dir = std::path::PathBuf::from(&model_dir);
    if route == Route::GemmaCpu {
        let missing: Vec<&str> = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ]
        .into_iter()
        .filter(|name| !dir.join(name).is_file())
        .collect();
        if !missing.is_empty() {
            skip(
                route,
                &format!(
                    "checkpoint_absent model_dir={model_dir} missing={}",
                    missing.join(",")
                ),
            );
        }
        return run_gemma_cpu(&dir, runs);
    }
    let tokenizer_path = dir.join("tokenizer.json");
    let format = detect_format(&dir);
    let format_ok = match route {
        Route::Cpu => matches!(format, ModelFormat::Safetensors),
        Route::LatticeMetal => matches!(format, ModelFormat::Q4),
        Route::LatticeServe => matches!(format, ModelFormat::Q4 | ModelFormat::Safetensors),
        Route::GemmaCpu => false,
    };
    if !format_ok || !tokenizer_path.is_file() {
        skip(
            route,
            &format!("checkpoint_absent model_dir={model_dir} format={format:?}"),
        );
    }
    match route {
        Route::Cpu => run_cpu(&dir, runs),
        Route::LatticeMetal | Route::LatticeServe => run_metal(route, &dir, format, runs),
        Route::GemmaCpu => Err("gemma-cpu is dispatched before format detection".into()),
    }
}

fn run_gemma_cpu(dir: &std::path::Path, runs: usize) -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::Tokenizer as _;
    use lattice_inference::model::gemma4_config::Gemma4Config;
    use lattice_inference::model::gemma4_model::Gemma4Model;
    use lattice_inference::tokenizer::GemmaBpeTokenizer;
    use std::time::Instant;

    let route = Route::GemmaCpu;
    println!("ROUTE route={} binary=lattice backend=cpu", route.name());
    let max_context = Gemma4Config::from_model_dir(dir)?.max_position_embeddings;
    // Truncation at the context window, never below it, so the context check
    // sees the whole prompt. `tokenize_batch` pads to the longest input rather
    // than to `max_seq_len`, so a one-prompt batch carries no padding.
    let tokenizer = GemmaBpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))?
        .with_max_seq_len(max_context);
    let adapter = GemmaPromptAdapter::from_model_dir(dir, &tokenizer)?;
    let load = Instant::now();
    let model = Gemma4Model::from_safetensors(dir)?;
    println!("LOAD route={} load_ms={:.3}", route.name(), ms(load));

    for case in GEMMA_POSITIVE_CASES {
        for run in 0..=runs {
            let body = case.body(run);
            let start = Instant::now();
            let mut tokenized = None;
            let prepared = gemma_prepare(
                &adapter,
                &body,
                |p| {
                    let batch = tokenizer.tokenize_batch(&[p]);
                    let len = batch.first().map_or(0, |t| t.pre_truncation_len);
                    tokenized = batch.into_iter().next();
                    len
                },
                max_context,
            )?
            .map_err(|e| format!("{}: {e:?}", case.name()))?;
            let tokenized =
                tokenized.ok_or_else(|| format!("{}: prompt not tokenized", case.name()))?;
            if tokenized.real_length != tokenized.pre_truncation_len {
                return Err(format!("{}: prompt was truncated", case.name()).into());
            }
            let prompt_ids = &tokenized.input_ids[..tokenized.real_length];
            let prepare_ms = ms(start);
            let prompt_tokens = prompt_ids.len();

            let start = Instant::now();
            let mut first_token_ms = None;
            model.generate_streaming_with_cancel(
                prompt_ids,
                &prepared.gen_cfg,
                |_| {
                    first_token_ms = Some(ms(start));
                    false
                },
                || false,
            )?;
            let first_token_ms = first_token_ms
                .ok_or_else(|| format!("{}: generation produced no token", case.name()))?;
            if run == 0 {
                continue;
            }
            println!(
                "RESULT route={} case={} run={run} prompt_tokens={prompt_tokens} \
                 prepare_ms={prepare_ms:.3} worker_prepare_ms=na admit_ms=na \
                 first_token_ms={first_token_ms:.3}",
                route.name(),
                case.name(),
            );
        }
    }
    let token_len = |p: &str| {
        tokenizer
            .tokenize_batch(&[p])
            .first()
            .map_or(0, |t| t.pre_truncation_len)
    };
    let code = check_refusal(gemma_prepare(
        &adapter,
        &unsupported_role_body(),
        token_len,
        max_context,
    )?)?;
    println!(
        "RESULT route={} case=unsupported_role refused=1 code={code}",
        route.name()
    );
    let code = check_refusal_code(
        Case::Reasoning.name(),
        gemma_prepare(&adapter, &Case::Reasoning.body(1), token_len, max_context)?,
        REFUSED_CODE,
    )?;
    println!(
        "RESULT route={} case={} refused=1 code={code}",
        route.name(),
        Case::Reasoning.name()
    );
    let code = check_refusal_code(
        "unsupported_modality",
        gemma_prepare(
            &adapter,
            &unsupported_modality_body(),
            token_len,
            max_context,
        )?,
        VISION_REFUSED_CODE,
    )?;
    println!(
        "RESULT route={} case=unsupported_modality refused=1 code={code}",
        route.name()
    );
    Ok(())
}

fn run_cpu(dir: &std::path::Path, runs: usize) -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::Tokenizer as _;
    use lattice_inference::model::qwen35::Qwen35Model;
    use std::time::Instant;

    let route = Route::Cpu;
    println!("ROUTE route={} binary=lattice backend=cpu", route.name());
    let load = Instant::now();
    let model = Qwen35Model::from_safetensors(dir)?;
    println!("LOAD route={} load_ms={:.3}", route.name(), ms(load));

    for case in POSITIVE_CASES {
        for run in 0..=runs {
            let body = case.body(run);
            let start = Instant::now();
            let prepared = lattice_prepare(
                &body,
                |p| model.tokenizer().tokenize(p).real_length,
                model.max_context(),
            )?
            .map_err(|e| format!("{}: {e:?}", case.name()))?;
            let gen_cfg = lattice_gen_cfg(
                prepared.max_tokens,
                prepared.temperature,
                prepared.top_p,
                prepared.seed,
                prepared.stop_strings.clone(),
                prepared.reasoning_budget,
                prepared.logprobs,
            );
            let prepare_ms = ms(start);
            let prompt_tokens = model.tokenizer().tokenize(&prepared.prompt).real_length;

            let start = Instant::now();
            let mut first_token_ms = None;
            model.generate_streaming_with_cancel(
                &prepared.prompt,
                &gen_cfg,
                |_| {
                    first_token_ms = Some(ms(start));
                    false
                },
                || false,
            )?;
            let first_token_ms = first_token_ms
                .ok_or_else(|| format!("{}: generation produced no token", case.name()))?;
            if run == 0 {
                continue;
            }
            println!(
                "RESULT route={} case={} run={run} prompt_tokens={prompt_tokens} \
                 prepare_ms={prepare_ms:.3} worker_prepare_ms=na admit_ms=na \
                 first_token_ms={first_token_ms:.3}",
                route.name(),
                case.name(),
            );
        }
    }
    let code = check_refusal(lattice_prepare(
        &unsupported_role_body(),
        |p| model.tokenizer().tokenize(p).real_length,
        model.max_context(),
    )?)?;
    println!(
        "RESULT route={} case=unsupported_role refused=1 code={code}",
        route.name()
    );
    Ok(())
}

#[cfg(not(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
)))]
fn run_metal(
    route: Route,
    _dir: &std::path::Path,
    _format: lattice_inference::model_format::ModelFormat,
    _runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    skip(
        route,
        "feature_absent need=macos,f16,metal-gpu,serve,bench-internals",
    );
}

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
fn run_metal(
    route: Route,
    dir: &std::path::Path,
    format: lattice_inference::model_format::ModelFormat,
    runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::Tokenizer as _;
    use lattice_inference::forward::metal_qwen35::{MetalQwen35State, format_chat_template};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::model_format::ModelFormat;
    use lattice_inference::serve::lora::ResidencyLimits;
    use lattice_inference::serve::metal_worker::bench_support::check_prompt_fits_window;
    use lattice_inference::serve::metal_worker::{
        ContextWindowPolicy, MetalWorker, VisionRuntime, WorkerEvent, WorkerMetadata,
    };
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use std::time::Instant;

    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    let binary = match route {
        Route::LatticeServe => "lattice_serve",
        _ => "lattice",
    };
    println!("ROUTE route={} binary={binary} backend=metal", route.name());
    let tokenizer = BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))?;
    let worker_tokenizer = tokenizer.clone();
    let loader_dir = dir.to_path_buf();
    let policy = match route {
        Route::LatticeServe => ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        _ => ContextWindowPolicy::PromptAndMaxTokens,
    };

    let load = Instant::now();
    let (owner, client, meta) = MetalWorker::spawn_with_vision(
        move || {
            let tokenizer_path = loader_dir.join("tokenizer.json");
            // `lattice serve` allocates its fixed 4096-token context;
            // `lattice_serve` requests the checkpoint's configured window.
            let state = match (route, format) {
                (Route::LatticeServe, ModelFormat::Safetensors) => {
                    let model = Qwen35Model::from_safetensors(&loader_dir)
                        .map_err(|e| format!("safetensors load failed: {e}"))?;
                    let cfg = model.config().clone();
                    let context = cfg.max_position_embeddings;
                    MetalQwen35State::new(model.weights(), &cfg, context)
                        .map_err(|e| format!("Metal init failed: {e}"))?
                }
                _ => {
                    let cfg = Qwen35Config::from_model_dir(&loader_dir)
                        .map_err(|e| format!("config.json load failed: {e}"))?;
                    let context = match route {
                        Route::LatticeServe => cfg.max_position_embeddings,
                        _ => LATTICE_METAL_MAX_CONTEXT,
                    };
                    MetalQwen35State::from_q4_dir(&loader_dir, &tokenizer_path, &cfg, context)
                        .map_err(|e| format!("Q4 model load failed: {e}"))?
                }
            };
            let model_max_context = match route {
                Route::LatticeServe => state.max_context(),
                _ => LATTICE_METAL_MAX_CONTEXT,
            };
            Ok((
                state,
                worker_tokenizer,
                WorkerMetadata {
                    format: format!("{format:?}"),
                    model_max_context,
                    context_window_policy: policy,
                },
            ))
        },
        VisionRuntime::unsupported(),
        1,
        ResidencyLimits::default(),
    )
    .map_err(|e| format!("worker start: {e}"))?;
    drop(owner);
    let model_max_context = meta.model_max_context;
    println!("LOAD route={} load_ms={:.3}", route.name(), ms(load));

    let prepare = |body: &[u8]| -> Result<Result<EngineRequest, ApiError>, String> {
        match route {
            Route::LatticeServe => lattice_serve_prepare(body, model_max_context),
            _ => Ok(lattice_prepare(
                body,
                |p| tokenizer.tokenize(p).real_length,
                LATTICE_METAL_MAX_CONTEXT,
            )?
            .map(|prepared| {
                let cfg = lattice_gen_cfg(
                    prepared.max_tokens,
                    prepared.temperature,
                    prepared.top_p,
                    prepared.seed,
                    prepared.stop_strings.clone(),
                    prepared.reasoning_budget,
                    prepared.logprobs,
                );
                (prepared.messages, cfg)
            })),
        }
    };

    for case in POSITIVE_CASES {
        for run in 0..=runs {
            let body = case.body(run);
            let start = Instant::now();
            let (messages, cfg) = prepare(&body)?.map_err(|e| format!("{}: {e:?}", case.name()))?;
            let prepare_ms = ms(start);

            let start = Instant::now();
            let prompt = format_chat_template(&messages);
            let prompt_tokens = tokenizer.tokenize(&prompt).real_length;
            check_prompt_fits_window(policy, model_max_context, prompt_tokens, &cfg)
                .map_err(|e| format!("{}: {e:?}", case.name()))?;
            let worker_prepare_ms = ms(start);

            let (cancel_guard, cancel_rx) = lattice_inference::serve::cancel_pair();
            let start = Instant::now();
            let mut rx = client
                .submit_with_lora(messages, cfg, cancel_rx, Vec::new())
                .map_err(|e| format!("{}: admission: {e:?}", case.name()))?;
            let admit_ms = ms(start);
            let first_token_ms = match rx.blocking_recv() {
                Some(WorkerEvent::Delta(_)) => ms(start),
                Some(WorkerEvent::Rejected(e)) => {
                    return Err(format!("{}: worker rejected: {e:?}", case.name()).into());
                }
                Some(other) => {
                    return Err(format!(
                        "{}: worker ended before the first token: {other:?}",
                        case.name()
                    )
                    .into());
                }
                None => {
                    return Err(format!("{}: worker closed the job", case.name()).into());
                }
            };
            drop(cancel_guard);
            while rx.blocking_recv().is_some() {}
            if run == 0 {
                continue;
            }
            println!(
                "RESULT route={} case={} run={run} prompt_tokens={prompt_tokens} \
                 prepare_ms={prepare_ms:.3} worker_prepare_ms={worker_prepare_ms:.3} \
                 admit_ms={admit_ms:.3} first_token_ms={first_token_ms:.3}",
                route.name(),
                case.name(),
            );
        }
    }
    let code = check_refusal(prepare(&unsupported_role_body())?)?;
    println!(
        "RESULT route={} case=unsupported_role refused=1 code={code}",
        route.name()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_CONTEXT: usize = 4096;

    #[test]
    fn route_names_round_trip_and_unknown_routes_are_refused() {
        for route in [
            Route::Cpu,
            Route::LatticeMetal,
            Route::LatticeServe,
            Route::GemmaCpu,
        ] {
            assert_eq!(Route::parse(route.name()), Ok(route));
        }
        assert!(Route::parse("metal").is_err());
        assert!(Route::parse("").is_err());
    }

    #[test]
    fn positive_cases_pass_both_handler_preparations() {
        for case in POSITIVE_CASES {
            let prepared = lattice_prepare(&case.body(1), |_| 32, TEST_CONTEXT)
                .unwrap()
                .unwrap_or_else(|e| panic!("{}: {e:?}", case.name()));
            assert!(prepared.stream, "{}", case.name());
            assert!(prepared.prompt.contains("Request 1."), "{}", case.name());
            let (messages, cfg) = lattice_serve_prepare(&case.body(1), TEST_CONTEXT)
                .unwrap()
                .unwrap_or_else(|e| panic!("{}: {e:?}", case.name()));
            assert_eq!(messages.len(), prepared.messages.len(), "{}", case.name());
            let expected_budget = (case == Case::Reasoning).then_some(64);
            assert_eq!(
                prepared.reasoning_budget,
                expected_budget,
                "{}",
                case.name()
            );
            assert_eq!(cfg.reasoning_budget, expected_budget, "{}", case.name());
        }
    }

    #[test]
    fn run_tag_changes_the_rendered_prompt() {
        let first = lattice_prepare(&Case::Control.body(1), |_| 32, TEST_CONTEXT)
            .unwrap()
            .unwrap();
        let second = lattice_prepare(&Case::Control.body(2), |_| 32, TEST_CONTEXT)
            .unwrap()
            .unwrap();
        assert_ne!(first.prompt, second.prompt);
    }

    #[test]
    fn unsupported_role_is_refused_with_the_exact_error_on_both_preparations() {
        let body = unsupported_role_body();
        assert_eq!(
            check_refusal(lattice_prepare(&body, |_| 32, TEST_CONTEXT).unwrap()),
            Ok(REFUSED_CODE)
        );
        assert_eq!(
            check_refusal(lattice_serve_prepare(&body, TEST_CONTEXT).unwrap()),
            Ok(REFUSED_CODE)
        );
    }

    /// The Gemma adapter over the committed E2B config and tokenizer fixtures,
    /// with the checkpoint's recorded `generation_config.json`.
    fn gemma_adapter() -> GemmaPromptAdapter {
        let fixtures =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma4");
        let matrix: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixtures.join("chat_template_matrix.json")).unwrap(),
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::copy(
            fixtures.join("e2b_config.json"),
            dir.path().join("config.json"),
        )
        .unwrap();
        std::fs::copy(
            fixtures.join("tokenizer/tokenizer_config.json"),
            dir.path().join("tokenizer_config.json"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            matrix["generation_config_json"].as_str().unwrap(),
        )
        .unwrap();
        let tokenizer = lattice_inference::tokenizer::GemmaBpeTokenizer::from_tokenizer_json(
            &fixtures.join("tokenizer/tokenizer.json"),
        )
        .unwrap();
        GemmaPromptAdapter::from_model_dir(dir.path(), &tokenizer).unwrap()
    }

    #[test]
    fn gemma_route_prepares_its_positive_cases_and_refuses_the_negatives() {
        let adapter = gemma_adapter();
        for case in GEMMA_POSITIVE_CASES {
            let prepared = gemma_prepare(&adapter, &case.body(1), str::len, TEST_CONTEXT)
                .unwrap()
                .unwrap_or_else(|e| panic!("{}: {e:?}", case.name()));
            assert!(prepared.stream, "{}", case.name());
            assert!(
                prepared.prompt.starts_with("<bos><|turn>"),
                "{}",
                case.name()
            );
            assert!(prepared.prompt.contains("Request 1."), "{}", case.name());
            assert!(!prepared.gen_cfg.enable_thinking, "{}", case.name());
        }
        assert_eq!(
            check_refusal(
                gemma_prepare(&adapter, &unsupported_role_body(), str::len, TEST_CONTEXT).unwrap()
            ),
            Ok(REFUSED_CODE)
        );
        assert_eq!(
            check_refusal_code(
                "reasoning",
                gemma_prepare(&adapter, &Case::Reasoning.body(1), str::len, TEST_CONTEXT).unwrap(),
                REFUSED_CODE,
            ),
            Ok(REFUSED_CODE)
        );
        assert_eq!(
            check_refusal_code(
                "unsupported_modality",
                gemma_prepare(
                    &adapter,
                    &unsupported_modality_body(),
                    str::len,
                    TEST_CONTEXT
                )
                .unwrap(),
                VISION_REFUSED_CODE,
            ),
            Ok(VISION_REFUSED_CODE)
        );
    }

    #[test]
    fn refusal_check_rejects_acceptance_and_other_errors() {
        assert!(check_refusal(Ok(())).is_err());
        let other: Result<(), ApiError> = Err(ApiError::BadRequest {
            message: "role 'developer' is not supported by this server".into(),
            code: REFUSED_CODE,
        });
        assert!(check_refusal(other).is_err());
    }
}
