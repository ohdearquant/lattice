# ADR-063: Serving Architecture --- CLI, HTTP Server, API Compatibility

**Status**: Proposed
**Date**: 2026-05-27
**Crate**: lattice-inference (new `serve` module + unified binary)
**Research**: RQ-5 (`workspaces/20260527/05.md`, 1583 lines)
**Issues**: #91 (CLI), #92 (daemon), #93 (OpenAI API), #94 (Anthropic API)
**Depends on**: ADR-048 (Continuous Batching), ADR-046 (XGrammar Structured Output), ADR-047 (Paged KV Cache)

---

## Context

### The product surface gap

Lattice has zero product surface. There are 9 standalone binaries in `crates/inference/src/bin/`, each solving one narrow problem:

| Binary             | Purpose                                                |
| ------------------ | ------------------------------------------------------ |
| `backfill_qwen3`   | Re-embed atoms from BGE-Small-384d to Qwen3-1024d      |
| `bench_decode_ab`  | A/B decode slope benchmark (Metal e2e path)            |
| `bench_logit_dump` | Logit dump for divergence analysis vs MLX              |
| `chat_metal`       | Interactive chat with Qwen3.5-2B/27B (Q4) on Metal     |
| `eval_perplexity`  | Perplexity evaluator for Qwen3.5 on text corpora       |
| `quantize_q4`      | Stream quantizer: BF16 safetensors to Q4_0 `.q4` files |
| `quantize_quarot`  | QuaRot offline converter: Hadamard-rotated Q4_0        |
| `qwen35_debug`     | Debug diagnostic for Qwen3.5-2B forward pass           |
| `qwen35_generate`  | Text generation demo                                   |

No unified CLI. No server. No HTTP handler. No API compatibility layer. A new user must `cargo run --release --bin chat_metal -- --model-dir /path/to/weights` to get a single chat response. There is no model acquisition, no model registry, no way to serve requests concurrently, and no way for external tools to talk to lattice over HTTP.

### Competitive landscape

Every competitor has `serve` as a first-class command:

| System                | Core lang      | Product surface                                                                                          | Hardware center     | Key strength                                                                                                                   | Lattice wedge                                                                                                                                                                                               |
| --------------------- | -------------- | -------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **llama.cpp**         | C/C++          | Very strong: CLI, server, OpenAI/Anthropic, embeddings, grammars, continuous batching, parallel decoding | Broad hardware      | Massive ecosystem, GGUF, mature local inference                                                                                | C++ complexity; less composable/verifiable                                                                                                                                                                  |
| **Ollama**            | Go + llama.cpp | Excellent UX: pull/run/serve/list/show/rm                                                                | Local dev machines  | Best consumer/dev UX                                                                                                           | Opaque scheduling/quantization internals                                                                                                                                                                    |
| **MLX-LM**            | Python + MLX   | Simple generate/server                                                                                   | Apple Silicon       | Apple-native, MLX models                                                                                                       | Python surface; docs warn server is not production-hardened                                                                                                                                                 |
| **vLLM / Inferact**   | Python + CUDA  | Excellent OpenAI server + bench tooling                                                                  | NVIDIA datacenter   | PagedAttention, throughput, production serving                                                                                 | Not Apple/Metal-first; not Rust-native                                                                                                                                                                      |
| **SGLang / RadixArk** | Python/PyTorch | Strong serving + RadixAttention                                                                          | Datacenter GPUs     | Prefix caching, agent workflows, structured generation                                                                         | Not Rust/Apple-first; heavier Python stack                                                                                                                                                                  |
| **Candle**            | Rust           | Framework/library                                                                                        | CPU/CUDA/WASM       | Rust ML framework, safetensors, examples                                                                                       | More framework than polished LLM serving product                                                                                                                                                            |
| **mistral.rs**        | Rust           | Strong: CLI run/serve, OpenAI API, quantization, Metal, multimodal, tool use, SDKs                       | Broad (CUDA+Metal)  | Nearest Rust competitor: zero-config HF loading, quantization breadth (GGUF/GPTQ/EXL2/HQQ), vision models, agent/tool features | Direct competitor; broader model+quant support. Lattice wedge: research-composable architecture (10 attn mechanisms, QuaRot, architecture search DSL), verified inference, Apple-Silicon-first optimization |
| **Burn**              | Rust           | Framework (not LLM-serving product)                                                                      | CPU/CUDA/WASM/Metal | Type-safe Rust ML framework, auto-diff, training                                                                               | ML framework, not polished inference server; no LLM-specific serving features                                                                                                                               |
| **lattice**           | Pure Rust      | **Currently missing**                                                                                    | Apple Silicon first | 10 attention mechanisms, QuaRot Q4, speculative decoding, embeddings, formal verification path                                 | Must add CLI/server/API immediately                                                                                                                                                                         |

### Market validation for inference infrastructure

The inference stack is now an investor category:

| Company          | Round                                   | Positioning                                     |
| ---------------- | --------------------------------------- | ----------------------------------------------- |
| **Inferact**     | $150M seed, Jan 2026, $800M valuation   | Commercialization of vLLM                       |
| **RadixArk**     | $100M seed, May 2026, led by Accel      | Commercialization of SGLang                     |
| **Fireworks AI** | $250M Series C, Oct 2025, $4B valuation | Inference platform for open models, $280M ARR   |
| **RunPod**       | $20M seed, May 2024                     | GPU cloud and serverless inference endpoints    |
| **FriendliAI**   | $20M seed extension, Aug 2025           | Enterprise inference speed + GPU cost reduction |
| **Baseten**      | $75M Series C, Feb 2025                 | Mission-critical AI inference infrastructure    |

The market understands "faster/cheaper inference." Lattice needs a sharper wedge than "we serve models." The gap inventory (`gap_inventory_20260527.md`) scored CLI+Daemon at 80/100 priority -- the single highest-impact gap.

---

## Decision

Ship a product surface before adding another model feature. The seed-demo bar is:

1. One-command model acquisition.
2. OpenAI-compatible streaming chat.
3. Anthropic-compatible streaming messages.
4. Concurrent request scheduling with cancellation.
5. Credible benchmarks against llama.cpp, MLX, and Ollama.
6. A clear technical wedge: pure Rust, Metal-native, QuaRot Q4, speculative decoding, continuous batching, embeddings, and a path toward formal verification.

### D1: Unified CLI via `clap`

A single `lattice` binary with subcommands replaces the 9 standalone binaries.

#### Command tree

```text
lattice
  pull       Download/prepare a model from Hugging Face Hub
  chat       Interactive terminal chat
  complete   One-shot prompt completion
  embed      Generate embeddings
  serve      OpenAI/Anthropic-compatible HTTP server
  bench      Benchmark model, serving, embedding, and scheduler
  info       Show system/model/backend information
  quantize   Convert/quantize model artifacts
  cache      Inspect/remove downloaded models and KV/prefix caches
```

#### Priority for seed demo

| Priority | Command            | Rationale                                       |
| -------: | ------------------ | ----------------------------------------------- |
|       P0 | `lattice serve`    | Converts library into product                   |
|       P0 | `lattice chat`     | Fastest visible demo path                       |
|       P0 | `lattice pull`     | Non-Rust users need model acquisition           |
|       P0 | `lattice bench`    | Investors need numbers                          |
|       P1 | `lattice embed`    | RAG/product wedge                               |
|       P1 | `lattice info`     | Helps demos and bug reports                     |
|       P2 | `lattice quantize` | Not required if pre-quantized demo model exists |
|       P2 | `lattice complete` | Covered by `chat --prompt`                      |
|       P2 | `lattice cache`    | Useful after MVP                                |

#### `lattice pull` flags

```text
lattice pull <REPO_ID>
  --revision <REV>            HF revision/branch (default: main)
  --quant <METHOD>            Quantize after download: quarot-q4, q4, none
  --alias <NAME>              Friendly name for model registry
  --cache-dir <PATH>          Override cache directory
  --allow-pattern <GLOB>      Only download matching files (e.g. "*.safetensors")
  --ignore-pattern <GLOB>     Skip matching files (e.g. "*.bin")
  --yes                       Skip confirmation prompt
  --offline                   Fail if artifacts missing locally
```

`lattice pull` behavior:

1. Resolve HF repo + revision via `hf-hub` crate.
2. Read model card, config, tokenizer metadata.
3. Check disk space before download.
4. Download with progress bars.
5. Verify checksums where available.
6. Convert/quantize if `--quant` specified.
7. Register alias in model registry.
8. Print a runnable command: `lattice chat <alias>`.

Auto-pull: `lattice chat Qwen/Qwen3-0.6B` prompts "Model not local. Download ~1.2 GB? [Y/n]" unless `--yes` or `--offline`.

#### `lattice serve` flags

```text
lattice serve <MODEL>
  --host <ADDR>               Bind address (default: 127.0.0.1)
  --port <PORT>               Bind port (default: 8080)
  --openai                    Enable OpenAI-compatible endpoints
  --anthropic                 Enable Anthropic-compatible endpoints
  --parallel <N>              Concurrent decode slots (default: 4)
  --ctx <N>                   Context window per slot (default: 8192)
  --max-queue <N>             Admission control queue depth (default: 64)
  --max-loaded-models <N>     v1: always 1
  --kv-budget <SIZE>          KV cache memory budget (e.g. 8GiB)
  --keep-alive <DUR>          Idle model unload timer (default: 10m)
  --prefix-cache <MODE>       none | radix (default: none)
  --spec <MODE>               Speculative decoding: ngram | mtp | none
  --chat-template <PATH>      Override Jinja template
  --api-key <KEY>             Require Bearer token for auth
```

#### `lattice chat` flags

```text
lattice chat <MODEL>
  --ctx <N>                   Context window (default: 8192)
  --temperature <F>           Sampling temperature (default: 0.7)
  --top-p <F>                 Nucleus sampling (default: 0.95)
  --max-tokens <N>            Max generation length (default: 512)
  --system <MSG>              System prompt
  --prompt <MSG>              One-shot mode (non-interactive)
  --seed <N>                  Deterministic sampling
  --response-format <FMT>     text | json_object | json_schema:<path>
```

#### `lattice bench` flags

```text
lattice bench <MODEL>
  --suite <NAME>              serving | decode | prefill | embed | lifecycle
  --endpoint <URL>            Benchmark running server (e.g. http://127.0.0.1:8080/v1)
  --prompt-len <N>            Prompt token count (default: 512)
  --decode-len <N>            Decode token count (default: 128)
  --concurrency <LIST>        Comma-separated concurrency levels (e.g. 1,2,4,8)
  --runs <N>                  Measurement iterations (default: 30)
  --warmup <N>                Warmup iterations (default: 3)
  --compare <LIST>            Frameworks to compare: llama.cpp,mlx,ollama
  --json <PATH>               Output JSON report
  --live                      Stream live metrics during benchmark
```

### D2: Model registry

Local model storage with provenance metadata.

#### Directory layout

```text
~/.cache/lattice/
  hub/
    models--Qwen--Qwen3-0.6B/
      snapshots/<revision>/
  models/
    qwen3-0.6b-q4/
      lattice-model.toml          # manifest: name, alias, source, quant, date
      tokenizer.json              # HF tokenizer
      tokenizer_config.json       # HF tokenizer config (chat_template lives here)
      chat_template.jinja         # extracted/overridden Jinja template
      model.safetensors           # or *.q4 for quantized
      quantization.json           # quant method, seed, params
      checksums.json              # SHA-256 per file
```

#### Manifest schema (`lattice-model.toml`)

```toml
[model]
name = "Qwen/Qwen3-0.6B"
alias = "qwen3-0.6b-q4"
revision = "main"
download_date = "2026-05-27T10:30:00Z"

[quantization]
method = "quarot-q4"
seed = 42
original_dtype = "bf16"

[source]
hub = "huggingface"
repo_id = "Qwen/Qwen3-0.6B"
commit_sha = "abc123..."

[checksums]
"model.q4" = "sha256:..."
"tokenizer.json" = "sha256:..."
```

#### `hf-hub` integration

Use the Rust `hf-hub` crate (not `huggingface-cli` Python) to maintain the pure-Rust constraint. The crate provides async and blocking clients for repo/file operations and downloads. Candle also uses `hf-hub`, providing a Rust ecosystem precedent.

### D3: HTTP server architecture

```text
HTTP clients
   |
   v
axum / tokio HTTP server
   |  parse JSON, auth, CORS, request ID (via Tower middleware)
   |
   v
Request normalizer
   |  chat template (MiniJinja), tokenizer, sampling params, response_format
   |
   v
bounded MPSC channel (tokio::sync::mpsc, capacity 64)
   |
   v
dedicated inference scheduler thread (NOT on tokio runtime)
   |  continuous batching (ADR-048), cancellation polling, KV page allocation
   |
   v
lattice engine / Metal backend
   |  synchronous GPU submission + wait_until_completed
   |
   v
per-request token channel (tokio::sync::mpsc, capacity 128)
   |
   v
SSE stream to client
```

#### Why GPU work MUST run on a dedicated OS thread

Metal command buffer submission is synchronous: `commit()` + `wait_until_completed()` blocks the calling thread until the GPU finishes. If this runs on a tokio worker thread, it starves all HTTP handling for the duration of each GPU step (potentially 10-100ms). The inference scheduler runs on a dedicated `std::thread` spawned with `Builder::new().name("lattice-inference")`.

The tokio runtime handles only: HTTP accept/parse, JSON serialization, SSE stream construction, channel I/O.

The scheduler thread handles: request queuing, batch construction, GPU submission, token distribution.

Communication between the two is a bounded `tokio::sync::mpsc` channel. The bounded capacity (64 pending requests) provides backpressure: when the queue is full, the HTTP handler returns `503 Service Unavailable` immediately rather than buffering unboundedly.

#### `AppState` struct

```rust
#[derive(Clone)]
pub struct AppState {
    /// Channel to the inference scheduler thread.
    pub engine_tx: mpsc::Sender<EngineRequest>,
    /// Default model alias when request omits `model` field.
    pub default_model: String,
    /// Chat template engine (MiniJinja).
    pub templates: Arc<TemplateEngine>,
    /// Tokenizer for prompt encoding.
    pub tokenizer: Arc<BpeTokenizer>,
    /// Model metadata for /v1/models.
    pub model_info: Arc<ModelInfo>,
}
```

#### Router setup

```rust
use axum::{Router, routing::{get, post}};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/v1/embeddings", post(embeddings))
        // Anthropic-compatible endpoint
        .route("/v1/messages", post(messages))
        // Operational
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        // Middleware stack
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .layer(RequestIdLayer::new())
        .with_state(state)
}
```

#### SSE handler for `/v1/chat/completions`

The handler creates a per-request cancellation token, submits an `EngineRequest` to the scheduler, and returns an SSE stream that yields `chat.completion.chunk` objects:

```rust
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let model = req.model.clone()
        .unwrap_or_else(|| state.default_model.clone());

    // 1. Apply HF-compatible chat template.
    let prompt = state.templates
        .render_chat_prompt(&model, &req.messages, true)
        .map_err(|e| ApiError::bad_request(&format!("chat_template_error: {e}")))?;

    // 2. Per-request token channel. Bounded to prevent unbounded memory growth.
    let (out_tx, mut out_rx) = mpsc::channel::<EngineEvent>(128);

    // 3. Cancellation token: owned by HTTP stream, observed by scheduler.
    let cancel = CancellationToken::new();

    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_unix();

    let engine_req = EngineRequest {
        request_id: request_id.clone(),
        model: model.clone(),
        prompt,
        params: SamplingParams::from_req(&req),
        out: out_tx,
        cancel: cancel.clone(),
    };

    // 4. Submit to scheduler. Bounded channel provides backpressure.
    //    Use try_send for immediate admission control — do NOT await capacity.
    state.engine_tx.try_send(engine_req)
        .map_err(|e| match e {
            tokio::sync::mpsc::error::TrySendError::Full(_) =>
                ApiError::service_unavailable("queue_full"),
            tokio::sync::mpsc::error::TrySendError::Closed(_) =>
                ApiError::service_unavailable("engine_not_running"),
        })?;

    // 5. Construct SSE stream.
    let stream = async_stream::stream! {
        let mut cancel_guard = CancelOnDrop::new(cancel);

        // First chunk: assistant role (OpenAI convention).
        yield Ok(event_json(&chunk_role(&request_id, created, &model)));

        while let Some(ev) = out_rx.recv().await {
            match ev {
                EngineEvent::Token { text, .. } => {
                    yield Ok(event_json(
                        &chunk_token(&request_id, created, &model, text)
                    ));
                }
                EngineEvent::Done { finish_reason } => {
                    yield Ok(event_json(
                        &chunk_done(&request_id, created, &model, finish_reason)
                    ));
                    yield Ok(Event::default().data("[DONE]"));
                    cancel_guard.disarm();
                    break;
                }
                EngineEvent::Error { message } => {
                    let err = serde_json::json!({
                        "error": { "message": message, "type": "server_error",
                                   "param": null, "code": "engine_error" }
                    });
                    yield Ok(Event::default().data(err.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
        // If stream is dropped before Done, CancelOnDrop fires cancellation.
    };

    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive")))
}
```

The `CancelOnDrop` guard ensures that when a client disconnects mid-stream, the cancellation token fires, the scheduler evicts the request at the next scheduling boundary, and KV pages are released.

#### Anthropic `/v1/messages` handler

The Anthropic Messages API uses named SSE event types rather than data-only events:

```text
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc","type":"message",
       "role":"assistant","content":[],"model":"qwen3-0.6b-q4",
       "stop_reason":null,"stop_sequence":null,
       "usage":{"input_tokens":42,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,
       "content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,
       "delta":{"type":"text_delta","text":"KV"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta",
       "delta":{"stop_reason":"end_turn","stop_sequence":null},
       "usage":{"output_tokens":12}}

event: message_stop
data: {"type":"message_stop"}
```

The handler reuses the same `EngineRequest` submission path as the OpenAI handler but wraps the token stream in Anthropic's event envelope. The system prompt is extracted from the request body's top-level `system` field (Anthropic convention) rather than a system-role message (OpenAI convention).

#### Request normalizer

The request normalizer translates between API-specific request formats and the internal `EngineRequest`:

```rust
pub struct EngineRequest {
    pub request_id: String,
    pub model: String,
    pub prompt: String,           // tokenized chat template output
    pub params: SamplingParams,
    pub out: mpsc::Sender<EngineEvent>,
    pub cancel: CancellationToken,
}

pub struct SamplingParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub response_format: Option<ResponseFormat>,
}
```

Responsibilities:

1. **Chat template rendering**: Load the model's Jinja template from `tokenizer_config.json["chat_template"]` via MiniJinja. Fallback chain: (1) per-model template from registry, (2) `chat_template.jinja` file, (3) built-in fallback by model family, (4) explicit `--chat-template` flag. Support templates for Qwen, Llama, Mistral, Gemma, Phi families at launch.

2. **Sampling parameter mapping**: Both OpenAI and Anthropic use `temperature`, `top_p`, `max_tokens`. OpenAI adds `stop`, `seed`, `response_format`. Anthropic adds `top_k`. Map to internal `SamplingParams` with sensible defaults (temperature=0.7, top_p=0.95, max_tokens=512).

3. **Model alias resolution**: `"model": "qwen3-0.6b-q4"` resolves through the model registry to the physical model path and metadata.

4. **Response format dispatch**: `response_format: {"type": "json_object"}` activates the XGrammar engine (ADR-046) with a JSON grammar. `{"type": "json_schema", "json_schema": {...}}` converts the schema to a grammar. Unsupported features return `400 unsupported_schema_feature`.

### D4: Inference scheduler

This is the most critical component. The scheduler is the single point of coordination between concurrent HTTP requests and the serial GPU resource.

#### Architecture

```rust
pub fn spawn_inference_thread(
    mut rx: mpsc::Receiver<EngineRequest>,
    mut engine: LatticeEngine,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("lattice-inference".to_string())
        .spawn(move || {
            let mut scheduler = ContinuousBatchScheduler::new(&mut engine);

            loop {
                // Drain newly arrived requests without blocking.
                while let Ok(req) = rx.try_recv() {
                    scheduler.enqueue(req);
                }

                if scheduler.has_work() {
                    scheduler.step();
                    continue;
                }

                // No active work: block until at least one request arrives.
                match rx.blocking_recv() {
                    Some(req) => scheduler.enqueue(req),
                    None => break, // HTTP server shut down.
                }
            }
        })
}
```

#### Scheduling tick

Each `step()` call performs one iteration of the continuous batching loop:

```rust
impl ContinuousBatchScheduler {
    pub fn step(&mut self) {
        // 1. Cancel completed/disconnected requests before scheduling GPU work.
        self.reap_cancelled();

        // 2. Decode gets priority for interactivity.
        let decode_batch = self.build_decode_microbatch();

        // 3. Fill remaining budget with chunked prefill (ADR-048).
        let prefill_batch = self.build_prefill_chunk(512); // 512-token chunks

        // 4. Submit one Metal command buffer. Blocks only this OS thread.
        let results = self.engine.run_step_blocking(decode_batch, prefill_batch);

        // 5. Stream tokens back. If receiver is gone, mark sequence cancelled.
        for result in results {
            match result {
                StepResult::Token { seq_id, text, token_id } => {
                    let seq = self.sequence_mut(seq_id);
                    if seq.out.blocking_send(EngineEvent::Token { text, token_id })
                        .is_err()
                    {
                        seq.cancel.cancel();
                    }
                }
                StepResult::Finished { seq_id, finish_reason } => {
                    let seq = self.sequence_mut(seq_id);
                    let _ = seq.out.blocking_send(
                        EngineEvent::Done { finish_reason }
                    );
                    self.free_sequence(seq_id);
                }
            }
        }
    }
}
```

#### Continuous batching integration (ADR-048)

The scheduler implements the `FifoScheduler` from ADR-048:

- **Chunked prefill**: Long prompts are split into 512-token chunks and interleaved with decode steps. This bounds prefill latency spikes to ~100ms on M2 Max for Qwen3.5-2B Q4, preventing head-of-line blocking.
- **Iteration-level scheduling**: Requests enter and exit the execution window at iteration boundaries, not request boundaries. Finished sequences are evicted and new ones fill their slots.
- **GPU as single resource**: One Metal `MTLCommandBuffer` per scheduling tick containing both prefill and decode compute encoders. Apple Silicon GPUs may overlap independent work, but LLM kernels saturate memory bandwidth -- treat the model as a single resource.

#### KV cache page allocation from PagedKVCache pool

The scheduler allocates KV pages from the `PagedKVCache` (ADR-047) pool:

- Each new request gets pages allocated on demand as prefill progresses.
- Page size: 256 tokens (matching existing `PagedKVCache` configuration).
- When a request completes or is cancelled, its pages are returned to the pool (or retained in the prefix cache if RadixAttention is enabled).
- Admission control: new requests are rejected with `503` if `kv_free_pages < PREFILL_RESERVE_PAGES`.

#### Per-request cancellation

Cancellation is boundary-based, not preemptive:

1. Client disconnects -- axum drops the SSE stream.
2. `CancelOnDrop` calls `cancel.cancel()`.
3. Scheduler observes `is_cancelled()` before the next GPU submission (in `reap_cancelled()`).
4. Sequence is removed from active decode batches.
5. KV pages are released or returned to prefix cache.
6. Output channel is dropped.

If a Metal command buffer is already executing, we do not attempt fine-grained token-level preemption. The command buffer completion boundary is the cancellation boundary. This is another reason to keep prefill chunks at 512 tokens.

#### Speculative decoding integration

When `--spec ngram` or `--spec mtp` is set, the scheduler integrates with ADR-006's speculative decoding:

- Draft tokens are proposed by the speculator (N-gram or MTP).
- Verification runs as part of the regular decode step.
- Accepted tokens are streamed to the client; rejected tokens are discarded.
- The scheduler treats speculative sequences identically to regular sequences for batching and cancellation purposes.

### D5: OpenAI Chat Completions API (`POST /v1/chat/completions`)

#### Supported request fields (v1)

| Field                                     | v1 behavior                                                                       |
| ----------------------------------------- | --------------------------------------------------------------------------------- |
| `model`                                   | Required unless server has one loaded default                                     |
| `messages`                                | Required. `role`: system/user/assistant. `content`: string or content-parts array |
| `stream`                                  | Support both `true` and `false`. Non-streaming accumulates tokens then returns    |
| `temperature`                             | Implement. Default 0.7                                                            |
| `top_p`                                   | Implement. Default 0.95                                                           |
| `max_tokens`                              | Implement. Default 512                                                            |
| `stop`                                    | Implement. String or array of strings                                             |
| `seed`                                    | Implement. Deterministic sampling when set                                        |
| `response_format: {"type":"text"}`        | Default. Normal decoding                                                          |
| `response_format: {"type":"json_object"}` | JSON grammar-constrained decoding (ADR-046)                                       |
| `response_format: {"type":"json_schema"}` | Convert schema to grammar; `400` if unsupported features                          |
| `tools`, `tool_choice`                    | Parse; v1 rejects with clear error unless model/template supports it              |
| `logprobs`                                | Do not fake. Return `400 unsupported_feature`                                     |
| `n > 1`                                   | Return `400` for v1                                                               |
| Unknown fields                            | Ignore unless strict mode enabled                                                 |

#### Streaming wire format

```text
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1710000000,
       "model":"qwen3-0.6b-q4","choices":[{"index":0,
       "delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1710000000,
       "model":"qwen3-0.6b-q4","choices":[{"index":0,
       "delta":{"content":"KV"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1710000000,
       "model":"qwen3-0.6b-q4","choices":[{"index":0,
       "delta":{"content":" cache"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1710000000,
       "model":"qwen3-0.6b-q4","choices":[{"index":0,
       "delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

First chunk always includes `delta.role = "assistant"` with empty content. Token chunks have `delta.content` only. Final chunk has `finish_reason` with no content. `data: [DONE]` terminates the stream.

#### Non-streaming response format

```json
{
  "id": "chatcmpl-abc",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "qwen3-0.6b-q4",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "KV cache stores precomputed key-value pairs..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 18,
    "total_tokens": 60
  }
}
```

#### Error response format

```json
{
  "error": {
    "message": "Model 'nonexistent' not found",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

HTTP status codes: `400` for bad requests, `401` for auth failures, `404` for unknown models, `503` for engine not running / queue full.

### D6: Anthropic Messages API (`POST /v1/messages`)

#### Supported request fields (v1)

| Field            | v1 behavior                                                 |
| ---------------- | ----------------------------------------------------------- |
| `model`          | Required                                                    |
| `messages`       | Required. `role`: user/assistant only (system is top-level) |
| `system`         | Top-level system prompt (Anthropic convention)              |
| `stream`         | Support both `true` and `false`                             |
| `max_tokens`     | Required                                                    |
| `temperature`    | Implement. Default 1.0 (Anthropic default)                  |
| `top_p`          | Implement                                                   |
| `top_k`          | Implement (not in OpenAI)                                   |
| `stop_sequences` | Implement                                                   |
| `tools`          | Parse; v1 rejects                                           |
| `metadata`       | Parse; store for logging                                    |

#### Streaming event types

| Event                 | When               | Payload                                                             |
| --------------------- | ------------------ | ------------------------------------------------------------------- |
| `message_start`       | First event        | Full message shell with `id`, `role`, `model`, `usage.input_tokens` |
| `content_block_start` | Before first token | `index: 0`, `content_block: {"type": "text", "text": ""}`           |
| `content_block_delta` | Each token         | `index: 0`, `delta: {"type": "text_delta", "text": "..."}`          |
| `content_block_stop`  | After last token   | `index: 0`                                                          |
| `message_delta`       | After content done | `delta: {"stop_reason": "end_turn"}`, `usage.output_tokens`         |
| `message_stop`        | Final event        | Empty                                                               |

### D7: Chat template rendering

Do not hardcode templates. HF chat templates are Jinja templates stored with the tokenizer; using the wrong control tokens materially degrades model behavior.

Use **MiniJinja** as the Rust template engine. Candle's chat-template support uses MiniJinja and loads templates from `tokenizer_config.json`, providing a Rust precedent.

#### Template registry

```rust
pub struct ChatTemplateRegistry {
    by_model: HashMap<String, CompiledTemplate>,
    fallback_by_family: HashMap<ModelFamily, CompiledTemplate>,
}

pub struct TemplateContext<'a> {
    messages: &'a [ChatMessage],
    add_generation_prompt: bool,
    bos_token: Option<&'a str>,
    eos_token: Option<&'a str>,
    tools: Option<&'a serde_json::Value>,
}
```

#### Load order

1. **`--chat-template <PATH>` CLI flag** (highest priority, always wins)
2. `tokenizer_config.json["chat_template"]`
3. `chat_template.jinja` file in model directory
4. Built-in fallback by model family (Qwen, Llama, Mistral, Gemma, Phi)
5. Explicit `--chat-template path.jinja`

#### Required compatibility helpers for MiniJinja

```text
raise_exception()          # HF templates use this for unsupported features
strftime_now()             # Timestamp injection
tojson                     # JSON serialization filter
dict/items compatibility   # Python dict patterns
trim_blocks / lstrip_blocks behavior
```

### D8: RadixAttention prefix cache

For multi-turn chat, coding agents, and RAG workloads, prefix caching dramatically reduces TTFT by reusing KV cache entries for shared prompt prefixes.

#### Data structure

Lattice's `PagedKVCache(page_size = 256)` uses a radix tree over token IDs with namespace isolation:

```rust
pub struct CacheNamespace {
    pub model_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub chat_template_hash: [u8; 32],
    pub rope_config_hash: [u8; 32],
    pub quant_policy_hash: [u8; 32],
    pub adapter_hash: Option<[u8; 32]>, // LoRA namespace isolation
}
```

Namespace must include adapter identity: same tokens but different LoRA adapter means different KV. vLLM's prefix-caching docs explicitly include LoRA ID in the KV block hash.

#### Page-aligned matching (v1)

```text
matched_len = floor(longest_token_prefix / prefix_page_size) * prefix_page_size
// prefix_page_size defaults to 64 (ADR-047, prefix.rs:111) — NOT 256.
// Using 256 here would conflict with ADR-047's accepted page size.
recompute_tail = longest_token_prefix - matched_len
```

v1 uses page-aligned matching only. With `prefix_page_size=64` (ADR-047 default), this loses up to 63 reusable tokens per request but has simple correctness and zero-copy page reuse. v2 adds token-aligned matching with partial-page tails for short system prompts.

#### Eviction policy

SGLang-style LRU with reference counting:

1. Evict only pages/nodes with `active_refcount == 0`.
2. Prefer least-recently-used.
3. Prefer leaves before internal nodes (preserve high-fanout shared prefixes).
4. Pinned pages for active sequences are never evicted.
5. Under severe memory pressure, cached pages yield to running batches.

### D9: Cross-framework benchmark suite

#### Two tracks

**Track A -- closest numerical equivalence**: Use the least-transformed model format across frameworks. FP16/BF16 where possible for strict comparability. Then "best practical Q4" as a separate track.

| Framework | Model format                 |
| --------- | ---------------------------- |
| lattice   | safetensors / lattice native |
| llama.cpp | GGUF                         |
| Ollama    | same GGUF where possible     |
| MLX       | MLX-converted weights        |

**Track B -- product-realistic comparison**: Use each framework as users actually use it. This is the investor-relevant result.

#### Workloads

| Workload         | Prompt len | Output len | Concurrency   | Primary metric                    |
| ---------------- | ---------- | ---------- | ------------- | --------------------------------- |
| Interactive chat | 256        | 128        | 1             | TTFT, tok/s                       |
| Long prompt chat | 2048       | 128        | 1             | TTFT, prefill tok/s               |
| Decode stress    | 128        | 512        | 1             | decode tok/s                      |
| Concurrent chat  | 512        | 128        | 1,2,4,8       | aggregate tok/s, p99 latency      |
| RAG answer       | 4096       | 256        | 1,4           | TTFT, memory, p99                 |
| Embedding        | 128,512    | n/a        | batch 1,16,64 | embeddings/sec, tokens/sec        |
| Model lifecycle  | n/a        | n/a        | n/a           | cold load, warm load, peak memory |

#### Statistical rigor

```text
warmup: 3 runs
measurement: 30 runs for single-request tests
load tests: 60-180 seconds per concurrency level
report: mean, median, p95, p99, stddev, 95% CI
outlier policy: predeclared; do not silently delete
randomize benchmark order to avoid thermal bias
```

#### Metric definitions

```text
TTFT = first_token_time - request_send_time
decode_tok_s = generated_tokens / (last_token_time - first_token_time)
end_to_end_tok_s = generated_tokens / (last_token_time - request_send_time)
prefill_tok_s = prompt_tokens / prefill_time
throughput = total_generated_tokens / wall_clock_seconds
p99_inter_token_latency = p99(delta between streamed chunks per request)
queue_delay = scheduler_admit_time - request_arrival_time
```

#### Seed-demo targets (Qwen-class 0.6B-0.8B Q4 on M2 Max)

| Metric                             | Good       | Seed-impressive |
| ---------------------------------- | ---------- | --------------- |
| Warm TTFT, 128-token prompt        | <150 ms    | <80 ms          |
| Warm TTFT, 2k-token prompt         | <600 ms    | <300 ms         |
| Cached TTFT, shared system prompt  | <100 ms    | <50 ms          |
| Single-stream decode               | >180 tok/s | >250 tok/s      |
| Prompt processing, 512 tokens      | >4k tok/s  | >8k tok/s       |
| 4 concurrent streams aggregate     | >350 tok/s | >600 tok/s      |
| p99 inter-token latency, 4 streams | <100 ms    | <60 ms          |
| Model load, warm filesystem cache  | <2 s       | <1 s            |
| Peak memory, model + 4x4k ctx      | <3 GB      | <2 GB           |

These are benchmark targets, not achieved numbers. Do not present as achieved unless measured in this session.

### D10: Competitive positioning

**Avoid:**

- "We are a faster Ollama."
- "We are llama.cpp in Rust."
- "We are vLLM for Mac."

**Use:**

> lattice is a pure-Rust inference microkernel for local and edge LLM systems: OpenAI-compatible at the surface, research-composable in the core, and Apple-Silicon-native from day one.

The "microkernel" framing separates a minimal trusted core from pluggable modules:

**Trusted core** (the scheduler):

- Scheduler + decode loop
- KV cache manager
- Sampling/grammar constraints
- Backend abstraction (Metal, CPU)
- Model registry
- API compatibility layer

**Pluggable modules** (the research platform):

- Attention mechanism (10 implementations: GQA, GDN, Flash, NSA, ...)
- Quantization policy (QuaRot Q4, Q8, FP16)
- Draft model (N-gram, MTP speculative decoding)
- Chat template (per-model Jinja)
- LoRA adapter (hot-swap, KV namespace isolation)
- Embedding backend (BERT, sentence-transformers)
- Prefix cache (RadixAttention)
- Verification harness (Lean4 path via styx)

This gives investors a coherent systems story: minimal core, high extensibility, fault isolation, clear boundaries for formal methods.

---

## Alternatives considered

| Alternative                      | Pros                               | Cons                                                                      | Decision |
| -------------------------------- | ---------------------------------- | ------------------------------------------------------------------------- | -------- |
| **actix-web** instead of axum    | Battle-tested, good performance    | Worse tokio integration, no Tower middleware, less ergonomic SSE          | Rejected |
| **hyper directly**               | Maximum control                    | Too low-level for JSON parsing, routing, middleware                       | Rejected |
| **Unix socket** for local daemon | Lower latency than TCP             | HTTP over localhost sufficient for v1; adds complexity                    | Deferred |
| **gRPC**                         | Type-safe, bidirectional streaming | OpenAI/Anthropic compatibility requires HTTP+JSON; gRPC adds protobuf dep | Deferred |
| **Python wrapper** (like MLX-LM) | Faster to prototype                | Violates pure-Rust constraint; performance overhead                       | Rejected |
| **Embed HTTP in each binary**    | Incremental, no unified CLI needed | Duplicates server logic; no unified model registry; terrible UX           | Rejected |
| **Axum + GPU on tokio**          | Simpler (no dedicated thread)      | Metal blocks tokio workers, starving HTTP handling                        | Rejected |

---

## Dependencies and risk

### New crate dependencies

| Crate                 | Purpose                               | Risk                                           |
| --------------------- | ------------------------------------- | ---------------------------------------------- |
| `axum`                | HTTP server, routing, SSE             | Well-maintained, tokio team. Low risk          |
| `tower`, `tower-http` | Middleware (CORS, trace, compression) | Part of tokio ecosystem. Low risk              |
| `minijinja`           | Jinja2 chat template rendering        | Active maintenance, used by Candle. Low risk   |
| `hf-hub`              | Hugging Face model downloads          | Active, used by Candle. API stability moderate |
| `clap`                | CLI argument parsing                  | De facto standard. Negligible risk             |
| `async-stream`        | SSE stream construction               | Thin macro crate. Low risk                     |
| `tokio-util`          | `CancellationToken`                   | Part of tokio ecosystem. Low risk              |
| `indicatif`           | Progress bars for `lattice pull`      | Optional, display only. Negligible risk        |

Note: `tokio` and `serde_json` are already workspace dependencies. `axum` pulls in `hyper` but does not add a new async runtime.

### Scope boundaries for v1

**In scope:**

- `lattice pull`, `lattice chat`, `lattice serve`, `lattice bench`, `lattice info`
- `/v1/chat/completions` (streaming + non-streaming)
- `/v1/messages` (streaming)
- `/v1/models`, `/health`
- Single-model serving
- Continuous batching scheduler
- Per-request cancellation
- Chat template rendering via MiniJinja
- Model registry with alias system
- JSON/JSON-schema response format (via ADR-046)

**Deferred to v2:**

- Function calling / tool use
- Vision / multimodal input
- `/v1/embeddings` via HTTP (use `lattice embed` CLI for now)
- Multi-model concurrent serving
- LoRA hot-swap via API
- Token-aligned prefix cache (partial pages)
- Log-probability endpoints
- `stream_options.include_usage`

### Risk: scope creep

The seed demo is `lattice pull -> lattice serve -> curl` in under 60 seconds. Every feature not on that critical path is a distraction. The v1 boundary is deliberately narrow: chat completions + messages + streaming + benchmarks. No function calling, no vision, no assistants API, no fine-tuning API.

---

## Consequences

- **Seed-round demo**: `lattice pull -> lattice serve -> curl` in under 60 seconds. This is the pitch.
- **Ecosystem integration**: OpenAI-compatible endpoint enables LangChain, Cursor, Continue, Aider, LlamaIndex -- any tool that speaks the OpenAI Chat Completions API.
- **Benchmark comparisons**: `lattice bench` enables apples-to-apples throughput comparison with llama.cpp server, MLX-LM serve, Ollama.
- **Architecture unlock**: The dedicated scheduler thread + MPSC channel pattern is the foundation for all future serving features (multi-model, LoRA hot-swap, prefix cache, priority scheduling).
- **Dependency footprint**: Adds ~6 crates to the workspace, all from the tokio ecosystem. No new async runtimes, no C dependencies, pure Rust maintained.

---

## Implementation plan

### v0.3 Seed Boundary (hard scope cut)

**In scope for v0.3 (seed demo):** P1-P5 + P8. The demo is `lattice pull -> lattice serve -> curl`. Single-model, single-request HTTP serving with OpenAI streaming. Benchmark harness for competitive numbers.

**Deferred to v0.4+:** Anthropic API (P6), continuous batching (P7), prefix cache, speculative serving, structured output, embeddings HTTP, multi-client benchmark polish.

| Phase | Scope                                                                            | Issues | Est. effort    | v0.3?                         |
| ----- | -------------------------------------------------------------------------------- | ------ | -------------- | ----------------------------- |
| P1    | CLI skeleton (`clap` subcommands, model registry, `lattice pull`)                | #91    | 3-4 days       | **Yes**                       |
| P2    | `lattice chat` (interactive terminal, wraps existing Metal path)                 | #91    | 2-3 days       | **Yes**                       |
| P3    | HTTP server skeleton (axum, health, `/v1/models`)                                | #92    | 2-3 days       | **Yes**                       |
| P4    | Inference scheduler (dedicated thread, MPSC, single-request decode)              | #92    | **2-3 weeks**  | **Yes** (single-request only) |
| P5    | OpenAI `/v1/chat/completions` (streaming + non-streaming)                        | #93    | 3-4 days       | **Yes**                       |
| P6    | Anthropic `/v1/messages` (streaming + non-streaming)                             | #94    | 3-4 days       | v0.4                          |
| P7    | Continuous batching (multi-request, chunked prefill, KV lifecycle, cancellation) | #92    | **6-10 weeks** | v0.4                          |
| P8    | `lattice bench --suite serving` (cross-framework comparison)                     | #84    | 3-4 days       | **Yes**                       |

**Note on P4/P7 effort revision**: ADR-048 describes the Metal `forward_step` migration as a "non-trivial refactor." The single-request scheduler (P4) is simpler — no batch construction, KV lifecycle, or cancellation — but still requires the dedicated-thread + channel architecture, request/response marshaling, error handling, and Metal pipeline integration. Comparable serving loops (vLLM, SGLang, llama.cpp server) took months; P4's single-request scope is feasible in 2-3 weeks, P7's full batching in 6-10 weeks. Previous estimates of 3-4 days (P4) and 4-5 days (P7) were not credible.

**Benchmark targets require measured baselines.** Before any investor-facing claim, measure llama.cpp, MLX-LM, and mistral.rs on the same hardware (M2 Max), same model (Qwen3-0.6B Q4), same prompt/decode lengths. The target table should include: hardware, OS, framework version/commit, command line, quantization, prompt/decode lengths, TTFT, prefill tok/s, decode tok/s, p95 latency, peak memory.

The first seed-round milestone: **a non-Rust developer can install lattice, pull a model, chat with it, serve it behind the OpenAI API, run Aider/Continue/LangChain against it, and see credible latency/throughput numbers on an M2 Max.**

---

## References

- RQ-5 research memo: `workspaces/20260527/05.md`
- ADR-048: Continuous Batching with Disaggregated Prefill/Decode
- ADR-046: XGrammar Structured Output Engine
- ADR-047: Paged KV Cache with Prefix Reuse
- ADR-006: Speculative Decoding
- axum SSE docs: https://docs.rs/axum/latest/axum/response/sse/
- OpenAI Chat Completions streaming: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events/
- Anthropic streaming messages: https://docs.anthropic.com/en/api/messages-streaming
- MiniJinja: https://github.com/mitsuhiko/minijinja
- hf-hub: https://github.com/huggingface/hf-hub
- llama.cpp server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Ollama FAQ (concurrency): https://docs.ollama.com/faq
- SGLang RadixAttention: https://lmsys.org/blog/2024-01-17-sglang/
- vLLM automatic prefix caching: https://docs.vllm.ai/en/v0.8.3/design/automatic_prefix_caching.html
- Apple Metal best practices: https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/PersistentObjects.html
