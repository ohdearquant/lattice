# `lattice serve` HTTP API

The README shows one bare `curl` example against `/v1/chat/completions`. This document covers
the rest of the surface: exact request/response shapes, streaming, error responses, validation
order, and behavior that is easy to get wrong if you only read the happy path. Everything below
was verified against `crates/inference/src/bin/lattice.rs`'s `mod serve` (the `lattice serve`
subcommand's implementation) and confirmed live against a running server built from this
repository.

## A naming note before anything else

This codebase has **two** separately built HTTP servers with confusingly similar names:

- **`lattice serve`** — a subcommand of the unified `lattice` CLI binary
  (`crates/inference/src/bin/lattice.rs`). This is the general-purpose, OpenAI-compatible server
  the README documents (Quick Start and "### HTTP API" sections), with an active development
  history (unified-CLI ADR-063, SSE streaming, stop sequences, native Q4 checkpoint support,
  `lattice doctor` preflight). **This document is about `lattice serve`.**
- **`lattice_serve`** — a separate, standalone binary
  (`crates/inference/src/bin/lattice_serve.rs`), purpose-built as the internal HTTP daemon the
  macOS Lattice Studio app spawns and talks to (introduced in PR #435). It has its own, narrower
  route set (`GET /`, `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`) and its own
  disconnect-cancellation behavior (PR #552/#606) that `lattice serve` does not have (see
  "Streaming" below). It is not what the README's HTTP API section documents, and it is out of
  scope for this document.

If you arrived here from an issue or note that points at `lattice_serve.rs` specifically: the
README's actual HTTP API example — the thing that issue was asking to be expanded — targets
`lattice serve` (the CLI subcommand), not the standalone `lattice_serve` binary. This document
covers the one the README documents.

## Starting the server

```bash
# CPU (safetensors checkpoint)
cargo build --release -p lattice-inference --bin lattice --features f16
./target/release/lattice serve --model ~/.lattice/models/qwen3.5-0.8b --port 8080

# Metal GPU (native Q4 checkpoint) — see docs/q4-quantization.md for producing one
cargo build --release -p lattice-inference --bin lattice --features metal-gpu,f16
./target/release/lattice serve \
  --model ~/.lattice/models/qwen3.5-0.8b-q4-quarot \
  --tokenizer-dir ~/.lattice/models/qwen3.5-0.8b \
  --port 8080
```

The model directory's contents pick the backend automatically: a `model.safetensors` file (or
`.index.json` for a sharded checkpoint) selects the CPU path; the presence of any `*.q4` file
selects the Metal path (requires the `metal-gpu` feature at build time — without it, a Q4
directory is rejected with a clear error rather than silently falling back to CPU). `--model-id`
lets you set the identifier clients must send back in `"model"`; if omitted, it's derived from the
model directory's basename (`qwen3.5-0.8b`, `qwen3.5-0.8b-q4-quarot`, etc.).

Startup output:

```
Loading model from ~/.lattice/models/qwen3.5-0.8b...
Model loaded. Serving as 'qwen3.5-0.8b'.
Listening on 127.0.0.1:8080  (model: qwen3.5-0.8b, max_tokens default: 64)
  POST /v1/chat/completions
  GET  /health
```

That printed route list is exhaustive — this is the complete router:

```rust
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(1_048_576)) // 1 MiB request body cap
        .with_state(state)
}
```

There is no `/v1/models`, `/v1/completions`, or any admin/metrics endpoint. If you need a model
listing endpoint, that's `lattice_serve` (the other binary), not this one.

Shut down with Ctrl-C — it's a graceful shutdown, not an immediate kill:

```
^CShutdown signal received, draining connections...
```

(confirmed live: the process exits cleanly after printing this, rather than dropping in-flight
connections.)

## Auth, rate limiting, and concurrency

None of this is implemented today — worth stating explicitly, since issue #601 asks for it:

- **No authentication.** There is no API-key check, bearer-token check, or any other
  `Authorization` handling anywhere in the router — it's exactly the two routes plus the
  body-size layer shown above. Anyone who can reach the listening address can call it.
- **No rate limiting, no per-request admission control.** There is no request-count or
  concurrency-limiting middleware in front of the handlers. The only thing that rejects a request
  before it reaches model code is the 1 MiB body-size cap already shown above.
- **CPU backend: not serialized by the server, but not free either.** Each CPU request's
  `generate` call runs as blocking work on a Tokio blocking-pool task
  (`tokio::task::spawn_blocking`, `crates/inference/src/bin/lattice.rs`), so multiple CPU requests
  can execute concurrently up to Tokio's blocking-pool size — this is not a hard concurrency-1
  limit, but concurrent CPU requests still contend for the same CPU cores and memory.
- **Metal/Q4 backend: effectively concurrency-1.** All Metal generation is funneled through one
  dedicated worker thread (an `mpsc` channel into a single OS thread holding the `!Send` Metal
  state, `crates/inference/src/bin/lattice.rs`) — a deliberate design choice matching how one local
  GPU device actually works, not an oversight. Two concurrent requests against a Q4-backed
  `lattice serve` run back-to-back, not in parallel; the later request's connection simply stays
  open until its turn in the channel comes up.

None of this makes `lattice serve` unsafe to run locally or behind your own reverse proxy that adds
auth and rate limiting — it just means `lattice serve` itself provides neither, so don't expose it
directly to an untrusted network.

## `GET /health`

```
$ curl -s http://127.0.0.1:8080/health
{"status":"ok"}
```

Always 200 once the server is listening; there's no dependency check behind it (it doesn't verify
the model is still loadable/usable, just that the process is up and routing requests).

## `POST /v1/chat/completions` — non-streaming

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

Real response (captured against a running server in this repo):

```json
{
  "id": "chatcmpl-1783130141-0",
  "object": "chat.completion",
  "created": 1783130141,
  "model": "qwen3.5-0.8b",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "<think>\n\n</think>\n\nHello!" },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 16, "completion_tokens": 6, "total_tokens": 22 }
}
```

Notes on real fields you'll see:

- `id` is `chatcmpl-{unix_seconds}-{request_counter}` — the counter is a per-process
  `AtomicU64` that makes IDs unique across concurrent requests within the same second, not a
  globally unique/opaque token.
- `content` can include Qwen3.5's `<think>...</think>` reasoning block as a literal prefix in the
  message text — this server does not separate reasoning from the final answer into different
  response fields (no OpenAI-style `reasoning_content`); it's all one `content` string, exactly as
  the model emitted it. An empty `<think>\n\n</think>` before the real answer, as above, is normal
  for a simple prompt.
- `finish_reason` is `"stop"` when generation ended via EOS/stop-token/stop-string, `"length"` when
  the `max_tokens` budget was exhausted first. There is no `"content_filter"` or `"tool_calls"`
  value — those OpenAI finish reasons don't apply here.

### Request fields

```rust
pub struct ChatCompletionRequest {
    pub model: String,                              // required, must match the served model_id
    pub messages: Vec<Message>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,        // alias; must agree with max_tokens if both set
    pub temperature: Option<f32>,                    // default 0.7, range [0.0, 2.0]
    pub top_p: Option<f32>,                          // default 0.9, range (0.0, 1.0]
    pub stream: Option<bool>,
    pub stop: Option<Value>,                         // string, or array of 1-4 non-empty strings
    pub seed: Option<u64>,
    pub response_format: Option<ResponseFormat>,     // only {"type": "text"} accepted
    pub tools: Option<Value>,                        // rejected if present
    pub tool_choice: Option<Value>,                  // rejected if present
    pub logprobs: Option<bool>,                      // rejected if true (see "logprobs" below)
    pub n: Option<usize>,                            // rejected if > 1
}
```

`message.content` accepts either a plain string or an OpenAI-style content-parts array, but only
`{"type": "text", "text": "..."}` parts — an image/audio/file part is rejected with 400, not
silently dropped. `messages[].role` must be `"system"`, `"user"`, or `"assistant"`; `"tool"` and
`"developer"` are explicitly named and rejected (`"role 'tool' is not supported by this server"`);
anything else gets a generic `"unsupported role '...'"` message.

The struct's own doc comment on the `stream` field currently reads "SSE streaming — not yet
supported; rejected with 400" — **this is stale and wrong**. Streaming is fully implemented (see
below); the comment simply wasn't updated when it was added. Trust the behavior below (confirmed
both by the passing test `reject_unsupported_stream_true_ok` and by a live request), not that
comment.

### Validation order

Requests are validated in a fixed sequence; the first failure wins. Useful to know so you can
predict which error you'll get when more than one thing is wrong with a request:

1. JSON body parses and is under the 1 MiB limit.
2. `reject_unsupported`: `tools`/`tool_choice` present, `logprobs: true`, `n > 1`,
   `response_format.type != "text"`.
3. `model` matches the server's loaded model ID.
4. `messages` is non-empty.
5. The **last** message has role `"user"` (a Qwen ChatML constraint — the conversation must end on
   a user turn for the model to have something to respond to).
6. `max_tokens`/`max_completion_tokens`, `temperature`, `top_p` are all in range.
7. Every message renders into ChatML (role + content-part checks).
8. The rendered prompt's token count plus `max_tokens` fits the model's context window.
9. `stop` parses into valid stop strings.

### Rejected requests — exact error shapes

Every error response uses this envelope, live-verified for all three status codes:

```json
{ "error": { "message": "...", "type": "invalid_request_error", "code": "...", "param": null } }
```

`type` is `"invalid_request_error"` for both 400 and 413 responses, `"server_error"` for 500.
`param` is always `null` today — no field ever populates it. Some real examples:

```
$ curl -s -w '\n%{http_code}\n' http://127.0.0.1:8080/v1/chat/completions \
    -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"hi"}],"logprobs":true}'
{"error":{"message":"logprobs is not supported by this server","type":"invalid_request_error","code":"unsupported_feature","param":null}}
400
```

```
$ curl -s -w '\n%{http_code}\n' http://127.0.0.1:8080/v1/chat/completions \
    -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"hi"}],"n":2}'
{"error":{"message":"n > 1 is not supported","type":"invalid_request_error","code":"unsupported_feature","param":null}}
400
```

Malformed JSON (here, a syntax error — a missing `}`):

```
$ curl -s -w '\n%{http_code}\n' http://127.0.0.1:8080/v1/chat/completions \
    -d '{"model": "qwen3.5-0.8b", "messages": [ { "role": "user", "content": "hi" ] }'
{"error":{"message":"invalid JSON request body","type":"invalid_request_error","code":"invalid_request_body","param":null}}
400
```

The underlying parser error is logged server-side only
(`eprintln!("invalid request body: {}", ...)`) — it is deliberately never forwarded to the client.
For the malformed request above, the server's stderr shows:

```
invalid request body: Failed to parse the request body as JSON: messages[0].?: expected `,` or `}` at line 1 column 75
```

Note the literal backticks around `,` and `}` — that's serde_json's own error-message formatting,
not markdown. If you're debugging a malformed-request issue, check the server's stderr, not the
response body; the client only ever sees the generic `"invalid JSON request body"` message above.

A request body over 1 MiB gets HTTP 413, not 400:

```
$ curl -s -w '\n%{http_code}\n' http://127.0.0.1:8080/v1/chat/completions \
    --data-binary @big_payload.json
{"error":{"message":"request body exceeds 1 MiB limit","type":"invalid_request_error","code":"request_body_too_large","param":null}}
413
```

### `logprobs` — rejected today, in review for support

As of this writing, `"logprobs": true` is unconditionally rejected
(`"logprobs is not supported by this server"`, code `unsupported_feature`) — confirmed both by
source and by a live request above. **PR #620** (issue #585, still open/draft as of this writing)
adds real OpenAI-compatible `logprobs`/`top_logprobs` support to this exact endpoint's
non-streaming path, with `top_logprobs` range-checked to `0..=20` and required to pair with
`logprobs: true`. Two things worth knowing if you're reading this close to when that PR lands:

- Per that PR's own description, `stream: true` combined with `logprobs: true` will still be
  rejected with 400 even after it merges — streaming logprobs is explicitly out of scope for that
  change.
- That PR's own description also notes that the cross-turn prefix-cache generation path (see
  [`docs/cross-turn-cache.md`](cross-turn-cache.md)) doesn't populate logprobs and isn't reachable
  from this HTTP server anyway — not a regression from that change, just a documented gap.

If you're reading this after #620 has merged, verify the current behavior against
`reject_unsupported` in `lattice.rs` directly rather than trusting this paragraph — it will be
stale at that point.

## `POST /v1/chat/completions` — streaming (SSE)

Set `"stream": true`. The response is `text/event-stream`, one JSON object per `data:` line,
matching OpenAI's chunk sequence exactly: a role chunk first, then one chunk per text delta, then
a finish chunk with an empty `delta: {}`, then the literal `data: [DONE]`.

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"Count to 3."}],"max_tokens":24,"temperature":0.0,"stream":true}'
```

Real captured output (trimmed):

```
data: {"id":"chatcmpl-1783130154-1","object":"chat.completion.chunk","created":1783130154,"model":"qwen3.5-0.8b","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-1783130154-1","object":"chat.completion.chunk","created":1783130154,"model":"qwen3.5-0.8b","choices":[{"index":0,"delta":{"content":"<think>"}}]}

data: {"id":"chatcmpl-1783130154-1","object":"chat.completion.chunk","created":1783130154,"model":"qwen3.5-0.8b","choices":[{"index":0,"delta":{"content":"Here"}}]}

... (one chunk per token) ...

data: {"id":"chatcmpl-1783130154-1","object":"chat.completion.chunk","created":1783130154,"model":"qwen3.5-0.8b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

`ChunkDelta`'s `role`/`content` fields are both `#[serde(skip_serializing_if = "Option::is_none")]`
— that's why the finish chunk's `delta` serializes as the literal empty object `{}` rather than
`{"role":null,"content":null}`. The connection also gets axum's default SSE keep-alive (a comment
ping if 15 seconds pass with no event), so a slow prefill before the first token won't look like a
dead connection to an intermediate proxy.

Two caveats worth knowing before you build on this:

- **No disconnect-cancellation.** If the client disconnects mid-stream, generation keeps running
  to the `max_tokens` cap on the server — the dropped send errors are silently ignored, but nothing
  stops the underlying generation early. This is called out directly in the handler's own comment:
  "per-token backpressure / disconnect-cancellation is a future refinement." The separate
  `lattice_serve` daemon binary _does_ have this (a `CancelOnDrop`/`watch::channel` mechanism added
  in PR #552/#606) — `lattice serve` does not, as of this writing.
- **An internal generation failure mid-stream is invisible in the stream shape.** If the
  generation task errors out partway through (an engine panic path, an invariant violation), the
  server emits a normal-looking finish chunk (`finish_reason: "stop"`) followed by `[DONE]` —
  exactly what a successful completion looks like. The only trace of the failure is a server-side
  `eprintln!` (`"generation error (streaming): ..."` or `"generation invariant violation: ..."`).
  A client cannot distinguish "finished normally" from "failed partway through" from the SSE
  stream alone.

## Context window and token-budget limits

`lattice serve` enforces a **hardcoded `max_tokens_cap` of 4096** — this is not a CLI flag; it's a
literal in `main()`'s `Command::Serve` handling. A request asking for more is rejected:

```
{"error":{"message":"max_tokens 8000 exceeds server limit 4096","type":"invalid_request_error","code":"max_tokens_exceeds_limit","param":null}}
```

Separately, for the Metal/Q4 backend specifically, the usable context window is capped at
`MetalChatBackend::MAX_CACHE_LEN` (4096 tokens) regardless of the loaded model's actual
`max_position_embeddings` (Qwen3.5-0.8B's config reports 262144) — `lattice doctor` will show you
this cap directly (see [`docs/q4-quantization.md`](q4-quantization.md)). If your rendered prompt's
token count plus `max_tokens` exceeds the effective context window, you get:

```
{"error":{"message":"prompt (X tokens) plus max_tokens (Y) exceeds model context window (Z)","type":"invalid_request_error","code":"context_length_exceeded","param":null}}
```

The CPU (safetensors) backend doesn't have this particular cap — its `max_context()` comes from
the model's own config — but the 4096 `max_tokens_cap` still applies to both backends equally.

## A realistic multi-turn example

The server is stateless per request — there is no session/conversation ID, and (as covered in
[`docs/cross-turn-cache.md`](cross-turn-cache.md)) no cross-turn KV cache reuse either. Every
request must carry the full conversation history in `messages`, and every request re-prefills that
entire history from scratch:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [
      {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of France is Paris."},
      {"role": "user", "content": "And its population?"}
    ],
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

Feeding a prior assistant turn's `<think>...</think>` block back in as history is fine — the
server doesn't strip or special-case it; it's rendered into ChatML like any other assistant
message content. There is no requirement to strip reasoning blocks between turns.

## Summary

- `lattice serve` (not the separate `lattice_serve` binary) is the OpenAI-compatible server this
  document covers: `GET /health`, `POST /v1/chat/completions`, nothing else.
- Non-streaming and streaming (SSE) both work today; the request struct's doc comment claiming
  streaming is unsupported is stale — verify against `reject_unsupported` and its tests, not that
  comment.
- Every error is the OpenAI error envelope shape (`error.message`/`type`/`code`/`param`), with
  `code` distinguishing specific failure reasons; malformed-JSON parser detail and internal
  generation failures are logged server-side only, never returned to the client.
- `logprobs` is rejected today; PR #620 (in review) adds it non-streaming-only.
- `max_tokens` is hard-capped at 4096 server-wide; the Metal/Q4 backend additionally caps total
  context at 4096 regardless of the model's own configured maximum.
- No cross-turn caching, no disconnect-cancellation on the streaming path, and a mid-stream
  internal failure is indistinguishable from a normal finish in the SSE shape itself.
