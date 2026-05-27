# ADR-063 Strategic Positioning and Seed-Round Readiness Review

Verdict: REQUEST CHANGES
Findings: 4 High, 4 Medium, 1 Low

ADR-063 is directionally strong: `pull -> serve -> curl`, OpenAI/Anthropic compatibility, and Apple-Silicon-first benchmarking are the right seed-demo primitives. It is not seed-round ready as written because the competitor table misses the nearest Rust serving competitor, the scheduler estimate is materially under-scoped, the CLI/API contracts are not complete enough for implementation, and the benchmark targets are not anchored to measured MLX/llama.cpp baselines.

## Findings

### [High] Competitive table misses the closest Rust serving competitor

Evidence: `docs/adr/ADR-063-serving-architecture.md:36` starts the competitive table, and `docs/adr/ADR-063-serving-architecture.md:38-44` lists llama.cpp, Ollama, MLX-LM, vLLM/Inferact, SGLang/RadixArk, Candle, and lattice, but omits `mistral.rs` and Burn.

Why this matters: The `pure Rust` wedge is not defensible unless the ADR positions against `mistral.rs`, which now has a CLI, `run`, `serve`, OpenAI-compatible HTTP API documentation, model auto-detection, quantization, Metal support, Rust/Python SDKs, and a much broader product surface. Burn should also be acknowledged, but as a Rust tensor/deep-learning framework rather than a polished LLM serving product. The current table makes lattice look more differentiated than it is.

Suggested fix: Add rows for `mistral.rs` and Burn. For `mistral.rs`, explicitly compare against zero-config HF model loading, OpenAI-compatible serve, quantization breadth, multimodality, agent/tool features, and Metal support. Reframe lattice's wedge as narrower and more technical: verified/research-composable inference microkernel, QuaRot/Qwen/embedding specialization, CPU-first constraints, and a simpler auditable scheduler.

### [High] Scheduler effort estimate is not credible against ADR-048's own risk model

Evidence: `docs/adr/ADR-063-serving-architecture.md:963` estimates 3-4 days for the inference scheduler, and `docs/adr/ADR-063-serving-architecture.md:966` estimates 4-5 days for continuous batching. But ADR-048 says the current Metal `forward_step` synchronization "must change" and calls the migration "a non-trivial refactor" at `docs/adr/ADR-048-continuous-batching.md:276-281`. ADR-063 also requires cancellation, KV page allocation, per-request output channels, chunked prefill, decode priority, and Metal command-buffer integration at `docs/adr/ADR-063-serving-architecture.md:471-579`.

Why this matters: The 7-9 day total implies a near-trivial scheduler, while comparable serving loops usually take weeks to months once load shedding, cancellation, cache lifecycle, streaming correctness, and benchmarks are included. This weakens seed diligence because the plan can miss the promised demo date by a large factor.

Suggested fix: Split the plan into a hard v0.3 seed slice and a later scheduler track. A realistic v0.3 could be: single-model HTTP server, single-request decode, OpenAI streaming, one demo model, and benchmark harness. Treat continuous batching plus cancellation/KV lifecycle as a 6-10 week workstream, or explicitly justify why existing `crates/inference/src/batch/` code reduces that to less.

### [High] Backpressure contract contradicts the pseudocode

Evidence: ADR-063 states the bounded channel returns `503 Service Unavailable` immediately when full at `docs/adr/ADR-063-serving-architecture.md:273`, but the OpenAI handler uses `state.engine_tx.send(engine_req).await` at `docs/adr/ADR-063-serving-architecture.md:355-357`. Awaiting `send` waits for capacity; it does not immediately reject a full queue.

Why this matters: This is a load-shedding bug in the spec. Under overload, HTTP handlers can pile up waiting for queue capacity instead of preserving bounded latency and giving clients a clear retry signal.

Suggested fix: Specify `try_send` for immediate admission control, mapping `TrySendError::Full` to `503 queue_full` and `Closed` to `503 engine_not_running`. If fairness requires waiting, specify a small timeout and expose queue-delay metrics.

### [High] Benchmark targets are not seed-ready without same-hardware baselines

Evidence: ADR-063 defines benchmark tracks at `docs/adr/ADR-063-serving-architecture.md:784-797` and seed-demo targets for "Qwen-class 0.6B-0.8B Q4 on M2 Max" at `docs/adr/ADR-063-serving-architecture.md:834-848`, but it does not include measured llama.cpp, MLX-LM, Ollama, or mistral.rs baselines on the same machine, same prompt set, same model class, and same quantization.

Why this matters: The targets may be plausible, but they are not yet competitive claims. A 0.6B-0.8B Q4 demo is too small by itself for seed-round positioning unless it is paired with an Apple Silicon baseline matrix and at least one larger model tier, such as 4B/7B-class, where users feel real latency and memory constraints.

Suggested fix: Add a benchmark target table with: hardware, OS, framework version/commit, command line, model, quantization, prompt/decode lengths, concurrency, TTFT, prefill tok/s, decode tok/s, p95/p99 latency, peak memory, and thermal policy. Require "match or beat MLX-LM and llama.cpp on M2 Max for Qwen3-0.6B Q4; show credible slope on Qwen3-4B or 7B-class" before any investor-facing claim.

### [Medium] CLI tree is not complete enough to become a clap contract

Evidence: The command tree lists `pull`, `chat`, `complete`, `embed`, `serve`, `bench`, `info`, `quantize`, and `cache` at `docs/adr/ADR-063-serving-architecture.md:80-90`, but detailed flags are only specified for `pull`, `serve`, `chat`, and `bench` at `docs/adr/ADR-063-serving-architecture.md:107-181`. The v1 scope then includes `info` but not `complete`, `quantize`, or `cache` at `docs/adr/ADR-063-serving-architecture.md:918-929`.

Why this matters: Implementers do not have a stable clap subcommand tree. The ADR also lacks global flags needed across commands: `--cache-dir`, `--config`, `--backend cpu|metal|wgpu|auto`, `--device`, `--hf-token`, `--offline`, `--log-level`, `--json`, and `--model-dir`/registry override. `serve` also needs explicit knobs for CORS policy, request timeout, max batch tokens, prefill chunk size, shutdown/drain behavior, and whether OpenAI/Anthropic routes are strict or permissive.

Suggested fix: Add a Rust-shaped `Cli` / `Commands` sketch with global options and per-command structs. Mark deferred commands as hidden/unstable or remove them from the v0.3 tree. Update `bench --compare` to include `mistral.rs` once that competitor row is added.

### [Medium] API compatibility section is minimal, but not complete enough for SDK compatibility

Evidence: OpenAI streaming chunks are specified at `docs/adr/ADR-063-serving-architecture.md:612-634`, and that minimal shape is broadly correct for Chat Completions streaming. Anthropic support claims both streaming and non-streaming at `docs/adr/ADR-063-serving-architecture.md:684`, but only streaming event types are specified at `docs/adr/ADR-063-serving-architecture.md:693-702`. Anthropic's official stream also allows `ping` and `error` events and states that `message_delta.usage` is cumulative; ADR-063 only lists the happy-path events.

Why this matters: "OpenAI/Anthropic-compatible" is an SDK contract, not just a sample payload. Without non-streaming Anthropic response JSON, stream error events, ping behavior, and conformance tests against the official SDKs, demos with Claude Code, Aider, Continue, LangChain, or LlamaIndex can fail in integration details even when token streaming works.

Suggested fix: Add conformance fixtures for OpenAI streaming/non-streaming and Anthropic streaming/non-streaming. For Anthropic, define the non-streaming `Message` response, optional `ping`, `event: error`, cumulative usage semantics, and unknown-event tolerance. For OpenAI, keep `stream_options.include_usage` deferred only if tests prove target clients do not require it.

### [Medium] v0.3 boundary is not explicit, and prefix caching leaks into the seed scope

Evidence: ADR-063 declares a narrow seed path at `docs/adr/ADR-063-serving-architecture.md:940-942`, but `serve` exposes `--prefix-cache <MODE>` at `docs/adr/ADR-063-serving-architecture.md:148`, D8 specifies RadixAttention prefix cache at `docs/adr/ADR-063-serving-architecture.md:744-783`, and the benchmark targets include cached TTFT at `docs/adr/ADR-063-serving-architecture.md:840`. The v1 scope list at `docs/adr/ADR-063-serving-architecture.md:916-929` does not include prefix cache, and the 8-phase plan at `docs/adr/ADR-063-serving-architecture.md:956-968` has no prefix-cache phase.

Why this matters: The ADR says "every feature not on the critical path is a distraction," but then mixes core seed demo, structured output, speculative decoding, prefix caching, cross-framework benchmarking, and future research-platform positioning. That creates a plan that can sprawl well past v0.3 even if the table sums to 22-28 days.

Suggested fix: Add a hard "v0.3 seed boundary" section. Put only `pull`, `chat`, `serve`, OpenAI streaming, basic Anthropic streaming or defer Anthropic, `bench --suite serving`, one model, and one backend in v0.3. Move prefix caching, speculative serving, JSON schema response format, embeddings HTTP, and multi-client benchmark polish to v0.4+ unless they are required for the seed demo.

### [Medium] Prefix-cache page math conflicts with ADR-047 and code

Evidence: ADR-063's D8 uses `floor(longest_token_prefix / 256) * 256` at `docs/adr/ADR-063-serving-architecture.md:765-772`. ADR-047 explicitly keeps 256-token pages for the live `PagedKVCache` but introduces a `prefix_page_size` default of 64 tokens at `docs/adr/ADR-047-paged-kv-cache.md:133-135`; the code matches that with `DEFAULT_PREFIX_PAGE_SIZE: usize = 64` at `crates/inference/src/kv_cache/prefix.rs:111-112`.

Why this matters: The cached TTFT target and D8 design assume coarser 256-token matching than the accepted prefix-cache ADR and implementation. This can mislead both scheduler implementation and benchmark interpretation.

Suggested fix: Replace the hard-coded `/ 256` with `prefix_page_size`, default 64. If ADR-063 wants live-page-aligned matching for v1, explicitly override ADR-047 and explain the performance/correctness tradeoff.

### [Low] Chat-template override precedence is backwards

Evidence: `--chat-template <PATH>` is documented as an override at `docs/adr/ADR-063-serving-architecture.md:150`, but the load order places the explicit flag after tokenizer config, `chat_template.jinja`, and built-in fallback at `docs/adr/ADR-063-serving-architecture.md:727-732`.

Why this matters: An override should win deterministically. As written, the precedence order is ambiguous and easy to implement incorrectly.

Suggested fix: Put explicit `--chat-template` first, or state that the flag is applied after default resolution and always replaces the selected template.

## Accuracy Notes

- llama.cpp claims in the table are broadly accurate: its server docs list OpenAI-compatible routes, Anthropic Messages compatibility, embeddings, parallel decoding, continuous batching, schema-constrained JSON, function calling, and speculative decoding.
- MLX-LM claims are broadly accurate: it is Python/MLX, Apple-Silicon-focused, has generate/chat/server surfaces, and its own server docs warn it is not recommended for production because it only implements basic security checks.
- vLLM claims are directionally accurate but should be phrased as "GPU/datacenter-first" rather than "CUDA-only"; its docs include OpenAI-compatible serving, benchmarking, prefix caching, structured outputs, speculative decoding, CPU/TPU pages, and many production integrations.
- The market-validation funding claims were not fully reviewed in this pass; any round/valuation/ARR claims should get citations before the ADR is used externally.

## What I Checked

- Read `docs/adr/ADR-063-serving-architecture.md` end to end.
- Read `docs/adr/ADR-048-continuous-batching.md` for scheduler context and risk alignment.
- Spot-checked ADR-047 prefix-cache page-size contract and current `crates/inference/src/kv_cache/prefix.rs`.
- Checked current repo state for existing binaries, lack of a unified `lattice` clap binary, and existing batch scheduler primitives.
- Checked primary/current external sources for llama.cpp server, MLX-LM server, vLLM serving/features, Burn, mistral.rs, OpenAI Chat Completions, and Anthropic streaming.

## What I Did Not Check

- I did not run benchmarks or validate the Apple Silicon performance targets empirically.
- I did not validate all funding/market-validation claims.
- I did not run local CI; this was a strategy/spec review, not a code-change review.
- I did not post or edit GitHub issues #91-#94.

## Recommended Next Steps

1. Revise the competitive table first, especially `mistral.rs`, because it changes the strategic wedge.
2. Add a hard v0.3 seed boundary and move non-critical serving features out of the 8-phase plan.
3. Replace the scheduler estimate with a milestone plan that separates single-request HTTP serving from continuous batching/cancellation/KV lifecycle.
4. Add API compatibility fixtures using official OpenAI and Anthropic SDKs.
5. Add an Apple Silicon baseline matrix before presenting benchmark targets externally.

## External Sources Checked

- llama.cpp server README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- MLX-LM README and server docs: https://github.com/ml-explore/mlx-lm and https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md
- vLLM docs: https://docs.vllm.ai/en/latest/serving/online_serving/
- mistral.rs README/docs: https://github.com/EricLBuehler/mistral.rs and https://ericlbuehler.github.io/mistral.rs/
- Burn framework: https://github.com/tracel-ai/burn and https://burn.dev/
- OpenAI Chat Completions docs: https://developers.openai.com/api/reference/resources/chat
- Anthropic streaming docs: https://platform.claude.com/docs/en/build-with-claude/streaming

Domain utility: SKIPPED - `mcp__lore__suggest` / `compose` tools were not available in this session; I used the local `review-spec-alignment` rubric instead.
