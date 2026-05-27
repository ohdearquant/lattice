# Gap Inventory — 2026-05-27

**Source**: KG exploration (lattice-inference `6c0a97df`, lattice `1c51f097`) + 33 open GitHub issues + codebase audit.
**Graph state**: lattice-inference has 90+ edges (implements 60+ concepts). Rich in attention/quantization/speculative concepts. Thin in serving/training/validation.

---

## I. Roadmap Gaps (researched-but-unbuilt, high downstream dependency)

### RG-1: Serving Stack (CLI + Daemon + HTTP API)

**Issues**: #91 (CLI), #92 (daemon), #93 (OpenAI API), #94 (Anthropic API)
**KG state**: Zero lattice project entities for serving. Concepts exist (Continuous Batching `1440a85a`, PagedAttention `c1d9f859`, Chunked Prefill `018193b3`) but no `implements` edge to any server code.
**Codebase**: 9 standalone binaries in `crates/inference/src/bin/` — no unified CLI, no server, no HTTP handler. Zero axum/actix/hyper references.
**Why it matters**: Without a server, lattice is a library. Every competitor (llama.cpp, MLX-LM, Ollama, vLLM) has `serve` as a first-class command. This is the #1 blocker for seed-round positioning — investors need a demo, not a crate.
**Downstream**: blocks benchmark comparisons (#84), blocks real user adoption, blocks khive's local-inference path.
**Score**: incoming_depends=8, implements=0 → **80** (highest priority)

### RG-2: f16 → int8 → int4 KV Cache Quantization Chain

**Issues**: #121 (f16), #118 (int8/int4), #120 (fused kernel), #122 (pre-RoPE K), #123 (WHT/SRFT rotation)
**KG state**: KV Cache Quantization `36aba582` exists with `implements` weight=0.7 (partial — Paged/Flat exist but no quantized storage). WHT `8f5a97ad` is `part_of` lattice-inference but the KV quant application is unbuilt.
**Codebase**: `kv_cache/flat.rs`, `kv_cache/paged.rs`, `kv_cache/prefix.rs` — all f32 only. No f16/int8/int4 storage paths.
**Why it matters**: f16 KV is a free 2x memory win (zero quality cost). The chain f16→int8→int4 enables 4-8x longer contexts or more concurrent sequences. Table-stakes for competitive serving (llama.cpp has `--cache-type-k q4_0`, vLLM has FP8 KV).
**Downstream**: blocks prefix cache wiring (#124), blocks radix tree (#127), blocks batch throughput bench (#119).
**Score**: incoming_depends=5, implements=0 → **50**

### RG-3: End-to-End LoRA Training Pipeline

**Issues**: #88 (e2e pipeline), #61 (adapt_step), #60 (SafeTensors save), #10 (JitAdapter placeholder), #11 (DistillationPipeline placeholder)
**KG state**: LoRA `0648de39` has `implements` weight=1.0 (load/apply exists). But SONA `b47da992`, Micro-LoRA `9e78216f`, adapt_step concept — all have zero implementing code. JIT Adaptation `cacdc201` explicitly flagged as placeholder.
**Codebase**: `train/jit.rs` — `train_step_cpu` returns synthetic loss, touches no weights. `lora/apply.rs` — inference only. Zero backward pass anywhere in the codebase.
**Why it matters**: khive's brain pack needs online gradient steps. Without training, lattice-tune is incomplete middleware. The LoRA ecosystem (load → apply → train → save → serve) is 60% built (load+apply+serve) and 40% missing (train+save).
**Score**: incoming_depends=4, implements=0 → **40**

### RG-4: Metal Flash Attention for Prefill

**Issues**: #126 (Metal FA2 prefill), #85 (MLX kernel study), #77 (GPU contention)
**KG state**: FlashAttention-2 `63602a7f` has `implements` weight=0.9 (CPU tiled path exists). Metal Fused Attention `48ee18b2` has `implements` weight=0.9 (decode-only). No prefill FA2 Metal entity.
**Codebase**: `metal_qwen35.rs:15546` lines — decode attention is fused, but prefill dispatches `(kv_heads, 1, 1)` threadgroups processing all seq_len rows per KV head. No Q-dimension tiling.
**Why it matters**: At context >2K-4K, attention dominates prefill cost. llama.cpp with `--flash-attn` beats MLX by 9 seconds at 8.5K context. This is the single biggest TTFT improvement available.
**Score**: incoming_depends=3, implements=0 → **30**

---

## II. Architectural Gaps

### Missing Layers

| Domain            | Concepts                                                                 | Projects                             | Gap                                         |
| ----------------- | ------------------------------------------------------------------------ | ------------------------------------ | ------------------------------------------- |
| serving           | 5 (daemon, OpenAI API, Anthropic API, SSE streaming, model registry)     | 0                                    | **CRITICAL** — no product surface           |
| training/backprop | 4 (backward pass, gradient computation, optimizer state, loss functions) | 0                                    | **HIGH** — LoRA apply without LoRA train    |
| benchmarking      | 3 (cross-framework, quality comparison, regression gate)                 | 0.5 (ADR-058 regression gate exists) | **MEDIUM** — can't prove competitive claims |

### Orphan Layers

- **Metal shaders** — 15,546 lines in a single file (`metal_qwen35.rs`). Issue #86 proposes extraction into `.metal` files. Currently untestable in isolation, unreviewable in diffs, unreusable across model architectures.
- **PrefixPageCache** — fully built data structure (#124 notes it's "complete with LRU, token-hash keyed, LoRA-namespaced") but never queried by the forward path. An orphan layer that produces zero benefit.
- **XGrammar Engine** — ADR-046, implemented but not exposed through any user-facing surface (no CLI `--format json`, no API `response_format`).

---

## III. Feature Direction Gaps (enables-into-void)

### Enables → Unbuilt

| Source (implemented)          | enables →          | Target (unbuilt)                    | Issue       |
| ----------------------------- | ------------------ | ----------------------------------- | ----------- |
| PagedKVCache                  | prefix reuse       | PrefixPageCache wiring into prefill | #124        |
| QuaRot (weight Q4)            | KV cache rotation  | WHT/SRFT rotation for KV int4       | #123        |
| Continuous Batching (ADR-048) | multi-user serving | HTTP server + daemon                | #92         |
| LoRA Hook Trait               | online adaptation  | adapt_step gradient API             | #61         |
| XGrammar Engine               | structured output  | API `response_format` param         | #93 Phase 2 |
| Speculative Decoding (MTP)    | throughput boost   | serving integration                 | #92         |

### Decision Debt (competes_with cliques, no winner chosen)

- **KV quantization approach**: Per-channel symmetric vs asymmetric min/max vs per-token — issue #118 lists options but no ADR decision
- **Metal shader organization**: include_str! vs build.rs metal-lib compilation vs runtime compilation — issue #86 lists options, no decision
- **LoRA training math**: Full autograd vs analytical LoRA-only gradients vs CPU-only placeholder — issue #88 lists but doesn't commit

---

## IV. Research Direction Gaps

### Papers Referenced in Issues but NOT in KG

| Paper                                                 | Referenced in | KG Status                                                       | Priority                                           |
| ----------------------------------------------------- | ------------- | --------------------------------------------------------------- | -------------------------------------------------- |
| arxiv:2605.05699 — Apple Silicon int4 KV with SRFT    | #123          | **MISSING**                                                     | P0 — directly applicable to lattice's Metal target |
| Sarathi-Serve (OSDI 2024, arxiv:2403.02310)           | #125          | **MISSING**                                                     | P1 — chunked prefill optimization evidence         |
| POD-Attention (ASPLOS 2025)                           | #125          | **MISSING**                                                     | P1 — energy-per-token at different chunk sizes     |
| KIVI (arxiv:2402.02750) — asymmetric 2-bit KV         | #118, #122    | **MISSING** — only snippet in KV Cache Quantization description | P1                                                 |
| KVQuant (arxiv:2401.18079) — pre-RoPE K quantization  | #122          | **MISSING** — only mentioned in issue                           | P1                                                 |
| Draw Things Metal FA2                                 | #126          | **MISSING** — open source reference implementation              | P0                                                 |
| RateQuant (arxiv:2605.06675) — optimal K/V bit budget | #123          | **MISSING**                                                     | P2                                                 |
| TurboQuant on MLX — WHT + 3-bit Lloyd-Max             | #123          | **MISSING**                                                     | P1                                                 |

### Narrow Framing (single approach, no alternatives recorded)

- **Metal attention**: Only FlashAttention-2 style considered. No evaluation of MPS/MPSGraph integration (Apple's private AMX path), no evaluation of metal-flash-attention PyPI package approach.
- **KV quantization rotation**: Only WHT mentioned. No comparison with Householder vs SRFT vs no rotation (KIVI approach works for some models without rotation).
- **Server framework**: Issues mention axum but no comparison with actix-web, tower-http, or hyper directly. No evaluation of Unix socket vs HTTP for local daemon.

---

## Frontier Ranking (Top 20)

| Rank | Concept / Work Item                      | Score | Reason                                                          |
| ---- | ---------------------------------------- | ----- | --------------------------------------------------------------- |
| 1    | **Unified CLI + Daemon** (#91, #92)      | 80    | Zero product surface. Blocks all user-facing demos.             |
| 2    | **OpenAI-compatible API** (#93)          | 75    | Blocks ecosystem integration (LangChain, Cursor, Continue).     |
| 3    | **f16 KV Cache Storage** (#121)          | 50    | Free 2x memory win. Foundation for entire quant chain.          |
| 4    | **Metal Flash Attention Prefill** (#126) | 30    | Biggest single TTFT improvement. Closes MLX gap.                |
| 5    | **PrefixPageCache wiring** (#124)        | 28    | Built but unwired. Enables system prompt caching.               |
| 6    | **Cross-framework bench suite** (#84)    | 25    | Evidence for investor pitch. Currently unsubstantiated claims.  |
| 7    | **Codex review follow-ups** (#116)       | 24    | 15 deferred items from embed show. CI gate gap is real.         |
| 8    | **Qwen3-Embedding divergence** (#103)    | 22    | Forward-pass bug, not tokenizer. Analyst investigation pending. |
| 9    | **LoRA SafeTensors save** (#60)          | 20    | Completes load/save roundtrip. Small scope, high value.         |
| 10   | **adapt_step gradient API** (#61)        | 20    | Unblocks khive brain pack. Minimal viable training.             |
| 11   | **Anthropic Messages API** (#94)         | 18    | Parallel to OpenAI API. Doubles ecosystem compatibility.        |
| 12   | **QuaRot step 3c binary** (#24)          | 18    | Offline conversion pipeline. QuaRot math done, binary missing.  |
| 13   | **Metal shader extraction** (#86)        | 16    | Prerequisite for sustainable kernel development.                |
| 14   | **Model-dependent chunk_size** (#125)    | 15    | Sarathi-Serve evidence: 1.5-2x prefill throughput at 1024.      |
| 15   | **Batch throughput benchmark** (#119)    | 14    | Validates continuous batching actually helps.                   |
| 16   | **MLX Metal kernel study** (#85)         | 14    | Identifies the 18%→27% bandwidth gap techniques.                |
| 17   | **E2E LoRA training pipeline** (#88)     | 12    | Large scope (~2 weeks). Needs backward pass engine.             |
| 18   | **Consolidated docs page** (#16)         | 10    | Lattice supports more than README shows. Signal loss.           |
| 19   | **Pre-commit clippy tightening** (#87)   | 8     | Small scope, prevents CI round-trips.                           |
| 20   | **Golden logit snapshot fix** (#31)      | 8     | 2x logit bug on main. Small but builds trust.                   |

---

## V. Research Platform Gaps (architecture exploration infrastructure)

Lattice's unique value is NOT "another inference engine" — it's a **composable architecture exploration platform**. 10 attention variants, QuaRot, GDN hybrid layers, speculative decoding — these are evidence of a pluggable research substrate. But the _infrastructure for exploration_ is missing.

### What exists

- 10 attention modules (`attention/{standard,gqa,flash,flash_causal,gdn,gdn_fused,gated,differential,native_sparse,decode}.rs`)
- `LayerType` enum with `Vec<LayerType>` per-layer dispatch (composable layer schedule)
- 17 Criterion benchmarks covering individual kernels
- `compute_perplexity()` on both CPU and Metal paths
- Strided perplexity evaluation with configurable window/stride

### RP-1: No Common Attention Trait (composability ceiling)

Each attention module is a standalone struct with a different API (`compute_attention`, `gdn_forward`, `gqa_attention`, etc.). There's no `trait Attention { fn forward(&self, q, k, v, mask) -> Output }` that lets you swap mechanisms without rewriting the forward pass. The `LayerType` enum dispatches between GDN and GQA, but adding a third option (e.g., NSA, Differential) requires editing the forward loop, not plugging in a new impl.

### RP-2: No Experiment Runner / Auto-Benchmarking

`make bench-compare` compares two git refs (A/B testing). But there's no way to say:

- "Run Qwen3.5-0.8B with [GDN×18, GQA×6] vs [GDN×12, NSA×6, GQA×6] and compare PPL + tok/s"
- "Sweep chunk_size from 256 to 2048, report prefill latency at each"
- "Compare f32 vs f16 vs int8 KV cache quality+speed"

This requires: config-driven model construction + automated evaluation + results database.

### RP-3: No Metrics Beyond PPL and tok/s

Missing: attention entropy, layer-wise activation norms, gradient flow statistics (when training exists), KV cache utilization, memory bandwidth utilization (roofline model), speculative decoding acceptance rate tracking, per-layer latency profiling.

These metrics are what enable architectural insight — "layer 12 has low attention entropy → candidate for linear attention replacement."

### RP-4: No Training-as-Exploration Loop

Architecture search needs: train small adapter → measure quality delta → compare configs. Without any training capability (#88), the exploration loop is: run inference → measure PPL. That's read-only exploration. Write-access (training) enables "does swapping layer 12 from GQA to GDN hurt quality if we LoRA-adapt the surrounding layers?"

### RP-5: No Experiment Tracking / Results Database

No way to record "config X with commit Y → PPL Z, tok/s W, memory M" persistently. Each bench run's Criterion output is ephemeral. The `perf-baselines` branch tracks CI baselines but not experimental configurations.

---

## Strategic Synthesis: Seed-Round Critical Path

```
Phase 1 (THIS SHOW): Product Surface
  CLI + Daemon + OpenAI API + bench-compare evidence
  → lattice becomes usable by humans, not just Rust authors

Phase 2: Memory Efficiency
  f16 KV → prefix cache wiring → int8 KV → fused attention kernel
  → competitive memory footprint, longer contexts

Phase 3: Performance Parity
  Metal FA2 prefill → MLX kernel techniques → chunk_size tuning
  → close the MLX/llama.cpp throughput gap with evidence

Phase 4: Training Loop
  adapt_step → SafeTensors save → E2E LoRA pipeline
  → complete LoRA lifecycle, unblock khive brain
```

Phase 1 is the show. Phases 2-4 are follow-up shows or issue-by-issue work.

**The pitch is NOT "faster inference engine."** The pitch is: **"lattice is the only pure-Rust composable research platform for transformer architecture exploration — with 10 swappable attention mechanisms, rotation-aware quantization, and production-grade Metal acceleration."** The serving stack makes it usable. The composability makes it unique. The metrics make it a research tool. The training loop makes it self-improving.
