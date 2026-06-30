# ADR-ALIGNMENT-AUDIT: ADR Set Classification as of 2026-06-30

**Status**: Informational
**Date**: 2026-06-30
**Scope**: `docs/adr/` (65 numbered ADRs + `INDEX.md`)

Source ref: `origin/main` at `4772d5b` via `/Users/lion/khive-work/worktrees/docs-adr-alignment`.

Inventory basis: `../explorer/adr-inventory.md` listed 65 markdown files under `docs/adr/` on `origin/main`: 64 numbered ADRs plus `INDEX.md`. This change adds ADR-065 (the companion methodology ADR), bringing committed HEAD to 65 numbered ADRs plus `INDEX.md`. Statuses below are classified from source evidence or concrete source gaps, not ADR prose alone. If an ADR makes no current code-checkable claim beyond an implemented path, it is marked `current` with the relevant source path.

## Classification Table

| ADR | Title | Status | One-line Evidence |
|-----|-------|--------|-------------------|
| 001 | Pure Rust Transformer Engine | current | Core pure-Rust inference crate exists under `crates/inference/src/lib.rs`, with CPU/Metal backends and HuggingFace/safetensors loaders under `crates/inference/src/{forward,weights,model}`. |
| 002 | SIMD Dispatch Strategy | current | Runtime CPU SIMD dispatch exists in `crates/inference/src/forward/cpu/simd.rs`; NEON fallback/source exists in `crates/inference/src/forward/neon.rs`. |
| 003 | SafeTensors Weight Loading | current | SafeTensors loaders and storage formats exist in `crates/inference/src/weights/f32_weights.rs` and `crates/inference/src/weights/f16_weights.rs`. |
| 004 | KV Cache Design | current | `FlatKVCache` and `PagedKVCache` are exported from `crates/inference/src/kv_cache/mod.rs:13`; f16 flat-cache behavior is implemented in `crates/inference/src/kv_cache/flat.rs`. |
| 005 | Tokenizer Architecture | current | `Tokenizer` implementations remain in `crates/inference/src/tokenizer/`; added-token rendering shipped via `crates/inference/src/tokenizer/bpe.rs:49` and `common.rs:791`. |
| 006 | Speculative Decoding | current | `NgramSpeculator`, `MtpVerifier`, KV rollback, and GDN snapshot restore are implemented in `crates/inference/src/speculative.rs:488`, `1273`, and `1347`. |
| 007 | Rotary Positional Encoding (RoPE) | current | RoPE split-half convention is exercised by MTP and Qwen paths; source evidence includes `crates/inference/src/speculative.rs` plus Metal MTP gates at `crates/inference/src/forward/metal_qwen35.rs:9667`. |
| 008 | LoRA Injection via Trait Hook | stale | The base hook exists in `crates/inference/src/lora_hook.rs`, but the multi-adapter statement is stale: shipped decode mixture uses CPU pre-blend in `crates/inference/src/forward/metal_qwen35.rs:4301`, not hook chaining. |
| 009 | Model Architectures (BERT and Qwen3) | current | BERT, Qwen, and Qwen3.5 modules are present under `crates/inference/src/model/{bert.rs,qwen.rs,qwen35/}`. |
| 010 | Attention Mechanisms | superseded | Primitive attention modules exist, but the taxonomy is now owned by `AttentionKind` in `crates/inference/src/attention/mod.rs:57` per ADR-059. |
| 011 | Sampling Strategies | current | Sampling implementation is present in `crates/inference/src/sampling.rs` and Qwen3.5 sampling integration in `crates/inference/src/model/qwen35/sampling.rs`. |
| 012 | Runtime SIMD Detection Strategy | current | `lattice-embed` SIMD modules exist under `crates/embed/src/simd/`, including `dot_product.rs`, `cosine.rs`, and tiered quantized paths. |
| 013 | lattice-embed as SIMD Foundation Layer | current | `crates/embed/src/lib.rs` re-exports SIMD and embedding primitives; vector kernels are centralized under `crates/embed/src/simd/`. |
| 014 | Embedding Service | current | `EmbeddingService` and `NativeEmbeddingService` are present in `crates/embed/src/service/mod.rs` and `native.rs:63`. |
| 015 | Sharded LRU Embedding Cache | current | `EmbeddingCache` and cached service wrapper exist in `crates/embed/src/cache.rs:137` and `crates/embed/src/service/cached.rs:42`. |
| 016 | Embedding Model Variants | current | `EmbeddingModel`, model configs, and provenance live in `crates/embed/src/model.rs` and are re-exported from `crates/embed/src/lib.rs`. |
| 017 | NativeEmbeddingService Orchestration | current | `NativeEmbeddingService` owns model selection and generation in `crates/embed/src/service/native.rs:63` and implements `EmbeddingService` at `native.rs:310`. |
| 018 | Quantized Vector Tiers | current | `QuantizedVector`, `QuantizedData`, and tiered approximate search exist in `crates/embed/src/simd/quantized.rs:75` and `tier.rs:103`. |
| 019 | Backfill and Migration Pipeline | current | `BackfillCoordinator`, migration controller, and routing types exist in `crates/embed/src/backfill/coordinator.rs:77` and `migration/`. |
| 020 | Neural Network Primitives | current | `Network` and layer primitives are present in `crates/fann/src/network/mod.rs:90` and `crates/fann/src/layer.rs`. |
| 021 | Zero-Allocation Inference | current | `lattice-fann` exposes network inference with reusable state in `crates/fann/src/network/`; no contradictory source gap was found in the current fann surface. |
| 022 | Gradient Guard Strategy | current | `GradientGuardStrategy` exists and is wired into backprop at `crates/fann/src/training/gradient.rs:57` and `backprop.rs:313`. |
| 023 | Activation Functions | current | Activation primitives are implemented in `crates/fann/src/activation.rs` and re-exported through `crates/fann/src/lib.rs`. |
| 024 | Network Builder Pattern | current | `NetworkBuilder` exists in `crates/fann/src/network/builder.rs:29`. |
| 025 | GPU Backend | current | WGPU GPU backend modules exist under `crates/fann/src/gpu/` and are feature-gated from `crates/fann/src/lib.rs`. |
| 026 | Training Loop | current | Backprop training types and `TrainingConfig` are implemented under `crates/fann/src/training/`, including `backprop.rs` and `mod.rs`. |
| 027 | Fine-Tuning Pipeline | current | `lattice-tune` exports distill, train, data, registry, and LoRA modules from `crates/tune/src/lib.rs:137`. |
| 028 | Multi-Provider Teacher Strategy | current | `TeacherProvider` covers Claude/OpenAI/Gemini/Local/Custom at `crates/tune/src/distill/teacher/mod.rs:16`. |
| 029 | Model Registry with Lineage | current | `ModelRegistry`, `RegisteredModel`, and storage backends exist in `crates/tune/src/registry/storage/registry.rs:23` and `registry/mod.rs`. |
| 030 | Knowledge Distillation Pipeline | current | `DistillationPipeline`, `DistillationConfig`, and teacher config are implemented in `crates/tune/src/distill/pipeline/` and `teacher/`. |
| 031 | LoRA Adapter Management | current | `LoraAdapter`, PEFT/MLX safetensors load/save, loader guards, and manifest modules exist under `crates/tune/src/lora/`. |
| 032 | Training Callbacks | current | `TrainingCallback`, `LoggingCallback`, and callback wiring exist in `crates/tune/src/train/loop/callbacks.rs:8` and `loop/mod.rs:32`. |
| 033 | JIT Adaptation | current | `JitAdapter` and strategies are present in `crates/tune/src/train/jit.rs:195`, with known training placeholder behavior documented by ADR-056 status. |
| 034 | Dataset Pipeline | current | `Dataset`, `DatasetConfig`, stats, split, and batching live in `crates/tune/src/data/dataset.rs:13` and `164`. |
| 035 | Sinkhorn-Knopp Balanced OT Solver | current | `SinkhornSolver`, `SinkhornConfig`, and `SinkhornWorkspace` exist in `crates/transport/src/sinkhorn.rs:33`, `70`, and `370`. |
| 036 | Log-Domain Numerical Stability | current | Log-domain solver and stable helpers are present in `crates/transport/src/sinkhorn_log.rs` and `logsumexp.rs`. |
| 037 | Cost Matrix Abstractions | current | `CostMatrix`, `DenseCostMatrix`, and pairwise/closure costs exist in `crates/transport/src/cost.rs:255` and `268`. |
| 038 | Wasserstein Barycenters | current | Fixed/free support barycenter code is implemented in `crates/transport/src/barycenter.rs:23`, `89`, and `283`. |
| 039 | Sinkhorn Divergence | current | Debiased `SinkhornDivergence` and point-set helpers exist in `crates/transport/src/divergence.rs:34` and `84`. |
| 040 | Gated Attention (G1 SDPA-Output Gating) | current | Gated attention source exists in `crates/inference/src/attention/gated.rs`; taxonomy integration is in `attention/mod.rs:57`. |
| 041 | Differential Attention | current | Differential attention implementation exists in `crates/inference/src/attention/differential.rs`. |
| 042 | Native Sparse Attention | current | Native sparse attention implementation exists in `crates/inference/src/attention/native_sparse.rs`. |
| 043 | LoRA Serving Verification and Qwen3.5-0.8B Support | current | Qwen/Metal LoRA loading and verification paths exist in `crates/inference/src/forward/metal_qwen35.rs` plus `crates/tune/src/lora/safetensors.rs`. |
| 044 | QuaRot Hadamard-Rotated 4-bit Quantization | current | QuaRot converter, rotation, plan, and forward-equivalence code exist under `crates/inference/src/quant/quarot/`. |
| 045 | QuaRot + LoRA Composition at Inference Time | stale | Single-adapter QuaRot+LoRA shipped, but mixture and compatibility prose drifted: CPU pre-blend exists at `metal_qwen35.rs:4301`; manifest rev fields are stored at `manifest.rs:49` and `51` but not load-compared. |
| 046 | XGrammar Structured Output Engine | partially-implemented | Grammar engine and caps exist in `crates/inference/src/grammar/`, but `lattice_serve` sets `grammar: None` and does not expose `response_format` in `crates/inference/src/bin/lattice_serve.rs:117` and `202`. |
| 047 | Paged KV Cache with Prefix Reuse | superseded | Paged KV exists in `crates/inference/src/kv_cache/paged.rs:374`, but KV element-format ownership moved to f16/int8/int4 work in ADR-062 and `flat.rs` f16 paths. |
| 048 | Continuous Batching with Disaggregated Prefill/Decode | superseded | Batch modules exist in `crates/inference/src/batch/`, but GPU-worker/product serving ownership moved to ADR-063 and `crates/inference/src/bin/lattice_serve.rs`. |
| 049 | Vision Encoder Integration (Qwen-VL Path) | aspirational | No vision encoder module or Qwen-VL runtime path was found under `crates/inference/src/`; current source tree has text/Qwen/embedding model paths only. |
| 050 | Rejection Sampling for Speculative Decoding | current | Speculative verifier, MTP target trait, rollback, and acceptance code exist in `crates/inference/src/speculative.rs:1273` and following. |
| 051 | QuaRot-MTP Rotation Reconciliation | current | MTP partial RoPE and QuaRot/MTP reconciliation paths are exercised through `crates/inference/src/speculative.rs` and `crates/inference/src/quant/quarot/lora.rs`. |
| 052 | GDN State Management for Speculative Rollback | current | `GatedDeltaNetState::snapshot`/`restore_from` exist in `crates/inference/src/attention/gdn.rs:102` and `109`, and MTP restore is wired in `speculative.rs:1347`. |
| 053 | MoE Metal Dispatch with Expert Coalescing | current | MoE and Metal dispatch evidence exists in `crates/inference/src/model/qwen35/moe.rs` and `crates/inference/src/forward/metal_qwen35.rs`. |
| 054 | Rotation-Aware LoRA Training (RoLoRA Integration) | aspirational | Serving-side QuaRot LoRA exists in `crates/inference/src/quant/quarot/lora.rs`, but no RoLoRA training module was found under `crates/tune/src`. |
| 055 | Online Distribution Drift Detection via Sinkhorn Divergence | partially-implemented | `OnlineDriftDetector` is implemented and exported at `crates/transport/src/online_drift.rs:109` and `transport/src/lib.rs:119`; no inference `DriftSampler` bridge exists. |
| 056 | LoRA Tuning Pipeline | partially-implemented | Full `train/lora/` pipeline was not created, but bounded LoRA training shipped via `crates/tune/src/lora/train.rs:209` and `train_grad_full.rs`. |
| 057 | LoRA Full-Lifecycle Consumer API | current | Lifecycle gaps are closed by `save_peft_safetensors`, `adapt_step`, BERT hook, and docs in `crates/tune/src/lora/` and `crates/inference/src/model/cross_encoder.rs`. |
| 058 | CPU Performance Regression Tracking in CI | superseded | ADR status is superseded and current CI gate taxonomy lives in ADR-064 plus `.github/workflows/ci.yml`. |
| 059 | Composable Layer Architecture | partially-implemented | `AttentionKind` is implemented in `crates/inference/src/attention/mod.rs:57`; broader composable layer architecture remains proposed. |
| 060 | Pruning Toolbox | partially-implemented | D2 block influence and masks shipped (`crates/inference/src/pruning.rs:62`, `qwen35_config.rs:619`, `metal_qwen35.rs:5082`); D1/D3/D4/D5 remain absent. |
| 061 | Inference Metrics & Experiment Runner Infrastructure | partially-implemented | `LayerMetrics` and `ForwardMetrics` exist in `crates/inference/src/metrics.rs:30` and `58`; `MetricSink`/runner pieces remain proposed. |
| 062 | Metal FA2 Prefill + KV Cache Quantization Chain | partially-implemented | f16 `FlatKVCache` behavior exists in `crates/inference/src/kv_cache/flat.rs`, but FA2 prefill and int8/int4 KV chain remain proposed. |
| 063 | Serving Architecture --- CLI, HTTP Server, API Compatibility | stale | ADR says no server/HTTP listener, but `lattice_serve` exists with OpenAI-compatible routes and warm stateless reset at `crates/inference/src/bin/lattice_serve.rs:1`, `231`, and `677`. |
| 064 | CI Gate Taxonomy and Promotion Policy | current | `rustdoc-lint` and `unwrap-in-lib-lint` informational jobs are present in `.github/workflows/ci.yml:180` and `208`; the methodology ADR it anticipated was added as ADR-065 in this change. |
| 065 | Feature Promotion Gates --- Merge Measured Primitives, Not Research Ideas | current | Companion methodology ADR added in this change; encodes the G0--G4 promotion-gate ladder, the 8-question issue template, and the adaptive gating-vector pattern (`docs/adr/ADR-065-feature-promotion-gates.md`). |
| INDEX | Architecture Decision Records | current | Index lists the tracked ADR set in `docs/adr/INDEX.md`; inventory count matches `git ls-tree origin/main docs/adr` at 64 numbered ADRs plus `INDEX.md`. |

## Summary

| Status | Count |
|--------|------:|
| current | 50 |
| stale | 3 |
| superseded | 4 |
| partially-implemented | 7 |
| aspirational | 2 |

ADRs requiring implementer action (stale fact corrections and partial-status notes) are detailed in the accompanying update manifest. ADRs marked `partially-implemented` but already accurately status-noted are not necessarily all manifest entries; only those with drifted prose or no existing status note appear in the manifest.
