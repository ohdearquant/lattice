# Architecture Decision Records

Global ADR index for the Lattice project. Numbered sequentially, grouped by crate.

## lattice-inference (ADR-001 to ADR-011, ADR-040 to ADR-053)

| ADR                                            | Title                                                 |
| ---------------------------------------------- | ----------------------------------------------------- |
| [001](ADR-001-pure-rust-transformer-engine.md) | Pure Rust Transformer Engine                          |
| [002](ADR-002-simd-dispatch.md)                | SIMD Dispatch Strategy                                |
| [003](ADR-003-safetensors-loading.md)          | SafeTensors Weight Loading                            |
| [004](ADR-004-kv-cache.md)                     | KV Cache Design                                       |
| [005](ADR-005-tokenizer-architecture.md)       | Tokenizer Architecture                                |
| [006](ADR-006-speculative-decoding.md)         | Speculative Decoding                                  |
| [007](ADR-007-rope-positional-encoding.md)     | Rotary Positional Encoding (RoPE)                     |
| [008](ADR-008-lora-injection.md)               | LoRA Injection via Trait Hook                         |
| [009](ADR-009-model-architectures.md)          | Model Architectures (BERT and Qwen3)                  |
| [010](ADR-010-attention-mechanisms.md)         | Attention Mechanisms                                  |
| [011](ADR-011-sampling-strategies.md)          | Sampling Strategies                                   |
| [040](ADR-040-gated-attention.md)              | Gated Attention (G1 SDPA-Output)                      |
| [041](ADR-041-differential-attention.md)       | Differential Attention                                |
| [042](ADR-042-native-sparse-attention.md)      | Native Sparse Attention                               |
| [043](ADR-043-lora-serving-verification.md)    | LoRA Serving Verification                             |
| [044](ADR-044-quarot-rotated-quantization.md)  | QuaRot Hadamard-Rotated Quantization                  |
| [045](ADR-045-quarot-lora-composition.md)      | QuaRot + LoRA Composition                             |
| [046](ADR-046-structured-output.md)            | Structured Output (JSON Schema Constrained Decoding)  |
| [047](ADR-047-paged-kv-cache.md)               | Paged KV Cache                                        |
| [048](ADR-048-continuous-batching.md)          | Continuous Batching with Disaggregated Prefill/Decode |
| [049](ADR-049-vision-encoder.md)               | Vision Encoder                                        |
| [050](ADR-050-rejection-sampling.md)           | Rejection Sampling for Speculative Decoding           |
| [051](ADR-051-quarot-mtp-rotation.md)          | QuaRot-MTP Rotation Reconciliation                    |
| [052](ADR-052-gdn-speculative-state.md)        | GDN State Management for Speculative Rollback         |
| [053](ADR-053-moe-metal-dispatch.md)           | MoE Metal Dispatch with Expert Coalescing             |

## lattice-embed (ADR-012 to ADR-019)

| ADR                                        | Title                                |
| ------------------------------------------ | ------------------------------------ |
| [012](ADR-012-simd-strategy.md)            | Runtime SIMD Detection Strategy      |
| [013](ADR-013-simd-foundation.md)          | SIMD Foundation Layer                |
| [014](ADR-014-embedding-service.md)        | Embedding Service                    |
| [015](ADR-015-embedding-cache.md)          | Sharded LRU Embedding Cache          |
| [016](ADR-016-model-variants.md)           | Embedding Model Variants             |
| [017](ADR-017-native-embedding-service.md) | NativeEmbeddingService Orchestration |
| [018](ADR-018-quantized-vectors.md)        | Quantized Vector Tiers               |
| [019](ADR-019-backfill-pipeline.md)        | Backfill and Migration Pipeline      |

## lattice-fann (ADR-020 to ADR-026)

| ADR                                     | Title                     |
| --------------------------------------- | ------------------------- |
| [020](ADR-020-neural-net-primitives.md) | Neural Network Primitives |
| [021](ADR-021-zero-alloc.md)            | Zero-Allocation Inference |
| [022](ADR-022-gradient-guards.md)       | Gradient Guard Strategy   |
| [023](ADR-023-activation-functions.md)  | Activation Functions      |
| [024](ADR-024-network-builder.md)       | Network Builder Pattern   |
| [025](ADR-025-gpu-backend.md)           | GPU Backend               |
| [026](ADR-026-training-loop.md)         | Training Loop             |

## lattice-tune (ADR-027 to ADR-034, ADR-054, ADR-056 to ADR-057)

| ADR                                          | Title                                 |
| -------------------------------------------- | ------------------------------------- |
| [027](ADR-027-finetuning-pipeline.md)        | Fine-Tuning Pipeline                  |
| [028](ADR-028-teacher-providers.md)          | Multi-Provider Teacher Strategy       |
| [029](ADR-029-model-registry.md)             | Model Registry with Lineage           |
| [030](ADR-030-knowledge-distillation.md)     | Knowledge Distillation Pipeline       |
| [031](ADR-031-lora-adapter-management.md)    | LoRA Adapter Management               |
| [032](ADR-032-training-callbacks.md)         | Training Callbacks                    |
| [033](ADR-033-jit-adaptation.md)             | JIT Adaptation                        |
| [034](ADR-034-dataset-pipeline.md)           | Dataset Pipeline                      |
| [054](ADR-054-rolora-rotation-aware-lora.md) | Rotation-Aware LoRA Training (RoLoRA) |
| [056](ADR-056-lora-tuning-pipeline.md)       | LoRA Tuning Pipeline                  |
| [057](ADR-057-lora-consumer-api.md)          | LoRA Full-Lifecycle Consumer API      |

## lattice-transport (ADR-035 to ADR-039, ADR-055)

| ADR                                      | Title                               |
| ---------------------------------------- | ----------------------------------- |
| [035](ADR-035-sinkhorn-solver.md)        | Sinkhorn-Knopp Balanced OT Solver   |
| [036](ADR-036-log-domain-stability.md)   | Log-Domain Numerical Stability      |
| [037](ADR-037-cost-matrices.md)          | Cost Matrix Abstractions            |
| [038](ADR-038-barycenters.md)            | Wasserstein Barycenters             |
| [039](ADR-039-divergence-measures.md)    | Sinkhorn Divergence                 |
| [055](ADR-055-online-drift-detection.md) | Online Distribution Drift Detection |

## workspace / CI (ADR-058)

| ADR                                      | Title                                     |
| ---------------------------------------- | ----------------------------------------- |
| [058](ADR-058-cpu-perf-regression-ci.md) | CPU Performance Regression Tracking in CI |

## architecture / research (ADR-059 through ADR-065) — Proposed

| ADR                                                | Title                                                         | Status       | Depends on   |
| -------------------------------------------------- | ------------------------------------------------------------- | ------------ | ------------ |
| [059](ADR-059-composable-layer-architecture.md)    | Composable Layer Architecture                                 | Proposed     | ADR-010      |
| [060](ADR-060-pruning-toolbox.md)                  | Pruning Toolbox                                               | Proposed     | ADR-044, 059 |
| [061](ADR-061-inference-metrics-infrastructure.md) | Inference Metrics & Experiment Runner                         | Proposed     | ADR-059      |
| [062](ADR-062-metal-fa2-prefill.md)                | Metal FA2 Prefill + KV Cache Quantization                     | Proposed     | ADR-047      |
| [063](ADR-063-serving-architecture.md)             | Serving Architecture — CLI, HTTP, API Compat                  | Proposed     | ADR-048, 046 |
| [064](ADR-064-ci-gate-taxonomy.md)                 | CI Gate Taxonomy and Promotion Policy                         | Proposed     | ADR-058      |
| [065](ADR-065-feature-promotion-gates.md)          | Feature promotion gates: merge measured primitives, not ideas | Proposed     | ADR-064      |
| [066](ADR-066-output-correctness-gate-architecture.md) | Output-correctness gate architecture                      | Proposed     | ADR-064, ADR-065 |

## informational

| ADR                                         | Title                             | Status       |
| ------------------------------------------- | --------------------------------- | ------------ |
| [AUDIT](ADR-ALIGNMENT-AUDIT.md)             | ADR Alignment Audit (2026-06-30)  | Informational |
