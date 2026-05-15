# Architecture Decision Records

Global ADR index for the Lattice project. Numbered sequentially, grouped by crate.

## lattice-inference (ADR-001 to ADR-011, ADR-040 to ADR-044)

| ADR                                            | Title                                |
| ---------------------------------------------- | ------------------------------------ |
| [001](ADR-001-pure-rust-transformer-engine.md) | Pure Rust Transformer Engine         |
| [002](ADR-002-simd-dispatch.md)                | SIMD Dispatch Strategy               |
| [003](ADR-003-safetensors-loading.md)          | SafeTensors Weight Loading           |
| [004](ADR-004-kv-cache.md)                     | KV Cache Design                      |
| [005](ADR-005-tokenizer-architecture.md)       | Tokenizer Architecture               |
| [006](ADR-006-speculative-decoding.md)         | Speculative Decoding                 |
| [007](ADR-007-rope-positional-encoding.md)     | Rotary Positional Encoding (RoPE)    |
| [008](ADR-008-lora-injection.md)               | LoRA Injection via Trait Hook        |
| [009](ADR-009-model-architectures.md)          | Model Architectures (BERT and Qwen3) |
| [010](ADR-010-attention-mechanisms.md)         | Attention Mechanisms                 |
| [011](ADR-011-sampling-strategies.md)          | Sampling Strategies                  |
| [040](ADR-040-gated-attention.md)              | Gated Attention (G1 SDPA-Output)     |
| [041](ADR-041-differential-attention.md)       | Differential Attention               |
| [042](ADR-042-native-sparse-attention.md)      | Native Sparse Attention              |
| [043](ADR-043-lora-serving-verification.md)    | LoRA Serving Verification            |
| [044](ADR-044-quarot-rotated-quantization.md)  | QuaRot Hadamard-Rotated Quantization |

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

## lattice-tune (ADR-027 to ADR-034)

| ADR                                       | Title                           |
| ----------------------------------------- | ------------------------------- |
| [027](ADR-027-finetuning-pipeline.md)     | Fine-Tuning Pipeline            |
| [028](ADR-028-teacher-providers.md)       | Multi-Provider Teacher Strategy |
| [029](ADR-029-model-registry.md)          | Model Registry with Lineage     |
| [030](ADR-030-knowledge-distillation.md)  | Knowledge Distillation Pipeline |
| [031](ADR-031-lora-adapter-management.md) | LoRA Adapter Management         |
| [032](ADR-032-training-callbacks.md)      | Training Callbacks              |
| [033](ADR-033-jit-adaptation.md)          | JIT Adaptation                  |
| [034](ADR-034-dataset-pipeline.md)        | Dataset Pipeline                |

## lattice-transport (ADR-035 to ADR-039)

| ADR                                    | Title                             |
| -------------------------------------- | --------------------------------- |
| [035](ADR-035-sinkhorn-solver.md)      | Sinkhorn-Knopp Balanced OT Solver |
| [036](ADR-036-log-domain-stability.md) | Log-Domain Numerical Stability    |
| [037](ADR-037-cost-matrices.md)        | Cost Matrix Abstractions          |
| [038](ADR-038-barycenters.md)          | Wasserstein Barycenters           |
| [039](ADR-039-divergence-measures.md)  | Sinkhorn Divergence               |
