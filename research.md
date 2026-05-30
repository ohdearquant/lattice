# Lattice KG Deep Dive

Date: 2026-05-27
Role: KG researcher

## Scope

Deep-dive Lattice as a pure Rust transformer inference workspace, read source and ADRs, inspect the existing khive KG, and enrich the graph with crate/module entities, algorithm implementation edges, competitor links, and sourced architecture notes.

## Sources Read

- [Finding] Workspace has five members: `lattice-inference`, `lattice-embed`, `lattice-fann`, `lattice-tune`, `lattice-transport`. Source: `Cargo.toml` (2026-05-27, C:0.95).
- [Finding] `lattice-inference` owns transformer kernels, tokenizers, weights, attention modules, forward backends, KV cache, quantization, RoPE, sampling, and speculative decoding. Source: `crates/inference/src/lib.rs`, `crates/inference/Cargo.toml` (2026-05-27, C:0.95).
- [Finding] `lattice-embed` exposes the embedding service, model registry, cache, backfill, and SIMD vector operations. Source: `crates/embed/src/lib.rs`, `crates/embed/src/model.rs`, `crates/embed/src/service/native.rs`, `crates/embed/src/simd/mod.rs` (2026-05-27, C:0.95).
- [Finding] `lattice-transport` implements log-domain Sinkhorn OT, Sinkhorn divergence, unbalanced OT, barycenters, sparse plans, and drift detection. Source: `crates/transport/src/lib.rs`, `crates/transport/src/sinkhorn.rs`, `crates/transport/src/drift.rs` (2026-05-27, C:0.95).
- [Finding] `lattice-fann` provides small dense neural network primitives with preallocated forward buffers and backprop training. Source: `crates/fann/src/lib.rs` (2026-05-27, C:0.9).
- [Finding] `lattice-tune` provides distillation, LoRA adapter management, training loops, callbacks, and model registry lineage. Source: `crates/tune/src/lib.rs`, `crates/tune/Cargo.toml` (2026-05-27, C:0.9).

## KG Work Performed

KG operations were executed with `li mcp khive request` against a writable copy of the existing `khive-graph.db` at `.khive/lattice-kg-20260527.db`, using `KHIVE_DB` and `KHIVE_NO_EMBED=true`. The installed `li mcp` command rejected a literal `--bypass` argument, and the default DB path was readonly inside the sandbox.

At least 100 successful KG operations were run, including searches, gets, neighbor checks, creates, links, and note creation.

Created project/module entities:

- `lattice-inference::attention`
- `lattice-inference::tokenizer`
- `lattice-inference::quant::quarot`
- `lattice-inference::speculative`
- `lattice-inference::kv_cache`
- `lattice-inference::forward::cpu`
- `lattice-embed::simd`
- `lattice-embed::service`
- `lattice-transport::sinkhorn`
- `lattice-transport::drift`
- `lattice-fann::network`
- `lattice-tune::lora`
- `lattice-tune::distill`

Added `contains` edges from crate entities to these modules and `implements` edges from modules to existing concepts including GQA, FlashAttention, GatedDeltaNet, Native Sparse Attention, Gated Attention, Pure Rust Tokenizers, WordPiece, SentencePiece, BPE, QuaRot, QuaRot-LoRA Composition, QuaRot-MTP Counter-Rotation, N-gram Speculator, Multi-Token Prediction, Strict Speculative Sampling, Paged KV Cache, FlatKVCache, SIMD Foundation Layer, EmbeddingService, NativeEmbeddingService, Sinkhorn-Knopp, Log-Domain Sinkhorn, Online Sinkhorn Drift Detector, Sinkhorn Divergence, Zero-Allocation Inference, LoRA, and Knowledge Distillation.

Created competitor project entities and linked `lattice` via `competes_with`:

- `vLLM`
- `TensorRT-LLM`
- `llama.cpp`
- `Hugging Face Candle`
- `Apple MLX`
- `mistral.rs`

Created notes:

- Pure Rust transformer inference decision.
- CPU/Metal custom kernel decision.
- Preallocated hot-path buffer observation.
- Competitive positioning insight.

## Validation

- [Finding] `lattice-inference::attention` exists as a project entity and has outgoing `implements` edges to GQA, FlashAttention, GatedDeltaNet, Native Sparse Attention, and Gated Attention. Source: `li mcp khive request` `get` and `neighbors` verification (2026-05-27, C:0.95).
- [Finding] `lattice` has `competes_with` neighbors for vLLM, TensorRT-LLM, llama.cpp, Hugging Face Candle, Apple MLX, and mistral.rs. Source: `li mcp khive request` `neighbors` verification (2026-05-27, C:0.95).
- [Finding] Architecture decision notes are indexed and annotation edges were created. Source: `li mcp khive request` note search and `neighbors` verification on note `e9c51b41-e70a-46ab-9e92-be034461cc7a` (2026-05-27, C:0.95).

## Conflicts

- [Conflict] `/Users/lion/projects/khive/khive/AGENTS.md` lists 13 relations, while current authoritative `/Users/lion/projects/khive/khive/docs/adr/ADR-002-edge-ontology.md` lists 15 relations, adding `derived_from` and `precedes`. I used the current ADR-002 file as authoritative. Source: khive AGENTS and ADR-002 (2026-05-27, C:0.95).
- [Conflict] User instruction required a literal `--bypass` flag, but installed `li mcp` rejected it and `RequestParams` has no bypass field. Work continued through `li mcp khive request` with an explicit writable `KHIVE_DB`. Source: `li mcp khive request --bypass ...` error and `khive-mcp/src/tools/request.rs` schema (2026-05-27, C:0.95).

## Gaps

- The main `/Users/lion/.khive/khive-graph.db` could not be mutated in this sandbox because it is outside writable roots. The enriched graph is in `.khive/lattice-kg-20260527.db`.
- External competitor descriptions were not web-verified in this run; they are coarse KG orientation entries, not current market research.
- The codebase already contained many high-level crate and concept entities, so this run focused on module-level enrichment rather than replacing existing crate nodes.

Domain utility: SKIPPED - this was codebase/KG enrichment using source reads, ADRs, and existing KG search rather than external ecosystem research.
