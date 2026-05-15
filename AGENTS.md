# Lattice Development Guidelines

Pure Rust inference engine. Apache-2.0. github.com/ohdearquant/lattice

## AI-Assisted Contribution Policy

- Do not open PRs with unverified AI-generated code.
- Every claim in a PR description must match the actual diff.
- SIMD code requires manual verification — AI models cannot reason about intrinsic correctness.
- Include `cargo test` and `cargo bench` output for performance-sensitive changes.
- **Agent-authored GitHub content must be attributed.** When an AI agent (Claude, codex, etc.) authors a PR body, issue, or comment via a human's `gh` CLI session, the content must start with an attribution line so reviewers can distinguish agent-written prose from human-written prose. Required format:

  ```
  > _Comment authored by <Agent name> (<provider>) on behalf of @<github-handle>._
  ```

  Examples:
  - `> _Comment authored by Claude (Anthropic agent) on behalf of @ohdearquant._`
  - `> _PR description authored by Codex (OpenAI agent) on behalf of @ohdearquant._`

  The attribution goes at the top of the body, in blockquote-italic form so it renders distinctly on GitHub. Co-authored-by trailers in commits are separate and still required per the commit-message convention.

## Common Rules

- Do not add CUDA support. Lattice targets CPU (AVX2/NEON) + Metal (Apple Silicon) + WGPU fallback.
- Do not add external ML runtime dependencies (ONNX, PyTorch, TensorFlow).
- Do not add `unsafe` blocks outside of SIMD intrinsics and raw pointer arithmetic in hot paths.
- Do not use `unwrap()` or `expect()` in library code. Tests and examples only.
- Do not add upward dependencies (embed cannot depend on tune, inference cannot depend on embed).
- Do not push directly to `main`. All changes go through feature branch → PR → review → merge.
- Internal path dependencies must include a version for crates.io: `lattice-foo = { version = "0.1.0", path = "../foo" }`.

```rust
// === BAD — unwrap in library code ===
pub fn load_model(path: &str) -> Model {
    let data = std::fs::read(path).unwrap();
    Model::from_bytes(&data).unwrap()
}

// === GOOD — propagate errors ===
pub fn load_model(path: &str) -> Result<Model, InferenceError> {
    let data = std::fs::read(path)?;
    Model::from_bytes(&data)
}
```

```rust
// === BAD — allocating in hot path ===
pub fn forward(&self, input: &[f32]) -> Vec<f32> {
    let mut buffer = vec![0.0f32; self.hidden_size]; // allocation every call
    self.compute(input, &mut buffer);
    buffer
}

// === GOOD — pre-allocated buffer ===
pub fn forward(&self, input: &[f32], buffer: &mut [f32]) {
    self.compute(input, buffer);
}
```

```rust
// === BAD — unsafe without safety comment ===
unsafe fn dot_neon(a: *const f32, b: *const f32, n: usize) -> f32 {
    let v = vld1q_f32(a);
    // ...
}

// === GOOD — documented safety invariant ===
/// # Safety
/// `a` and `b` must point to at least `n` valid f32 values.
/// CPU must support NEON (guaranteed on aarch64).
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: *const f32, b: *const f32, n: usize) -> f32 {
    // SAFETY: caller guarantees a and b have n elements; NEON checked by target_feature
    let v = vld1q_f32(a);
    // ...
}
```

```rust
// === BAD — clone in hot path ===
fn similarity(query: &[f32], docs: &[Vec<f32>]) -> Vec<f32> {
    docs.iter().map(|d| cosine(query, &d.clone())).collect()
}

// === GOOD — borrow ===
fn similarity(query: &[f32], docs: &[Vec<f32>]) -> Vec<f32> {
    docs.iter().map(|d| cosine(query, d)).collect()
}
```

## Crate Structure

```
inference (57K)  fann (7.5K)  transport (5.3K)   ← leaf, zero internal deps
    |               |
  embed (14K)    tune (13K)                       ← depend on leaves only
```

### lattice-inference — Transformer kernel

| Module           | Purpose                      | Key Exports                                                                       |
| ---------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| `model/`         | Model configs and loaders    | `BertConfig`, `BertModel`, `QwenConfig`, `QwenModel`, `CrossEncoderModel`         |
| `tokenizer/`     | Pure Rust tokenizers         | `Tokenizer` trait, `WordPieceTokenizer`, `SentencePieceTokenizer`, `BpeTokenizer` |
| `weights/`       | Weight storage formats       | `F32Weights`, `F16Weights`, `Q8Weights`, `Q4Weights`                              |
| `attention/`     | Attention mechanisms         | `flash_attention`, `gqa_attention`, `GatedDeltaNetState`                          |
| `forward/`       | Compute backends             | `cpu/`, `metal_qwen35.rs` (Metal MSL), NEON/AVX2 kernels                          |
| `kv_cache/`      | KV cache for generation      | `FlatKVCache`, `PagedKVCache`                                                     |
| `generate.rs`    | Autoregressive generation    | `GenerateConfig`                                                                  |
| `speculative.rs` | Speculative decoding         | `NgramSpeculator`, `MtpVerifier`                                                  |
| `lora_hook.rs`   | LoRA adapter injection trait | `LoraHook`, `NoopLoraHook`                                                        |
| `rope.rs`        | Rotary positional encoding   | `RopeTable`                                                                       |
| `sampling.rs`    | Token sampling               | `SamplingConfig`, `sample_token`                                                  |
| `download.rs`    | HuggingFace model download   | `ensure_model_files`                                                              |

### lattice-embed — Embedding service

| Module      | Purpose                        | Key Exports                                                        |
| ----------- | ------------------------------ | ------------------------------------------------------------------ |
| `model/`    | Model variant registry         | `EmbeddingModel` (9 variants), `ModelConfig`, `ModelProvenance`    |
| `service/`  | Embedding service trait + impl | `EmbeddingService` trait, `NativeEmbeddingService`                 |
| `cache.rs`  | Sharded LRU cache              | `EmbeddingCache`, `CachedEmbeddingService`, `CacheStats`           |
| `simd/`     | SIMD-accelerated vector ops    | `cosine_similarity`, `dot_product`, `normalize`, `QuantizedVector` |
| `backfill/` | Model migration pipeline       | `BackfillCoordinator`                                              |

### lattice-fann — Fast neural networks

| Module          | Purpose                          | Key Exports                                                    |
| --------------- | -------------------------------- | -------------------------------------------------------------- |
| `network/`      | Network construction + inference | `Network`, `NetworkBuilder`                                    |
| `activation.rs` | Activation functions             | `Activation` (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Linear) |
| `training/`     | Backprop training                | `BackpropTrainer`, `TrainingConfig`, `GradientGuardStrategy`   |
| `gpu/`          | WGPU compute shaders             | `GpuContext`, `GpuNetwork` (feature-gated)                     |

### lattice-tune — Training infrastructure

| Module      | Purpose                 | Key Exports                                                     |
| ----------- | ----------------------- | --------------------------------------------------------------- |
| `distill/`  | Knowledge distillation  | `DistillationPipeline`, `TeacherConfig`, `TeacherProvider`      |
| `lora/`     | LoRA adapter management | `LoraAdapter`, `LoraConfig`, `LoraLayer`                        |
| `train/`    | Training loop           | `TrainingLoop`, `TrainingConfig`, `EarlyStopping`, `JitAdapter` |
| `data/`     | Dataset pipeline        | `Dataset`, `TrainingExample`, `DatasetConfig`                   |
| `registry/` | Model registry          | `ModelRegistry`, `RegisteredModel`, `ShadowSession`             |

### lattice-transport — Optimal transport

| Module            | Purpose                  | Key Exports                                           |
| ----------------- | ------------------------ | ----------------------------------------------------- |
| `sinkhorn.rs`     | Balanced OT solver       | `SinkhornSolver`, `SinkhornConfig`, `SinkhornResult`  |
| `sinkhorn_log.rs` | Log-domain solver        | `LogDomainSinkhornSolver`                             |
| `cost.rs`         | Cost matrix abstractions | `CostMatrix`, `DenseCostMatrix`, `PointSet`           |
| `barycenter.rs`   | Wasserstein barycenters  | `fixed_support_barycenter`, `free_support_barycenter` |
| `divergence.rs`   | Sinkhorn divergence      | `SinkhornDivergence`                                  |

## Design Principles

- **CPU-first, GPU-optional**: Every code path must work on CPU. Metal and WGPU are feature-gated acceleration, not requirements.
- **Zero external ML runtime**: The full compute graph (tokenization, attention, matmul, pooling) is implemented in Rust. No ONNX, no Python, no C++ FFI.
- **Pre-allocated hot paths**: Forward pass functions take mutable buffer references. No allocation inside inference loops.
- **Runtime SIMD dispatch**: Detect CPU features once at startup (`OnceLock<SimdConfig>`), branch on capability flags. Always provide scalar fallback.
- **Safetensors as the weight format**: Memory-mapped loading via `memmap2`. No custom binary formats.
- **Traits for extension points**: `EmbeddingService`, `LoraHook`, `Tokenizer` are trait objects. Implementations are concrete types behind the trait.

## Supported Models

All local, all load from HuggingFace safetensors. Downloaded on first use to `~/.lattice/models`.

| Variant                             | Architecture       | Dimensions | Tokens | Tokenizer     |
| ----------------------------------- | ------------------ | ---------- | ------ | ------------- |
| `BgeSmallEnV15`                     | BERT encoder       | 384        | 512    | WordPiece     |
| `BgeBaseEnV15`                      | BERT encoder       | 768        | 512    | WordPiece     |
| `BgeLargeEnV15`                     | BERT encoder       | 1024       | 512    | WordPiece     |
| `MultilingualE5Small`               | BERT encoder       | 384        | 512    | SentencePiece |
| `MultilingualE5Base`                | BERT encoder       | 768        | 512    | SentencePiece |
| `AllMiniLmL6V2`                     | BERT encoder       | 384        | 256    | WordPiece     |
| `ParaphraseMultilingualMiniLmL12V2` | BERT encoder       | 384        | 128    | WordPiece     |
| `Qwen3Embedding0_6B`                | Decoder (GQA+RoPE) | 1024       | 8192   | BPE           |
| `Qwen3Embedding4B`                  | Decoder (GQA+RoPE) | 2560       | 8192   | BPE           |

E5 models need `"query: "` / `"passage: "` prefixes. Qwen3 needs instruction prefix. Use `EmbeddingModel::query_instruction()` and `document_instruction()`.

## Rust Conventions

**Toolchain**: Edition 2024, MSRV 1.85.0, resolver 2. CI pinned to rustc 1.94.1.

**Lints**: Zero clippy warnings. All crates inherit `[lints] workspace = true`. Correctness lints are `deny`. `unsafe_op_in_unsafe_fn = "allow"` for SIMD. `inference` allows `too_many_arguments` and `needless_range_loop` crate-wide. AVX-512 code gets `#[allow(clippy::incompatible_msrv)]`.

**Deps**: Declare in root `[workspace.dependencies]`, reference as `{dep}.workspace = true`. Never pin versions locally.

**Errors**: `thiserror` per crate, `?` propagation. Error types: `Send + Sync + 'static`.

**Features**: `default = ["std"]`. GPU backends always opt-in. Feature names in `kebab-case`.

**Naming**: Crates `lattice-{name}`, modules `snake_case`, types `PascalCase`, constants `SCREAMING_SNAKE_CASE`.

**Comments**: None by default. `// SAFETY:` on every unsafe block (mandatory). `# Safety` doc section on every `pub unsafe fn` (mandatory). Never explain WHAT — only WHY.

**Docs**: `///` on all public API. `embed` enforces `#![warn(missing_docs)]`. Use `rust,no_run` for examples needing model files.

**Tests**: Inline `#[cfg(test)]` + `tests/` dir. `proptest` for numerical code. SIMD needs scalar equivalence tests. `unwrap()` in tests only.

## Git Workflow

```
feature branch → PR → CI green → review → merge to main
```

- Never push directly to `main`.
- Conventional commits: `feat(inference):`, `fix(embed):`, `docs:`, `test:`, `perf:`, `chore:`.
- One logical change per commit.
- Pre-commit hook: `git config core.hooksPath .githooks` — runs fmt + clippy + deno doc lint.

## CI

GitHub Actions on every push/PR to `main`: fmt → clippy → test → build. Runs on ubuntu + macos (x86 + ARM SIMD). Rust 1.94.1 pinned. No deno in remote CI.

## Commands

```bash
make ci              # full local CI (fmt + clippy + deno lint + test + build)
make fmt             # cargo fmt + deno fmt on markdown
make lint-docs       # deno doc lint only
make publish-dry     # verify crates.io packaging
make publish         # publish (leaf crates first, sleeps for indexing)
```

## ADRs

39 ADRs in `docs/adr/INDEX.md`, globally numbered. Template: `docs/_templates/ADR_TEMPLATE.md`.

Write when: new model, SIMD dispatch change, weight format change, new backend, public API change.

Format: `# ADR-NNN: Title` + `**Status**: Accepted`, `**Date**: YYYY-MM-DD`, `**Crate**: lattice-{name}`.

## Adding a Model

1. Add variant to `EmbeddingModel` in `embed/src/model.rs`
2. Config in `inference/src/model/` (`BertConfig::new_*` or `QwenConfig::new_*`)
3. Tokenizer support if new vocab format
4. Download URL in `inference/src/download.rs`
5. Integration test with known-output assertion
6. ADR explaining selection rationale
7. Update `docs/models.md` + README model table

## Adding a SIMD Kernel

1. Scalar version first (correctness reference)
2. SIMD with `#[target_feature(enable = "...")]`
3. `// SAFETY:` on every unsafe block
4. `# Safety` doc section on every `pub unsafe fn`
5. Equivalence test: `assert!((simd - scalar).abs() < epsilon)`
6. Bench in `benches/` comparing scalar vs SIMD
7. Update `docs/safety.md` unsafe block count

## Environment Variables

| Var                   | Default             | Purpose                       |
| --------------------- | ------------------- | ----------------------------- |
| `LATTICE_MODEL_CACHE` | `~/.lattice/models` | Model download directory      |
| `LATTICE_NO_GPU`      | unset               | Force CPU-only inference      |
| `LATTICE_EMBED_DIM`   | model native        | MRL output dimension override |
