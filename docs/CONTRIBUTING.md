# Contributing

## Requirements

- Rust stable (current release). AVX-512 kernels additionally require nightly.
- No system Python, ONNX Runtime, or external ML library required.

## Building

```sh
git clone https://github.com/ohdearquant/lattice
cd lattice
cargo build
```

## Running Tests

```sh
cargo test --workspace
```

Tests that require downloaded model weights are marked `#[ignore]`:

```sh
# Run including model-dependent tests (downloads weights on first run)
cargo test --workspace -- --include-ignored
```

## Running CI Locally

The full CI check sequence:

```sh
# Format
cargo fmt --all --check

# Lint (all lints must pass — correctness is set to deny)
cargo clippy --workspace --all-features -- -D warnings

# Tests
cargo test --workspace

# Doc tests
cargo test --doc --workspace
```

Or if a `Makefile` is present:

```sh
make ci
```

## Code Style

**Formatting**: `cargo fmt`. No exceptions. The CI check is `--check` mode.

**Lints**: `cargo clippy -D warnings`. The workspace `Cargo.toml` denies the entire
`correctness` lint group. `type_complexity`, `too_many_arguments`, and `large_enum_variant`
are warnings. Fix or add a targeted `#[allow]` with a comment explaining why.

**`unsafe` blocks**: Every unsafe block must have a comment immediately above it stating
the safety invariant being upheld. PRs that add unsafe without a justifying comment will
not be merged. See `docs/safety.md` for the full policy.

**BLAS-style function signatures**: `lattice-inference` kernel functions sometimes have
more than 7 arguments because grouping them into structs would require heap allocation in
hot paths. `#![allow(clippy::too_many_arguments)]` is set crate-wide for this reason.
Avoid adding many-argument functions outside the kernel layer.

**No stubs.** Do not merge placeholder implementations. If a function's body is not
ready, do not open the PR.

## Pre-commit Hook

To run the format and lint checks before every commit:

```sh
# .git/hooks/pre-commit
#!/bin/sh
cargo fmt --all --check && cargo clippy --workspace -- -D warnings
```

## PR Process

1. Fork and create a branch from `main`.
2. Keep PRs focused — one logical change per PR.
3. Run `cargo test --workspace` and `cargo fmt --all --check` locally before pushing.
4. PRs touching any `unsafe` block require a second reviewer (label: `unsafe-review`).
5. Update `docs/` if the change affects a public API, a supported model, or a benchmark.
6. Squash fixup commits before requesting review.

## Adding a New SIMD Kernel

SIMD kernels live in:

- `crates/embed/src/simd/` — distance ops (cosine, dot product, euclidean, normalize, binary, quantized)
- `crates/inference/src/forward/cpu/` — matmul, activation, norm, softmax, tiled variants
- `crates/inference/src/forward/neon.rs` — NEON-specific paths
- `crates/inference/src/forward/metal.rs` — Metal GPU dispatch

Steps:

1. Write the scalar fallback first and test it.
2. Add the SIMD path behind a runtime feature check (check `src/simd/tier.rs` in `embed`
   or the `cfg(target_arch)` patterns in `inference/forward/`).
3. Add a unit test asserting the SIMD and scalar paths produce values within `1e-4` of each
   other on the same input.
4. Add a Criterion benchmark.
5. Document the `unsafe` block with its safety invariant.

## Adding a New Embedding Model

See `docs/models.md §Adding a New Model` for the full checklist. Short version:

1. Add `BertConfig` or `QwenConfig` in `crates/inference/src/model/`
2. Wire the HuggingFace ID into `crates/inference/src/download.rs`
3. Add the `EmbeddingModel` variant in `crates/embed/src/model.rs`
4. Implement all match arms (`dimensions`, `max_input_tokens`, `query_instruction`, `model_id`)
5. Wire the new variant into `NativeEmbeddingService` in `crates/embed/src/service/native.rs`
6. Add `FromStr` patterns and a `Display` arm
7. Add dimension and round-trip tests

## Workspace Structure

```
lattice/
  Cargo.toml              — workspace manifest, shared dependencies, workspace lints
  crates/
    inference/            — transformer kernels (experimental, high churn)
      src/
        model/            — BertConfig, QwenConfig, cross_encoder
        tokenizer/        — WordPiece, SentencePiece, BPE
        weights/          — f32, f16, Q8, Q4 weight formats
        attention/        — standard, GQA, Flash, GDN
        forward/          — CPU (AVX2/NEON/tiled), Metal, WGPU
      benches/            — Criterion benchmarks
      examples/           — end-to-end bench_* programs
    embed/                — public embedding service
      src/
        model.rs          — EmbeddingModel enum + ModelConfig
        service/          — EmbeddingService trait, NativeEmbeddingService, CachedEmbeddingService
        simd/             — SIMD distance and normalize kernels
        backfill/         — re-embedding coordinator
        migration/        — zero-downtime model swap controller
      benches/
    fann/                 — fast tiny neural nets
      src/
        network.rs        — Network, NetworkBuilder
        layer.rs          — Layer
        activation.rs     — Activation enum
        training/         — BackpropTrainer, TrainingConfig
    tune/                 — training pipeline
      src/
        data/             — TrainingExample, Dataset
        distill/          — teacher API, DistillationPipeline
        train/            — TrainingLoop, Optimizer, JIT
        registry/         — ModelRegistry, RollbackController
        lora/             — LoraAdapter, LoraLayer
    transport/            — optimal transport math
      src/
        sinkhorn.rs       — balanced Sinkhorn
        sinkhorn_log.rs   — log-domain, epsilon-scaling
        unbalanced.rs     — KL-relaxed marginals
        barycenter.rs     — Wasserstein barycenters
        drift.rs          — embedding drift detection
      benches/
  docs/                   — this documentation
    architecture.md
    getting-started.md
    models.md
    benchmarks.md
    safety.md
    CONTRIBUTING.md
    adr/                  — Architecture Decision Records
```

## License

Apache-2.0. All contributions must be submitted under the same license.
