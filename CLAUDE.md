# Lattice — Coding Conventions & Agent Guide

**Project**: Lattice — Pure Rust inference engine
**License**: Apache-2.0
**Repo**: github.com/ohdearquant/lattice

---

## Workspace Structure

```
lattice/
├── crates/
│   ├── inference/    lattice-inference   57K LOC   Transformer kernel (SIMD, GPU, tokenizers)
│   ├── embed/        lattice-embed       14K LOC   Embedding service (models, cache, similarity)
│   ├── fann/         lattice-fann        7.5K LOC  Fast tiny neural nets (zero-alloc forward)
│   ├── tune/         lattice-tune        13K LOC   Training (distillation, LoRA, registry)
│   └── transport/    lattice-transport   5.3K LOC  Optimal transport math (Sinkhorn, Wasserstein)
├── docs/             Top-level documentation
├── scripts/          CI/CD scripts
└── Cargo.toml        Workspace root
```

### Dependency Graph (strict — no cycles)

```
inference   fann   transport     (leaf crates — zero internal deps)
    |         |
    v         v
  embed     tune                 (depend on leaves only)
```

Breaking changes flow downward. Never add an upward dependency.

---

## Rust Conventions

### Edition & Toolchain

- **Edition**: 2024
- **MSRV**: 1.85.0
- **Resolver**: 2

### Formatting & Linting

```bash
cargo fmt --all                          # format
cargo clippy --workspace -- -D warnings  # lint (zero warnings policy)
make ci                                  # full pipeline: fmt + clippy + test + build
```

All crates inherit workspace lints via `[lints] workspace = true`.

### Workspace Lint Policy

```toml
[workspace.lints.clippy]
correctness = { level = "deny", priority = -1 }
needless_return = "warn"
redundant_closure_for_method_calls = "warn"
manual_let_else = "warn"
implicit_clone = "warn"
cloned_instead_of_copied = "warn"
uninlined_format_args = "warn"
type_complexity = "warn"
too_many_arguments = "warn"
large_enum_variant = "warn"

[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "allow"    # SIMD intrinsics in unsafe fns (edition 2024)
```

### Error Handling

- Use `thiserror` for library error types. Each crate has its own error enum.
- Never `unwrap()` or `expect()` in library code. Use `?` propagation.
- `unwrap()` is acceptable in tests and examples only.
- Error types must implement `std::error::Error + Send + Sync + 'static`.

### Unsafe Code

- Unsafe blocks are **allowed only for SIMD intrinsics and raw pointer arithmetic in hot paths**.
- Every `unsafe` block must have a `// SAFETY:` comment explaining the invariant.
- New unsafe code requires justification — prefer safe alternatives.
- Use `#[target_feature]` annotations on SIMD functions.
- Crate-level `#![allow(clippy::too_many_arguments)]` and `#![allow(clippy::needless_range_loop)]` are permitted in `inference` for BLAS-style kernel signatures.
- For function-level suppressions, use `#[allow(...)]` with a brief comment explaining why.

### Naming

- Crate names: `lattice-{name}` (kebab-case)
- Module names: `snake_case`
- Types: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Feature flags: `kebab-case` in Cargo.toml

### Dependencies

- Workspace dependencies are declared in the root `Cargo.toml` under `[workspace.dependencies]`.
- All crate Cargo.toml files reference `{dep}.workspace = true` — never pin versions locally.
- Adding a new external dependency requires justification. Prefer zero-dep or `#[no_std]`-compatible crates.
- Never add a dependency from a higher crate to a lower one (embed → inference is OK, inference → embed is NOT).

### Feature Flags

- `default = ["std"]` for most crates.
- GPU backends (`metal-gpu`, `wgpu-gpu`) are always opt-in.
- Optional internal dependencies use feature gates (e.g., `native` in embed enables inference).
- Test-only features use the `dev-dependencies` section, not feature flags.

---

## Code Style

### Comments

- Default: no comments. Code should be self-documenting.
- Write a comment only when the WHY is non-obvious: a safety invariant, a performance trick, a workaround.
- Never explain WHAT the code does if a good name already does that.
- `// SAFETY:` comments on every unsafe block (mandatory).

### Documentation

- All public types, traits, and functions must have `//!` or `///` doc comments.
- `lattice-embed` enforces `#![warn(missing_docs)]`.
- Doc comments should include a one-line summary, then optionally a longer explanation.
- Include `# Examples` in doc comments for key public API functions.
- Use ````rust,no_run` for examples that require model files.

### Tests

- Unit tests go in `#[cfg(test)] mod tests` within the source file.
- Integration tests go in `tests/` directory per crate.
- Test names: `test_{what}_{expected_behavior}` (e.g., `test_cosine_similarity_unit_vectors`).
- Use `proptest` for property-based testing of numerical code.
- SIMD code must have scalar equivalence tests (same input → same output ± epsilon).

### Performance-Critical Code

- Hot-path functions use pre-allocated buffers — never allocate in a loop.
- Prefer `&[f32]` slices over `Vec<f32>` in function signatures.
- SIMD dispatch: runtime feature detection, with scalar fallback.
- Benchmark before and after changes to hot paths: `cargo bench -p lattice-{crate}`.
- No `clone()` in hot paths — borrow or copy.

---

## Git & PR Workflow

### Branches

- `main` is the release branch. Always clean, always builds.
- Feature branches: `{topic}` (e.g., `add-bge-m3-model`, `simd-avx512`).
- No long-lived branches.

### Commits

- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `perf:`, `chore:`.
- Scope is the crate name: `feat(inference): add Qwen3.5 MoE support`.
- One logical change per commit. Don't mix refactoring with features.

### Pre-Commit

The `.githooks/pre-commit` hook runs `cargo fmt --check` and `cargo clippy -- -D warnings`. To install:

```bash
git config core.hooksPath .githooks
```

### CI

CI runs on every push to `main` and every PR:

- Format check
- Clippy with `-D warnings`
- Tests on Ubuntu + macOS (covers x86 + ARM SIMD)
- Release build

Run locally before pushing: `make ci`

---

## ADR (Architecture Decision Records)

ADRs document significant technical decisions. They live in each crate's `docs/adr/` directory.

Template: `docs/_templates/ADR_TEMPLATE.md`

When to write an ADR:

- Adding a new model architecture
- Changing SIMD dispatch strategy
- Modifying the weight format or loading mechanism
- Adding a new hardware backend
- Changing the public API surface

---

## Adding a New Model

1. Add the variant to `EmbeddingModel` enum in `embed/src/model.rs`
2. Implement `BertConfig::new_variant()` or `QwenConfig::new_variant()` in `inference/src/model/`
3. Add tokenizer support if needed (new vocab format)
4. Add download URL mapping in `inference/src/download.rs`
5. Write integration test with known-output vector assertion
6. Write ADR explaining model selection rationale
7. Update `docs/models.md` and the README model table

---

## Adding a New SIMD Kernel

1. Implement scalar version first — this is the correctness reference
2. Add SIMD version with `#[target_feature(enable = "...")]`
3. Add `// SAFETY:` comment on every unsafe block
4. Write equivalence test: `assert!((simd_result - scalar_result).abs() < epsilon)`
5. Add benchmark in `benches/` comparing scalar vs SIMD
6. Update `docs/safety.md` unsafe block count

---

## Environment Variables

| Variable              | Default             | Description                               |
| --------------------- | ------------------- | ----------------------------------------- |
| `LATTICE_MODEL_CACHE` | `~/.lattice/models` | Model download directory                  |
| `LATTICE_NO_GPU`      | unset               | Set to `1` to force CPU-only inference    |
| `LATTICE_EMBED_DIM`   | model default       | Override embedding output dimension (MRL) |

---

## Quick Reference

```bash
make ci              # full local CI (fmt + clippy + test + build)
make fmt             # format all code
make clippy          # lint check
make test            # run all tests
make build           # release build
make publish-dry     # verify crates.io packaging
make publish         # publish to crates.io (leaf crates first)
```
