# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All lattice crates share one
workspace version; the workspace release notes are in
[`docs/releases/`](https://github.com/ohdearquant/lattice/tree/main/docs/releases).

## [Unreleased]

## [0.12.1] - 2026-10-14

Documentation and package metadata only; no code changes.

### Documentation

- crates.io keywords and categories; docs.rs builds the documentation with every feature enabled (#1936).
- README install snippets pin `0.12` (#1936).

## [0.12.0] - 2026-10-13

### Fixed

- `train_grad_full --gradcheck` refuses `--load` and `--save`, which it used to accept and ignore (#1894).

## 0.1.0 to 0.11.1

These releases are described in the workspace release notes, `docs/releases/v<version>.md`.
The entries below were recorded in this file under the Unreleased heading until 2026-08-11
and shipped in releases up to and including 0.10.0.

### Changed

- **Breaking (#1366):** `LoraConfig` gained a required public field, `dtype: String`,
  the tensor dtype label the adapter's in-memory `f32` buffers were converted from
  (e.g. `"f32"`, `"f16"`, `"bf16"`), matching the shared
  `lattice_fann::lora::LoraDescriptor::dtype` this crate now round-trips through.
  Since `lattice-tune` has not published a `0.9.0` release (the latest published
  version is `0.8.0`), this is a permitted break under `0.x` semver rather than a
  violation, but any `struct LoraConfig { .. }` literal written against `0.8.0`
  will not compile against `0.9.0` without adding the new field — there is no
  `Default` impl and the struct is not `#[non_exhaustive]`. Add
  `dtype: "f32".into()` (or the actual source dtype label) to each literal to
  migrate. Code built from a [`lattice_fann::lora::LoraDescriptor`] via
  `LoraConfig::from_descriptor` is unaffected.
