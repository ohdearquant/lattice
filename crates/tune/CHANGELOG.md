# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
