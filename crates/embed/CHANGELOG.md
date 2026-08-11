# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `EmbeddingModel::max_instruction_bytes` returns the byte length of the model's
  longest retrieval instruction, and zero for symmetric models that take no
  prefix. A length guard that runs on already-prepared text sizes itself with
  this; a guard on caller text uses `MAX_TEXT_BYTES` unchanged.
- `EmbeddingService::embed_with_role` is the single entry point for role-aware
  embedding. It has a default body, so existing implementations of the trait
  continue to compile and behave as before without being edited.

### Changed

- The published text-length cap is now checked against the text the caller
  supplied, before the retrieval instruction is prepended. Previously a caller
  submitting text at exactly `MAX_TEXT_BYTES` could be rejected for bytes the
  service itself added. `embed_query` and `embed_passage` are now thin wrappers
  over `embed_with_role`, so all three role paths share one ordering decision.
- `CachedEmbeddingService` keys the caller's text together with the role tag
  rather than keying prepared text. This is equivalent, because the instruction
  is a function of the role and model configuration already present in the key.
- `TextTooLong` now reports the bound that actually rejected the input, so the
  reported maximum is one the caller can relate to their own input.
- The in-memory embedding cache key and `ModelProvenance::hash` are now derived
  with SHA-256 rather than BLAKE3, and the `blake3` dependency is dropped from
  the workspace. Both values keep their existing shape: a 32-byte cache key and
  a 64-character lowercase hex string.
  **This is a breaking change for `ModelProvenance::hash`.** `ModelProvenance`
  is documented as stable and is `Serialize`/`Deserialize`, and its `hash` is
  computed from its own persisted `model_id`, `loaded_at_iso`, and `model`
  fields via a published formula, so a consumer holding a serialized record
  can recompute the digest independently and compare it against `hash`.
  Recomputing with SHA-256 verifies records produced from this release
  onward. A record produced by an earlier release carries a BLAKE3 digest
  that SHA-256 recomputation will never match, and the serialized struct
  carries no explicit algorithm or version field.
  A consumer that keeps both formulas is not blocked by that: because all
  three inputs are themselves persisted fields, it can recompute both
  digests over the documented input and see which one matches `hash`,
  which classifies the record without any external release metadata. A
  consumer that supports SHA-256 alone cannot verify older records, and
  has to track which release produced each record out of band, re-derive
  the affected records, or stop verifying the older ones.
  The embedding cache key is not affected by that compatibility concern.
  Its construction is published (see "Key construction" in
  `crates/embed/docs/model.md`, repeated in `crates/embed/docs/design.md`),
  but the cache itself lives in memory for the life of the process, and its
  key scheme is explicitly documented as unstable and not to be persisted
  across sessions or process versions. A changed key therefore invalidates
  no cache-owned data and breaks no supported cross-version contract.
  Nothing stops an external caller from storing the public 32-byte key
  itself, but such a key is outside the documented contract and a stored
  one may simply miss after an upgrade.

#### Compatibility note for external implementors

The default `embed_with_role` reaches the backend by calling `embed`, and the
text it passes there is the prepared text, longer than the caller's by up to
`EmbeddingModel::max_instruction_bytes`. An external implementation that
enforces the exact `MAX_TEXT_BYTES` cap inside its own `embed` will therefore
still reject caller text at the cap.

This behavior is not new. `embed_query` and `embed_passage` have always applied
the instruction and then called `embed`, so such an implementation received
lengthened text in prior releases too, and nothing about that path changes here.
What changes is that the situation is now nameable and fixable: size the guard
with `max_instruction_bytes`, or override `embed_with_role` to reach the backend
directly. Both implementations in this crate override it.

Keeping `embed` the sole abstract method is a deliberate backward-compatibility
choice. Making `embed_with_role` abstract would deliver the cap unconditionally
but would break every existing implementor at compile time.

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2024-12-16

Initial release.

- Initial implementation of embedding generation
- SIMD-accelerated vector operations
- Local embedding support via fastembed (BGE-small default)
- LRU caching for embedding results with blake3 hashing
- Async embedding generation with Tokio runtime
- Benchmarks for SIMD operations and embedding performance
