# ADR-103: Splitting lattice-inference into focused crates behind a facade

**Status**: Proposed (2026-10-19)\
**Date**: 2026-10-19\
**Origin**: roadmap item "Inference crate decomposition" (docs/ROADMAP.md), 2026-10-18\
**Crate**: lattice-inference and the crates extracted from it; workspace manifests, publish script, CI\
**Related**: ADR-102 (in-crate reorganisation and the 3000-line file cap; complementary, see "Relation to ADR-102")

## Decision summary

`lattice-inference` is split into focused crates along module boundaries that already exist. `lattice-inference`
remains the published entry point: a facade that depends on the extracted crates and re-exports each one at the path
it has today, so every public `lattice_inference::...` path, type and feature keeps working. Shared vocabulary types,
starting with `InferenceError`, move to one small base crate so that extracted crates and the facade use the same types
rather than converting between copies. Every extraction is a zero-behaviour move with a fixed acceptance list. The
first extraction is the tokenizer, together with the base crate.

## Context

Measured at `main` b58ce287 with `wc -l` over each crate's `src/`:

| Crate               | Library source lines |
| ------------------- | -------------------: |
| `lattice-inference` |              353,172 |
| `lattice-tune`      |               25,698 |
| `lattice-embed`     |               22,554 |
| `lattice-fann`      |               13,516 |
| `lattice-transport` |                5,723 |

`lattice-inference` is about 84% of the workspace's library source. Its largest top-level modules are `forward`
(89,894 lines), `model` (57,355), `bin` (29,771), `weights` (22,509), `serve` (21,709), `attention` (16,348), `quant`
(15,691), `grammar` (13,595), `vision` (11,033), `decoder` (9,282) and `tokenizer` (8,243). Every change to any of them
recompiles the whole crate, review ownership is crate-wide, and a downstream user who needs one tokenizer compiles a
Metal inference engine.

Facts about the first candidate, the tokenizer module (`src/tokenizer/`, 13 files), read at the same commit:

- Outside its own files its non-test code uses one thing from the rest of the crate: `crate::error::InferenceError`.
  Its tests additionally use `crate::grammar` and `crate::model::qwen35_config`.
- About ten `pub(crate)` items are used by other modules of the crate, for example `IncrementalDetokenizer`,
  `detokenize`, `decode_tokens`, `token_ids_for_content`, `append_token_bytes`, `DEFAULT_BPE_MAX_SEQ_LEN`,
  `max_token_byte_span` and `tokenize_fragments_with_inserted_ids`.
- Two of those items are compiled only under feature combinations that belong to consumers (`serve` with
  `metal-gpu`; `serve`, `f16`, `router-learning` and `metal-gpu` together).
- `InferenceError` (`src/error.rs`, 104 lines) depends on nothing in the crate. It is `#[non_exhaustive]` and is
  documented as a stable type consumed by `lattice-embed`.

These are the four problems every extraction meets: a shared error type, crate-private items that cross the new
boundary, feature gates owned by consumers, and tests that exercise more than one module.

## Decisions

### D1. The facade keeps every public path

`lattice-inference` depends on each extracted crate and re-exports it where the module lives today, for example
`pub use lattice_tokenizer as tokenizer;` or a `pub mod tokenizer { pub use lattice_tokenizer::*; }` shim when the
facade module also holds items that stay behind. Re-exports, never copies, so a type reached through the old path and
the new crate is the same type. Crate-root re-exports (`pub use crate::...` in `lib.rs`) are kept unchanged.

### D2. Shared vocabulary lives in one base crate

`InferenceError` moves to a base crate (working name `lattice-inference-core`) and is re-exported at
`lattice_inference::error::InferenceError` and at the crate root. Extracted crates return that same type, so no
public signature changes. The base crate holds only types that two or more extracted crates need; it gains a type only
in the change that makes a second crate depend on it. It has no Metal, no I/O beyond what the moved types already use,
and no default features.

### D3. When a module is ready to move

A module is extractable when its non-test code depends only on the standard library, external crates, the base crate,
and crates already extracted. The check is a search of the module's non-test sources for `crate::` and `super::` paths
that leave it, with a known-positive control in the same run. Tests that exercise the module together with code that
stays behind (the tokenizer's grammar tests, for example) stay in `lattice-inference` as integration tests, unchanged
except for their paths.

### D4. Crate-private items that cross the boundary

Every `pub(crate)` item used from outside the moving module is listed in the extraction change with one of three
dispositions:

1. **Public**: it becomes a documented public item of the new crate, because it is genuinely useful there.
2. **Moves with its consumer**: the code that calls it moves into the new crate instead.
3. **Hidden**: `#[doc(hidden)] pub`, with a comment naming the consumer, for items that are implementation glue. Hidden
   items are not re-exported by the facade and are not part of the supported API.

No item is widened without a listed disposition.

### D5. Features

A feature that gates code inside a moved module becomes a feature of the new crate with the same name and meaning. The
facade's feature of that name forwards to it (`serve = [..., "lattice-tokenizer/serve"]`). An item gated on a
combination of consumer features is either made unconditional when it is plain CPU code with no extra dependencies, or
gated on a single feature of the new crate that the facade enables when all the consumer features are on. New crates
add no default features of their own; the facade's defaults are unchanged.

### D6. Acceptance of every extraction change

1. **Same tests.** The set of test names across the workspace (`cargo test --workspace --all-features -- --list`, plus
   the Metal feature set on Apple silicon) is identical before and after, allowing only the crate prefix of moved
   tests to change. The count is stated in the change.
2. **No public change.** `cargo semver-checks` on `lattice-inference` against its last published release reports no
   removed or changed item, and every crate's public API listing is attached to the change.
3. **Clippy and docs.** `cargo clippy --workspace --all-targets -D warnings` on the CI feature sets, and `cargo doc`
   for the facade with its docs.rs feature list, both clean.
4. **Measurement.** The bench-compare disposition follows the reachability rules in the repository's contribution
   guide. A moved module that a declared measurement target reaches gets a before/after comparison of that target.
5. **Pure move.** No logic, signature or name changes in the same change. Renames and API redesigns are separate
   changes after the move lands.

### D7. Dependency direction

Extracted crates may depend on the base crate and on each other, in one direction only. None depends on
`lattice-inference`. `lattice-embed` and `lattice-tune` keep depending on the facade; they may later depend on a
narrower crate in their own changes. A CI check over `cargo metadata` fails if any extracted crate depends on
`lattice-inference` or if the extracted crates form a cycle. The crate-ownership and publishing sections of the
contribution guide are updated in the same change that adds a crate.

### D8. Naming, versioning and publishing

New crates are named `lattice-<area>`, share the workspace version, and are published in dependency order before the
facade. The publish script's order is re-derived from the path dependencies in the change that adds a crate. Each new
crate carries a README, `package.metadata.docs.rs` features, and the workspace's lint and licence settings.

### D9. Order

1. Base crate plus tokenizer: the smallest real case, and it exercises D1 to D8 end to end.
2. Grammar, and weight and quantization formats.
3. The shared Metal runtime and per-model definitions, after ADR-102's in-crate split has brought the Metal files
   under the 3000-line cap, so that code is not cut twice.
4. Serving and the binaries last, since they depend on nearly everything.

Each step is its own change and lands only after the previous one is released or explicitly batched into one release.

## Relation to ADR-102

ADR-102 reorganises code inside `lattice-inference` and states that it introduces no new crate. That remains true of
ADR-102's own changes. This ADR is the separate decision to introduce crates, and it reuses ADR-102's acceptance
discipline (pure moves, measured targets, file cap). Where both apply to the Metal code, ADR-102's in-crate split comes
first.

## Alternatives considered

- **Keep one crate and rely on modules and the file cap.** Reviewability improves, but compile scope, dependency
  weight for narrow users, and crate-level ownership do not. Rejected as the end state; it is the interim state.
- **Split without a facade** (users import the new crates directly). Breaks every existing import and forces
  downstream changes for no behavioural gain. Rejected.
- **Per-crate error types with `From` conversions into `InferenceError`.** Changes the return type of every moved
  public function, which is a breaking change at the facade. Rejected in favour of D2.
- **One large split in a single change.** Unreviewable and not bisectable. Rejected in favour of D9's order.

## Consequences

- Downstream users see no change unless they opt into a narrower crate.
- The workspace gains crates to publish and version; the publish script and release notes cover them.
- Compile and review scope for tokenizer, grammar and format work shrinks to the crate that owns it.
- Hidden cross-crate items (D4.3) are a cost that should fall over time as consumers move.
