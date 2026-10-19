# ADR-103: Splitting lattice-inference into focused crates behind a facade

**Status**: Proposed (2026-10-19)\
**Date**: 2026-10-19\
**Origin**: roadmap item "Inference crate decomposition" (docs/ROADMAP.md), 2026-10-18\
**Crate**: lattice-inference and the crates extracted from it; workspace manifests, publish script, CI\
**Related**: ADR-102 (in-crate reorganisation and the 3000-line file cap; complementary, see "Relation to ADR-102")

## Direction

The project maintainer's direction, 2026-10-18, quoted as given:

> add to roadmap, inference crate decomposition, will do those in a harmless fashion in high level public API level,
> inference crate itself might be the aggragator interface for acceessing stuff. right now inference crate is huge

and, on the tokenizer as the first extraction:

> if you think that's reasonable enough of a scope, sure., we probably want some kind of standard around these,

This ADR is that standard. "Harmless at the public API level" is D1 and D6, "the aggregator interface" is D1 and D7,
"tokenizer first" is D9.

## Decision summary

`lattice-inference` is split into focused crates along module boundaries that already exist. `lattice-inference`
remains the published entry point: a facade that depends on the extracted crates and re-exports their items at the
paths they have today, so every public `lattice_inference::...` path, type and feature keeps working. `InferenceError`
moves to one small base crate so that extracted crates and the facade share one error type. Every extraction is a
zero-behaviour move with a fixed readiness check and a fixed acceptance list. The first extraction is the tokenizer,
together with the base crate.

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

Facts about the first candidate, the tokenizer module (`src/tokenizer/`), read at the same commit:

- Outside its own files its non-test code uses one thing from the rest of the crate:
  `crate::error::InferenceError`. Its tests additionally use `crate::grammar` and `crate::model::qwen35_config`, and
  two doc comments link into the rest of the crate (`bpe.rs`, links to `crate::model::qwen35` and `crate::grammar`).
- Crate-private items used from other modules include `BpeTokenizer::newline_boundary_table` (called from
  `decoder/qwen_metal.rs` and `model/qwen35/generation.rs`), `token_ids_for_content`, `max_token_byte_span`,
  `tokenize_fragments_with_inserted_ids`, `DEFAULT_BPE_MAX_SEQ_LEN` and `IncrementalDetokenizer`.
  `detokenize::bytes_to_unicode` is already public through `lattice_inference::model::qwen35`.
- Three of those items are compiled only under feature combinations that belong to consumers:
  `max_token_byte_span` (`serve`, `f16`, `router-learning`, `metal-gpu` on macOS, or `test`),
  `tokenize_fragments_with_inserted_ids` (`metal-gpu` and `serve`), and `IncrementalDetokenizer::push` (`metal-gpu` on
  macOS, or `test`). CPU test suites that run without `metal-gpu` call `push` and `max_token_byte_span`.
- `InferenceError` (`src/error.rs`) depends on nothing in the crate and is `#[non_exhaustive]`. Two trait impls on it
  live in other modules: `From<GrammarError>` (`grammar/engine.rs`) and `From<PredictionError>` (`decoder.rs`).

Two tool facts shape the acceptance rules. Both were measured with `cargo-semver-checks` 0.50.0 on a minimal workspace
that moves a struct, a function and an error enum out of a crate and re-exports them from it:

- `cargo semver-checks` reports every moved item as removed (`struct_missing`, `function_missing`, `enum_missing`)
  whether the facade re-exports them by name with `#[doc(inline)]` or by glob. Controls in the same run behaved
  correctly: identical code passed, and a really removed function failed. Re-exports from another crate are not
  followed, so the facade check can neither pass on an extraction nor see moved items afterwards.
- The repository's semver job checks default features on Linux only, and under 0.x a workspace minor bump makes it
  skip every check (this is already disclosed by the job itself).

## Decisions

### D1. The facade keeps every public path

`lattice-inference` depends on each extracted crate and keeps each module at its current path as a small shim that
re-exports named items:

```rust
pub mod tokenizer {
    #[doc(inline)]
    pub use lattice_tokenizer::{bpe, common, BpeTokenizer, Tokenizer, TokenizedInput /* ... the full current list */};
}
```

The list is exactly today's public surface of that module, and the crate-root re-exports in `lib.rs` keep their
current names. Re-exports, never copies, so a type reached through the old path and the new crate is the same type.
The facade uses neither a whole-crate alias (`pub use lattice_tokenizer as tokenizer;`) nor a glob
(`pub use lattice_tokenizer::*;`): both forward everything the new crate makes public, including the internal items
D4 keeps out of the supported API, and a glob can collide with another glob.

### D2. `InferenceError` lives in one base crate

The base crate is `lattice-inference-core` (`lattice-core` is taken on crates.io by an unrelated crate). It is created
in the first extraction and holds `InferenceError`, re-exported at `lattice_inference::error::InferenceError` and at
the crate root. Extracted crates return that same type, so no public signature changes. The tokenizer crate does not
own it, because the next candidates (weight and quantization formats, grammar) also return it and have no reason to
depend on a tokenizer.

Rules for the base crate:

1. No dependency on any other lattice crate, no Metal, no default features.
2. `InferenceError` variants carry only standard-library or primitive payloads, never a type from an extracted crate
   (that would create a cycle). A crate with a richer error defines its own error type and implements
   `From<ItsError> for InferenceError`, as `grammar` already does by converting to a string.
3. A new variant is a change to the base crate first, released with the crate that needs it.
4. Any type other than `InferenceError` enters the base crate only in the change that makes a second extracted crate
   need it.

### D3. When a module is ready to move

A module is extractable when all of these hold, each checked by a search with a known-positive control in the same
run:

1. **Outbound paths.** Its non-test sources, doc comments included, reach outside it only into the standard library,
   external crates, the base crate and crates already extracted. Doc links to code that stays behind become plain code
   spans in the move. Self-paths such as `crate::tokenizer::...` are rewritten to the new crate root.
2. **Inbound impls.** No code that stays behind holds an inherent `impl` of a moving type (it stops compiling once
   the type lives in another crate), and every trait impl on a moving type that stays behind still satisfies the
   orphan rule. Inherent impls split across modules exist in the crate today (`impl Qwen35Model` in
   `forward/batch_prefill.rs`, `impl PreparationHandle` in three files), so this check is not hypothetical for later
   steps.
3. **Field access.** No code that stays behind reads or writes a private or crate-private field of a moving type.

Tests that exercise the module together with code that stays behind (the tokenizer's grammar tests, for example) stay
in `lattice-inference` as integration tests, unchanged except for their paths.

### D4. Crate-private items that cross the boundary

Every `pub(crate)` item of the moving module that is used from outside it is listed in the extraction change with one
of three dispositions, in this order of preference:

1. **Stays crate-private** in the new crate, when its outside callers can move with it or use public API instead.
2. **Public**: a documented public item of the new crate, when it is a coherent part of that crate's job (for example
   `BpeTokenizer::token_ids_for_content`).
3. **Internal**: a free function or extension trait inside `#[doc(hidden)] pub mod __internal` of the new crate,
   called by the facade's code by its full path and never re-exported by the facade. This is for engine-specific glue.

An internal item is never an inherent method of a type the facade re-exports: methods travel with their type, so a
hidden inherent method would be callable through the facade path. No item is widened without a listed disposition.

### D5. Features

A feature that gates code inside a moved module because the code needs an optional dependency becomes a feature of the
new crate with the same name and meaning, and the facade's feature of that name forwards to it
(`serve = [..., "lattice-tokenizer/serve"]`).

A `cfg` that exists only because the item's callers are compiled under some combination of consumer features (the
three tokenizer items listed under Context) is removed in the move, and the item compiles unconditionally. A gate
cannot be kept: Cargo features cannot express "all of these features together", and `cfg(test)` is false in a
dependency when the facade's tests run, so a kept gate would break the CPU test suites that call these items today.
New crates add no default features of their own; the facade's defaults are unchanged.

### D6. Acceptance of every extraction change

1. **Same tests.** The set of `(crate, test name)` pairs listed by `cargo test --lib --tests -- --list` for
   `lattice-inference` before the move equals the union over `lattice-inference` and the new crate after it, apart from
   the crate prefix of moved tests. Test targets declared `test = false` or `harness = false` do not answer `--list`, and are accounted for by name.
   The counts are stated in the change, and every CI feature-matrix job passes.
2. **Same public surface.** Because `cargo semver-checks` does not follow re-exports from another crate (Context),
   the repository's semver job reports the moved items as missing on an extraction change. The change states this and
   carries two instruments instead:
   - an integration test in `lattice-inference` that names every moved public item through both its facade path and
     its new crate path and proves they are the same item (for types, by assigning one to the other; for traits and
     functions, by use), together with the trait bounds callers rely on (`Send`, `Sync`, `std::error::Error`, `From`
     conversions);
   - a listing of the facade's public items that resolves re-exports across the workspace's rustdoc JSON, identical
     before and after the move.
3. **Clippy and docs.** `cargo clippy` with `-D warnings` on the CI feature sets, and the CI rustdoc command with its
   broken-link lints over the whole workspace, so links in the new crate are checked too.
4. **Measurement.** The bench-compare disposition follows the reachability rules in the repository's contribution
   guide. A moved module that a declared measurement target reaches gets a before/after comparison of that target. A
   move also changes build inputs (manifests, crate boundaries), which the guide treats as requiring a measured target
   where one reaches the code.
5. **CI reach.** Workflow path filters that select work by source path (for example the end-to-end parity gate,
   which watches `crates/inference/src/**` and `crates/embed/src/**`) are extended to the new crate's paths in the same
   change, so moved code keeps triggering the gates it triggered before.
6. **Pure move.** No logic, signature or name changes in the same change beyond what D3 to D5 require. Renames and API
   redesigns are separate changes after the move lands.

### D7. Dependency direction

Extracted crates may depend on the base crate and on each other, in one direction only. None depends on
`lattice-inference`. `lattice-embed` and `lattice-tune` keep depending on the facade; they may later depend on a
narrower crate in their own changes. A CI check over `cargo metadata` fails if any extracted crate depends on
`lattice-inference` or if the extracted crates form a cycle. The crate-ownership and publishing sections of the
contribution guide are updated in the same change that adds a crate.

### D8. Naming, versioning and publishing

1. New crates are named `lattice-<area>`, share the workspace version, licence, lints and repository metadata, and
   carry a README and `package.metadata.docs.rs` features. Each name is checked free on crates.io in the change that
   introduces it.
2. The publish script's tiers are re-derived from path dependencies in that change. With the base crate and the
   tokenizer the order becomes: `lattice-fann`, `lattice-transport`, `lattice-inference-core`; then
   `lattice-tokenizer`; then `lattice-inference`; then `lattice-embed`, `lattice-tune`, with an index wait between
   tiers.
3. A crate that has never been published cannot be resolved from the registry, so a registry-backed dry run of its
   dependents fails until it exists. The first release with a new crate states how its publish was rehearsed.
4. A new crate joins the semver job's package list in the first change after its first release, when a baseline
   exists. From then on the extracted crate is checked directly, which is where its API is now defined.

### D9. Order

1. The base crate and the tokenizer: the smallest real case, and it exercises D1 to D8 end to end.
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
- **The tokenizer crate owns `InferenceError`.** Saves one crate now and forces weight, quantization and grammar
  crates to depend on a tokenizer later, or a second move of the error type. Rejected in favour of D2.
- **One large split in a single change.** Unreviewable and not bisectable. Rejected in favour of D9's order.

## Consequences

- Downstream users see no change unless they opt into a narrower crate.
- The workspace gains crates to publish and version; the publish script and release notes cover them.
- Compile and review scope for tokenizer, grammar and format work shrinks to the crate that owns it.
- The semver job on `lattice-inference` stops seeing items that moved out of it; those items are checked on their
  own crates after first release (D8.4), and the facade's surface by the D6.2 instruments.
- Internal glue modules (D4.3) are a cost that should fall over time as consumers move.
