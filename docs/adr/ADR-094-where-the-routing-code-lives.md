# ADR-094: Where the routing code lives, and what feeds it

**Status**: Accepted (2026-09-21)
**Date**: 2026-09-20
**Crate**: lattice-inference / lattice-fann / lattice-tune

## Context

ADR-093 decided that mixture routing runs once per request. Nothing in the serving path calls a
router today: `crates/inference/src/serve/` contains no reference to `AdapterRouter`,
`update_router` or the `mixture` module at all. Wiring one in runs into three structural facts that
have to be settled before any of it is written, because each of them decides where code goes rather
than what it does.

**The router is behind a non-default feature and the serving path is not.** `pub mod mixture` is
gated by `#[cfg(feature = "mixture")]` at `crates/inference/src/lib.rs:100`, and `mixture` is not
in the crate's default set. The serving path is unconditional. So a call from `serve/` into the
router either drags a `cfg` onto the call site or the gate moves.

**The update primitive is in the wrong crate for the direction the call has to run.** `route` is in
`lattice-inference` (`mixture.rs:412`), but the thing that learns — `update_router`, `FeedbackEvent`,
`ReplayBuffer`, `DiagonalFisher`, `RouterUpdateConfig` — is in `lattice-tune`
(`crates/tune/src/lora/router_update.rs`, 1203 lines). The dependency runs `tune → inference`
(`crates/tune/Cargo.toml:57`, optional, under `inference-hook` and `train-backward`) and inference
declares no dependency on tune at all, against 14 declared path deps in its manifest. A feedback
endpoint in `serve/` calling `update_router` would invert that, which this repository bans outright.

**The context source is already in the serving crate.** `BertModel` is `lattice-inference`'s own
(`crates/inference/src/model/bert.rs:211`, re-exported at `lib.rs:191`), and
`serve/embeddings.rs` already loads and serves through it behind `--embedding-model`. Feeding
`route(context_vector, ...)` from an in-process embedding model needs no new crate reach.

## Decision

**1. The context vector comes from the in-process embedding model, with an explicit refusal when
none is loaded.** The gate was trained on that representation, so this reuses it as trained. When
no embedding model is configured, a request that would route refuses and says so; it does not fall
back to a default mixture silently, because a silent fallback makes "the router is enabled" and
"the router ran" indistinguishable at every later reading.

**2. The `mixture` gate moves off the call site.** The rule is that a `cfg` never lands on the
serving call. Either the router module stops being feature-gated, or the serving path acquires a
gate-free façade whose non-`mixture` build is a compiled-in refusal rather than an absent symbol.
The second is preferred: it keeps the feature's build-size argument intact while making the missing
configuration a runtime answer a caller can read, which is the same shape as decision 1's refusal.

The façade is a concrete type with a `cfg`-selected body, not a trait. One implementor is not a
trait's reason to exist, and a trait here would add a dispatch seam whose only caller is the one
this ADR is wiring.

**3. `router_update.rs` moves from `lattice-tune` to `lattice-fann`, behind `online-router`.** Its
only tune-specific dependency is tune's error type; everything substantive it imports is already
`lattice_fann::{Network, training::{RlooConfig, RlooTrainer}}` (`router_update.rs:12-15`), and
`lattice-fann`'s `online-router` feature exists precisely to expose `RlooTrainer` and
`DiagonalFisher` (`crates/fann/Cargo.toml:16-17`). `lattice-inference` already depends on
`lattice-fann` under `mixture` (`crates/inference/Cargo.toml:70,123`), so the move makes the update
path reachable from the serving crate with the dependency direction unchanged. `lattice-tune`
re-exports the module so its two existing callers — `examples/prompt_router.rs` and
`tests/router_loop_closure.rs` — keep compiling unmodified, and that is the move's acceptance test.

## Alternatives considered

- **`cfg`-gate the serving call site.** Rejected, and this is the alternative the code invites: it
  is the smallest diff and it makes the serving path's behaviour depend on a build flag that
  nothing at the HTTP surface reports. A deployment with the feature off then answers routing
  requests as though routing were off by design.
- **Add `lattice-inference → lattice-tune`.** Rejected: it inverts a declared dependency direction
  and makes the serving crate carry the training stack.
- **Run the update out of process, over the replay buffer as a file.** Not rejected on the merits —
  it is a reasonable shape for a large training job — but it answers a different question than a
  feedback endpoint with a documented cadence does, and it is available later without any of the
  decisions here being wrong.
- **A second embedding model loaded for routing only.** Rejected: a second copy of a model the
  process already holds, and a second thing to configure, to answer a question the loaded one
  answers.
- **The served model's own prompt hidden state as the gate input.** This is the input a learned
  gate should end up on, and it removes both the second model and the per-request CPU embedding.
  It needs the gate retrained on that representation, so it is a follow-on rather than an
  alternative to decision 1, and it is recorded here so that decision 1 is read as a first wiring
  rather than as an endpoint.

## Consequences

- Decision 3 is a move, not a rewrite, so it is reversible by reverting one commit, and its
  acceptance is mechanical: `lattice-tune`'s existing example and test compile and pass unchanged.
- Decision 2's façade makes "no router configured" a response rather than a link error, which is
  what lets ADR-095's `GET` report the router's state honestly in a build without the feature.
- Decision 1 ties routing availability to the embedding model's presence. That coupling is real and
  is the reason the refusal is explicit: an operator who configures routing without an embedding
  model gets an error naming the missing piece, not a quietly unrouted service.

## Open, and deliberately not decided here

The numeric cost of the per-request embedding pass, which is unmeasured and belongs with the first
end-to-end rather than with this topology.
