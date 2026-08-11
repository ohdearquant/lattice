VERDICT: FIXED, REGRESSION TEST ADDED, MUTATION-VERIFIED, GATES GREEN

## Defect

`POST /v1/embeddings` (`crates/inference/src/bin/lattice_serve.rs`) bounded concurrent
`encode_batch` jobs with a 4-slot `Semaphore` (`EmbeddingState::admission`). The admission
permit was bound in the request handler's own stack frame
(`let _admission_permit = try_acquire_embedding_slot(...)`) while the CPU work ran via
`tokio::task::spawn_blocking(move || ...)`.

A `spawn_blocking` task that is already running is not cancelled by dropping its
`JoinHandle`. When a client disconnected mid-request, the handler's future was dropped at
its `.await` point, which dropped `_admission_permit` and released the slot immediately --
while the already-started `encode_batch` job kept running on the blocking thread pool.
Repeated disconnects could therefore accumulate concurrent `encode_batch` jobs past the
configured cap; the cap only bounded connected clients.

## Fix

Moved the permit into the `spawn_blocking` closure so ownership -- and release -- is tied
to the job's completion rather than to the handler future's lifetime. Extracted the
`spawn_blocking` call into a small helper, `run_embedding_encode(permit, work)`, generic
over the blocking closure, so the fix is directly unit-testable:

```rust
async fn run_embedding_encode(
    permit: tokio::sync::OwnedSemaphorePermit,
    work: impl FnOnce() -> Result<Vec<Vec<f32>>, lattice_inference::InferenceError> + Send + 'static,
) -> Result<Result<Vec<Vec<f32>>, lattice_inference::InferenceError>, tokio::task::JoinError> {
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        work()
    })
    .await
}
```

The handler's call site (`embeddings()`) now passes its `admission_permit` into this
helper instead of holding it across the `.await` in its own frame. The 503 rejection path
in `try_acquire_embedding_slot` is unchanged.

## Why the helper extraction

`EmbeddingState::model` is a concrete `BertModel`, not a trait object, and `BertModel` has
no synthetic/mock constructor (existing tests load a real model from
`LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR`, `#[ignore]`d when that env var is unset). There
is no way to give the real HTTP handler an artificially slow `encode_batch` without a
downloaded model directory. `run_embedding_encode` is generic over the blocking closure,
so a deterministic test closure exercises the exact permit-lifetime mechanics the fix
changed, without needing a model at all.

## Regression test

`crates/inference/src/bin/lattice_serve.rs`,
`imp::tests::embedding_permit_survives_dropped_request_future`:

- Acquires the single slot of a 1-capacity `Semaphore`, then calls `run_embedding_encode`
  with a closure gated by two channels (a "started" signal and a "release" gate) -- a
  deterministic handshake, not a sleep.
- Runs the call inside `tokio::spawn`, awaits the "started" signal, then calls
  `JoinHandle::abort()` on the outer task. At that point the task is suspended awaiting the
  inner `spawn_blocking` `JoinHandle` (not actively running), so `abort()` drops the task's
  future there -- the same event a dropped/disconnected request future produces.
- Asserts `admission.available_permits() == 0` immediately after the abort resolves: the
  slot must still be held because the blocking job is still running.
- Releases the blocking closure and polls (bounded, 5ms steps, 1s cap) until
  `available_permits() == 1`, confirming the slot frees once the job actually completes.

### Mutation check

Command: `cargo test -p lattice-inference --bin lattice_serve --features
f16,metal-gpu,test-utils -- embedding_permit_survives_dropped_request_future`

1. Fix in place: `test imp::tests::embedding_permit_survives_dropped_request_future ... ok`
   (`test result: ok. 1 passed`).
2. Fix reverse-applied (`run_embedding_encode` body restored to bind the permit outside the
   `spawn_blocking` closure, matching the pre-fix shape), file touched to force a rebuild:
   test failed --
   `assertion left == right failed: admission slot must stay held while the encode_batch
   job is still running, not release just because the request future was dropped
   left: 1 right: 0`.
3. Fix restored: `test imp::tests::embedding_permit_survives_dropped_request_future ... ok`
   again. `git diff --stat` confirmed no residual diff from the temporary mutation.

## Gates (all commands run under `RUSTC_WRAPPER=""`, flock-serialized)

- `cargo test -p lattice-inference --test metal_measurement_lock_contract` -- 53 passed, 0
  failed. This edit inserted the new helper above `imp::load_model`'s two
  `MetalQwen35State` construction-exemption sites, shifting both by the same +19 lines
  (`1772:47` -> `1791:47`, `1792:47` -> `1811:47`, verified as pure line movement --
  column and enclosing-function identity unchanged); `recorded_position` updated for both
  in `crates/inference/tests/metal_measurement_lock_contract.rs`.
- `cargo test -p lattice-inference --bin lattice_serve --features f16,metal-gpu,test-utils`
  -- 108 passed, 0 failed, 4 ignored (pre-existing, gated on
  `LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR`).
- `cargo test -p lattice-inference --bin lattice --features f16,metal-gpu,test-utils` -- 173
  passed, 0 failed.
- `cargo clippy -p lattice-inference -- -D warnings` -- clean.
- `cargo fmt --check` -- clean (one block reformatted by `cargo fmt` during development,
  re-verified clean after).
- `cargo semver-checks --package lattice-inference` -- `no semver update required` (196
  checks: 196 pass, 57 skip). Binary-only change, as expected.

## Scope note (not fixed, out of scope for this change)

`crates/inference/src/bin/lattice/serve.rs` also defines a `POST /v1/embeddings` handler
(a second binary target, pooled text/image embeddings for vision-language checkpoints).
It has no admission-cap `Semaphore` at all -- every `encode`/`embed_items` call runs
unbounded via `spawn_blocking` with no concurrency cap to begin with, so it does not share
the permit-scope defect this change fixes (there is no permit to scope). Flagging as a
separate, pre-existing gap rather than folding an unrelated new admission-control feature
into this fix.
