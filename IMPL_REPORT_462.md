# Implementation report: cross-turn KV prefix reuse in `chat_metal` (#462)

## What changed

Issue #462 asked for the chat serve loop to stop re-prefilling the entire
conversation on every turn. #516 already shipped the underlying engine
primitive — `MetalCrossTurnPrefixCache` / `PrefixReuseMode` and the
`*_with_prefix_cache` entry points on `MetalQwen35State` — but nothing in
any of the three serving binaries called them; `chat_metal.rs` still called
`reset_state()` unconditionally before every request. This change wires the
existing cache-aware API into `chat_metal.rs`'s two request paths:

- `emit_json_generation` (the `--json` single-shot / driven-by-external-app
  path): replaced the unconditional `reset_state()` + `generate_streaming`
  call with `generate_streaming_with_prefix_cache(CrossTurnSlotId::DEFAULT,
  ...)`. A cache/generation error is reported as a `{"ev":"error"}` protocol
  event instead of tearing down the process, since the engine already resets
  its own live state on that path.
- The interactive REPL loop: replaced `reset_state()` +
  `chat_completion_streaming` with `chat_completion_streaming_with_prefix_cache`.
  On error, the unanswered turn is popped back out of `history` so the user
  can retry cleanly.

Both call sites use a single fixed slot (`CrossTurnSlotId::DEFAULT`), which
matches `chat_metal`'s single-session, single-process design — there is
exactly one conversation live at a time, so one slot is all this binary
needs. Neither call site changed anything about *when* a full refill happens;
that decision (safe append vs. mismatch vs. LoRA change vs. GDN-mode flip)
already lives entirely inside the engine and is unconditionally
fail-closed — a first request, an edited history, a divergent prompt, or an
adapter swap all still fall back to a full prefill exactly as before, they
just no longer pay for it via an explicit `reset_state()` call from the
binary.

Status line output for both paths now also prints `cache: {mode} reused
{reused}/{prompt}` so this is observable at the terminal without extra
tooling.

No change to the cache implementation itself, no change to any other
serving binary, no change to any Metal kernel or dispatch geometry.

## Correctness

Added `cross_turn_cache_chat_completion_matches_full_reprefill` in
`crates/inference/src/forward/metal_qwen35.rs`, next to the existing
`cross_turn_cache_multiturn_token_identity_matches_full_reprefill`. It drives
`chat_completion_streaming_with_prefix_cache` (the same call `chat_metal`'s
REPL loop now makes) across a multi-turn conversation and asserts the
generated tokens are byte-identical to a from-scratch `reset_state()` +
`chat_completion_streaming` run over the same history, and that
`reused_tokens > 0` on a later turn once `ExactAppend` engages (i.e. the
assertion cannot pass by silently falling back to `FullRefill` the whole
time).

All ten `cross_turn_cache_*` tests in `metal_qwen35.rs` (nine pre-existing +
the one new test above) plus the rest of the `cross_turn`-tagged suite (the
core cache-logic tests from #516) pass solo, `--test-threads=1`, against a
real Metal device:

```
cargo test -p lattice-inference --features "f16,metal-gpu" --lib cross_turn -- --test-threads=1
```

27/27 passed. This includes the tests directly guarding the invariants this
task was told to respect: `cross_turn_cache_slot_isolation_evicts_other_slot`,
`cross_turn_cache_lora_load_unload_invalidates`,
`cross_turn_cache_invalidated_by_gdn_mode_flip`,
`cross_turn_cache_prefix_mismatch_falls_back_to_full_refill`,
`cross_turn_cache_interleaved_plain_path_invalidates`,
`cross_turn_cache_interleaved_raw_forward_invalidates`, and
`cross_turn_cache_exact_equal_retry_full_refills`.

A broader, non-cross-turn-filtered run of the whole `metal_qwen35` module
(170 tests total) was also attempted this session as an extra check beyond
what the task required. It surfaced pre-existing failures unrelated to this
diff — `gdn_chunked_prefill_vs_serial_prefill_logit_parity`,
`gdn_chunked_state_vs_serial_state_diff`, and three `lm_head_*` real-checkpoint
agreement tests — while running on a machine with other, unrelated Metal-GPU
test processes active concurrently (this repo's own engineering notes flag
concurrent GPU load as a source of both timing and numeric corruption, not
just noise). Re-running the two tests originally flagged as failing, solo
with the GPU otherwise idle (verified via process listing immediately
before), resolved both:

- `gdn_chunked_prefill_vs_serial_prefill_logit_parity` — passes cleanly
  solo (`ok` in the isolated run). The `gdn_chunked_*` failures track a
  separate, already-known GDN chunked-vs-serial parity issue being fixed
  independently; nothing in this diff touches GDN chunking.
- `f16_kv_metal_path_executes_and_reports_capability` — not a regression at
  all: it's an opt-in hardware-path-proof probe that panics with an explicit
  message (`f16 KV Metal probe must run with LATTICE_KV_F16=1`, then
  `... LATTICE_METAL_PATH_PROOF=1`) unless both are set. With both env vars
  set it passes in 0.1s. It was never exercising its real path in the
  bare run.

Neither of those two, nor any of the other module-level failures, are
reachable from this change: the only edit inside `metal_qwen35.rs` is the
one new, additive test above — no production code in that file changed.
`chat_metal.rs` is a separate binary target not included in `--lib` test
runs at all.

## Performance

Dedicated Criterion bench added at
`crates/inference/benches/cross_turn_prefix_cache_bench.rs`
(`required-features = ["metal-gpu", "f16"]`, so it's simply absent from the
default target set on non-macOS/non-metal-gpu builds — no `SKIP:` guard
needed). It replays a nine-turn travel-assistant conversation against the
real local `~/.lattice/models/qwen3.5-0.8b` checkpoint and compares, at two
conversation depths:

- `full_reprefill`: `chat_metal`'s pre-#462 behavior — `reset_state()` then
  `chat_completion_streaming` over the entire history, every call.
- `cache_aware_incremental`: the #462 path — history is replayed once
  through `chat_completion_streaming_with_prefix_cache` to build the cache
  (excluded from timing, via `iter_batched`), then the timed call is the
  single next turn.

```
cargo bench -p lattice-inference --features "f16,metal-gpu" --bench cross_turn_prefix_cache_bench -- cross_turn_prefix_cache
```

Results (this session, `--quick`, sample_size=10; ranges are Criterion's
[lower, point estimate, upper] confidence interval):

| depth | full_reprefill | cache_aware_incremental | direct delta |
|---|---|---|---|
| 2 prior turns | 463.28 / **498.76** / 538.29 ms | 502.21 / **559.68** / 610.60 ms | +12.2% (slower) |
| 8 prior turns | 839.57 / **900.11** / 970.38 ms | 513.12 / **578.77** / 650.57 ms | **-35.7%** |

Read by depth-scaling instead of same-depth delta, the result is
unambiguous and matches the acceptance criterion's own wording ("turn N's
prefill cost is O(new tokens) instead of O(total tokens) when the prefix is
unchanged"):

- `full_reprefill` grows **+80.5%** from depth 2 to depth 8 (498.76ms ->
  900.11ms) — cost scales with total conversation length, as expected when
  every turn re-prefills everything.
- `cache_aware_incremental` grows only **+3.4%** over the same span
  (559.68ms -> 578.77ms) — essentially flat, because each timed call only
  prefills the one new turn regardless of how much prior conversation sits
  behind it in the cache.

The 2-prior-turns same-depth comparison looks like a wash (even a small
apparent regression) because both paths generate the same fixed 32 decode
tokens, and at shallow depth the *prefill* savings this change targets are
small relative to that fixed decode cost, so the two paths are within noise
of each other there. The 8-prior-turns comparison isolates the prefill cost
enough to show a clear, direct 35.7% reduction, and the depth-scaling
comparison above is the cleanest evidence that the change does what #462
asked for.

Caveat: this machine had several other unrelated build/test/bench jobs
running concurrently in sibling worktrees for most of this session
(contention, not a correctness issue — see Verification below), so treat
the absolute millisecond figures as directionally reliable rather than
tightly precise. The qualitative result — flat cache-aware cost vs. growing
full-reprefill cost as depth increases — is a large (multiples, not a few
percent) effect, well outside what contention noise alone would plausibly
produce.

`make bench-compare` (the repo's mandatory generic regression gate):

```
$ make bench-compare
=== bench-compare: origin/main (4ad5d7cf3) vs HEAD (4ad5d7cf3) ===
```

`bench-compare.sh` always benches `elementwise_cpu_bench` (lattice-inference:
rms_norm, layer_norm, silu, gelu, softmax_attention, elementwise_mul — all
defined in `forward/cpu/elementwise.rs`) and `simd` (lattice-embed) only, and
this change touches neither file, neither crate, no SIMD kernel, no
dispatch geometry. The comparison ran against the real working-tree diff
(the "HEAD" side builds directly in this worktree, uncommitted changes
included — the identical short SHA shown for both sides is just because
`git rev-parse` reports the parent commit, not a sign the two builds were
identical). The gate reported large swings in both directions across
almost every group (some >80% in each direction, including internally
inconsistent pairs like `softmax_attention/128` at -31% next to
`softmax_attention/512` at +216%), which is the signature of the same
machine-contention effect discussed above, not a real regression — this
change cannot touch elementwise or SIMD code by construction. This is
exactly why a dedicated bench (above) exists for the code path #462
actually changed. Per this repo's own guidance for a change bench-compare's
fixed group list can't observe: bench-compare showed no attributable
change.

## Verification performed this session

- `cargo test -p lattice-inference --features "f16,metal-gpu" --lib
  cross_turn -- --test-threads=1` — 27/27 passed.
- `cargo test -p lattice-inference --features "f16,metal-gpu" --lib -- \
  --test-threads=1 f16_kv_metal_path_executes_and_reports_capability \
  gdn_chunked_prefill_vs_serial_prefill_logit_parity` (solo, GPU otherwise
  idle, verified via process listing beforehand) — both pass; see
  Correctness above for the two-line explanation of why they showed up as
  failures during the broader, contended, out-of-scope module run.
- `cargo clippy --workspace --all-targets -- -D warnings` (default
  features) — clean.
- `cargo clippy -p lattice-inference --all-targets --features
  "f16,metal-gpu" -- -D warnings` (covers the `chat_metal` binary, the new
  bench, and the lib/tests together) — clean.
- `cargo fmt --all -- --check` — clean.
- `cargo bench -p lattice-inference --features "f16,metal-gpu" --bench
  cross_turn_prefix_cache_bench` — ran end-to-end against the real local
  Qwen3.5-0.8B checkpoint, see Performance above.
- `make bench-compare` — see Performance above.
- No `unwrap()`/`expect()` introduced; the one pre-existing `.expect()` in
  `chat_metal.rs` (`HOME not set`) predates this change and is a legitimate
  startup precondition.

A full, unfiltered `cargo test -p lattice-inference --lib` run and a
170-test unfiltered run of just the `metal_qwen35` module were both
attempted this session as extra, broader-than-required checks. Neither
finished cleanly (machine-wide contention from other concurrent work made
both impractically slow — the module-only run was still only 29% done
after 57 minutes), so I did not wait them out; the targeted, solo,
idle-GPU verification above is the actual basis for this report's
correctness claim, not those two.

## Stretch goals not attempted

`lattice_serve.rs` and `lattice.rs` were explicit stretch goals, to attempt
only if the existing cache API cleanly extends without new slot-allocation
design work. It doesn't, for two independent reasons found this session:

1. `MetalCrossTurnPrefixCache` is a single `Option<MetalCrossTurnPrefixEntry>`,
   not a per-slot map — `insert()` unconditionally replaces whatever entry
   is currently retained regardless of the `CrossTurnSlotId` passed in. A
   multi-session server (`lattice_serve`, `lattice.rs`) with more than one
   concurrent conversation would have each new request's cache write evict
   every other session's cache entry, silently degrading every other live
   session back to full re-prefill. That's safe (fail-closed, never wrong
   output) but defeats the point for any server handling more than one
   session at a time, which is the normal case for those two binaries.
   Turning this into a real per-slot cache is a separate, scoped change to
   `kv_cache::cross_turn`, not a one-line wiring change.
2. `MetalQwen35State` is not `Send` (it holds raw `metal::*` FFI fields
   directly), a pre-existing constraint already tracked separately against
   `lattice_serve`'s own Q4 serving path. Any multi-session, multi-request
   server built on top of it needs its own answer to that before cross-turn
   caching is even the next question.

`chat_metal.rs` sidesteps both: it is single-process, single-session, and
already serializes all requests through one `MetalQwen35State` on one
thread, so `CrossTurnSlotId::DEFAULT` and a single retained cache entry are
exactly correct, not a simplification.

Concrete follow-up for whoever picks up `lattice_serve`/`lattice.rs`: extend
`MetalCrossTurnPrefixCache` to a small bounded map keyed by
`CrossTurnSlotId` (with an eviction policy — LRU is the obvious default)
before wiring `*_with_prefix_cache` into either multi-session binary.
