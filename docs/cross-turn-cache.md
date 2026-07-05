# Cross-Turn KV Prefix Cache

A multi-turn chat conversation re-sends its entire history as the prompt on every turn (the
client is stateless; the server/library is not expected to remember anything between calls).
Without a cache, that means re-prefilling every prior message from scratch on every single turn
— cost that grows with conversation length and, for Qwen3.5's hybrid GQA+GDN architecture, also
means recomputing 18 of 24 layers' recurrent GDN state from the beginning every time.

The cross-turn prefix cache (issue #462) lets a new turn that safely extends the previous one
reuse that retained KV/GDN state and prefill only the new suffix, instead of re-prefilling the
whole conversation. This document covers the actual Rust API, what currently calls it, and — just
as important — what does **not** call it yet, since that gap is easy to assume away incorrectly.

## Where this cache lives, and where it doesn't (read this first)

The underlying cache (`crate::kv_cache::cross_turn` plus the Metal runtime hooks in
`MetalQwen35State`) shipped in PR #516 (merged). At that point, **nothing in the codebase called
it** — it was infrastructure with zero consumers. PR #619 later wired the first real caller, and
it wires exactly one: **`chat_metal`** (`crates/inference/src/bin/chat_metal.rs`), both its
`--json` one-shot/`--serve` path and its interactive REPL.

**`lattice_serve` (the separate HTTP daemon built for the macOS Studio app) now calls the
cancel-aware cache methods** on a single sticky `CrossTurnSlotId::DEFAULT` slot, since it is a
single-worker, single-model binary: its one Metal state, on its one worker thread, calling
`chat_completion_streaming_with_prefix_cache_and_cancel` instead of unconditionally
`reset_state()` + `chat_completion_streaming_with_cancel` before every request. A request whose
messages are not an append onto the currently retained entry (a new conversation, edited history,
or a second client's unrelated prompt interleaved onto the same slot) still falls back to
`FullRefill` — honestly, never incorrectly — exactly like every other cache-miss case in this
document. Cache stats (`slot`, `prompt_tokens`, `reused_tokens`, `mode`) are logged to stderr per
request; there is no response-surface (HTTP header/JSON field/SSE event) telemetry yet.

**`lattice serve` (the OpenAI-compatible HTTP server in `lattice.rs`) now calls the cache-aware
methods too**, on the same single sticky `CrossTurnSlotId::DEFAULT` slot as `lattice_serve`: its
Metal worker thread calls `generate_streaming_with_prefix_cache` instead of the plain,
non-cache-aware `generate_streaming` on every request. `POST /v1/chat/completions` no longer
re-prefills the entire prompt from scratch when a turn is a safe append onto the previous one —
the same `FullRefill` fallback applies for a new conversation, edited history, or an interleaved
second client on the same slot. Cache stats (`mode`, `reused_tokens`, `prefetched_tokens`,
`prompt_tokens`) are logged to stderr per request, matching `lattice_serve`'s telemetry. This path
has no client-disconnect cancellation today (unlike `lattice_serve`'s
`_and_cancel` variant), so it uses the plain `generate_streaming_with_prefix_cache` wrapper.

See [`docs/serve-http-api.md`](serve-http-api.md) for the full `lattice serve` HTTP API — this
document is only about the library-level cache, not the HTTP surface.

The underlying cache remains a **single-entry** store (see below) — safe for one interactive
process (or one sticky server slot) talking to one conversation at a time, but a server handling
several simultaneous, genuinely interleaved conversations will still see the later ones evict the
earlier ones' retained state, forcing a full re-prefill rather than corrupting anything. Making the
cache itself safe for concurrent multi-session residency (a bounded multi-entry store with proven
KV/GDN ownership) is called out explicitly as follow-up work, not something already done.

So: if you're calling `lattice_serve`'s or `lattice serve`'s HTTP API from one active conversation
at a time, this cache now helps you; if several conversations interleave through either, expect
honest fallback to full re-prefill for whichever conversation was not most recently active. The
rest of this document covers exactly what "safely extends" means and where the sharp edges are.

## The library API

Everything here lives on `MetalQwen35State` (`lattice_inference::forward::metal_qwen35`) and in
`lattice_inference::kv_cache`.

### Identifying a conversation: `CrossTurnSlotId`

```rust
use lattice_inference::kv_cache::CrossTurnSlotId;

let slot = CrossTurnSlotId::DEFAULT;        // the single-client / local-use slot (value 0)
let slot = CrossTurnSlotId::new(0xABCD_1234); // e.g. a hashed session key, for your own multiplexing
```

`CrossTurnSlotId` is a thin `u64` wrapper. Distinct slot IDs are guaranteed never to read each
other's retained state — but see "One entry, not one entry per slot" below before assuming two
slots can be used concurrently with both benefiting from caching at once.

### The two cache-aware entry points

These are siblings of the existing (non-cached) generation methods, not a flag on them — the
plain methods are untouched and never consult the cache:

```rust
pub fn generate_streaming_with_prefix_cache<F>(
    &mut self,
    slot_id: CrossTurnSlotId,
    prompt: &str,
    tokenizer: &BpeTokenizer,
    gen_cfg: &GenerateConfig,
    on_token: F,
) -> Result<CachedGenerateOutput, InferenceError>
where
    F: FnMut(&str, u32) -> bool;

pub fn chat_completion_streaming_with_prefix_cache<F>(
    &mut self,
    slot_id: CrossTurnSlotId,
    messages: &[ChatMessage],
    tokenizer: &BpeTokenizer,
    gen_cfg: &GenerateConfig,
    on_token: F,
) -> Result<CachedChatCompletionOutput, InferenceError>
where
    F: FnMut(&str, u32) -> bool;
```

`chat_completion_streaming_with_prefix_cache` is the one you want for an actual multi-turn
conversation: pass your full growing `Vec<ChatMessage>` history each turn (same as the
non-cached `chat_completion_streaming`) — it formats the ChatML prompt internally and appends the
`<|im_end|>` stop token, then calls the prefix-cache-aware generation path underneath.

You no longer need to call `reset_state()` yourself before each turn. The old pattern —
`metal.reset_state(); metal.chat_completion_streaming(&history, ...)` — becomes just
`metal.chat_completion_streaming_with_prefix_cache(slot_id, &history, ...)`. A first request, a
divergent/edited history, an adapter change, or any other case where reuse isn't safe falls back
to a full internal reset-and-reprefill on its own — you don't need to detect that yourself.

### Reading what happened: `CachedGenerateOutput` / `CachedChatCompletionOutput`

Both cache-aware calls return the normal output type plus cache statistics for that call:

```rust
pub struct CrossTurnCacheStats {
    pub slot_id: CrossTurnSlotId,
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    pub prefetched_tokens: usize,
    pub mode: PrefixReuseMode,
}

pub struct CachedGenerateOutput {
    pub output: GenerateOutput,       // identical to the non-cached return type
    pub cache: CrossTurnCacheStats,
}
// CachedChatCompletionOutput is the same shape with `output: ChatCompletionOutput`.
```

This is your "did it actually reuse anything" signal — check `cache.mode` and
`cache.reused_tokens` rather than assuming caching happened just because you called the
`_with_prefix_cache` method. `chat_metal` prints a line like this to stderr after every turn. The
`--json` path (`--json --prompt` / `--json --serve`) and the interactive REPL use two different
shapes:

```
# --json mode
[chat_metal] GPU Metal q4: 812 prompt + 40 gen in 620ms = 64.5 tok/s | cache: ExactAppend reused 780/812

# interactive REPL
[812 prompt + 40 gen in 620.0ms = 64.5 tok/s | GPU Metal q4 | cache: ExactAppend reused 780/812]
```

`model_format` here is `"bf16"` or `"q4"` — there is no separate `"q4-quarot"` label; a
QuaRot-converted checkpoint loads and reports as plain `"q4"`. `reused_tokens` out of
`prompt_tokens` tells you how much of this turn's prompt didn't need to be re-prefilled; `mode`
tells you which of the three `PrefixReuseMode` variants applied.

### `PrefixReuseMode`: what actually happened

```rust
pub enum PrefixReuseMode {
    /// No usable overlap: reset state and prefill the whole prompt.
    FullRefill,
    /// The cached entry is an exact prefix of the new prompt and its GDN
    /// snapshot is taken at that same boundary — reuse it verbatim.
    ExactAppend,
    /// Reuse only up to an earlier exact GDN checkpoint, then replay/prefill
    /// forward. v2: no v1 caller currently supplies checkpoints, so this
    /// variant is never produced by the current Metal integration.
    ReplayFromCheckpoint { checkpoint_len: usize },
}
```

In practice, with today's callers, you will only ever see `FullRefill` or `ExactAppend`.
`ReplayFromCheckpoint` exists in the type for a planned sparse-checkpoint extension but nothing in
the current Metal integration produces it — don't design around it yet.

## What makes a turn "safely extend" the previous one

The planner (`plan_prefix_reuse` in `kv_cache::cross_turn`) only ever claims `ExactAppend` when
**all** of the following hold:

1. The new prompt's tokens are an exact superset of the retained entry's tokens, sharing the
   entire retained prefix (an exact token-ID match, not a fuzzy/semantic one).
2. That shared prefix covers the entry's _entire_ represented length (partial mid-history reuse
   without an explicit checkpoint isn't attempted).
3. The new prompt actually has a nonempty suffix beyond that shared prefix — an exact repeat of
   the previous prompt with nothing new falls back to `FullRefill` rather than treating a
   zero-length suffix as a (disallowed) no-op prefill.
4. A `CrossTurnPrefixMetadata` fingerprint matches exactly between the retained entry and the
   current request:

```rust
pub struct CrossTurnPrefixMetadata {
    pub model_fingerprint: u64,
    pub tokenizer_fingerprint: u64,
    pub adapter_id: AdapterId,
    pub vocab_size: usize,
    pub max_cache_len: usize,
    pub kv_f16: bool,
    pub rope_theta_bits: u64,
    pub partial_rotary_factor_bits: Option<u32>,
    pub layer_pattern_hash: u64,
    pub chat_template_version: u32,
}
```

Any mismatch — a different loaded model, a different tokenizer, a LoRA adapter swapped in or out,
a different KV dtype, RoPE configuration, layer pattern, or chat template version — invalidates
the entry and forces `FullRefill`. This is deliberately conservative: the module's own doc comment
puts it as "any divergence falls back to `PrefixReuseMode::FullRefill`" — decline-beats-fabricate
for token-identity correctness. You do not get a wrong-but-fast answer; you get a slow-but-correct
one.

Practical implications for structuring a conversation to actually benefit:

- **Only append to history; don't edit or delete earlier turns.** Editing an earlier user message
  or regenerating an earlier assistant turn changes the token sequence at a position before the
  end, so the longest-common-prefix against the retained entry stops at the edit point (or
  becomes 0 if the edit is near the start) — every turn after an edit pays a full re-prefill until
  the next unedited extension.
- **Don't interleave unrelated conversations through the same slot.** See the next section — a
  second conversation on the same slot doesn't get its own separate cache entry, it evicts the
  first one's.
- **A LoRA adapter swap invalidates the cache** (it's part of the fingerprint) — expect a full
  re-prefill on the turn immediately after swapping adapters.

## One entry, not one entry per slot

This is the sharpest edge in the current design, and it's easy to misread from "distinct slots
never share state" alone. The Metal-side cache (`MetalCrossTurnPrefixCache`) retains **at most one
entry, for one slot, at a time** — it is deliberately not an unbounded per-slot map.

`get`/`take` both check the slot ID on the one entry that exists, so a lookup for a _different_
slot than the one currently retained is always a clean miss (never a stale or wrong-slot hit) —
that's the isolation guarantee. But it also means: if you generate on slot A, then slot B, then
slot A again, that third call is a full refill — slot B's generation evicted slot A's entry, it
did not get its own independent storage alongside it. If your process genuinely interleaves
multiple concurrent conversations through one `MetalQwen35State`, only the most recently generated
one benefits from caching at any given moment; the others pay full re-prefill every time they get
a turn. This is the multi-session thrash a single shared worker sees when several simultaneous
chats share one sticky slot, as `lattice_serve` and `lattice serve` both now do: correct (a losing
conversation just gets an honest full re-prefill) but not the multi-session residency a bounded
multi-entry cache would give.

PR #516 originally considered an unbounded per-slot map, but the underlying full-attention KV
buffer is one mutable live buffer shared by the whole process. A stale map entry could restore GDN
state from one conversation while attention silently read KV rows another generation had since
overwritten. The implemented fix caps retention at one entry and makes any other mutation of live
state — a different slot's generation, the plain non-cache-aware generate path, an explicit
`reset_state()`, or a LoRA load/unload — invalidate the entry before that mutation proceeds.

## Failure handling

If a cache-aware call fails partway through (a restore or prefill error), the engine does not
leave the cache or live model state in a possibly-inconsistent condition for the next call to
trip over:

```rust
match self.generate_streaming_with_prefix_cache_and_cancel_inner(slot_id, prompt, tokenizer, gen_cfg, on_token, should_cancel) {
    Ok(out) => Ok(out),
    Err(e) => {
        // Fail-closed: state and cache may disagree after a
        // restore/prefill/save error, so drop both rather than
        // risk a later turn reusing an inconsistent boundary.
        self.reset_state();
        self.cross_turn_prefix_cache.remove(slot_id);
        Err(e)
    }
}
```

Both the live model state and the retained cache entry for that slot are cleared before the error
is returned to you. The next call on that slot — even a retry of the same conversation — starts
from a clean `FullRefill`, not from a half-restored state. `chat_metal`'s own error handling relies
on exactly this: on a generation failure it just surfaces an error event (or, in the REPL, drops
the unanswered turn) and continues, because it knows the engine already put itself back into a
consistent state.

## Using it from `chat_metal`

`chat_metal`'s interactive REPL builds a growing `Vec<ChatMessage>` history and now calls the
cache-aware entry point instead of resetting state on every turn:

```rust
let cache_result = metal.chat_completion_streaming_with_prefix_cache(
    CrossTurnSlotId::DEFAULT,
    &history,
    &tokenizer,
    &gen_cfg,
    |delta, _| {
        print!("{delta}");
        std::io::stdout().flush().ok();
        response_text.push_str(delta);
        true
    },
);
```

The old unconditional `metal.reset_state()` call that used to precede every turn is removed
entirely — the cache-aware call now owns that decision. The one-shot `--json --prompt` and
`--json --serve` paths (used by the macOS Studio app to drive `chat_metal` as a subprocess) get
the same treatment via `emit_json_generation`, using `metal.generate_streaming_with_prefix_cache`
with the same `CrossTurnSlotId::DEFAULT` slot, since that binary only ever handles one
conversation per process. Both call sites drop the `output` field out of the returned
`Cached*Output` for the actual generated text/token count, and log the `cache` field's
`mode`/`reused_tokens`/`prompt_tokens` alongside the existing tok/s line so you can see the cache
behavior on every turn without instrumenting anything yourself.

The PR also adds a Metal integration test,
`cross_turn_cache_chat_completion_matches_full_reprefill`, that drives a growing multi-turn
history through this wrapper and asserts token-identity against an independent from-scratch full
re-prefill, plus a nonzero reused-token count on at least one turn — so the test cannot pass via
the fallback path alone, and a dedicated Criterion bench,
`cross_turn_prefix_cache_bench`, to measure the reuse path directly.

## Summary

- The cache is reachable today through direct `MetalQwen35State` calls, `chat_metal`,
  `lattice_serve`, and `lattice serve` (the OpenAI-compatible HTTP server in `lattice.rs`) — all on
  a sticky `CrossTurnSlotId::DEFAULT` slot, with honest full-refill fallback on any mismatch.
- Use `CrossTurnSlotId::DEFAULT` for a single local conversation; only one entry is ever retained
  process-wide, so multiplexing several conversations through distinct slot IDs on one
  `MetalQwen35State` does not give each of them independent caching — the most recent one wins and
  evicts the rest.
- Reuse requires an exact token-prefix match plus an exact match on a ten-field fingerprint
  (model, tokenizer, adapter, vocab, cache length, KV dtype, RoPE config, layer pattern, chat
  template version). Any mismatch, or any edit to earlier conversation turns, forces a full
  re-prefill rather than a wrong answer.
- Check `cache.mode` and `cache.reused_tokens` on the returned `Cached*Output` (or the
  `chat_metal` stderr log line) to confirm reuse actually happened — don't assume it did just
  because you called the `_with_prefix_cache` method.
- On any internal failure, both live model state and the cache entry are cleared before the error
  propagates — the next call starts clean.
