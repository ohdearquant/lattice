# Generation entry-point behavior matrix (post-#787)

Analysis-only deliverable for [#822](https://github.com/ohdearquant/lattice/issues/822).
No source changes. Every cell below was verified by reading the current code at
`origin/main` commit `d18f50ebb5e0bdd1e696d7fb62b6c309ab6cec12` (the commit PR #787 /
ADR-080 C3 landed on). Re-verify line numbers before relying on this document once
`crates/inference/src/forward/metal_qwen35.rs` moves again — it changes frequently.

This document exists to answer one question before epic #614 stage 1 (shared
generation preflight) starts: **what does each entry point actually do today**, so a
future consolidation can preserve every intentional difference and eliminate only the
accidental ones. Nothing here is a recommendation to change behavior.

## Entry points covered

| # | Column label              | Function(s)                                                                      | Location                                                        |
| - | ------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1 | CPU f16                   | `generate_f16`                                                                   | `crates/inference/src/forward/cpu_f16.rs:833`                   |
| 2 | CPU q8                    | `generate_q8`                                                                    | `crates/inference/src/forward/cpu_q8.rs:686`                    |
| 3 | CPU q8 NEON               | `generate_q8_neon`                                                               | `crates/inference/src/forward/neon_forward.rs:871`              |
| 4 | Metal direct              | `MetalQwen35State::generate`                                                     | `crates/inference/src/forward/metal_qwen35.rs:8591`             |
| 5 | Metal streaming (wrapper) | `MetalQwen35State::generate_streaming`                                           | `crates/inference/src/forward/metal_qwen35.rs:12913`            |
| 6 | Metal streaming (impl)    | `MetalQwen35State::generate_streaming_with_cancel`                               | `crates/inference/src/forward/metal_qwen35.rs:12947`            |
| 7 | Metal prefix-cache        | `MetalQwen35State::generate_streaming_with_prefix_cache_and_cancel` (+ `_inner`) | `crates/inference/src/forward/metal_qwen35.rs:15330` (+`15377`) |

Column 5 is a thin pass-through: its entire body is
`self.generate_streaming_with_cancel(prompt, tokenizer, gen_cfg, on_token, || false)`
(`metal_qwen35.rs:12923`). None of the ten behaviors below depend on the
`should_cancel` closure, so columns 5 and 6 have identical outcomes in every row;
column 5 is kept as its own column only because the issue lists it as a distinct
entry point callers actually call.

Column 7 is a public wrapper (`generate_streaming_with_prefix_cache_and_cancel`,
`:15330`) around a private `_inner` (`:15377`). Where the wrapper and `_inner` do
different things (this matters a great deal for row 10), both are cited.

Shared guard helpers referenced throughout, all in
`crates/inference/src/model/qwen35/generation.rs`: `check_grammar_not_set` (`:1940`),
`check_logprobs_not_set` (`:1959`), `check_stop_strings_not_set` (`:1984`),
`check_reasoning_budget_not_set` (`:2012`), `check_mtp_not_requested` (`:2042`,
`#[cfg(all(target_os = "macos", feature = "metal-gpu"))]`).

## Matrix

### 1. Explicit RNG seed 0 vs non-zero vs unset

All seven paths run byte-identical seed logic (duplicated per file, no shared helper):

```rust
let mut rng_state = match gen_cfg.seed {
    Some(s) => if s == 0 { 1 } else { s },
    None => {
        let t = SystemTime::now().duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64).unwrap_or(0x12345678_9abcdef0);
        if t == 0 { 1 } else { t }
    }
};
```

| Path                   | Location                      |
| ---------------------- | ----------------------------- |
| CPU f16                | `cpu_f16.rs:842-858`          |
| CPU q8                 | `cpu_q8.rs:695-711`           |
| CPU q8 NEON            | `neon_forward.rs:880-896`     |
| Metal direct           | `metal_qwen35.rs:8602-8618`   |
| Metal streaming (impl) | `metal_qwen35.rs:12963-12979` |
| Metal prefix-cache     | `metal_qwen35.rs:15404-15420` |

Outcome, all paths: `seed: Some(0)` is remapped to `1` (0 would leave the xorshift-style
generator stuck); `Some(nonzero)` passes through unchanged; `None` falls back to
wall-clock nanoseconds since epoch, itself remapped `0 → 1` in the pathological case.
**No divergence.**

### 2. Empty prompt

**RESOLVED (#856, unified on `Err`):** all seven entry points now reject an empty
prompt with the same typed `Err(InferenceError::Inference("empty prompt"))`, via one
shared preflight, `check_prompt_not_empty` (`model/qwen35/generation.rs:2227`). This
replaced each CPU path's own inline copy of the check and added the equivalent call to
the four Metal paths, which previously accepted an empty prompt and returned a normal
empty `Ok` completion (see "Original divergence" below for what that looked like and
why maintainer sign-off ruled the CPU behavior as the contract to keep).

| Path               | Location                                                                                        | Outcome                                                                             |
| ------------------ | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| CPU f16            | `cpu_f16.rs:868` (call site)                                                                    | `Err(InferenceError::Inference("empty prompt"))`                                    |
| CPU q8             | `cpu_q8.rs:721` (call site)                                                                     | `Err(InferenceError::Inference("empty prompt"))`                                    |
| CPU q8 NEON        | `neon_forward.rs:906` (call site)                                                               | `Err(InferenceError::Inference("empty prompt"))`                                    |
| Metal direct       | `metal_qwen35.rs:8711` (call site)                                                              | `Err(InferenceError::Inference("empty prompt"))`                                    |
| Metal streaming    | `metal_qwen35.rs:13250` (call site, in `generate_streaming_with_cancel`)                        | `Err(InferenceError::Inference("empty prompt"))`; `on_token` never invoked          |
| Metal prefix-cache | `metal_qwen35.rs:15985` (call site, in the public wrapper, _before_ `_inner` -- see note below) | `Err(InferenceError::Inference("empty prompt"))`; no cache entry created or evicted |

Note on the prefix-cache path: the guard had to land in the public wrapper
(`generate_streaming_with_prefix_cache_and_cancel`), not in `_inner`, alongside the
existing `check_logprobs_not_set`/`check_mtp_not_requested` preflights (row 5/8's
"CONTRACT, preserve" pattern) — `_inner`'s caller unconditionally evicts the cache slot
on any `Err` it returns, so a preflight-only rejection has to return before `_inner` is
even called, exactly like those two guards. `_inner` no longer carries its own
empty-prompt early-return at all (dead code once the wrapper always rejects first);
removing rather than duplicating it as an `Err` avoided reintroducing the same
destructive-eviction hazard those two existing guards were already routed around.

**Scope note:** "unified" above means the seven `MetalQwen35State`/CPU-forward paths
this document tracks (eight physical call sites once the CPU dispatcher/streaming
entry points that call into the three CPU forward functions are counted separately —
see #915's PR body). The separate, model-agnostic public helper
`lattice_inference::speculative::generate_with_speculation`
(`crates/inference/src/speculative.rs`) also rejects an empty prompt with the same
typed error as of 0.7.0 (#916), but remains outside this document's path count.

Re-verify these line numbers before relying on them; `metal_qwen35.rs` changes
frequently (see the file-level warning at the top of this document).

<details>
<summary>Original divergence (pre-#856), preserved for history</summary>

| Path               | Location (pre-#856)           | Outcome (pre-#856)                                                                                                                                                               |
| ------------------ | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU f16            | `cpu_f16.rs:865-869`          | `Err(InferenceError::Inference("empty prompt"))`                                                                                                                                 |
| CPU q8             | `cpu_q8.rs:718-722`           | `Err(InferenceError::Inference("empty prompt"))`                                                                                                                                 |
| CPU q8 NEON        | `neon_forward.rs:903-907`     | `Err(InferenceError::Inference("empty prompt"))`                                                                                                                                 |
| Metal direct       | `metal_qwen35.rs:8625-8635`   | `Ok(GenerateOutput{ text:"", stopped:false, stop_reason:None, .. })`                                                                                                             |
| Metal streaming    | `metal_qwen35.rs:12982-12995` | Same `Ok` shape as Metal direct; `on_token` never invoked                                                                                                                        |
| Metal prefix-cache | `metal_qwen35.rs:15422-15448` | `Ok(CachedGenerateOutput{ output: GenerateOutput{ stopped:false, stop_reason:None, .. }, cache: CrossTurnCacheStats{ prompt_tokens:0, reused_tokens:0, mode:FullRefill, .. } })` |

**Divergence (was CONTRACT, preserve; now RESOLVED, see above):** all three CPU paths
rejected an empty prompt with a typed `Err`; all three Metal paths accepted it and
returned a normal empty `Ok` completion with `stop_reason: None` -- itself an
invariant-violating result shape (a "completed" generation that neither stopped nor
gives a reason). Question 2 in "Questions for maintainer sign-off" below flagged this
split explicitly; the maintainer ruling (recorded on #856) was to unify on the CPU
`Err` behavior in the shared preflight rather than preserve the split.

</details>

### 3. Zero-budget (`max_new_tokens == 0`)

| Path               | Location                      | Outcome                                                                    |
| ------------------ | ----------------------------- | -------------------------------------------------------------------------- |
| CPU f16            | `cpu_f16.rs:874-884`          | `Ok`, `generated_tokens:0`, `stop_reason:Some(Length)`                     |
| CPU q8             | `cpu_q8.rs:727-737`           | Same shape                                                                 |
| CPU q8 NEON        | `neon_forward.rs:912-922`     | Same shape                                                                 |
| Metal direct       | `metal_qwen35.rs:8640-8650`   | Same shape                                                                 |
| Metal streaming    | `metal_qwen35.rs:13001-13011` | Same shape; comment notes `on_token` never invoked                         |
| Metal prefix-cache | `metal_qwen35.rs:15449-15468` | Same shape, plus `CrossTurnCacheStats{ reused_tokens:0, mode:FullRefill }` |

All seven paths agree: zero budget is a valid request answered before any
prefill/sampling work, with `stop_reason: Some(StopReason::Length)`. **No divergence.**

### 4. Grammar requested

| Path               | Location                                                                                                            | Outcome                                                                                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU f16            | `cpu_f16.rs:890` → `check_grammar_not_set` (`generation.rs:1940-1950`)                                              | `Err(InvalidInput("grammar-constrained decoding is not yet supported on this path; ..."))` before any state allocation                                                                                      |
| CPU q8             | `cpu_q8.rs:742` → same guard                                                                                        | Same error                                                                                                                                                                                                  |
| CPU q8 NEON        | `neon_forward.rs:928` → same guard                                                                                  | Same error                                                                                                                                                                                                  |
| Metal direct       | init `metal_qwen35.rs:8733`; masked at prefill `8739-8751` and each decode step `8901-8912`; route gate `8713-8715` | Supported: masks logits every step, fails closed (`InvalidInput`, "grammar constraint blocked every token...") if the grammar admits nothing; `use_compact` requires `grammar.is_none()`                    |
| Metal streaming    | init `13083`; masked at prefill `13136-13148`, decode `13309-13320`; route gate `13062-13065`                       | Supported, identical masking/fail-closed behavior; `use_compact` requires `grammar.is_none() && logprobs.is_none()`                                                                                         |
| Metal prefix-cache | route gate `15556-15559`; masked each decode step `15834-15845`                                                     | Supported, identical masking/fail-closed behavior; `use_compact` requires `grammar.is_none() && logprobs.is_none()` (the `logprobs.is_none()` half of this AND is always true here in practice — see row 5) |

**Divergence (CONTRACT, already known — #822 "compact-route eligibility"):** Metal
direct's compact-route gate (`8713-8715`) checks `grammar.is_none()` only. Both Metal
streaming (`13062-13065`) and Metal prefix-cache (`15556-15559`) additionally require
`logprobs.is_none()`. All three CPU paths reject grammar outright rather than gating a
route on it.

### 5. Logprobs requested

| Path               | Location                                                              | Outcome                                                                                                                                                                                                                                                |
| ------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CPU f16            | `cpu_f16.rs:893` → `check_logprobs_not_set`                           | `Err(InvalidInput(...))`; `token_logprobs` is always `vec![]` regardless (`cpu_f16.rs:882,960,1015`)                                                                                                                                                   |
| CPU q8             | `cpu_q8.rs:745` → same guard                                          | Same error; `token_logprobs` always `vec![]`                                                                                                                                                                                                           |
| CPU q8 NEON        | `neon_forward.rs:931` → same guard                                    | Same error; `token_logprobs` always `vec![]`                                                                                                                                                                                                           |
| Metal direct       | `metal_qwen35.rs:8669` → same guard, comment `8662-8668`              | `Err(InvalidInput(...))`: this entry point unconditionally returns `token_logprobs: vec![]` on every return path (including MTP/self-spec delegations), so accepting the request would silently drop it                                                |
| Metal streaming    | no `check_logprobs_not_set` call anywhere in `12947-13439` (verified) | **Supported.** `token_logprobs: Vec<TokenLogprob>` populated via `DecodePolicy::init`/`transition` (`13042, 13221-13230, 13347-13359`); every return path returns `token_logprobs: token_logprobs.clone()` (e.g. `13437`), never a hardcoded empty vec |
| Metal prefix-cache | wrapper `15354` → `check_logprobs_not_set`                            | `Err(InvalidInput(...))`, rejected before `_inner` runs at all                                                                                                                                                                                         |

**Divergence (CONTRACT, preserve):** logprobs are supported on exactly one of the
seven paths — Metal streaming (`generate_streaming` / `generate_streaming_with_cancel`).
All six others reject a logprobs request with a typed error rather than silently
dropping it. This is also why Metal streaming's `use_compact` gate (row 4) needs the
extra `logprobs.is_none()` clause: compact mode only tracks a truncated top-k
candidate set, not the full logits a logprobs capture needs, and this is the one path
where a real request for logprobs can reach that gate at all — on Metal direct and
Metal prefix-cache, `gen_cfg.logprobs` is always `None` by the time route selection
runs (rejected upstream), so their identical-looking `logprobs.is_none()` clause is
always true in practice (dead-but-documented condition on prefix-cache, per the
comment at `metal_qwen35.rs:15713-15721`; direct doesn't carry the clause at all
because it has no need to).

### 6. Stop strings requested

| Path               | Location                                                                                                                                                                        | Outcome                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU f16            | `cpu_f16.rs:896` → `check_stop_strings_not_set`                                                                                                                                 | `Err(InvalidInput("stop_strings is not yet supported on this generation path; ..."))`; only EOS-token stopping (`should_stop_token`) exists in the loop |
| CPU q8             | `cpu_q8.rs:748` → same guard                                                                                                                                                    | Same error, same EOS-only loop                                                                                                                          |
| CPU q8 NEON        | `neon_forward.rs:934` → same guard                                                                                                                                              | Same error, same EOS-only loop                                                                                                                          |
| Metal direct       | `metal_qwen35.rs:8846-8853` (construct `StopStringMatcher`), matched at `8855-8873` (prefill token) and `8943-8950` (decode loop)                                               | Supported directly, no guard call                                                                                                                       |
| Metal streaming    | folded into `DecodePolicy`'s `stop_mode`, constructed `13221-13230` with `streaming: true` ("selects `StopMode::Streaming`'s incremental byte-holdback", comment `13216-13220`) | Supported directly, no guard call                                                                                                                       |
| Metal prefix-cache | `DecodePolicy::init(..., true)` at `15728-15737`, comment `15723-15727`                                                                                                         | Supported directly (`streaming: true` selects the same incremental holdback mode), no guard call                                                        |

All three Metal paths support stop strings; all three CPU paths reject them. **This is
the same fault line as row 5**, not an independent divergence — CPU paths have not had
stop-string/reasoning-budget/logprobs wiring added to their decode loops at all
(ADR-080 C3, #783 added the fail-closed guards rather than the wiring). No new
divergence to record beyond what rows 4/5/7/8 already show.

### 7. Reasoning budget requested

| Path               | Location                                                                                                                                                                                                                                                                 | Outcome                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| CPU f16            | `cpu_f16.rs:897` → `check_reasoning_budget_not_set`                                                                                                                                                                                                                      | `Err(InvalidInput(...))` |
| CPU q8             | `cpu_q8.rs:749` → same guard                                                                                                                                                                                                                                             | Same error               |
| CPU q8 NEON        | `neon_forward.rs:935` → same guard                                                                                                                                                                                                                                       | Same error               |
| Metal direct       | `metal_qwen35.rs:8660` → same guard, rationale comment `8652-8659` ("budget-forcing... is not wired into this decode loop, unlike `generate_streaming` below")                                                                                                           | `Err(InvalidInput(...))` |
| Metal streaming    | `think_close_id` resolved conditionally `13087-13091`; threaded into `DecodePolicy::init` (`13221`); consumed via `policy.cap()` (`13286`) and the budget-exhausted break (`13401-13404`); no guard call                                                                 | **Supported**            |
| Metal prefix-cache | same `think_close_id` pattern `15578-15582`; threaded into `DecodePolicy::init` at `15730` (comment `15710-15712`: "shared with `generate_streaming` above and the CPU ... loops" — this last clause about CPU loops is stale/inaccurate, see note below); no guard call | **Supported**            |

**Divergence (CONTRACT, preserve):** reasoning budget is supported on Metal streaming
and Metal prefix-cache only. Metal direct and all three CPU paths reject it.

Note for maintainer attention (not resolved here): the comment at
`metal_qwen35.rs:15710-15712` describes the reasoning-budget policy as "shared ...
with the CPU `model::qwen35::generation` loops," but per row 7 above, none of the three
CPU paths in scope for this matrix (`generate_f16`, `generate_q8`, `generate_q8_neon`)
implement reasoning-budget forcing — they reject it via `check_reasoning_budget_not_set`.
The comment may be referring to a different CPU entry point not in this matrix's scope
(e.g. `Qwen35Model::generate` in `model/qwen35/generation.rs` itself, which several of
the guard doc-comments describe as implementing these features directly) rather than
being wrong — this needs a maintainer check, see the questions section.

### 8. `enable_mtp` requested

| Path               | Location                                                                                                                                                                                                           | Outcome                                                                                                                                                                                                                                                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU f16            | no code path (verified by full-file grep; `mtp_*` appears only in test-fixture `Qwen35Config` literals)                                                                                                            | Field is never read; no effect, no error                                                                                                                                                                                                                                                                                                                 |
| CPU q8             | same (verified by grep)                                                                                                                                                                                            | Field is never read; no effect, no error                                                                                                                                                                                                                                                                                                                 |
| CPU q8 NEON        | same (verified by grep)                                                                                                                                                                                            | Field is never read; no effect, no error                                                                                                                                                                                                                                                                                                                 |
| Metal direct       | read/acted on directly at `metal_qwen35.rs:8754-8774`: `mtp_enabled = gen_cfg.enable_mtp.unwrap_or_else(\|\| LATTICE_MTP env set)`; `use_mtp = mtp_route_active(...)`; if true, delegates to `generate_greedy_mtp` | **Supported** as a route decision, not a guard: if `mtp_route_active`'s other preconditions (`self.session.mtp.is_some()`, greedy decoding, `!use_compact`, no grammar/stop_strings/reasoning_budget/logprobs, `repetition_penalty == 1.0`) aren't all met, `enable_mtp: true` is silently _not_ honored — falls through to plain sampling with no error |
| Metal streaming    | zero matches for `enable_mtp` / `check_mtp_not_requested` / `mtp_route_active` in the full function body `12947-13439` (verified by grep)                                                                          | Field is never read; no effect, no error — MTP is completely inert on this path                                                                                                                                                                                                                                                                          |
| Metal prefix-cache | wrapper `15355` → `check_mtp_not_requested` (`generation.rs:2042-2056`), same `enable_mtp`/`LATTICE_MTP` resolution as Metal direct                                                                                | **Rejected**: `Err(InvalidInput("enable_mtp (or LATTICE_MTP) is not supported on the cross-turn prefix-cache generation path, which has no MTP draft/verify wiring; use the Metal generate() / generate_streaming() paths, which implement MTP"))`                                                                                                       |

**Divergence (CONTRACT, partially already known):** the issue's "known cross-path
discrepancies" section documents that `check_mtp_not_requested` has exactly one call
site (Metal prefix-cache) and that no other path calls it. What is not yet documented
prior to this matrix: **among the six paths that don't call the guard, only Metal
direct actually implements an MTP fast path.** The three CPU paths and Metal streaming
all simply never look at `gen_cfg.enable_mtp` — a caller requesting MTP on those four
paths gets ordinary sampling with no indication MTP was skipped, which is a different
outcome from Metal direct's route-eligibility fallthrough (same "greedy accept, no
error" surface behavior, but for different reasons: total absence of the field vs. an
unmet route precondition) and a different outcome again from Metal prefix-cache's hard
rejection. **This third-way split (implements it / silently ignores it / hard-rejects
it) is flagged below as a question for maintainer sign-off** — it is not called out in
the issue as an existing intentional discrepancy the way the compact-route and
context-bound splits are.

### 9. Over-context request (`prompt_len` alone vs `prompt_len + max_new_tokens`)

| Path               | Location                                                                | Comparison                                                                                               | On violation                                                                                                                                                             |
| ------------------ | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CPU f16            | `cpu_f16.rs:905-912`                                                    | `prompt_len.saturating_add(gen_cfg.max_new_tokens) > max_context` (`max_context = rope.max_positions()`) | `Err(InferenceError::Inference("prompt ({prompt_len} tokens) plus max_new_tokens ({n}) exceeds model context window ({max_context})"))`                                  |
| CPU q8             | `cpu_q8.rs:757-764`                                                     | Same comparison                                                                                          | Same error text                                                                                                                                                          |
| CPU q8 NEON        | `neon_forward.rs:940-947`                                               | Same comparison                                                                                          | Same error text                                                                                                                                                          |
| Metal direct       | `metal_qwen35.rs:8675-8685`                                             | `prompt_len > self.max_context()` (`max_context()` = `session.kv_cache.max_cache_len`)                   | `Ok(GenerateOutput{ stopped:true, stop_reason:Some(KvFull), .. })` — **not an error**                                                                                    |
| Metal streaming    | `metal_qwen35.rs:13023-13033`, extended rationale comment `13013-13022` | Same `prompt_len` alone comparison                                                                       | Same `Ok`/`KvFull` shape                                                                                                                                                 |
| Metal prefix-cache | `metal_qwen35.rs:15469-15488`                                           | Same `prompt_len` alone comparison                                                                       | `Ok(CachedGenerateOutput{ output: GenerateOutput{ stopped:true, stop_reason:Some(KvFull), .. }, cache: CrossTurnCacheStats{ mode:FullRefill, .. } })` — **not an error** |

**Divergence (CONTRACT, already tracked as #743, open — do not fold a fix into this
matrix):** all three CPU paths bound `prompt_len + max_new_tokens` and fail with a
typed `Err`. All three Metal paths bound `prompt_len` alone and fail _open_ by GPU
convention — returning a normal `Ok` result with `stop_reason: KvFull` rather than an
error — meaning a Metal path will admit and completely run a request the CPU paths
would reject at entry, only discovering the shortfall (if it occurs) mid-decode via
each loop's own `seq_len >= max_cache_len` break (Metal direct: `8886-8889`/
`8952-8955`; Metal streaming: `13298-13301`/`13397-13400`).

### 10. Prefix-cache mutation-on-error (does a rejected/early-exit request leave a warmed cache slot untouched?)

| Path               | Applicability                                                                                                                                                                                                                  | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU f16            | N/A — no persistent prefix cache parameter exists on this path (verified: zero matches for `prefix_cache`/`PrefixCache` in the file)                                                                                           | All mutable state (`gdn_states`, `kv_cache`, `scratch`) is freshly constructed _after_ every guard (`cpu_f16.rs:917-921`, guards at `865,874,890-897,906`). Nothing to corrupt.                                                                                                                                                                                                                                                                                                                                                                                                        |
| CPU q8             | Same — N/A                                                                                                                                                                                                                     | Same ordering (`cpu_q8.rs:769-773` after guards at `718,727,742-749,757`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| CPU q8 NEON        | Same — N/A, but this path additionally pre-reserves capacity (`kv_cache.reserve`, `scratch.ensure_capacity`, `neon_forward.rs:957-962`) after the context guard — still strictly post-validation, still on freshly-owned state | Same conclusion                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Metal direct       | Owns `self.cross_turn_prefix_cache` indirectly via `reset_state()` (`metal_qwen35.rs:9298-9337`), called at `8688`                                                                                                             | The three early-return guards (empty prompt `8625`, zero-budget `8640`, over-context `8675`) all return _before_ `reset_state()` at `8688`, so a rejected/short-circuited request never reaches it. **But**: every _accepted_ call unconditionally clears the entire `cross_turn_prefix_cache` as a side effect of `reset_state()` (`metal_qwen35.rs:9337`, `self.cross_turn_prefix_cache.clear()`) — this is a full-cache wipe, not scoped to one slot, and it is not conditioned on success/failure of the generation that follows it, only on having passed the three early guards. |
| Metal streaming    | Same `reset_state()` call, at `13035`, after the same three early-return guards (`12982, 13001, 13023`)                                                                                                                        | Same finding as Metal direct: rejected requests don't reach `reset_state()`; any accepted call wipes the whole cross-turn cache regardless of whether decoding then succeeds.                                                                                                                                                                                                                                                                                                                                                                                                          |
| Metal prefix-cache | Owns `self.cross_turn_prefix_cache` directly, per-slot (`slot_id`)                                                                                                                                                             | See detailed walkthrough below. Summary: validation strictly precedes mutation for every guard; the wrapper/`_inner` split exists specifically to keep preflight-only rejections from ever reaching the destructive per-slot eviction path.                                                                                                                                                                                                                                                                                                                                            |

**Metal prefix-cache walkthrough** (the path this behavior matters most for, since it's
the only one of the seven with a real per-slot cross-turn cache):

The wrapper (`generate_streaming_with_prefix_cache_and_cancel`, `:15330`) runs
`check_logprobs_not_set` (`:15354`) and `check_mtp_not_requested` (`:15355`) _before_
calling `_inner`, then wraps the `_inner` call:

```rust
match self.generate_streaming_with_prefix_cache_and_cancel_inner(...) {
    Ok(out) => Ok(out),
    Err(e) => {
        self.reset_state();
        self.cross_turn_prefix_cache.remove(slot_id);
        Err(e)
    }
}
```

That `Err` arm is unconditional — _any_ error from `_inner` evicts the slot, on the
assumption that an error means the recurrent state and the cache now potentially
disagree. The wrapper doc comment (`:15343-15353`) states the reason the two preflight
checks live here rather than inside `_inner`: if they ran inside `_inner` (after
`_inner`'s own reuse-planning/restore code had already touched the slot), their
rejection would flow through this same `Err` arm and evict a slot the request never
should have been allowed to touch in the first place — including a slot from a
completely unrelated prior turn that happened to share the same `slot_id`. `_inner`'s
mirrored comment (`:15393-15400`) confirms the same reasoning from the callee side.

Inside `_inner`, the remaining three early-return cases — empty prompt (`:15429`),
zero-budget (`:15449`), and over-context (`:15469`) — all return `Ok(...)` before line
`15491`'s `plan_cross_turn_reuse` call (the first point at which the cache's
reuse-planning, and hence any mutation, could begin). Because they return `Ok`, they
never enter the wrapper's `Err` arm either. So across all five rejection/early-exit
behaviors on this path (logprobs, enable_mtp, empty prompt, zero-budget,
over-context), none commits a cache mutation before its guard fires, and the two paths
that DO route through the wrapper's blanket post-`_inner`-error eviction (logprobs,
enable_mtp) are specifically kept out of `_inner` so that blanket eviction never
executes on their account.

**No divergence in the strict "does an error leave a stale mutation behind" sense** —
across all seven paths, no guard fires after a mutation it should have preceded. The
one behavior worth flagging as a real difference in blast radius (not an error-path
bug, but adjacent and worth maintainer awareness) is under questions below.

## Questions for maintainer sign-off

These are observations this matrix surfaced that are **not** already covered by the
issue's pre-documented "known cross-path discrepancies" (CPU/Metal context bound #743,
`enable_mtp` preflight scope, compact-route eligibility). None of them are resolved
here; they are recorded for explicit maintainer judgment on whether each is CONTRACT
(preserve as-is) or an accident worth its own follow-up issue.

1. **Three-way `enable_mtp` split, not just "one path checks, others don't."** Row 8:
   of the six paths that don't call `check_mtp_not_requested`, Metal direct actually
   _implements_ MTP behind a route-eligibility fallthrough (silently falls through to
   plain sampling if any precondition is unmet), while the three CPU paths and Metal
   streaming have _zero_ code referencing `enable_mtp` at all (inert field). Is the
   "implements it opportunistically, else silently plain-samples" behavior on Metal
   direct intentional, or should an unmet-precondition MTP request surface some signal
   to the caller (as it does, hard, on Metal prefix-cache)?

2. **RESOLVED (#856).** Empty-prompt contract split (row 2) was a hard `Err` vs `Ok`
   divergence, independent of the already-tracked over-context split (row 9): all
   three CPU paths rejected an empty prompt; all three Metal paths accepted it and
   returned an empty completion. Maintainer ruling (recorded on #856): not
   intentional -- unify on the CPU `Err` behavior via a shared preflight, since the
   Metal `Ok` shape (`stopped: false, stop_reason: None`) is itself
   invariant-violating (a "completed" generation that neither stopped nor gives a
   reason), not a deliberate no-op contract for interactive callers. See row 2 above
   for the resulting shared guard and the caller audit performed before landing it.

3. **Stale-looking doc comment at `metal_qwen35.rs:15710-15712`** (row 7): claims the
   reasoning-budget policy is "shared ... with the CPU `model::qwen35::generation`
   loops," but none of the three CPU paths in this matrix's scope
   (`generate_f16`/`generate_q8`/`generate_q8_neon`) implement reasoning-budget
   forcing — they reject it. Is this comment referring to a different CPU entry point
   outside this matrix's seven (e.g. `Qwen35Model::generate` in
   `model/qwen35/generation.rs`), or is it stale and should be corrected in a follow-up
   docs-only PR?

4. **Full-cache wipe as a side effect of any accepted Metal direct/streaming call**
   (row 10): `reset_state()` unconditionally clears the _entire_
   `cross_turn_prefix_cache` (not scoped to any one slot) on every call to
   `generate`/`generate_streaming` that passes the three early guards, per the comment
   at `metal_qwen35.rs:9329-9336` ("every public path that resets live KV/GDN state...
   routes through here, so this is the single place that must invalidate any retained
   cross-turn entry"). This appears to be intentional invalidation (the comment cites
   #516 remediation D3), but it means a caller alternating between
   `generate()`/`generate_streaming()` and `generate_streaming_with_prefix_cache_and_cancel()`
   on the same `MetalQwen35State` will silently lose all cross-turn cache entries the
   moment it calls the former. Worth an explicit maintainer confirmation that this is
   the intended lifecycle boundary, since it's easy for a future caller to trip over.

## Summary table (compact cross-reference)

| Behavior                | CPU f16                              | CPU q8          | CPU q8 NEON     | Metal direct                                                    | Metal streaming (wrapper)                                        | Metal streaming (impl) | Metal prefix-cache                                                                                                                 |
| ----------------------- | ------------------------------------ | --------------- | --------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Seed 0/nonzero/unset    | remap 0→1, time fallback             | same            | same            | same                                                            | same                                                             | same                   | same                                                                                                                               |
| Empty prompt (#856)     | `Err`                                | `Err`           | `Err`           | `Err`                                                           | `Err`                                                            | `Err`                  | `Err`                                                                                                                              |
| Zero budget             | `Ok` Length                          | `Ok` Length     | `Ok` Length     | `Ok` Length                                                     | `Ok` Length                                                      | `Ok` Length            | `Ok` Length                                                                                                                        |
| Grammar                 | `Err`                                | `Err`           | `Err`           | supported, route needs `grammar.is_none()`                      | supported, route needs `grammar.is_none() && logprobs.is_none()` | same as wrapper        | supported, route needs `grammar.is_none() && logprobs.is_none()`                                                                   |
| Logprobs                | `Err`                                | `Err`           | `Err`           | `Err`                                                           | **supported**                                                    | **supported**          | `Err`                                                                                                                              |
| Stop strings            | `Err`                                | `Err`           | `Err`           | supported                                                       | supported                                                        | supported              | supported                                                                                                                          |
| Reasoning budget        | `Err`                                | `Err`           | `Err`           | `Err`                                                           | **supported**                                                    | **supported**          | **supported**                                                                                                                      |
| `enable_mtp`            | inert (no code)                      | inert (no code) | inert (no code) | implemented, silent fallthrough if ineligible                   | inert (no code)                                                  | inert (no code)        | rejected, `Err`                                                                                                                    |
| Over-context bound      | `prompt_len + max_new_tokens`, `Err` | same            | same            | `prompt_len` alone, `Ok`/KvFull                                 | same                                                             | same                   | same                                                                                                                               |
| Cache mutation on error | N/A, no cache                        | N/A             | N/A             | guard precedes `reset_state()`; accepted calls wipe whole cache | same                                                             | same                   | validation strictly precedes mutation; wrapper/`_inner` split isolates preflight rejections from the destructive per-slot eviction |
