//! Neutral generation types: the request configuration and the result struct that every
//! decoder path in this crate produces, independent of model family.
//!
//! These four types ([`GenerateConfig`], [`GenerateOutput`], [`TokenLogprob`], [`TopLogprob`])
//! were defined in `model::qwen35_config` until they were moved here under ADR-092, which makes
//! the crate-root path canonical: `lattice_inference::GenerateConfig`. The old
//! `model::qwen35_config` paths still resolve, through deprecated aliases, so nothing downstream
//! breaks.
//!
//! **One dependency points the other way, and it is deliberate.** [`GenerateConfig`]'s
//! [`Default`] seeds `stop_token_ids` with `QWEN_CHAT_IM_END_TOKEN_ID`, a Qwen chat-template
//! token id owned by `model::qwen35_config`. Making the default neutral would silently stop 75
//! call sites treating `im_end` as a stop token, so the import stays and is documented here
//! rather than removed. The type is neutral in shape; its default is not, and a reader deciding
//! what `GenerateConfig::default()` means needs to know that without reading the impl.
//!
//! Not to be confused with `model::qwen35::generation`, which is the Qwen CPU decode loop.

use crate::grammar::GrammarEngine;
use crate::model::qwen35::stop_strings::{
    StopStringMatcher, earliest_stop_match_from, stop_scan_search_start,
};
use crate::model::qwen35_config::{QWEN_CHAT_IM_END_TOKEN_ID, decode_cap, force_close_think};
use crate::sampling::compute_step_logprobs;
use crate::stop_reason::StopReason;
use std::sync::Arc;

/// **Unstable**: sampling configuration for text generation; fields may expand.
#[derive(Clone)]
#[non_exhaustive]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Min-p: keep tokens with probability at least `min_p * max_probability`,
    /// applied before top-p. 0.0 or NaN = disabled (the default); other
    /// values clamp to `[0.0, 1.0]`. See `crate::sampling::SamplingConfig::min_p`
    /// and `crate::sampling::Sampler::with_min_p` for the shared semantics.
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// Random seed for sampling. `None` = seed from system time.
    pub seed: Option<u64>,
    /// Additional stop token IDs (beyond EOS). Generation stops on any of these.
    pub stop_token_ids: Vec<u32>,
    /// When false, the caller is responsible for priming the prompt so no reasoning block
    /// is produced; for Qwen that is the `QWEN3_NO_THINK_PREFIX` token sequence in
    /// `model::qwen35_config`. This flag does not prime the prompt itself.
    pub enable_thinking: bool,
    /// Enable multi-token prediction when the model has MTP weights loaded.
    /// Replaces the `LATTICE_MTP` env var for programmatic control.
    /// `None` = defer to `LATTICE_MTP` env var (backwards-compatible default).
    pub enable_mtp: Option<bool>,
    /// Optional grammar-constrained decoding engine (ADR-046).
    ///
    /// When set, `mask_logits` is called on CPU logits before sampling on every step.
    /// The Metal path copies logits to CPU before sampling — no additional GPU transfer needed.
    pub grammar: Option<Arc<GrammarEngine>>,
    /// Additional string-level stop sequences. When any appears in the output, generation
    /// halts and the matched text is excluded. Empty = disabled (default; parity-safe).
    pub stop_strings: Vec<String>,
    /// Reasoning-budget forcing (s1-style): after this many reasoning tokens are
    /// generated without a `</think>`, force-inject `</think>` to commit the model
    /// to an answer. `None`, `Some(0)`, or [`enable_thinking`](Self::enable_thinking)
    /// `== false` = disabled (no behaviour change) -- with thinking disabled there is
    /// no reasoning block for a forced `</think>` to close, so the budget is inert
    /// regardless of its value. See `GenerateConfig::effective_reasoning_budget`.
    pub reasoning_budget: Option<usize>,
    /// Capture per-token log-probabilities (OpenAI `logprobs`/`top_logprobs`).
    /// `None` (default) disables capture entirely -- no extra allocation or
    /// computation is added to the decode loop. `Some(n)` captures the
    /// sampled token's log-probability plus its `n` highest-probability
    /// alternatives at every generated step (`n == 0` is valid: report only
    /// the sampled token's log-probability, no alternatives).
    pub logprobs: Option<usize>,
}

impl std::fmt::Debug for GenerateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerateConfig")
            .field("max_new_tokens", &self.max_new_tokens)
            .field("temperature", &self.temperature)
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .field("min_p", &self.min_p)
            .field("repetition_penalty", &self.repetition_penalty)
            .field("seed", &self.seed)
            .field("stop_token_ids", &self.stop_token_ids)
            .field("enable_thinking", &self.enable_thinking)
            .field("enable_mtp", &self.enable_mtp)
            .field("grammar", &self.grammar.as_ref().map(|_| "<GrammarEngine>"))
            .field("stop_strings", &self.stop_strings)
            .field("reasoning_budget", &self.reasoning_budget)
            .field("logprobs", &self.logprobs)
            .finish()
    }
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.1,
            seed: None,
            stop_token_ids: vec![QWEN_CHAT_IM_END_TOKEN_ID],
            enable_thinking: true,
            enable_mtp: None,
            grammar: None,
            stop_strings: vec![],
            reasoning_budget: None,
            logprobs: None,
        }
    }
}

impl GenerateConfig {
    /// `reasoning_budget` as every decode-path consumer must see it: `None`
    /// whenever `enable_thinking` is false, since a budget without a
    /// reasoning block to close is inert (see the field's own doc).
    ///
    /// This is the single point where that contract is enforced. Every
    /// `decode_cap` / `check_context_budget` / `DecodePolicy` call site reads
    /// this instead of the raw `reasoning_budget` field, so a future
    /// consumer that does the same is correct by construction instead of
    /// having to re-derive the `enable_thinking` gate itself.
    pub(crate) fn effective_reasoning_budget(&self) -> Option<usize> {
        self.reasoning_budget.filter(|_| self.enable_thinking)
    }
}

/// One alternative token considered at a single generation step, paired with
/// its log-probability under the reporting distribution computed by
/// [`crate::sampling`]'s logprobs support. Ordered by descending probability
/// when produced via [`GenerateConfig::logprobs`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopLogprob {
    pub token_id: u32,
    pub logprob: f32,
}

/// Per-token log-probability data for one generated token. Only populated
/// when [`GenerateConfig::logprobs`] is `Some`; `GenerateOutput::token_logprobs`
/// stays empty otherwise (no extra allocation on the default path).
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    /// The token id that was actually sampled/generated at this step.
    pub token_id: u32,
    /// Natural-log probability of `token_id` under a temperature-scaled
    /// softmax over that step's raw logits.
    pub logprob: f32,
    /// The requested number of highest-probability alternatives at this step
    /// (may include `token_id` itself), sorted by descending probability.
    /// Empty when `logprobs` was requested with a `top_logprobs` count of 0.
    pub top: Vec<TopLogprob>,
}

/// **Unstable**: text generation output struct; fields may expand with streaming support.
///
/// # Stop-token contract (#613)
///
/// When generation ends because EOS or a configured `stop_token_ids` entry is
/// hit, that terminating token is **excluded** from `token_ids` and `text` —
/// it is never appended to the output. Every generation entry point across
/// this crate (CPU and Metal) honours this contract (see the
/// `stop_token_contract` test module for the cross-path regression sweep).
/// `generated_tokens` always equals `token_ids.len()`.
///
/// **`stop_strings` behave differently (#632).** A
/// `stop_strings` match truncates `text` to the point where the match begins,
/// but the token(s) whose decoded text completed the match are **not**
/// removed from `token_ids`/`generated_tokens` — the implementation cannot
/// "un-generate" a token once it has been decoded and appended (see
/// `decode_loop_with_stops` / `earliest_stop_match` in
/// `crate::model::qwen35::generation`). So for a `stop_strings` stop,
/// `token_ids.len()` (== `generated_tokens`) can exceed the number of tokens
/// whose text actually survived in the truncated `text`. The EOS /
/// `stop_token_ids` exclusion guarantee above does not extend to this case.
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    /// Generated text (excluding prompt).
    pub text: String,
    /// Generated token IDs. Excludes the terminating token for EOS /
    /// `stop_token_ids` stops (see the stop-token contract above), but a
    /// `stop_strings` stop retains the token(s) that completed the match —
    /// see the `stop_strings` note above.
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Total tokens generated (excluding prompt).
    pub generated_tokens: usize,
    /// True when generation ended via a stop condition (EOS, a stop token, or a
    /// stop string); false when it ended by reaching `max_new_tokens`. Serve maps
    /// this to the OpenAI `finish_reason` ("stop" vs "length").
    pub stopped: bool,
    /// Why generation terminated. `Some` on every real generation exit; `None` only on
    /// non-generation returns that have no issue-listed cause.
    pub stop_reason: Option<StopReason>,
    /// Per-step log-probability data, one entry per generated token, in
    /// generation order. Empty unless `GenerateConfig::logprobs` was `Some`.
    pub token_logprobs: Vec<TokenLogprob>,
}

/// Outcome of [`DecodePolicy::transition`], the one per-step call every decode
/// loop drives through (ADR-080 C3, PR #787).
pub(crate) enum StepOutcome {
    /// The (possibly budget-overridden) token was rejected by the backend's
    /// own grammar advance before ever being pushed. The loop must stop with
    /// `stopped = true`, `stop_reason = Grammar`.
    GrammarStop,
    /// The (possibly budget-overridden) token is EOS / a stop-token id and
    /// was never pushed. The loop must stop with `stopped = true`,
    /// `stop_reason = Eos`.
    Eos,
    /// [`DecodePolicy::stop_check`] — driven internally from `self.stop_mode`
    /// (PR #787; see [`StopMode`]) —
    /// reported that a configured stop string matched as of this token. The
    /// token was pushed (via `push`) and every other backend-neutral
    /// per-step control already applied before `stop_check` ran; the loop
    /// must stop with `stopped = true`, `stop_reason = Eos`.
    Stopped,
    /// [`DecodePolicy::stop_check`] reported that the caller's
    /// streaming sink (`emit_confirmed`) can no longer consume output (e.g. a
    /// dropped SSE receiver) — not a stop condition. The loop must stop with
    /// `stopped = false`, `stop_reason = Interrupt`.
    Interrupted,
    /// The token was pushed (via the caller's `push` callback), `stop_check`
    /// reported [`StopCheckOutcome::Continue`], and every backend-neutral
    /// per-step control (logprobs, reasoning-end capture, answer-budget
    /// accounting) has already been applied for it.
    Emitted {
        /// The actually-emitted token id (post budget-override).
        token_id: u32,
        /// Whether the answer-budget window has closed as of this token —
        /// the loop should break after this iteration (in addition to its
        /// normal cap) when this is `true`.
        answer_budget_exhausted: bool,
    },
}

/// Outcome of the mandatory per-step stop-check [`DecodePolicy::transition`]
/// drives internally (PR #787: a mandatory
/// `stop_check` closure could still compile as a trivial
/// `|_, _| StopCheckOutcome::Continue` for a configuration that actually had
/// stop strings — an arbitrary outcome-producing closure cannot be forced to
/// consult real matcher state. `transition` no longer accepts one at all; see
/// [`StopMode`] and [`DecodePolicy::stop_check`]).
pub(crate) enum StopCheckOutcome {
    /// No stop-string match yet (or `stop_strings` is not configured for
    /// this generation at all) — keep decoding.
    Continue,
    /// A configured stop string matched as of this token. The callback has
    /// already truncated/finalized the backend's own accumulated output
    /// (text buffer or streaming sink) before returning this.
    Stopped,
    /// The caller's streaming sink (`on_token`) signaled it can no longer
    /// consume output — not a stop condition.
    Interrupted,
}

/// Backend-neutral decode-policy state (ADR-080 C3): reasoning-budget
/// accounting and logprobs formatting, shared by every canonical/streaming
/// decode loop — CPU [`decode_loop`], [`decode_loop_with_stops`], both
/// branches of [`Qwen35Model::generate_streaming_with_cancel`], and the Metal
/// `generate_streaming` / `generate_streaming_with_prefix_cache_and_cancel_inner`
/// loops in `crate::forward::metal_qwen35` — via one atomic per-step
/// transition ([`DecodePolicy::transition`]): each backend keeps
/// `forward_step`, grammar masking, sampling, and its own token vectors
/// (`generated_ids` / `all_ids` or the Metal equivalents) entirely to itself,
/// hands `transition` the token its own pipeline just sampled plus three
/// backend callbacks (grammar-advance, EOS/stop-token check, the push into
/// its own vectors) and raw per-token I/O primitives for the stop check
/// (`decode_delta`, a `text`/`token_logprob_end_offsets` buffer pair, and
/// `emit_confirmed` — see [`StopMode`] below), and gets back a
/// [`StepOutcome`] that already reflects budget-override, reasoning-block
/// tracking, logprobs recording, reasoning-end capture, the stop check, and
/// the answer-budget check — in that fixed order, every time, for every
/// site.
///
/// Before this struct existed, this exact bookkeeping (`think_close_id`
/// resolution, `thinking_closed` / `reasoning_end_len` tracking, the
/// `decode_cap` / `force_close_think` calls, and the answer-budget break
/// condition) was hand-duplicated across six independent decode loops —
/// exactly the drift ADR-080 C3 exists to prevent: a seventh loop could add
/// its own copy and silently diverge from the other six. The struct
/// originally exposed each of these as a separate method
/// (`apply_override` / `note_emitted` / `record_logprob` /
/// `capture_reasoning_end` / `answer_budget_exhausted`), which let a call
/// site choreograph a subset of them and skip another — that was exactly the
/// failure mode observed live: the Metal prefix-cache loop
/// called four of the five and silently never called `record_logprob`.
/// `transition` replaces all five with the one call above; the five
/// constituent methods are now private to this module, so a caller in a
/// different module (e.g. `crate::forward::metal_qwen35`) cannot reach any of
/// them individually even by mistake — omitting `transition` is the only way
/// to skip a control, and doing so breaks every one of these behaviors at
/// once rather than silently dropping just one.
///
/// Stop-string matching (PR #787): the streaming vs non-streaming consumption shapes genuinely
/// differ (incremental byte-holdback via [`StopStringMatcher`] vs full-text
/// rescan via `earliest_stop_match_from`), but which one applies — and
/// whether checking happens at all — is now [`StopMode`], a value chosen
/// exactly once from the real `gen_cfg.stop_strings` at [`DecodePolicy::init`]
/// time and stored privately on the policy. A caller can no longer supply a
/// closure that *decides* the stop outcome (a prior `stop_check` parameter,
/// which could compile as a trivial `|_, _| Continue` for any configuration
/// regardless of what `stop_strings` actually held); it supplies only raw
/// per-token I/O primitives — a decoded delta (`decode_delta`) and a
/// confirmed-text sink (`emit_confirmed`) — and [`DecodePolicy::stop_check`]
/// (called from both [`DecodePolicy::check_initial_stop`], for the
/// prefill-derived first token, and `transition`, for every token after)
/// dispatches on `self.stop_mode` to decide, using the real adapter for that
/// mode, not caller-supplied decision logic.
pub(crate) struct DecodePolicy {
    reasoning_budget: Option<usize>,
    enable_thinking: bool,
    max_new_tokens: usize,
    logprobs: Option<usize>,
    think_close_id: Option<u32>,
    thinking_closed: bool,
    reasoning_end_len: Option<usize>,
    stop_mode: StopMode,
}

/// The stop-string check adapter a [`DecodePolicy`] owns (PR #787). The only place a value of this
/// type is ever produced is the private [`StopMode::for_config`], called once
/// from [`DecodePolicy::init`] on the real `gen_cfg.stop_strings` — there is
/// no public constructor, so a caller cannot independently choose (or swap
/// in) `Disabled` for a configuration that actually has stop strings: the
/// variant a given policy drives is fixed by the config it was built from,
/// not by anything a call site writes.
enum StopMode {
    /// `gen_cfg.stop_strings` was empty at construction — there is nothing to
    /// match, so [`DecodePolicy::stop_check`] only threads decoded text
    /// through to the caller's sink (still needed for streaming callers'
    /// `on_token`; a no-op for `decode_loop`, which has no text pipeline at
    /// all).
    Disabled,
    /// Streaming incremental byte-holdback: the owned [`StopStringMatcher`]
    /// ensures a partial match never reaches the caller's confirmed-text
    /// sink. Used by every streaming call site with `stop_strings` set (CPU
    /// `generate_streaming_with_cancel`'s stop-string branch, both Metal
    /// streaming loops).
    Streaming(StopStringMatcher),
    /// Non-streaming full-text rescan, bounded to the suffix that could
    /// contain a new match (`stop_scan_search_start`). Used only by CPU
    /// `decode_loop_with_stops` (via `Qwen35Model::generate`'s stop-string
    /// branch), which has no external consumer to hold text back from.
    FullScan {
        stop_strings: Vec<String>,
        max_stop: usize,
    },
}

impl StopMode {
    fn for_config(stop_strings: &[String], streaming: bool) -> Self {
        if stop_strings.is_empty() {
            StopMode::Disabled
        } else if streaming {
            StopMode::Streaming(StopStringMatcher::new(stop_strings))
        } else {
            let max_stop = stop_strings.iter().map(String::len).max().unwrap_or(1);
            StopMode::FullScan {
                stop_strings: stop_strings.to_vec(),
                max_stop,
            }
        }
    }
}

impl DecodePolicy {
    /// The first-step transition (PR #787): constructs the policy AND atomically records the
    /// prefill-derived first token's logprob in the same call, so there is no
    /// longer any way to build a `DecodePolicy` without also recording its
    /// first token's logprob. Before this, `new()` only built the struct and
    /// left every call site to separately invoke the freestanding
    /// `crate::sampling::record_logprob` for that one token — three
    /// call sites (this module's `generate()` / `generate_streaming_with_cancel()`,
    /// and Metal's `generate_streaming`) duplicated that call independently,
    /// the exact drift pattern already proved live
    /// once for the *other* four constituent methods (see the struct-level
    /// doc comment above).
    ///
    /// `think_close_id` is resolved by the caller with [`resolve_reasoning_close_token`]
    /// since each backend reaches its tokenizer differently.
    /// `first_emitted_id` / `first_generated_len` seed
    /// `thinking_closed` / `reasoning_end_len` from the token already sampled and pushed
    /// before the decode loop starts (the prefill-derived first token), covering the
    /// `reasoning_budget == 1` edge case exactly as the six duplicated call sites did.
    /// `first_logits` / `temperature` are the same values the free-function
    /// `record_logprob` call used to take directly. `streaming` selects which
    /// [`StopMode`] a non-empty `gen_cfg.stop_strings` resolves to
    /// (`Streaming`'s incremental holdback vs `FullScan`'s full-text rescan;
    /// see [`StopMode::for_config`]) — pass `true` for every streaming caller
    /// (CPU `generate_streaming_with_cancel`, both Metal streaming loops),
    /// `false` for non-streaming callers (`Qwen35Model::generate`). An empty
    /// `stop_strings` always resolves to `Disabled` regardless of `streaming`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn init(
        gen_cfg: &GenerateConfig,
        think_close_id: Option<u32>,
        token_logprobs: &mut Vec<TokenLogprob>,
        first_emitted_id: u32,
        first_logits: &[f32],
        temperature: f32,
        first_generated_len: usize,
        streaming: bool,
    ) -> Self {
        let thinking_closed = Some(first_emitted_id) == think_close_id;
        let reasoning_end_len = if thinking_closed {
            Some(first_generated_len)
        } else {
            None
        };
        let policy = Self {
            reasoning_budget: gen_cfg.effective_reasoning_budget(),
            enable_thinking: gen_cfg.enable_thinking,
            max_new_tokens: gen_cfg.max_new_tokens,
            logprobs: gen_cfg.logprobs,
            think_close_id,
            thinking_closed,
            reasoning_end_len,
            stop_mode: StopMode::for_config(&gen_cfg.stop_strings, streaming),
        };
        policy.record_logprob(token_logprobs, first_logits, first_emitted_id, temperature);
        policy
    }

    /// Total decode-loop iteration cap (`rb + max_new_tokens + 1` when budgeted,
    /// `max_new_tokens` otherwise) — see [`decode_cap`].
    pub(crate) fn cap(&self) -> usize {
        decode_cap(self.reasoning_budget, self.max_new_tokens)
    }

    /// Overrides `sampled_id` with the forced `</think>` token when the reasoning
    /// budget is exhausted and the block is still open; a no-op pass-through
    /// otherwise. Call after sampling, before grammar-advance (the actually-emitted
    /// token, post-override, is what grammar must advance on).
    ///
    /// Private (PR #787): only reachable through
    /// [`DecodePolicy::transition`], which owns the full per-step ordering.
    fn apply_override(&self, generated_len: usize, sampled_id: u32) -> u32 {
        force_close_think(
            self.reasoning_budget,
            self.enable_thinking,
            self.thinking_closed,
            generated_len,
            self.think_close_id,
        )
        .unwrap_or(sampled_id)
    }

    /// Marks the thinking block closed when `next_id` (the actually-emitted,
    /// post-override token) is the `</think>` token. Call after grammar-advance
    /// succeeds, before the EOS/stop-token check — mirrors the original inline
    /// ordering across all six sites.
    ///
    /// Private (PR #787): only reachable through
    /// [`DecodePolicy::transition`], which owns the full per-step ordering.
    fn note_emitted(&mut self, next_id: u32) {
        if Some(next_id) == self.think_close_id {
            self.thinking_closed = true;
        }
    }

    /// Captures the answer-budget window start the first time the thinking block
    /// closes, using the generated-token count *after* the token was pushed (so
    /// `</think>` itself is the last reasoning token, not the first answer token).
    /// A no-op once already captured or while the block is still open.
    ///
    /// Private (PR #787): only reachable through
    /// [`DecodePolicy::transition`], which owns the full per-step ordering.
    fn capture_reasoning_end(&mut self, generated_len_after_push: usize) {
        if self.thinking_closed && self.reasoning_end_len.is_none() {
            self.reasoning_end_len = Some(generated_len_after_push);
        }
    }

    /// True once `max_new_tokens` answer tokens have followed the `</think>` close
    /// point — the decode loop should break on this, in addition to its normal cap.
    ///
    /// Private (PR #787): only reachable through
    /// [`DecodePolicy::transition`], which owns the full per-step ordering.
    fn answer_budget_exhausted(&self, generated_len: usize) -> bool {
        self.reasoning_end_len
            .is_some_and(|end| generated_len.saturating_sub(end) >= self.max_new_tokens)
    }

    /// Appends one decode step's logprob data to `token_logprobs` when
    /// `self.logprobs` requests it; a no-op otherwise (so callers can invoke
    /// it unconditionally on every step -- the softmax pass over the full
    /// vocabulary is paid only when logprobs were actually requested).
    ///
    /// This is the ONLY place in the crate that pushes onto a
    /// `token_logprobs: &mut Vec<TokenLogprob>` accumulator (PR #787):
    /// `crate::sampling` exposes only
    /// the pure computation (`compute_step_logprobs`), not a freestanding
    /// "record" function a sibling decode call site could invoke directly
    /// to recreate the exact duplicate-choreography bug this method's
    /// privacy already closes for the other four constituent methods.
    ///
    /// Private (PR #787): only reachable through
    /// [`DecodePolicy::transition`] / [`DecodePolicy::init`], which own the
    /// full per-step ordering.
    fn record_logprob(
        &self,
        token_logprobs: &mut Vec<TokenLogprob>,
        logits: &[f32],
        token_id: u32,
        temperature: f32,
    ) {
        let Some(top_n) = self.logprobs else {
            return;
        };
        let (logprob, top) = compute_step_logprobs(logits, token_id, temperature, top_n);
        token_logprobs.push(TokenLogprob {
            token_id,
            logprob,
            top,
        });
    }

    /// The stop-check adapter dispatch (PR #787): drives whichever [`StopMode`] this policy was
    /// constructed with, given the caller's freshly decoded delta text for
    /// the current token. The caller supplies no decision logic at all — only
    /// the decoded text and a sink for whatever text is confirmed safe to
    /// release (`emit_confirmed`, called with the post-holdback-safe
    /// substring for `Streaming`, the raw delta for `Disabled`, never for
    /// `FullScan`, which has no external consumer). Shared by
    /// [`DecodePolicy::check_initial_stop`] (the prefill-derived first token,
    /// called once before the decode loop) and `transition` (every token
    /// after) — the same `self.stop_mode` instance is mutated across both
    /// calls, so `Streaming`'s incremental byte-holdback state carries over
    /// correctly from the first token onward, exactly as it did when each
    /// call site constructed and drove its own matcher by hand.
    ///
    /// Private (PR #787): only
    /// reachable through the two methods above.
    fn stop_check(
        &mut self,
        token_logprobs: &mut Vec<TokenLogprob>,
        text: &mut String,
        token_logprob_end_offsets: &mut Vec<usize>,
        delta: &str,
        mut emit_confirmed: impl FnMut(&str) -> bool,
    ) -> StopCheckOutcome {
        match &mut self.stop_mode {
            StopMode::Disabled => {
                if delta.is_empty() {
                    return StopCheckOutcome::Continue;
                }
                text.push_str(delta);
                if emit_confirmed(delta) {
                    StopCheckOutcome::Continue
                } else {
                    StopCheckOutcome::Interrupted
                }
            }
            StopMode::Streaming(matcher) => {
                let mut interrupted = false;
                let stop_matched = matcher.push(delta, &mut |s| {
                    if !s.is_empty() {
                        text.push_str(s);
                        if !interrupted && !emit_confirmed(s) {
                            interrupted = true;
                        }
                    }
                });
                if interrupted {
                    StopCheckOutcome::Interrupted
                } else if stop_matched {
                    StopCheckOutcome::Stopped
                } else {
                    StopCheckOutcome::Continue
                }
            }
            StopMode::FullScan {
                stop_strings,
                max_stop,
            } => {
                let prev_len = text.len();
                if !delta.is_empty() {
                    text.push_str(delta);
                }
                // Keep the offset tracker in lockstep with token_logprobs'
                // conditional growth (record_logprob is a no-op unless
                // gen_cfg.logprobs is set).
                if token_logprobs.len() > token_logprob_end_offsets.len() {
                    token_logprob_end_offsets.push(text.len());
                }
                let search_start = stop_scan_search_start(text, prev_len, *max_stop);
                if let Some(hit) = earliest_stop_match_from(text, stop_strings, search_start) {
                    text.truncate(hit);
                    truncate_token_logprobs_to_retained_text(
                        token_logprobs,
                        token_logprob_end_offsets,
                        hit,
                    );
                    StopCheckOutcome::Stopped
                } else {
                    StopCheckOutcome::Continue
                }
            }
        }
    }

    /// Checks the prefill-derived first token's already-decoded delta text
    /// against this policy's stop-mode (PR #787), before the decode loop starts — the first token is
    /// pushed and its logprob recorded by [`DecodePolicy::init`] outside
    /// `transition`'s per-step scope (it has no preceding grammar-advance /
    /// EOS check of its own to run through `transition` for), so its
    /// stop-string check needs its own entry point. Uses the SAME
    /// `self.stop_mode` instance `transition` will keep driving for every
    /// subsequent token, so `Streaming`'s byte-holdback state is continuous
    /// across the boundary — critical for a match that spans the first and
    /// second tokens, which a freshly-constructed second matcher would miss.
    pub(crate) fn check_initial_stop(
        &mut self,
        token_logprobs: &mut Vec<TokenLogprob>,
        text: &mut String,
        token_logprob_end_offsets: &mut Vec<usize>,
        delta: &str,
        emit_confirmed: impl FnMut(&str) -> bool,
    ) -> StopCheckOutcome {
        self.stop_check(
            token_logprobs,
            text,
            token_logprob_end_offsets,
            delta,
            emit_confirmed,
        )
    }

    /// The one per-step transition (ADR-080 C3, PR
    /// #787): atomically applies, in the fixed order every decode loop
    /// requires, the reasoning-budget override, the backend's grammar-advance
    /// callback, the emitted-token bookkeeping, the backend's EOS/stop-token
    /// callback, the backend's push callback (into its own `generated_ids` /
    /// `all_ids` or Metal-equivalent vectors), per-token logprobs recording,
    /// reasoning-end capture, the owned stop-check adapter, and the
    /// answer-budget check.
    ///
    /// `grammar_advance` and `is_eos` are backend callbacks because grammar
    /// masking/advance and EOS/stop-token identification remain genuinely
    /// backend-specific per ADR-080 C3 scope — each backend owns its own
    /// `GrammarState` and `cfg.eos_token_id` / `stop_token_ids` wiring, and
    /// `grammar_advance` must run on the *actually-emitted* (post-override)
    /// token before `is_eos` sees it, exactly mirroring the inline ordering
    /// every site used before this method existed. `push` is a callback
    /// because the token vectors are owned by the caller and are also read on
    /// the *next* loop iteration (`all_ids.last()` feeds the next
    /// `forward_step`) — the caller cannot hand that ownership to the policy.
    ///
    /// `decode_delta` / `text` / `token_logprob_end_offsets` / `emit_confirmed`
    /// (PR #787) replace an earlier
    /// arbitrary outcome-producing `stop_check` closure: the caller supplies
    /// only raw I/O (decode a token to text; a buffer to accumulate into; an
    /// offset tracker only `StopMode::FullScan` consults; a sink for
    /// confirmed-safe text), and [`DecodePolicy::stop_check`] — driven from
    /// `self.stop_mode`, fixed at construction from the real
    /// `gen_cfg.stop_strings` — decides the outcome. A call site can no
    /// longer claim `Continue` for a configuration that actually has stop
    /// strings, because it no longer produces the outcome at all.
    ///
    /// Returns [`StepOutcome::GrammarStop`] / [`StepOutcome::Eos`] without
    /// ever calling `push` when the token is rejected before emission
    /// (matching the existing contract that a stop token is never present in
    /// `token_ids`); [`StepOutcome::Stopped`] / [`StepOutcome::Interrupted`]
    /// when the stop-check adapter reports either outcome (the answer-budget
    /// check is skipped in both cases, matching every site's original control
    /// flow, which broke out of the loop before ever reaching it); or
    /// [`StepOutcome::Emitted`] once the token has been pushed and every
    /// remaining control, including a `Continue` stop-check, applied.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transition(
        &mut self,
        token_logprobs: &mut Vec<TokenLogprob>,
        sampled_id: u32,
        logits: &[f32],
        temperature: f32,
        generated_len_before: usize,
        mut grammar_advance: impl FnMut(u32) -> bool,
        mut is_eos: impl FnMut(u32) -> bool,
        mut push: impl FnMut(u32),
        mut decode_delta: impl FnMut(u32) -> String,
        text: &mut String,
        token_logprob_end_offsets: &mut Vec<usize>,
        mut emit_confirmed: impl FnMut(&str, u32) -> bool,
    ) -> StepOutcome {
        let next_id = self.apply_override(generated_len_before, sampled_id);

        if !grammar_advance(next_id) {
            return StepOutcome::GrammarStop;
        }

        self.note_emitted(next_id);

        if is_eos(next_id) {
            return StepOutcome::Eos;
        }

        push(next_id);
        let generated_len_after = generated_len_before + 1;

        self.record_logprob(token_logprobs, logits, next_id, temperature);
        self.capture_reasoning_end(generated_len_after);

        let delta = decode_delta(next_id);
        let stop_outcome = self.stop_check(
            token_logprobs,
            text,
            token_logprob_end_offsets,
            &delta,
            |s| emit_confirmed(s, next_id),
        );
        match stop_outcome {
            StopCheckOutcome::Stopped => return StepOutcome::Stopped,
            StopCheckOutcome::Interrupted => return StepOutcome::Interrupted,
            StopCheckOutcome::Continue => {}
        }

        StepOutcome::Emitted {
            token_id: next_id,
            answer_budget_exhausted: self.answer_budget_exhausted(generated_len_after),
        }
    }

    /// Natural-end flush (decode loop ended by cap/EOS/grammar-stop, not by
    /// `stop_check` reporting `Stopped`/`Interrupted`). `tail` is the
    /// detokenizer's own end-of-generation flush (`detok.finish()`).
    ///
    /// A no-op for `Disabled` beyond appending+emitting `tail` directly (there
    /// is nothing held back to reconcile) and for `FullScan` (the
    /// non-streaming caller owns its own tail-flush against its `full` buffer
    /// directly, e.g. `decode_loop_with_stops`, since it has no external
    /// consumer to hold text back from in the first place). Only `Streaming`
    /// mode's owned [`StopStringMatcher`] can be holding back up to
    /// `max_stop - 1` unconfirmed bytes that must be reconciled once the
    /// token source is exhausted — mirrors `StopStringMatcher::finish`
    /// exactly, since that is the only mode this call does real work for.
    ///
    /// Returns `true` when the tail flush itself completed a stop match
    /// (`Streaming` only; always `false` for `Disabled`/`FullScan`).
    pub(crate) fn finish_stop(
        &mut self,
        text: &mut String,
        tail: &str,
        mut emit_confirmed: impl FnMut(&str) -> bool,
    ) -> bool {
        match &mut self.stop_mode {
            StopMode::Disabled => {
                if !tail.is_empty() {
                    text.push_str(tail);
                    emit_confirmed(tail);
                }
                false
            }
            StopMode::Streaming(matcher) => {
                matcher.finish(tail, &mut |s| {
                    if !s.is_empty() {
                        text.push_str(s);
                        emit_confirmed(s);
                    }
                });
                matcher.stopped()
            }
            StopMode::FullScan { .. } => false,
        }
    }
}

/// Drops trailing `token_logprobs` entries whose decoded text extends past
/// `retained_len` (the text length after a stop-string match truncates the
/// output).
///
/// A stop match can complete mid-token or even mid-multi-token (an
/// incrementally-detokenized delta may itself span several sampled tokens),
/// so more than one trailing token can end up with text that no longer
/// appears in the truncated output. The OpenAI `logprobs.content` shape is
/// one entry per whole token; a token whose text was only partially retained
/// can't be represented as a partial entry, so it — and any token after it —
/// is dropped rather than left describing text the caller never receives in
/// `message.content` (#620).
///
/// `token_logprob_end_offsets[i]` must be the length of the accumulated
/// output text immediately after token `i`'s delta was appended, and the two
/// slices must be the same length (both grow in lockstep, gated on the same
/// `gen_cfg.logprobs.is_some()` condition — see call sites).
pub(crate) fn truncate_token_logprobs_to_retained_text(
    token_logprobs: &mut Vec<TokenLogprob>,
    token_logprob_end_offsets: &[usize],
    retained_len: usize,
) {
    debug_assert_eq!(token_logprobs.len(), token_logprob_end_offsets.len());
    let keep = token_logprob_end_offsets.partition_point(|&end| end <= retained_len);
    token_logprobs.truncate(keep);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three facts that make the prefill-derived first token safe to route
    /// through the ordinary per-step sequence.
    ///
    /// ADR-090 D2 requires the first-token exception to be *characterized*
    /// rather than inferred from later steps, because the canonical CPU entry
    /// runs a differently shaped step 0: it advances grammar on the sampled ID
    /// and never calls `apply_override` or `answer_budget_exhausted`, while
    /// [`DecodePolicy::transition`] calls both. Reading both paths at
    /// `model/qwen35/generation.rs:196-300` against `transition` shows every
    /// control is either present at step 0 in an equivalent form or unable to
    /// fire there at all. These tests pin the "unable to fire" half, which is
    /// the half that is a property of this module rather than of the entry.
    ///
    /// They exist because the shared driver may unify step 0 with later steps
    /// only while these hold, and each would be broken by an ordinary-looking
    /// change (a `>` relaxed to `>=`, a budget baseline moved) with no other
    /// test in this crate going red.
    fn cfg_with(max_new_tokens: usize, reasoning_budget: Option<usize>) -> GenerateConfig {
        GenerateConfig {
            max_new_tokens,
            reasoning_budget,
            enable_thinking: true,
            logprobs: None,
            ..Default::default()
        }
    }

    #[test]
    fn force_close_think_cannot_fire_on_the_first_generated_token() {
        let close = 248_069_u32;

        // budget == 1 is the discriminating case and the one an existing test in
        // qwen35_config does not cover: it is where `generated_so_far >= budget`
        // comes closest to holding at step 0, so it is where a wrong baseline
        // would first show. It must still not fire.
        assert_eq!(
            force_close_think(Some(1), true, false, 0, Some(close)),
            None,
            "budget 1 must not force on the first generated token"
        );
        assert_eq!(
            force_close_think(Some(2), true, false, 0, Some(close)),
            None,
            "budget 2 must not force on the first generated token"
        );

        // Must-FIRE control. Without it this test would keep passing if
        // `force_close_think` were changed to return None unconditionally.
        assert_eq!(
            force_close_think(Some(1), true, false, 1, Some(close)),
            Some(close),
            "budget 1 must force once one token has been generated"
        );
    }

    #[test]
    fn answer_budget_cannot_be_exhausted_by_the_first_generated_token() {
        let close = 248_069_u32;
        let cfg = cfg_with(1, Some(4));
        let mut logprobs = Vec::new();
        let logits = vec![0.0_f32; 8];

        // The only way step 0 sets `reasoning_end_len` is by emitting `</think>`
        // as the very first token, which gives end == 1 with exactly one token
        // generated. That is the earliest the answer budget could possibly be
        // exhausted, so it is the case to pin.
        let policy = DecodePolicy::init(
            &cfg,
            Some(close),
            &mut logprobs,
            close,
            &logits,
            cfg.temperature,
            1,
            false,
        );
        assert_eq!(
            policy.reasoning_end_len,
            Some(1),
            "a first token equal to the close ID must capture the reasoning end"
        );
        assert!(
            !policy.answer_budget_exhausted(1),
            "the answer budget must not be exhausted by the first generated token \
             while max_new_tokens >= 1"
        );

        // Must-FIRE control at the boundary: with max_new_tokens == 1 and the
        // reasoning block closed at length 1, length 2 is the first exhausted
        // length. Without this arm the assertion above would survive
        // `answer_budget_exhausted` being stubbed to false.
        assert!(
            policy.answer_budget_exhausted(2),
            "one answer token past the reasoning end must exhaust a budget of 1"
        );
    }

    #[test]
    fn a_zero_length_request_never_reaches_the_first_token_path() {
        // The proof above is conditional on `max_new_tokens >= 1`. That holds
        // because the canonical entry returns before sampling when the request
        // asks for nothing (`model/qwen35/generation.rs`, the
        // `max_new_tokens == 0` guard). Pinned here as the premise it is: if the
        // guard ever moves, `answer_budget_exhausted(1)` with max_new_tokens 0
        // would be `0 >= 0`, i.e. true on the first token.
        let cfg = cfg_with(0, Some(4));
        let mut logprobs = Vec::new();
        let logits = vec![0.0_f32; 8];
        let policy = DecodePolicy::init(
            &cfg,
            Some(248_069),
            &mut logprobs,
            248_069,
            &logits,
            cfg.temperature,
            1,
            false,
        );
        assert!(
            policy.answer_budget_exhausted(1),
            "documents WHY the entry guard is load-bearing: with max_new_tokens 0 \
             the first token would exhaust the answer budget immediately"
        );
    }
}
