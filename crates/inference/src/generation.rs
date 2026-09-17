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
use crate::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;
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
