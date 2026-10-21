//! Qwen3.5 generation, streaming generation, prefill/decode loops, stop-streamer utilities, and stop-token helpers.
use super::cache::{ForwardScratch, KvCache};
use super::model::Qwen35Model;
// Row R03 deleted this module's last production callers of `sample_token`/`DecodePolicy`/
// `StepOutcome` (the `generate_inline`/`generate_streaming_inline`/`decode_loop`/
// `decode_loop_with_stops` family): production CPU generation now routes entirely through
// `QwenCpuSession`/`decoder::driver::run_into`. This module's own tests still exercise
// `DecodePolicy`/`StepOutcome`/`sample_token` directly (see `mod tests` below), so the
// import is `#[cfg(test)]`-gated rather than deleted.
#[cfg(test)]
use super::sampling::sample_token;
use super::stop_strings::earliest_stop_match;
use crate::attention::gdn::GatedDeltaNetState;
use crate::decoder::Cancellation;
use crate::decoder::driver;
use crate::decoder::qwen_cpu::QwenCpuSession;
use crate::error::InferenceError;
#[cfg(test)]
use crate::generation::{DecodePolicy, StepOutcome};
use crate::generation::{GenerateConfig, GenerateOutput, truncate_token_logprobs_to_retained_text};
use crate::model::qwen35_config::{Qwen35Config, decode_cap};
use crate::stop_reason::StopReason;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;
use lattice_tokenizer::__internal::{
    IncrementalDetokenizer, decode_tokens, incremental_detokenizer_finish,
    incremental_detokenizer_new, incremental_detokenizer_push_into,
};

/// Adapts the streaming API's `should_cancel: impl FnMut() -> bool` (which may
/// capture a `Receiver`-style handle and mutate on each poll) to
/// `decoder::Cancellation`, whose blanket impl covers only non-mutating `Fn`
/// closures (see that trait's own doc comment, which names this exact
/// conversion as a future row's job). `RefCell` gives `is_cancelled(&self)` a
/// runtime-checked path to the one mutable call it needs; `decoder::driver::run_into`
/// calls it strictly sequentially from a single thread, so the borrow never
/// conflicts.
struct FnMutCancellation<'a, F: FnMut() -> bool>(&'a std::cell::RefCell<F>);

impl<F: FnMut() -> bool> Cancellation for FnMutCancellation<'_, F> {
    fn is_cancelled(&self) -> bool {
        let mut should_cancel = self.0.borrow_mut();
        (*should_cancel)()
    }
}

/// Test-only toggle forcing the pre-delegation serial prefill path
/// (`prefill_tokens`) instead of `prefill_tokens_batched_for_generate`.
///
/// Exists solely so `generate` / `generate_streaming` tests can reproduce the
/// exact old-path token sequence in-process (no duplicated ~300-line copy of
/// `generate`'s body) and assert it against the new batched-prefill path.
/// Guarded by `#[cfg(test)]` end to end, so it does not exist in non-test
/// builds; production behaviour is unaffected. Tests that use it must hold
/// `SERIAL_PREFILL_TEST_LOCK` for the duration, since this is process-global
/// mutable state and `cargo test` runs tests in parallel by default.
#[cfg(test)]
pub(crate) static FORCE_SERIAL_PREFILL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Serializes tests that toggle `FORCE_SERIAL_PREFILL`, so they never race
/// against each other (racing against unrelated dense-model tests is
/// harmless: both prefill paths are required to produce identical output,
/// which is exactly the invariant this feature exists to preserve).
#[cfg(test)]
pub(crate) static SERIAL_PREFILL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn force_serial_prefill() -> bool {
    FORCE_SERIAL_PREFILL.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(not(test))]
pub(crate) fn force_serial_prefill() -> bool {
    false
}

impl Qwen35Model {
    /// **Unstable**: autoregressive text generation with temperature/top-k/top-p sampling.
    ///
    /// Delegates to `Self::generate_with_trace` and discards the driver trace (ADR-090
    /// row C, decomposition "Open question 3, ANSWERED"): this entry's signature and
    /// observable behaviour are unchanged, and the trace exists for the migration's own
    /// tests to see, not for callers of the public API.
    pub fn generate(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, InferenceError> {
        self.generate_with_trace(prompt, gen_cfg)
            .map(|(output, _trace)| output)
    }

    /// ADR-090 row C dispatch point (row R03: unconditional -- every canonical Qwen3.5 CPU
    /// generate/stream request now routes through [`Self::generate_via_driver`]
    /// (`decoder::driver::run_into` over a [`crate::decoder::qwen_cpu::QwenCpuSession`]), grammar
    /// and logprobs included; the pre-driver `generate_inline`/`decode_loop`/
    /// `decode_loop_with_stops` fallback this comment used to describe is deleted).
    ///
    /// Crate-private and not `#[cfg(test)]`-gated (decomposition "Open question 3,
    /// ANSWERED"): a marker that only exists under `cfg(test)` would make the shipped path
    /// and the tested path differ in the one respect the test is about.
    pub(crate) fn generate_with_trace(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) -> Result<(GenerateOutput, driver::DriverTrace), InferenceError> {
        self.generate_via_driver(prompt, gen_cfg)
    }

    /// ADR-090 row C, extended by row R03 to every canonical Qwen3.5 CPU generate request
    /// (grammar and logprobs included -- no exceptions remain): constructs a
    /// [`QwenCpuSession`] and runs [`driver::run_into`] over it. Every preflight step through
    /// `think_close_id` resolution is unchanged from the pre-driver implementation -- only
    /// the decode mechanics after prefill are re-routed. `gen_cfg.seed` and
    /// `gen_cfg.temperature` go straight into [`QwenCpuSession::new`], which owns the RNG
    /// draw and the per-request `gdn_states`/`kv_cache`/`scratch` allocation from here on
    /// (see that constructor's own doc comment for why it captures exactly those two
    /// fields). `gen_cfg.grammar` is read by [`driver::run_into`] itself now, not by this
    /// constructor -- the driver owns grammar transitions (ADR-090 D1).
    fn generate_via_driver(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) -> Result<(GenerateOutput, driver::DriverTrace), InferenceError> {
        let cfg = &self.config;

        let input = self.tokenizer.tokenize(prompt);
        let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
        let prompt_len = prompt_ids.len();

        check_prompt_not_empty(prompt_len)?;

        if gen_cfg.max_new_tokens == 0 {
            return Ok((
                GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: Some(StopReason::Length),
                    token_logprobs: vec![],
                },
                driver::DriverTrace::default(),
            ));
        }

        let max_context = self.max_context();
        check_context_budget(
            prompt_len,
            gen_cfg.effective_reasoning_budget(),
            gen_cfg.max_new_tokens,
            max_context,
        )?;

        let think_close_id = resolve_reasoning_close_token(
            &self.tokenizer,
            gen_cfg.reasoning_budget,
            gen_cfg.enable_thinking,
            cfg.vocab_size,
        )?;
        let reasoning_newline_boundary = think_close_id
            .map(|_| lattice_tokenizer::__internal::newline_boundary_table(&self.tokenizer));

        let mut session =
            QwenCpuSession::new(self, prompt_ids.clone(), gen_cfg.temperature, gen_cfg.seed);
        session.reserve_generation(prompt_len.saturating_add(decode_cap(
            gen_cfg.effective_reasoning_budget(),
            gen_cfg.max_new_tokens,
        )));

        // Non-streaming callers never cancel and never need raw-event/tail-flush
        // hooks; only the streaming dispatch target (`generate_streaming_via_driver`)
        // supplies real ones.
        let never_cancel = || false;

        if gen_cfg.stop_strings.is_empty() {
            let mut throwaway_text = String::new();
            let mut throwaway_offsets: Vec<usize> = Vec::new();
            let result = driver::run_into(
                &mut session,
                gen_cfg,
                think_close_id,
                reasoning_newline_boundary.clone(),
                &prompt_ids,
                cfg.eos_token_id,
                false,
                &never_cancel,
                |_generated_len| {},
                |_next_id, _out| {},
                &mut throwaway_text,
                &mut throwaway_offsets,
                |_delta, _next_id| true,
                || {},
                String::new,
            )?;

            let text = decode_tokens(&self.tokenizer, &result.generated_ids);

            Ok((
                GenerateOutput {
                    text,
                    token_ids: result.generated_ids.clone(),
                    prompt_tokens: prompt_len,
                    generated_tokens: result.generated_ids.len(),
                    stopped: result.stopped,
                    stop_reason: Some(result.stop_reason),
                    token_logprobs: result.token_logprobs,
                },
                result.trace,
            ))
        } else {
            let mut detok = incremental_detokenizer_new();
            let mut full = String::new();
            let mut token_logprob_end_offsets: Vec<usize> = Vec::new();

            let result = driver::run_into(
                &mut session,
                gen_cfg,
                think_close_id,
                reasoning_newline_boundary,
                &prompt_ids,
                cfg.eos_token_id,
                false,
                &never_cancel,
                |_generated_len| {},
                |next_id, out| {
                    incremental_detokenizer_push_into(&mut detok, &self.tokenizer, next_id, out)
                },
                &mut full,
                &mut token_logprob_end_offsets,
                |_delta, _next_id| true,
                || {},
                String::new,
            )?;

            let mut token_logprobs = result.token_logprobs;
            let mut stopped = result.stopped;
            let mut stop_reason = result.stop_reason;

            // Mirrors `decode_loop_with_stops`'s own post-loop tail-flush exactly
            // (same detokenizer-finish/rescan/truncate sequence, same "already
            // stopped wins" precedence).
            if let Some(tail) = finish_detokenizer(&mut detok, result.confirmed_stop_string_match)
                && !tail.is_empty()
            {
                full.push_str(&tail);
                if let Some(hit) = earliest_stop_match(&full, &gen_cfg.stop_strings) {
                    full.truncate(hit);
                    truncate_token_logprobs_to_retained_text(
                        &mut token_logprobs,
                        &token_logprob_end_offsets,
                        hit,
                    );
                    if !stopped {
                        stopped = true;
                        stop_reason = StopReason::Eos;
                    }
                }
            }

            Ok((
                GenerateOutput {
                    text: full,
                    token_ids: result.generated_ids.clone(),
                    prompt_tokens: prompt_len,
                    generated_tokens: result.generated_ids.len(),
                    stopped,
                    stop_reason: Some(stop_reason),
                    token_logprobs,
                },
                result.trace,
            ))
        }
    }

    /// Streaming variant of [`Self::generate`] — identical token sequence, but invokes
    /// `on_token` with incremental text deltas after each generated token.
    ///
    /// # Parity safety
    ///
    /// The body below is a deliberate copy of `generate` rather than a refactor of
    /// the shared path. This ensures that no change here can silently alter the
    /// non-streaming `generate` path, which is pinned by the e2e-parity CI gate
    /// (greedy token match vs HF transformers). `on_token` is the only addition.
    ///
    /// `should_cancel = || false` convenience form of
    /// [`Self::generate_streaming_with_cancel`]; both share this one
    /// implementation.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<GenerateOutput, InferenceError> {
        self.generate_streaming_with_cancel(
            prompt,
            gen_cfg,
            |delta| {
                on_token(delta);
                true
            },
            || false,
        )
    }

    /// Cancellation-aware sibling of [`Self::generate_streaming`] (ADR-080 C2,
    /// ports the Metal `MetalQwen35State::generate_streaming_with_cancel`
    /// contract to the CPU backend, closing #744: previously `lattice.rs`'s CPU
    /// streaming path had no way to observe a client disconnect at all and ran
    /// to the token cap after the client left).
    ///
    /// `should_cancel` is polled independently of `on_token`: before the
    /// prefill pass starts, immediately after it returns, and at the top of
    /// every decode iteration — all before any further work runs for that
    /// step. `on_token` itself also stops generation the moment it returns
    /// `false` (the caller could not forward the delta, e.g. the SSE receiver
    /// was dropped). Either signal short-circuits the trailing
    /// incomplete-UTF-8 flush too, since the caller is no longer consuming
    /// the stream by then. Both stopping paths report
    /// `stopped: false, stop_reason: Some(StopReason::Interrupt)` — a
    /// cancellation is not an OpenAI "stop condition", matching the Metal
    /// contract exactly.
    ///
    /// Thin wrapper over [`Self::generate_streaming_with_observer`] with a
    /// no-op raw-event observer, so every existing caller (production
    /// serving paths, tests) keeps this exact 4-argument signature and zero
    /// behavior change.
    pub fn generate_streaming_with_cancel<F, C>(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<GenerateOutput, InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
    {
        self.generate_streaming_with_observer(prompt, gen_cfg, on_token, should_cancel, |_| {})
    }

    /// [`Self::generate_streaming_with_cancel`] plus a raw generation-lifecycle
    /// observer: fires [`RawGenEvent::PrefillEnd`] once, after the prefill forward pass has
    /// produced logits and *before* the first token is sampled, and
    /// [`RawGenEvent::RawToken`] once per token that actually becomes part of
    /// `GenerateOutput` (both the prefill-derived first token and every
    /// decode-loop token), in generation order, with a 1-based
    /// monotonically-increasing `index` equal to `generated_ids.len()` at
    /// the moment of firing.
    ///
    /// This is deliberately independent of `on_token`'s text deltas: the
    /// incremental UTF-8 detokenizer buffers incomplete multi-byte
    /// codepoints, so one text delta is not guaranteed to equal one sampled
    /// token, and `prefill_end` measured off the first delta would fire
    /// *after* sampling rather than before it. `on_raw_event` fires exactly
    /// at the raw-token boundary and is unaffected by detokenizer buffering,
    /// so a caller measuring prefill/decode timing (`--emit-phase-events` in
    /// `qwen35_generate.rs`) gets an event count that always equals
    /// `GenerateOutput.generated_tokens` by construction: it is fired from
    /// the exact same `push` control-flow point that increments
    /// `generated_ids`, never from a step that grammar-stops or EOS-stops
    /// before the token is pushed.
    ///
    /// Delegates to `Self::generate_streaming_with_trace` and discards the driver
    /// trace, mirroring [`Self::generate`]'s relationship to
    /// `Self::generate_with_trace` (ADR-092 arm 1): this entry's signature and
    /// observable behaviour are unchanged.
    pub fn generate_streaming_with_observer<F, C, O>(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
        on_raw_event: O,
    ) -> Result<GenerateOutput, InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
        O: FnMut(RawGenEvent),
    {
        self.generate_streaming_with_trace(prompt, gen_cfg, on_token, should_cancel, on_raw_event)
            .map(|(output, _trace)| output)
    }

    /// ADR-092 arm 1 dispatch point, streaming sibling of [`Self::generate_with_trace`]
    /// (row R03: unconditional, same as that function -- routes through
    /// [`Self::generate_streaming_via_driver`] for every canonical Qwen3.5 CPU streaming
    /// request, grammar and logprobs included).
    ///
    /// Crate-private and not `#[cfg(test)]`-gated, matching `generate_with_trace`: a
    /// marker that only exists under `cfg(test)` would make the shipped path and the
    /// tested path differ in the one respect the test is about.
    pub(crate) fn generate_streaming_with_trace<F, C, O>(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
        on_raw_event: O,
    ) -> Result<(GenerateOutput, driver::DriverTrace), InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
        O: FnMut(RawGenEvent),
    {
        self.generate_streaming_via_driver(prompt, gen_cfg, on_token, should_cancel, on_raw_event)
    }

    /// Driver-routed dispatch target -- streaming sibling of [`Self::generate_via_driver`],
    /// which it mirrors closely: same tokenize/preflight/context-budget/reasoning-close
    /// sequence, same [`QwenCpuSession`], same `driver::run_into` call. Two differences, both
    /// load-bearing for the streaming contract:
    ///
    /// 1. **Cancellation.** `should_cancel` here is `FnMut` (it may capture a
    ///    `Receiver`-style handle), while `decoder::Cancellation`'s blanket impl covers
    ///    only non-mutating `Fn` closures. [`FnMutCancellation`] bridges the two via a
    ///    `RefCell`, per that trait's own doc comment.
    /// 2. **Raw-token eventing.** `decoder::driver::run_into` calls its dedicated `on_push`
    ///    parameter exactly once per token that becomes part of `generated_ids`,
    ///    immediately at the push, in generation order, with `generated_ids.len()` at
    ///    that moment -- the same guarantee [`RawGenEvent::RawToken`]'s own doc comment
    ///    relies on. This entry fires `RawToken` from that hook directly, using the
    ///    length it is handed as the index, rather than from `decode_delta`: `decode_delta`
    ///    runs later in `driver::run_into`'s fixed per-step order (after `record_metadata`'s
    ///    session-scored logprob call whenever `gen_cfg.logprobs` is set), so firing from
    ///    it would let the event's timing drift with that call's cost instead of marking
    ///    the true push boundary.
    ///
    /// Always builds a real incremental detokenizer and real output buffers (unlike
    /// `generate_via_driver`'s fast/stop-strings split, which throws the delta away when
    /// nothing needs to observe it): a streaming caller needs `on_token` fired
    /// incrementally regardless of whether `stop_strings` is set, and `driver::run_into`'s own
    /// `policy` (built from `gen_cfg.stop_strings`) already decides internally whether
    /// that incremental work does anything.
    fn generate_streaming_via_driver<F, C, O>(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        mut on_token: F,
        should_cancel: C,
        on_raw_event: O,
    ) -> Result<(GenerateOutput, driver::DriverTrace), InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
        O: FnMut(RawGenEvent),
    {
        let cfg = &self.config;

        let input = self.tokenizer.tokenize(prompt);
        let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
        let prompt_len = prompt_ids.len();

        check_prompt_not_empty(prompt_len)?;

        if gen_cfg.max_new_tokens == 0 {
            return Ok((
                GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: Some(StopReason::Length),
                    token_logprobs: vec![],
                },
                driver::DriverTrace::default(),
            ));
        }

        let max_context = self.max_context();
        check_context_budget(
            prompt_len,
            gen_cfg.effective_reasoning_budget(),
            gen_cfg.max_new_tokens,
            max_context,
        )?;

        let think_close_id = resolve_reasoning_close_token(
            &self.tokenizer,
            gen_cfg.reasoning_budget,
            gen_cfg.enable_thinking,
            cfg.vocab_size,
        )?;
        let reasoning_newline_boundary = think_close_id
            .map(|_| lattice_tokenizer::__internal::newline_boundary_table(&self.tokenizer));

        let mut session =
            QwenCpuSession::new(self, prompt_ids.clone(), gen_cfg.temperature, gen_cfg.seed);
        session.reserve_generation(prompt_len.saturating_add(decode_cap(
            gen_cfg.effective_reasoning_budget(),
            gen_cfg.max_new_tokens,
        )));

        let should_cancel_cell = std::cell::RefCell::new(should_cancel);
        let cancel = FnMutCancellation(&should_cancel_cell);

        // Shared via `RefCell` rather than plain captures: `on_push` (called once per
        // pushed token, before `decode_delta` runs for that same token -- see
        // `driver::run_into`'s doc comment), `decode_delta` (also called once per pushed
        // token), and `on_prefill_end` (called once, before any token) are three
        // separate closures passed into the same `driver::run_into` call, and both `on_push`
        // and `on_prefill_end` need `on_raw_event`; `decode_delta` and `finish_tail`
        // both need `detok`. No two of these are ever live concurrently (`driver::run_into`
        // calls them strictly sequentially, single-threaded), so the runtime borrow
        // check never conflicts.
        let on_raw_event_cell = std::cell::RefCell::new(on_raw_event);
        let detok_cell = std::cell::RefCell::new(incremental_detokenizer_new());
        let mut text = String::new();
        let mut token_logprob_end_offsets: Vec<usize> = Vec::new();

        let result = driver::run_into(
            &mut session,
            gen_cfg,
            think_close_id,
            reasoning_newline_boundary,
            &prompt_ids,
            cfg.eos_token_id,
            true,
            &cancel,
            // Fires `RawToken` at `driver::run_into`'s own `on_push` hook, i.e. immediately
            // at the push that grows `generated_ids` to `generated_len` -- never from
            // `decode_delta` below, which `driver::run_into` calls later in the fixed
            // per-step order (after `record_metadata`'s session-scored logprob call
            // whenever `gen_cfg.logprobs` is set). `generated_len` IS the index this
            // event must report (`RawGenEvent::RawToken`'s own doc comment), so no
            // local counter is needed to reproduce it.
            |generated_len| {
                (*on_raw_event_cell.borrow_mut())(RawGenEvent::RawToken {
                    index: generated_len,
                });
            },
            |next_id, out| {
                incremental_detokenizer_push_into(
                    &mut detok_cell.borrow_mut(),
                    &self.tokenizer,
                    next_id,
                    out,
                )
            },
            &mut text,
            &mut token_logprob_end_offsets,
            |delta, _next_id| on_token(delta),
            || {
                (*on_raw_event_cell.borrow_mut())(RawGenEvent::PrefillEnd);
                // Test-only seam (mirrors `generate_streaming_inline`'s own call to the
                // same function, at the equivalent point: immediately before the first
                // `sample_token` -- here, immediately before the first `session.select`,
                // which is where that sampling happens for the driver route). No-op
                // outside `cfg(test)`.
                #[cfg(test)]
                test_record_first_sample_entry();
            },
            || incremental_detokenizer_finish(&mut detok_cell.borrow_mut()),
        )?;

        Ok((
            GenerateOutput {
                text,
                token_ids: result.generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: result.generated_ids.len(),
                stopped: result.stopped,
                stop_reason: Some(result.stop_reason),
                token_logprobs: result.token_logprobs,
            },
            result.trace,
        ))
    }
}

/// Raw generation-lifecycle event fired by
/// [`Qwen35Model::generate_streaming_with_observer`], independent of the
/// caller's text-delta callback and unaffected by incremental UTF-8
/// detokenizer buffering: a text delta is not guaranteed to equal one raw
/// sampled token, and measuring `prefill_end` off the first delta fires it
/// after sampling instead of before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RawGenEvent {
    /// Fired exactly once, after the prefill forward pass has produced
    /// logits and before the first token is sampled -- the true
    /// prefill/decode boundary.
    PrefillEnd,
    /// Fired once per token that actually becomes part of
    /// `GenerateOutput.token_ids` (never for a token that grammar-stops or
    /// EOS-stops before being pushed), in generation order. `index` is
    /// 1-based and monotonically increasing, equal to `generated_ids.len()`
    /// at the moment of firing -- so a caller counting these events always
    /// gets a count equal to `GenerateOutput.generated_tokens`.
    RawToken { index: usize },
}

// Test-only seam: the existing
// mutation-sensitive test below only proves `PrefillEnd` fires before the
// `RawToken` *callback*, not before `sample_token` itself. A `PrefillEnd`
// moved to just after `generated_ids.push` (after sampling, before the
// `RawToken` callback) would keep that test green while silently folding
// sampling time into the reported prefill interval. `test_record_first_sample_entry`
// is called from the exact point `sample_token` is about to run for the
// prefill-derived first token; a test's `on_raw_event` closure calls
// `test_mark_prefill_end_seen` when it observes `PrefillEnd`, and the two
// are compared to answer "did PrefillEnd fire before the *first sample*",
// not just "before the first raw-token callback". Zero production effect:
// every item here is `#[cfg(test)]` and compiles to nothing outside tests.
#[cfg(test)]
thread_local! {
    static TEST_PREFILL_END_SEEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TEST_FIRST_SAMPLE_SAW_PREFILL_END: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Resets the seam's thread-local state. Call at the start of any test that
/// reads `test_take_first_sample_saw_prefill_end` afterward.
#[cfg(test)]
fn test_reset_sample_seam() {
    TEST_PREFILL_END_SEEN.with(|c| c.set(false));
    TEST_FIRST_SAMPLE_SAW_PREFILL_END.with(|c| c.set(None));
}

/// Called by a test's `on_raw_event` closure when it observes
/// `RawGenEvent::PrefillEnd`.
#[cfg(test)]
fn test_mark_prefill_end_seen() {
    TEST_PREFILL_END_SEEN.with(|c| c.set(true));
}

/// Called from production code (under `#[cfg(test)]`) immediately before the
/// first call to `sample_token`. Records, the first time only, whether
/// `PrefillEnd` had already been observed by that point.
#[cfg(test)]
fn test_record_first_sample_entry() {
    TEST_FIRST_SAMPLE_SAW_PREFILL_END.with(|seen_at_sample| {
        if seen_at_sample.get().is_none() {
            let seen = TEST_PREFILL_END_SEEN.with(std::cell::Cell::get);
            seen_at_sample.set(Some(seen));
        }
    });
}

/// Reads the value recorded by `test_record_first_sample_entry`.
#[cfg(test)]
fn test_take_first_sample_saw_prefill_end() -> Option<bool> {
    TEST_FIRST_SAMPLE_SAW_PREFILL_END.with(std::cell::Cell::get)
}

pub(crate) fn initial_rng_state(seed: Option<u64>) -> u64 {
    match seed {
        Some(s) => {
            if s == 0 {
                1
            } else {
                s
            }
        }
        None => {
            use std::time::SystemTime;
            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x12345678_9abcdef0);
            if t == 0 { 1 } else { t }
        }
    }
}

fn finish_detokenizer(
    detok: &mut IncrementalDetokenizer,
    confirmed_stop_string_match: bool,
) -> Option<String> {
    (!confirmed_stop_string_match).then(|| incremental_detokenizer_finish(detok))
}

pub(crate) fn prefill_tokens(
    model: &Qwen35Model,
    prompt_ids: &[u32],
    gdn_states: &mut [GatedDeltaNetState],
    kv_cache: &mut KvCache,
    scratch: &mut ForwardScratch,
) {
    let prompt_len = prompt_ids.len();
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        model.forward_step(token_id, pos, gdn_states, kv_cache, scratch);
        if pos < prompt_len - 1 {
            kv_cache.seq_len += 1;
        }
    }
}

/// Returns true when `token_id` is EOS or is in the `stop_token_ids` list.
///
/// `pub(crate)` — all call sites live within this crate. Keeping the function
/// crate-private avoids leaking a low-level sampling helper as part of the
/// public API.
pub(crate) fn should_stop_token(
    cfg: &Qwen35Config,
    gen_cfg: &GenerateConfig,
    token_id: u32,
) -> bool {
    token_id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&token_id)
}

/// Returns `Err(InvalidInput)` when `gen_cfg.grammar` is set, to prevent the
/// grammar field from being silently ignored on paths that have not yet wired
/// grammar masking (#397).
///
/// Grammar masking (`mask_logits` + `advance`) requires a per-step wiring loop
/// inside each generate path. The following paths have not yet been wired and
/// delegate to this guard to fail closed rather than silently producing
/// unconstrained output when a caller sets `gen_cfg.grammar`:
///
/// - `generate_q8` (`forward/cpu_q8.rs`)
/// - `generate_f16` (`forward/cpu_f16.rs`)
/// - `generate_q8_neon` (`forward/neon_forward.rs`)
/// - `multimodal_generate_preflight` (`forward/metal_qwen35.rs`)
///
/// The base CPU `generate()` / `generate_streaming()` paths in this module wire
/// grammar directly and therefore do not call this guard.
pub(crate) fn check_grammar_not_set(gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
    if gen_cfg.grammar.is_some() {
        return Err(InferenceError::InvalidInput(
            "grammar-constrained decoding is not yet supported on this path; \
             use the Qwen3.5 CPU generate() / generate_streaming(), which implement \
             grammar masking"
                .into(),
        ));
    }
    Ok(())
}

/// Sibling guard to [`check_grammar_not_set`]: fails closed instead of silently
/// dropping a `logprobs` request on a generation path that has not been wired
/// to capture per-step log-probabilities (#585). Same five paths, same
/// rationale — see `check_grammar_not_set` for the full list.
///
/// The base CPU `generate()` / `generate_streaming()` paths in this module
/// wire logprobs capture directly and therefore do not call this guard.
pub(crate) fn check_logprobs_not_set(gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
    if gen_cfg.logprobs.is_some() {
        return Err(InferenceError::InvalidInput(
            "per-token logprobs are not yet supported on this generation path; \
             use the Qwen3.5 CPU generate() / generate_streaming() or the Metal \
             generate_streaming(), which implement logprobs capture"
                .into(),
        ));
    }
    Ok(())
}

/// Sibling guard to [`check_grammar_not_set`] / [`check_logprobs_not_set`]
/// (ADR-080 C3, #783): fails closed instead of silently dropping a
/// `stop_strings` request on a generation path that has not wired
/// string-level stop matching into its decode loop.
///
/// Callers: `generate_f16` (`forward/cpu_f16.rs`), `generate_q8`
/// (`forward/cpu_q8.rs`), `generate_q8_neon` (`forward/neon_forward.rs`).
///
/// The base CPU `generate()` / `generate_streaming()` paths in this module,
/// and the Metal `generate()` / `generate_streaming()` / `generate_multimodal`
/// family, all wire `stop_strings` matching directly and therefore do not
/// call this guard.
pub(crate) fn check_stop_strings_not_set(gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
    if !gen_cfg.stop_strings.is_empty() {
        return Err(InferenceError::InvalidInput(
            "stop_strings is not yet supported on this generation path; \
             use the Qwen3.5 CPU generate() / generate_streaming() or the Metal \
             generate() / generate_streaming(), which implement stop-string matching"
                .into(),
        ));
    }
    Ok(())
}

/// Sibling guard to [`check_stop_strings_not_set`] (ADR-080 C3, #783): fails
/// closed instead of silently dropping a `reasoning_budget` request on a
/// generation path that has not wired budget-forcing (`decode_cap` /
/// `force_close_think`) into its decode loop.
///
/// Same CPU caller list as [`check_stop_strings_not_set`]. On the Metal side,
/// the plain `generate()` entry point and `multimodal_generate_preflight`
/// (`generate_multimodal`) both call this guard directly; the MTP and
/// self-speculative greedy fast paths never see a set `reasoning_budget` in
/// the first place — their route predicates (`mtp_route_active` /
/// `self_spec_route_active`) exclude it, falling through to plain
/// `generate()`, which rejects it via this same guard.
///
/// The base CPU `generate()` / `generate_streaming()` paths in this module,
/// and the Metal `generate_streaming()` family, wire reasoning-budget forcing
/// directly and therefore do not call this guard.
pub(crate) fn check_reasoning_budget_not_set(
    gen_cfg: &GenerateConfig,
) -> Result<(), InferenceError> {
    if gen_cfg.reasoning_budget.is_some() {
        return Err(InferenceError::InvalidInput(
            "reasoning_budget is not yet supported on this generation path; \
             use the Qwen3.5 CPU generate() / generate_streaming() or the Metal \
             generate_streaming(), which implement reasoning-budget forcing"
                .into(),
        ));
    }
    Ok(())
}

/// The reasoning-close marker every thinking-budget path resolves against:
/// the shared generation resolver below, and the cross-turn KV cache
/// fingerprint (`MetalQwen35State::generate_streaming`'s
/// `tokenizer_fingerprint` hash), which must key on the same marker the
/// close-token resolver validates so cache reuse and budget enforcement
/// never disagree about which tokenizer content they mean by "the reasoning
/// close marker".
pub(crate) const REASONING_CLOSE_MARKER: &str = "</think>";

/// Resolve the closing marker required to enforce an active reasoning budget.
///
/// `reasoning_budget` is inert when `enable_thinking` is false: with thinking
/// disabled the caller is contracted to have primed the prompt so no
/// reasoning block is produced (see [`GenerateConfig::enable_thinking`]), so
/// there is nothing for a forced `</think>` to close. This mirrors
/// [`force_close_think`], whose own `enable_thinking` guard already treats
/// that combination as a no-op -- resolving (and possibly rejecting on) a
/// close marker that forcing could never use would make the two functions
/// disagree about what the field means. So this returns `Ok(None)` without
/// touching the tokenizer whenever `enable_thinking` is false, exactly like
/// the `None`/`Some(0)` budget case below.
///
/// When thinking is enabled, exact token-content lookup covers both
/// base-vocabulary entries and added tokens regardless of their `special`
/// flag. An active budget fails closed if the tokenizer has no representable
/// `</think>` marker, if more than one id renders as that marker (the decode
/// loop recognises only one, so the rest would close the block in the output
/// while the budget kept counting), or if the resolved id
/// is outside `vocab_size` (a mismatched or malicious tokenizer can declare
/// `</think>` in its base vocabulary at an id the loaded model has no
/// embedding/logit row for) -- the id is later forced into decoding, and
/// `forward_step` only asserts on out-of-range ids rather than returning a
/// typed error, so this is the one place that can fail closed instead.
pub(crate) fn resolve_reasoning_close_token(
    tokenizer: &BpeTokenizer,
    reasoning_budget: Option<usize>,
    enable_thinking: bool,
    vocab_size: usize,
) -> Result<Option<u32>, InferenceError> {
    if !enable_thinking || !matches!(reasoning_budget, Some(budget) if budget > 0) {
        return Ok(None);
    }

    // Ambiguity is the same defect as absence and has to fail the same way.
    // `DecodePolicy` compares the emitted id against ONE close-marker id, so if
    // the tokenizer spells `</think>` at two different ids -- which the tiers
    // permit, since `vocab` and `rendered_added` are independent inputs and
    // added-token ids sit beyond the base range -- then resolving to the first
    // tier silently discards the other. A generation that emitted the discarded
    // alias would render `</think>` while the policy went on treating the
    // reasoning block as open, spending the budget and forcing a second marker.
    // That is a silently unenforced budget reached by a different route, which
    // is the exact failure this resolver exists to refuse.
    let ids =
        lattice_tokenizer::__internal::token_ids_for_content(tokenizer, REASONING_CLOSE_MARKER);
    let token_id = match ids.as_slice() {
        [] => {
            return Err(InferenceError::InvalidInput(
                "reasoning_budget cannot be enforced because the tokenizer has no </think> token"
                    .into(),
            ));
        }
        [only] => *only,
        ambiguous => {
            let mut sorted = ambiguous.to_vec();
            sorted.sort_unstable();
            return Err(InferenceError::InvalidInput(format!(
                "reasoning_budget cannot be enforced because the tokenizer maps </think> to \
                 more than one token id ({sorted:?}); the decode loop recognises a single \
                 close-marker id, so any other id rendering </think> would close the reasoning \
                 block in the output while the budget kept treating it as open"
            )));
        }
    };

    if (token_id as usize) >= vocab_size {
        return Err(InferenceError::InvalidInput(format!(
            "reasoning_budget cannot be enforced because the tokenizer's </think> \
             token id {token_id} is out of range for the model vocabulary \
             (vocab_size {vocab_size})"
        )));
    }

    Ok(Some(token_id))
}

/// Sibling guard to [`check_reasoning_budget_not_set`] (PR #787): fails
/// closed instead of silently ignoring an active MTP
/// request on a generation path that never reads `gen_cfg.enable_mtp`.
///
/// Resolves `enable_mtp` exactly like the Metal `generate()` entry point
/// (`gen_cfg.enable_mtp.unwrap_or_else(|| LATTICE_MTP env set)`), so a caller
/// or environment combination that would activate MTP on the direct path is
/// rejected here too, rather than silently falling back to plain per-token
/// decode with no indication MTP was skipped.
///
/// Sole caller: the Metal cross-turn prefix-cache path
/// (`generate_streaming_with_prefix_cache_and_cancel`), which has no MTP
/// draft/verify wiring at all -- gated identically to that Metal-only
/// consumer (same gate as the `DecodePolicy`/`StepOutcome` re-export in
/// `mod.rs`) so non-metal-gpu builds don't carry an unused function.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) fn check_mtp_not_requested(gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
    let mtp_enabled = gen_cfg
        .enable_mtp
        .unwrap_or_else(|| crate::env_switch_enabled("LATTICE_MTP"));
    if mtp_enabled {
        return Err(InferenceError::InvalidInput(
            "enable_mtp (or LATTICE_MTP) is not supported on the cross-turn \
             prefix-cache generation path, which has no MTP draft/verify \
             wiring; use the Metal generate() / generate_streaming() paths, \
             which implement MTP"
                .into(),
        ));
    }
    Ok(())
}

/// Preflight guard unifying the empty-prompt contract across every Qwen3.5
/// generation entry point (#856).
///
/// Before this fix, `docs/generation-entrypoint-matrix.md` row 2 recorded a
/// hard behavior split: the three CPU forward paths (`generate_f16`,
/// `generate_q8`, `generate_q8_neon`) rejected an empty prompt with this
/// same typed `Err`, inline and duplicated per file, while the four Metal
/// paths (`MetalQwen35State::generate`, `generate_streaming` /
/// `generate_streaming_with_cancel`, and
/// `generate_streaming_with_prefix_cache_and_cancel`) silently accepted it
/// and returned a normal-looking empty `Ok(GenerateOutput { stopped: false,
/// stop_reason: None, .. })` — a result shape that is itself
/// invariant-violating (a "completed" generation that neither stopped nor
/// gives a reason). Maintainer sign-off on the matrix (PR #855) ruled this
/// unify on the CPU behavior; this function is that shared preflight, and
/// every one of the seven entry points now calls it instead of carrying its
/// own inline `if prompt_len == 0` copy.
///
/// Unlike the `check_*_not_set` siblings above (which reject an unsupported
/// *config field*), this guard rejects the *input*: `prompt_len` is the
/// tokenized prompt length every call site already computes before any
/// state allocation, so this call replaces each site's own inline check
/// rather than adding a new one alongside it.
pub(crate) fn check_prompt_not_empty(prompt_len: usize) -> Result<(), InferenceError> {
    if prompt_len == 0 {
        return Err(InferenceError::Inference("empty prompt".into()));
    }
    Ok(())
}

/// Rejects tokenizer output that cannot index the configured embedding table.
///
/// Standalone CPU generation drivers accept their tokenizer and model config
/// independently, so tokenizer validity alone does not prove that every prompt
/// ID is below `vocab_size`. Perform one cold ingress scan before any decoder
/// state allocation instead of branching inside the per-token forward path.
pub(crate) fn check_prompt_ids_in_vocab(
    prompt_ids: &[u32],
    vocab_size: usize,
) -> Result<(), InferenceError> {
    if let Some(&bad_id) = prompt_ids.iter().find(|&&id| id as usize >= vocab_size) {
        return Err(InferenceError::InvalidInput(format!(
            "prompt contains out-of-vocabulary token id {bad_id} (vocab_size={vocab_size})"
        )));
    }
    Ok(())
}

/// Shared total-context admission bound (#922): rejects a request whose
/// prompt fits the window alone but whose prompt plus decode budget does
/// not, closing exactly the gap between "will prefill" and "will actually
/// be able to decode the requested completion."
///
/// This is the same "prompt plus requested completion fits the window"
/// bound `generate` / `generate_streaming` (this module) apply inline —
/// `effective_new` is `decode_cap(reasoning_budget, max_new_tokens)`, the
/// budget-extended cap (equal to `max_new_tokens` when reasoning is
/// unbudgeted), matching the true furthest position the decode loop can
/// reach. Before #922, the three Metal entry points in
/// `forward::metal_qwen35` (`generate`, `generate_streaming_with_cancel`,
/// `generate_streaming_with_prefix_cache_and_cancel`) only checked
/// `prompt_len > max_context`, so a request that fit the window by itself
/// but would run the decode budget past it was admitted and either
/// silently truncated mid-decode (the streaming paths self-limit on
/// `seq_len >= max_cache_len`) or, worse, indexed the RoPE table
/// out-of-bounds in the non-streaming `generate` (a release panic). Direct
/// library callers and `chat_metal` hit this gap directly; HTTP callers
/// behind `lattice_serve`/`lattice.rs` were already shielded by
/// `serve::metal_worker::check_prompt_fits_window`'s independent
/// (pre-existing) enforcement of the same total bound.
///
/// Every call site is expected to have already confirmed `prompt_len > 0`
/// (via [`check_prompt_not_empty`]) and `max_new_tokens > 0` (typically via
/// an early `max_new_tokens == 0` return) before calling this — those are
/// separate, distinct guards with their own contracts and error shapes;
/// this function only enforces the arithmetic bound.
pub(crate) fn check_context_budget(
    prompt_len: usize,
    reasoning_budget: Option<usize>,
    max_new_tokens: usize,
    max_context: usize,
) -> Result<(), InferenceError> {
    let effective_new = decode_cap(reasoning_budget, max_new_tokens);
    if prompt_len.saturating_add(effective_new) > max_context {
        let reasoning_detail = reasoning_budget
            .map(|budget| format!(", reasoning_budget={budget}"))
            .unwrap_or_default();
        return Err(InferenceError::Inference(format!(
            "prompt ({prompt_len} tokens) plus effective decode cap ({effective_new} tokens; \
             max_new_tokens={max_new_tokens}{reasoning_detail}) exceeds \
             model context window ({max_context})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
