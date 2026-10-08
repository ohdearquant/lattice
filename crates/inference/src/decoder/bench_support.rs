//! Measurement-only seams for the per-token allocation instrument (ADR-090 D7, row R01).
//!
//! Compiled only with `--features bench-internals`; the default public API and every
//! shipped decode path are unchanged. Nothing here alters what the Qwen CPU route does:
//! the streaming entry point below calls the production dispatch function as it is, and
//! the decorator around the real session forwards every call to it unmodified.
//!
//! **Why this exists.** `inference_perf` registers a counting allocator, and a global
//! allocator belongs to the final binary, so the measured consumers have to be driven
//! from that bench. The shared driver and the session trait are crate-private, so the
//! bench reaches them through this module instead of widening their visibility.
//!
//! **Three entry points, three questions.**
//!
//! * [`run_qwen_cpu_streaming`]: the production streaming entry itself
//!   (`generate_streaming_with_trace`), with the raw-event observer the bench uses to
//!   bracket the warm region. This is the real consumer.
//! * [`run_qwen_cpu_probed`]: the same driver over the same `QwenCpuSession`, with the
//!   session wrapped so the bench can act inside `select` and `decode`. This is where a
//!   deliberately retained allocation is placed; the retaining code lives in the bench,
//!   never here, so no build contains it unless the bench is the one running.
//! * [`run_interface_only`]: the same driver over a session that does no model work. Its
//!   own allocation total is the bounded interface-only control: an allocation added to
//!   the per-token interface cannot be hidden behind a removal elsewhere in a model.
//!
//! The probed entry mirrors `Qwen35Model::generate_streaming_via_driver` step for step
//! (tokenize, preflight, session, `driver::run`) for requests without a reasoning budget. The bench proves the mirror is faithful
//! rather than assuming it: an unchanged probe must reproduce the production entry's
//! token ids and allocation counts.

use super::driver;
use super::qwen_cpu::QwenCpuSession;
use super::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionId, PredictionLedger, SelectOutcome, SelectionCandidate,
    SelectionRequest, StepStamp, TokenMetadata,
};
use crate::error::InferenceError;
use crate::generation::GenerateConfig;
use crate::model::qwen35::{Qwen35Model, RawGenEvent, check_prompt_not_empty};
use crate::tokenizer::Tokenizer;
use crate::tokenizer::detokenize::IncrementalDetokenizer;

/// Hooks fired by the wrapped session, in the worker that runs the call, immediately
/// after the real `select` / `decode` returns. They sit inside the per-token interface:
/// each token's `select` and `decode` is reached exactly once per iteration.
pub trait InterfaceProbe {
    fn after_select(&mut self) {}
    fn after_decode(&mut self) {}
}

/// The unchanged interface: both hooks do nothing and allocate nothing.
pub struct NoProbe;

impl InterfaceProbe for NoProbe {}

/// What one call through the shared driver produced.
pub struct RouteRun {
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    /// Predictions opened (`select` calls that returned a candidate).
    pub opened: usize,
    /// Predictions consumed (`decode` calls).
    pub consumed: usize,
    /// Capacity, in `f32` elements, of layer 0's key cache immediately after prefill.
    /// `None` on routes with no model cache.
    pub initial_kv_capacity_floats: Option<usize>,
}

/// The production streaming entry with a raw-event observer. The route marker is the
/// driver trace it returns: `consumed + 1 == opened` only when the loop ran through
/// `driver::run`.
pub fn run_qwen_cpu_streaming(
    model: &Qwen35Model,
    prompt: &str,
    gen_cfg: &GenerateConfig,
    on_raw_event: &mut dyn FnMut(RawGenEvent),
) -> Result<RouteRun, InferenceError> {
    let (output, trace) =
        model.generate_streaming_with_trace(prompt, gen_cfg, |_| true, || false, on_raw_event)?;
    Ok(RouteRun {
        token_ids: output.token_ids,
        prompt_tokens: output.prompt_tokens,
        opened: trace.opened,
        consumed: trace.consumed,
        initial_kv_capacity_floats: None,
    })
}

struct ProbedSession<'p, S: DecoderSession> {
    inner: S,
    probe: &'p mut dyn InterfaceProbe,
    capacity_of: fn(&S) -> Option<usize>,
    initial_capacity: Option<usize>,
}

impl<S: DecoderSession> DecoderSession for ProbedSession<'_, S> {
    fn capabilities(&self) -> &ExecutionCapabilities {
        self.inner.capabilities()
    }

    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        let stamp = self.inner.prefill(cancel);
        self.initial_capacity = (self.capacity_of)(&self.inner);
        stamp
    }

    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        let stamp = self.inner.decode(accepted, cancel);
        self.probe.after_decode();
        stamp
    }

    fn select(&mut self, request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        let outcome = self.inner.select(request);
        self.probe.after_select();
        outcome
    }

    fn metadata(
        &mut self,
        prediction: PredictionId,
        final_token: u32,
        request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        self.inner.metadata(prediction, final_token, request)
    }

    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError> {
        self.inner.finish(disposition)
    }
}

/// The streaming driver call over the real Qwen CPU session, with `probe` wrapped around
/// the session. Refuses a request with a reasoning budget: the instrument's arms never set
/// one, and this mirror does not carry that branch's setup.
pub fn run_qwen_cpu_probed(
    model: &Qwen35Model,
    prompt: &str,
    gen_cfg: &GenerateConfig,
    probe: &mut dyn InterfaceProbe,
    on_raw_event: &mut dyn FnMut(RawGenEvent),
) -> Result<RouteRun, InferenceError> {
    let cfg = &model.config;

    let input = model.tokenizer.tokenize(prompt);
    let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
    let prompt_len = prompt_ids.len();
    check_prompt_not_empty(prompt_len)?;
    if gen_cfg.effective_reasoning_budget().is_some() {
        return Err(InferenceError::InvalidInput(
            "the probed route does not carry the reasoning-budget setup".into(),
        ));
    }
    if prompt_len.saturating_add(gen_cfg.max_new_tokens) > model.max_context() {
        return Err(InferenceError::InvalidInput(
            "prompt plus max_new_tokens exceeds the model context".into(),
        ));
    }
    let think_close_id = None;

    let mut session = ProbedSession {
        inner: QwenCpuSession::new(model, prompt_ids.clone(), gen_cfg.temperature, gen_cfg.seed),
        probe,
        capacity_of: QwenCpuSession::first_layer_key_capacity,
        initial_capacity: None,
    };

    let never_cancel = || false;
    let on_raw_event_cell = std::cell::RefCell::new(on_raw_event);
    let detok_cell = std::cell::RefCell::new(IncrementalDetokenizer::new());
    let mut text = String::new();
    let mut token_logprob_end_offsets: Vec<usize> = Vec::new();

    let result = driver::run(
        &mut session,
        gen_cfg,
        think_close_id,
        &prompt_ids,
        cfg.eos_token_id,
        true,
        &never_cancel,
        |generated_len| {
            (*on_raw_event_cell.borrow_mut())(RawGenEvent::RawToken {
                index: generated_len,
            });
        },
        |next_id| detok_cell.borrow_mut().push(&model.tokenizer, next_id),
        &mut text,
        &mut token_logprob_end_offsets,
        |_delta, _next_id| true,
        || (*on_raw_event_cell.borrow_mut())(RawGenEvent::PrefillEnd),
        || detok_cell.borrow_mut().finish(),
    )?;

    Ok(RouteRun {
        token_ids: result.generated_ids,
        prompt_tokens: prompt_len,
        opened: result.trace.opened,
        consumed: result.trace.consumed,
        initial_kv_capacity_floats: session.initial_capacity,
    })
}

/// A session that does no model work: it accepts every prediction and always proposes the
/// same token. Whatever the driver, the policy and the ledger allocate around it is all that
/// an interface-only run counts.
struct NullSession {
    ledger: PredictionLedger,
    capabilities: ExecutionCapabilities,
    prompt_len: usize,
    evaluated: usize,
}

impl DecoderSession for NullSession {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &self.capabilities
    }

    fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        self.evaluated = self.prompt_len;
        Ok(StepStamp {
            evaluated_len: self.evaluated,
            prediction: None,
        })
    }

    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        _cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        self.ledger.consume(accepted.prediction)?;
        self.evaluated += 1;
        Ok(StepStamp {
            evaluated_len: self.evaluated,
            prediction: None,
        })
    }

    fn select(&mut self, _request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        let prediction = self.ledger.open();
        Ok(SelectOutcome::Candidate(SelectionCandidate {
            candidate_id: INTERFACE_ONLY_TOKEN,
            prediction,
        }))
    }

    fn metadata(
        &mut self,
        _prediction: PredictionId,
        _final_token: u32,
        _request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        Err(InferenceError::InvalidInput(
            "the interface-only session scores nothing".into(),
        ))
    }

    fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
        Ok(())
    }
}

const INTERFACE_ONLY_TOKEN: u32 = 1;

/// The shared driver over [`NullSession`]. `prompt_len` prompt tokens, `gen_cfg` as given,
/// the same streaming mode and no-op detokenizer the fast path uses.
pub fn run_interface_only(
    prompt_len: usize,
    gen_cfg: &GenerateConfig,
    probe: &mut dyn InterfaceProbe,
    on_raw_event: &mut dyn FnMut(RawGenEvent),
) -> Result<RouteRun, InferenceError> {
    let prompt_ids: Vec<u32> = vec![INTERFACE_ONLY_TOKEN; prompt_len];
    let mut session = ProbedSession {
        inner: NullSession {
            ledger: PredictionLedger::new(),
            capabilities: ExecutionCapabilities::default(),
            prompt_len,
            evaluated: 0,
        },
        probe,
        capacity_of: |_| None,
        initial_capacity: None,
    };

    let never_cancel = || false;
    let on_raw_event_cell = std::cell::RefCell::new(on_raw_event);
    let mut text = String::new();
    let mut token_logprob_end_offsets: Vec<usize> = Vec::new();

    let result = driver::run(
        &mut session,
        gen_cfg,
        None,
        &prompt_ids,
        u32::MAX,
        true,
        &never_cancel,
        |generated_len| {
            (*on_raw_event_cell.borrow_mut())(RawGenEvent::RawToken {
                index: generated_len,
            });
        },
        |_next_id| String::new(),
        &mut text,
        &mut token_logprob_end_offsets,
        |_delta, _next_id| true,
        || (*on_raw_event_cell.borrow_mut())(RawGenEvent::PrefillEnd),
        String::new,
    )?;

    Ok(RouteRun {
        token_ids: result.generated_ids,
        prompt_tokens: prompt_len,
        opened: result.trace.opened,
        consumed: result.trace.consumed,
        initial_kv_capacity_floats: None,
    })
}
