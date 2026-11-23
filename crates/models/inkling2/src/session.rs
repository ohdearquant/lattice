//! One generation request behind `lattice_arch::decoder::DecoderSession`: the prompt evaluated in
//! chunks, then a sampled token and one forward pass per accepted token.

use lattice_arch::__internal::{
    check_prompt_not_empty, compute_step_logprobs, initial_rng_state, sample_full_logits,
};
use lattice_arch::decoder::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionError, PredictionId, PredictionLedger, SelectOutcome,
    SelectionCandidate, SelectionRequest, StepStamp, TokenMetadata,
};
use lattice_inference_core::InferenceError;

use crate::cache::Inkling2Cache;
use crate::model::Inkling2Model;

/// Prompt tokens per forward call during prefill. Cancellation is checked between calls.
pub(crate) const PREFILL_CHUNK: usize = 64;

/// Grammar masks, log-probabilities, stop strings and reasoning budgets are not supported, so
/// the driver refuses a request that asks for any of them before the first session call.
const CAPABILITIES: ExecutionCapabilities = ExecutionCapabilities::none();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Created: only `prefill` and `finish` are accepted.
    Fresh,
    /// The logits of the next position are held for `select` and `metadata`.
    Ready,
    /// A forward call was cancelled or failed part way, so the cache holds part of it: only
    /// `finish` is accepted.
    Broken,
    /// Finished: nothing is accepted.
    Finished,
}

impl Phase {
    fn describe(self) -> &'static str {
        match self {
            Self::Fresh => "not prefilled",
            Self::Ready => "ready",
            Self::Broken => "broken by an interrupted or failed forward call",
            Self::Finished => "finished",
        }
    }
}

/// A decoding session over a borrowed [`Inkling2Model`], owning its cache, the logits of the
/// next position and its sampling state.
pub(crate) struct Inkling2CpuSession<'model> {
    model: &'model Inkling2Model,
    cache: Inkling2Cache,
    prompt_ids: Vec<u32>,
    logits: Vec<f32>,
    rng_state: u64,
    temperature: f32,
    ledger: PredictionLedger,
    phase: Phase,
    prefill_chunk: usize,
}

impl<'model> Inkling2CpuSession<'model> {
    /// A session for `prompt_ids` with an empty cache. `temperature` scales the log-probabilities
    /// `metadata` reports; `seed` starts the sampler, from the clock when `None`.
    ///
    /// # Errors
    ///
    /// The prediction ledger's error when the process has no ledger identity left.
    pub(crate) fn new(
        model: &'model Inkling2Model,
        prompt_ids: Vec<u32>,
        temperature: f32,
        seed: Option<u64>,
    ) -> Result<Self, InferenceError> {
        Ok(Self {
            model,
            cache: model.new_cache(),
            prompt_ids,
            logits: Vec::new(),
            rng_state: initial_rng_state(seed),
            temperature,
            ledger: PredictionLedger::new()?,
            phase: Phase::Fresh,
            prefill_chunk: PREFILL_CHUNK,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_prefill_chunk(mut self, tokens: usize) -> Self {
        self.prefill_chunk = tokens.max(1);
        self
    }

    #[cfg(test)]
    pub(crate) fn logits(&self) -> &[f32] {
        &self.logits
    }

    #[cfg(test)]
    pub(crate) fn position(&self) -> usize {
        self.cache.position()
    }

    fn require(&self, phase: Phase, call: &str) -> Result<(), InferenceError> {
        if self.phase == phase {
            Ok(())
        } else {
            Err(InferenceError::InvalidInput(format!(
                "{call} is out of order: the session is {}",
                self.phase.describe()
            )))
        }
    }

    /// Runs `tokens` after the cache and keeps the logits of the last one. A failed call leaves
    /// part of it in the cache, so the session is broken.
    fn run(&mut self, tokens: &[u32]) -> Result<(), InferenceError> {
        match self.model.forward(tokens, &mut self.cache) {
            Ok(logits) => {
                self.logits = logits;
                Ok(())
            }
            Err(error) => {
                self.phase = Phase::Broken;
                Err(error)
            }
        }
    }
}

impl DecoderSession for Inkling2CpuSession<'_> {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &CAPABILITIES
    }

    /// Evaluates the prompt in chunks of up to [`PREFILL_CHUNK`] tokens. A cancellation seen
    /// before the first chunk changes nothing; one seen between chunks breaks the session.
    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        self.require(Phase::Fresh, "prefill")?;
        check_prompt_not_empty(self.prompt_ids.len())?;
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before prefill".into()));
        }
        let prompt = std::mem::take(&mut self.prompt_ids);
        let mut result = Ok(());
        for (index, chunk) in prompt.chunks(self.prefill_chunk).enumerate() {
            if index > 0 && cancel.is_cancelled() {
                self.phase = Phase::Broken;
                result = Err(InferenceError::Inference("cancelled during prefill".into()));
                break;
            }
            result = self.run(chunk);
            if result.is_err() {
                break;
            }
        }
        self.prompt_ids = prompt;
        result?;
        self.ledger.reset();
        self.phase = Phase::Ready;
        Ok(StepStamp::new(self.cache.position(), None))
    }

    fn prefill_interruptible(
        &mut self,
        cancel: &dyn Cancellation,
    ) -> Result<Option<StepStamp>, InferenceError> {
        let interrupted = std::cell::Cell::new(false);
        let observe = || {
            let cancelled = cancel.is_cancelled();
            interrupted.set(interrupted.get() || cancelled);
            cancelled
        };
        match self.prefill(&observe) {
            Ok(stamp) => Ok(Some(stamp)),
            Err(_) if interrupted.get() => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Consumes the accepted token's prediction and runs the token. A token outside the
    /// vocabulary, a stale prediction and a cancellation are refused before anything changes.
    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        self.require(Phase::Ready, "decode")?;
        let vocab = self.model.config().unpadded_vocab_size;
        if accepted.final_id as usize >= vocab {
            return Err(InferenceError::InvalidInput(format!(
                "token {} is outside the vocabulary of {vocab} tokens",
                accepted.final_id
            )));
        }
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before decode".into()));
        }
        self.ledger.consume(accepted.prediction)?;
        self.run(&[accepted.final_id])?;
        Ok(StepStamp::new(self.cache.position(), None))
    }

    /// Samples from the held logits with the request's sampling settings and opens the
    /// candidate's prediction. A grammar mask is refused: the session does not declare grammar.
    fn select(&mut self, request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        self.require(Phase::Ready, "select")?;
        if request.grammar_mask.is_some() {
            return Err(InferenceError::InvalidInput(
                "grammar-constrained selection is not supported by the Inkling 2 session".into(),
            ));
        }
        let candidate = sample_full_logits(
            &self.logits,
            request.config,
            request.history,
            &mut self.rng_state,
            request.config.min_p,
            0.0,
        );
        let prediction = self.ledger.open()?;
        Ok(SelectOutcome::Candidate(SelectionCandidate::new(
            candidate, prediction,
        )))
    }

    fn metadata(
        &mut self,
        prediction: PredictionId,
        final_token: u32,
        request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        self.require(Phase::Ready, "metadata")?;
        let Some(top_n) = request.top_logprobs else {
            return Err(InferenceError::InvalidInput(
                "metadata capture is disabled in this request".into(),
            ));
        };
        if !self.ledger.is_live(prediction) {
            return Err(PredictionError::Stale.into());
        }
        if final_token as usize >= self.logits.len() {
            return Err(InferenceError::InvalidInput(format!(
                "token {final_token} is outside the vocabulary of {} tokens",
                self.logits.len()
            )));
        }
        let (logprob, top) =
            compute_step_logprobs(&self.logits, final_token, self.temperature, top_n);
        Ok(TokenMetadata::new(prediction, final_token, logprob, top))
    }

    fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
        if self.phase == Phase::Finished {
            return Err(InferenceError::InvalidInput(
                "finish is out of order: the session is finished".into(),
            ));
        }
        self.ledger.invalidate();
        self.phase = Phase::Finished;
        Ok(())
    }
}
