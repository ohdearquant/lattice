use super::*;
use crate::decoder::{
    PredictionId, PredictionLedger, SelectionCandidate, StepStamp, TokenMetadata,
};
use std::cell::Cell;

struct RecordingSession {
    caps: ExecutionCapabilities,
    ledger: PredictionLedger,
    prediction: Option<PredictionId>,
    calls: Vec<&'static str>,
    finishes: Vec<FinishDisposition>,
    failure: Option<&'static str>,
    finish_error: bool,
}

impl RecordingSession {
    fn new(failure: Option<&'static str>) -> Self {
        Self {
            caps: ExecutionCapabilities {
                grammar: true,
                logprobs: true,
                stop_strings: true,
                reasoning_budget: true,
            },
            ledger: PredictionLedger::new(),
            prediction: None,
            calls: Vec::new(),
            finishes: Vec::new(),
            failure,
            finish_error: false,
        }
    }

    fn record(&mut self, call: &'static str) -> Result<(), InferenceError> {
        self.calls.push(call);
        if self.failure == Some(call) {
            return Err(InferenceError::Inference(call.into()));
        }
        Ok(())
    }
}

impl DecoderSession for RecordingSession {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &self.caps
    }

    fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        self.record("prefill")?;
        Ok(StepStamp {
            evaluated_len: 1,
            prediction: None,
        })
    }

    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        _cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        self.record("decode")?;
        self.ledger.consume(accepted.prediction)?;
        Ok(StepStamp {
            evaluated_len: 2,
            prediction: None,
        })
    }

    fn select(&mut self, _request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        self.record("select")?;
        if self.failure == Some("grammar") {
            return Ok(SelectOutcome::GrammarExhausted);
        }
        let prediction = self.ledger.open();
        self.prediction = Some(prediction);
        Ok(SelectOutcome::Candidate(SelectionCandidate {
            candidate_id: 1,
            prediction,
        }))
    }

    fn metadata(
        &mut self,
        prediction: PredictionId,
        final_token: u32,
        _request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        self.record("metadata")?;
        Ok(TokenMetadata {
            prediction,
            final_token_id: final_token,
            final_logprob: 0.0,
            top: Vec::new(),
        })
    }

    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError> {
        self.finishes.push(disposition);
        self.ledger.invalidate();
        self.record("finish")?;
        if self.finish_error {
            return Err(InferenceError::Inference("finish failed".into()));
        }
        Ok(())
    }
}

fn drive(
    session: &mut RecordingSession,
    cancel: &dyn Cancellation,
) -> Result<DriverResult, InferenceError> {
    let config = GenerateConfig {
        max_new_tokens: 2,
        logprobs: Some(0),
        ..Default::default()
    };
    run(
        session,
        &config,
        None,
        &[0],
        99,
        true,
        cancel,
        |_| {},
        |_| "a".into(),
        &mut String::new(),
        &mut Vec::new(),
        |_, _| true,
        || {},
        String::new,
    )
}

#[test]
fn ordinary_completion_and_every_cancel_checkpoint_finish_once() {
    for cancel_at in [None, Some(0), Some(1), Some(2)] {
        let mut session = RecordingSession::new(None);
        let polls = Cell::new(0);
        let cancel = || {
            let at = polls.get();
            polls.set(at + 1);
            cancel_at == Some(at)
        };
        let result = drive(&mut session, &cancel).expect("driver result");
        assert_eq!(
            result.stop_reason,
            if cancel_at.is_some() {
                StopReason::Interrupt
            } else {
                StopReason::Length
            }
        );
        let expected = if matches!(cancel_at, Some(1 | 2)) {
            FinishDisposition::Poisoned
        } else {
            FinishDisposition::Reusable
        };
        assert_eq!(
            session.finishes,
            [expected],
            "cancel checkpoint {cancel_at:?}"
        );
        if let Some(prediction) = session.prediction {
            assert!(!session.ledger.is_live(prediction));
        }
        if cancel_at == Some(0) {
            assert_eq!(session.calls, ["finish"]);
        }
    }
}

#[test]
fn ordinary_execution_errors_finish_poisoned_once() {
    for failure in ["prefill", "select", "decode", "metadata", "grammar"] {
        let mut session = RecordingSession::new(Some(failure));
        let result = drive(&mut session, &|| false);
        assert!(result.is_err(), "{failure}");
        assert!(session.calls.contains(&if failure == "grammar" {
            "select"
        } else {
            failure
        }));
        assert_eq!(session.finishes, [FinishDisposition::Poisoned], "{failure}");
        if let Some(prediction) = session.prediction {
            assert!(!session.ledger.is_live(prediction));
        }
    }
}

#[test]
fn finish_errors_preserve_the_primary_execution_error() {
    let mut session = RecordingSession::new(Some("prefill"));
    session.finish_error = true;
    let result = drive(&mut session, &|| false);
    assert!(matches!(result, Err(InferenceError::Inference(message)) if message == "prefill"));
    assert_eq!(session.finishes, [FinishDisposition::Poisoned]);

    let mut session = RecordingSession::new(None);
    session.finish_error = true;
    let result = drive(&mut session, &|| false);
    assert!(
        matches!(result, Err(InferenceError::Inference(message)) if message == "finish failed")
    );
    assert_eq!(session.finishes, [FinishDisposition::Reusable]);
}
