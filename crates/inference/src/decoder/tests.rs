use super::*;

// -----------------------------------------------------------------
// Named rule 1: a stale PredictionId is rejected; a live one is accepted
// by the same call.
// -----------------------------------------------------------------

#[test]
fn consume_rejects_a_stale_id_but_accepts_a_live_one() {
    let mut ledger = PredictionLedger::new();
    let first = ledger.open();
    ledger.reset(); // invalidates `first`; it is now genuinely stale
    let second = ledger.open();

    assert_eq!(
        ledger.consume(first),
        Err(PredictionError::Stale),
        "a prefill/reset-invalidated id must be rejected"
    );
    // Control: the same call (`consume`), same ledger, live id is accepted.
    assert_eq!(ledger.consume(second), Ok(()));
}

// -----------------------------------------------------------------
// Named rule 2: an AcceptedToken cannot be consumed twice; the first
// consume succeeds.
// -----------------------------------------------------------------

#[test]
fn accepted_token_cannot_be_consumed_twice() {
    let mut ledger = PredictionLedger::new();
    let id = ledger.open();
    let token = AcceptedToken {
        final_id: 7,
        prediction: id,
    };

    // Control: the first consume of this token's prediction succeeds.
    assert_eq!(ledger.consume(token.prediction), Ok(()));
    // The same token's prediction cannot be consumed a second time.
    assert_eq!(
        ledger.consume(token.prediction),
        Err(PredictionError::Stale)
    );
    assert_eq!(token.final_id, 7);
}

// -----------------------------------------------------------------
// Named rule 3: a cancelled prediction is neither selectable nor
// scoreable; before the cancel, the same id passes the same
// eligibility check.
// -----------------------------------------------------------------

#[test]
fn cancelled_prediction_is_not_eligible_control_checks_before_cancel() {
    let mut ledger = PredictionLedger::new();
    let id = ledger.open();

    // Control: before cancellation, the id passes the eligibility check a
    // concrete session's select/metadata operations would consult.
    assert!(ledger.is_live(id));

    ledger.invalidate(); // cancellation, failure, or finish

    assert!(!ledger.is_live(id));
}

// -----------------------------------------------------------------
// Named rule 4: reset() (another prefill) invalidates the live
// prediction, with the same before/after control.
// -----------------------------------------------------------------

#[test]
fn reset_invalidates_the_live_prediction() {
    let mut ledger = PredictionLedger::new();
    let id = ledger.open();
    assert!(ledger.is_live(id)); // control: live immediately after open

    ledger.reset();

    assert!(!ledger.is_live(id));
}

// -----------------------------------------------------------------
// The open()-while-live decision (supersede, not refuse): documented
// above on `PredictionLedger::open`, tested here.
// -----------------------------------------------------------------

#[test]
fn opening_while_one_is_live_supersedes_the_previous_prediction() {
    let mut ledger = PredictionLedger::new();
    let first = ledger.open();
    assert!(ledger.is_live(first)); // control: live immediately after open

    let second = ledger.open(); // opening again while `first` is still live

    assert_ne!(first, second);
    assert!(
        !ledger.is_live(first),
        "the superseded id must no longer be live"
    );
    assert!(
        ledger.is_live(second),
        "the newly opened id must be the ledger's one live prediction"
    );
}

#[test]
fn one_ledgers_prediction_is_not_live_or_consumable_in_another() {
    // Two sessions, each with its own ledger, both at sequence zero. Before this type
    // carried a ledger identity, both minted an identical id, and since `is_live` and
    // `consume` compare ids by value, session A's live prediction was accepted and
    // consumed by session B -- the cross-session confusion the whole type exists to
    // prevent, reported as a clean success.
    let mut a = PredictionLedger::new();
    let mut b = PredictionLedger::new();
    let from_a = a.open();
    let from_b = b.open();

    assert_ne!(
        from_a, from_b,
        "two fresh ledgers must not mint equal ids at the same sequence"
    );
    // Control: each id IS live in the ledger that minted it, so the rejections below
    // are about issuer identity and not about a ledger that simply has nothing live.
    assert!(a.is_live(from_a));
    assert!(b.is_live(from_b));

    assert!(!b.is_live(from_a), "a foreign id must not be eligible");
    assert_eq!(
        b.consume(from_a),
        Err(PredictionError::Stale),
        "a foreign id must not consume this ledger's live prediction"
    );
    // And the victim's own prediction survived the foreign attempt.
    assert!(b.is_live(from_b));
}

#[test]
fn ids_are_never_reused_across_resets() {
    // The property an `epoch` field appeared to provide. It is `seq` monotonicity that
    // actually provides it, so it is pinned here instead of implied by a field nothing
    // reads. A ledger that reset its counter would hand out an id equal to a retired
    // one, and `is_live` would then answer `true` for a genuinely stale prediction.
    let mut ledger = PredictionLedger::new();
    let mut seen = Vec::new();
    for _ in 0..4 {
        seen.push(ledger.open());
        ledger.reset();
    }
    let first = seen[0];
    assert!(
        seen.iter().skip(1).all(|id| *id != first),
        "a reset must not let the ledger re-mint a retired id: {seen:?}"
    );
    assert!(
        !ledger.is_live(first),
        "and the retired id must not be live after three further opens"
    );
}

#[test]
fn the_blanket_impl_covers_a_capturing_non_mutating_closure() {
    // The covered case, stated as a test rather than only in the doc comment: a
    // closure that captures state and reads it is `Fn`, so it is a `Cancellation`.
    // A closure that MUTATES its capture is `FnMut` and is not accepted here; that
    // arm cannot be written as a passing test, which is why the limitation lives in
    // the doc comment above and in the row that has to convert the serving path.
    let flag = std::cell::Cell::new(false);
    let cancel = || flag.get();
    assert!(!(&cancel as &dyn Cancellation).is_cancelled());
    flag.set(true);
    assert!((&cancel as &dyn Cancellation).is_cancelled());
}

// -----------------------------------------------------------------
// Object safety of DecoderSession, plus a happy-path exercise of
// every remaining pub(crate) item in this module through the trait's
// full lifecycle (prefill -> select -> metadata -> decode -> finish).
// -----------------------------------------------------------------

struct DummySession {
    caps: ExecutionCapabilities,
    ledger: PredictionLedger,
}

impl DecoderSession for DummySession {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &self.caps
    }

    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before prefill".into()));
        }
        Ok(StepStamp {
            evaluated_len: 3,
            prediction: None,
        })
    }

    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        _cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        self.ledger.consume(accepted.prediction)?;
        Ok(StepStamp {
            evaluated_len: 4,
            prediction: None,
        })
    }

    fn select(&mut self, _request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        let prediction = self.ledger.open();
        Ok(SelectOutcome::Candidate(SelectionCandidate {
            candidate_id: 11,
            prediction,
        }))
    }

    fn metadata(
        &mut self,
        prediction: PredictionId,
        final_token: u32,
        request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        let top = match request.top_logprobs {
            Some(_) => vec![TopLogprob {
                token_id: final_token,
                logprob: -0.1,
            }],
            None => Vec::new(),
        };
        Ok(TokenMetadata {
            prediction,
            final_token_id: final_token,
            final_logprob: -0.1,
            top,
        })
    }

    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError> {
        if disposition == FinishDisposition::Poisoned {
            self.ledger.invalidate();
        }
        Ok(())
    }
}

#[test]
fn default_interruptible_prefill_preserves_driver_owned_cancellation() {
    let mut session = DummySession {
        caps: ExecutionCapabilities::default(),
        ledger: PredictionLedger::new(),
    };
    let polls = std::cell::Cell::new(0);
    let cancel = || {
        polls.set(polls.get() + 1);
        true
    };
    let stamp = session.prefill_interruptible(&cancel).unwrap().unwrap();
    assert_eq!(stamp.evaluated_len, 3);
    assert_eq!(stamp.prediction, None);
    assert_eq!(polls.get(), 0);
    assert!(session.prefill(&cancel).is_err());
    assert_eq!(polls.get(), 1);
}

#[test]
fn decoder_session_is_object_safe() {
    let mut dummy = DummySession {
        caps: ExecutionCapabilities {
            logprobs: true,
            ..ExecutionCapabilities::default()
        },
        ledger: PredictionLedger::new(),
    };

    // The coercion below is the object-safety proof: it only type-checks if every
    // `DecoderSession` method is dispatchable through a vtable.
    let session: &mut dyn DecoderSession = &mut dummy;

    assert!(session.capabilities().logprobs);

    let cancel = || false;
    let stamp = session
        .prefill(&cancel)
        .expect("dummy prefill always succeeds when not cancelled");
    assert_eq!(stamp.evaluated_len, 3);

    let cfg = GenerateConfig::default();
    let history: Vec<u32> = Vec::new();
    let request = SelectionRequest {
        config: &cfg,
        history: &history,
        grammar_mask: None,
    };
    let candidate = match session
        .select(&request)
        .expect("dummy select always succeeds")
    {
        SelectOutcome::Candidate(c) => c,
        SelectOutcome::GrammarExhausted => panic!("dummy select never reports exhaustion"),
    };

    // Policy finalization and requested metadata happen before decode (D2's
    // lifecycle order), against the prediction's pre-advance scoring view.
    let meta_request = MetadataRequest {
        top_logprobs: Some(1),
    };
    let metadata = session
        .metadata(candidate.prediction, candidate.candidate_id, &meta_request)
        .expect("dummy metadata always succeeds");
    assert_eq!(metadata.final_token_id, candidate.candidate_id);
    assert_eq!(metadata.top.len(), 1);

    let accepted = AcceptedToken {
        final_id: metadata.final_token_id,
        prediction: candidate.prediction,
    };
    session
        .decode(&accepted, &cancel)
        .expect("dummy decode consumes the freshly opened prediction");

    session
        .finish(FinishDisposition::Reusable)
        .expect("dummy finish always succeeds");
}
