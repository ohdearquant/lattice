use super::*;

/// A fresh ledger, shared by the decoder and driver unit tests.
pub(super) fn fresh_ledger() -> PredictionLedger {
    PredictionLedger::new().expect("a fresh ledger identity")
}

// -----------------------------------------------------------------
// Named rule 1: a stale PredictionId is rejected; a live one is accepted
// by the same call.
// -----------------------------------------------------------------

#[test]
fn consume_rejects_a_stale_id_but_accepts_a_live_one() {
    let mut ledger = fresh_ledger();
    let first = ledger.open().expect("an unspent sequence number");
    ledger.reset(); // invalidates `first`; it is now genuinely stale
    let second = ledger.open().expect("an unspent sequence number");

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
    let mut ledger = fresh_ledger();
    let id = ledger.open().expect("an unspent sequence number");
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
    let mut ledger = fresh_ledger();
    let id = ledger.open().expect("an unspent sequence number");

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
    let mut ledger = fresh_ledger();
    let id = ledger.open().expect("an unspent sequence number");
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
    let mut ledger = fresh_ledger();
    let first = ledger.open().expect("an unspent sequence number");
    assert!(ledger.is_live(first)); // control: live immediately after open

    // Opening again while `first` is still live.
    let second = ledger.open().expect("an unspent sequence number");

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
    let mut a = fresh_ledger();
    let mut b = fresh_ledger();
    let from_a = a.open().expect("an unspent sequence number");
    let from_b = b.open().expect("an unspent sequence number");

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
    let mut ledger = fresh_ledger();
    let mut seen = Vec::new();
    for _ in 0..4 {
        seen.push(ledger.open().expect("an unspent sequence number"));
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
fn a_spent_identity_counter_refuses_instead_of_wrapping() {
    // The last identity below the counter's ceiling is still handed out; the ceiling itself
    // is not, and the refusal leaves the counter where it was, so a retry is refused again
    // rather than wrapping to identity zero, which the process's first ledger holds.
    let counter = AtomicU64::new(u64::MAX - 1);
    assert_eq!(allocate_ledger_id(&counter), Ok(u64::MAX - 1));
    assert_eq!(
        allocate_ledger_id(&counter),
        Err(PredictionError::Exhausted)
    );
    assert_eq!(
        allocate_ledger_id(&counter),
        Err(PredictionError::Exhausted)
    );
    assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
}

#[test]
fn a_spent_sequence_refuses_without_retiring_the_live_prediction() {
    let mut ledger = fresh_ledger();
    ledger.next_seq = u64::MAX - 1;
    let last = ledger.open().expect("the last unspent sequence number");
    assert_eq!(ledger.open(), Err(PredictionError::Exhausted));
    // The refused open changed nothing: the last minted id is still the live one and is
    // still consumable once.
    assert!(ledger.is_live(last));
    assert_eq!(ledger.next_seq, u64::MAX);
    assert_eq!(ledger.consume(last), Ok(()));
    assert_eq!(ledger.open(), Err(PredictionError::Exhausted));
}

#[test]
fn prediction_errors_display_and_convert_with_the_same_text() {
    for (error, text) in [
        (
            PredictionError::Stale,
            "stale prediction id: no matching live prediction in the ledger",
        ),
        (
            PredictionError::Exhausted,
            "prediction ledger exhausted: no unused prediction id remains",
        ),
    ] {
        assert_eq!(error.to_string(), text);
        let source: &dyn std::error::Error = &error;
        assert!(source.source().is_none());
        match InferenceError::from(error) {
            InferenceError::Inference(message) => assert_eq!(message, text),
            other => panic!("{error:?} converted to {other:?}"),
        }
    }
}

#[test]
fn constructors_set_every_field() {
    let mut ledger = fresh_ledger();
    let id = ledger.open().expect("an unspent sequence number");

    let accepted = AcceptedToken::new(3, id);
    assert_eq!((accepted.final_id, accepted.prediction), (3, id));
    let candidate = SelectionCandidate::new(4, id);
    assert_eq!((candidate.candidate_id, candidate.prediction), (4, id));
    let stamp = StepStamp::new(5, Some(id));
    assert_eq!((stamp.evaluated_len, stamp.prediction), (5, Some(id)));
    let top = vec![TopLogprob {
        token_id: 6,
        logprob: -0.5,
    }];
    let metadata = TokenMetadata::new(id, 7, -0.25, top.clone());
    assert_eq!(metadata.prediction, id);
    assert_eq!(metadata.final_token_id, 7);
    assert_eq!(metadata.final_logprob, -0.25);
    assert_eq!(metadata.top, top);

    assert_eq!(MetadataRequest::disabled().top_logprobs, None);
    assert_eq!(MetadataRequest::with_top_logprobs(0).top_logprobs, Some(0));
    assert_eq!(MetadataRequest::with_top_logprobs(5).top_logprobs, Some(5));

    assert_eq!(
        ExecutionCapabilities::none(),
        ExecutionCapabilities::default()
    );
    let each = [
        ExecutionCapabilities::none().with_grammar(true),
        ExecutionCapabilities::none().with_logprobs(true),
        ExecutionCapabilities::none().with_stop_strings(true),
        ExecutionCapabilities::none().with_reasoning_budget(true),
    ];
    let flags =
        |c: &ExecutionCapabilities| [c.grammar, c.logprobs, c.stop_strings, c.reasoning_budget];
    for (index, caps) in each.iter().enumerate() {
        let mut expected = [false; 4];
        expected[index] = true;
        assert_eq!(
            flags(caps),
            expected,
            "builder {index} set the wrong control"
        );
        assert_eq!(
            flags(
                &caps
                    .with_grammar(false)
                    .with_logprobs(false)
                    .with_stop_strings(false)
                    .with_reasoning_budget(false)
            ),
            [false; 4],
            "builder {index} cannot be withdrawn"
        );
    }

    let config = GenerateConfig::default();
    let history = [1u32, 2];
    let plain = SelectionRequest::new(&config, &history);
    assert!(std::ptr::eq(plain.config, &config));
    assert_eq!(plain.history, &history);
    assert!(plain.grammar_mask.is_none());
    let mask = |logits: &mut [f32]| -> Result<(), InferenceError> {
        logits[0] = f32::NEG_INFINITY;
        Ok(())
    };
    let masked = plain.with_grammar_mask(&mask);
    let mut logits = [0.0f32; 2];
    (masked.grammar_mask.expect("the mask was set"))(&mut logits).expect("the mask applies");
    assert_eq!(logits[0], f32::NEG_INFINITY);
    assert_eq!(masked.history, &history);
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
// every remaining public item in this module through the trait's
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
        let prediction = self.ledger.open().expect("an unspent sequence number");
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
        ledger: fresh_ledger(),
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
        ledger: fresh_ledger(),
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
