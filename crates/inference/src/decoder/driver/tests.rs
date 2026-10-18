use super::*;
use crate::decoder::{
    PredictionId, PredictionLedger, SelectionCandidate, StepStamp, TokenMetadata,
};
use crate::grammar::GrammarSpec;
use std::sync::Arc;

#[test]
fn interrupted_prefill_uses_real_cancel_and_finishes_poisoned_once() {
    struct InterruptedSession {
        caps: ExecutionCapabilities,
        finishes: Vec<FinishDisposition>,
    }

    impl DecoderSession for InterruptedSession {
        fn capabilities(&self) -> &ExecutionCapabilities {
            &self.caps
        }

        fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
            panic!("driver must use interruptible prefill")
        }

        fn prefill_interruptible(
            &mut self,
            cancel: &dyn Cancellation,
        ) -> Result<Option<StepStamp>, InferenceError> {
            assert!(cancel.is_cancelled());
            Ok(None)
        }

        fn decode(
            &mut self,
            _accepted: &AcceptedToken,
            _cancel: &dyn Cancellation,
        ) -> Result<StepStamp, InferenceError> {
            panic!("cancelled prefill must not decode")
        }

        fn select(
            &mut self,
            _request: &SelectionRequest<'_>,
        ) -> Result<SelectOutcome, InferenceError> {
            panic!("cancelled prefill must not select")
        }

        fn metadata(
            &mut self,
            _prediction: PredictionId,
            _final_token: u32,
            _request: &MetadataRequest,
        ) -> Result<TokenMetadata, InferenceError> {
            panic!("cancelled prefill must not score")
        }

        fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError> {
            self.finishes.push(disposition);
            Ok(())
        }
    }

    let mut session = InterruptedSession {
        caps: ExecutionCapabilities::default(),
        finishes: Vec::new(),
    };
    let polls = std::cell::Cell::new(0);
    let cancel = || {
        let poll = polls.get();
        polls.set(poll + 1);
        poll > 0
    };
    let mut text = String::new();
    let mut offsets = Vec::new();
    let result = run(
        &mut session,
        &GenerateConfig::default(),
        None,
        &[2, 3],
        999,
        false,
        &cancel,
        |_| panic!("cancelled prefill must not publish"),
        |_| panic!("cancelled prefill must not detokenize"),
        &mut text,
        &mut offsets,
        |_, _| panic!("cancelled prefill must not emit"),
        || panic!("cancelled prefill must not report completed prefill"),
        || panic!("cancelled prefill must not flush"),
    )
    .unwrap();
    assert_eq!(result.stop_reason, StopReason::Interrupt);
    assert!(result.generated_ids.is_empty());
    assert_eq!(session.finishes, [FinishDisposition::Poisoned]);
    assert_eq!(polls.get(), 2);
}

/// A one-token grammar (`root ::= "a"`) over a one-token vocabulary -- just enough for
/// `gen_cfg.grammar` to be `Some`, which is all [`check_capabilities`] inspects. Neither
/// test below reaches masking: `FakeSession::prefill` errors out before `select` is ever
/// called, so the grammar is never actually run.
fn a_trivial_grammar() -> Arc<GrammarEngine> {
    Arc::new(
        GrammarEngine::new(
            &GrammarSpec::Gbnf("root ::= \"a\"\n".into()),
            vec![b"a".to_vec()],
        )
        .expect("trivial one-token grammar must compile"),
    )
}

/// Declares a fixed [`ExecutionCapabilities`] and otherwise never runs: every method past
/// `capabilities`/`prefill` panics if reached, so a test that gets past
/// [`check_capabilities`] fails at `prefill`'s distinct sentinel error rather than at a
/// silent no-op -- the two refusal points cannot be confused with each other.
struct FakeSession {
    caps: ExecutionCapabilities,
}

impl DecoderSession for FakeSession {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &self.caps
    }

    fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        Err(InferenceError::Inference(
            "FakeSession::prefill reached -- check_capabilities let this request through".into(),
        ))
    }

    fn decode(
        &mut self,
        _accepted: &AcceptedToken,
        _cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        unreachable!("FakeSession::prefill always errors before decode is reached")
    }

    fn select(&mut self, _request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        unreachable!("FakeSession::prefill always errors before select is reached")
    }

    fn metadata(
        &mut self,
        _prediction: PredictionId,
        _final_token: u32,
        _request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        unreachable!("FakeSession::prefill always errors before metadata is reached")
    }

    fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
        Ok(())
    }
}

fn run_with_capabilities(caps: ExecutionCapabilities) -> Result<DriverResult, InferenceError> {
    let mut session = FakeSession { caps };
    let gen_cfg = GenerateConfig {
        grammar: Some(a_trivial_grammar()),
        ..Default::default()
    };
    let cancel = || false;
    let mut text = String::new();
    let mut offsets = Vec::new();
    run(
        &mut session,
        &gen_cfg,
        None,
        &[0u32],
        999,
        false,
        &cancel,
        |_generated_len| {},
        |_next_id| String::new(),
        &mut text,
        &mut offsets,
        |_delta, _id| true,
        || {},
        String::new,
    )
}

// -----------------------------------------------------------------
// Rework item 2: an undeclared-but-requested control is a hard
// `InferenceError`, in every build -- not a `debug_assert!` a release
// binary silently drops. A session declaring `grammar: false` while
// `gen_cfg.grammar` is set must be refused before any session method
// past `capabilities` is ever called.
// -----------------------------------------------------------------

#[test]
fn request_buffers_reserve_history() {
    for (reasoning_budget, max_new_tokens) in [(None, 0), (None, 16), (Some(7), 16)] {
        let config = GenerateConfig {
            reasoning_budget,
            enable_thinking: reasoning_budget.is_some(),
            max_new_tokens,
            ..Default::default()
        };
        let prompt = [1, 2, 3];
        let cap = decode_cap(config.effective_reasoning_budget(), max_new_tokens);
        let mut history = prompt_history(&prompt, &config);
        assert_eq!(history, prompt);
        assert!(history.capacity() >= prompt.len() + cap);
        let pointer = history.as_ptr();
        let capacity = history.capacity();
        for _ in 0..cap {
            history.push(4);
            assert_eq!(history.capacity(), capacity);
            assert_eq!(history.as_ptr(), pointer);
        }
    }
}

#[test]
fn request_buffers_reuse_driver_delta() {
    let mut session = ScriptedSession {
        caps: ExecutionCapabilities::default(),
        ledger: PredictionLedger::new(),
        script: vec![TEXT_TOKEN],
        selects: 0,
    };
    let config = GenerateConfig {
        max_new_tokens: 8,
        ..Default::default()
    };
    let mut text = String::new();
    let mut offsets = Vec::new();
    let mut calls = 0;
    let mut capacity = 0;
    let mut pointer = std::ptr::null();
    let result = run_into(
        &mut session,
        &config,
        None,
        &[0],
        SCRIPTED_EOS,
        true,
        &|| false,
        |_| {},
        |_, delta| {
            assert!(delta.is_empty());
            if calls == 0 {
                delta.reserve(128);
                capacity = delta.capacity();
                pointer = delta.as_ptr();
            } else {
                assert_eq!(delta.capacity(), capacity);
                assert_eq!(delta.as_ptr(), pointer);
            }
            delta.push('a');
            calls += 1;
        },
        &mut text,
        &mut offsets,
        |delta, _| {
            assert_eq!(delta, "a");
            true
        },
        || {},
        String::new,
    )
    .unwrap();
    assert_eq!(calls, config.max_new_tokens);
    assert_eq!(result.generated_ids.len(), calls);
    assert_eq!(text, "a".repeat(calls));
}

#[test]
fn undeclared_grammar_capability_is_a_hard_error() {
    let result = run_with_capabilities(ExecutionCapabilities {
        grammar: false,
        ..ExecutionCapabilities::default()
    });
    match result {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("grammar"),
                "refusal message should name the missing capability, got: {msg}"
            );
        }
        Err(other_err) => panic!(
            "a session declaring grammar: false with gen_cfg.grammar set must be refused \
                 with InvalidInput before any session method past capabilities() runs; got a \
                 different error instead: {other_err:?}"
        ),
        Ok(_) => panic!(
            "a session declaring grammar: false with gen_cfg.grammar set must be refused; \
                 got Ok(_) instead"
        ),
    }
}

/// Passing control for the test above: the same request, against a session that DOES
/// declare grammar support, must get past `check_capabilities` -- proven by reaching
/// `FakeSession::prefill`'s distinct sentinel error rather than the capability refusal.
/// Without this control, the test above could pass for the wrong reason (e.g. a
/// `check_capabilities` that always refuses regardless of `caps`).
#[test]
fn declared_grammar_capability_passes_the_check() {
    let result = run_with_capabilities(ExecutionCapabilities {
        grammar: true,
        ..ExecutionCapabilities::default()
    });
    match result {
        Err(InferenceError::Inference(msg)) => {
            assert!(
                msg.contains("FakeSession::prefill reached"),
                "a session declaring grammar: true must get past check_capabilities and \
                     reach prefill; got a different error instead: {msg}"
            );
        }
        Err(other_err) => panic!(
            "expected FakeSession::prefill's sentinel error (proving check_capabilities let \
                 the request through); got a different error instead: {other_err:?}"
        ),
        Ok(_) => panic!(
            "expected FakeSession::prefill's sentinel error; got Ok(_) instead -- \
                 FakeSession::prefill should be unreachable-if-not-erroring"
        ),
    }
}

// -----------------------------------------------------------------
// `on_push` vs `metadata` ordering, with `gen_cfg.logprobs` set. `on_push`
// must fire for a token strictly before that token's `metadata` call (the
// session-scored logprob lookup `DecodePolicy::transition_inner`'s
// `record_metadata` callback routes to), for every generated token -- the
// prefill-derived first token included, not just the decode-loop tokens.
// -----------------------------------------------------------------

/// Mutation sensitivity: `run`'s `on_push` call sites (step 0, and the push
/// closure passed to `transition_with_metadata`) both fire before the metadata
/// call for that same token; a change that fires `on_push` from `decode_delta`
/// instead (the pre-fix design, which needed a caller-side counter because
/// `decode_delta` receives only a token id, never the push-time length) would
/// fire it *after* `metadata` at every step, since `transition_inner` calls
/// `record_metadata` before `decode_delta` in its fixed order -- see that
/// method's own doc comment. That reorders every pair in `events` to
/// `[Metadata, Push(n), Metadata, Push(n+1), ...]`, and the `assert_eq!` below
/// fails on the very first pair.
#[test]
fn on_push_fires_before_metadata_for_every_token_with_logprobs_enabled() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Event {
        Push(usize),
        Metadata,
    }

    /// Always accepts one fixed candidate token (never EOS, never a configured
    /// stop token), drives `PredictionLedger` correctly through select/decode,
    /// and records every `metadata` call into `events` -- enough to run a
    /// `gen_cfg.logprobs`-enabled request through several full decode-loop
    /// iterations and observe the `on_push`/`metadata` ordering for more than
    /// just the prefill-derived first token.
    struct RecordingSession {
        caps: ExecutionCapabilities,
        ledger: PredictionLedger,
        events: std::rc::Rc<std::cell::RefCell<Vec<Event>>>,
    }

    const CANDIDATE_ID: u32 = 7;

    impl DecoderSession for RecordingSession {
        fn capabilities(&self) -> &ExecutionCapabilities {
            &self.caps
        }

        fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
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
            self.ledger.consume(accepted.prediction)?;
            Ok(StepStamp {
                evaluated_len: 2,
                prediction: None,
            })
        }

        fn select(
            &mut self,
            _request: &SelectionRequest<'_>,
        ) -> Result<SelectOutcome, InferenceError> {
            Ok(SelectOutcome::Candidate(SelectionCandidate {
                candidate_id: CANDIDATE_ID,
                prediction: self.ledger.open(),
            }))
        }

        fn metadata(
            &mut self,
            prediction: PredictionId,
            final_token: u32,
            _request: &MetadataRequest,
        ) -> Result<TokenMetadata, InferenceError> {
            self.events.borrow_mut().push(Event::Metadata);
            Ok(TokenMetadata {
                prediction,
                final_token_id: final_token,
                final_logprob: -0.1,
                top: Vec::new(),
            })
        }

        fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::<Event>::new()));
    let mut session = RecordingSession {
        caps: ExecutionCapabilities {
            logprobs: true,
            ..ExecutionCapabilities::default()
        },
        ledger: PredictionLedger::new(),
        events: events.clone(),
    };
    // `logprobs: Some(_)` is what makes `record_metadata` a real, non-skipped
    // call at every step (see `DecodePolicy::transition_inner`'s doc comment) --
    // the ordering this test exists to pin has no observable effect otherwise.
    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        logprobs: Some(0),
        ..Default::default()
    };
    let cancel = || false;
    let mut text = String::new();
    let mut offsets = Vec::new();
    let events_for_push = events.clone();

    let result = run(
        &mut session,
        &gen_cfg,
        None,
        &[0u32],
        999, // eos_token_id; CANDIDATE_ID (7) and the default stop_token_ids
        // (QWEN_CHAT_IM_END_TOKEN_ID, 248_046) never match it, so the loop
        // runs to gen_cfg.max_new_tokens rather than stopping early.
        false,
        &cancel,
        move |generated_len| {
            events_for_push
                .borrow_mut()
                .push(Event::Push(generated_len));
        },
        |_next_id| String::new(),
        &mut text,
        &mut offsets,
        |_delta, _id| true,
        || {},
        String::new,
    )
    .expect("a fixed non-EOS candidate with no grammar/stop-strings must run to the cap");

    assert_eq!(
        result.generated_ids,
        vec![CANDIDATE_ID; 3],
        "the fixed candidate must be accepted on every step"
    );

    let expected: Vec<Event> = (1..=3usize)
        .flat_map(|index| [Event::Push(index), Event::Metadata])
        .collect();
    assert_eq!(
        events.borrow().clone(),
        expected,
        "on_push must fire for token n strictly before that token's metadata call, for \
             every one of the 3 generated tokens (the prefill-derived first token included)"
    );
}

// -----------------------------------------------------------------
// A caller rejecting the natural-end tail flush must end the request
// with `Interrupt`, exactly as a mid-decode rejection does.
// -----------------------------------------------------------------

const TEXT_TOKEN: u32 = 7;
const SCRIPTED_EOS: u32 = 999;

/// Returns `script[i]` from the i-th `select` call (the last entry repeats), driving
/// `PredictionLedger` correctly through select/decode.
struct ScriptedSession {
    caps: ExecutionCapabilities,
    ledger: PredictionLedger,
    script: Vec<u32>,
    selects: usize,
}

impl DecoderSession for ScriptedSession {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &self.caps
    }

    fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
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
        self.ledger.consume(accepted.prediction)?;
        Ok(StepStamp {
            evaluated_len: 2,
            prediction: None,
        })
    }

    fn select(&mut self, _request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        let index = self.selects.min(self.script.len() - 1);
        self.selects += 1;
        Ok(SelectOutcome::Candidate(SelectionCandidate {
            candidate_id: self.script[index],
            prediction: self.ledger.open(),
        }))
    }

    fn metadata(
        &mut self,
        _prediction: PredictionId,
        _final_token: u32,
        _request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        unreachable!("gen_cfg.logprobs is None, so metadata is never requested")
    }

    fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
        Ok(())
    }
}

/// Streams `script` with `stop_strings = ["ZZ"]`, every token decoding to "a". The sink
/// returns false only on call number `reject_call` (1-based). Returns the result and
/// every string the sink was handed, in order.
fn run_streaming_script(
    script: Vec<u32>,
    max_new_tokens: usize,
    reject_call: usize,
) -> (DriverResult, Vec<String>) {
    let mut session = ScriptedSession {
        caps: ExecutionCapabilities {
            stop_strings: true,
            ..ExecutionCapabilities::default()
        },
        ledger: PredictionLedger::new(),
        script,
        selects: 0,
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens,
        stop_strings: vec!["ZZ".to_string()],
        ..Default::default()
    };
    let cancel = || false;
    let mut text = String::new();
    let mut offsets = Vec::new();
    let mut emitted: Vec<String> = Vec::new();
    let result = run(
        &mut session,
        &gen_cfg,
        None,
        &[0u32],
        SCRIPTED_EOS,
        true,
        &cancel,
        |_generated_len| {},
        |_next_id| "a".to_string(),
        &mut text,
        &mut offsets,
        |delta, _id| {
            emitted.push(delta.to_string());
            emitted.len() != reject_call
        },
        || {},
        String::new,
    )
    .expect("a scripted non-grammar stream must not error");
    (result, emitted)
}

/// "ZZ" holds back one byte, so the third "a" of a 3-token run reaches the caller only
/// through `finish_stop`: calls 1 and 2 are mid-decode, call 3 is the tail flush.
#[test]
fn rejected_tail_flush_after_length_cap_reports_interrupt() {
    let (control, emitted) = run_streaming_script(vec![TEXT_TOKEN], 3, usize::MAX);
    assert_eq!(
        emitted,
        vec!["a", "a", "a"],
        "fixture shape: 2 loop emits + 1 flush"
    );
    assert_eq!(
        control.stop_reason,
        StopReason::Length,
        "control: all accepted"
    );
    assert!(!control.stopped);

    let (result, emitted) = run_streaming_script(vec![TEXT_TOKEN], 3, 3);
    assert_eq!(emitted.len(), 3, "the rejected call must be the tail flush");
    assert_eq!(
        result.stop_reason,
        StopReason::Interrupt,
        "a sink rejecting the final flush must end the request as Interrupt, like a \
             mid-decode rejection"
    );
    assert!(!result.stopped, "Interrupt always reports stopped: false");
    assert!(!result.confirmed_stop_string_match);
}

/// Same, for a stream that ends on EOS (`stopped: true` before the flush): the rejected
/// flush must still report `Interrupt` and clear `stopped`.
#[test]
fn rejected_tail_flush_after_eos_reports_interrupt() {
    let script = vec![TEXT_TOKEN, TEXT_TOKEN, SCRIPTED_EOS];
    let (control, emitted) = run_streaming_script(script.clone(), 8, usize::MAX);
    assert_eq!(
        emitted,
        vec!["a", "a"],
        "fixture shape: 1 loop emit + 1 flush"
    );
    assert_eq!(
        control.stop_reason,
        StopReason::Eos,
        "control: all accepted"
    );
    assert!(control.stopped);

    let (result, emitted) = run_streaming_script(script, 8, 2);
    assert_eq!(emitted.len(), 2, "the rejected call must be the tail flush");
    assert_eq!(result.stop_reason, StopReason::Interrupt);
    assert!(!result.stopped, "Interrupt always reports stopped: false");
}
