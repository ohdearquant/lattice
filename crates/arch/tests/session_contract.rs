//! The decoder session contract as a crate outside `lattice-arch` sees it (ADR-103 Amendment 21
//! item 3): a session built only from the public constructors, holding an `Rc` (so neither
//! `Send` nor `Sync`), and run through `driver::run` over `&mut dyn DecoderSession`.

use lattice_arch::decoder::driver::{DriverResult, run};
use lattice_arch::decoder::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionError, PredictionId, PredictionLedger, SelectOutcome,
    SelectionCandidate, SelectionRequest, StepStamp, TokenMetadata,
};
use lattice_arch::generation::{GenerateConfig, TopLogprob};
use lattice_arch::stop_reason::StopReason;
use lattice_inference_core::InferenceError;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

const EOS: u32 = 99;
const THINK_CLOSE: u32 = 7;

/// What the test reads back after the session has been moved into a driver call or dropped.
#[derive(Default)]
struct Observed {
    calls: Vec<&'static str>,
    finishes: Vec<FinishDisposition>,
    opened: Vec<PredictionId>,
    /// `(candidate, final token)` for every `metadata` call that scored.
    scored: Vec<(u32, u32)>,
    released: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fresh,
    Ready,
    Finished,
}

struct ScriptSession {
    observed: Rc<RefCell<Observed>>,
    caps: ExecutionCapabilities,
    ledger: PredictionLedger,
    prompt_len: usize,
    script: Vec<u32>,
    evaluated: usize,
    phase: Phase,
    candidate: Option<u32>,
    fail_at: Option<&'static str>,
}

impl ScriptSession {
    fn new(prompt_len: usize, script: Vec<u32>) -> Self {
        Self {
            observed: Rc::default(),
            caps: ExecutionCapabilities::none()
                .with_logprobs(true)
                .with_stop_strings(true)
                .with_reasoning_budget(true),
            ledger: PredictionLedger::new().expect("a fresh ledger identity"),
            prompt_len,
            script,
            evaluated: 0,
            phase: Phase::Fresh,
            candidate: None,
            fail_at: None,
        }
    }

    fn enter(&self, call: &'static str, phase: Phase) -> Result<(), InferenceError> {
        self.observed.borrow_mut().calls.push(call);
        if self.phase != phase {
            return Err(InferenceError::InvalidInput(format!(
                "{call} out of order: the session is {:?}",
                self.phase
            )));
        }
        Ok(())
    }

    fn fail(&self, call: &'static str) -> Result<(), InferenceError> {
        if self.fail_at == Some(call) {
            return Err(InferenceError::Inference(format!("{call} failed")));
        }
        Ok(())
    }
}

impl Drop for ScriptSession {
    fn drop(&mut self) {
        self.observed.borrow_mut().released = true;
    }
}

impl DecoderSession for ScriptSession {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &self.caps
    }

    fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        self.enter("prefill", Phase::Fresh)?;
        self.fail("prefill")?;
        self.evaluated = self.prompt_len;
        self.phase = Phase::Ready;
        Ok(StepStamp::new(self.evaluated, None))
    }

    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        _cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        self.enter("decode", Phase::Ready)?;
        self.ledger.consume(accepted.prediction)?;
        self.candidate = None;
        self.fail("decode")?;
        self.evaluated += 1;
        Ok(StepStamp::new(self.evaluated, None))
    }

    fn select(&mut self, _request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        self.enter("select", Phase::Ready)?;
        if self.ledger.has_live() {
            return Err(InferenceError::InvalidInput(
                "select while a prediction is live".into(),
            ));
        }
        self.fail("select")?;
        let candidate = self.script[(self.evaluated - self.prompt_len) % self.script.len()];
        let prediction = self.ledger.open()?;
        self.candidate = Some(candidate);
        self.observed.borrow_mut().opened.push(prediction);
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
        self.enter("metadata", Phase::Ready)?;
        let Some(top_n) = request.top_logprobs else {
            return Err(InferenceError::InvalidInput(
                "metadata capture is disabled in this request".into(),
            ));
        };
        if !self.ledger.is_live(prediction) {
            return Err(PredictionError::Stale.into());
        }
        self.fail("metadata")?;
        let candidate = self.candidate.expect("a live prediction has its candidate");
        self.observed
            .borrow_mut()
            .scored
            .push((candidate, final_token));
        let top = (0..top_n)
            .map(|rank| TopLogprob {
                token_id: rank as u32,
                logprob: -1.0 - rank as f32,
            })
            .collect();
        Ok(TokenMetadata::new(prediction, final_token, -0.5, top))
    }

    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError> {
        self.observed.borrow_mut().calls.push("finish");
        if self.phase == Phase::Finished {
            return Err(InferenceError::InvalidInput("finish after finish".into()));
        }
        self.ledger.invalidate();
        self.candidate = None;
        self.phase = Phase::Finished;
        self.observed.borrow_mut().finishes.push(disposition);
        Ok(())
    }
}

fn config(max_new_tokens: usize) -> GenerateConfig {
    let mut config = GenerateConfig::default();
    config.max_new_tokens = max_new_tokens;
    config
}

fn drive(
    session: &mut dyn DecoderSession,
    config: &GenerateConfig,
    prompt: &[u32],
    think_close_id: Option<u32>,
    boundary: Option<Arc<[bool]>>,
    cancel: &dyn Cancellation,
) -> Result<DriverResult, InferenceError> {
    run(
        session,
        config,
        think_close_id,
        boundary,
        prompt,
        EOS,
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

fn count(observed: &Observed, call: &str) -> usize {
    observed.calls.iter().filter(|&&c| c == call).count()
}

fn stale_text() -> String {
    InferenceError::from(PredictionError::Stale).to_string()
}

fn select(session: &mut dyn DecoderSession, config: &GenerateConfig) -> SelectionCandidate {
    match session
        .select(&SelectionRequest::new(config, &[]))
        .expect("select")
    {
        SelectOutcome::Candidate(candidate) => candidate,
        _ => panic!("no grammar is set, so select yields a candidate"),
    }
}

#[test]
fn an_outside_session_runs_through_the_driver_and_refuses_every_call_after_finish() {
    let mut session = ScriptSession::new(2, vec![5, 6, 4]);
    let observed = Rc::clone(&session.observed);
    let mut config = config(3);
    config.logprobs = Some(2);
    let result = drive(&mut session, &config, &[1, 2], None, None, &|| false)
        .expect("a three-token request");

    assert_eq!(result.generated_ids, [5, 6, 4]);
    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(result.token_logprobs.len(), 3);
    {
        let observed = observed.borrow();
        assert_eq!(observed.calls.first(), Some(&"prefill"));
        assert_eq!(observed.calls.last(), Some(&"finish"));
        assert_eq!(observed.finishes, [FinishDisposition::Reusable]);
        assert_eq!(result.trace.opened, count(&observed, "select"));
        assert_eq!(result.trace.consumed, count(&observed, "decode"));
        assert_eq!(observed.scored, [(5, 5), (6, 6), (4, 4)]);
    }

    let evaluated = session.evaluated;
    let last = *observed
        .borrow()
        .opened
        .last()
        .expect("an opened prediction");
    let contract: &mut dyn DecoderSession = &mut session;
    assert!(contract.prefill(&|| false).is_err(), "prefill after finish");
    assert!(
        contract
            .select(&SelectionRequest::new(&config, &[]))
            .is_err(),
        "select after finish"
    );
    assert!(
        contract
            .decode(&AcceptedToken::new(4, last), &|| false)
            .is_err(),
        "decode after finish"
    );
    assert!(
        contract
            .metadata(last, 4, &MetadataRequest::with_top_logprobs(0))
            .is_err(),
        "metadata after finish"
    );
    assert!(
        contract.finish(FinishDisposition::Reusable).is_err(),
        "a second finish"
    );
    assert_eq!(
        session.evaluated, evaluated,
        "a refused call moved the session"
    );
    assert_eq!(observed.borrow().finishes.len(), 1);
    assert_eq!(observed.borrow().scored.len(), 3);
}

#[test]
fn stale_foreign_and_consumed_ids_are_refused_without_a_state_change() {
    let config = config(4);
    let mut a = ScriptSession::new(1, vec![3]);
    let mut b = ScriptSession::new(1, vec![8]);
    a.prefill(&|| false).expect("prefill a");
    b.prefill(&|| false).expect("prefill b");

    let first = select(&mut a, &config);
    a.decode(
        &AcceptedToken::new(first.candidate_id, first.prediction),
        &|| false,
    )
    .expect("the first decode consumes the live prediction");
    let after_first = a.evaluated;

    let consumed = a.decode(
        &AcceptedToken::new(first.candidate_id, first.prediction),
        &|| false,
    );
    assert_eq!(consumed.err().map(|e| e.to_string()), Some(stale_text()));
    assert_eq!(a.evaluated, after_first, "a consumed id moved the session");

    let second = select(&mut a, &config);
    let foreign = select(&mut b, &config);
    assert!(
        a.decode(
            &AcceptedToken::new(foreign.candidate_id, foreign.prediction),
            &|| false,
        )
        .is_err(),
        "an id minted by another session's ledger"
    );
    assert!(
        a.metadata(
            foreign.prediction,
            8,
            &MetadataRequest::with_top_logprobs(0)
        )
        .is_err(),
        "metadata for another session's id"
    );
    let stale = a.metadata(first.prediction, 3, &MetadataRequest::with_top_logprobs(0));
    assert_eq!(stale.err().map(|e| e.to_string()), Some(stale_text()));
    assert_eq!(a.evaluated, after_first);
    assert!(a.observed.borrow().scored.is_empty());

    // Control: the live prediction survived every refusal and is still accepted, once.
    a.decode(
        &AcceptedToken::new(second.candidate_id, second.prediction),
        &|| false,
    )
    .expect("the live prediction decodes");
    assert_eq!(a.evaluated, after_first + 1);
    // And b's own prediction was never consumed by a's refusals.
    b.decode(
        &AcceptedToken::new(foreign.candidate_id, foreign.prediction),
        &|| false,
    )
    .expect("b's live prediction decodes");
}

#[test]
fn preflight_refusals_call_nothing_and_a_zero_budget_finishes_without_prefill() {
    let mut session = ScriptSession::new(1, vec![3]);
    let error = drive(&mut session, &config(2), &[], None, None, &|| false)
        .err()
        .expect("an empty prompt is refused");
    assert!(
        matches!(&error, InferenceError::InvalidInput(message) if message.contains("non-empty prompt")),
        "{error:?}"
    );
    assert!(session.observed.borrow().calls.is_empty());

    let undeclared = {
        let mut logprobs = config(2);
        logprobs.logprobs = Some(0);
        let mut stops = config(2);
        stops.stop_strings = vec!["stop".into()];
        let mut reasoning = config(2);
        reasoning.reasoning_budget = Some(1);
        [
            ("logprobs", logprobs),
            ("stop_strings", stops),
            ("reasoning_budget", reasoning),
        ]
    };
    for (control, request) in &undeclared {
        let mut session = ScriptSession::new(1, vec![3]);
        session.caps = ExecutionCapabilities::none();
        let error = drive(&mut session, request, &[1], None, None, &|| false)
            .err()
            .expect("an undeclared control is refused");
        assert!(
            matches!(&error, InferenceError::InvalidInput(message) if message.contains(control)),
            "{control}: {error:?}"
        );
        assert!(session.observed.borrow().calls.is_empty(), "{control}");
    }

    let mut session = ScriptSession::new(1, vec![3]);
    let result = drive(&mut session, &config(0), &[1], None, None, &|| false)
        .expect("a zero budget is not an error");
    assert!(result.generated_ids.is_empty());
    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(session.observed.borrow().calls, ["finish"]);
    assert_eq!(
        session.observed.borrow().finishes,
        [FinishDisposition::Reusable]
    );

    // Control: the same session type, a one-token prompt and a declared-free request reach
    // prefill.
    let mut control = ScriptSession::new(1, vec![3]);
    control.caps = ExecutionCapabilities::none();
    let result =
        drive(&mut control, &config(1), &[1], None, None, &|| false).expect("a one-token request");
    assert_eq!(result.generated_ids, [3]);
    assert_eq!(control.observed.borrow().calls.first(), Some(&"prefill"));
}

#[test]
fn metadata_scores_the_final_token_under_the_candidates_prediction() {
    // A reasoning budget of one with a line-boundary first token makes the decode policy
    // replace the second candidate with the think-close token, so the final token differs
    // from the candidate the session sampled.
    let mut session = ScriptSession::new(1, vec![1]);
    let observed = Rc::clone(&session.observed);
    let mut request = config(2);
    request.enable_thinking = true;
    request.reasoning_budget = Some(1);
    request.logprobs = Some(0);
    let mut boundary = vec![false; 16];
    boundary[1] = true;
    let result = drive(
        &mut session,
        &request,
        &[1],
        Some(THINK_CLOSE),
        Some(boundary.into()),
        &|| false,
    )
    .expect("a budgeted request");
    assert_eq!(result.generated_ids[..2], [1, THINK_CLOSE]);
    assert_eq!(observed.borrow().scored[..2], [(1, 1), (1, THINK_CLOSE)]);

    // Direct: the returned record names the prediction and the final token it was asked
    // about, and a disabled request is refused without retiring the live prediction.
    let mut session = ScriptSession::new(1, vec![1]);
    session.prefill(&|| false).expect("prefill");
    let candidate = select(&mut session, &config(1));
    let disabled = session.metadata(candidate.prediction, 2, &MetadataRequest::disabled());
    assert!(
        matches!(&disabled, Err(InferenceError::InvalidInput(message)) if message.contains("disabled")),
        "{disabled:?}"
    );
    let scored = session
        .metadata(
            candidate.prediction,
            2,
            &MetadataRequest::with_top_logprobs(1),
        )
        .expect("the prediction is still live");
    assert_eq!(scored.prediction, candidate.prediction);
    assert_eq!(scored.final_token_id, 2);
    assert_eq!(scored.top.len(), 1);
}

#[test]
fn cancellation_and_errors_finish_once_and_drop_releases_the_session() {
    for cancel_at in [0, 1] {
        let mut session = ScriptSession::new(1, vec![3]);
        let observed = Rc::clone(&session.observed);
        let polls = Cell::new(0);
        let cancel = || {
            let at = polls.get();
            polls.set(at + 1);
            at == cancel_at
        };
        let result = drive(&mut session, &config(3), &[1], None, None, &cancel)
            .expect("an interrupted request still returns its result");
        assert_eq!(result.stop_reason, StopReason::Interrupt, "{cancel_at}");
        let expected = if cancel_at == 0 {
            FinishDisposition::Reusable
        } else {
            FinishDisposition::Poisoned
        };
        assert_eq!(observed.borrow().finishes, [expected], "{cancel_at}");
        drop(session);
        assert!(observed.borrow().released);
    }

    for failure in ["prefill", "select", "decode", "metadata"] {
        let mut session = ScriptSession::new(1, vec![3]);
        session.fail_at = Some(failure);
        let observed = Rc::clone(&session.observed);
        let mut request = config(3);
        request.logprobs = Some(0);
        let error = drive(&mut session, &request, &[1], None, None, &|| false)
            .err()
            .expect("a failing session call fails the request");
        assert!(
            matches!(&error, InferenceError::Inference(message) if *message == format!("{failure} failed")),
            "{failure}: {error:?}"
        );
        assert_eq!(
            observed.borrow().finishes,
            [FinishDisposition::Poisoned],
            "{failure}"
        );
        drop(session);
        assert!(observed.borrow().released, "{failure}");
    }

    // A session dropped without `finish` releases its resources too.
    let mut session = ScriptSession::new(1, vec![3]);
    let observed = Rc::clone(&session.observed);
    session.prefill(&|| false).expect("prefill");
    drop(session);
    let observed = observed.borrow();
    assert!(observed.released);
    assert!(observed.finishes.is_empty());
}
