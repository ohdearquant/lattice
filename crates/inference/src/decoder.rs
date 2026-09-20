//! Crate-private decoder session vocabulary (ADR-090 D1/D2).
//!
//! This module lands the type surface D1's illustrative `DecoderSession` trait sketches,
//! plus the ledger mechanism that makes "a stale prediction id is rejected" a state fact
//! the type system enforces rather than a comment a future caller has to remember. Nothing
//! here is wired to a driver or a concrete session yet: there is no autoregressive loop, no
//! `QwenCpuSession`/`GemmaCpuSession`/`QwenMetalSession`, and no caller anywhere else in the
//! crate. That lands in a later row; see the ADR for the full lifecycle this vocabulary will
//! eventually serve.
//!
//! Everything in this module is `pub(crate)`.

use crate::error::InferenceError;
use crate::generation::{GenerateConfig, TopLogprob};
use crate::grammar::GrammarEngine;
use std::sync::atomic::{AtomicU64, Ordering};

/// First concrete [`DecoderSession`] implementation (ADR-090 row B): the CPU dense
/// Qwen3.5 entry point wrapped over this module's typed vocabulary.
pub(crate) mod qwen_cpu;

// ---------------------------------------------------------------------------
// PredictionId / PredictionLedger
// ---------------------------------------------------------------------------

/// Opaque identity for one open prediction: a sampled candidate awaiting policy
/// finalization and, if accepted, consumption by the next decode step.
///
/// `Copy` and comparable, but mintable only by [`PredictionLedger::open`] -- there is no
/// `From<u64>`, no public constructor, and both fields are private to this module (visible
/// to this module's own `tests` submodule, never to the rest of the crate). A caller
/// holding a `PredictionId` therefore has proof it went through the ledger that owns its
/// validity, not a bare integer it could have fabricated.
///
/// **`ledger` is not the `epoch` field this type used to carry.** That one was bumped on
/// every reset and read by nothing: `seq` is monotonic within a ledger and never reused, so
/// it already made ids unique across resets, which is pinned by a test below. `ledger`
/// answers a different question the sequence cannot: WHICH ledger minted this. Without it
/// two freshly constructed ledgers both mint `seq = 0`, and since `is_live`/`consume`
/// compare ids by value, session A's live prediction would be accepted and consumed by
/// session B -- the exact cross-session confusion this type exists to make impossible. The
/// id is assigned once per ledger from a process-wide counter, not once per token, so the
/// decode path pays no atomic per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PredictionId {
    ledger: u64,
    seq: u64,
}

/// Source of ledger identities. Relaxed ordering is sufficient: the only property required
/// is that no two `fetch_add` results are equal, which `fetch_add` guarantees on its own
/// without any happens-before relationship between ledgers.
static NEXT_LEDGER_ID: AtomicU64 = AtomicU64::new(0);

/// The mechanism that makes "a stale prediction id is rejected" a state fact rather than a
/// comment. Owns its own identity, a sequence counter that is monotonic within that
/// identity and never reused, and at most one live prediction.
///
/// D2: "There is one current prediction per evaluated prefix." Selection consumes no
/// token; eligibility survives candidate selection and final-token metadata reads, and
/// ends at consumption, cancellation, failure, finish, or another prefill/reset.
///
/// No `Default`: constructing a ledger takes an identity from a process-wide counter, and a
/// `Default` impl would hide that side effect behind a call that reads as inert.
#[derive(Debug)]
pub(crate) struct PredictionLedger {
    id: u64,
    next_seq: u64,
    live: Option<PredictionId>,
}

impl PredictionLedger {
    /// A ledger with a fresh identity no other ledger in this process shares.
    pub(crate) fn new() -> Self {
        Self {
            id: NEXT_LEDGER_ID.fetch_add(1, Ordering::Relaxed),
            next_seq: 0,
            live: None,
        }
    }

    /// Mint a new prediction id and make it the ledger's one live prediction.
    ///
    /// **Decision (documented and tested below): supersede, not refuse.** `open` cannot
    /// report failure -- its signature returns `PredictionId`, not a `Result` -- so a
    /// caller that opens a second prediction while the first is still live cannot be told
    /// no. The ledger instead enforces its own "at most one live prediction" invariant by
    /// retiring the old id before minting the new one: the superseded id fails the same
    /// `is_live` check a cancelled, consumed, or reset-invalidated id would fail, through
    /// the same mechanism, rather than inventing a second way for a `PredictionId` to go
    /// bad. A driver that opens twice per step has a bug either way; this choice makes
    /// that bug surface as an ordinary stale-id rejection on whichever id the driver kept,
    /// instead of two simultaneously "live" ids racing to be consumed.
    pub(crate) fn open(&mut self) -> PredictionId {
        let id = PredictionId {
            ledger: self.id,
            seq: self.next_seq,
        };
        self.next_seq += 1;
        self.live = Some(id);
        id
    }

    /// Whether `id` is the ledger's current live prediction. This is the eligibility check
    /// a concrete session's `select`/`metadata` operations must consult: a cancelled,
    /// consumed, superseded, or reset-invalidated id is never live again.
    pub(crate) fn is_live(&self, id: PredictionId) -> bool {
        self.live == Some(id)
    }

    /// Another prefill/reset happened: the live prediction, if any, is no longer eligible.
    ///
    /// **Behaviourally identical to [`Self::invalidate`] today, and that is stated rather
    /// than hidden.** The two exist as separate entry points because the call sites are
    /// different -- a prefill is not a cancellation -- and because a reader at either call
    /// site should not have to decide which generic verb to reach for. Nothing currently
    /// distinguishes an id invalidated by a reset from one invalidated by a cancellation.
    /// If a later row needs that distinction, it has to be added and tested deliberately;
    /// it must not be assumed present because the methods have different names.
    pub(crate) fn reset(&mut self) {
        self.live = None;
    }

    /// Cancellation, failure, or finish: the live prediction, if any, is no longer
    /// eligible. See [`Self::reset`] on why these are two methods with one behaviour.
    pub(crate) fn invalidate(&mut self) {
        self.live = None;
    }

    /// Consume `id`: the live prediction must currently be `id`, and after this call it no
    /// longer is. A second `consume` of the same id -- whether the original caller retried
    /// it or a different caller captured a copy -- finds no matching live prediction and is
    /// rejected the same way any other stale id would be.
    pub(crate) fn consume(&mut self, id: PredictionId) -> Result<(), PredictionError> {
        if self.live == Some(id) {
            self.live = None;
            Ok(())
        } else {
            Err(PredictionError::Stale)
        }
    }
}

/// Crate-private failure taxonomy for the prediction ledger. See
/// `impl From<PredictionError> for InferenceError` for how this maps into the crate's
/// public error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PredictionError {
    /// `consume` was called with an id that is not the ledger's current live prediction:
    /// already consumed, superseded, cancelled, or invalidated by a prefill/reset.
    Stale,
}

/// Maps into the crate's general execution-failure bucket. `InferenceError` is
/// `#[non_exhaustive]`; this row does not add a new public variant. A stale prediction id
/// is a session-internal protocol error (the driver misused the ledger), not
/// caller-supplied invalid input, so it is not routed through `InvalidInput`.
impl From<PredictionError> for InferenceError {
    fn from(value: PredictionError) -> Self {
        match value {
            PredictionError::Stale => InferenceError::Inference(
                "stale prediction id: no matching live prediction in the ledger".into(),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// AcceptedToken / SelectionCandidate / SelectionRequest
// ---------------------------------------------------------------------------

/// Binds an accepted final token id to the prediction that produced its candidate: "the
/// final token pending evaluation" in D1's vocabulary. Fields are readable in-crate, but
/// the only way to obtain a `PredictionId` to put in one is [`PredictionLedger::open`], so
/// construction is gated by the ledger even without a dedicated constructor function.
///
/// Not `Clone`, not `Copy` -- but that is a hint, not the enforcement, and the difference
/// matters. D1's `decode(&mut self, accepted: &AcceptedToken, ..)` takes a SHARED borrow, so
/// the type cannot stop a caller from passing the same token twice; and the embedded
/// `PredictionId` is `Copy`, so the identity can be lifted out and reused on its own.
/// Consume-at-most-once is therefore a property of [`PredictionLedger::consume`], which
/// clears the live slot so the second attempt finds nothing and is rejected exactly like any
/// other stale id. The non-`Clone` shape only removes the most casual way to get it wrong.
#[derive(Debug)]
pub(crate) struct AcceptedToken {
    pub(crate) final_id: u32,
    pub(crate) prediction: PredictionId,
}

/// A sampled candidate id and the prediction it belongs to. Sampling alone authorizes
/// neither publication nor consumption (D1); the candidate becomes an [`AcceptedToken`]
/// only after policy finalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SelectionCandidate {
    pub(crate) candidate_id: u32,
    pub(crate) prediction: PredictionId,
}

/// What a concrete session's `select` operation borrows to sample the next candidate.
///
/// RNG ownership stays in the driver/session, never in this request: randomness crosses
/// this boundary as already-drawn values baked into the session's own state, never as a
/// sampler handle passed through the request. That is what lets `select` take
/// `&SelectionRequest` (shared borrow) instead of a mutable one -- the mutation happens
/// through `&mut self` on the session, not through this type. Note for a future row: the
/// legacy `TokenSampler::sample` draws its RNG internally rather than accepting
/// already-drawn values; reconciling that draw schedule with this boundary is row B/C
/// work and is not solved here.
#[derive(Clone, Copy)]
pub(crate) struct SelectionRequest<'a> {
    pub(crate) config: &'a GenerateConfig,
    pub(crate) history: &'a [u32],
    pub(crate) grammar: Option<&'a GrammarEngine>,
}

// ---------------------------------------------------------------------------
// MetadataRequest / TokenMetadata
// ---------------------------------------------------------------------------

/// What the entry profile asked for from the `metadata` operation. Mirrors
/// [`GenerateConfig::logprobs`] exactly: `None` disables metadata capture entirely; `Some(n)`
/// requests the final token's log-probability plus its `n` highest-probability
/// alternatives (`n == 0` is valid: report only the final token's log-probability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetadataRequest {
    pub(crate) top_logprobs: Option<usize>,
}

/// The result of scoring one final token against the prediction that produced its
/// candidate. Identifies both: `prediction` names which ledger entry was scored,
/// `final_token_id` names which token id was scored -- these can differ from the
/// originally sampled candidate id when policy finalization overrides it (D1's
/// policy-final-token role), so a caller cannot recover the final id from the prediction
/// alone.
///
/// `final_logprob` is named for what it is on purpose: it is the *final* token's
/// log-probability under the prediction's pre-advance scoring view, never the candidate's.
/// Reuses [`crate::generation::TopLogprob`] rather than defining a second alternative-token
/// type.
#[derive(Debug, Clone)]
pub(crate) struct TokenMetadata {
    pub(crate) prediction: PredictionId,
    pub(crate) final_token_id: u32,
    pub(crate) final_logprob: f32,
    pub(crate) top: Vec<TopLogprob>,
}

// ---------------------------------------------------------------------------
// StepStamp
// ---------------------------------------------------------------------------

/// Identifies the evaluated input prefix and the current prediction, if one was produced,
/// after a `prefill` or `decode` step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StepStamp {
    pub(crate) evaluated_len: usize,
    pub(crate) prediction: Option<PredictionId>,
}

// ---------------------------------------------------------------------------
// ExecutionCapabilities
// ---------------------------------------------------------------------------

/// Capabilities negotiated per concrete model + backend + entry profile (D3).
///
/// Every field here names an unsupported-control refusal that already exists in this
/// crate today, so this type describes real, currently-varying behavior rather than
/// speculative future switches:
///
/// - `grammar` mirrors `check_grammar_not_set` (`model::qwen35::generation`): the base
///   Qwen CPU `generate()`/`generate_streaming()` wire grammar masking directly, while
///   `generate_q8`, `generate_f16`, `generate_q8_neon` fail closed on a set
///   `GenerateConfig::grammar` (#397).
/// - `logprobs` mirrors `check_logprobs_not_set`: the same base CPU paths and the Metal
///   `generate_streaming()` wire per-step logprob capture directly, while the same
///   standalone wrappers fail closed on a set `GenerateConfig::logprobs` (#585).
/// - `stop_strings` mirrors `check_stop_strings_not_set` (ADR-080 C3, #783): the base CPU
///   and Metal families wire stop-string matching directly, while the standalone CPU
///   wrappers fail closed on a non-empty `GenerateConfig::stop_strings`.
/// - `reasoning_budget` mirrors `check_reasoning_budget_not_set` (ADR-080 C3, #783): the
///   base CPU paths and Metal `generate_streaming()` family wire budget-forcing directly,
///   while the standalone CPU wrappers fail closed on a set `GenerateConfig::reasoning_budget`.
///
/// Left out, deliberately: prefix-cache and speculative-execution capability fields. D3
/// lists "Qwen prefix cache" and "Qwen speculative routes" as their own entry profiles
/// today, but there is no generic, per-session negotiation of them yet -- only Qwen has
/// the methods at all, and nothing in this row (no driver, no concrete session) would read
/// such a field. Naming a field for behavior nothing here can point at would be exactly
/// the guess-shaped field this ADR forbids; that negotiation is row B/C work, once a
/// concrete session and driver exist to consult it. One permissive family-level boolean
/// must not widen a narrower entry point (D1); this type adds no `everything: bool`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ExecutionCapabilities {
    pub(crate) grammar: bool,
    pub(crate) logprobs: bool,
    pub(crate) stop_strings: bool,
    pub(crate) reasoning_budget: bool,
}

// ---------------------------------------------------------------------------
// FinishDisposition
// ---------------------------------------------------------------------------

/// How a session's execution ended, and whether its typed state may be reused for another
/// prefill (D2). Distinct from [`crate::stop_reason::StopReason`], which answers *why*
/// generation stopped (EOS, length, grammar dead end, ...) -- a question orthogonal to
/// whether the session's internal state is still valid to reuse. This row does not
/// duplicate that type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinishDisposition {
    /// Generation completed normally, or a failure occurred before any state mutation:
    /// the session's typed state remains valid and may be reused for another prefill.
    Reusable,
    /// A partial execution mutated session state without completing. The session must be
    /// destroyed or reset through its typed owner -- never made reusable by changing a
    /// common sequence-length integer (D2).
    Poisoned,
}

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

/// Object-safe cancellation check, so a concrete session can hold `&dyn Cancellation`
/// without a generic parameter.
///
/// Today cancellation is a generic closure parameter (`should_cancel: C where C: FnMut()
/// -> bool`) at `model::qwen35::generation`'s `generate_streaming_with_cancel` and its
/// siblings; this row does not change that call site.
///
/// **The blanket impl below is narrower than that call site's bound, and the gap is real.**
/// `is_cancelled` takes `&self`, which is forced by D1's `cancel: &dyn Cancellation`
/// shape, so the impl can only cover `Fn() -> bool`. An `FnMut` closure that mutates
/// captured state is not `Fn` and does not adapt. Worse for the serving path specifically:
/// `crates/inference/src/bin/lattice/serve.rs` threads cancellation as an erased
/// `&mut dyn FnMut() -> bool`, and `&mut dyn FnMut()` implements neither `Fn` nor this
/// trait. Today's concrete closure there (`move || *cancel_rx.borrow()`) only reads its
/// capture and so is `Fn`, but the type it travels under is not, which is what a future
/// row will actually have to convert. So the covered case is: a non-mutating closure,
/// passed unerased. Everything else needs either a wrapper with interior mutability or a
/// change to the erased type along the serving chain. That choice belongs to the row that
/// wires a driver through it, and it is a choice, not a formality.
pub(crate) trait Cancellation {
    fn is_cancelled(&self) -> bool;
}

impl<F> Cancellation for F
where
    F: Fn() -> bool,
{
    fn is_cancelled(&self) -> bool {
        self()
    }
}

// ---------------------------------------------------------------------------
// DecoderSession
// ---------------------------------------------------------------------------

/// The crate-private, object-safe execution boundary D1 introduces. A concrete session
/// owns or safely borrows its immutable model and owns its mutable cache, scratch,
/// position, validated layer plan, and backend resources. No implementation lands in this
/// row; object safety is proven by the `&dyn DecoderSession` coercion in this module's
/// tests.
pub(crate) trait DecoderSession {
    fn capabilities(&self) -> &ExecutionCapabilities;

    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError>;

    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError>;

    fn select(
        &mut self,
        request: &SelectionRequest<'_>,
    ) -> Result<SelectionCandidate, InferenceError>;

    fn metadata(
        &mut self,
        prediction: PredictionId,
        final_token: u32,
        request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError>;

    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError>;
}

#[cfg(test)]
mod tests {
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

        fn select(
            &mut self,
            _request: &SelectionRequest<'_>,
        ) -> Result<SelectionCandidate, InferenceError> {
            let prediction = self.ledger.open();
            Ok(SelectionCandidate {
                candidate_id: 11,
                prediction,
            })
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
            grammar: None,
        };
        let candidate = session
            .select(&request)
            .expect("dummy select always succeeds");

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
}
