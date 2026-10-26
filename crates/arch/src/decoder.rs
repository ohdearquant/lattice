//! Decoder session vocabulary (ADR-090 D1/D2): the type surface D1's `DecoderSession` trait
//! sketches, plus the ledger mechanism that makes "a stale prediction id is rejected" a state fact
//! the type system enforces rather than a comment a future caller has to remember.
//!
//! [`driver`] holds the one autoregressive loop over `&mut dyn DecoderSession`. Grammar ownership
//! lives in `driver`, not in this trait: `select` reads a borrowed mask through
//! [`SelectionRequest::grammar_mask`] and the driver calls `advance` /
//! `is_complete_without_continuation` on its own owned engine and state -- see `driver`'s own
//! module doc comment for the full per-step ordering and the reasoning for that split.
//!
//! **Unstable**: this module is the public session contract (ADR-103 Amendment 21 item 3), and it
//! may still change in a minor release. A model family implements [`DecoderSession`] and runs it
//! with [`driver::run`]; the records the trait exchanges are `#[non_exhaustive]` and built with
//! their constructors, so an implementation outside this crate can return every value the trait
//! asks for.
//!
//! **One request per session.** A session serves one request and [`DecoderSession::finish`] ends
//! it; the call order and what an implementation must refuse are stated once, on
//! [`DecoderSession`].
//!
//! **Loaded owners.** [`DecoderModel`] is a loaded model that opens sessions: each
//! [`DecoderModel::session`] borrows the owner and one [`SessionRequest`] for the session's
//! lifetime, and the driver reads its prompt, configuration and token policy from that same
//! request. A loaded owner is local to the worker that loaded it; a consumer moves the loader and
//! its [`crate::family::LoadRequest`] across threads, never the owner or a session.
//!
//! The speculative extension (the typed draft-verification protocol the Qwen Metal routes use)
//! stays internal and is not part of this module.

use crate::error::InferenceError;
use crate::generation::{GenerateConfig, TopLogprob};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// The autoregressive driver (ADR-090 row C): one loop over `&mut dyn DecoderSession`
/// that drives the shared decode policy unchanged.
pub mod driver;

/// The speculative extension of the session contract and its driver (ADR-090 D2, D6). Internal:
/// reached by the facade through `__internal`, never re-exported here.
pub(crate) mod speculative;

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
///
/// Opacity stops fabrication, not misuse of a genuine id: a session checks every id it is
/// handed against its own ledger before it reads metadata or changes state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictionId {
    ledger: u64,
    seq: u64,
}

/// Source of ledger identities. Relaxed ordering is sufficient: the only property required
/// is that no two allocations return the same value, which a single atomic read-modify-write
/// guarantees on its own without any happens-before relationship between ledgers.
static NEXT_LEDGER_ID: AtomicU64 = AtomicU64::new(0);

/// Takes the next identity from `counter`, refusing once it is spent instead of wrapping to an
/// identity an earlier ledger already holds. Takes the counter as an argument so the exhaustion
/// arm is testable without spending the process-wide one.
fn allocate_ledger_id(counter: &AtomicU64) -> Result<u64, PredictionError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| PredictionError::Exhausted)
}

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
pub struct PredictionLedger {
    id: u64,
    next_seq: u64,
    live: Option<PredictionId>,
}

impl PredictionLedger {
    /// A ledger with a fresh identity no other ledger in this process shares.
    ///
    /// # Errors
    ///
    /// [`PredictionError::Exhausted`] once the process has spent every ledger identity; the
    /// counter never wraps to an identity an earlier ledger holds.
    pub fn new() -> Result<Self, PredictionError> {
        Ok(Self {
            id: allocate_ledger_id(&NEXT_LEDGER_ID)?,
            next_seq: 0,
            live: None,
        })
    }

    /// Mint a new prediction id and make it the ledger's one live prediction.
    ///
    /// **Decision (documented and tested below): supersede, not refuse.** A caller that
    /// opens a second prediction while the first is still live is not told no. The ledger
    /// instead enforces its own "at most one live prediction" invariant by retiring the old
    /// id before minting the new one: the superseded id fails the same `is_live` check a
    /// cancelled, consumed, or reset-invalidated id would fail, through the same mechanism,
    /// rather than inventing a second way for a `PredictionId` to go bad. A driver that opens
    /// twice per step has a bug either way; this choice makes that bug surface as an ordinary
    /// stale-id rejection on whichever id the driver kept, instead of two simultaneously
    /// "live" ids racing to be consumed.
    ///
    /// # Errors
    ///
    /// [`PredictionError::Exhausted`] once this ledger has minted every sequence number. The
    /// refusal changes nothing: the live prediction, if any, stays live.
    pub fn open(&mut self) -> Result<PredictionId, PredictionError> {
        let next_seq = self
            .next_seq
            .checked_add(1)
            .ok_or(PredictionError::Exhausted)?;
        let id = PredictionId {
            ledger: self.id,
            seq: self.next_seq,
        };
        self.next_seq = next_seq;
        self.live = Some(id);
        Ok(id)
    }

    /// Whether `id` is the ledger's current live prediction. This is the eligibility check
    /// a concrete session's `select`/`metadata` operations must consult: a cancelled,
    /// consumed, superseded, or reset-invalidated id is never live again.
    pub fn is_live(&self, id: PredictionId) -> bool {
        self.live == Some(id)
    }

    /// Whether the ledger holds a live prediction.
    pub fn has_live(&self) -> bool {
        self.live.is_some()
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
    pub fn reset(&mut self) {
        self.live = None;
    }

    /// Cancellation, failure, or finish: the live prediction, if any, is no longer
    /// eligible. See [`Self::reset`] on why these are two methods with one behaviour.
    pub fn invalidate(&mut self) {
        self.live = None;
    }

    /// Consume `id`: the live prediction must currently be `id`, and after this call it no
    /// longer is. A second `consume` of the same id -- whether the original caller retried
    /// it or a different caller captured a copy -- finds no matching live prediction and is
    /// rejected the same way any other stale id would be.
    pub fn consume(&mut self, id: PredictionId) -> Result<(), PredictionError> {
        if self.live == Some(id) {
            self.live = None;
            Ok(())
        } else {
            Err(PredictionError::Stale)
        }
    }
}

/// Failure taxonomy for the prediction ledger. See
/// `impl From<PredictionError> for InferenceError` for how this maps into the crate's
/// public error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PredictionError {
    /// `consume` was called with an id that is not the ledger's current live prediction:
    /// already consumed, superseded, cancelled, or invalidated by a prefill/reset.
    Stale,
    /// The process has spent every ledger identity, or this ledger has spent every sequence
    /// number. Reported instead of wrapping to an identity that is already in use.
    Exhausted,
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictionError::Stale => {
                f.write_str("stale prediction id: no matching live prediction in the ledger")
            }
            PredictionError::Exhausted => {
                f.write_str("prediction ledger exhausted: no unused prediction id remains")
            }
        }
    }
}

impl std::error::Error for PredictionError {}

/// Maps into the crate's general execution-failure bucket. `InferenceError` is
/// `#[non_exhaustive]`; this row does not add a new public variant. A stale prediction id
/// is a session-internal protocol error (the driver misused the ledger), not
/// caller-supplied invalid input, so it is not routed through `InvalidInput`.
impl From<PredictionError> for InferenceError {
    fn from(value: PredictionError) -> Self {
        InferenceError::Inference(value.to_string())
    }
}

// ---------------------------------------------------------------------------
// AcceptedToken / SelectionCandidate / SelectionRequest
// ---------------------------------------------------------------------------

/// Binds an accepted final token id to the prediction that produced its candidate: "the
/// final token pending evaluation" in D1's vocabulary. Fields are readable, and
/// [`Self::new`] builds one, but the only way to obtain a `PredictionId` to put in it is
/// [`PredictionLedger::open`], so construction is gated by the ledger.
///
/// Not `Clone`, not `Copy` -- but that is a hint, not the enforcement, and the difference
/// matters. D1's `decode(&mut self, accepted: &AcceptedToken, ..)` takes a SHARED borrow, so
/// the type cannot stop a caller from passing the same token twice; and the embedded
/// `PredictionId` is `Copy`, so the identity can be lifted out and reused on its own.
/// Consume-at-most-once is therefore a property of [`PredictionLedger::consume`], which
/// clears the live slot so the second attempt finds nothing and is rejected exactly like any
/// other stale id. The non-`Clone` shape only removes the most casual way to get it wrong.
#[derive(Debug)]
#[non_exhaustive]
pub struct AcceptedToken {
    pub final_id: u32,
    pub prediction: PredictionId,
}

impl AcceptedToken {
    /// The final token `final_id`, accepted for the prediction `prediction`.
    pub const fn new(final_id: u32, prediction: PredictionId) -> Self {
        Self {
            final_id,
            prediction,
        }
    }
}

/// A sampled candidate id and the prediction it belongs to. Sampling alone authorizes
/// neither publication nor consumption (D1); the candidate becomes an [`AcceptedToken`]
/// only after policy finalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SelectionCandidate {
    pub candidate_id: u32,
    pub prediction: PredictionId,
}

impl SelectionCandidate {
    /// The candidate `candidate_id`, sampled from the prediction `prediction`.
    pub const fn new(candidate_id: u32, prediction: PredictionId) -> Self {
        Self {
            candidate_id,
            prediction,
        }
    }
}

/// Grammar mask closure carried by [`SelectionRequest::grammar_mask`]: applies grammar
/// masking to a session's own logits in place, or fails with the same error
/// `lattice_grammar::GrammarEngine::mask_logits` returns. Named as its own alias (rather than
/// spelled out inline) purely to keep `SelectionRequest`'s and `decoder::driver::run`'s type
/// signatures readable -- clippy's `type_complexity` lint flags the unaliased form.
pub type GrammarMaskFn<'a> = dyn Fn(&mut [f32]) -> Result<(), InferenceError> + 'a;

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
///
/// `grammar_mask` is the one field that is not `Copy`-trivial to reason about but still lets
/// this type stay `#[derive(Copy)]`: a shared reference to a `dyn Fn` is `Copy` regardless of
/// what the closure behind it captures. The driver builds this closure once per `run()` call
/// over its own owned `GrammarEngine`/`GrammarState` (interior mutability via `RefCell`, since
/// masking needs `&mut GrammarState` but `select` only ever sees `&SelectionRequest`), and
/// hands a borrow of it to every step's request. `None` when no grammar was set on
/// `GenerateConfig` -- the ADR-090 D1 owner of grammar transitions is the driver, not any
/// concrete session, so a session with no grammar support simply never receives `Some` here.
/// See `decoder::driver`'s module doc comment for the full per-step ordering.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct SelectionRequest<'a> {
    pub config: &'a GenerateConfig,
    pub history: &'a [u32],
    pub grammar_mask: Option<&'a GrammarMaskFn<'a>>,
}

impl<'a> SelectionRequest<'a> {
    /// A request to select from `config` over `history`, with no grammar mask.
    pub const fn new(config: &'a GenerateConfig, history: &'a [u32]) -> Self {
        Self {
            config,
            history,
            grammar_mask: None,
        }
    }

    /// The same request, with `mask` applied to the session's logits before selection.
    pub const fn with_grammar_mask(self, mask: &'a GrammarMaskFn<'a>) -> Self {
        Self {
            grammar_mask: Some(mask),
            ..self
        }
    }
}

/// Outcome of [`DecoderSession::select`]: either a real sampled candidate, or a signal that
/// grammar masking blocked every token, i.e. no finite logit remained after applying
/// [`SelectionRequest::grammar_mask`]. This variant does not by itself distinguish "the
/// grammar reached a valid accepting state with no further legal continuation" (a completed
/// grammar -- generation ends here with no new token sampled) from "no legal continuation
/// exists and the grammar is not in an accepting state" (a hard error): resolving that
/// requires the grammar engine and state, which the session no longer holds (ADR-090 D1: the
/// driver owns grammar transitions). The driver makes that call itself right after `select`
/// returns this variant, via its own owned `is_complete_without_continuation` check, and
/// returns `InferenceError::GrammarConstraintBlocked` directly (never as a variant of this
/// enum) for the not-complete case -- mirroring the pre-driver inline loops' identical
/// distinction, just relocated to the caller that still has the state to make it.
#[non_exhaustive]
pub enum SelectOutcome {
    Candidate(SelectionCandidate),
    GrammarExhausted,
}

// ---------------------------------------------------------------------------
// MetadataRequest / TokenMetadata
// ---------------------------------------------------------------------------

/// What the entry profile asked for from the `metadata` operation. Mirrors
/// [`GenerateConfig::logprobs`] exactly: `None` ([`Self::disabled`]) disables metadata capture
/// entirely, and a session refuses a direct `metadata` call carrying it with
/// [`InferenceError::InvalidInput`]; `Some(n)` ([`Self::with_top_logprobs`]) requests the final
/// token's log-probability plus its `n` highest-probability alternatives (`n == 0` is valid:
/// report only the final token's log-probability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct MetadataRequest {
    pub top_logprobs: Option<usize>,
}

impl MetadataRequest {
    /// Metadata capture disabled. A direct `metadata` call with this request is refused.
    pub const fn disabled() -> Self {
        Self { top_logprobs: None }
    }

    /// The final token's log-probability and its `n` highest-probability alternatives.
    pub const fn with_top_logprobs(n: usize) -> Self {
        Self {
            top_logprobs: Some(n),
        }
    }
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
#[non_exhaustive]
pub struct TokenMetadata {
    // `driver::run`'s `record_metadata` closure trusts the session's own bookkeeping and
    // reads only `final_logprob`/`top`; these two identity fields exist for the
    // consistency guarantee this doc comment describes and are exercised by this
    // module's and `qwen_cpu`'s tests (e.g. `assert_eq!(metadata.prediction, ...)`).
    pub prediction: PredictionId,
    pub final_token_id: u32,
    pub final_logprob: f32,
    pub top: Vec<TopLogprob>,
}

impl TokenMetadata {
    /// `final_token_id`, scored against `prediction`: its log-probability and the
    /// highest-probability alternatives the request asked for, in descending order.
    pub fn new(
        prediction: PredictionId,
        final_token_id: u32,
        final_logprob: f32,
        top: Vec<TopLogprob>,
    ) -> Self {
        Self {
            prediction,
            final_token_id,
            final_logprob,
            top,
        }
    }
}

// ---------------------------------------------------------------------------
// StepStamp
// ---------------------------------------------------------------------------

/// Identifies the evaluated input prefix and the current prediction, if one was produced,
/// after a `prefill` or `decode` step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StepStamp {
    pub evaluated_len: usize,
    pub prediction: Option<PredictionId>,
}

impl StepStamp {
    /// `evaluated_len` input tokens evaluated, with `prediction` as the current prediction.
    pub const fn new(evaluated_len: usize, prediction: Option<PredictionId>) -> Self {
        Self {
            evaluated_len,
            prediction,
        }
    }
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
///
/// Built from [`Self::none`] (also the `Default`) with one `with_*` call per declared
/// control.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExecutionCapabilities {
    pub grammar: bool,
    pub logprobs: bool,
    pub stop_strings: bool,
    pub reasoning_budget: bool,
}

impl ExecutionCapabilities {
    /// No optional control declared.
    pub const fn none() -> Self {
        Self {
            grammar: false,
            logprobs: false,
            stop_strings: false,
            reasoning_budget: false,
        }
    }

    /// Declares, or withdraws, grammar-constrained selection.
    pub const fn with_grammar(self, grammar: bool) -> Self {
        Self { grammar, ..self }
    }

    /// Declares, or withdraws, per-token log-probabilities.
    pub const fn with_logprobs(self, logprobs: bool) -> Self {
        Self { logprobs, ..self }
    }

    /// Declares, or withdraws, stop-string matching.
    pub const fn with_stop_strings(self, stop_strings: bool) -> Self {
        Self {
            stop_strings,
            ..self
        }
    }

    /// Declares, or withdraws, a reasoning budget.
    pub const fn with_reasoning_budget(self, reasoning_budget: bool) -> Self {
        Self {
            reasoning_budget,
            ..self
        }
    }
}

// ---------------------------------------------------------------------------
// FinishDisposition
// ---------------------------------------------------------------------------

/// How a session's execution ended, and whether the resources behind it are still valid
/// (D2). Distinct from [`crate::stop_reason::StopReason`], which answers *why* generation
/// stopped (EOS, length, grammar dead end, ...) -- a question orthogonal to whether the
/// session's internal state is still valid. This row does not duplicate that type.
///
/// Either way, `finish` ends the session's use through [`DecoderSession`]: the disposition
/// tells the session's typed owner what it may do with the underlying resources, never that
/// the same session accepts a second request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishDisposition {
    /// Generation completed normally, or a failure occurred before any state mutation:
    /// the typed owner may keep the underlying resources and reset them for its next
    /// session.
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
/// `crates/serve/src/bin/lattice/serve.rs` threads cancellation as an erased
/// `&mut dyn FnMut() -> bool`, and `&mut dyn FnMut()` implements neither `Fn` nor this
/// trait. Today's concrete closure there (`move || *cancel_rx.borrow()`) only reads its
/// capture and so is `Fn`, but the type it travels under is not, which is what a future
/// row will actually have to convert. So the covered case is: a non-mutating closure,
/// passed unerased. Everything else needs either a wrapper with interior mutability or a
/// change to the erased type along the serving chain. That choice belongs to the row that
/// wires a driver through it, and it is a choice, not a formality.
pub trait Cancellation {
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

/// The object-safe execution boundary D1 introduces. A concrete session
/// owns or safely borrows its immutable model and owns its mutable cache, scratch,
/// position, validated layer plan, and backend resources. The trait has no `Send` or `Sync`
/// requirement: a session may hold thread-bound resources and is created, run and dropped
/// on one thread.
///
/// A session serves one request, in the order prefill, then `select` / `decode` steps, then
/// `finish`. An implementation must refuse, before changing any state, a call out of that
/// order, a prediction id that is not its ledger's live prediction (stale, consumed, or
/// minted by another session), and any call after `finish`; [`driver::run`] makes no such
/// call, and finishes the session once on every path that returns after its preflight. Dropping
/// a session without `finish` must still release its resources.
pub trait DecoderSession {
    /// The optional controls this session supports. The driver refuses a request that uses
    /// a control the session does not declare, before any other call.
    fn capabilities(&self) -> &ExecutionCapabilities;

    /// Evaluates the session's prompt and leaves its prediction for the first output
    /// position ready for `select`.
    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError>;

    /// Opts into cancellation during prefill; `None` requires a poisoned finish.
    /// The default ignores `cancel` and runs [`Self::prefill`] to completion, leaving the
    /// driver's own checkpoints before and after it. An override promises cancellation at
    /// its own checkpoints, not a bound on how long a cancelled prefill keeps running.
    fn prefill_interruptible(
        &mut self,
        _cancel: &dyn Cancellation,
    ) -> Result<Option<StepStamp>, InferenceError> {
        self.prefill(&|| false).map(Some)
    }

    /// Evaluates `accepted`, consuming its prediction, and leaves the prediction for the
    /// next position ready for `select`.
    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError>;

    /// Samples a candidate from the current prediction and opens it in the session's
    /// ledger.
    fn select(&mut self, request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError>;

    /// Scores `final_token` against `prediction`, which must be the ledger's live
    /// prediction. A request with metadata disabled ([`MetadataRequest::disabled`]) is
    /// refused with [`InferenceError::InvalidInput`].
    fn metadata(
        &mut self,
        prediction: PredictionId,
        final_token: u32,
        request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError>;

    /// Ends the request and its live prediction, if any. See [`FinishDisposition`].
    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError>;
}

// ---------------------------------------------------------------------------
// SessionRequest / DecoderModel
// ---------------------------------------------------------------------------

/// One admitted generation request: the prompt token ids, the generation configuration and the
/// token policy of the model family, owned so that the session and the driver read the same
/// values.
///
/// The token policy is the end-of-sequence id and, when a reasoning budget is enforced, the id of
/// the token that closes a reasoning block together with the table of tokens that end a line
/// (indexed by token id). [`driver::run`] takes them as `eos_token_id`, `think_close_id` and
/// `reasoning_newline_boundary`. A [`DecoderModel`] refuses a request whose policy disagrees with
/// the loaded model.
#[derive(Clone)]
pub struct SessionRequest {
    prompt_ids: Vec<u32>,
    config: GenerateConfig,
    eos_token_id: u32,
    think_close_id: Option<u32>,
    reasoning_newline_boundary: Option<Arc<[bool]>>,
}

impl SessionRequest {
    /// A request for `prompt_ids` under `config`, ending at `eos_token_id`, with no reasoning
    /// close token.
    pub fn new(prompt_ids: Vec<u32>, config: GenerateConfig, eos_token_id: u32) -> Self {
        Self {
            prompt_ids,
            config,
            eos_token_id,
            think_close_id: None,
            reasoning_newline_boundary: None,
        }
    }

    /// Sets the token that closes a reasoning block and the table of line-ending tokens, both
    /// needed to enforce `config.reasoning_budget`.
    pub fn with_reasoning_close(
        mut self,
        think_close_id: u32,
        newline_boundary: Arc<[bool]>,
    ) -> Self {
        self.think_close_id = Some(think_close_id);
        self.reasoning_newline_boundary = Some(newline_boundary);
        self
    }

    /// The prompt token ids.
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    /// The generation configuration.
    pub fn config(&self) -> &GenerateConfig {
        &self.config
    }

    /// The end-of-sequence token id.
    pub const fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// The token that closes a reasoning block, if set.
    pub const fn think_close_id(&self) -> Option<u32> {
        self.think_close_id
    }

    /// Whether each token id ends a line, if a reasoning close token is set.
    pub fn reasoning_newline_boundary(&self) -> Option<&Arc<[bool]>> {
        self.reasoning_newline_boundary.as_ref()
    }
}

/// Turns a loaded model's token ids into text.
///
/// Read-only over the model's tokenizer: it decodes and streams, and has no way to change the
/// tokenizer or reach the model that handed it out. A consumer holds it beside a session, which
/// borrows the model mutably, so that borrow stays the only one.
pub trait TextDecoder: Send + Sync {
    /// The text of `ids`. Bytes that do not form valid UTF-8 are replaced with U+FFFD.
    fn decode(&self, ids: &[u32]) -> String;

    /// A new incremental decoder for one generated sequence.
    fn stream(&self) -> Box<dyn TextStream + '_>;
}

/// Incremental decoding of one generated sequence, token by token.
pub trait TextStream {
    /// Appends to `out` the text that `id` completes. Bytes of a character that a later token
    /// may complete are held back. This is the shape [`driver::run_into`] takes for its
    /// `decode_delta`, which clears the buffer before each call.
    fn push_into(&mut self, id: u32, out: &mut String);

    /// The bytes still held back at the end of the sequence, with invalid UTF-8 replaced by
    /// U+FFFD; empty when none are held.
    fn finish(&mut self) -> String;
}

/// A loaded decoder model that opens [`DecoderSession`]s.
///
/// No `Send` or `Sync` requirement: an owner may hold thread-bound backend state, and is loaded,
/// used and dropped on one worker. A session borrows the owner mutably for its whole life, so an
/// owner runs one session at a time.
pub trait DecoderModel {
    /// Tokenizes `prompt` and resolves the model's token policy for `config` into a request
    /// that [`DecoderModel::session`] accepts.
    ///
    /// The request is owned and borrows nothing from `self`: a caller admits with `&self`,
    /// keeps the request, and then passes it to `session`, which borrows the owner mutably.
    ///
    /// When `config.max_new_tokens` is 0 nothing will be generated, so neither the context
    /// budget nor the reasoning close token is checked, and the request carries no close token.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] when `prompt` tokenizes to no tokens, and when the prompt's
    /// token count plus the decode cap of `config` exceeds the loaded model's context window.
    /// [`InferenceError::InvalidInput`] when `config` enforces a reasoning budget and the model
    /// has no usable token that closes a reasoning block.
    fn admit(&self, prompt: &str, config: GenerateConfig)
    -> Result<SessionRequest, InferenceError>;

    /// The decoder of this model's token ids, shared with the model.
    fn text(&self) -> Arc<dyn TextDecoder>;

    /// Opens a session for `request`. Refuses, before allocating session state, a request the
    /// loaded model cannot serve: an empty prompt, a token id outside the vocabulary, a token
    /// policy that differs from the model's, or a prompt and decode budget past its context.
    fn session<'m>(
        &'m mut self,
        request: &'m SessionRequest,
    ) -> Result<Box<dyn DecoderSession + 'm>, InferenceError>;
}

#[cfg(test)]
mod tests;
