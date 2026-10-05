//! Byte-level pushdown automaton (PDA) for context-free grammar matching.
//!
//! # Design
//!
//! The grammar is compiled to a set of *rules*, each of which is a set of
//! alternatives, each alternative being an ordered sequence of *symbols*
//! (either a terminal byte or a non-terminal rule reference). A parse in
//! progress is a stack of `StackFrame`s:
//!
//! ```text
//! frame = (rule_id, alt_idx, sym_pos)
//! stack = [frame, ...]        bottom = root rule, top = innermost rule
//! ```
//!
//! In every frame below the top, `sym_pos` points at the non-terminal that the
//! frame above it is currently matching.
//!
//! # One parse is not enough
//!
//! The matcher does not commit to a single parse. A `GrammarState` holds the
//! whole *set* of stacks that are consistent with the bytes seen so far, and
//! `advance_byte(b)` advances every stack that can consume `b`, expanding
//! non-terminals into each of their alternatives on the way, and drops the
//! rest. Two alternatives that begin with the same byte therefore both stay
//! alive until a later byte tells them apart, and a byte is rejected only when
//! no stack can consume it. There is no backtracking and no record of which
//! bytes a frame has consumed: a parse that turns out to be wrong simply stops
//! being in the set. This is the design llama.cpp uses for GBNF.
//!
//! After every byte the set is sorted and deduplicated, so two states that
//! hold the same stacks compare equal. The vocabulary partition keys on that
//! set.
//!
//! # Limits
//!
//! * `MAX_PDA_DEPTH` bounds the length of one stack. A left-recursive or
//!   cyclic grammar pushes frames without consuming a byte, so a step that
//!   needs a stack past the bound stops there (issue #343).
//! * `MAX_LIVE_STACKS` bounds how many distinct stacks one step may keep alive
//!   and `MAX_EXPANSION_STEPS` bounds how many structural expansion steps it
//!   may take.
//!
//! A grammar that needs more than any of the three is too ambiguous or too
//! deeply nested for set-of-stacks matching, and `advance_byte` returns
//! `StepResult::StackLimitExceeded`. That outcome is deliberately separate from
//! `StepResult::Rejected`, so that a caller cannot mistake an overloaded
//! matcher for a grammar that refuses the input.
//!
//! # Grammar representation
//!
//! A `Rule` is a named set of alternatives, each alternative being an ordered
//! list of `Symbol`s:
//!
//! ```text
//! Rule { name, alts: Vec<Vec<Symbol>> }
//! Symbol::Terminal(u8)
//! Symbol::NonTerminal(rule_id)
//! Symbol::AnyByte   — matches any single byte (used for GBNF `.` and `[^...]`)
//! ```
//!
//! The root rule has id 0 (by convention enforced by `CompiledGrammar`). A
//! rule with no alternatives matches the empty string.
//!
//! # The UTF-8 contract, and where it stops
//!
//! **This automaton matches BYTES, not characters.** `Symbol::AnyByte`, which is what
//! GBNF `.` and `[^...]` compile to, accepts any of the 256 byte values. That includes
//! `0xFF`, which is not legal anywhere in UTF-8, and lone `0x80..=0xBF` continuation
//! bytes. It is deliberate: byte-level `.` is GBNF's established meaning, and narrowing
//! it to "one UTF-8 scalar" would silently change the language accepted by every grammar
//! ported from a byte-level GBNF implementation.
//!
//! The consequence is a seam that neither side of it used to name. A grammar built from
//! `.` or `[^...]` can accept a byte sequence that is not well-formed UTF-8; the
//! detokenizer then renders those bytes through `String::from_utf8_lossy`, substituting
//! `U+FFFD`. **So the returned string is not necessarily the byte sequence this automaton
//! validated, and re-validating that string against the same grammar can fail.** What
//! grammar-constrained decoding guarantees is a property of the emitted bytes, not of the
//! decoded string.
//!
//! Two things bound that in practice, and both are checkable rather than hoped for:
//!
//! * A grammar that uses neither `.` nor a negated class cannot reach the case at all,
//!   because every other terminal names one specific byte.
//! * The JSON-schema compiler does not use `AnyByte` for string contents. It models
//!   well-formed 2/3/4-byte UTF-8 with the correct lead and continuation ranges
//!   (issue #931), so schema-constrained output is valid UTF-8 by construction and its
//!   decoded string does re-validate.
//!
//! The other end of the seam is documented on `tokenizer::detokenize::decode_tokens`.
//!
//! # State machine encoding
//!
//! A `GrammarState` encodes the full PDA configuration:
//!
//! ```text
//! stacks: Vec<Vec<StackFrame>>   sorted, deduplicated
//! partial_bytes: Vec<u8>         bytes of current token received so far
//! complete: bool                 some stack can finish with no further input
//! ```
//!
//! The automaton starts with one stack holding a single frame for the root
//! rule whose alternative is not chosen yet (`UNCHOSEN_ALT`); the first byte
//! expands it into one stack per root alternative.
//! `advance_byte(b)` returns whether the byte `b` is accepted (some stack can
//! consume it) and updates the stack set in-place.
//!
//! `can_accept_more()` returns whether the current stack set can still
//! accept additional input (used for context-dependent token masking).
//! `is_complete()` returns whether some stack can finish through nullable
//! symbols alone, i.e. whether the bytes so far are a complete match.

use std::collections::HashMap;

/// A single rule alternative: an ordered sequence of symbols.
pub type Alt = Vec<Symbol>;

/// An element in a grammar rule alternative.
#[derive(Debug, Clone, PartialEq)]
pub enum Symbol {
    /// Matches a single literal byte.
    Terminal(u8),
    /// Matches any single byte (GBNF `.`).
    AnyByte,
    /// Expands into the named rule.
    NonTerminal(usize),
}

/// Maximum stack length a byte step may build.
///
/// A left-recursive or cyclic grammar (`root ::= root`, or a JSON-Schema `$ref`
/// cycle) makes the matcher push non-terminal frames without ever consuming a
/// byte, growing the stack without bound — a hang reachable from untrusted
/// grammar input at `GrammarEngine::new` (issue #343). A step that would grow a
/// stack past this depth fails with `StepResult::StackLimitExceeded` instead of
/// running out of memory. That is a limit of the matcher, not a verdict on the
/// input: a productive recursion such as `root ::= "a" root | ""` accepts every
/// run of `a` bytes, and the byte that needs the stack to pass this depth is
/// reported as a limit rather than as a rejection. The bound is far above any
/// real nesting: `serde_json` itself caps recursion at 128, and each JSON level
/// expands to only a handful of PDA frames, so 8192 frames is unreachable by a
/// well-formed grammar on well-formed output.
pub(crate) const MAX_PDA_DEPTH: usize = 8192;

/// Maximum number of distinct stacks one byte step may keep alive, counting
/// both the stacks it has produced and the ones still waiting to be explored.
///
/// A grammar whose ambiguity keeps more parses open than this at once is too
/// ambiguous for set-of-stacks matching. The step then fails with
/// `StepResult::StackLimitExceeded` instead of growing without bound or
/// pretending the grammar rejected the byte. The bound is far above what the
/// schema compiler produces: an object with N optional properties keeps about
/// N stacks open at a key boundary, and identical stacks are merged before
/// they are counted.
pub(crate) const MAX_LIVE_STACKS: usize = 1024;

/// Maximum number of structural expansion steps one byte step may take before
/// it gives up.
///
/// A structural step opens the top frame of one stack: it chooses among the
/// alternatives of a rule, enters a non-terminal, or leaves an exhausted frame.
/// Distinct live stacks are not the only cost: nested nullable alternatives and
/// long runs of nullable symbols can multiply the steps a byte has to take even
/// when no parse survives it, so the work needs its own bound. Exceeding it
/// reports `StepResult::StackLimitExceeded`, like `MAX_LIVE_STACKS`.
pub(crate) const MAX_EXPANSION_STEPS: usize = 16 * MAX_LIVE_STACKS;

/// A compiled grammar rule: a name and a set of alternatives.
#[derive(Debug, Clone)]
pub struct Rule {
    /// Human-readable name (for debugging).
    pub name: String,
    /// The alternatives for this rule, in priority order.
    pub alts: Vec<Alt>,
}

/// The compiled grammar: a flat list of rules, root at index 0.
#[derive(Debug, Clone)]
pub struct CompiledGrammar {
    pub rules: Vec<Rule>,
}

impl CompiledGrammar {
    /// Number of rules in the grammar.
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Return the root rule (index 0), or `None` when the grammar has no rules.
    pub fn root(&self) -> Option<&Rule> {
        self.rules.first()
    }
}

/// `StackFrame::alt_idx` of a frame whose rule has been entered but whose
/// alternative has not been chosen yet.
///
/// The matcher replaces it with one frame per alternative of the rule when it
/// next steps the stack, so only a freshly pushed frame (and the root frame of
/// the initial state) ever carries it.
pub const UNCHOSEN_ALT: usize = usize::MAX;

/// One frame on a PDA stack.
///
/// `Ord` sorts the stacks of a `GrammarState` into a canonical order, and
/// `Eq + Hash` let a `GrammarState`'s `(stacks, complete)` pair — the same
/// identity `states_equal` in `engine.rs` compares by — key a `HashMap` for
/// state-revisit / memoization profiling.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StackFrame {
    /// Index into `CompiledGrammar::rules`.
    pub rule_id: usize,
    /// Index into `rules[rule_id].alts`, or [`UNCHOSEN_ALT`].
    pub alt_idx: usize,
    /// Position within the chosen alternative (0 = before the first symbol).
    pub sym_pos: usize,
}

/// Runtime state of the PDA for one decode sequence.
///
/// Clone this at each step to enable parallel-beam grammar tracking.  The
/// cost is O(total frames over the live stacks): a single stack of 2-6 frames
/// at positions where only one parse is open, a few more where alternatives
/// that share a prefix have not been told apart yet.
#[derive(Debug, Clone)]
pub struct GrammarState {
    /// The live stacks; the top of each stack is its last element. Sorted and
    /// free of duplicates after every accepted byte, so equal sets of stacks
    /// compare equal.
    pub stacks: Vec<Vec<StackFrame>>,
    /// Bytes accumulated within the current token (context-dependent checks).
    pub partial_token_bytes: Vec<u8>,
    /// `true` once the root rule has been fully matched (EOS is valid).
    pub complete: bool,
}

impl GrammarState {
    /// Initial state: a single stack with one frame at the root rule, its
    /// alternative not yet chosen, at sym_pos 0.
    pub fn initial() -> Self {
        Self {
            stacks: vec![vec![StackFrame {
                rule_id: 0,
                alt_idx: UNCHOSEN_ALT,
                sym_pos: 0,
            }]],
            partial_token_bytes: Vec::new(),
            complete: false,
        }
    }

    /// Returns true if the automaton has consumed all input and is in an
    /// accepting configuration (some stack is empty or all its remaining
    /// frames are at rules whose alternatives can complete with zero bytes).
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Returns true if the automaton could potentially accept more bytes.
    /// Used during context-dependent token inspection.
    pub fn can_accept_more(&self) -> bool {
        !self.complete || self.stacks.iter().any(|stack| !stack.is_empty())
    }
}

pub(crate) fn initial_grammar_state(grammar: &CompiledGrammar) -> GrammarState {
    let mut state = GrammarState::initial();
    state.complete = is_accepting(&state, grammar);
    state
}

// ---------------------------------------------------------------------------
// PDA execution engine
// ---------------------------------------------------------------------------

/// A byte step needed more stacks than `MAX_LIVE_STACKS` allows, more
/// expansion work than `MAX_EXPANSION_STEPS` allows, or a stack deeper than
/// `MAX_PDA_DEPTH`.
///
/// This is a limit of the matcher, not a verdict on the input: it says the
/// grammar is too ambiguous or too deeply nested to track, not that the grammar
/// rejects the byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackLimitError;

impl std::fmt::Display for StackLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "grammar is too ambiguous or too deeply nested to match: one step needed more than {MAX_LIVE_STACKS} live stacks, {MAX_EXPANSION_STEPS} expansion steps or {MAX_PDA_DEPTH} stack frames"
        )
    }
}
impl std::error::Error for StackLimitError {}

/// Result of attempting to advance the PDA by one byte.
#[derive(Debug, Clone, PartialEq)]
pub enum StepResult {
    /// The byte was accepted; the state has been updated.
    Accepted,
    /// No live stack can consume the byte; the state is unchanged.
    Rejected,
    /// The step needed more live stacks, more expansion work or a deeper stack
    /// than the matcher allows (see [`StackLimitError`]); the state is
    /// unchanged.
    ///
    /// This says nothing about whether the grammar admits the byte, so it must
    /// not be handled as a grammar rejection. Compare against
    /// [`StepResult::Accepted`] rather than against `Rejected` when deciding
    /// whether a byte went through.
    StackLimitExceeded,
}

/// One parse in progress; the top frame is the last element.
type Stack = Vec<StackFrame>;

/// What the top of a stack needs next.
enum Top {
    /// The stack is empty: the root rule has been matched in full.
    Empty,
    /// The top symbol matches one byte: `Some(b)` for a terminal, `None` for
    /// `Symbol::AnyByte`.
    Byte(Option<u8>),
    /// The top has to be expanded before it can match a byte: an alternative
    /// not chosen yet, a non-terminal, or an exhausted frame.
    Open,
    /// The stack is deeper than `MAX_PDA_DEPTH`: the step cannot tell whether
    /// it could match, so it fails with a stack limit.
    TooDeep,
    /// The stack can never match anything: it names a rule, alternative or
    /// non-terminal that does not exist.
    Dead,
}

fn classify_top(stack: &[StackFrame], grammar: &CompiledGrammar) -> Top {
    let Some(frame) = stack.last() else {
        return Top::Empty;
    };
    if stack.len() > MAX_PDA_DEPTH {
        // Cyclic / left-recursive grammar pushing frames without progress
        // (issue #343). Stop rather than grow the stack unbounded.
        return Top::TooDeep;
    }
    let Some(rule) = grammar.rules.get(frame.rule_id) else {
        return Top::Dead;
    };
    // A rule with no alternatives matches the empty string.
    if rule.alts.is_empty() || frame.alt_idx == UNCHOSEN_ALT {
        return Top::Open;
    }
    let Some(alt) = rule.alts.get(frame.alt_idx) else {
        return Top::Dead;
    };
    match alt.get(frame.sym_pos) {
        None => Top::Open,
        Some(Symbol::Terminal(t)) => Top::Byte(Some(*t)),
        Some(Symbol::AnyByte) => Top::Byte(None),
        Some(Symbol::NonTerminal(rule_id)) => {
            if grammar.rules.get(*rule_id).is_some() {
                Top::Open
            } else {
                Top::Dead
            }
        }
    }
}

fn byte_matches(want: Option<u8>, b: u8) -> bool {
    want.is_none_or(|w| w == b)
}

/// Advance a `GrammarState` by one byte `b` against `grammar`.
///
/// Every live stack is stepped: a stack whose next symbol is a terminal takes
/// the byte if it matches, and a stack that needs expanding is expanded into
/// each alternative that could consume `b`. Stacks that cannot consume `b` are
/// dropped. The byte is accepted when at least one stack survives, and the
/// surviving stacks are sorted and deduplicated.
///
/// Returns [`StepResult::Rejected`] when no stack survives, and
/// [`StepResult::StackLimitExceeded`] when the step needed more stacks, more
/// work or a deeper stack than `MAX_LIVE_STACKS` / `MAX_EXPANSION_STEPS` /
/// `MAX_PDA_DEPTH` allow. In both cases `state` is left unchanged.
pub fn advance_byte(state: &mut GrammarState, grammar: &CompiledGrammar, b: u8) -> StepResult {
    let next = match step_stacks(&state.stacks, grammar, b) {
        Ok(next) => next,
        Err(StackLimitError) => return StepResult::StackLimitExceeded,
    };
    if next.is_empty() {
        return StepResult::Rejected;
    }
    state.stacks = next;
    state.partial_token_bytes.push(b);
    // Check for completion after consuming the byte.
    state.complete = is_accepting(state, grammar);
    StepResult::Accepted
}

/// Step every stack in `stacks` by `b`; the result is sorted and deduplicated.
///
/// Expansion and the byte filter run together: an alternative whose first
/// symbol is a terminal other than `b` is never materialised, so a rule with
/// many alternatives that start with different bytes (a character class)
/// costs one scan of the alternatives rather than one stack per alternative.
///
/// Fails with [`StackLimitError`] when more than `MAX_LIVE_STACKS` distinct
/// stacks are alive at once (checked as each sibling is created), when the
/// step takes more than `MAX_EXPANSION_STEPS` structural expansion steps (one
/// per `open_top` call), or when a stack grows deeper than `MAX_PDA_DEPTH`.
fn step_stacks(
    stacks: &[Stack],
    grammar: &CompiledGrammar,
    b: u8,
) -> Result<Vec<Stack>, StackLimitError> {
    let mut next: Vec<Stack> = Vec::new();
    let mut pending: Vec<Stack> = Vec::new();
    for stack in stacks {
        match classify_top(stack, grammar) {
            Top::Byte(want) if byte_matches(want, b) => {
                let mut stepped = stack.clone();
                consume_top(&mut stepped, grammar);
                next.push(stepped);
            }
            Top::Open => pending.push(stack.clone()),
            Top::TooDeep => return Err(StackLimitError),
            Top::Byte(_) | Top::Empty | Top::Dead => {}
        }
    }
    bound_live(&mut pending, &mut next)?;

    let mut steps_left = MAX_EXPANSION_STEPS;
    while let Some(mut stack) = pending.pop() {
        loop {
            match classify_top(&stack, grammar) {
                Top::Empty | Top::Dead => break,
                Top::TooDeep => return Err(StackLimitError),
                Top::Byte(want) => {
                    if byte_matches(want, b) {
                        consume_top(&mut stack, grammar);
                        next.push(stack);
                    }
                    break;
                }
                Top::Open => {
                    steps_left = steps_left.checked_sub(1).ok_or(StackLimitError)?;
                    if !open_top(&mut stack, grammar, b, &mut pending, &mut next)? {
                        break;
                    }
                }
            }
        }
        bound_live(&mut pending, &mut next)?;
    }

    next.sort_unstable();
    next.dedup();
    Ok(next)
}

/// Fail when more than [`MAX_LIVE_STACKS`] distinct stacks are alive.
///
/// Identical stacks are merged first, so duplicates produced by alternatives
/// that lead to the same parse never count against the limit.
fn bound_live(pending: &mut Vec<Stack>, next: &mut Vec<Stack>) -> Result<(), StackLimitError> {
    if pending.len() + next.len() <= MAX_LIVE_STACKS {
        return Ok(());
    }
    pending.sort_unstable();
    pending.dedup();
    next.sort_unstable();
    next.dedup();
    if pending.len() + next.len() > MAX_LIVE_STACKS {
        Err(StackLimitError)
    } else {
        Ok(())
    }
}

/// Expand the top frame of `stack` by one structural step so that it can
/// eventually match `b`. The top must be [`Top::Open`].
///
/// An unchosen alternative is replaced by one frame per alternative that could
/// consume `b`: the siblings go to `pending` and the last one is kept in
/// place. Each sibling is counted together with `pending` and `next` as it is
/// created, so a rule with more matching alternatives than `MAX_LIVE_STACKS`
/// fails before all of them are cloned. Returns `Ok(false)` when no alternative
/// can lead to a match of `b`, or the stack names something that does not
/// exist, so the stack is dropped.
fn open_top(
    stack: &mut Stack,
    grammar: &CompiledGrammar,
    b: u8,
    pending: &mut Vec<Stack>,
    next: &mut Vec<Stack>,
) -> Result<bool, StackLimitError> {
    let Some(&StackFrame {
        rule_id,
        alt_idx,
        sym_pos,
    }) = stack.last()
    else {
        return Ok(false);
    };
    let Some(rule) = grammar.rules.get(rule_id) else {
        return Ok(false);
    };
    if rule.alts.is_empty() {
        collapse_exhausted(stack, grammar);
        return Ok(true);
    }
    if alt_idx == UNCHOSEN_ALT {
        let mut kept: Option<usize> = None;
        for (candidate, alt) in rule.alts.iter().enumerate() {
            if let Some(Symbol::Terminal(first)) = alt.first()
                && *first != b
            {
                continue;
            }
            if let Some(earlier) = kept.replace(candidate) {
                let mut sibling = stack.clone();
                if let Some(top) = sibling.last_mut() {
                    top.alt_idx = earlier;
                }
                pending.push(sibling);
                bound_live(pending, next)?;
            }
        }
        let Some(chosen) = kept else {
            return Ok(false);
        };
        if let Some(top) = stack.last_mut() {
            top.alt_idx = chosen;
        }
        return Ok(true);
    }
    let Some(alt) = rule.alts.get(alt_idx) else {
        return Ok(false);
    };
    match alt.get(sym_pos) {
        // Exhausted frame: pop it and move the parent past the non-terminal.
        None => {
            collapse_exhausted(stack, grammar);
            Ok(true)
        }
        Some(Symbol::NonTerminal(rule_id)) => {
            let Some(referenced) = grammar.rules.get(*rule_id) else {
                return Ok(false);
            };
            if referenced.alts.is_empty() {
                // A rule with no alternatives matches the empty string.
                if let Some(top) = stack.last_mut() {
                    top.sym_pos += 1;
                }
            } else {
                stack.push(StackFrame {
                    rule_id: *rule_id,
                    alt_idx: UNCHOSEN_ALT,
                    sym_pos: 0,
                });
            }
            Ok(true)
        }
        Some(Symbol::Terminal(_)) | Some(Symbol::AnyByte) => Ok(false),
    }
}

/// Consume the byte matched by the top symbol: advance its position and pop
/// every frame that this exhausts.
fn consume_top(stack: &mut Stack, grammar: &CompiledGrammar) {
    if let Some(top) = stack.last_mut() {
        top.sym_pos += 1;
    }
    collapse_exhausted(stack, grammar);
}

/// Pop exhausted frames from the top of the stack after a successful byte match.
/// A frame is exhausted when `sym_pos >= alt.len()`.
fn collapse_exhausted(stack: &mut Vec<StackFrame>, grammar: &CompiledGrammar) {
    loop {
        match stack.last() {
            None => break,
            Some(frame) => {
                let Some(rule) = grammar.rules.get(frame.rule_id) else {
                    // Invalid rule id: cannot determine whether this frame is
                    // exhausted. Stop collapsing rather than index out of
                    // bounds; the next operation on this state re-validates
                    // the frame through the same guarded path and rejects.
                    break;
                };
                // No alts: already dead-ended; pop.
                if rule.alts.is_empty() {
                    stack.pop();
                    if let Some(parent) = stack.last_mut() {
                        parent.sym_pos += 1;
                    }
                    continue;
                }
                let Some(alt) = rule.alts.get(frame.alt_idx) else {
                    break;
                };
                if frame.sym_pos < alt.len() {
                    break;
                }
                stack.pop();
                if let Some(parent) = stack.last_mut() {
                    parent.sym_pos += 1;
                }
            }
        }
    }
}

/// Returns true if `state` is in an accepting configuration.
///
/// A state is accepting if at least one live stack can be resolved with zero
/// additional bytes — i.e., all remaining symbols on that stack are
/// *nullable* (can derive the empty string).
fn is_accepting(state: &GrammarState, grammar: &CompiledGrammar) -> bool {
    state
        .stacks
        .iter()
        .any(|stack| stack_is_accepting(stack, grammar))
}

/// Returns true if every symbol still to be matched on `stack` is nullable.
///
/// The stack represents a nested call structure.  The bottom frame contains
/// the root rule; child frames sit on top.  Each non-bottom frame is the
/// expansion of the NonTerminal at `parent.sym_pos`.  Once a child frame
/// completes, the parent advances past that symbol (sym_pos + 1).
///
/// For the purposes of the nullable check:
/// - The **top** (innermost) frame must be nullable from its current `sym_pos`.
/// - Each **non-top** frame must be nullable from `sym_pos + 1` (the current
///   symbol at `sym_pos` is the one being expanded by the frame above it).
/// - A frame whose alternative is not chosen yet is nullable when any
///   alternative of its rule is.
fn stack_is_accepting(stack: &[StackFrame], grammar: &CompiledGrammar) -> bool {
    let n = stack.len();
    for (i, frame) in stack.iter().enumerate() {
        let Some(rule) = grammar.rules.get(frame.rule_id) else {
            return false;
        };
        if rule.alts.is_empty() {
            // A rule with no alts is an empty / epsilon rule — always nullable.
            continue;
        }
        if frame.alt_idx == UNCHOSEN_ALT {
            if !(0..rule.alts.len())
                .any(|alt_idx| remaining_is_nullable(grammar, frame.rule_id, alt_idx, 0))
            {
                return false;
            }
            continue;
        }
        if frame.alt_idx >= rule.alts.len() {
            return false;
        }
        // Non-top frames: the child frame is handling the symbol at sym_pos,
        // so check nullable from sym_pos + 1.
        // Top frame: check nullable from sym_pos itself.
        let check_from = if i == n - 1 {
            frame.sym_pos
        } else {
            frame.sym_pos + 1
        };
        if !remaining_is_nullable(grammar, frame.rule_id, frame.alt_idx, check_from) {
            return false;
        }
    }
    true
}

#[derive(Clone, Copy)]
enum NullableFrame {
    /// Checking whether `grammar.rules[rule_id].alts[alt_idx][pos..]` is nullable.
    Alt {
        rule_id: usize,
        alt_idx: usize,
        pos: usize,
    },
    /// Checking whether any of `grammar.rules[rule_id].alts[alt_idx..]` is nullable.
    Rule { rule_id: usize, alt_idx: usize },
}

std::thread_local! {
    /// Per-thread worklist + cycle-guard for `remaining_is_nullable`, reused
    /// across calls instead of allocated fresh each time.
    ///
    /// `is_accepting` calls this function once per PDA stack frame, and
    /// `is_accepting` itself runs after every accepted byte (`advance_byte`)
    /// and at grammar-state construction (`initial_grammar_state`) — a
    /// per-candidate-token hot path (see `grammar_mask_bench`). A fresh
    /// `Vec`/`HashSet` per call put a guaranteed heap allocation on that path
    /// even for the overwhelmingly common shallow (depth-1, no `NonTerminal`)
    /// case. Reusing thread-local buffers keeps the walk itself unchanged —
    /// still an explicit heap-backed worklist, never native recursion — while
    /// making the allocation one-time-per-thread instead of one-time-per-call:
    /// `clear()` retains capacity, so after the first call reaches a given
    /// depth, subsequent calls (including calls as deep as
    /// `deeply_nested_nullable_chain_accepts_on_bounded_stack`'s
    /// `MAX_PDA_DEPTH`-length chain) reuse that capacity with zero further
    /// allocation. `remaining_is_nullable` does not call itself or otherwise
    /// re-enter this function while a borrow is live, so `borrow_mut` never
    /// contends within one thread; each thread gets its own buffers, so
    /// parallel-beam grammar tracking across threads never contends either.
    static NULLABLE_SCRATCH: std::cell::RefCell<(Vec<NullableFrame>, std::collections::HashSet<usize>)> =
        std::cell::RefCell::new((Vec::new(), std::collections::HashSet::new()));
}

/// Returns true if `grammar.rules[rule_id].alts[alt_idx][pos..]` can derive
/// the empty string, i.e. every remaining symbol is nullable.
///
/// This mirrors the natural mutually-recursive definition (an alt is nullable
/// iff every remaining symbol is nullable; a non-terminal is nullable iff
/// *some* alternative of the rule it names is nullable, or the rule has no
/// alternatives, which matches the empty string) but walks an explicit,
/// heap-allocated worklist (`stack`) instead of native call frames. A cyclic
/// grammar is bounded by the per-path `visited` guard below, same as before;
/// an *acyclic* grammar is bounded by the number of distinct rules it can
/// reference on one path (at most `grammar.rules.len()`), since `visited`
/// forbids revisiting a rule id — either way the frames live on `stack`, not
/// the native stack, so neither shape can overflow it. `MAX_PDA_DEPTH` is a
/// different bound (the live PDA execution stack) and does not apply here.
fn remaining_is_nullable(
    grammar: &CompiledGrammar,
    rule_id: usize,
    alt_idx: usize,
    pos: usize,
) -> bool {
    use NullableFrame as Frame;

    NULLABLE_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let (stack, visited) = &mut *scratch;
        stack.clear();
        visited.clear();
        stack.push(Frame::Alt {
            rule_id,
            alt_idx,
            pos,
        });
        // The boolean result of the frame that just finished, to be consumed
        // by the frame now on top of `stack`.
        let mut pending: Option<bool> = None;

        loop {
            let Some(&frame) = stack.last() else {
                return pending.unwrap_or(true);
            };
            match frame {
                Frame::Alt {
                    rule_id,
                    alt_idx,
                    mut pos,
                } => {
                    let alt = &grammar.rules[rule_id].alts[alt_idx];
                    if let Some(sub) = pending.take() {
                        // Resuming after the NonTerminal at `pos` was checked.
                        if let Symbol::NonTerminal(rid) = alt[pos] {
                            visited.remove(&rid);
                        }
                        if !sub {
                            stack.pop();
                            pending = Some(false);
                            continue;
                        }
                        pos += 1;
                    }
                    match alt.get(pos) {
                        None => {
                            stack.pop();
                            pending = Some(true);
                        }
                        Some(Symbol::Terminal(_)) | Some(Symbol::AnyByte) => {
                            stack.pop();
                            pending = Some(false);
                        }
                        Some(Symbol::NonTerminal(rid)) => {
                            let rid = *rid;
                            // Persist the (possibly advanced) `pos` before descending.
                            if let Some(top) = stack.last_mut() {
                                *top = Frame::Alt {
                                    rule_id,
                                    alt_idx,
                                    pos,
                                };
                            }
                            if !visited.insert(rid) {
                                // Already checking this rule on this path (cycle):
                                // conservatively non-nullable.
                                stack.pop();
                                pending = Some(false);
                            } else {
                                stack.push(Frame::Rule {
                                    rule_id: rid,
                                    alt_idx: 0,
                                });
                            }
                        }
                    }
                }
                Frame::Rule {
                    rule_id,
                    mut alt_idx,
                } => {
                    if let Some(sub) = pending.take() {
                        if sub {
                            stack.pop();
                            pending = Some(true);
                            continue;
                        }
                        alt_idx += 1;
                    }
                    let Some(rule) = grammar.rules.get(rule_id) else {
                        stack.pop();
                        pending = Some(false);
                        continue;
                    };
                    if rule.alts.is_empty() {
                        // A rule with no alternatives matches the empty string.
                        stack.pop();
                        pending = Some(true);
                    } else if alt_idx >= rule.alts.len() {
                        stack.pop();
                        pending = Some(false);
                    } else {
                        if let Some(top) = stack.last_mut() {
                            *top = Frame::Rule { rule_id, alt_idx };
                        }
                        stack.push(Frame::Alt {
                            rule_id,
                            alt_idx,
                            pos: 0,
                        });
                    }
                }
            }
        }
    })
}

/// Simulate advancing the PDA from `state` by consuming all bytes of `token`.
///
/// Returns `SimResult::Accept` if `token` is fully accepted (all bytes consumed
/// and the resulting state is valid), `SimResult::ContextDependent` if some
/// bytes were consumed but the automaton is mid-grammar-boundary, and
/// `SimResult::Reject` if any byte was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum SimResult {
    /// All bytes accepted; resulting state is valid.
    Accept,
    /// Some bytes consumed but not all; context-dependent token.
    ContextDependent,
    /// Byte rejected.
    Reject,
    /// A byte step exceeded the matcher's stack limits (see
    /// [`StepResult::StackLimitExceeded`]), so the token cannot be classified.
    /// This is not a rejection by the grammar.
    StackLimitExceeded,
}

/// Simulate consuming all bytes of `token` from state `start`.
/// Does not mutate `start`; returns a classification.
pub fn simulate_token(
    start: &GrammarState,
    grammar: &CompiledGrammar,
    token: &[u8],
) -> (SimResult, GrammarState) {
    let mut state = start.clone();
    for (i, &b) in token.iter().enumerate() {
        match advance_byte(&mut state, grammar, b) {
            StepResult::Accepted => {}
            StepResult::Rejected => {
                if i > 0 {
                    return (SimResult::ContextDependent, state);
                }
                return (SimResult::Reject, state);
            }
            StepResult::StackLimitExceeded => return (SimResult::StackLimitExceeded, state),
        }
    }
    (SimResult::Accept, state)
}

// ---------------------------------------------------------------------------
// Grammar builders for use by json_schema.rs and gbnf.rs
// ---------------------------------------------------------------------------

/// Error from a `GrammarBuilder` operation.
#[derive(Debug, Clone, PartialEq)]
pub struct BuilderError(pub String);

impl std::fmt::Display for BuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "grammar builder error: {}", self.0)
    }
}
impl std::error::Error for BuilderError {}

/// Builder for assembling a `CompiledGrammar`.
pub struct GrammarBuilder {
    rules: Vec<Rule>,
    name_to_id: HashMap<String, usize>,
}

impl GrammarBuilder {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    /// Reserve a rule slot by name and return its id.
    /// If the name already exists, return its id without creating a new slot.
    pub fn reserve(&mut self, name: &str) -> usize {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = self.rules.len();
        self.rules.push(Rule {
            name: name.to_string(),
            alts: Vec::new(),
        });
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    /// Add alternatives to an already-reserved rule.
    ///
    /// Returns an error rather than panicking when `id` was never returned by
    /// `reserve` on this builder (e.g. a foreign or stale id).
    pub fn set_alts(&mut self, id: usize, alts: Vec<Alt>) -> Result<(), BuilderError> {
        let Some(rule) = self.rules.get_mut(id) else {
            return Err(BuilderError(format!(
                "set_alts: rule id {id} was not reserved on this builder ({} rule(s) reserved)",
                self.rules.len()
            )));
        };
        rule.alts = alts;
        Ok(())
    }

    /// Reserve and immediately set alternatives.
    ///
    /// `id` is always freshly reserved above, so this can never hit the
    /// unreserved-id case `set_alts` guards against.
    pub fn add_rule(&mut self, name: &str, alts: Vec<Alt>) -> usize {
        let id = self.reserve(name);
        self.rules[id].alts = alts;
        id
    }

    /// Look up the id of a previously reserved rule.
    pub fn rule_id(&self, name: &str) -> Option<usize> {
        self.name_to_id.get(name).copied()
    }

    /// Consume the builder and produce a `CompiledGrammar`.
    ///
    /// The grammar is not validated: a rule without alternatives, the root
    /// included, matches the empty string.
    pub fn build(self) -> CompiledGrammar {
        CompiledGrammar { rules: self.rules }
    }
}

impl Default for GrammarBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a grammar that matches exactly `b"ab"`.
    fn ab_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        b.add_rule(
            "root",
            vec![vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')]],
        );
        b.build()
    }

    /// Grammar: root = 'a' | 'b'
    fn or_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        b.add_rule(
            "root",
            vec![vec![Symbol::Terminal(b'a')], vec![Symbol::Terminal(b'b')]],
        );
        b.build()
    }

    /// Grammar: root = digit+  where digit = '0' | '1' | ... | '9'
    fn digits_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        let digit_id = b.reserve("digit");
        let digit_alts: Vec<Alt> = (b'0'..=b'9')
            .map(|byte| vec![Symbol::Terminal(byte)])
            .collect();
        b.set_alts(digit_id, digit_alts).unwrap();

        // root = digit digit_rest
        // digit_rest = digit digit_rest | ε  (implemented as digit_rest = [empty alt])
        let rest_id = b.reserve("digit_rest");
        b.set_alts(
            rest_id,
            vec![
                vec![Symbol::NonTerminal(digit_id), Symbol::NonTerminal(rest_id)],
                vec![], // epsilon
            ],
        )
        .unwrap();

        let root_id = b.reserve("root");
        b.set_alts(
            root_id,
            vec![vec![
                Symbol::NonTerminal(digit_id),
                Symbol::NonTerminal(rest_id),
            ]],
        )
        .unwrap();
        // Ensure root is at index 0.
        let mut grammar = b.build();
        // Swap root to position 0.
        let root_pos = grammar.rules.iter().position(|r| r.name == "root").unwrap();
        grammar.rules.swap(0, root_pos);
        // Fix up any NonTerminal references after the swap.
        let orig_root_id = root_pos;
        let swapped_to_id = 0usize;
        if orig_root_id != 0 {
            for rule in &mut grammar.rules {
                for alt in &mut rule.alts {
                    for sym in alt.iter_mut() {
                        if let Symbol::NonTerminal(rid) = sym {
                            if *rid == orig_root_id {
                                *rid = swapped_to_id;
                            } else if *rid == 0 {
                                *rid = orig_root_id;
                            }
                        }
                    }
                }
            }
        }
        grammar
    }

    /// Minimal grammar isolating the trailing-comma class, free of the
    /// JSON-schema compiler:
    ///   root = '[' body ']'
    ///   body = elem tail | ε
    ///   tail = ',' elem tail | ε
    ///   elem = 'x'
    /// A trailing comma (`[x,]`) must reject: once the `,` is consumed only the
    /// parse that took `tail`'s `',' elem tail` alternative is alive, and it
    /// needs an `elem` next, not `]`. The ε arm of `tail` belongs to another
    /// stack that never saw the `,`.  The nullable `body`/`tail` ε arms must
    /// still let valid forms through (refs #353).
    fn comma_list_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root"); // index 0
        let body_id = b.reserve("body");
        let tail_id = b.reserve("tail");
        let elem_id = b.reserve("elem");
        b.set_alts(elem_id, vec![vec![Symbol::Terminal(b'x')]])
            .unwrap();
        b.set_alts(
            tail_id,
            vec![
                vec![
                    Symbol::Terminal(b','),
                    Symbol::NonTerminal(elem_id),
                    Symbol::NonTerminal(tail_id),
                ],
                vec![], // epsilon
            ],
        )
        .unwrap();
        b.set_alts(
            body_id,
            vec![
                vec![Symbol::NonTerminal(elem_id), Symbol::NonTerminal(tail_id)],
                vec![], // epsilon
            ],
        )
        .unwrap();
        b.set_alts(
            root_id,
            vec![vec![
                Symbol::Terminal(b'['),
                Symbol::NonTerminal(body_id),
                Symbol::Terminal(b']'),
            ]],
        )
        .unwrap();
        b.build()
    }

    fn accepts_str(g: &CompiledGrammar, input: &[u8]) -> bool {
        let state = GrammarState::initial();
        let (result, final_state) = simulate_token(&state, g, input);
        result == SimResult::Accept && final_state.is_complete()
    }

    #[test]
    fn comma_list_rejects_trailing_comma() {
        let g = comma_list_grammar();
        assert!(accepts_str(&g, b"[]")); // nullable body reaches ε, no byte consumed
        assert!(accepts_str(&g, b"[x]")); // clean tail reaches ε
        assert!(accepts_str(&g, b"[x,x]")); // nested tail
        assert!(!accepts_str(&g, b"[x,]")); // trailing comma: the ε stack never saw the ','
        assert!(!accepts_str(&g, b"[x,x,]")); // same, one level deeper
        assert!(!accepts_str(&g, b"[,]")); // leading comma
    }

    /// Grammar: root = "ab" | "ac" (two alternatives with one shared first byte).
    fn shared_first_byte_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        b.add_rule(
            "root",
            vec![
                vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')],
                vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'c')],
            ],
        );
        b.build()
    }

    #[test]
    fn alternatives_sharing_a_first_byte_both_stay_alive() {
        let g = shared_first_byte_grammar();
        for follower in [b'b', b'c'] {
            let mut state = GrammarState::initial();
            assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
            assert_eq!(
                state.stacks.len(),
                2,
                "one stack per alternative that took 'a'"
            );
            assert!(
                state.stacks.windows(2).all(|pair| pair[0] < pair[1]),
                "stacks must be sorted and free of duplicates"
            );
            assert!(!state.is_complete());
            assert_eq!(
                advance_byte(&mut state, &g, follower),
                StepResult::Accepted,
                "follower {:?}",
                follower as char
            );
            assert!(state.is_complete());
        }

        let mut state = GrammarState::initial();
        assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
        let before = state.clone();
        assert_eq!(advance_byte(&mut state, &g, b'd'), StepResult::Rejected);
        assert_eq!(state.stacks, before.stacks);
    }

    /// `root ::= "ab" | "a" tail`, `tail ::= "x" | ""`. After "a" the first
    /// stack (alternative 0) still needs "b" while the second can finish
    /// through the nullable `tail`, so completeness has to look past the first
    /// stack.
    #[test]
    fn completion_is_judged_over_every_live_stack() {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let tail_id = b.reserve("tail");
        b.set_alts(tail_id, vec![vec![Symbol::Terminal(b'x')], vec![]])
            .unwrap();
        b.set_alts(
            root_id,
            vec![
                vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')],
                vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(tail_id)],
            ],
        )
        .unwrap();
        let g = b.build();

        let mut state = GrammarState::initial();
        assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
        assert_eq!(state.stacks.len(), 2);
        assert!(state.is_complete(), "\"a\" completes through `tail`");
        assert!(accepts_str(&g, b"ab"));
        assert!(accepts_str(&g, b"ax"));
        assert!(!accepts_str(&g, b"abx"));
    }

    /// `root ::= x root | ""`, `x ::= "a" | "a"`. The two `x` alternatives lead
    /// to the same parse, so without merging identical stacks the live set
    /// would double on every byte.
    #[test]
    fn identical_stacks_are_merged_after_every_byte() {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let x_id = b.reserve("x");
        b.set_alts(
            x_id,
            vec![vec![Symbol::Terminal(b'a')], vec![Symbol::Terminal(b'a')]],
        )
        .unwrap();
        b.set_alts(
            root_id,
            vec![
                vec![Symbol::NonTerminal(x_id), Symbol::NonTerminal(root_id)],
                vec![],
            ],
        )
        .unwrap();
        let g = b.build();

        let mut state = GrammarState::initial();
        for i in 0..40 {
            assert_eq!(
                advance_byte(&mut state, &g, b'a'),
                StepResult::Accepted,
                "byte {i}"
            );
            assert_eq!(
                state.stacks.len(),
                1,
                "byte {i}: identical stacks must collapse into one"
            );
        }
        assert!(state.is_complete());
    }

    /// `n` alternatives that all start with "a" and differ only in their index,
    /// so each one is a distinct live stack after that byte.
    fn many_distinct_alternatives_grammar(n: usize) -> CompiledGrammar {
        CompiledGrammar {
            rules: vec![Rule {
                name: "root".to_string(),
                alts: vec![vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')]; n],
            }],
        }
    }

    #[test]
    fn live_stack_limit_is_inclusive_and_reported_distinctly() {
        let at_limit = many_distinct_alternatives_grammar(MAX_LIVE_STACKS);
        let mut state = GrammarState::initial();
        assert_eq!(
            advance_byte(&mut state, &at_limit, b'a'),
            StepResult::Accepted
        );
        assert_eq!(state.stacks.len(), MAX_LIVE_STACKS);

        let past_limit = many_distinct_alternatives_grammar(MAX_LIVE_STACKS + 1);
        let mut state = GrammarState::initial();
        let before = state.clone();
        assert_eq!(
            advance_byte(&mut state, &past_limit, b'a'),
            StepResult::StackLimitExceeded
        );
        assert_eq!(state.stacks, before.stacks);
        assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
        assert_eq!(state.complete, before.complete);

        // The token-level classification keeps the same distinction instead of
        // folding the limit into a rejection.
        let (result, _) = simulate_token(&GrammarState::initial(), &past_limit, b"ab");
        assert_eq!(result, SimResult::StackLimitExceeded);

        // A byte that no alternative takes is still an ordinary rejection.
        let mut state = GrammarState::initial();
        assert_eq!(
            advance_byte(&mut state, &past_limit, b'z'),
            StepResult::Rejected
        );
    }

    /// `root ::= root "a" | "b"` is left-recursive with a base case: every
    /// level of expansion yields one more distinct stack, so the live-stack
    /// limit reports it instead of the byte looking rejected.
    #[test]
    fn left_recursive_grammar_with_a_base_case_reports_the_stack_limit() {
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "root".to_string(),
                alts: vec![
                    vec![Symbol::NonTerminal(0), Symbol::Terminal(b'a')],
                    vec![Symbol::Terminal(b'b')],
                ],
            }],
        };
        let mut state = GrammarState::initial();
        assert_eq!(
            advance_byte(&mut state, &grammar, b'b'),
            StepResult::StackLimitExceeded
        );
    }

    /// `root ::= "a" E` with `E` a rule that has no alternatives (it matches
    /// the empty string): the completion analysis must treat `E` as nullable,
    /// as execution does when it steps past the reference.
    #[test]
    fn referenced_rule_without_alternatives_is_nullable() {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let empty_id = b.reserve("E");
        b.set_alts(
            root_id,
            vec![vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(empty_id)]],
        )
        .unwrap();
        let g = b.build();

        let mut state = GrammarState::initial();
        assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
        assert!(
            state.is_complete(),
            "\"a\" completes through the empty rule"
        );
        assert!(accepts_str(&g, b"a"));
        assert!(!accepts_str(&g, b"ab"));

        // The same rule reached before any byte: `root ::= E`.
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let empty_id = b.reserve("E");
        b.set_alts(root_id, vec![vec![Symbol::NonTerminal(empty_id)]])
            .unwrap();
        assert!(initial_grammar_state(&b.build()).is_complete());
    }

    /// `root ::= "a" root | ""` accepts every run of `a` bytes, each one a
    /// level deeper on the stack. The byte that needs the stack past
    /// `MAX_PDA_DEPTH` is reported as a limit, not as a rejection, and leaves
    /// the state as it was.
    #[test]
    fn recursion_past_the_depth_cap_reports_the_stack_limit() {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        b.set_alts(
            root_id,
            vec![
                vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(root_id)],
                vec![],
            ],
        )
        .unwrap();
        let g = b.build();

        let mut state = GrammarState::initial();
        for i in 0..MAX_PDA_DEPTH {
            assert_eq!(
                advance_byte(&mut state, &g, b'a'),
                StepResult::Accepted,
                "byte {i} is inside the depth cap"
            );
        }
        assert!(state.is_complete());
        let before = state.clone();
        assert_eq!(
            advance_byte(&mut state, &g, b'a'),
            StepResult::StackLimitExceeded
        );
        assert_eq!(state.stacks, before.stacks);
        assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
    }

    /// `root ::= E{n} "a"` with `E` a rule without alternatives: one stack, one
    /// structural step per `E`, no sibling and no depth. Choosing root's
    /// alternative is one more step, so `MAX_EXPANSION_STEPS - 1` references
    /// fit the budget and `MAX_EXPANSION_STEPS` do not.
    fn empty_rule_run_grammar(references: usize) -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let empty_id = b.reserve("E");
        let mut alt = vec![Symbol::NonTerminal(empty_id); references];
        alt.push(Symbol::Terminal(b'a'));
        b.set_alts(root_id, vec![alt]).unwrap();
        b.build()
    }

    /// The expansion budget is charged for every structural step a stack
    /// takes, not once per stack taken off the pending list: a single stack
    /// walking a long run of nullable references pays for each of them.
    #[test]
    fn expansion_budget_is_charged_per_structural_step() {
        let within = empty_rule_run_grammar(MAX_EXPANSION_STEPS - 1);
        let mut state = GrammarState::initial();
        assert_eq!(
            advance_byte(&mut state, &within, b'a'),
            StepResult::Accepted
        );

        let over = empty_rule_run_grammar(MAX_EXPANSION_STEPS);
        let mut state = GrammarState::initial();
        assert_eq!(
            advance_byte(&mut state, &over, b'a'),
            StepResult::StackLimitExceeded
        );
    }

    /// Creating siblings counts against `MAX_LIVE_STACKS` as they are made: a
    /// rule with far more matching alternatives than the limit fails after the
    /// first `MAX_LIVE_STACKS + 1` clones instead of cloning all of them first.
    #[test]
    fn sibling_generation_stops_at_the_live_stack_limit() {
        let g = many_distinct_alternatives_grammar(4 * MAX_LIVE_STACKS);
        let mut stack = vec![StackFrame {
            rule_id: 0,
            alt_idx: UNCHOSEN_ALT,
            sym_pos: 0,
        }];
        let mut pending = Vec::new();
        let mut next = Vec::new();
        assert_eq!(
            open_top(&mut stack, &g, b'a', &mut pending, &mut next),
            Err(StackLimitError)
        );
        assert_eq!(
            pending.len(),
            MAX_LIVE_STACKS + 1,
            "sibling creation must stop as soon as the limit is passed"
        );
    }

    #[test]
    fn ab_grammar_accepts_ab() {
        let g = ab_grammar();
        let mut state = GrammarState::initial();
        assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
        assert!(!state.is_complete()); // not done yet
        assert_eq!(advance_byte(&mut state, &g, b'b'), StepResult::Accepted);
        assert!(state.is_complete());
    }

    #[test]
    fn ab_grammar_rejects_ba() {
        let g = ab_grammar();
        let mut state = GrammarState::initial();
        assert_eq!(advance_byte(&mut state, &g, b'b'), StepResult::Rejected);
    }

    #[test]
    fn ab_grammar_rejects_partial_a_then_wrong() {
        let g = ab_grammar();
        let mut state = GrammarState::initial();
        advance_byte(&mut state, &g, b'a');
        assert_eq!(advance_byte(&mut state, &g, b'x'), StepResult::Rejected);
    }

    #[test]
    fn rejected_byte_leaves_state_intact_and_resumes() {
        // A rejected byte must not corrupt the matcher: the consumed prefix
        // stays committed and the correct continuation still completes. This
        // locks the rollback-on-reject contract that `advance_byte` relies on
        // (`step_stacks` builds the next stack set separately and `advance_byte`
        // only commits it on success, so no outer snapshot is needed).
        //
        // The grammar must be *nested* so the rejecting byte is checked
        // against a stack with a child frame on it:
        //   root  ::= "a" child
        //   child ::= "bc"
        // Feeding `a`,`b` descends into `child` (frame pushed, one byte
        // consumed). The wrong byte at child's second position matches no
        // stack, and the state must keep that child frame so the correct `c`
        // can still complete. A flat grammar never exercises this and would
        // make the test vacuous.
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let child_id = b.reserve("child");
        b.set_alts(
            child_id,
            vec![vec![Symbol::Terminal(b'b'), Symbol::Terminal(b'c')]],
        )
        .unwrap();
        b.set_alts(
            root_id,
            vec![vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(child_id)]],
        )
        .unwrap();
        let g = b.build();

        let mut state = GrammarState::initial();
        assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
        assert_eq!(advance_byte(&mut state, &g, b'b'), StepResult::Accepted);
        assert_eq!(advance_byte(&mut state, &g, b'x'), StepResult::Rejected);
        assert_eq!(advance_byte(&mut state, &g, b'c'), StepResult::Accepted);
        assert!(state.complete);
    }

    #[test]
    fn or_grammar_accepts_a_or_b() {
        let g = or_grammar();
        let mut s = GrammarState::initial();
        assert_eq!(advance_byte(&mut s, &g, b'a'), StepResult::Accepted);

        let mut s2 = GrammarState::initial();
        assert_eq!(advance_byte(&mut s2, &g, b'b'), StepResult::Accepted);
    }

    #[test]
    fn or_grammar_rejects_c() {
        let g = or_grammar();
        let mut s = GrammarState::initial();
        assert_eq!(advance_byte(&mut s, &g, b'c'), StepResult::Rejected);
    }

    #[test]
    fn dead_child_alternative_does_not_block_a_sibling() {
        let mut builder = GrammarBuilder::new();
        let root_id = builder.reserve("root");
        let dead_id = builder.reserve("dead");
        builder
            .set_alts(dead_id, vec![vec![Symbol::Terminal(b'y')]])
            .unwrap();
        builder
            .set_alts(
                root_id,
                vec![
                    vec![Symbol::NonTerminal(dead_id)],
                    vec![Symbol::Terminal(b'x')],
                ],
            )
            .unwrap();
        let grammar = builder.build();
        let mut state = GrammarState::initial();

        assert_eq!(
            advance_byte(&mut state, &grammar, b'x'),
            StepResult::Accepted
        );
        assert!(state.complete);
    }

    /// A production-valid nested grammar must reach a root sibling without
    /// growing the native call stack while it exhausts uncommitted parents.
    ///
    /// The 64 KiB native thread stack makes a recursive parent walk overflow before
    /// it can accept `x`; the iterative walk completes within the same bound.
    #[test]
    fn deeply_nested_parent_fallback_accepts_on_bounded_stack() {
        let depth = MAX_PDA_DEPTH / 2;
        let mut rules = Vec::with_capacity(depth);
        rules.push(Rule {
            name: "root".to_string(),
            alts: vec![vec![Symbol::NonTerminal(1)], vec![Symbol::Terminal(b'x')]],
        });
        for rule_id in 1..depth - 1 {
            rules.push(Rule {
                name: String::new(),
                alts: vec![vec![Symbol::NonTerminal(rule_id + 1)]],
            });
        }
        rules.push(Rule {
            name: String::new(),
            alts: vec![vec![Symbol::Terminal(b'y')]],
        });
        let grammar = CompiledGrammar { rules };

        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(move || {
                let mut state = GrammarState::initial();
                assert_eq!(
                    advance_byte(&mut state, &grammar, b'x'),
                    StepResult::Accepted
                );
                assert!(state.complete);
            })
            .expect("bounded-stack regression thread spawns")
            .join()
            .expect("iterative fallback must not overflow the bounded stack");
    }

    /// A long *acyclic* chain of distinct nullable wrapper rules has no cycle
    /// for `remaining_is_nullable`'s `visited` guard to catch, so unlike the
    /// cyclic case its only historical bound was call-frame depth.
    /// `is_accepting` runs on `initial_grammar_state`, before any byte is
    /// consumed — a single-frame state referencing the head of such a chain
    /// must not overflow the native stack even though `state.stacks.len()`
    /// never leaves 1 (the nullability walk descends the *static* rule graph,
    /// not the PDA execution stack `MAX_PDA_DEPTH` bounds).
    #[test]
    fn deeply_nested_nullable_chain_accepts_on_bounded_stack() {
        let depth = MAX_PDA_DEPTH;
        let mut rules = Vec::with_capacity(depth);
        for rule_id in 0..depth - 1 {
            rules.push(Rule {
                name: String::new(),
                alts: vec![vec![Symbol::NonTerminal(rule_id + 1)]],
            });
        }
        // The chain's tail is epsilon, so the whole chain is nullable.
        rules.push(Rule {
            name: String::new(),
            alts: vec![vec![]],
        });
        let grammar = CompiledGrammar { rules };

        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(move || {
                let state = initial_grammar_state(&grammar);
                assert!(state.is_complete());
            })
            .expect("bounded-stack regression thread spawns")
            .join()
            .expect("iterative nullability walk must not overflow the bounded stack");
    }

    fn nested_terminal_grammar(depth: usize, terminal: u8) -> CompiledGrammar {
        let mut rules = Vec::with_capacity(depth);
        for rule_id in 0..depth - 1 {
            rules.push(Rule {
                name: String::new(),
                alts: vec![vec![Symbol::NonTerminal(rule_id + 1)]],
            });
        }
        rules.push(Rule {
            name: String::new(),
            alts: vec![vec![Symbol::Terminal(terminal)]],
        });
        CompiledGrammar { rules }
    }

    #[test]
    fn nesting_depth_limit_matches_recursive_boundary() {
        let at_limit = nested_terminal_grammar(MAX_PDA_DEPTH, b'x');
        let mut state = GrammarState::initial();
        assert_eq!(
            advance_byte(&mut state, &at_limit, b'x'),
            StepResult::Accepted
        );

        // One rule deeper needs a stack past the cap: the matcher reports its
        // limit instead of rejecting a byte the grammar accepts.
        let past_limit = nested_terminal_grammar(MAX_PDA_DEPTH + 1, b'x');
        let mut state = GrammarState::initial();
        assert_eq!(
            advance_byte(&mut state, &past_limit, b'x'),
            StepResult::StackLimitExceeded
        );
    }

    /// Grammar: root = "a" nonterm | "x" ; nonterm = "cd"
    /// Root reserved first so it lands at index 0.
    fn leading_terminal_then_nt_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let nt_id = b.reserve("nonterm");
        b.set_alts(
            nt_id,
            vec![vec![Symbol::Terminal(b'c'), Symbol::Terminal(b'd')]],
        )
        .unwrap();
        b.set_alts(
            root_id,
            vec![
                vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(nt_id)],
                vec![Symbol::Terminal(b'x')],
            ],
        )
        .unwrap();
        b.build()
    }

    #[test]
    fn leading_terminal_then_nt_accepts_valid() {
        let g = leading_terminal_then_nt_grammar();
        // "acd" via alt-0, "x" via alt-1 must both still be accepted.
        let s0 = GrammarState::initial();
        let (r_acd, _) = simulate_token(&s0, &g, b"acd");
        assert_eq!(r_acd, SimResult::Accept);
        let s1 = GrammarState::initial();
        let (r_x, _) = simulate_token(&s1, &g, b"x");
        assert_eq!(r_x, SimResult::Accept);
    }

    #[test]
    fn simulate_token_full_match() {
        let g = ab_grammar();
        let state = GrammarState::initial();
        let (result, _) = simulate_token(&state, &g, b"ab");
        assert_eq!(result, SimResult::Accept);
    }

    #[test]
    fn simulate_token_reject() {
        let g = ab_grammar();
        let state = GrammarState::initial();
        let (result, _) = simulate_token(&state, &g, b"ba");
        assert_eq!(result, SimResult::Reject);
    }

    #[test]
    fn simulate_token_partial_is_context_dependent() {
        let g = ab_grammar();
        let state = GrammarState::initial();
        // Token "ax" — first byte 'a' accepted, second 'x' rejected mid-token.
        let (result, _) = simulate_token(&state, &g, b"ax");
        assert_eq!(result, SimResult::ContextDependent);
    }

    #[test]
    fn state_partial_bytes_recorded() {
        let g = ab_grammar();
        let mut state = GrammarState::initial();
        advance_byte(&mut state, &g, b'a');
        assert_eq!(state.partial_token_bytes, vec![b'a']);
        advance_byte(&mut state, &g, b'b');
        assert_eq!(state.partial_token_bytes, vec![b'a', b'b']);
    }

    #[test]
    fn any_byte_matches_any_value() {
        let mut b = GrammarBuilder::new();
        b.add_rule("root", vec![vec![Symbol::AnyByte]]);
        let g = b.build();
        for byte in [b'a', b'z', b'0', b'\n', 0xffu8] {
            let mut s = GrammarState::initial();
            assert_eq!(advance_byte(&mut s, &g, byte), StepResult::Accepted);
            assert!(s.is_complete());
        }
    }

    #[test]
    fn digits_grammar_accepts_single_digit() {
        let g = digits_grammar();
        let state = GrammarState::initial();
        let (result, _) = simulate_token(&state, &g, b"5");
        assert_eq!(result, SimResult::Accept);
    }

    #[test]
    fn digits_grammar_accepts_multi_digit() {
        let g = digits_grammar();
        let state = GrammarState::initial();
        let (result, final_state) = simulate_token(&state, &g, b"123");
        assert_eq!(result, SimResult::Accept);
        assert!(final_state.is_complete());
    }

    #[test]
    fn digits_grammar_rejects_letter() {
        let g = digits_grammar();
        let state = GrammarState::initial();
        let (result, _) = simulate_token(&state, &g, b"abc");
        assert_eq!(result, SimResult::Reject);
    }

    #[test]
    fn grammar_builder_reserve_idempotent() {
        let mut builder = GrammarBuilder::new();
        let id1 = builder.reserve("foo");
        let id2 = builder.reserve("foo");
        assert_eq!(id1, id2);
    }

    #[test]
    fn missing_root_rule_id_rejects_without_panicking() {
        let grammar = CompiledGrammar { rules: Vec::new() };
        let mut state = GrammarState::initial();
        let before = state.clone();

        assert_eq!(
            advance_byte(&mut state, &grammar, b'x'),
            StepResult::Rejected
        );
        assert_eq!(state.stacks, before.stacks);
        assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
        assert_eq!(state.complete, before.complete);
    }

    #[test]
    fn out_of_range_state_rule_id_rejects_without_panicking() {
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "root".to_string(),
                alts: vec![vec![Symbol::Terminal(b'x')]],
            }],
        };
        let mut state = GrammarState {
            stacks: vec![vec![StackFrame {
                rule_id: 1,
                alt_idx: 0,
                sym_pos: 0,
            }]],
            partial_token_bytes: Vec::new(),
            complete: false,
        };
        let before = state.clone();

        assert_eq!(
            advance_byte(&mut state, &grammar, b'x'),
            StepResult::Rejected
        );
        assert_eq!(state.stacks, before.stacks);
        assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
        assert_eq!(state.complete, before.complete);
    }

    #[test]
    fn out_of_range_non_terminal_rule_id_rejects_without_panicking() {
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "root".to_string(),
                alts: vec![vec![Symbol::NonTerminal(1)]],
            }],
        };
        let mut state = GrammarState::initial();
        let before = state.clone();

        assert_eq!(
            advance_byte(&mut state, &grammar, b'x'),
            StepResult::Rejected
        );
        assert_eq!(state.stacks, before.stacks);
        assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
        assert_eq!(state.complete, before.complete);
    }

    #[test]
    fn set_alts_out_of_range_id_returns_err() {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        assert_eq!(root_id, 0);

        // No rule was ever reserved at id 5: out of range for a 1-rule builder.
        let err = b
            .set_alts(5, vec![vec![Symbol::Terminal(b'x')]])
            .expect_err("set_alts on an unreserved id must return Err, not panic");
        assert!(err.0.contains('5'), "error should name the bad id: {err:?}");
    }

    #[test]
    fn set_alts_in_range_id_succeeds() {
        let mut b = GrammarBuilder::new();
        let id = b.reserve("root");
        b.set_alts(id, vec![vec![Symbol::Terminal(b'x')]])
            .expect("set_alts on a freshly reserved id must succeed");
        let g = b.build();
        assert!(accepts_str(&g, b"x"));
    }

    #[test]
    fn out_of_range_alt_idx_rejects_without_panicking() {
        // A directly-constructed StackFrame (public fields) can carry an
        // alt_idx past the rule's alternative count even though the rule id
        // itself is valid and non-empty. The main advance loop must reject
        // this the same way it already rejects an out-of-range rule id,
        // rather than indexing `rule.alts[alt_idx]` unchecked.
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "root".to_string(),
                alts: vec![vec![Symbol::Terminal(b'x')]], // exactly one alt
            }],
        };
        let mut state = GrammarState {
            stacks: vec![vec![StackFrame {
                rule_id: 0,
                alt_idx: 7, // no alt at index 7
                sym_pos: 0,
            }]],
            partial_token_bytes: Vec::new(),
            complete: false,
        };
        let before = state.clone();

        assert_eq!(
            advance_byte(&mut state, &grammar, b'x'),
            StepResult::Rejected
        );
        assert_eq!(state.stacks, before.stacks);
        assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
        assert_eq!(state.complete, before.complete);
    }

    #[test]
    fn dangling_ancestor_rule_id_rejects_on_mismatch_without_panicking() {
        // A non-top (ancestor) frame can carry a rule id that no longer
        // exists in the grammar — the state a dangling NonTerminal reference
        // would leave behind. The top frame is valid but its only alt
        // mismatches the incoming byte, so the stack is dropped without ever
        // being popped into the invalid ancestor. The byte must reject, not
        // index `grammar.rules[rule_id]` unchecked.
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "child".to_string(),
                alts: vec![vec![Symbol::Terminal(b'z')]],
            }],
        };
        let mut state = GrammarState {
            stacks: vec![vec![
                StackFrame {
                    rule_id: 999, // dangling: no such rule
                    alt_idx: 0,
                    sym_pos: 0,
                },
                StackFrame {
                    rule_id: 0, // valid, but its only alt won't match b'x'
                    alt_idx: 0,
                    sym_pos: 0,
                },
            ]],
            partial_token_bytes: Vec::new(),
            complete: false,
        };
        let before = state.clone();

        assert_eq!(
            advance_byte(&mut state, &grammar, b'x'),
            StepResult::Rejected
        );
        assert_eq!(state.stacks, before.stacks);
        assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
        assert_eq!(state.complete, before.complete);
    }

    /// `r ::= "" | "a"` (epsilon first), `root ::= r "b"`.
    fn epsilon_first_alt_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let r_id = b.reserve("r");
        b.set_alts(r_id, vec![vec![], vec![Symbol::Terminal(b'a')]])
            .unwrap();
        b.set_alts(
            root_id,
            vec![vec![Symbol::NonTerminal(r_id), Symbol::Terminal(b'b')]],
        )
        .unwrap();
        b.build()
    }

    /// Same shape, epsilon last: `r ::= "a" | ""`. Control proving the fix
    /// does not depend on alternative order (this direction already worked).
    fn epsilon_last_alt_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let r_id = b.reserve("r");
        b.set_alts(r_id, vec![vec![Symbol::Terminal(b'a')], vec![]])
            .unwrap();
        b.set_alts(
            root_id,
            vec![vec![Symbol::NonTerminal(r_id), Symbol::Terminal(b'b')]],
        )
        .unwrap();
        b.build()
    }

    /// `r ::= "" | "a"`, `root ::= r "a" "c"`: accepting "aac" needs both of
    /// `r`'s choices to stay open across the first byte (#322).
    fn epsilon_first_multi_byte_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let r_id = b.reserve("r");
        b.set_alts(r_id, vec![vec![], vec![Symbol::Terminal(b'a')]])
            .unwrap();
        b.set_alts(
            root_id,
            vec![vec![
                Symbol::NonTerminal(r_id),
                Symbol::Terminal(b'a'),
                Symbol::Terminal(b'c'),
            ]],
        )
        .unwrap();
        b.build()
    }

    /// Deeply nested `r_i ::= "" | (r_{i+1} "a") | (r_{i+1} "b")`, base case
    /// `r_depth ::= "" | "a" | "b"`; each level doubles fresh re-exploration
    /// of the next, so exploring every parse is ~2^depth work.
    fn nested_nullable_branch_grammar(depth: usize) -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let levels: Vec<usize> = (0..=depth).map(|i| b.reserve(&format!("r{i}"))).collect();
        b.set_alts(
            levels[depth],
            vec![
                vec![],
                vec![Symbol::Terminal(b'a')],
                vec![Symbol::Terminal(b'b')],
            ],
        )
        .unwrap();
        for i in (0..depth).rev() {
            let next = levels[i + 1];
            b.set_alts(
                levels[i],
                vec![
                    vec![],
                    vec![Symbol::NonTerminal(next), Symbol::Terminal(b'a')],
                    vec![Symbol::NonTerminal(next), Symbol::Terminal(b'b')],
                ],
            )
            .unwrap();
        }
        b.set_alts(
            root_id,
            vec![vec![Symbol::NonTerminal(levels[0]), Symbol::Terminal(b'z')]],
        )
        .unwrap();
        b.build()
    }

    /// #322: an epsilon-first alternative must not permanently commit. `"ab"`
    /// falls back to `r`'s `"a"` once `root`'s `"b"` mismatches; `"b"` keeps
    /// working via the epsilon choice; `"c"` / `"aab"` still reject.
    #[test]
    fn epsilon_first_alternative_is_retried_on_later_mismatch() {
        let g = epsilon_first_alt_grammar();
        assert!(accepts_str(&g, b"ab"), "\"ab\" must accept via r ::= \"a\"");
        assert!(accepts_str(&g, b"b"), "\"b\" must accept via r ::= \"\"");
        assert!(!accepts_str(&g, b"c"), "\"c\" matches neither alternative");
        assert!(
            !accepts_str(&g, b"aab"),
            "\"aab\" has one 'a' too many for r ::= \"\" | \"a\""
        );
    }

    /// Epsilon *last* control: must keep accepting both forms, unaffected by
    /// alternative order (this direction predates #322 and must not regress).
    #[test]
    fn epsilon_last_alternative_still_accepts_both_forms() {
        let g = epsilon_last_alt_grammar();
        assert!(accepts_str(&g, b"ab"), "\"ab\" must accept via r ::= \"a\"");
        assert!(accepts_str(&g, b"b"), "\"b\" must accept via r ::= \"\"");
        assert!(!accepts_str(&g, b"c"));
        assert!(!accepts_str(&g, b"aab"));
    }

    /// `r`'s epsilon choice for "aac" is still open after byte 1: the stack
    /// that took `r ::= "a"` and the stack that took `r ::= ""` both stay
    /// alive, and byte 2 settles which one the input follows.
    #[test]
    fn epsilon_first_choice_stays_open_across_a_later_byte() {
        let g = epsilon_first_multi_byte_grammar();
        assert!(
            accepts_str(&g, b"aac"),
            "\"aac\" must accept via r ::= \"a\""
        );
    }

    /// Exploring every parse of `depth`-nested branches is ~2^depth work,
    /// intractable if unbounded; completing at all proves the expansion budget
    /// capped it. The step reports the limit rather than a rejection, because
    /// the matcher gave up before it could tell whether "x" fits.
    #[test]
    fn nested_nullable_alternatives_report_the_expansion_limit() {
        let g = nested_nullable_branch_grammar(40);
        let mut state = GrammarState::initial();
        assert_eq!(
            advance_byte(&mut state, &g, b'x'),
            StepResult::StackLimitExceeded
        );
        let (result, _) = simulate_token(&GrammarState::initial(), &g, b"x");
        assert_eq!(result, SimResult::StackLimitExceeded);
    }

    #[test]
    fn dangling_ancestor_rule_id_stops_collapse_without_panicking() {
        // Same dangling-ancestor shape as above, but reached via the
        // post-match `collapse_exhausted` walk instead of a mismatch:
        // the top frame's byte DOES match and exhausts its only alt, so
        // popping it advances into the invalid ancestor while collapsing.
        let grammar = CompiledGrammar {
            rules: vec![Rule {
                name: "child".to_string(),
                alts: vec![vec![Symbol::Terminal(b'x')]],
            }],
        };
        let mut state = GrammarState {
            stacks: vec![vec![
                StackFrame {
                    rule_id: 999, // dangling: no such rule
                    alt_idx: 0,
                    sym_pos: 0,
                },
                StackFrame {
                    rule_id: 0, // valid; matches b'x' and then exhausts
                    alt_idx: 0,
                    sym_pos: 0,
                },
            ]],
            partial_token_bytes: Vec::new(),
            complete: false,
        };

        // Must not panic. The byte match itself still succeeds (the top
        // frame's own terminal matched); what must not panic is the
        // subsequent collapse into the dangling ancestor.
        let result = advance_byte(&mut state, &grammar, b'x');
        assert_eq!(result, StepResult::Accepted);
    }

    #[test]
    fn root_is_rule_zero_when_present() {
        let grammar = ab_grammar();

        let root = grammar.root().expect("a grammar with rules has a root");

        assert_eq!(root.name, "root");
    }

    #[test]
    fn root_of_a_grammar_without_rules_is_none() {
        let grammar = GrammarBuilder::new().build();

        assert!(grammar.root().is_none());
    }
}
