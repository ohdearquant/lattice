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
    ///
    /// This takes no grammar, so it cannot know whether the root accepts the
    /// empty string: `complete` is `false` even when it does. A caller that
    /// holds the grammar should start from [`GrammarState::for_grammar`].
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

    /// Initial state for `grammar`: the same stack as [`GrammarState::initial`],
    /// with `complete` set from the grammar, so a root that accepts the empty
    /// string reports [`GrammarState::is_complete`] before any byte.
    pub fn for_grammar(grammar: &CompiledGrammar) -> Self {
        let mut state = Self::initial();
        state.complete = is_accepting(&state, grammar);
        state
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
    GrammarState::for_grammar(grammar)
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
mod tests;
