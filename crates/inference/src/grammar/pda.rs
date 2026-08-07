//! Byte-level pushdown automaton (PDA) for context-free grammar matching.
//!
//! # Design
//!
//! The grammar is compiled to a set of *rules*, each of which is a sequence
//! of *symbols* (either a terminal byte or a non-terminal rule reference).
//! Execution is modelled as a stack of `StackFrame`s:
//!
//! ```text
//! frame = (rule_id, position_within_rule, alt_index)
//! ```
//!
//! The `advance_byte` operation pops frames that have been fully consumed,
//! pushes frames for non-terminal expansions, and checks whether the current
//! terminal symbol matches the incoming byte.
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
//! The root rule has id 0 (by convention enforced by `CompiledGrammar`).
//!
//! # State machine encoding
//!
//! A `GrammarState` encodes the full PDA configuration:
//!
//! ```text
//! stack: Vec<StackFrame>
//!   StackFrame { rule_id, alt_idx, sym_pos, consumed }
//! partial_bytes: Vec<u8>  — bytes of current token received so far
//! ```
//!
//! `consumed` records whether a byte has been consumed under a frame's current
//! alternative; it gates backtracking so a committed frame is never switched to
//! a sibling alternative (no input rewind). See [`StackFrame::consumed`].
//!
//! The automaton starts with a single frame at `(root, 0, 0)`.
//! `advance_byte(b)` returns whether the byte `b` is accepted (the PDA can
//! make progress) and updates the stack in-place.
//!
//! `can_accept_more()` returns whether the current stack state can still
//! accept additional input (used for context-dependent token masking).
//! `is_complete()` returns whether a terminal state has been reached.

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

/// Maximum PDA stack depth before an advance is rejected.
///
/// A left-recursive or cyclic grammar (`root ::= root`, or a JSON-Schema `$ref`
/// cycle) makes `try_advance_stack` push non-terminal frames without ever
/// consuming a byte, growing the stack without bound — a hang reachable from
/// untrusted grammar input at `GrammarEngine::new` (issue #343). Capping the
/// depth turns that into a bounded rejection (the grammar becomes a dead
/// grammar that accepts nothing) instead of an OOM. The bound is far above any
/// real nesting: `serde_json` itself caps recursion at 128, and each JSON level
/// expands to only a handful of PDA frames, so 8192 frames is unreachable by a
/// well-formed grammar on well-formed output.
pub(crate) const MAX_PDA_DEPTH: usize = 8192;

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

    /// Return the root rule (index 0).
    pub fn root(&self) -> &Rule {
        &self.rules[0]
    }
}

/// One frame on the PDA execution stack.
///
/// `Eq + Hash` (added alongside `PartialEq`, structurally over the same
/// fields) let a `GrammarState`'s `(stack, complete)` pair — the same
/// identity `states_equal` in `engine.rs` already compares by — key a
/// `HashMap` for state-revisit / memoization profiling without changing any
/// existing comparison semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StackFrame {
    /// Index into `CompiledGrammar::rules`.
    pub rule_id: usize,
    /// Index into `rules[rule_id].alts`.
    pub alt_idx: usize,
    /// Position within the chosen alternative (0 = before the first symbol).
    pub sym_pos: usize,
    /// `true` once any input byte has been consumed while this frame has been
    /// at its current `alt_idx` (set for every frame on the stack on each byte
    /// match; reset when the frame switches to a new alternative). A frame with
    /// `consumed == true` is *committed* to its alternative: because this is a
    /// no-rewind byte matcher, switching it to a sibling alternative would
    /// re-interpret already-consumed bytes as if they never existed. The
    /// backtracking logic refuses that switch, which is what closes the
    /// trailing-comma / optional-collapse over-acceptance class (#353). Note it
    /// tracks byte consumption, not `sym_pos`: a frame can advance `sym_pos`
    /// past nullable nonterminals without consuming a byte, and such a frame
    /// must still be free to switch (otherwise `[]` and similar over-reject).
    pub consumed: bool,
}

/// Runtime state of the PDA for one decode sequence.
///
/// Clone this at each step to enable parallel-beam grammar tracking.  The
/// cost is O(stack depth), which for well-formed JSON is at most O(nesting
/// depth) — typically 2-6 frames.
#[derive(Debug, Clone)]
pub struct GrammarState {
    /// Execution stack; top of stack is the last element.
    pub stack: Vec<StackFrame>,
    /// Bytes accumulated within the current token (context-dependent checks).
    pub partial_token_bytes: Vec<u8>,
    /// `true` once the root rule has been fully matched (EOS is valid).
    pub complete: bool,
}

impl GrammarState {
    /// Initial state: single frame at root rule, alt 0, sym_pos 0.
    pub fn initial() -> Self {
        Self {
            stack: vec![StackFrame {
                rule_id: 0,
                alt_idx: 0,
                sym_pos: 0,
                consumed: false,
            }],
            partial_token_bytes: Vec::new(),
            complete: false,
        }
    }

    /// Returns true if the automaton has consumed all input and is in an
    /// accepting configuration (stack is empty or all remaining frames are at
    /// rules whose alternatives can complete with zero bytes).
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Returns true if the automaton could potentially accept more bytes.
    /// Used during context-dependent token inspection.
    pub fn can_accept_more(&self) -> bool {
        !self.complete || !self.stack.is_empty()
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

/// Result of attempting to advance the PDA by one byte.
#[derive(Debug, Clone, PartialEq)]
pub enum StepResult {
    /// The byte was accepted; the state has been updated.
    Accepted,
    /// The byte was rejected by the current grammar state.
    Rejected,
}

/// Advance a `GrammarState` by one byte `b` against `grammar`.
///
/// The algorithm:
/// 1. Inspect the top frame.
/// 2. Get the current symbol at `(rule_id, alt_idx, sym_pos)`.
/// 3. If terminal: match against `b`.  If match, increment `sym_pos`.
///    If the frame is exhausted, pop it and increment sym_pos of the parent
///    (repeatedly until a non-exhausted frame is found or the stack is empty).
/// 4. If non-terminal: push a new frame for the referenced rule (alt 0, pos 0)
///    and retry step 1 — but we do not consume a byte when pushing, so we
///    loop until we reach a terminal.
///
/// If no alternative can accept `b`, try other alternatives for the current
/// frame's rule via backtracking.
pub fn advance_byte(state: &mut GrammarState, grammar: &CompiledGrammar, b: u8) -> StepResult {
    // `try_advance_byte` operates on a clone of `state.stack` and only writes it
    // back on success, so `state.stack` is already left untouched on rejection.
    // No outer snapshot/restore is needed (would be one redundant clone per byte).
    if try_advance_byte(state, grammar, b) {
        state.partial_token_bytes.push(b);
        // Check for completion after consuming the byte.
        state.complete = is_accepting(state, grammar);
        StepResult::Accepted
    } else {
        StepResult::Rejected
    }
}

/// Attempt to advance the PDA by byte `b`.  Returns `true` on success,
/// `false` on rejection.  Mutates `state.stack` in place.
fn try_advance_byte(state: &mut GrammarState, grammar: &CompiledGrammar, b: u8) -> bool {
    // The PDA loop: walk down non-terminals until we hit a terminal.
    // We may need to backtrack across alternative choices.

    // Work on a copy of the stack to support backtracking.
    let mut stack = state.stack.clone();

    if try_advance_stack(&mut stack, grammar, b) {
        state.stack = stack;
        return true;
    }
    false
}

/// Advance `stack` by byte `b`. Returns true on success.
fn try_advance_stack(stack: &mut Vec<StackFrame>, grammar: &CompiledGrammar, b: u8) -> bool {
    loop {
        if stack.is_empty() {
            // Stack empty and we still have a byte to consume → reject.
            return false;
        }
        if stack.len() > MAX_PDA_DEPTH {
            // Cyclic / left-recursive grammar pushing frames without progress
            // (issue #343). Reject rather than grow the stack unbounded.
            return false;
        }

        let frame_idx = stack.len() - 1;
        let frame = &stack[frame_idx];
        let Some(rule) = grammar.rules.get(frame.rule_id) else {
            return false;
        };

        // Rule with no alternatives: dead end → reject via next-alt or backtrack.
        if rule.alts.is_empty() {
            if !try_next_alt(stack, grammar, frame_idx) {
                return false;
            }
            continue;
        }

        let alt = &rule.alts[frame.alt_idx];

        if frame.sym_pos >= alt.len() {
            // Current alternative exhausted: pop frame, advance parent.
            stack.pop();
            if let Some(parent) = stack.last_mut() {
                parent.sym_pos += 1;
            }
            // Continue the loop to handle parent frame.
            continue;
        }

        let sym = &alt[frame.sym_pos].clone();
        match sym {
            Symbol::Terminal(t) => {
                if *t == b {
                    // Match: a byte is consumed. Mark every frame currently on
                    // the stack as having consumed under its current alternative
                    // — the matching frame is the top, and every frame below it
                    // is an ancestor whose active subtree just consumed this byte
                    // (#353). Then advance position and pop exhausted frames.
                    mark_consumed(stack);
                    stack[frame_idx].sym_pos += 1;
                    collapse_exhausted(stack, grammar);
                    return true;
                } else {
                    // Byte doesn't match this terminal. Try this frame's next
                    // alternative; `try_next_alt` enforces the consumed-guard:
                    // a frame that has consumed a byte under its current
                    // alternative is committed and cannot switch (the consumed
                    // bytes cannot be "un-consumed" — no input rewind), while a
                    // frame that only advanced `sym_pos` past nullable
                    // nonterminals is still free to switch (e.g. a tail rule
                    // reaching its `ε` alternative). This single path replaces
                    // the old `sym_pos == 0` heuristic and closes both halves of
                    // #353 (the trailing-comma over-acceptance and the
                    // nullable-prefix over-rejection).
                    if !try_next_alt(stack, grammar, frame_idx) {
                        return false;
                    }
                    continue;
                }
            }
            Symbol::AnyByte => {
                // AnyByte matches any single byte: a byte is consumed.
                mark_consumed(stack);
                stack[frame_idx].sym_pos += 1;
                collapse_exhausted(stack, grammar);
                return true;
            }
            Symbol::NonTerminal(rule_id) => {
                let rid = *rule_id;
                // Push a new frame for the non-terminal's first alt.
                // Before pushing, check if the referenced rule has any alts.
                let Some(referenced_rule) = grammar.rules.get(rid) else {
                    return false;
                };
                if referenced_rule.alts.is_empty() {
                    // Empty rule = epsilon; advance past the non-terminal.
                    stack[frame_idx].sym_pos += 1;
                    continue;
                }
                stack.push(StackFrame {
                    rule_id: rid,
                    alt_idx: 0,
                    sym_pos: 0,
                    consumed: false,
                });
                // Continue loop: now top frame is the pushed non-terminal.
            }
        }
    }
}

/// Try alternative `alt_idx + 1` for the rule at `frame_idx`.
fn try_next_alt(
    stack: &mut Vec<StackFrame>,
    grammar: &CompiledGrammar,
    mut frame_idx: usize,
) -> bool {
    // Consumed-guard (#353). If this frame has consumed an input byte under its
    // current alternative it is committed: switching it to a sibling alternative
    // — or popping it to switch an ancestor — would re-interpret the consumed
    // bytes as if they never existed, and this byte-level matcher has no input
    // rewind. Reject instead. `mark_consumed` flags every frame on the stack at
    // each byte match, so any ancestor of a committed frame is committed too.
    // This is the guard whose absence over-accepted trailing commas (`[1,]`,
    // `{"r":1,"o1":2,}`). It tracks byte consumption, not `sym_pos`, so a frame
    // that only advanced past nullable nonterminals (e.g. an optional `ws`
    // before a `,`) is NOT committed and still reaches its `ε` alternative below
    // — which is what keeps `[]`, `[5]`, and `[1,2]` accepting (the
    // over-rejection dual).
    //
    // SOUNDNESS (no over-acceptance): rejecting here never admits invalid input.
    // After a byte is consumed, escalating to an ancestor's sibling alternative
    // could only re-interpret that already-consumed byte under a different rule;
    // with no rewind, any acceptance it produced would be an over-accept, never
    // a faithful parse. Refusing the switch is the correct direction.
    //
    // KNOWN LIMITATION (not complete; pre-existing, NOT introduced by #353): for
    // grammars with shared-prefix sibling alternatives a faithful parse can
    // genuinely need to switch siblings *after* consuming the shared prefix —
    // e.g. enum `["foo","food"]`, where `"food"` requires the second member once
    // `foo` is consumed. A no-rewind single-stack matcher cannot do this and
    // over-REJECTS the longer member (verified byte-identical on origin/main —
    // this guard neither causes nor fixes it). Over-rejection is the safe
    // direction for constrained decoding (the model simply cannot emit one valid
    // member; it never emits invalid output). Making shared-prefix alternatives
    // complete needs ambiguity-preserving matching (a trie/NFA compiled form or
    // parallel active stacks) and is out of scope here. See the
    // `shared_prefix_enum_known_limitation` regression anchor in json_schema.rs.
    loop {
        if stack[frame_idx].consumed {
            return false;
        }

        let rule_id = stack[frame_idx].rule_id;
        let next_alt = stack[frame_idx].alt_idx + 1;
        let num_alts = grammar.rules[rule_id].alts.len();

        if next_alt < num_alts {
            // Switch to the next alternative in the same rule (reset position and the
            // consumed flag — the new alternative has consumed nothing yet).
            stack[frame_idx].alt_idx = next_alt;
            stack[frame_idx].sym_pos = 0;
            stack[frame_idx].consumed = false;
            // Truncate any frames pushed during the failed attempt.
            stack.truncate(frame_idx + 1);
            return true;
        }

        // No more alternatives at this (uncommitted) level: pop the frame and
        // inspect the parent. Sound because this frame consumed no byte under
        // its current alternative, so removing it un-interprets nothing.
        if frame_idx == 0 {
            return false;
        }
        stack.truncate(frame_idx);
        frame_idx -= 1;
    }
}

/// Mark every frame currently on the stack as having consumed a byte under its
/// current alternative. Called on each successful byte match: the matching
/// frame is the top of the stack and every frame below it is an ancestor whose
/// active subtree just consumed the byte, so all of them become committed to
/// their current alternative (#353). See [`StackFrame::consumed`].
fn mark_consumed(stack: &mut [StackFrame]) {
    for frame in stack.iter_mut() {
        frame.consumed = true;
    }
}

/// Pop exhausted frames from the top of the stack after a successful byte match.
/// A frame is exhausted when `sym_pos >= alt.len()`.
fn collapse_exhausted(stack: &mut Vec<StackFrame>, grammar: &CompiledGrammar) {
    loop {
        match stack.last() {
            None => break,
            Some(frame) => {
                let rule = &grammar.rules[frame.rule_id];
                // No alts: already dead-ended; pop.
                if rule.alts.is_empty() {
                    stack.pop();
                    if let Some(parent) = stack.last_mut() {
                        parent.sym_pos += 1;
                    }
                    continue;
                }
                let alt = &rule.alts[frame.alt_idx];
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
/// A state is accepting if all remaining work on the stack can be resolved
/// with zero additional bytes — i.e., all remaining symbols are *nullable*
/// (can derive the empty string).
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
fn is_accepting(state: &GrammarState, grammar: &CompiledGrammar) -> bool {
    let n = state.stack.len();
    for (i, frame) in state.stack.iter().enumerate() {
        if frame.rule_id >= grammar.rules.len() {
            return false;
        }
        let rule = &grammar.rules[frame.rule_id];
        if rule.alts.is_empty() {
            // A rule with no alts is an empty / epsilon rule — always nullable.
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

/// Returns true if `grammar.rules[rule_id].alts[alt_idx][pos..]` can derive
/// the empty string, i.e. every remaining symbol is nullable.
///
/// This mirrors the natural mutually-recursive definition (an alt is nullable
/// iff every remaining symbol is nullable; a non-terminal is nullable iff
/// *some* alternative of the rule it names is nullable) but walks an explicit,
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
    #[derive(Clone, Copy)]
    enum Frame {
        /// Checking whether `grammar.rules[rule_id].alts[alt_idx][pos..]` is nullable.
        Alt {
            rule_id: usize,
            alt_idx: usize,
            pos: usize,
        },
        /// Checking whether any of `grammar.rules[rule_id].alts[alt_idx..]` is nullable.
        Rule { rule_id: usize, alt_idx: usize },
    }

    let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut stack = vec![Frame::Alt {
        rule_id,
        alt_idx,
        pos,
    }];
    // The boolean result of the frame that just finished, to be consumed by
    // the frame now on top of `stack`.
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
                        *stack.last_mut().unwrap() = Frame::Alt {
                            rule_id,
                            alt_idx,
                            pos,
                        };
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
                if alt_idx >= rule.alts.len() {
                    stack.pop();
                    pending = Some(false);
                } else {
                    *stack.last_mut().unwrap() = Frame::Rule { rule_id, alt_idx };
                    stack.push(Frame::Alt {
                        rule_id,
                        alt_idx,
                        pos: 0,
                    });
                }
            }
        }
    }
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
        }
    }
    (SimResult::Accept, state)
}

// ---------------------------------------------------------------------------
// Grammar builders for use by json_schema.rs and gbnf.rs
// ---------------------------------------------------------------------------

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
    pub fn set_alts(&mut self, id: usize, alts: Vec<Alt>) {
        self.rules[id].alts = alts;
    }

    /// Reserve and immediately set alternatives.
    pub fn add_rule(&mut self, name: &str, alts: Vec<Alt>) -> usize {
        let id = self.reserve(name);
        self.set_alts(id, alts);
        id
    }

    /// Look up the id of a previously reserved rule.
    pub fn rule_id(&self, name: &str) -> Option<usize> {
        self.name_to_id.get(name).copied()
    }

    /// Consume the builder and produce a `CompiledGrammar`.
    ///
    /// Panics if the root rule (index 0, name "root") has no alternatives.
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
        b.set_alts(digit_id, digit_alts);

        // root = digit digit_rest
        // digit_rest = digit digit_rest | ε  (implemented as digit_rest = [empty alt])
        let rest_id = b.reserve("digit_rest");
        b.set_alts(
            rest_id,
            vec![
                vec![Symbol::NonTerminal(digit_id), Symbol::NonTerminal(rest_id)],
                vec![], // epsilon
            ],
        );

        let root_id = b.reserve("root");
        b.set_alts(
            root_id,
            vec![vec![
                Symbol::NonTerminal(digit_id),
                Symbol::NonTerminal(rest_id),
            ]],
        );
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

    /// Minimal grammar isolating the no-rewind trailing-comma class, free of
    /// the JSON-schema compiler:
    ///   root = '[' body ']'
    ///   body = elem tail | ε
    ///   tail = ',' elem tail | ε
    ///   elem = 'x'
    /// A trailing comma (`[x,]`) must reject: the `tail` frame that consumed
    /// the `,` byte may not backtrack to its ε alternative once a byte is
    /// committed.  The nullable `body`/`tail` ε arms must still let valid
    /// forms through (refs #353).
    fn comma_list_grammar() -> CompiledGrammar {
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root"); // index 0
        let body_id = b.reserve("body");
        let tail_id = b.reserve("tail");
        let elem_id = b.reserve("elem");
        b.set_alts(elem_id, vec![vec![Symbol::Terminal(b'x')]]);
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
        );
        b.set_alts(
            body_id,
            vec![
                vec![Symbol::NonTerminal(elem_id), Symbol::NonTerminal(tail_id)],
                vec![], // epsilon
            ],
        );
        b.set_alts(
            root_id,
            vec![vec![
                Symbol::Terminal(b'['),
                Symbol::NonTerminal(body_id),
                Symbol::Terminal(b']'),
            ]],
        );
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
        assert!(!accepts_str(&g, b"[x,]")); // trailing comma: dirty tail must not switch to ε
        assert!(!accepts_str(&g, b"[x,x,]")); // same, one level deeper
        assert!(!accepts_str(&g, b"[,]")); // leading comma
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
        // (`try_advance_byte` clones the stack and only commits it on success,
        // so no outer snapshot is needed).
        //
        // The grammar must be *nested* so the rejecting byte forces
        // `try_advance_stack` to truncate a child frame before it fails:
        //   root  ::= "a" child
        //   child ::= "bc"
        // Feeding `a`,`b` descends into `child` (frame pushed, one byte
        // consumed). The wrong byte at child's second position truncates the
        // child frame, then exhausts root's alternatives and returns false,
        // mutating the working stack en route. Only the inner clone keeps
        // `state.stack` intact so the correct `c` can still complete. A flat
        // grammar (reject at the root frame returns false without mutating)
        // never exercises this and would make the test vacuous.
        let mut b = GrammarBuilder::new();
        let root_id = b.reserve("root");
        let child_id = b.reserve("child");
        b.set_alts(
            child_id,
            vec![vec![Symbol::Terminal(b'b'), Symbol::Terminal(b'c')]],
        );
        b.set_alts(
            root_id,
            vec![vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(child_id)]],
        );
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
    fn dead_child_backtracks_to_parent_alternative() {
        let mut builder = GrammarBuilder::new();
        let root_id = builder.reserve("root");
        let dead_id = builder.reserve("dead");
        builder.set_alts(dead_id, vec![vec![Symbol::Terminal(b'y')]]);
        builder.set_alts(
            root_id,
            vec![
                vec![Symbol::NonTerminal(dead_id)],
                vec![Symbol::Terminal(b'x')],
            ],
        );
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
    /// must not overflow the native stack even though `state.stack.len()`
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

        let past_limit = nested_terminal_grammar(MAX_PDA_DEPTH + 1, b'x');
        let mut state = GrammarState::initial();
        assert_eq!(
            advance_byte(&mut state, &past_limit, b'x'),
            StepResult::Rejected
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
        );
        b.set_alts(
            root_id,
            vec![
                vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(nt_id)],
                vec![Symbol::Terminal(b'x')],
            ],
        );
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
        assert_eq!(state.stack, before.stack);
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
            stack: vec![StackFrame {
                rule_id: 1,
                alt_idx: 0,
                sym_pos: 0,
                consumed: false,
            }],
            partial_token_bytes: Vec::new(),
            complete: false,
        };
        let before = state.clone();

        assert_eq!(
            advance_byte(&mut state, &grammar, b'x'),
            StepResult::Rejected
        );
        assert_eq!(state.stack, before.stack);
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
        assert_eq!(state.stack, before.stack);
        assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
        assert_eq!(state.complete, before.complete);
    }
}
