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
//!   StackFrame { rule_id, alt_idx, sym_pos }
//! partial_bytes: Vec<u8>  — bytes of current token received so far
//! ```
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
#[derive(Debug, Clone, PartialEq)]
pub struct StackFrame {
    /// Index into `CompiledGrammar::rules`.
    pub rule_id: usize,
    /// Index into `rules[rule_id].alts`.
    pub alt_idx: usize,
    /// Position within the chosen alternative (0 = before the first symbol).
    pub sym_pos: usize,
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
///    (recursively until a non-exhausted frame is found or the stack is empty).
/// 4. If non-terminal: push a new frame for the referenced rule (alt 0, pos 0)
///    and retry step 1 — but we do not consume a byte when pushing, so we
///    loop until we reach a terminal.
///
/// If no alternative can accept `b`, try other alternatives for the current
/// frame's rule via backtracking.
pub fn advance_byte(state: &mut GrammarState, grammar: &CompiledGrammar, b: u8) -> StepResult {
    // We snapshot the stack on entry; if all alternatives fail, restore it.
    let snapshot = state.stack.clone();

    if try_advance_byte(state, grammar, b) {
        state.partial_token_bytes.push(b);
        // Check for completion after consuming the byte.
        state.complete = is_accepting(state, grammar);
        StepResult::Accepted
    } else {
        // Restore snapshot — byte was rejected.
        state.stack = snapshot;
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

/// Recursively advance `stack` by byte `b`.  Returns true on success.
fn try_advance_stack(stack: &mut Vec<StackFrame>, grammar: &CompiledGrammar, b: u8) -> bool {
    loop {
        if stack.is_empty() {
            // Stack empty and we still have a byte to consume → reject.
            return false;
        }

        let frame_idx = stack.len() - 1;
        let frame = &stack[frame_idx];
        let rule = &grammar.rules[frame.rule_id];

        // Rule with no alternatives: dead end → reject via next-alt or backtrack.
        if rule.alts.is_empty() {
            return try_next_alt(stack, grammar, b, frame_idx);
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
                    // Match: advance position in current alternative.
                    stack[frame_idx].sym_pos += 1;
                    // Pop any exhausted frames.
                    collapse_exhausted(stack, grammar);
                    return true;
                } else {
                    // Byte doesn't match this terminal.
                    // Only switch alternatives when no bytes have been consumed
                    // in the current alternative (sym_pos == 0).  Once we have
                    // consumed bytes under one alternative, backtracking to a
                    // sibling alternative would create an inconsistent state
                    // (the consumed bytes cannot be "un-consumed").
                    if frame.sym_pos == 0 {
                        return try_next_alt(stack, grammar, b, frame_idx);
                    } else {
                        // Mid-alternative mismatch: propagate to parent.
                        if frame_idx == 0 {
                            return false;
                        }
                        stack.truncate(frame_idx);
                        let parent_idx = stack.len() - 1;
                        return try_next_alt(stack, grammar, b, parent_idx);
                    }
                }
            }
            Symbol::AnyByte => {
                // AnyByte matches any single byte.
                stack[frame_idx].sym_pos += 1;
                collapse_exhausted(stack, grammar);
                return true;
            }
            Symbol::NonTerminal(rule_id) => {
                let rid = *rule_id;
                // Push a new frame for the non-terminal's first alt.
                // Before pushing, check if the referenced rule has any alts.
                if grammar.rules[rid].alts.is_empty() {
                    // Empty rule = epsilon; advance past the non-terminal.
                    stack[frame_idx].sym_pos += 1;
                    continue;
                }
                stack.push(StackFrame {
                    rule_id: rid,
                    alt_idx: 0,
                    sym_pos: 0,
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
    b: u8,
    frame_idx: usize,
) -> bool {
    let rule_id = stack[frame_idx].rule_id;
    let next_alt = stack[frame_idx].alt_idx + 1;
    let num_alts = grammar.rules[rule_id].alts.len();

    if next_alt >= num_alts {
        // No more alternatives at this level.
        // Try backtracking to the parent.
        if frame_idx == 0 {
            return false;
        }
        // Pop this frame, try alternatives at the parent level.
        stack.truncate(frame_idx);
        // Retry with parent — but we need to try the parent's next alt or
        // propagate rejection.  We do this by trying the parent's next alt.
        let parent_idx = stack.len() - 1;
        return try_next_alt(stack, grammar, b, parent_idx);
    }

    // Switch to next alternative in the same rule (reset sym_pos).
    stack[frame_idx].alt_idx = next_alt;
    stack[frame_idx].sym_pos = 0;
    // Truncate any frames pushed during the failed attempt.
    stack.truncate(frame_idx + 1);
    // Retry with new alt.
    try_advance_stack(stack, grammar, b)
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
        let alt = &rule.alts[frame.alt_idx];
        // Non-top frames: the child frame is handling the symbol at sym_pos,
        // so check nullable from sym_pos + 1.
        // Top frame: check nullable from sym_pos itself.
        let check_from = if i == n - 1 {
            frame.sym_pos
        } else {
            frame.sym_pos + 1
        };
        if !remaining_is_nullable(
            grammar,
            alt,
            check_from,
            &mut std::collections::HashSet::new(),
        ) {
            return false;
        }
    }
    true
}

/// Returns true if the symbols `alt[pos..]` can all derive the empty string.
fn remaining_is_nullable(
    grammar: &CompiledGrammar,
    alt: &[Symbol],
    pos: usize,
    visited: &mut std::collections::HashSet<usize>,
) -> bool {
    for sym in &alt[pos..] {
        match sym {
            Symbol::Terminal(_) | Symbol::AnyByte => return false,
            Symbol::NonTerminal(rid) => {
                if !visited.insert(*rid) {
                    // Already checking this rule (cycle): conservatively non-nullable.
                    return false;
                }
                if !rule_is_nullable(grammar, *rid, visited) {
                    visited.remove(rid);
                    return false;
                }
                visited.remove(rid);
            }
        }
    }
    true
}

/// Returns true if rule `rule_id` has at least one alternative that can
/// derive the empty string.
fn rule_is_nullable(
    grammar: &CompiledGrammar,
    rule_id: usize,
    visited: &mut std::collections::HashSet<usize>,
) -> bool {
    if rule_id >= grammar.rules.len() {
        return false;
    }
    for alt in &grammar.rules[rule_id].alts {
        if remaining_is_nullable(grammar, alt, 0, visited) {
            return true;
        }
    }
    false
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
}
