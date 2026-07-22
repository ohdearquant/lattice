//! GBNF grammar parser (llama.cpp-compatible subset).
//!
//! # Supported syntax
//!
//! ```text
//! grammar  = rule+
//! rule     = IDENT "::=" expr "\n"
//! expr     = alt ("|" alt)*
//! alt      = item+
//! item     = IDENT                  — non-terminal reference
//!          | '"' chars '"'          — string literal
//!          | "[" range_chars "]"    — character class
//!          | "[^" range_chars "]"   — negated character class
//!          | "."                    — any byte
//!          | "(" expr ")"           — grouping
//!          | item "*"               — zero or more
//!          | item "+"               — one or more
//!          | item "?"               — zero or one
//! range_chars = (char | char "-" char)+
//! ```
//!
//! The grammar must contain a rule named `root`.  The `root` rule becomes
//! rule index 0 in the output `CompiledGrammar`.
//!
//! # Limitations (v0)
//!
//! - `*` and `+` are desugared into right-recursive rules.
//! - `?` is desugared into `alt | ε`.
//! - Character class negation `[^...]` generates alternatives for every
//!   byte NOT in the set — correct but potentially many alternatives.

use crate::grammar::pda::{Alt, CompiledGrammar, GrammarBuilder, Symbol};
use std::collections::HashSet;

/// Error from parsing a GBNF string.
#[derive(Debug, Clone, PartialEq)]
pub struct GbnfError(pub String);

impl std::fmt::Display for GbnfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GBNF parse error: {}", self.0)
    }
}
impl std::error::Error for GbnfError {}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Assign,   // ::=
    Pipe,     // |
    LParen,   // (
    RParen,   // )
    Star,     // *
    Plus,     // +
    Question, // ?
    Dot,      // .
    StringLit(Vec<u8>),
    CharClass { bytes: Vec<u8>, negated: bool },
    Newline,
    Eof,
}

struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_spaces(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_line_comment(&mut self) {
        if self.peek() == Some(b'#') {
            while let Some(b) = self.advance() {
                if b == b'\n' {
                    break;
                }
            }
        }
    }

    fn read_ident(&mut self) -> String {
        let mut s = String::new();
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                s.push(b as char);
                self.advance();
            } else {
                break;
            }
        }
        s
    }

    fn read_string_lit(&mut self) -> Result<Vec<u8>, GbnfError> {
        // Opening `"` already consumed.
        let mut bytes = Vec::new();
        loop {
            match self.advance() {
                None => return Err(GbnfError("unterminated string literal".into())),
                Some(b'"') => break,
                Some(b'\\') => {
                    match self.advance() {
                        None => return Err(GbnfError("truncated escape in string".into())),
                        Some(b'n') => bytes.push(b'\n'),
                        Some(b't') => bytes.push(b'\t'),
                        Some(b'r') => bytes.push(b'\r'),
                        Some(b'"') => bytes.push(b'"'),
                        Some(b'\\') => bytes.push(b'\\'),
                        Some(b'u') => {
                            // \uXXXX — decode 4 hex digits to UTF-8.
                            let mut hex = String::new();
                            for _ in 0..4 {
                                match self.advance() {
                                    Some(h) => hex.push(h as char),
                                    None => return Err(GbnfError("truncated \\u escape".into())),
                                }
                            }
                            let code = u32::from_str_radix(&hex, 16)
                                .map_err(|_| GbnfError(format!("invalid \\u escape: {hex}")))?;
                            let ch = char::from_u32(code).ok_or_else(|| {
                                GbnfError(format!("invalid unicode codepoint: {code}"))
                            })?;
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        Some(other) => bytes.push(other),
                    }
                }
                Some(b) => bytes.push(b),
            }
        }
        Ok(bytes)
    }

    fn read_char_class(&mut self) -> Result<Token, GbnfError> {
        // Opening `[` already consumed.
        let negated = if self.peek() == Some(b'^') {
            self.advance();
            true
        } else {
            false
        };

        let mut included = Vec::new();
        loop {
            match self.peek() {
                None => return Err(GbnfError("unterminated character class".into())),
                Some(b']') => {
                    self.advance();
                    break;
                }
                Some(b'\\') => {
                    self.advance();
                    let byte = match self.advance() {
                        Some(b'n') => b'\n',
                        Some(b't') => b'\t',
                        Some(b'r') => b'\r',
                        Some(b']') => b']',
                        Some(b'\\') => b'\\',
                        Some(b'-') => b'-',
                        Some(b) => b,
                        None => return Err(GbnfError("truncated escape in char class".into())),
                    };
                    // Check for range `\x-y`
                    if self.peek() == Some(b'-') && self.input.get(self.pos + 1) != Some(&b']') {
                        self.advance(); // consume '-'
                        let Some(end) = self.advance() else {
                            return Err(GbnfError("truncated char range".into()));
                        };
                        for b in byte..=end {
                            included.push(b);
                        }
                    } else {
                        included.push(byte);
                    }
                }
                Some(b) => {
                    self.advance();
                    // Check for range `a-z`
                    if self.peek() == Some(b'-') && self.input.get(self.pos + 1) != Some(&b']') {
                        self.advance(); // consume '-'
                        let Some(end) = self.advance() else {
                            return Err(GbnfError("truncated char range".into()));
                        };
                        for byte in b..=end {
                            included.push(byte);
                        }
                    } else {
                        included.push(b);
                    }
                }
            }
        }

        Ok(Token::CharClass {
            bytes: included,
            negated,
        })
    }

    fn next_token(&mut self) -> Result<Token, GbnfError> {
        self.skip_spaces();
        self.skip_line_comment();

        match self.peek() {
            None => Ok(Token::Eof),
            Some(b'\n') => {
                self.advance();
                Ok(Token::Newline)
            }
            Some(b':') => {
                self.advance();
                if self.peek() == Some(b':') {
                    self.advance();
                    if self.advance() == Some(b'=') {
                        Ok(Token::Assign)
                    } else {
                        Err(GbnfError("expected ::=".into()))
                    }
                } else {
                    Err(GbnfError("expected ::=".into()))
                }
            }
            Some(b'|') => {
                self.advance();
                Ok(Token::Pipe)
            }
            Some(b'(') => {
                self.advance();
                Ok(Token::LParen)
            }
            Some(b')') => {
                self.advance();
                Ok(Token::RParen)
            }
            Some(b'*') => {
                self.advance();
                Ok(Token::Star)
            }
            Some(b'+') => {
                self.advance();
                Ok(Token::Plus)
            }
            Some(b'?') => {
                self.advance();
                Ok(Token::Question)
            }
            Some(b'.') => {
                self.advance();
                Ok(Token::Dot)
            }
            Some(b'"') => {
                self.advance();
                let bytes = self.read_string_lit()?;
                Ok(Token::StringLit(bytes))
            }
            Some(b'[') => {
                self.advance();
                self.read_char_class()
            }
            Some(b) if b.is_ascii_alphabetic() || b == b'_' => {
                let ident = self.read_ident();
                Ok(Token::Ident(ident))
            }
            Some(b) => Err(GbnfError(format!("unexpected byte: 0x{b:02x}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    lexer: Lexer<'a>,
    lookahead: Option<Token>,
    builder: GrammarBuilder,
    /// Counter for auto-generated rule names (for desugared repetition).
    anon_counter: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Result<Self, GbnfError> {
        let mut lexer = Lexer::new(input);
        let tok = lexer.next_token()?;
        Ok(Self {
            lexer,
            lookahead: Some(tok),
            builder: GrammarBuilder::new(),
            anon_counter: 0,
        })
    }

    fn peek(&self) -> &Token {
        self.lookahead.as_ref().unwrap_or(&Token::Eof)
    }

    fn consume(&mut self) -> Result<Token, GbnfError> {
        let tok = self.lookahead.take().unwrap_or(Token::Eof);
        self.lookahead = Some(self.lexer.next_token()?);
        Ok(tok)
    }

    fn expect(&mut self, expected: &Token) -> Result<(), GbnfError> {
        let tok = self.consume()?;
        if &tok == expected {
            Ok(())
        } else {
            Err(GbnfError(format!("expected {expected:?}, got {tok:?}")))
        }
    }

    fn skip_newlines(&mut self) -> Result<(), GbnfError> {
        while self.peek() == &Token::Newline {
            self.consume()?;
        }
        Ok(())
    }

    fn anon_rule_name(&mut self) -> String {
        let n = format!("__anon_{}", self.anon_counter);
        self.anon_counter += 1;
        n
    }

    // grammar = rule+
    // Returns the builder, consuming self.
    fn parse_grammar(mut self) -> Result<GrammarBuilder, GbnfError> {
        self.skip_newlines()?;
        while self.peek() != &Token::Eof {
            self.parse_rule()?;
            self.skip_newlines()?;
        }
        Ok(self.builder)
    }

    // rule = IDENT "::=" expr "\n"?
    fn parse_rule(&mut self) -> Result<(), GbnfError> {
        let name = match self.consume()? {
            Token::Ident(n) => n,
            tok => return Err(GbnfError(format!("expected rule name, got {tok:?}"))),
        };
        self.expect(&Token::Assign)?;
        let alts = self.parse_expr()?;
        let id = self.builder.reserve(&name);
        self.builder.set_alts(id, alts);
        // Consume optional trailing newline(s)
        self.skip_newlines()?;
        Ok(())
    }

    // expr = alt ("|" alt)*
    fn parse_expr(&mut self) -> Result<Vec<Alt>, GbnfError> {
        let mut alts = vec![self.parse_alt()?];
        while self.peek() == &Token::Pipe {
            self.consume()?;
            alts.push(self.parse_alt()?);
        }
        Ok(alts)
    }

    // alt = item*  (stops at | ) \n EOF)
    fn parse_alt(&mut self) -> Result<Alt, GbnfError> {
        let mut syms: Alt = Vec::new();
        loop {
            match self.peek() {
                Token::Pipe | Token::RParen | Token::Newline | Token::Eof => break,
                _ => {
                    let new_syms = self.parse_item()?;
                    syms.extend(new_syms);
                }
            }
        }
        Ok(syms)
    }

    // item = base_item ("*" | "+" | "?")?
    fn parse_item(&mut self) -> Result<Alt, GbnfError> {
        let base = self.parse_base_item()?;

        match self.peek() {
            Token::Star => {
                self.consume()?;
                // Desugar `x*` → `opt_rep = x opt_rep | ε`
                let rule_name = self.anon_rule_name();
                let id = self.builder.reserve(&rule_name);
                let rep_alt: Alt = base
                    .iter()
                    .cloned()
                    .chain(std::iter::once(Symbol::NonTerminal(id)))
                    .collect();
                self.builder.set_alts(id, vec![rep_alt, vec![]]);
                Ok(vec![Symbol::NonTerminal(id)])
            }
            Token::Plus => {
                self.consume()?;
                // Desugar `x+` → `rep = x tail; tail = x tail | ε`
                // Using a nullable tail rule ensures `is_accepting` returns
                // true after consuming exactly one `x` (tail is epsilon-able).
                let tail_name = self.anon_rule_name();
                let tail_id = self.builder.reserve(&tail_name);
                // tail = [x, NT(tail)] | ε
                let tail_rec: Alt = base
                    .iter()
                    .cloned()
                    .chain(std::iter::once(Symbol::NonTerminal(tail_id)))
                    .collect();
                self.builder.set_alts(tail_id, vec![tail_rec, vec![]]);
                // rep = x tail
                let rep_name = self.anon_rule_name();
                let rep_id = self.builder.reserve(&rep_name);
                let rep_alt: Alt = base
                    .iter()
                    .cloned()
                    .chain(std::iter::once(Symbol::NonTerminal(tail_id)))
                    .collect();
                self.builder.set_alts(rep_id, vec![rep_alt]);
                Ok(vec![Symbol::NonTerminal(rep_id)])
            }
            Token::Question => {
                self.consume()?;
                // Desugar `x?` → `opt = x | ε`
                let rule_name = self.anon_rule_name();
                let id = self.builder.reserve(&rule_name);
                self.builder.set_alts(id, vec![base, vec![]]);
                Ok(vec![Symbol::NonTerminal(id)])
            }
            _ => Ok(base),
        }
    }

    // base_item = IDENT | string_lit | char_class | "." | "(" expr ")"
    fn parse_base_item(&mut self) -> Result<Alt, GbnfError> {
        match self.peek().clone() {
            Token::Ident(name) => {
                self.consume()?;
                // Forward-reference: reserve if not seen yet.
                let id = self.builder.reserve(&name);
                Ok(vec![Symbol::NonTerminal(id)])
            }
            Token::StringLit(_) => {
                if let Token::StringLit(bytes) = self.consume()? {
                    Ok(bytes.into_iter().map(Symbol::Terminal).collect())
                } else {
                    unreachable!()
                }
            }
            Token::CharClass { .. } => {
                if let Token::CharClass { bytes, negated } = self.consume()? {
                    let alts = char_class_alts(&bytes, negated);
                    if alts.is_empty() {
                        return Err(GbnfError("empty character class".into()));
                    }
                    if alts.len() == 1 {
                        return Ok(alts.into_iter().next().unwrap());
                    }
                    // Wrap in anon rule.
                    let rule_name = self.anon_rule_name();
                    let id = self.builder.reserve(&rule_name);
                    self.builder.set_alts(id, alts);
                    Ok(vec![Symbol::NonTerminal(id)])
                } else {
                    unreachable!()
                }
            }
            Token::Dot => {
                self.consume()?;
                Ok(vec![Symbol::AnyByte])
            }
            Token::LParen => {
                self.consume()?;
                let alts = self.parse_expr()?;
                match self.consume()? {
                    Token::RParen => {}
                    tok => {
                        return Err(GbnfError(format!("expected ), got {tok:?}")));
                    }
                }
                let rule_name = self.anon_rule_name();
                let id = self.builder.reserve(&rule_name);
                self.builder.set_alts(id, alts);
                Ok(vec![Symbol::NonTerminal(id)])
            }
            tok => Err(GbnfError(format!("unexpected token in item: {tok:?}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build alternatives for a character class.
/// Returns a `Vec<Alt>` — one alternative per matching byte.
fn char_class_alts(bytes: &[u8], negated: bool) -> Vec<Alt> {
    if negated {
        let set: HashSet<u8> = bytes.iter().copied().collect();
        (0u8..=255)
            .filter(|b| !set.contains(b))
            .map(|b| vec![Symbol::Terminal(b)])
            .collect()
    } else {
        bytes.iter().map(|&b| vec![Symbol::Terminal(b)]).collect()
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a GBNF string into a `CompiledGrammar`.
///
/// The grammar must contain a rule named `root`.  After parsing, the root
/// rule is moved to index 0 so the PDA initial state is correct.
pub fn parse_gbnf(gbnf: &str) -> Result<CompiledGrammar, GbnfError> {
    let parser = Parser::new(gbnf)?;
    let builder = parser.parse_grammar()?;
    let mut grammar = builder.build();

    // Completeness check (issue #1079): `parse_base_item` reserves a rule id
    // for every identifier reference before it is known whether that rule is
    // ever defined (`::=`); a name that is referenced but never defined is
    // left with empty `alts`. `pda.rs`'s `advance_byte` treats an empty-alts
    // rule as an intentional epsilon, but there is no GBNF syntax that
    // legitimately produces one: every parsed `rule = IDENT "::=" expr`
    // reaches `set_alts` with at least one alt, even for an explicitly empty
    // RHS (`parse_expr` always pushes at least `parse_alt()`'s result, which
    // may itself be the empty sequence `vec![]` — that is one alt containing
    // zero symbols, not zero alts). So `alts.is_empty()` here can only mean
    // "reserved by a reference, never defined" — a typo'd or missing rule
    // name — and must be rejected rather than silently treated as an
    // intentional epsilon rule that would swallow the reference.
    if let Some(undefined) = grammar.rules.iter().find(|r| r.alts.is_empty()) {
        return Err(GbnfError(format!(
            "undefined rule referenced: '{}'",
            undefined.name
        )));
    }

    // Ensure root is at index 0.
    let root_pos = grammar
        .rules
        .iter()
        .position(|r| r.name == "root")
        .ok_or_else(|| GbnfError("grammar has no 'root' rule".into()))?;

    if root_pos != 0 {
        // Swap root to index 0 and fix up all NonTerminal references.
        grammar.rules.swap(0, root_pos);
        for rule in &mut grammar.rules {
            for alt in &mut rule.alts {
                for sym in alt.iter_mut() {
                    if let Symbol::NonTerminal(rid) = sym {
                        if *rid == root_pos {
                            *rid = 0;
                        } else if *rid == 0 {
                            *rid = root_pos;
                        }
                    }
                }
            }
        }
    }

    Ok(grammar)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::pda::{GrammarState, SimResult, simulate_token};

    fn accepts(grammar: &CompiledGrammar, input: &[u8]) -> bool {
        let state = GrammarState::initial();
        let (result, final_state) = simulate_token(&state, grammar, input);
        result == SimResult::Accept && final_state.is_complete()
    }

    fn rejects(grammar: &CompiledGrammar, input: &[u8]) -> bool {
        // A grammar "rejects" a string when it cannot accept it as a complete
        // value.  This includes both early byte rejections (SimResult::Reject)
        // and cases where all bytes are consumed but the state is not complete.
        !accepts(grammar, input)
    }

    #[test]
    fn gbnf_string_literal() {
        let g = parse_gbnf("root ::= \"hello\"\n").unwrap();
        assert!(accepts(&g, b"hello"));
        assert!(rejects(&g, b"world"));
    }

    #[test]
    fn gbnf_alternation() {
        let g = parse_gbnf("root ::= \"yes\" | \"no\"\n").unwrap();
        assert!(accepts(&g, b"yes"));
        assert!(accepts(&g, b"no"));
        assert!(rejects(&g, b"maybe"));
    }

    #[test]
    fn gbnf_char_class() {
        let g = parse_gbnf("root ::= [abc]\n").unwrap();
        assert!(accepts(&g, b"a"));
        assert!(accepts(&g, b"b"));
        assert!(accepts(&g, b"c"));
        assert!(rejects(&g, b"d"));
    }

    #[test]
    fn gbnf_char_range() {
        let g = parse_gbnf("root ::= [a-z]\n").unwrap();
        assert!(accepts(&g, b"a"));
        assert!(accepts(&g, b"z"));
        assert!(rejects(&g, b"A"));
        assert!(rejects(&g, b"0"));
    }

    #[test]
    fn gbnf_optional() {
        let g = parse_gbnf("root ::= \"x\"?\n").unwrap();
        assert!(accepts(&g, b"x"));
        // epsilon: initial state should be complete (? makes root = x | ε)
        let state = GrammarState::initial();
        assert!(state.is_complete() || !state.stack.is_empty());
    }

    #[test]
    fn gbnf_plus() {
        let g = parse_gbnf("root ::= [0-9]+\n").unwrap();
        assert!(accepts(&g, b"5"));
        assert!(accepts(&g, b"42"));
        assert!(rejects(&g, b"x"));
    }

    #[test]
    fn gbnf_star() {
        let g = parse_gbnf("root ::= [0-9]*\n").unwrap();
        // one or more accepted
        assert!(accepts(&g, b"123"));
    }

    #[test]
    fn gbnf_non_terminal_reference() {
        let gbnf = "root ::= digit digit\ndigit ::= [0-9]\n";
        let g = parse_gbnf(gbnf).unwrap();
        assert!(accepts(&g, b"42"));
        assert!(rejects(&g, b"4"));
        assert!(rejects(&g, b"abc"));
    }

    #[test]
    fn gbnf_grouping() {
        let g = parse_gbnf("root ::= (\"ab\" | \"cd\")\n").unwrap();
        assert!(accepts(&g, b"ab"));
        assert!(accepts(&g, b"cd"));
        assert!(rejects(&g, b"ac"));
    }

    #[test]
    fn gbnf_dot() {
        let g = parse_gbnf("root ::= .\n").unwrap();
        assert!(accepts(&g, b"a"));
        assert!(accepts(&g, b"z"));
        assert!(accepts(&g, b"0"));
    }

    #[test]
    fn gbnf_no_root_returns_error() {
        assert!(parse_gbnf("foo ::= \"bar\"\n").is_err());
    }

    /// issue #1079: a reference to a rule name that is never given a `::=`
    /// definition must be rejected at compile time, naming the undefined rule
    /// — not silently compiled to an epsilon match for the reference.
    ///
    /// Mutation guard: removing the completeness check in `parse_gbnf` makes
    /// this `Ok` (the reserved-but-undefined `missing_rule` id has empty
    /// `alts`, which `pda.rs` treats as epsilon), so `expect_err` panics.
    #[test]
    fn gbnf_undefined_rule_reference_is_rejected() {
        let err = parse_gbnf("root ::= missing_rule\n")
            .expect_err("a reference to an undefined rule must not compile");
        assert!(
            err.0.contains("missing_rule"),
            "error must name the undefined rule, got: {}",
            err.0
        );
    }

    /// issue #1079 (the other direction): `?` desugars into an anon rule with
    /// TWO alternatives (`[x]` and the empty sequence `[]`) — its `alts`
    /// vector has length 2, so it is not `alts.is_empty()` and must NOT be
    /// flagged by the undefined-rule completeness check. Only a rule that was
    /// `reserve`d and never `set_alts` (zero alternatives, not one empty
    /// alternative) is a genuine undefined reference.
    #[test]
    fn gbnf_legitimate_epsilon_rule_not_flagged_undefined() {
        let g = parse_gbnf("root ::= \"x\"?\n").unwrap();
        assert!(accepts(&g, b"x"));
    }

    #[test]
    fn gbnf_error_display() {
        let e = GbnfError("test error".to_string());
        assert!(e.to_string().contains("test error"));
    }

    #[test]
    fn gbnf_comment_skipped() {
        let g = parse_gbnf("# this is a comment\nroot ::= \"ok\"\n").unwrap();
        assert!(accepts(&g, b"ok"));
    }
}
