use super::{
    Alt, GrammarBuilder, HashSet, MAX_ANYOF_BRANCHES, MAX_SCHEMA_DEPTH, MAX_STRING_LITERAL_BYTES,
    MAX_STRING_LITERALS, MAX_TRIE_DEPTH, REF_IGNORED_SIBLING_KEYS, SchemaError, Symbol, Value,
};

// ---------------------------------------------------------------------------
// Built-in primitive rules
// ---------------------------------------------------------------------------

/// Register primitive grammar rules into the builder.
pub(super) fn register_builtins(b: &mut GrammarBuilder) -> Result<(), SchemaError> {
    // ws = (' ' | '\t' | '\n' | '\r')* (zero or more whitespace bytes)
    let ws_id = b.reserve("ws");
    let ws_tail = b.reserve("ws_tail");
    b.set_alts(
        ws_tail,
        vec![
            vec![Symbol::Terminal(b' '), Symbol::NonTerminal(ws_tail)],
            vec![Symbol::Terminal(b'\t'), Symbol::NonTerminal(ws_tail)],
            vec![Symbol::Terminal(b'\n'), Symbol::NonTerminal(ws_tail)],
            vec![Symbol::Terminal(b'\r'), Symbol::NonTerminal(ws_tail)],
            vec![], // epsilon
        ],
    )?;
    b.set_alts(ws_id, vec![vec![Symbol::NonTerminal(ws_tail)]])?;

    // json_string = '"' string_inner '"'
    // string_inner = char* where char is one of:
    //   - an unescaped ASCII byte 0x20-0x7F excluding '"' and '\' (RFC 8259
    //     requires the control range U+0000-U+001F to be escaped)
    //   - a well-formed multi-byte UTF-8 sequence (2/3/4 bytes, restricted to
    //     the valid lead/continuation ranges from the Unicode UTF-8 table,
    //     issue #931)
    //   - a legal RFC 8259 escape: \" \\ \/ \b \f \n \r \t, or \u followed by
    //     exactly four hex digits.
    let str_id = b.reserve("json_string");
    let str_inner_id = b.reserve("json_string_inner");
    let str_char_id = b.reserve("json_string_char");

    // UTF-8 continuation byte: 0x80-0xBF.
    let cont_id = b.reserve("json_utf8_cont");
    b.set_alts(
        cont_id,
        (0x80u8..=0xBF)
            .map(|byte| vec![Symbol::Terminal(byte)])
            .collect(),
    )?;

    // Hex digit for \uXXXX escapes: 0-9, A-F, a-f.
    let hex_id = b.reserve("json_hex_digit");
    let hex_alts: Vec<Alt> = (b'0'..=b'9')
        .chain(b'A'..=b'F')
        .chain(b'a'..=b'f')
        .map(|byte| vec![Symbol::Terminal(byte)])
        .collect();
    b.set_alts(hex_id, hex_alts)?;

    // Legal escape body (the byte(s) following '\').
    let escape_id = b.reserve("json_escape");
    let mut escape_alts: Vec<Alt> = vec![
        vec![Symbol::Terminal(b'"')],
        vec![Symbol::Terminal(b'\\')],
        vec![Symbol::Terminal(b'/')],
        vec![Symbol::Terminal(b'b')],
        vec![Symbol::Terminal(b'f')],
        vec![Symbol::Terminal(b'n')],
        vec![Symbol::Terminal(b'r')],
        vec![Symbol::Terminal(b't')],
    ];
    escape_alts.push(vec![
        Symbol::Terminal(b'u'),
        Symbol::NonTerminal(hex_id),
        Symbol::NonTerminal(hex_id),
        Symbol::NonTerminal(hex_id),
        Symbol::NonTerminal(hex_id),
    ]);
    b.set_alts(escape_id, escape_alts)?;

    let mut char_alts: Vec<Alt> = Vec::new();

    // Escape sequence: '\' + a legal escape body.
    char_alts.push(vec![
        Symbol::Terminal(b'\\'),
        Symbol::NonTerminal(escape_id),
    ]);

    // Unescaped ASCII: 0x20-0x7F excluding '"' (0x22) and '\' (0x5c). Bytes
    // 0x00-0x1F are the RFC 8259 control range and MUST be escaped.
    for byte in 0x20u8..=0x7F {
        if byte != b'"' && byte != b'\\' {
            char_alts.push(vec![Symbol::Terminal(byte)]);
        }
    }

    // Well-formed 2-byte UTF-8: lead 0xC2-0xDF (0xC0-0xC1 would be an
    // overlong encoding of a 1-byte code point) + one continuation byte.
    // Each lead byte here is a distinct literal, so there is no shared-prefix
    // ambiguity among these alternatives (see the note on `e0_second_id`
    // below for why that distinction matters on this PDA).
    for lead in 0xC2u8..=0xDF {
        char_alts.push(vec![Symbol::Terminal(lead), Symbol::NonTerminal(cont_id)]);
    }

    // Well-formed 3-byte UTF-8 (Unicode Table 3-7), with the two lead bytes
    // that need a restricted first continuation byte routed through their
    // own sub-rule:
    //   0xE0        -> first continuation 0xA0-0xBF (else overlong)
    //   0xED        -> first continuation 0x80-0x9F (else a UTF-16 surrogate)
    //   0xE1-0xEC,
    //   0xEE-0xEF   -> first continuation 0x80-0xBF (the general case, cont_id)
    //
    // The restricted range MUST be its own `NonTerminal` rule rather than one
    // `str_char_id` alt per allowed second byte (e.g. 32 alts all starting
    // `Terminal(0xE0), Terminal(<second>), ...`). This PDA has no input
    // rewind (`consumed`-gated, see `try_next_alt`): once the shared
    // `Terminal(0xE0)` matches, the frame commits to whichever one of those
    // 32 alternatives it happened to try first, and a second byte that
    // matches a *different* alternative's literal is then rejected as a
    // mismatch instead of retried against a sibling alt. Pushing a fresh
    // frame for a sub-rule sidesteps this: the sub-rule frame has consumed
    // nothing yet when it starts trying its own alternatives.
    let e0_second_id = b.reserve("json_utf8_e0_second");
    b.set_alts(
        e0_second_id,
        (0xA0u8..=0xBF)
            .map(|byte| vec![Symbol::Terminal(byte)])
            .collect(),
    )?;
    let ed_second_id = b.reserve("json_utf8_ed_second");
    b.set_alts(
        ed_second_id,
        (0x80u8..=0x9F)
            .map(|byte| vec![Symbol::Terminal(byte)])
            .collect(),
    )?;
    char_alts.push(vec![
        Symbol::Terminal(0xE0),
        Symbol::NonTerminal(e0_second_id),
        Symbol::NonTerminal(cont_id),
    ]);
    char_alts.push(vec![
        Symbol::Terminal(0xED),
        Symbol::NonTerminal(ed_second_id),
        Symbol::NonTerminal(cont_id),
    ]);
    for lead in (0xE1u8..=0xEC).chain(0xEEu8..=0xEF) {
        char_alts.push(vec![
            Symbol::Terminal(lead),
            Symbol::NonTerminal(cont_id),
            Symbol::NonTerminal(cont_id),
        ]);
    }

    // Well-formed 4-byte UTF-8, with the two lead bytes that need a
    // restricted first continuation byte routed through their own sub-rule
    // (same shared-prefix reasoning as the 3-byte E0/ED case above):
    //   0xF0       -> first continuation 0x90-0xBF (else overlong)
    //   0xF4       -> first continuation 0x80-0x8F (caps code points at
    //                 U+10FFFF, the Unicode maximum)
    //   0xF1-0xF3  -> first continuation 0x80-0xBF (the general case, cont_id)
    let f0_second_id = b.reserve("json_utf8_f0_second");
    b.set_alts(
        f0_second_id,
        (0x90u8..=0xBF)
            .map(|byte| vec![Symbol::Terminal(byte)])
            .collect(),
    )?;
    let f4_second_id = b.reserve("json_utf8_f4_second");
    b.set_alts(
        f4_second_id,
        (0x80u8..=0x8F)
            .map(|byte| vec![Symbol::Terminal(byte)])
            .collect(),
    )?;
    char_alts.push(vec![
        Symbol::Terminal(0xF0),
        Symbol::NonTerminal(f0_second_id),
        Symbol::NonTerminal(cont_id),
        Symbol::NonTerminal(cont_id),
    ]);
    char_alts.push(vec![
        Symbol::Terminal(0xF4),
        Symbol::NonTerminal(f4_second_id),
        Symbol::NonTerminal(cont_id),
        Symbol::NonTerminal(cont_id),
    ]);
    for lead in 0xF1u8..=0xF3 {
        char_alts.push(vec![
            Symbol::Terminal(lead),
            Symbol::NonTerminal(cont_id),
            Symbol::NonTerminal(cont_id),
            Symbol::NonTerminal(cont_id),
        ]);
    }

    b.set_alts(str_char_id, char_alts)?;

    // json_string_inner = (json_string_char json_string_inner) | ε
    b.set_alts(
        str_inner_id,
        vec![
            vec![
                Symbol::NonTerminal(str_char_id),
                Symbol::NonTerminal(str_inner_id),
            ],
            vec![], // epsilon
        ],
    )?;
    b.set_alts(
        str_id,
        vec![vec![
            Symbol::Terminal(b'"'),
            Symbol::NonTerminal(str_inner_id),
            Symbol::Terminal(b'"'),
        ]],
    )?;

    // json_number = '-'? digit+ ('.' digit+)? (('e'|'E') ('+'|'-')? digit+)?
    // We approximate with a rule that matches the common case.
    let num_id = b.reserve("json_number");
    let digit_id = b.reserve("json_digit");
    let digit_alts: Vec<Alt> = (b'0'..=b'9')
        .map(|byte| vec![Symbol::Terminal(byte)])
        .collect();
    b.set_alts(digit_id, digit_alts)?;

    // json_digits = digit digit_tail
    // json_digit_tail = digit digit_tail | ε   (nullable tail ensures
    // is_accepting returns true after consuming ≥1 digit).
    let digits_id = b.reserve("json_digits");
    let digit_tail_id = b.reserve("json_digit_tail");
    b.set_alts(
        digit_tail_id,
        vec![
            vec![
                Symbol::NonTerminal(digit_id),
                Symbol::NonTerminal(digit_tail_id),
            ],
            vec![], // epsilon
        ],
    )?;
    b.set_alts(
        digits_id,
        vec![vec![
            Symbol::NonTerminal(digit_id),
            Symbol::NonTerminal(digit_tail_id),
        ]],
    )?;

    // json_nonzero = '1' | '2' | ... | '9'
    let nonzero_id = b.reserve("json_nonzero");
    let nonzero_alts: Vec<Alt> = (b'1'..=b'9')
        .map(|byte| vec![Symbol::Terminal(byte)])
        .collect();
    b.set_alts(nonzero_id, nonzero_alts)?;

    // json_int_part = '0' | nonzero digit_tail
    // Strict JSON integer part: forbids leading zeros (e.g. "01" rejected).
    // Alternatives differ on first byte ('0' vs 1-9), so the single-stack
    // PDA disambiguates them without backtracking (issue #310 finding #6).
    // json_digits is kept unchanged for fraction/exponent where leading zeros
    // ARE legal (e.g. "1.05", "1e08").
    let int_part_id = b.reserve("json_int_part");
    b.set_alts(
        int_part_id,
        vec![
            vec![Symbol::Terminal(b'0')],
            vec![
                Symbol::NonTerminal(nonzero_id),
                Symbol::NonTerminal(digit_tail_id),
            ],
        ],
    )?;

    // Optional sign
    let opt_sign_id = b.reserve("json_opt_sign");
    b.set_alts(opt_sign_id, vec![vec![Symbol::Terminal(b'-')], vec![]])?;

    // Optional fraction: '.' digits  (leading zeros legal here, e.g. 1.05)
    let opt_frac_id = b.reserve("json_opt_frac");
    b.set_alts(
        opt_frac_id,
        vec![
            vec![Symbol::Terminal(b'.'), Symbol::NonTerminal(digits_id)],
            vec![],
        ],
    )?;

    // Optional exponent: (e|E) (sign?) digits  (leading zeros legal, e.g. 1e08)
    let exp_sign_id = b.reserve("json_exp_sign");
    b.set_alts(
        exp_sign_id,
        vec![
            vec![Symbol::Terminal(b'+')],
            vec![Symbol::Terminal(b'-')],
            vec![],
        ],
    )?;
    let opt_exp_id = b.reserve("json_opt_exp");
    b.set_alts(
        opt_exp_id,
        vec![
            vec![
                Symbol::Terminal(b'e'),
                Symbol::NonTerminal(exp_sign_id),
                Symbol::NonTerminal(digits_id),
            ],
            vec![
                Symbol::Terminal(b'E'),
                Symbol::NonTerminal(exp_sign_id),
                Symbol::NonTerminal(digits_id),
            ],
            vec![],
        ],
    )?;

    // json_number = '-'? int_part ('.' digits)? (('e'|'E') sign? digits)?
    // Uses int_part (not digits) for the integer part to reject leading zeros.
    b.set_alts(
        num_id,
        vec![vec![
            Symbol::NonTerminal(opt_sign_id),
            Symbol::NonTerminal(int_part_id),
            Symbol::NonTerminal(opt_frac_id),
            Symbol::NonTerminal(opt_exp_id),
        ]],
    )?;

    // json_integer = '-'? int_part  (rejects leading zeros)
    let int_id = b.reserve("json_integer");
    b.set_alts(
        int_id,
        vec![vec![
            Symbol::NonTerminal(opt_sign_id),
            Symbol::NonTerminal(int_part_id),
        ]],
    )?;

    // json_boolean = "true" | "false"
    let bool_id = b.reserve("json_boolean");
    b.set_alts(bool_id, vec![bytes_to_alt(b"true"), bytes_to_alt(b"false")])?;

    // json_null = "null"
    let null_id = b.reserve("json_null");
    b.set_alts(null_id, vec![bytes_to_alt(b"null")])?;
    Ok(())
}

fn bytes_to_alt(bytes: &[u8]) -> Alt {
    bytes.iter().map(|&b| Symbol::Terminal(b)).collect()
}

pub(super) fn empty_object_as_alt_with_ws(ws_id: usize) -> Alt {
    vec![
        Symbol::Terminal(b'{'),
        Symbol::NonTerminal(ws_id),
        Symbol::Terminal(b'}'),
    ]
}

// ---------------------------------------------------------------------------
// Trie compiler for shared-prefix literal alternatives
// ---------------------------------------------------------------------------

/// Build one node of a literal trie into `builder`.
///
/// Each node groups its input sequences by first byte. A group becomes one
/// alternative: `[Terminal(byte)]` when every member ends here (leaf); a flat
/// `[Terminal(byte), Terminal(b0), Terminal(b1), ...]` when exactly one member
/// continues (single-child compression, the whole remaining tail emitted inline
/// instead of recursing); or `[Terminal(byte), NonTerminal(child)]` when two or
/// more members continue with distinct bytes (inner). Because every group's
/// alternatives diverge on their FIRST byte, the no-rewind single-stack PDA
/// picks the correct branch at `sym_pos == 0` without ever needing to backtrack.
///
/// Single-child compression bounds recursion depth by the longest SHARED prefix
/// rather than the longest member, so one long literal compiles to a single flat
/// alternative (matching the pre-trie encoding) and a member's length never
/// drives recursion. `depth` carries the current nesting level so the
/// shared-prefix recursion fails closed at `MAX_TRIE_DEPTH` instead of
/// overflowing the stack.
///
/// Callers guarantee the sequences are prefix-free (no sequence is a proper
/// prefix of another), so within any group either ALL tails are empty (leaf) or
/// ALL are non-empty (compressed or inner); mixed groups are impossible.
///
/// Rule-naming convention: `node_counter == 0` → root → `str_enum_{n}`;
/// subsequent inner nodes → `str_enum_{n}_{k}`.
pub(super) fn build_trie_node(
    builder: &mut GrammarBuilder,
    seqs: &[Vec<u8>],
    enum_n: usize,
    node_counter: &mut usize,
    depth: usize,
) -> Result<usize, SchemaError> {
    // Fail closed rather than overflow the stack. With single-child compression
    // below, recursion descends one level only per byte of a prefix SHARED by
    // two or more members, so this bound is unreachable for realistic schemas
    // and trips only on an adversarial set sharing an absurdly long prefix.
    if depth >= MAX_TRIE_DEPTH {
        return Err(SchemaError(format!(
            "enum/const shared-prefix nesting exceeds the supported depth ({MAX_TRIE_DEPTH})"
        )));
    }

    let k = *node_counter;
    *node_counter += 1;
    let rule_name = if k == 0 {
        format!("str_enum_{enum_n}")
    } else {
        format!("str_enum_{enum_n}_{k}")
    };
    let id = builder.reserve(&rule_name);

    // Group sequences by first byte (stable insertion order).
    let mut groups: Vec<(u8, Vec<Vec<u8>>)> = Vec::new();
    for seq in seqs {
        let Some((&first, tail)) = seq.split_first() else {
            return Err(SchemaError(
                "trie: empty literal sequence (prefix-free invariant violated)".into(),
            ));
        };
        match groups.iter_mut().find(|(b, _)| *b == first) {
            Some(entry) => entry.1.push(tail.to_vec()),
            None => groups.push((first, vec![tail.to_vec()])),
        }
    }

    let mut alts: Vec<Alt> = Vec::with_capacity(groups.len());
    for (byte, tails) in groups {
        if tails.iter().all(Vec::is_empty) {
            // Leaf: every member ends at this byte (exact duplicates collapse).
            alts.push(vec![Symbol::Terminal(byte)]);
        } else if tails.len() == 1 {
            // Single continuation: emit the whole remaining tail flat instead of
            // recursing, so recursion depth tracks the longest SHARED prefix,
            // not member length. A lone long literal becomes one flat
            // alternative and cannot drive deep recursion or per-byte rule blowup.
            let tail = &tails[0];
            let mut alt = Vec::with_capacity(tail.len() + 1);
            alt.push(Symbol::Terminal(byte));
            alt.extend(tail.iter().map(|&b| Symbol::Terminal(b)));
            alts.push(alt);
        } else {
            // Two or more continuations share this byte: recurse one level.
            let child_id = build_trie_node(builder, &tails, enum_n, node_counter, depth + 1)?;
            alts.push(vec![Symbol::Terminal(byte), Symbol::NonTerminal(child_id)]);
        }
    }

    builder.set_alts(id, alts)?;
    Ok(id)
}

/// Compile an `enum` schema (values can be any JSON type).
///
/// When all enum values are JSON strings, the opening and closing `"`
/// delimiters are factored out into the calling context and the alternatives
/// are trie-compiled so every branch diverges on its first byte.  This
/// avoids alternative ambiguity when multiple string values share a prefix.
///
/// For mixed-type or non-string enums the naive per-value serialisation is
/// used; those values have distinct first bytes in the common case and are
/// unambiguous.
///
/// # Residual limitation — unquoted numeric enums with a shared prefix
///
/// Integer or float enum members such as `[1, 10]` serialise to `1` and `10`
/// in the byte stream.  `1` IS a prefix of `10`, but there is no in-rule
/// terminator byte to make them prefix-free: the disambiguator is the
/// following delimiter character (`]`, `,`, `}`) which belongs to the
/// SURROUNDING grammar context, not to this rule.  The compile-time trie fix
/// therefore does not apply here.  A complete fix requires parallel active
/// stacks or an NFA-based matcher that can hold both possibilities open until
/// the surrounding context resolves them.  Until then, `enum [1, 10]` remains
/// a narrower known limitation: `10` is over-rejected by the no-rewind PDA
/// (the same safe direction as before).
pub(super) fn compile_enum(values: &[Value]) -> Result<Vec<Alt>, SchemaError> {
    // All-string enums are handled upstream via compile_string_type, which
    // trie-compiles the alternatives to avoid PDA ambiguity.  This path is
    // reached only for mixed-type or non-string enum values (e.g., const
    // arrays containing integers or null), whose first bytes are distinct in
    // the common case.
    //
    // DoS bound (issue #474): reject by raw cardinality before allocating one
    // `Alt` per member below. This is the same sibling-allocation pattern as
    // the string-enum and anyOf paths, applied to the non-string branch: a
    // mixed or non-string `enum` still reaches a per-member materialization
    // step that must be bounded up front, not after the Vec is built.
    if values.len() > MAX_STRING_LITERALS {
        return Err(SchemaError(format!(
            "enum value count ({}) exceeds the supported limit ({MAX_STRING_LITERALS})",
            values.len()
        )));
    }
    // DoS bound (issue #474, finding 3): check the total encoded byte budget
    // before allocating one `Alt` (a `Vec<Symbol>` sized by the value's JSON
    // byte length) per member below. The count cap above bounds how many
    // members there are but not how large each one is, so a handful of
    // oversized values under MAX_STRING_LITERALS can still drive an
    // unbounded allocation in the map/collect that follows.
    // `compile_string_type` enforces the analogous budget for the all-string
    // path; this mirrors it for the mixed/non-string path it does not cover.
    // saturating_add prevents an overflow in the running sum from wrapping
    // past the cap.
    let mut pre_byte_total: usize = 0;
    for v in values {
        let json_repr = serde_json::to_string(v)
            .map_err(|e| SchemaError(format!("cannot JSON-encode enum value: {e}")))?;
        pre_byte_total = pre_byte_total.saturating_add(json_repr.len());
        if pre_byte_total > MAX_STRING_LITERAL_BYTES {
            return Err(SchemaError(format!(
                "enum value encoded byte length exceeds the supported limit ({MAX_STRING_LITERAL_BYTES})"
            )));
        }
    }
    let alts = values
        .iter()
        .map(json_value_to_alt)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(alts)
}

/// Compile a `const` schema (a single fixed JSON value).
pub(super) fn compile_const(v: &Value) -> Result<Vec<Alt>, SchemaError> {
    Ok(vec![json_value_to_alt(v)?])
}

/// How an `anyOf`/`oneOf` sub-schema matches strings, for collapsing the
/// string-valued ambiguity class into a single `"` entry (see the `anyOf`
/// handler). `type`, `const`, and `enum` are conjunctive assertions, so a
/// sub-schema's string language is the INTERSECTION of the constraints it
/// states. A schema that may also accept a non-string value, or whose string
/// language is not reducible to a broad string or a fixed literal set from a
/// SINGLE sub-schema in isolation (a `$ref`, a nested `anyOf`/`oneOf`, an
/// untyped `{}`, a `{"type":["string",...]}` union, or an untyped mixed
/// `enum`) classifies as `None` here and stays an "other" branch by default.
/// `compile_any_of`'s caller loop gives three of those shapes — a `$ref`, a
/// nested `anyOf`/`oneOf`, and an untyped mixed `enum` — a second chance to
/// fold anyway (issue #473; see `ref_string_contribution`,
/// `flatten_any_of_branches`, `fold_string_members`) before finally treating
/// the branch as "other". Only a genuinely untyped `{}` and a
/// `{"type":["string",...]}` union remain unfoldable at compile time — both
/// need parallel-stack matching, not a literal fold.
pub(super) enum StrClass {
    /// Any JSON string (`{"type":"string"}` with no value constraint).
    /// `pattern` / `minLength` are not enforced by this compiler, so such a
    /// schema widens to any string — the same `json_string` rule
    /// `compile_string_type` emits for it standalone.
    Broad,
    /// A fixed set of string values (a string `const`, an all-string `enum`, or
    /// a `{"type":"string","enum":[...]}` intersected down to its string
    /// members). An EMPTY set is a dead branch — `const`/`enum` conflict, or a
    /// `type:"string"` enum with no string member — whose language is empty, so
    /// it contributes nothing to the union (it must NOT widen to any string).
    Literals(Vec<String>),
}

/// Classify a raw `anyOf`/`oneOf` sub-schema for string-class merging.
///
/// `type`, `const`, and `enum` are conjunctive assertions, so the branch's
/// string language is their INTERSECTION — this models that intersection rather
/// than taking the first keyword seen. A branch that may accept a non-string
/// value, or whose string language is not reducible to a broad string or a
/// fixed literal set, returns `None` and stays an "other". A dead branch (empty
/// intersection, e.g. `const`/`enum` conflict or a `type:"string"` enum with no
/// string member) returns `Literals(vec![])` so it contributes nothing — it
/// must NOT fall through to `None`, which would let the standalone path widen it
/// to any string.
pub(super) fn string_class_of(sub: &Value) -> Result<Option<StrClass>, SchemaError> {
    let Some(obj) = sub.as_object() else {
        return Ok(None);
    };
    // A `$ref` alongside `type`/`const`/`enum` needs the referenced target in
    // order to compute the conjunction, so this single-node classifier cannot
    // handle it. Return `None` so `compile_any_of` reclassifies via
    // `ref_string_contribution`, which resolves the target and hoists only a
    // faithfully intersected string set. Non-string cases fall through to the
    // general `$ref` path, which fails closed rather than widening (issue #473).
    if obj.contains_key("$ref") {
        return Ok(None);
    }
    // `type` gate: only an absent type or exactly `"string"` keeps the branch a
    // pure string class. A scalar non-string type, or a `type` array (for which
    // `as_str()` is `None` on a non-string `Value`), means the branch may accept
    // a non-string value, so it cannot fold into the string entry.
    let type_is_string = match obj.get("type") {
        None => false,
        Some(Value::String(s)) if s == "string" => true,
        _ => return Ok(None),
    };
    let const_val = obj.get("const");
    let enum_arr = obj.get("enum").and_then(Value::as_array);
    // DoS bound (issue #474): reject by raw JSON cardinality before either match
    // arm below can reach a step sized by this array — the const-narrowing
    // `.any()` scan in the `(Some(c), _)` arm, or the filter_map/to_string
    // deep-copy in the `(None, Some(arr))` arm. A sibling `const` narrows which
    // VALUES survive but never shrinks the RAW array itself, so checking
    // `arr.len()` here, ahead of the match, bounds both arms uniformly.
    if let Some(arr) = enum_arr
        && arr.len() > MAX_STRING_LITERALS
    {
        return Err(SchemaError(format!(
            "anyOf string literal count ({}) exceeds the supported limit ({MAX_STRING_LITERALS})",
            arr.len()
        )));
    }
    match (const_val, enum_arr) {
        (Some(c), enum_opt) => {
            // `const` is a single-value enum. The language is the intersection of
            // {const}, the `enum` (when present), and the type.
            let Some(cs) = c.as_str() else {
                // Non-string const: with `type:"string"` nothing satisfies both
                // (dead branch); with no type the branch accepts that non-string
                // value, which this string fold cannot represent.
                return Ok(if type_is_string {
                    Some(StrClass::Literals(Vec::new()))
                } else {
                    None
                });
            };
            // String const: fold it only if a present `enum` also contains it; a
            // conflicting `enum` makes the intersection empty (dead branch).
            Ok(match enum_opt {
                Some(arr) if !arr.iter().any(|v| v == c) => Some(StrClass::Literals(Vec::new())),
                _ => {
                    guard_literal_bytes(cs)?;
                    Some(StrClass::Literals(vec![cs.to_string()]))
                }
            })
        }
        (None, Some(arr)) => {
            if type_is_string {
                // `type:"string"` kills every non-string member; the language is
                // exactly the string members (possibly empty = dead branch).
                let mut lits = Vec::new();
                for v in arr {
                    if let Some(s) = v.as_str() {
                        guard_literal_bytes(s)?;
                        lits.push(s.to_string());
                    }
                }
                Ok(Some(StrClass::Literals(lits)))
            } else if !arr.is_empty() && arr.iter().all(Value::is_string) {
                // Untyped all-string enum: a pure literal set.
                let mut lits = Vec::new();
                for v in arr {
                    if let Some(s) = v.as_str() {
                        guard_literal_bytes(s)?;
                        lits.push(s.to_string());
                    }
                }
                Ok(Some(StrClass::Literals(lits)))
            } else {
                // Untyped mixed or empty enum: non-string members stay reachable
                // as an "other" branch (folding would drop those acceptances).
                Ok(None)
            }
        }
        (None, None) => Ok(if type_is_string {
            Some(StrClass::Broad)
        } else {
            None
        }),
    }
}

/// The string-language contribution of a `$ref`-bearing `anyOf` branch, as
/// computed by `ref_string_contribution`: the resolved terminal's string class
/// intersected with the `const`/`enum` narrowing collected along the chain
/// (issue #473).
pub(super) enum RefStr {
    /// The terminal may accept a non-string value and no chain node forced a
    /// string: contribute no string hoist and leave the branch in `other_subs`
    /// for `compile_schema` to handle. Also returned for an
    /// unresolvable/cyclic chain, so the `$ref` error still surfaces on that
    /// path.
    NotString,
    /// A chain node forces a string (`type:"string"`, a string `const`, or an
    /// all-string `enum`) but the terminal is non-string: the conjunction is the
    /// EMPTY language, so the caller drops the branch entirely (it contributes
    /// nothing and need not reach the generic fail-closed `$ref` path).
    Dead,
    /// The terminal is a broad string and no `const`/`enum` narrowed it: hoist a
    /// broad string entry, exactly as compiling the terminal would.
    Broad,
    /// The terminal string intersected with the chain narrowing is this fixed
    /// set (possibly empty — a dead string branch that hoists nothing).
    Literals(Vec<String>),
}

/// Combine a terminal's `string_class_of` result with the `const`/`enum`
/// narrowing collected along a `$ref` chain. Intersection can only shrink the
/// terminal's string language, so the result never over-accepts relative to
/// compiling the terminal directly (issue #473). `chain_forces_string` is true
/// when any node along the chain constrains the value to a string; conjoined
/// with a non-string terminal that is the empty language (`Dead`).
pub(super) fn combine_ref_str(
    terminal: Option<StrClass>,
    narrowing: Option<Vec<String>>,
    chain_forces_string: bool,
) -> RefStr {
    match terminal {
        None if chain_forces_string => RefStr::Dead,
        None => RefStr::NotString,
        Some(StrClass::Broad) => match narrowing {
            None => RefStr::Broad,
            Some(set) => RefStr::Literals(set),
        },
        Some(StrClass::Literals(term_lits)) => match narrowing {
            None => RefStr::Literals(term_lits),
            Some(set) => RefStr::Literals(intersect_string_sets(&term_lits, &set)),
        },
    }
}

/// The set of strings a SINGLE schema node permits via its own `const`/`enum`,
/// ignoring any `$ref` (whose target is classified separately). `None` = the
/// node states neither `const` nor `enum`, so it imposes no string narrowing.
/// `Some(set)` = strings are restricted to exactly `set` (the string members of
/// the `const`/`enum`); an EMPTY `set` means the node admits no string at all (a
/// non-string `const`, a `const`/`enum` conflict, or an enum with no string
/// member), which makes the whole chain a dead string branch.
pub(super) fn node_string_narrowing(node: &Value) -> Result<Option<Vec<String>>, SchemaError> {
    let Some(obj) = node.as_object() else {
        return Ok(None);
    };
    let const_val = obj.get("const");
    let enum_arr = obj.get("enum").and_then(Value::as_array);
    // DoS bound (issue #474), mirroring `string_class_of`: reject by raw enum
    // cardinality before the per-element scan/collect below can be sized by it.
    if let Some(arr) = enum_arr
        && arr.len() > MAX_STRING_LITERALS
    {
        return Err(SchemaError(format!(
            "anyOf string literal count ({}) exceeds the supported limit ({MAX_STRING_LITERALS})",
            arr.len()
        )));
    }
    let set: Vec<String> = match (const_val, enum_arr) {
        (Some(c), enum_opt) => match c.as_str() {
            // Non-string const: no string value satisfies it.
            None => Vec::new(),
            // String const: kept only if a present enum also contains it.
            Some(cs) => match enum_opt {
                Some(arr) if !arr.iter().any(|v| v == c) => Vec::new(),
                _ => vec![cs.to_string()],
            },
        },
        (None, Some(arr)) => arr
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        // No `const` and no `enum`: this node imposes no string narrowing.
        (None, None) => return Ok(None),
    };
    for s in &set {
        guard_literal_bytes(s)?;
    }
    Ok(Some(set))
}

/// Intersect `new` into the running narrowing accumulator. `None` in either
/// position is the identity ("no `const`/`enum` stated"), so it never widens the
/// accumulator; a stated set can only shrink it.
pub(super) fn intersect_narrowing(acc: &mut Option<Vec<String>>, new: Option<Vec<String>>) {
    let Some(new_set) = new else {
        return;
    };
    match acc {
        None => *acc = Some(new_set),
        Some(cur) => cur.retain(|s| new_set.contains(s)),
    }
}

/// The intersection of two string sets, preserving `a`'s order.
fn intersect_string_sets(a: &[String], b: &[String]) -> Vec<String> {
    a.iter().filter(|s| b.contains(s)).cloned().collect()
}

/// Does this node's OWN keywords force the value to be a string?
/// True iff the node states a string-ONLY `type` (the scalar `"string"`, or —
/// per draft-2020-12 §6.1.1, where `type` may be an array of type names — a
/// NON-EMPTY array whose every member is `"string"`), a string `const`, or a
/// non-empty `enum` whose members are all strings. When such a node sits
/// ANYWHERE on a `$ref` chain (the outer sub or an intermediate `$defs` link)
/// whose terminal is a NON-string target, the conjunction is the empty
/// language, so `compile_any_of` drops the branch exactly. A pure `$ref`, or
/// one whose keyword admits a non-string — a numeric `const`, an all-numeric
/// or mixed `enum`, or a `type` array that permits any non-string type
/// (`["string","null"]`, `["integer"]`, …) — returns `false`; the generic
/// compiler then either compiles the pure reference or rejects the narrowing
/// sibling it cannot represent.
pub(super) fn ref_sub_forces_string(sub: &Value) -> bool {
    let Some(obj) = sub.as_object() else {
        return false;
    };
    match obj.get("type") {
        Some(Value::String(s)) if s == "string" => return true,
        Some(Value::Array(arr))
            if !arr.is_empty() && arr.iter().all(|v| v.as_str() == Some("string")) =>
        {
            return true;
        }
        _ => {}
    }
    if let Some(c) = obj.get("const") {
        return c.is_string();
    }
    if let Some(arr) = obj.get("enum").and_then(Value::as_array) {
        return !arr.is_empty() && arr.iter().all(Value::is_string);
    }
    false
}

/// Recognize a branch that is a PURE nested `anyOf` union: an object with
/// EXACTLY one key, `anyOf`, mapping to an array. Only `anyOf` (inclusive
/// union) is recognized, because flattening it into the parent branch list is
/// unconditionally sound: a union of unions is the same union.
///
/// A nested `oneOf` is intentionally NOT recognized. `oneOf` is exclusive
/// ("exactly one branch matches"), and merging its branches into the parent's
/// OR-hoist silently drops that exclusion, over-accepting any input two
/// branches share (a no-rewind PDA cannot count matches to enforce it). The
/// safe direction is to leave every nested `oneOf` unflattened. A sibling key
/// alongside the `anyOf` (e.g. a co-occurring `description` or `type`)
/// likewise makes the branch not purely a union. Both non-recognized cases
/// fall through to the pre-existing `other_subs` path unchanged, which
/// over-rejects but never over-accepts (issue #473, shape 2).
fn as_pure_nested_union(sub: &Value) -> Option<&[Value]> {
    let obj = sub.as_object()?;
    if obj.len() != 1 {
        return None;
    }
    let arr = obj.get("anyOf")?;
    arr.as_array().map(Vec::as_slice)
}

/// Flatten `any_of`'s pure nested-union branches (`as_pure_nested_union`)
/// into `out`, so `compile_any_of`'s classification loop folds a broad or
/// literal string nested inside `{"anyOf":[{"anyOf":[{"type":"string"}]}]}`
/// exactly like a top-level branch (issue #473, shape 2). Non-union branches
/// pass through unchanged.
///
/// Enforces `MAX_ANYOF_BRANCHES` on the FLATTENED total incrementally as
/// `out` grows: checking only the raw top-level `any_of.len()` (the prior
/// single guard in `compile_any_of`) would let a small top-level array with
/// a wide nested union bypass the cap (issue #474 class). Recursion depth is
/// bounded by `MAX_SCHEMA_DEPTH` independent of `self.depth` — this
/// traversal walks the raw `Value` tree directly and never goes through
/// `compile_schema`'s depth-counting wrapper, so it needs its own bound to
/// reject a pathological `{"anyOf":[{"anyOf":[{"anyOf":[...]}]}]}` chain
/// before the native stack overflows, mirroring why that constant exists.
pub(super) fn flatten_any_of_branches<'v>(
    any_of: &'v [Value],
    depth: usize,
    out: &mut Vec<&'v Value>,
) -> Result<(), SchemaError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaError(format!(
            "anyOf/oneOf nesting depth exceeds the supported depth ({MAX_SCHEMA_DEPTH})"
        )));
    }
    for sub in any_of {
        match as_pure_nested_union(sub) {
            Some(inner) => flatten_any_of_branches(inner, depth + 1, out)?,
            None => out.push(sub),
        }
        if out.len() > MAX_ANYOF_BRANCHES {
            return Err(SchemaError(format!(
                "anyOf/oneOf branch count ({}) exceeds the supported limit ({MAX_ANYOF_BRANCHES})",
                out.len()
            )));
        }
    }
    Ok(())
}

/// The exact JSON values a branch is restricted to, if (and only if) that set
/// is closed-form and enumerable from the schema alone — a `const` (one
/// value) or an `enum` (a fixed list). Any other shape (a bare `type`, an
/// object/array schema, an unresolved `$ref`, ...) can admit an open-ended set
/// of values, so this returns `None` rather than guess: overlap between two
/// `None` branches is exactly the case #1077 leaves undetected and documented,
/// and it must stay silent rather than produce a false positive.
/// The set is the CONJUNCTION of every assertion in the branch, not the
/// `const`/`enum` alone: a branch also asserting `"type"` matches only the
/// literals of that type, and a branch whose assertions contradict each other
/// (`{"const": "a", "type": "integer"}`) matches nothing at all and returns the
/// empty set. Reporting the bare literal there would let the overlap check
/// reject a `oneOf` whose branches are in fact mutually exclusive. A branch
/// carrying an assertion this function cannot intersect returns `None` and is
/// skipped for the same reason: an unanalysed assertion may exclude the
/// literal, so claiming the branch matches it would manufacture an overlap.
fn closed_form_values(schema: &Value) -> Option<Vec<Value>> {
    let mut values: Vec<Value> = if let Some(v) = schema.get("const") {
        vec![v.clone()]
    } else if let Some(a) = schema.get("enum").and_then(Value::as_array) {
        a.clone()
    } else {
        return None;
    };

    for k in schema.as_object()?.keys() {
        let k = k.as_str();
        if k == "const" || k == "enum" || k == "type" || REF_IGNORED_SIBLING_KEYS.contains(&k) {
            continue;
        }
        return None;
    }

    // `const` alongside `enum`: the value must satisfy both.
    if schema.get("const").is_some()
        && let Some(enum_values) = schema.get("enum").and_then(Value::as_array)
    {
        values.retain(|v| enum_values.contains(v));
    }

    match schema.get("type") {
        None => {}
        Some(Value::String(t)) => values.retain(|v| value_matches_type(v, t)),
        Some(Value::Array(ts)) => values.retain(|v| {
            ts.iter()
                .any(|t| t.as_str().is_some_and(|t| value_matches_type(v, t)))
        }),
        Some(_) => return None,
    }

    Some(values)
}

/// Whether `v` is an instance of the JSON Schema primitive type named `t`.
/// An unrecognized type name matches nothing: the compiler rejects it as an
/// unsupported type on the normal path, and reporting no match here only ever
/// removes a literal from the overlap check.
pub(super) fn value_matches_type(v: &Value, t: &str) -> bool {
    match t {
        "string" => v.is_string(),
        "integer" => v.as_i64().is_some() || v.as_u64().is_some(),
        "number" => v.is_number(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "object" => v.is_object(),
        "array" => v.is_array(),
        _ => false,
    }
}

/// Parse a non-empty JSON Schema `type` assertion without applying a subtype
/// lattice. Literal name membership is the only proof used for `$ref` sibling
/// redundancy.
pub(super) fn schema_type_names(value: &Value) -> Result<Vec<&str>, SchemaError> {
    match value {
        Value::String(name) => Ok(vec![name]),
        Value::Array(names) if !names.is_empty() => names
            .iter()
            .map(|name| {
                name.as_str().ok_or_else(|| {
                    SchemaError(
                        "unsupported schema: `type` alongside `$ref` must contain only type names"
                            .to_string(),
                    )
                })
            })
            .collect(),
        _ => Err(unrepresentable_ref_narrowing("type")),
    }
}

pub(super) fn unrepresentable_ref_narrowing(key: &str) -> SchemaError {
    SchemaError(format!(
        "unsupported schema: `{key}` sibling alongside `$ref` requires an intersection with the referenced schema that this grammar cannot represent safely"
    ))
}

/// Reject a `oneOf` whose branches provably overlap on a literal value (issue
/// #1077). Only branches that reduce to a closed `const`/`enum` value set are
/// decidable here. An open-ended branch (`closed_form_values` returns `None`)
/// contributes nothing and is skipped, while the branches around it are still
/// checked against each other. Skipping rather than abandoning the whole check
/// at the first such branch is what keeps the outcome independent of where an
/// open-ended branch sits: `oneOf`'s branches are an unordered set, so the same
/// schema written in a different order has to compile the same way.
///
/// Rejecting on a shared literal is sound whatever the other branches admit —
/// that value matches more than one branch, which is precisely what `oneOf`
/// forbids. The converse does not hold: compiling is not a claim that
/// exclusivity is enforced. Overlap between two open-ended branches stays
/// undetected, which is the limitation #1077 leaves standing: a `oneOf` mixing
/// an `object` branch with a `const` branch compiles exactly as `anyOf` does
/// today, per the module doc's known-limitations section.
///
/// Bounded against the same DoS class as `flatten_any_of_branches`
/// (issue #474): branch count is capped at `MAX_ANYOF_BRANCHES` and the
/// cumulative literal count at `MAX_STRING_LITERALS`, both by skipping the
/// check (not erroring) once exceeded — the pre-existing `compile_any_of`
/// guards still reject an oversized schema on the normal path.
pub(super) fn check_one_of_exclusivity(branches: &[Value]) -> Result<(), SchemaError> {
    if branches.len() > MAX_ANYOF_BRANCHES {
        return Ok(());
    }
    let mut seen: HashSet<String> = HashSet::new();
    for (i, b) in branches.iter().enumerate() {
        let Some(values) = closed_form_values(b) else {
            continue;
        };
        if seen.len().saturating_add(values.len()) > MAX_STRING_LITERALS {
            return Ok(());
        }
        // A branch's own values join `seen` only once the whole branch has
        // been measured against it. A value repeated inside ONE branch
        // (`{"enum":["a","a"]}`, which draft 2020-12 permits — `enum` members
        // SHOULD be unique, not MUST) does not give an instance a second
        // matching branch, so folding the branch in value-by-value would make
        // it collide with itself and report an earlier branch that need not
        // exist.
        let mut branch_keys: Vec<String> = Vec::with_capacity(values.len());
        for v in &values {
            let key = overlap_key(v)?;
            if seen.contains(&key) {
                return Err(SchemaError(format!(
                    "oneOf branch {i} overlaps an earlier branch on literal value {v}: oneOf requires mutually exclusive branches"
                )));
            }
            branch_keys.push(key);
        }
        seen.extend(branch_keys);
    }
    Ok(())
}

/// The key on which two `oneOf` branches' literals are held to collide.
///
/// JSON Schema compares instances by value, and numbers are arbitrary-precision
/// decimals compared by mathematical value, so `1` and `1.0` are the SAME
/// instance. `serde_json`'s rendering preserves the spelling a number was
/// written with (`"1"` against `"1.0"`), which would let those two branches
/// pass as distinct. A number whose mathematical value is an integer inside
/// `i64` range is therefore keyed by that integer.
///
/// Deliberately NOT canonicalized through `f64`: past 2^53 distinct integers
/// share a single `f64`, and a collision here reports an overlap between
/// branches that are in fact exclusive — rejecting a correct schema. That is
/// the worse direction of error, so the conversion is applied only where it is
/// exact and every other value falls back to its rendering, which at worst
/// leaves an overlap undetected.
fn overlap_key(v: &Value) -> Result<String, SchemaError> {
    if let Value::Number(n) = v {
        if let Some(i) = n.as_i64() {
            return Ok(format!("n:{i}"));
        }
        if let Some(u) = n.as_u64() {
            return Ok(format!("n:{u}"));
        }
        // `f as i64` saturates rather than wrapping, so the range test is what
        // keeps two far-apart magnitudes off the same key. Both bounds are
        // exactly representable, and every `f64` in the half-open range whose
        // fractional part is zero converts to `i64` without loss.
        if let Some(f) = n.as_f64()
            && f.fract() == 0.0
            && f >= -(2f64.powi(63))
            && f < 2f64.powi(63)
        {
            return Ok(format!("n:{}", f as i64));
        }
    }
    serde_json::to_string(v)
        .map(|s| format!("j:{s}"))
        .map_err(|e| SchemaError(format!("cannot JSON-encode oneOf literal: {e}")))
}

/// Fold `values` into the running `literals`/`byte_total` accumulators shared
/// by every string-class-merging path in `compile_any_of` — a direct
/// `StrClass::Literals` branch, a resolved `$ref` target (issue #473 shape
/// 1), or a mixed-enum's string members (issue #473 shape 5). Enforces
/// `MAX_STRING_LITERALS` and `MAX_STRING_LITERAL_BYTES` incrementally (issue
/// #474/#478) so no caller path can accumulate literals past either cap by
/// deferring to a later check.
pub(super) fn fold_literals_into(
    literals: &mut Vec<String>,
    byte_total: &mut usize,
    values: Vec<String>,
) -> Result<(), SchemaError> {
    // DoS bound (issue #478): reject before the extend so that many small
    // foldable sources cannot accumulate `literals` past the cap via repeated
    // `Vec::extend` growth before the post-loop backstop in `compile_any_of`
    // ever runs.
    let total = literals.len() + values.len();
    if total > MAX_STRING_LITERALS {
        return Err(SchemaError(format!(
            "anyOf string literal count ({total}) exceeds the supported limit ({MAX_STRING_LITERALS})"
        )));
    }
    // DoS bound (issue #474, finding 4): accumulate the byte budget
    // incrementally as each source's literals are folded in, not only after
    // the loop ends. saturating_add prevents an overflow in the running sum
    // from wrapping past the cap.
    for s in &values {
        let json_repr = serde_json::to_string(s)
            .map_err(|e| SchemaError(format!("cannot JSON-encode string value: {e}")))?;
        *byte_total = byte_total.saturating_add(json_repr.len() - 1);
        if *byte_total > MAX_STRING_LITERAL_BYTES {
            return Err(SchemaError(format!(
                "string literal encoded byte length exceeds the supported limit ({MAX_STRING_LITERAL_BYTES})"
            )));
        }
    }
    literals.extend(values);
    Ok(())
}

/// For a branch that `string_class_of` classifies as `None` because it is an
/// untyped `enum` mixing string and non-string members (issue #473, shape 5),
/// return its definitely-accepted STRING members so the caller can fold them
/// into the hoisted string entry. The branch itself still goes to
/// `other_subs` unchanged — its non-string members have first bytes disjoint
/// from `"` and stay reachable there, so duplicating the strings into the
/// trie only ADDS an acceptance path; the branch's own (now-shadowed) string
/// alternative being redundant is harmless.
///
/// Returns an empty `Vec` (no fold) for anything that is not exactly this
/// shape. `$ref`, a `type`, and a `const` key are each re-checked here even
/// though `compile_any_of`'s only caller reaches this function solely when
/// `string_class_of(sub)` already returned `None` for this same `sub` (which
/// already excludes all three), so this function is correct standalone too,
/// not just under that precondition.
pub(super) fn fold_string_members(sub: &Value) -> Result<Vec<String>, SchemaError> {
    let Some(obj) = sub.as_object() else {
        return Ok(Vec::new());
    };
    if obj.contains_key("$ref") || obj.contains_key("type") || obj.contains_key("const") {
        return Ok(Vec::new());
    }
    let Some(arr) = obj.get("enum").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    // Defensive backstop mirroring `string_class_of`'s own raw-cardinality
    // guard, bounding this function's filter/collect below independent of
    // the caller. Currently unreachable via `compile_any_of` (its loop always
    // calls `string_class_of(sub)` on this same `sub` first, which already
    // enforces this identical bound on the same array before `None` can be
    // returned), kept in case this function is ever called from elsewhere.
    if arr.len() > MAX_STRING_LITERALS {
        return Err(SchemaError(format!(
            "anyOf string literal count ({}) exceeds the supported limit ({MAX_STRING_LITERALS})",
            arr.len()
        )));
    }
    let has_string = arr.iter().any(Value::is_string);
    let has_non_string = arr.iter().any(|v| !v.is_string());
    if !(has_string && has_non_string) {
        // All-string (already folded directly by `string_class_of` itself) or
        // all-non-string (nothing to fold) — not this function's shape.
        return Ok(Vec::new());
    }
    let mut lits = Vec::new();
    for v in arr {
        if let Some(s) = v.as_str() {
            guard_literal_bytes(s)?;
            lits.push(s.to_string());
        }
    }
    Ok(lits)
}

/// Reject a single JSON literal whose byte length exceeds the per-literal cap
/// before it is copied or expanded into grammar symbols. Bounds the
/// single-oversized-literal allocation uniformly across the const / enum-member
/// / object-key paths (issue #474).
pub(super) fn guard_literal_bytes(s: &str) -> Result<(), SchemaError> {
    if s.len() > MAX_STRING_LITERAL_BYTES {
        return Err(SchemaError(format!(
            "string literal byte length ({}) exceeds the supported limit ({MAX_STRING_LITERAL_BYTES})",
            s.len()
        )));
    }
    Ok(())
}

/// Reject any individual string anywhere in the schema whose byte length
/// exceeds the per-literal cap before compilation begins, so no downstream
/// const / enum-member / object-key / $defs-name / $ref path can serialize,
/// copy, or expand an oversized literal. A single entry-point pass bounds the
/// whole schema uniformly (issue #474).
pub(super) fn guard_schema_string_bytes(v: &Value) -> Result<(), SchemaError> {
    match v {
        Value::String(s) => guard_literal_bytes(s),
        Value::Array(items) => {
            for item in items {
                guard_schema_string_bytes(item)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for (key, val) in map {
                guard_literal_bytes(key)?;
                guard_schema_string_bytes(val)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Convert a concrete JSON value to a grammar alternative (sequence of terminal bytes).
fn json_value_to_alt(v: &Value) -> Result<Alt, SchemaError> {
    let json_str = serde_json::to_string(v)
        .map_err(|e| SchemaError(format!("cannot serialize enum value: {e}")))?;
    guard_literal_bytes(&json_str)?;
    Ok(json_str.bytes().map(Symbol::Terminal).collect())
}

/// Build a grammar alternative for the JSON literal string `key` (with quotes).
///
/// issue #310 finding #7: JSON-encode the key so that characters like `"` and
/// `\` are properly escaped in the grammar, matching valid JSON output.
pub(super) fn json_string_literal(key: &str) -> Result<Alt, SchemaError> {
    let json_repr = serde_json::to_string(key)
        .map_err(|e| SchemaError(format!("cannot JSON-encode property key: {e}")))?;
    guard_literal_bytes(&json_repr)?;
    // json_repr includes surrounding `"` quotes; emit them as terminals directly.
    Ok(json_repr.bytes().map(Symbol::Terminal).collect())
}

/// Extract the definition name from a `$ref` string like `#/$defs/Foo`.
pub(super) fn extract_ref_name(ref_str: &str) -> Result<&str, SchemaError> {
    let parts: Vec<&str> = ref_str.split('/').collect();
    match parts.as_slice() {
        ["#", "$defs", name] | ["#", "definitions", name] => Ok(name),
        _ => Err(SchemaError(format!(
            "unsupported $ref format: {ref_str} (only #/$defs/Name supported)"
        ))),
    }
}

/// The schema a property named by BOTH a `$ref` target and that `$ref`'s
/// `properties` sibling must satisfy: the intersection of the two, since
/// draft 2020-12 applies the sibling in addition to the reference rather than
/// in place of it.
///
/// This compiler has no `allOf` representation and `compile_object_fields`
/// borrows property schemas out of the source document, so an intersection can
/// only be honored when it IS one of the two operands: identical declarations,
/// or one side being the empty schema `{}`, which admits every value and so
/// contributes nothing. Every other collision is an explicit unsupported-schema
/// error — picking a side would compile a language that accepts values the
/// other side forbids, which for `{"type":"integer"}` against
/// `{"type":"string"}` means accepting a value the schema admits for no type at
/// all.
pub(super) fn intersect_property_schemas<'a>(
    key: &str,
    sibling: &'a Value,
    target: &'a Value,
) -> Result<&'a Value, SchemaError> {
    if sibling == target {
        return Ok(sibling);
    }
    if sibling.as_object().is_some_and(serde_json::Map::is_empty) {
        return Ok(target);
    }
    if target.as_object().is_some_and(serde_json::Map::is_empty) {
        return Ok(sibling);
    }
    Err(SchemaError(format!(
        "unsupported schema: property `{key}` is declared by both a $ref target and its `properties` sibling, and the intersection of the two declarations is not representable in this grammar"
    )))
}
