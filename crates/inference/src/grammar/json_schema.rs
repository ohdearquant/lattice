//! JSON Schema → `CompiledGrammar` compiler.
//!
//! # Supported subset (v0)
//!
//! - `type`: `"object"`, `"array"`, `"string"`, `"number"`, `"integer"`,
//!   `"boolean"`, `"null"`
//! - `properties` + `required` on objects
//! - `items` (uniform array) and `prefixItems` (tuple) on arrays
//! - `minItems` / `maxItems` on arrays
//! - `enum` and `const` on any type
//! - `anyOf` / `oneOf`
//! - `$ref` resolved within the same document (`#/$defs/...`,
//!   `#/definitions/...`)
//! - `$defs` / `definitions` for reusable sub-schemas
//!
//! # Deferred (not implemented)
//!
//! - External `$ref` (URI references)
//! - `pattern` constraints on strings
//! - `if` / `then` / `else`
//!
//! # Grammar encoding
//!
//! Each JSON Schema concept maps to a named rule in the `CompiledGrammar`:
//!
//! - Primitives: `json_string`, `json_number`, `json_integer`, `json_boolean`,
//!   `json_null`, `json_ws` (optional whitespace)
//! - Objects: `obj_{hash}` with one alternative per property permutation
//! - Arrays: `arr_{hash}`
//! - `$ref`-defined schemas get rule names from their path segment

use crate::grammar::pda::{Alt, CompiledGrammar, GrammarBuilder, Symbol};
use serde_json::Value;
use std::collections::HashMap;

/// Error returned by the JSON Schema compiler.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaError(pub String);

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON Schema compile error: {}", self.0)
    }
}

impl std::error::Error for SchemaError {}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compile a JSON Schema `Value` into a `CompiledGrammar`.
///
/// The schema is expected to be a JSON Schema object (an `{}` map).  Pass the
/// full document root — `$defs` / `definitions` will be extracted automatically.
pub fn compile_json_schema(schema: &Value) -> Result<CompiledGrammar, SchemaError> {
    let mut ctx = CompileCtx::new(schema)?;
    // Reserve "root" — may not be at index 0 because register_builtins() runs
    // first in CompileCtx::new().
    let root_id = ctx.builder.reserve("root");
    let root_alts = ctx.compile_schema(schema, &[])?;
    ctx.builder.set_alts(root_id, root_alts);
    // Ensure all deferred rules have been resolved.
    ctx.resolve_pending()?;
    let mut grammar = ctx.builder.build();

    // Move "root" to index 0 so that GrammarState::initial() (which starts
    // with rule_id=0) points to the correct entry rule.
    if root_id != 0 {
        grammar.rules.swap(0, root_id);
        // Fix up all NonTerminal references after the swap.
        for rule in &mut grammar.rules {
            for alt in &mut rule.alts {
                for sym in alt.iter_mut() {
                    if let crate::grammar::pda::Symbol::NonTerminal(rid) = sym {
                        if *rid == root_id {
                            *rid = 0;
                        } else if *rid == 0 {
                            *rid = root_id;
                        }
                    }
                }
            }
        }
    }

    Ok(grammar)
}

// ---------------------------------------------------------------------------
// Internal compile context
// ---------------------------------------------------------------------------

struct CompileCtx<'a> {
    /// Name-to-schema mapping for `$defs` / `definitions`.
    defs: HashMap<String, &'a Value>,
    builder: GrammarBuilder,
    /// Monotonic counter for generating collision-free string-enum rule names.
    enum_counter: usize,
}

impl<'a> CompileCtx<'a> {
    fn new(doc_root: &'a Value) -> Result<Self, SchemaError> {
        let mut defs = HashMap::new();
        // Collect $defs
        if let Some(defs_map) = doc_root.get("$defs").and_then(Value::as_object) {
            for (k, v) in defs_map {
                defs.insert(k.clone(), v);
            }
        }
        // Also accept legacy `definitions`
        if let Some(defs_map) = doc_root.get("definitions").and_then(Value::as_object) {
            for (k, v) in defs_map {
                defs.entry(k.clone()).or_insert(v);
            }
        }
        let mut builder = GrammarBuilder::new();
        // Pre-register built-in rules.
        register_builtins(&mut builder);
        Ok(Self {
            defs,
            builder,
            enum_counter: 0,
        })
    }

    /// Compile `schema` into a list of alternatives.
    fn compile_schema(
        &mut self,
        schema: &'a Value,
        _path: &[&str],
    ) -> Result<Vec<Alt>, SchemaError> {
        // Handle `$ref`
        if let Some(ref_str) = schema.get("$ref").and_then(Value::as_str) {
            return self.compile_ref(ref_str);
        }

        // Handle `enum`
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            // For string-typed enums (or all-string enum values), delegate to
            // compile_string_type which factors the `"` delimiters to avoid
            // alternative ambiguity in the PDA.
            let type_is_string = schema.get("type").and_then(Value::as_str) == Some("string");
            let all_strings = values.iter().all(Value::is_string);
            if type_is_string || all_strings {
                return Ok(self.compile_string_type(schema));
            }
            return compile_enum(values);
        }

        // Handle `const`
        if let Some(v) = schema.get("const") {
            return compile_const(v);
        }

        // Handle `anyOf` / `oneOf`
        if let Some(any_of) = schema
            .get("anyOf")
            .or_else(|| schema.get("oneOf"))
            .and_then(Value::as_array)
        {
            let mut all_alts: Vec<Alt> = Vec::new();
            for sub in any_of {
                let sub_alts = self.compile_schema(sub, _path)?;
                all_alts.extend(sub_alts);
            }
            return Ok(all_alts);
        }

        // Dispatch on `type`
        match schema.get("type").and_then(Value::as_str) {
            Some("object") => self.compile_object(schema),
            Some("array") => self.compile_array(schema),
            Some("string") => Ok(self.compile_string_type(schema)),
            Some("number") => {
                let id = self
                    .builder
                    .rule_id("json_number")
                    .ok_or_else(|| SchemaError("builtin json_number missing".into()))?;
                Ok(vec![vec![Symbol::NonTerminal(id)]])
            }
            Some("integer") => {
                let id = self
                    .builder
                    .rule_id("json_integer")
                    .ok_or_else(|| SchemaError("builtin json_integer missing".into()))?;
                Ok(vec![vec![Symbol::NonTerminal(id)]])
            }
            Some("boolean") => {
                let id = self
                    .builder
                    .rule_id("json_boolean")
                    .ok_or_else(|| SchemaError("builtin json_boolean missing".into()))?;
                Ok(vec![vec![Symbol::NonTerminal(id)]])
            }
            Some("null") => {
                let id = self
                    .builder
                    .rule_id("json_null")
                    .ok_or_else(|| SchemaError("builtin json_null missing".into()))?;
                Ok(vec![vec![Symbol::NonTerminal(id)]])
            }
            Some(other) => Err(SchemaError(format!("unsupported type: {other}"))),
            None => {
                // No type: treat as any JSON value.
                Ok(self.any_value_alts())
            }
        }
    }

    /// Compile `#/$defs/Name` or `#/definitions/Name` reference.
    fn compile_ref(&mut self, ref_str: &str) -> Result<Vec<Alt>, SchemaError> {
        let name = extract_ref_name(ref_str)?;
        // Look up in $defs
        let target: &'a Value = self
            .defs
            .get(name)
            .copied()
            .ok_or_else(|| SchemaError(format!("$ref not found: {ref_str}")))?;

        // Namespace `$defs` rules under a `def_` prefix so they share the rule
        // table with neither built-in rules (`ws`, `json_*`) nor generated
        // helpers (`str_enum_*`). A user definition named `ws` or `str_enum_0`
        // thus becomes `def_ws` / `def_str_enum_0` and can neither clobber nor
        // be aliased to those rules in any compile order (issue #310 #4).
        let rule_name = format!("def_{name}");

        // If we already have a rule for this def, emit a NonTerminal.
        if let Some(id) = self.builder.rule_id(&rule_name) {
            return Ok(vec![vec![Symbol::NonTerminal(id)]]);
        }

        // Reserve the rule (handles recursive references).
        let id = self.builder.reserve(&rule_name);
        // Compile the target and set alts.
        let alts = self.compile_schema(target, &[])?;
        self.builder.set_alts(id, alts);
        Ok(vec![vec![Symbol::NonTerminal(id)]])
    }

    /// Compile an `object` schema.
    fn compile_object(&mut self, schema: &'a Value) -> Result<Vec<Alt>, SchemaError> {
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .map(|m| m.iter().collect::<Vec<_>>());
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();

        match properties {
            None => {
                // No properties: any JSON object `{}` or `{"k":v,...}`.
                Ok(vec![empty_object_alt()])
            }
            Some(props) => {
                // Generate grammar: `{` ws property_list ws `}`
                // For v0 we emit a single alt with all properties in declaration order.
                // Required properties are mandated; optional properties are wrapped in an
                // "optional item" rule.
                let mut main_alt: Alt = vec![Symbol::Terminal(b'{')];
                // ws
                let ws_id = self.builder.rule_id("ws").unwrap();
                main_alt.push(Symbol::NonTerminal(ws_id));

                let mut first = true;
                for (key, val_schema) in &props {
                    let key_str: &str = key.as_str();
                    let is_req = required.contains(&key_str);

                    // Compile the value schema into a named rule.
                    let val_rule_name = format!("prop_val_{key_str}");
                    let val_id = if let Some(id) = self.builder.rule_id(&val_rule_name) {
                        id
                    } else {
                        let id = self.builder.reserve(&val_rule_name);
                        let val_alts = self.compile_schema(val_schema, &[])?;
                        self.builder.set_alts(id, val_alts);
                        id
                    };

                    // Build: `"key" ws ":" ws value`
                    let pair_rule_name = format!("pair_{key_str}");
                    let pair_id = if let Some(id) = self.builder.rule_id(&pair_rule_name) {
                        id
                    } else {
                        let id = self.builder.reserve(&pair_rule_name);
                        let mut pair_alt = json_string_literal(key_str);
                        pair_alt.push(Symbol::NonTerminal(ws_id));
                        pair_alt.push(Symbol::Terminal(b':'));
                        pair_alt.push(Symbol::NonTerminal(ws_id));
                        pair_alt.push(Symbol::NonTerminal(val_id));
                        self.builder.set_alts(id, vec![pair_alt]);
                        id
                    };

                    if !first {
                        // Separator: ws "," ws
                        main_alt.push(Symbol::NonTerminal(ws_id));
                        main_alt.push(Symbol::Terminal(b','));
                        main_alt.push(Symbol::NonTerminal(ws_id));
                    }

                    if is_req {
                        main_alt.push(Symbol::NonTerminal(pair_id));
                    } else {
                        // Optional pair: wrap in opt_pair rule.
                        let opt_name = format!("opt_{pair_rule_name}");
                        let opt_id = if let Some(id) = self.builder.rule_id(&opt_name) {
                            id
                        } else {
                            let id = self.builder.reserve(&opt_name);
                            let opt_alts = vec![
                                vec![Symbol::NonTerminal(pair_id)],
                                vec![], // epsilon: property absent
                            ];
                            self.builder.set_alts(id, opt_alts);
                            id
                        };
                        main_alt.push(Symbol::NonTerminal(opt_id));
                    }
                    first = false;
                }

                main_alt.push(Symbol::NonTerminal(ws_id));
                main_alt.push(Symbol::Terminal(b'}'));
                Ok(vec![main_alt])
            }
        }
    }

    /// Compile an `array` schema.
    fn compile_array(&mut self, schema: &'a Value) -> Result<Vec<Alt>, SchemaError> {
        let ws_id = self.builder.rule_id("ws").unwrap();
        let min_items = schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
        let max_items = schema
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|v| v as usize);

        if let Some(items_schema) = schema.get("items") {
            // Uniform array: all items share the same schema.
            let item_rule = "arr_item";
            let item_id = if let Some(id) = self.builder.rule_id(item_rule) {
                id
            } else {
                let id = self.builder.reserve(item_rule);
                let item_alts = self.compile_schema(items_schema, &[])?;
                self.builder.set_alts(id, item_alts);
                id
            };

            // Build the array as: `[` ws  (item (ws , ws item)*)? ws `]`
            let mut alt: Alt = vec![Symbol::Terminal(b'[')];
            alt.push(Symbol::NonTerminal(ws_id));

            if min_items == 0 && max_items.is_none_or(|m| m > 0) {
                // Optional content: build arr_body rule.
                let body_name = "arr_body";
                let body_id = if let Some(id) = self.builder.rule_id(body_name) {
                    id
                } else {
                    let id = self.builder.reserve(body_name);
                    // arr_body = item (ws , ws item)*
                    // We approximate repetition with a left-recursive rule.
                    // Since our PDA doesn't handle true left-recursion well,
                    // we use right-recursion: arr_body = item arr_tail
                    // arr_tail = (ws , ws item arr_tail) | ε
                    let tail_name = "arr_tail";
                    let tail_id = self.builder.reserve(tail_name);
                    let tail_alts = vec![
                        vec![
                            Symbol::NonTerminal(ws_id),
                            Symbol::Terminal(b','),
                            Symbol::NonTerminal(ws_id),
                            Symbol::NonTerminal(item_id),
                            Symbol::NonTerminal(tail_id),
                        ],
                        vec![], // epsilon
                    ];
                    self.builder.set_alts(tail_id, tail_alts);

                    let body_alt = vec![vec![
                        Symbol::NonTerminal(item_id),
                        Symbol::NonTerminal(tail_id),
                    ]];
                    self.builder.set_alts(id, body_alt);
                    id
                };

                // opt_arr_body = arr_body | ε
                let opt_name = "opt_arr_body";
                let opt_id = if let Some(id) = self.builder.rule_id(opt_name) {
                    id
                } else {
                    let id = self.builder.reserve(opt_name);
                    self.builder
                        .set_alts(id, vec![vec![Symbol::NonTerminal(body_id)], vec![]]);
                    id
                };
                alt.push(Symbol::NonTerminal(opt_id));
            } else {
                // min_items required items — unroll exactly.
                for _ in 0..min_items {
                    alt.push(Symbol::NonTerminal(item_id));
                }
            }

            alt.push(Symbol::NonTerminal(ws_id));
            alt.push(Symbol::Terminal(b']'));
            return Ok(vec![alt]);
        }

        // No items schema: any array `[]` or `[v, ...]`.
        let any_val_rule = "any_json_value";
        let any_id = if let Some(id) = self.builder.rule_id(any_val_rule) {
            id
        } else {
            let id = self.builder.reserve(any_val_rule);
            let any_alts = self.any_value_alts();
            self.builder.set_alts(id, any_alts);
            id
        };

        let tail_name = "any_arr_tail";
        let tail_id = if let Some(id) = self.builder.rule_id(tail_name) {
            id
        } else {
            let id = self.builder.reserve(tail_name);
            let tail_alts = vec![
                vec![
                    Symbol::NonTerminal(ws_id),
                    Symbol::Terminal(b','),
                    Symbol::NonTerminal(ws_id),
                    Symbol::NonTerminal(any_id),
                    Symbol::NonTerminal(id), // recursive
                ],
                vec![],
            ];
            self.builder.set_alts(id, tail_alts);
            id
        };

        let body_name = "any_arr_body";
        let body_id = if let Some(id) = self.builder.rule_id(body_name) {
            id
        } else {
            let id = self.builder.reserve(body_name);
            self.builder.set_alts(
                id,
                vec![
                    vec![Symbol::NonTerminal(any_id), Symbol::NonTerminal(tail_id)],
                    vec![],
                ],
            );
            id
        };

        let mut alt: Alt = vec![Symbol::Terminal(b'[')];
        alt.push(Symbol::NonTerminal(ws_id));
        alt.push(Symbol::NonTerminal(body_id));
        alt.push(Symbol::NonTerminal(ws_id));
        alt.push(Symbol::Terminal(b']'));
        Ok(vec![alt])
    }

    /// Compile `"type": "string"` potentially with an `enum` constraint.
    fn compile_string_type(&mut self, schema: &Value) -> Vec<Alt> {
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            let str_values: Vec<&str> = values.iter().filter_map(|v| v.as_str()).collect();
            if !str_values.is_empty() {
                // Factor the common `'"'` prefix and `'"'` suffix into a
                // wrapper so that string enum alternatives are disambiguated
                // inside the quoted region, not at the opening quote.
                //
                // Grammar: root → '"' enum_choices '"'
                //          enum_choices → alt0 | alt1 | ...
                //
                // Each enum occurrence gets a UNIQUE rule name via a monotonic
                // counter, so distinct enum sets never share alternatives
                // (issue #310 #4). User `$defs` rules live in a separate `def_*`
                // namespace (see `compile_ref`), so a user-named definition can
                // never collide with these `str_enum_*` helpers in either order.
                let choices_name = format!("str_enum_{}", self.enum_counter);
                self.enum_counter += 1;
                let choices_id = self.builder.reserve(&choices_name);
                let choice_alts: Vec<Alt> = str_values
                    .iter()
                    .map(|s| s.bytes().map(Symbol::Terminal).collect::<Alt>())
                    .collect();
                self.builder.set_alts(choices_id, choice_alts);
                return vec![vec![
                    Symbol::Terminal(b'"'),
                    Symbol::NonTerminal(choices_id),
                    Symbol::Terminal(b'"'),
                ]];
            }
        }
        // Default: any JSON string via built-in rule.
        if let Some(id) = self.builder.rule_id("json_string") {
            vec![vec![Symbol::NonTerminal(id)]]
        } else {
            // Fallback: inline minimal string rule (should not happen).
            vec![vec![Symbol::Terminal(b'"'), Symbol::Terminal(b'"')]]
        }
    }

    /// Returns alternatives for "any JSON value" (used when no type is given).
    fn any_value_alts(&mut self) -> Vec<Alt> {
        let Some(ws_id) = self.builder.rule_id("ws") else {
            return vec![];
        };
        let mut alts = Vec::new();
        if let Some(id) = self.builder.rule_id("json_string") {
            alts.push(vec![Symbol::NonTerminal(id)]);
        }
        if let Some(id) = self.builder.rule_id("json_number") {
            alts.push(vec![Symbol::NonTerminal(id)]);
        }
        if let Some(id) = self.builder.rule_id("json_boolean") {
            alts.push(vec![Symbol::NonTerminal(id)]);
        }
        if let Some(id) = self.builder.rule_id("json_null") {
            alts.push(vec![Symbol::NonTerminal(id)]);
        }
        // Inline empty object and empty array for v0.
        alts.push(empty_object_as_alt_with_ws(ws_id));
        alts.push(vec![
            Symbol::Terminal(b'['),
            Symbol::NonTerminal(ws_id),
            Symbol::Terminal(b']'),
        ]);
        alts
    }

    fn resolve_pending(&self) -> Result<(), SchemaError> {
        // All rules resolved inline via compile_schema and compile_ref.
        // No deferred resolution needed for v0.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Built-in primitive rules
// ---------------------------------------------------------------------------

/// Register primitive grammar rules into the builder.
fn register_builtins(b: &mut GrammarBuilder) {
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
    );
    b.set_alts(ws_id, vec![vec![Symbol::NonTerminal(ws_tail)]]);

    // json_string = '"' string_inner '"'
    // string_inner = char* where char = any byte except '"' and '\'
    //   plus '\\' followed by any byte (simplified escape handling).
    let str_id = b.reserve("json_string");
    let str_inner_id = b.reserve("json_string_inner");
    let str_char_id = b.reserve("json_string_char");

    // json_string_char: any byte except '"' (0x22) and '\' (0x5c),
    // or '\' followed by any byte.
    let mut char_alts: Vec<Alt> = Vec::new();
    // Escape sequence: '\' + any byte
    char_alts.push(vec![Symbol::Terminal(b'\\'), Symbol::AnyByte]);
    // Normal chars: all bytes except '"' (0x22) and '\' (0x5c)
    for byte in 0u8..=255 {
        if byte != b'"' && byte != b'\\' {
            char_alts.push(vec![Symbol::Terminal(byte)]);
        }
    }
    b.set_alts(str_char_id, char_alts);

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
    );
    b.set_alts(
        str_id,
        vec![vec![
            Symbol::Terminal(b'"'),
            Symbol::NonTerminal(str_inner_id),
            Symbol::Terminal(b'"'),
        ]],
    );

    // json_number = '-'? digit+ ('.' digit+)? (('e'|'E') ('+'|'-')? digit+)?
    // We approximate with a rule that matches the common case.
    let num_id = b.reserve("json_number");
    let digit_id = b.reserve("json_digit");
    let digit_alts: Vec<Alt> = (b'0'..=b'9')
        .map(|byte| vec![Symbol::Terminal(byte)])
        .collect();
    b.set_alts(digit_id, digit_alts);

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
    );
    b.set_alts(
        digits_id,
        vec![vec![
            Symbol::NonTerminal(digit_id),
            Symbol::NonTerminal(digit_tail_id),
        ]],
    );

    // json_nonzero = '1' | '2' | ... | '9'
    let nonzero_id = b.reserve("json_nonzero");
    let nonzero_alts: Vec<Alt> = (b'1'..=b'9')
        .map(|byte| vec![Symbol::Terminal(byte)])
        .collect();
    b.set_alts(nonzero_id, nonzero_alts);

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
    );

    // Optional sign
    let opt_sign_id = b.reserve("json_opt_sign");
    b.set_alts(opt_sign_id, vec![vec![Symbol::Terminal(b'-')], vec![]]);

    // Optional fraction: '.' digits  (leading zeros legal here, e.g. 1.05)
    let opt_frac_id = b.reserve("json_opt_frac");
    b.set_alts(
        opt_frac_id,
        vec![
            vec![Symbol::Terminal(b'.'), Symbol::NonTerminal(digits_id)],
            vec![],
        ],
    );

    // Optional exponent: (e|E) (sign?) digits  (leading zeros legal, e.g. 1e08)
    let exp_sign_id = b.reserve("json_exp_sign");
    b.set_alts(
        exp_sign_id,
        vec![
            vec![Symbol::Terminal(b'+')],
            vec![Symbol::Terminal(b'-')],
            vec![],
        ],
    );
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
    );

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
    );

    // json_integer = '-'? int_part  (rejects leading zeros)
    let int_id = b.reserve("json_integer");
    b.set_alts(
        int_id,
        vec![vec![
            Symbol::NonTerminal(opt_sign_id),
            Symbol::NonTerminal(int_part_id),
        ]],
    );

    // json_boolean = "true" | "false"
    let bool_id = b.reserve("json_boolean");
    b.set_alts(bool_id, vec![bytes_to_alt(b"true"), bytes_to_alt(b"false")]);

    // json_null = "null"
    let null_id = b.reserve("json_null");
    b.set_alts(null_id, vec![bytes_to_alt(b"null")]);
}

fn bytes_to_alt(bytes: &[u8]) -> Alt {
    bytes.iter().map(|&b| Symbol::Terminal(b)).collect()
}

fn empty_object_alt() -> Alt {
    vec![Symbol::Terminal(b'{'), Symbol::Terminal(b'}')]
}

fn empty_object_as_alt_with_ws(ws_id: usize) -> Alt {
    vec![
        Symbol::Terminal(b'{'),
        Symbol::NonTerminal(ws_id),
        Symbol::Terminal(b'}'),
    ]
}

/// Compile an `enum` schema (values can be any JSON type).
///
/// When all enum values are JSON strings, the opening and closing `"`
/// delimiters are factored out into the calling context.  This avoids
/// alternative ambiguity when multiple string values share the `"` prefix.
/// For mixed-type or non-string enums the naive per-value serialisation is
/// used; those values have distinct first bytes and are unambiguous.
fn compile_enum(values: &[Value]) -> Result<Vec<Alt>, SchemaError> {
    // Check if all values are strings.
    // All-string enums are handled upstream via compile_string_type, which
    // factors the `"` delimiters to avoid PDA ambiguity.  This path is
    // reached only for mixed-type or non-string enum values (e.g., const
    // arrays containing integers or null), whose first bytes are distinct.
    let alts = values
        .iter()
        .map(json_value_to_alt)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(alts)
}

/// Compile a `const` schema (a single fixed JSON value).
fn compile_const(v: &Value) -> Result<Vec<Alt>, SchemaError> {
    Ok(vec![json_value_to_alt(v)?])
}

/// Convert a concrete JSON value to a grammar alternative (sequence of terminal bytes).
fn json_value_to_alt(v: &Value) -> Result<Alt, SchemaError> {
    let json_str = serde_json::to_string(v)
        .map_err(|e| SchemaError(format!("cannot serialize enum value: {e}")))?;
    Ok(json_str.bytes().map(Symbol::Terminal).collect())
}

/// Build a grammar alternative for the JSON literal string `key` (with quotes).
fn json_string_literal(key: &str) -> Alt {
    let mut alt = vec![Symbol::Terminal(b'"')];
    alt.extend(key.bytes().map(Symbol::Terminal));
    alt.push(Symbol::Terminal(b'"'));
    alt
}

/// Extract the definition name from a `$ref` string like `#/$defs/Foo`.
fn extract_ref_name(ref_str: &str) -> Result<&str, SchemaError> {
    let parts: Vec<&str> = ref_str.split('/').collect();
    match parts.as_slice() {
        ["#", "$defs", name] | ["#", "definitions", name] => Ok(name),
        _ => Err(SchemaError(format!(
            "unsupported $ref format: {ref_str} (only #/$defs/Name supported)"
        ))),
    }
}

/// Compile a JSON Schema document into a `CompiledGrammar`.
///
/// This is the primary entry point used by `GrammarEngine::new`.
pub fn compile(schema: &Value) -> Result<CompiledGrammar, SchemaError> {
    compile_json_schema(schema)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::pda::{GrammarState, SimResult, simulate_token};

    fn compile_ok(schema_json: &str) -> CompiledGrammar {
        let v: Value = serde_json::from_str(schema_json).unwrap();
        compile(&v).unwrap()
    }

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
    fn compile_null_schema() {
        let g = compile_ok(r#"{"type":"null"}"#);
        assert!(accepts(&g, b"null"));
        assert!(rejects(&g, b"true"));
    }

    #[test]
    fn compile_boolean_schema() {
        let g = compile_ok(r#"{"type":"boolean"}"#);
        assert!(accepts(&g, b"true"));
        assert!(accepts(&g, b"false"));
        assert!(rejects(&g, b"null"));
    }

    #[test]
    fn compile_integer_schema() {
        let g = compile_ok(r#"{"type":"integer"}"#);
        assert!(accepts(&g, b"42"));
        assert!(accepts(&g, b"-7"));
        assert!(accepts(&g, b"0"));
    }

    #[test]
    fn compile_string_schema() {
        let g = compile_ok(r#"{"type":"string"}"#);
        assert!(accepts(&g, b"\"hello\""));
        assert!(accepts(&g, b"\"\""));
    }

    #[test]
    fn compile_string_enum() {
        let g = compile_ok(r#"{"type":"string","enum":["foo","bar"]}"#);
        assert!(accepts(&g, b"\"foo\""));
        assert!(accepts(&g, b"\"bar\""));
        assert!(rejects(&g, b"\"baz\""));
    }

    #[test]
    fn compile_const_value() {
        let g = compile_ok(r#"{"const":"hello"}"#);
        assert!(accepts(&g, b"\"hello\""));
        assert!(rejects(&g, b"\"world\""));
    }

    #[test]
    fn compile_const_integer() {
        let g = compile_ok(r#"{"const":42}"#);
        assert!(accepts(&g, b"42"));
        assert!(rejects(&g, b"43"));
    }

    #[test]
    fn compile_any_of() {
        let g = compile_ok(r#"{"anyOf":[{"type":"boolean"},{"type":"null"}]}"#);
        assert!(accepts(&g, b"true"));
        assert!(accepts(&g, b"false"));
        assert!(accepts(&g, b"null"));
    }

    #[test]
    fn compile_simple_object() {
        let g = compile_ok(
            r#"{
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }"#,
        );
        assert!(accepts(&g, b"{\"name\":\"Alice\"}"));
    }

    #[test]
    fn compile_empty_object() {
        let g = compile_ok(r#"{"type":"object"}"#);
        assert!(accepts(&g, b"{}"));
    }

    #[test]
    fn compile_array_any() {
        let g = compile_ok(r#"{"type":"array"}"#);
        assert!(accepts(&g, b"[]"));
    }

    #[test]
    fn schema_error_display() {
        let e = SchemaError("test".to_string());
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn unsupported_type_returns_error() {
        let v = serde_json::json!({"type": "binary"});
        assert!(compile_json_schema(&v).is_err());
    }

    #[test]
    fn ref_not_found_returns_error() {
        let v = serde_json::json!({"$ref": "#/$defs/Missing"});
        assert!(compile(&v).is_err());
    }

    #[test]
    fn defs_resolved() {
        let schema = serde_json::json!({
            "$defs": {
                "Status": {"type": "string", "enum": ["ok", "err"]}
            },
            "type": "object",
            "properties": {
                "status": {"$ref": "#/$defs/Status"}
            },
            "required": ["status"]
        });
        let g = compile(&schema).unwrap();
        assert!(accepts(&g, b"{\"status\":\"ok\"}"));
        assert!(accepts(&g, b"{\"status\":\"err\"}"));
    }
}
