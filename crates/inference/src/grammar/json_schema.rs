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

/// Maximum array cardinality (`minItems` / `maxItems`) the compiler will
/// materialize.
///
/// `build_cardinality_array_alt` emits one grammar rule (or symbol) per
/// required/optional item, and `build_bounded_tail` recurses once per unit of
/// `maxItems - minItems` slack. A parseable schema with `maxItems` or
/// `minItems` near `u64::MAX` (e.g. `18446744073709551615`) therefore overflows
/// the stack or exhausts memory at compile time — reachable from untrusted
/// schema input at `GrammarEngine::new` (issue #343). Constrained decoding has
/// no real use for arrays anywhere near this many items; reject beyond the cap
/// at the parse boundary with a typed error.
const MAX_ARRAY_CARDINALITY: usize = 4096;

/// Maximum schema recursion depth the compiler will descend.
///
/// `compile_schema` is the single recursion hub: `$ref` targets, object
/// properties, array items, and `anyOf`/`oneOf` branches all recurse back into
/// it. A parseable schema with a long `$defs` reference chain
/// (`A→B→C→…`, each a distinct unseen ref) or deeply nested objects therefore
/// recurses once per link with no natural bound — a compile-time stack overflow
/// reachable from untrusted schema input (issue #343, codex finding B). Cap the
/// depth: `serde_json` itself rejects JSON nested beyond 128, and no legitimate
/// schema chains references or nests anywhere near 512 levels.
const MAX_SCHEMA_DEPTH: usize = 512;

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
    /// Monotonic counter for generating collision-free array helper rule names
    /// (issue #310 finding #2: each array node gets a unique suffix to prevent
    /// rule sharing across distinct array schemas in the same document).
    array_counter: usize,
    /// Monotonic counter for generating collision-free object helper rule names.
    /// Same hazard as `array_counter`: two distinct object schemas that share a
    /// property key (e.g. the two branches of an `anyOf`) must not share the
    /// per-key value/pair rules, or the first branch's value type silently wins
    /// for both. Each object node gets a unique suffix.
    object_counter: usize,
    /// Current `compile_schema` recursion depth (issue #343: bound compile-time
    /// recursion through `$ref` chains and deep nesting).
    depth: usize,
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
            array_counter: 0,
            object_counter: 0,
            depth: 0,
        })
    }

    /// Compile `schema` into a list of alternatives.
    ///
    /// Thin recursion-depth wrapper around [`Self::compile_schema_inner`]: every
    /// recursive descent (refs, properties, items, `anyOf`) re-enters here, so a
    /// single depth cap bounds all compile-time recursion (issue #343).
    fn compile_schema(
        &mut self,
        schema: &'a Value,
        path: &[&str],
    ) -> Result<Vec<Alt>, SchemaError> {
        self.depth += 1;
        if self.depth > MAX_SCHEMA_DEPTH {
            self.depth -= 1;
            return Err(SchemaError(format!(
                "schema nesting / $ref chain exceeds the supported depth ({MAX_SCHEMA_DEPTH})"
            )));
        }
        let result = self.compile_schema_inner(schema, path);
        self.depth -= 1;
        result
    }

    fn compile_schema_inner(
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
                return self.compile_string_type(schema);
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
            Some("string") => self.compile_string_type(schema),
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

        let ws_id = self.builder.rule_id("ws").unwrap();
        match properties {
            None => {
                // No properties: empty object `{}`, tolerating interior ws (`{ }`).
                Ok(vec![empty_object_as_alt_with_ws(ws_id)])
            }
            Some(props) => {
                // Generate grammar: `{` ws property_list ws `}`
                // For v0 we emit a single alt with all properties in declaration order.
                // Required properties are mandated; optional properties are wrapped in an
                // "optional item" rule.
                // Unique per-object suffix so distinct object schemas sharing a
                // property key don't alias each other's value/pair rules.
                let obj_idx = self.object_counter;
                self.object_counter += 1;
                let mut main_alt: Alt = vec![Symbol::Terminal(b'{')];
                main_alt.push(Symbol::NonTerminal(ws_id));

                let mut first = true;
                for (key, val_schema) in &props {
                    let key_str: &str = key.as_str();
                    let is_req = required.contains(&key_str);

                    // Compile the value schema into a named rule.
                    let val_rule_name = format!("prop_val_{obj_idx}_{key_str}");
                    let val_id = if let Some(id) = self.builder.rule_id(&val_rule_name) {
                        id
                    } else {
                        let id = self.builder.reserve(&val_rule_name);
                        let val_alts = self.compile_schema(val_schema, &[])?;
                        self.builder.set_alts(id, val_alts);
                        id
                    };

                    // Build: `"key" ws ":" ws value`
                    let pair_rule_name = format!("pair_{obj_idx}_{key_str}");
                    let pair_id = if let Some(id) = self.builder.rule_id(&pair_rule_name) {
                        id
                    } else {
                        let id = self.builder.reserve(&pair_rule_name);
                        let mut pair_alt = json_string_literal(key_str)?;
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

        // issue #343: cap cardinality before materializing per-item rules.
        // `build_bounded_tail` recurses once per unit of slack and the required-
        // item loop emits one symbol group per `minItems`, so an absurd
        // `minItems`/`maxItems` (e.g. u64::MAX) overflows the stack / exhausts
        // memory at compile time. Reject at the parse boundary.
        if min_items > MAX_ARRAY_CARDINALITY {
            return Err(SchemaError(format!(
                "array schema minItems ({min_items}) exceeds the supported limit ({MAX_ARRAY_CARDINALITY})"
            )));
        }
        if let Some(max) = max_items {
            if max > MAX_ARRAY_CARDINALITY {
                return Err(SchemaError(format!(
                    "array schema maxItems ({max}) exceeds the supported limit ({MAX_ARRAY_CARDINALITY})"
                )));
            }
        }

        // issue #310 finding #2: validate cardinality constraints up front.
        if let Some(max) = max_items {
            if max < min_items {
                return Err(SchemaError(format!(
                    "array schema maxItems ({max}) < minItems ({min_items})"
                )));
            }
        }

        // issue #310 finding #3: handle prefixItems (tuple arrays).
        // prefixItems is a JSON array of per-position schemas.
        let prefix_items = schema
            .get("prefixItems")
            .and_then(Value::as_array)
            .map(Vec::as_slice);

        if let Some(prefix_schemas) = prefix_items {
            // Only process non-empty prefixItems here; empty prefixItems falls
            // through to the items / no-items handling below (JSON-Schema 2020-12:
            // empty prefixItems ≡ no prefixItems).
            if !prefix_schemas.is_empty() {
                // Assign a unique counter for this array node (only when we will
                // actually emit array helper rules).
                let n = self.array_counter;
                self.array_counter += 1;

                let p = prefix_schemas.len();

                // Cardinality validation against the fixed prefix length.
                if let Some(m) = max_items {
                    if m < p {
                        return Err(SchemaError(format!(
                            "array maxItems ({m}) < prefixItems length ({p})"
                        )));
                    }
                }
                if min_items > p {
                    return Err(SchemaError(format!(
                        "array minItems ({min_items}) > prefixItems length ({p}) is not supported with prefixItems"
                    )));
                }

                // Compile each positional item schema into a named rule.
                let mut pos_ids: Vec<usize> = Vec::with_capacity(p);
                for (i, pos_schema) in prefix_schemas.iter().enumerate() {
                    let pos_rule = format!("arr_{n}_prefix_{i}");
                    let pos_id = self.builder.reserve(&pos_rule);
                    let pos_alts = self.compile_schema(pos_schema, &[])?;
                    self.builder.set_alts(pos_id, pos_alts);
                    pos_ids.push(pos_id);
                }

                // Build: `[` ws p0 ws `,` ws p1 ws `,` ... ws
                let mut alt: Alt = vec![Symbol::Terminal(b'[')];
                alt.push(Symbol::NonTerminal(ws_id));
                for (i, &pid) in pos_ids.iter().enumerate() {
                    if i > 0 {
                        alt.push(Symbol::Terminal(b','));
                        alt.push(Symbol::NonTerminal(ws_id));
                    }
                    alt.push(Symbol::NonTerminal(pid));
                    alt.push(Symbol::NonTerminal(ws_id));
                }

                // If `items` is also present, append a bounded uniform tail.
                // p prefix items already emitted; max >= p validated above.
                // slack = None → unbounded; slack = Some(0) → epsilon (no extra items).
                if let Some(items_schema) = schema.get("items") {
                    let item_rule = format!("arr_{n}_item");
                    let item_id = self.builder.reserve(&item_rule);
                    let item_alts = self.compile_schema(items_schema, &[])?;
                    self.builder.set_alts(item_id, item_alts);

                    // depth=1: leading-comma continuation after the fixed prefix.
                    let slack = max_items.map(|m| m - p);
                    let tail_id = self.build_bounded_tail(n, 1, slack, item_id, ws_id);
                    alt.push(Symbol::NonTerminal(tail_id));
                }

                alt.push(Symbol::Terminal(b']'));
                return Ok(vec![alt]);
            }
            // else: empty prefixItems — fall through to items / no-items handling.
        }

        if let Some(items_schema) = schema.get("items") {
            // Uniform array: all items share the same schema.
            // issue #310 finding #2: use per-array unique rule names to prevent
            // collision when multiple array schemas appear in one document.
            let n = self.array_counter;
            self.array_counter += 1;

            let item_rule = format!("arr_{n}_item");
            let item_id = self.builder.reserve(&item_rule);
            let item_alts = self.compile_schema(items_schema, &[])?;
            self.builder.set_alts(item_id, item_alts);

            let alt = self.build_cardinality_array_alt(n, item_id, ws_id, min_items, max_items);
            return Ok(vec![alt]);
        }

        // No items schema and no prefixItems: any array `[]` or `[v, ...]`.
        let any_val_rule = "any_json_value";
        let any_id = if let Some(id) = self.builder.rule_id(any_val_rule) {
            id
        } else {
            let id = self.builder.reserve(any_val_rule);
            let any_alts = self.any_value_alts();
            self.builder.set_alts(id, any_alts);
            id
        };

        // issue #310 #2 (codex review): honor cardinality even with no `items` schema.
        if min_items > 0 || max_items.is_some() {
            let n = self.array_counter;
            self.array_counter += 1;
            let alt = self.build_cardinality_array_alt(n, any_id, ws_id, min_items, max_items);
            return Ok(vec![alt]);
        }

        // Unconstrained any-array: cache the shared rules.
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

    /// Build the full `[ ... ]` alternative for an array whose items all use
    /// `item_id`, honoring [min_items, max_items] cardinality.
    ///
    /// Precondition: max_items >= min_items (validated by the caller / global guard).
    fn build_cardinality_array_alt(
        &mut self,
        n: usize,
        item_id: usize,
        ws_id: usize,
        min_items: usize,
        max_items: Option<usize>,
    ) -> Alt {
        let mut alt: Alt = vec![Symbol::Terminal(b'[')];
        alt.push(Symbol::NonTerminal(ws_id));

        let slack = max_items.map(|m| m - min_items); // None = unbounded

        match (min_items, max_items) {
            (0, Some(0)) => {
                // Only `[]` is valid; no items at all.
            }
            (0, _) => {
                // All items are optional.  Build an optional head + optional tail.
                // slack == Some(0) means max == 0 which is handled above;
                // this arm is only reached when slack > 0 or slack is None.
                if slack != Some(0) {
                    // Build optional body: first item is optional, subsequent
                    // items each carry a leading comma.
                    let body_id = self.build_bounded_tail(n, 0, slack, item_id, ws_id);
                    alt.push(Symbol::NonTerminal(body_id));
                }
            }
            (min, _) => {
                // Emit exactly `min` required items (first without leading comma,
                // remaining with leading comma).
                alt.push(Symbol::NonTerminal(item_id));
                alt.push(Symbol::NonTerminal(ws_id));
                for _ in 1..min {
                    alt.push(Symbol::Terminal(b','));
                    alt.push(Symbol::NonTerminal(ws_id));
                    alt.push(Symbol::NonTerminal(item_id));
                    alt.push(Symbol::NonTerminal(ws_id));
                }

                // Append optional tail for additional items (up to slack more).
                if slack != Some(0) {
                    let tail_id = self.build_bounded_tail(n, 1, slack, item_id, ws_id);
                    alt.push(Symbol::NonTerminal(tail_id));
                }
            }
        }

        alt.push(Symbol::Terminal(b']'));
        alt
    }

    /// Build a rule representing "zero or more additional items (with leading commas),
    /// bounded to at most `slack` items (None = unbounded)".
    ///
    /// `depth` tracks whether this is the first item in an all-optional body (depth=0,
    /// no leading comma on the first item) or a continuation tail (depth>=1, leading
    /// comma required per item).
    ///
    /// Returns the rule id of the outermost optional rule.
    ///
    /// For unbounded tails (slack=None): right-recursive rule.
    /// For bounded tails (slack=Some(k)): nested-optional chain of depth k.
    fn build_bounded_tail(
        &mut self,
        arr_n: usize,
        depth: usize,
        slack: Option<usize>,
        item_id: usize,
        ws_id: usize,
    ) -> usize {
        match slack {
            None => {
                // Unbounded tail: arr_{n}_tail_{depth} = (,? ws item ws arr_{n}_tail_{depth}) | ε
                // When depth==0 (optional body), first item has no leading comma.
                // When depth>=1 (continuation), every item has a leading comma.
                let rule_name = format!("arr_{arr_n}_tail_{depth}");
                let id = self.builder.reserve(&rule_name);
                let recurse_id = id; // right-recursive reference to self
                let body_alt = if depth == 0 {
                    // First item: no leading comma; tail continues with depth=1.
                    // We build: item ws tail_{depth+1}
                    // But since we need the recursive rule to reference itself,
                    // build the continuation tail as a separate rule.
                    let cont_name = format!("arr_{arr_n}_tail_{}", depth + 1);
                    let cont_id = self.builder.reserve(&cont_name);
                    let cont_alt = vec![
                        Symbol::Terminal(b','),
                        Symbol::NonTerminal(ws_id),
                        Symbol::NonTerminal(item_id),
                        Symbol::NonTerminal(ws_id),
                        Symbol::NonTerminal(cont_id),
                    ];
                    self.builder
                        .set_alts(cont_id, vec![cont_alt, vec![] /* epsilon */]);
                    vec![
                        Symbol::NonTerminal(item_id),
                        Symbol::NonTerminal(ws_id),
                        Symbol::NonTerminal(cont_id),
                    ]
                } else {
                    // Continuation: leading comma + item + recurse.
                    vec![
                        Symbol::Terminal(b','),
                        Symbol::NonTerminal(ws_id),
                        Symbol::NonTerminal(item_id),
                        Symbol::NonTerminal(ws_id),
                        Symbol::NonTerminal(recurse_id),
                    ]
                };
                self.builder
                    .set_alts(id, vec![body_alt, vec![] /* epsilon */]);
                id
            }
            Some(0) => {
                // No slack: epsilon rule (no additional items allowed).
                let rule_name = format!("arr_{arr_n}_opt_{depth}_0");
                let id = self.builder.reserve(&rule_name);
                self.builder.set_alts(id, vec![vec![] /* epsilon */]);
                id
            }
            Some(k) => {
                // Bounded tail: nested-optional chain of depth k.
                // opt_{depth}_{k} = (leading_comma? item ws opt_{depth+1}_{k-1}) | ε
                let rule_name = format!("arr_{arr_n}_opt_{depth}_{k}");
                let id = self.builder.reserve(&rule_name);
                // Recursively build the inner optional (one fewer slot).
                let inner_id =
                    self.build_bounded_tail(arr_n, depth + 1, Some(k - 1), item_id, ws_id);
                let body_alt: Alt = if depth == 0 {
                    // Optional head: no leading comma on first item.
                    vec![
                        Symbol::NonTerminal(item_id),
                        Symbol::NonTerminal(ws_id),
                        Symbol::NonTerminal(inner_id),
                    ]
                } else {
                    // Continuation: leading comma before each additional item.
                    vec![
                        Symbol::Terminal(b','),
                        Symbol::NonTerminal(ws_id),
                        Symbol::NonTerminal(item_id),
                        Symbol::NonTerminal(ws_id),
                        Symbol::NonTerminal(inner_id),
                    ]
                };
                self.builder
                    .set_alts(id, vec![body_alt, vec![] /* epsilon */]);
                id
            }
        }
    }

    /// Compile `"type": "string"` potentially with an `enum` constraint.
    fn compile_string_type(&mut self, schema: &Value) -> Result<Vec<Alt>, SchemaError> {
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
                // issue #310 finding #7: JSON-encode each string value so that
                // special characters (e.g. `"`, `\`) are properly escaped in the
                // grammar, matching what valid JSON actually contains.
                // serde_json::to_string produces `"value"` with surrounding quotes;
                // strip them and emit the inner escaped bytes.
                let mut choice_alts: Vec<Alt> = Vec::with_capacity(str_values.len());
                for s in &str_values {
                    let json_repr = serde_json::to_string(s)
                        .map_err(|e| SchemaError(format!("cannot JSON-encode enum value: {e}")))?;
                    // json_repr is e.g. `"a\"b"` — strip the surrounding `"`.
                    let inner = &json_repr[1..json_repr.len() - 1];
                    choice_alts.push(inner.bytes().map(Symbol::Terminal).collect());
                }
                self.builder.set_alts(choices_id, choice_alts);
                return Ok(vec![vec![
                    Symbol::Terminal(b'"'),
                    Symbol::NonTerminal(choices_id),
                    Symbol::Terminal(b'"'),
                ]]);
            }
        }
        // Default: any JSON string via built-in rule.
        if let Some(id) = self.builder.rule_id("json_string") {
            Ok(vec![vec![Symbol::NonTerminal(id)]])
        } else {
            // Fallback: inline minimal string rule (should not happen).
            Ok(vec![vec![Symbol::Terminal(b'"'), Symbol::Terminal(b'"')]])
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
///
/// issue #310 finding #7: JSON-encode the key so that characters like `"` and
/// `\` are properly escaped in the grammar, matching valid JSON output.
fn json_string_literal(key: &str) -> Result<Alt, SchemaError> {
    let json_repr = serde_json::to_string(key)
        .map_err(|e| SchemaError(format!("cannot JSON-encode property key: {e}")))?;
    // json_repr includes surrounding `"` quotes; emit them as terminals directly.
    Ok(json_repr.bytes().map(Symbol::Terminal).collect())
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
    fn empty_object_accepts_interior_whitespace() {
        // A no-properties object schema must accept `{ }` (interior ws), not
        // only the byte-exact `{}`. The None arm previously emitted `{` `}`
        // with no ws rule between the braces.
        let g = compile_ok(r#"{"type":"object"}"#);
        assert!(accepts(&g, b"{}"));
        assert!(
            accepts(&g, b"{ }"),
            "empty no-properties object should accept interior whitespace"
        );
    }

    #[test]
    fn distinct_objects_sharing_a_key_do_not_alias() {
        // Two distinct object schemas (`a` and `b`) each declare a property
        // named `x` with a DIFFERENT value type. Without per-object rule-name
        // namespacing they would share a single `prop_val_x` rule and the first
        // compiled type (string) would silently win for both, rejecting `b.x`
        // as an integer. `a` and `b` are distinct keys, so there is no
        // shared-prefix ambiguity here — the PDA handles it cleanly.
        let g = compile_ok(
            r#"{
              "type":"object",
              "properties":{
                "a":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]},
                "b":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}
              },
              "required":["a","b"]
            }"#,
        );
        assert!(
            accepts(&g, b"{\"a\":{\"x\":\"s\"},\"b\":{\"x\":42}}"),
            "b.x must accept an integer; the string rule for a.x must not alias it"
        );
        assert!(
            rejects(&g, b"{\"a\":{\"x\":42},\"b\":{\"x\":42}}"),
            "a.x must still reject an integer (string-typed)"
        );
    }

    #[test]
    fn compile_array_any() {
        let g = compile_ok(r#"{"type":"array"}"#);
        assert!(accepts(&g, b"[]"));
        // Non-empty untyped arrays must accept too: the any-array tail rule
        // `tail ::= ws ',' ws value tail | ε` reaches the `| ε` alternative
        // after each element so the closing `]` can match.  Regression guard
        // for a mid-alternative-backtrack over-rejection (refs #353).
        assert!(accepts(&g, b"[5]"));
        assert!(accepts(&g, b"[1,2]"));
        assert!(accepts(&g, br#"[{}]"#));
        assert!(accepts(&g, b"[true]"));
    }

    #[test]
    fn any_array_rejects_trailing_comma() {
        // An untyped array (`body ::= any tail | ε`, `tail ::= ws ',' ws any
        // tail | ε`) must reject a trailing comma before `]`.  Previously the
        // PDA's no-rewind parent-backtrack switched the comma-consuming `tail`
        // frame to its `| ε` sibling AFTER the `,` byte was consumed, silently
        // admitting invalid JSON.  The per-frame byte-consumption guard refuses
        // that switch once a byte is committed.  Valid forms must still accept.
        let g = compile_ok(r#"{"type":"array"}"#);
        assert!(accepts(&g, b"[]"));
        assert!(accepts(&g, b"[5]"));
        assert!(accepts(&g, b"[1,2]"));
        assert!(rejects(&g, b"[1,]"));
        assert!(rejects(&g, b"[1,2,]"));
        assert!(rejects(&g, b"[,]"));
    }

    #[test]
    fn typed_array_rejects_trailing_comma() {
        // Same trailing-comma over-acceptance via the bounded-tail rule used
        // for typed arrays (`build_bounded_tail`).  The byte-consumption guard
        // closes both the untyped and typed array paths at once.
        let g = compile_ok(r#"{"type":"array","items":{"type":"integer"}}"#);
        assert!(accepts(&g, b"[]"));
        assert!(accepts(&g, b"[5]"));
        assert!(accepts(&g, b"[1,2]"));
        assert!(rejects(&g, b"[1,]"));
        assert!(rejects(&g, b"[1,2,]"));
    }

    #[test]
    fn shared_prefix_enum_known_limitation() {
        // KNOWN LIMITATION (pre-existing, not introduced by the #353 guard):
        // the no-rewind byte matcher cannot parse shared-prefix sibling
        // alternatives. For `enum ["foo","food"]` the first member commits the
        // shared `foo` prefix, so the longer sibling `food` is over-REJECTED.
        // This is the SAFE direction for constrained decoding (the model just
        // cannot emit one valid member); over-acceptance would be the dangerous
        // direction. The first member still accepts. This test documents the
        // behavior so a future ambiguity-preserving matcher (trie/NFA or
        // parallel stacks) has a regression anchor; it is verified identical on
        // origin/main (the byte-consumption guard neither causes nor fixes it).
        let g = compile_ok(r#"{"enum":["foo","food"]}"#);
        assert!(accepts(&g, b"\"foo\""));
        assert!(rejects(&g, b"\"food\""));
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
