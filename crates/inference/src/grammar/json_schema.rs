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
/// reachable from untrusted schema input (issue #343, finding B). Cap the
/// depth: `serde_json` itself rejects JSON nested beyond 128, and no legitimate
/// schema chains references or nests anywhere near 512 levels.
const MAX_SCHEMA_DEPTH: usize = 512;

/// Maximum number of properties in a single object schema.
///
/// `build_object_tail` builds a right-nested tail chain, recursing once per
/// property (the recursion is on property index `i`, independent of
/// `MAX_SCHEMA_DEPTH`, which bounds only nested-schema descent). A parseable
/// schema with an absurd number of properties therefore recurses once per
/// property at compile time — a stack-exhaustion vector reachable from
/// untrusted schema input at `GrammarEngine::new`, the object analogue of the
/// `MAX_ARRAY_CARDINALITY` array vector. Constrained decoding has no real use
/// for objects with anywhere near this many keys; reject beyond the cap at the
/// parse boundary with a typed error. Matches `MAX_ARRAY_CARDINALITY` so the
/// two boundaries stay consistent.
const MAX_OBJECT_PROPERTIES: usize = 4096;

/// Maximum trie recursion depth when compiling shared-prefix string-literal
/// alternations (`build_trie_node`). With single-child compression the trie
/// only recurses through the prefix SHARED by two or more members, so reaching
/// this bound requires two enum/const members that share an identical prefix
/// this many bytes long, which no realistic schema does. The cap keeps compile
/// recursion well below the stack limit and fails closed with a `SchemaError`
/// instead of aborting the process, mirroring `MAX_OBJECT_PROPERTIES`.
const MAX_TRIE_DEPTH: usize = 1024;
// DoS bound: cap the number of distinct string literals a single trie may hold.
const MAX_STRING_LITERALS: usize = 4096;
// DoS bound: cap the total encoded byte length across all literals in a single trie.
const MAX_STRING_LITERAL_BYTES: usize = 1024 * 1024;

/// Upper bound on `prefixItems` (tuple) positional schemas a single array node
/// may declare, bounding compile work and grammar rule count against adversarial
/// schemas. `compile_array` emits one named rule and one `compile_schema`
/// recursive call per positional item, so an unguarded `prefixItems` array with
/// no `maxItems` check drives unbounded allocation and recursion from untrusted
/// schema input (issue #474). Matches `MAX_ARRAY_CARDINALITY` so all per-item
/// cardinality boundaries stay consistent.
const MAX_PREFIX_ITEMS: usize = 4096;

// DoS bound: cap the number of anyOf/oneOf branches a single schema node may declare (issue #474).
const MAX_ANYOF_BRANCHES: usize = 4096;

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
    // Bound every schema string at the true public compilation entry: this fn is
    // itself `pub` (a direct caller bypasses `compile`), and the $defs-name path
    // in CompileCtx::new clones its key with no per-site byte guard, so the
    // schema-wide cap must run here to dominate every public path (issue #474).
    guard_schema_string_bytes(schema)?;
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
        // Each definition name is individually capped by the schema-wide entry
        // guard, but the map clones every key (`k.clone()`) and `$defs` has no
        // cardinality cap, so many under-cap names can still retain an unbounded
        // total. Bound the cumulative key bytes across BOTH `$defs` and
        // `definitions`, mirroring the enum/anyOf cumulative guards (issue #474).
        // saturating_add keeps the running sum from wrapping past the cap.
        let mut defs_name_bytes: usize = 0;
        // Collect $defs
        if let Some(defs_map) = doc_root.get("$defs").and_then(Value::as_object) {
            for (k, v) in defs_map {
                defs_name_bytes = defs_name_bytes.saturating_add(k.len());
                if defs_name_bytes > MAX_STRING_LITERAL_BYTES {
                    return Err(SchemaError(format!(
                        "schema definition name cumulative byte length exceeds the supported limit ({MAX_STRING_LITERAL_BYTES})"
                    )));
                }
                defs.insert(k.clone(), v);
            }
        }
        // Also accept legacy `definitions`
        if let Some(defs_map) = doc_root.get("definitions").and_then(Value::as_object) {
            for (k, v) in defs_map {
                defs_name_bytes = defs_name_bytes.saturating_add(k.len());
                if defs_name_bytes > MAX_STRING_LITERAL_BYTES {
                    return Err(SchemaError(format!(
                        "schema definition name cumulative byte length exceeds the supported limit ({MAX_STRING_LITERAL_BYTES})"
                    )));
                }
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

    /// Lower an `anyOf` / `oneOf` to a list of alternatives.
    ///
    /// The no-rewind PDA commits to an alternative once it consumes a byte (the
    /// #353/#380 consumed-guard), so every branch that can begin with `"`
    /// competes for the single opening quote: once one of them consumes it, the
    /// siblings become unreachable. Every JSON string shares that opening `"`,
    /// so listing string-valued branches separately strands all but the first
    /// (issue #310 — the over-rejection regression fixed here). Collapse the
    /// whole string-valued class into ONE branch with a single `"` entry:
    ///
    ///   * any broad string (`{"type":"string"}`; also `pattern` / `minLength`,
    ///     which this compiler does not enforce and so widens to any string)
    ///     makes the class the `json_string` rule. It subsumes every string
    ///     literal, so a valid string is never rejected and the literals need no
    ///     separate branch.
    ///   * otherwise the string literals (`const` / all-string `enum`) become a
    ///     single trie so each diverges inside the quoted region instead of
    ///     competing at the shared opening quote.
    ///
    /// Non-string branches (numbers, booleans, null, objects, arrays) never
    /// begin with `"`, so they follow the string entry in their original
    /// relative order and stay reachable: a non-`"` input diverges from the
    /// string entry at sym_pos == 0 and the PDA falls through.
    ///
    /// Three more shapes fold into the same string-valued class here, beyond
    /// what `string_class_of` alone can prove from a single sub-schema (issue
    /// #473):
    ///
    ///   * a pure nested `anyOf` union — a branch that is EXACTLY
    ///     `{"anyOf":[...]}` with no sibling key — is FLATTENED into this same
    ///     branch list (`flatten_any_of_branches`) before classification, so a
    ///     broad or literal string nested inside it classifies exactly like a
    ///     top-level branch would. A nested `oneOf` is deliberately NOT
    ///     flattened: `oneOf` is exclusive (a value matching two branches is
    ///     rejected), a no-rewind PDA cannot enforce that exclusion, and
    ///     merging an overlapping `oneOf` into this OR-hoist would over-accept
    ///     the overlap — so it stays in `other_subs`, over-rejecting as
    ///     `origin/main` does (issue #473 Tier 2);
    ///   * a `$ref` branch's string language is recovered by
    ///     `ref_string_contribution` (reusing the `$defs` map `compile_ref`
    ///     uses) as the terminal target's string class intersected with the
    ///     `const`/`enum` narrowing dropped along the chain, so a `$ref` to a
    ///     string rule folds too — narrowed, never widened. When such a branch
    ///     ALSO carries a string-forcing sibling (`type:"string"`, a string
    ///     `const`, or an all-string `enum`, see `ref_sub_forces_string`) but
    ///     its `$ref` resolves to a NON-string target, the conjunction is the
    ///     empty language, so the branch is DROPPED entirely rather than routed
    ///     to `other_subs` — otherwise `compile_ref` would materialize the
    ///     sibling-dropped non-string target and over-accept it;
    ///   * an untyped `enum` mixing string and non-string members has its
    ///     string members folded into the literal set (`fold_string_members`)
    ///     while the branch itself is ALSO kept in `other_subs`, so its
    ///     non-string members (disjoint first bytes from `"`) stay reachable
    ///     there.
    ///
    /// LIMITATION (pre-existing; `origin/main` rejects the same inputs): an
    /// untyped `{}` sibling and a `{"type":["string",...]}` type-array union
    /// still have their quoted inputs shadowed by the hoisted string entry once
    /// it consumes the opening `"` — both accept strings AND non-strings with no
    /// fixed literal set a compile-time fold could extract, so fixing them needs
    /// parallel-stack / NFA matching, which the current single no-rewind PDA
    /// does not support (issue #473). Every fold performed here can only narrow
    /// a sibling, never widen it — it never over-accepts.
    ///
    /// Kept `#[inline(never)]` and out of `compile_schema_inner` so these locals
    /// do not enlarge that function's per-recursion stack frame (see the call
    /// site: a deep `$ref` chain must hit the depth guard, not the native stack).
    #[inline(never)]
    fn compile_any_of(
        &mut self,
        any_of: &'a [Value],
        path: &[&str],
    ) -> Result<Vec<Alt>, SchemaError> {
        // DoS bound (issue #474 finding 2; extended by issue #473 shape 2):
        // reject by FLATTENED branch count before the classification loop
        // below can push an unbounded number of entries into `other_subs`
        // (each later driving one recursive `compile_schema` call). Bounding
        // only the raw top-level `any_of.len()` (the prior single guard)
        // would let a small top-level array with a wide nested pure union
        // bypass the cap entirely, so `flatten_any_of_branches` enforces
        // MAX_ANYOF_BRANCHES incrementally as it expands nested unions.
        let mut flat_subs: Vec<&'a Value> = Vec::new();
        flatten_any_of_branches(any_of, 0, &mut flat_subs)?;

        let mut broad_string = false;
        let mut literals: Vec<String> = Vec::new();
        let mut byte_total: usize = 0;
        let mut other_subs: Vec<&'a Value> = Vec::new();
        for sub in flat_subs.iter().copied() {
            match string_class_of(sub)? {
                Some(StrClass::Broad) => broad_string = true,
                Some(StrClass::Literals(values)) => {
                    fold_literals_into(&mut literals, &mut byte_total, values)?;
                }
                None => {
                    if sub.get("$ref").is_some() {
                        // Shape 1 (issue #473): a `$ref` branch. `compile_schema_inner`
                        // resolves `$ref` first and drops sibling `const`/`enum`, so the
                        // compiler's own language for the branch is the target's — an
                        // over-accept vs the true `$ref` ∧ sibling conjunction.
                        // `ref_string_contribution` recovers that conjunction (terminal
                        // string class ∩ chain narrowing); its result is never broader
                        // than compiling the terminal, so it can only narrow, never
                        // over-accept.
                        match self.ref_string_contribution(sub)? {
                            RefStr::Broad => broad_string = true,
                            RefStr::Literals(values) => {
                                // A string terminal has no non-string language, so the
                                // hoisted set fully represents the branch — it must NOT
                                // also go to `other_subs`, where `compile_ref` would
                                // re-add the broad, sibling-dropped target and reopen the
                                // over-accept. An empty set is a dead branch (empty
                                // intersection) that hoists nothing.
                                if !values.is_empty() {
                                    fold_literals_into(&mut literals, &mut byte_total, values)?;
                                }
                            }
                            RefStr::Dead => {
                                // A string-forcing node ANYWHERE along the `$ref` chain
                                // (`type:"string"`, a string `const`, or an all-string
                                // `enum`) conjoined with a non-string terminal is the
                                // EMPTY language: no value is both a string and (e.g.) an
                                // integer. Routing it to `other_subs` would let
                                // `compile_ref` materialize the sibling-dropped non-string
                                // terminal and accept, e.g., an integer the branch forbids
                                // — a genuine over-accept that `origin/main` did NOT commit
                                // (it rejected the number there). The forcing node may be
                                // an intermediate `$defs` link (`{$ref: N, type:"string"}`),
                                // not just the outer sub, so this decision is made across
                                // the whole chain in `ref_string_contribution`. Dropping
                                // the branch is exactly faithful.
                            }
                            RefStr::NotString => {
                                // No chain node forced a string and the terminal is
                                // non-string (or the chain was unresolvable): keep the
                                // prior behavior and leave the branch in `other_subs` so
                                // its non-string target stays reachable (and the pre-#473
                                // `$ref`-not-found error still surfaces on the unresolvable
                                // path).
                                other_subs.push(sub);
                            }
                        }
                    } else {
                        // Shape 5 (issue #473): a non-`$ref` untyped mixed enum folds its
                        // string members into `literals` (reachable via the hoisted trie)
                        // while `sub` itself still goes to `other_subs`, keeping its
                        // non-string members reachable too — folding can only ADD an
                        // acceptance path here, so the now-redundant string alternative
                        // left in `sub` is harmless (shadowed by the hoisted entry).
                        let folded = fold_string_members(sub)?;
                        if !folded.is_empty() {
                            fold_literals_into(&mut literals, &mut byte_total, folded)?;
                        }
                        other_subs.push(sub);
                    }
                }
            }
        }

        let mut merged: Vec<Alt> = Vec::new();
        if broad_string {
            match self.builder.rule_id("json_string") {
                Some(id) => merged.push(vec![Symbol::NonTerminal(id)]),
                None => merged.push(vec![Symbol::Terminal(b'"'), Symbol::Terminal(b'"')]),
            }
        } else if !literals.is_empty() {
            // DoS bound: reject before dedup so that a schema with more than
            // MAX_STRING_LITERALS duplicate branches cannot bypass the count cap
            // by relying on deduplication to shrink the set below the limit.
            // Unreachable once the per-extend guard above holds the invariant,
            // kept as a backstop mirroring compile_trie_literals' late guard.
            if literals.len() > MAX_STRING_LITERALS {
                return Err(SchemaError(format!(
                    "string literal count ({}) exceeds the supported limit ({MAX_STRING_LITERALS})",
                    literals.len()
                )));
            }
            // Distinct JSON strings are prefix-free once the closing `"` is
            // part of each leaf, so every literal diverges within the trie.
            // The byte budget was already enforced incrementally above as
            // `literals` was built, so no further pre-allocation check is
            // needed here.
            literals.sort();
            literals.dedup();
            let mut seqs: Vec<Vec<u8>> = Vec::with_capacity(literals.len());
            for s in &literals {
                let json_repr = serde_json::to_string(s)
                    .map_err(|e| SchemaError(format!("cannot JSON-encode string value: {e}")))?;
                // json_repr is e.g. `"a\"b"` — strip the surrounding `"`.
                let inner = &json_repr[1..json_repr.len() - 1];
                let mut seq: Vec<u8> = inner.bytes().collect();
                seq.push(b'"');
                seqs.push(seq);
            }
            let trie_root_id = self.compile_trie_literals(&seqs)?;
            merged.push(vec![
                Symbol::Terminal(b'"'),
                Symbol::NonTerminal(trie_root_id),
            ]);
        }
        for sub in other_subs {
            merged.extend(self.compile_schema(sub, path)?);
        }
        Ok(merged)
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

        // Handle `anyOf` / `oneOf`. The body lives in a separate, non-inlined
        // method (see `compile_any_of`) so its locals do not enlarge this
        // function's stack frame: a deep `$ref` chain re-enters
        // `compile_schema_inner` at every link, and the depth guard (#343) must
        // reject at MAX_SCHEMA_DEPTH before the native stack overflows. Folding
        // the anyOf locals in here regressed that headroom (a 2000-link chain
        // overflowed instead of returning the depth error).
        if let Some(any_of) = schema
            .get("anyOf")
            .or_else(|| schema.get("oneOf"))
            .and_then(Value::as_array)
        {
            return self.compile_any_of(any_of, _path);
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

    /// Compute the string-language contribution of a `$ref`-bearing `anyOf`
    /// branch as the resolved terminal's string class INTERSECTED with every
    /// `const`/`enum` narrowing dropped along the `$ref` chain (issue #473).
    ///
    /// `compile_schema_inner` resolves `$ref` FIRST and drops sibling keywords,
    /// so the compiler's own language for `{"$ref":<string def>,"enum":["a"]}`
    /// is the target's (any string) — an over-accept relative to the true
    /// draft-2020-12 conjunction `string ∩ {"a"} = {"a"}`. Hoisting that broad
    /// target into the shared string entry would let the union accept any
    /// string. Instead this walks the chain (`sub` and each intermediate
    /// `$defs` node), collects the string set any `const`/`enum` permits,
    /// classifies the terminal, and returns the intersection. The result is
    /// never BROADER than what the terminal alone would compile to, so it can
    /// never over-accept relative to origin; where a narrowing applies it is
    /// strictly more faithful.
    ///
    /// Returns `NotString` when the terminal may accept a non-string value AND
    /// no node in the chain forces a string (so a genuine non-string target is
    /// reachable), a `$ref` in the chain has an unsupported format, a definition
    /// name is unknown, or the chain does not terminate within
    /// `MAX_SCHEMA_DEPTH` (a cyclic `$defs` chain `A` -> `B` -> `A`) — in every
    /// such case the caller leaves `sub` in `other_subs` to compile for real,
    /// exactly as before this fix. Returns `Dead` when a node ANYWHERE in the
    /// chain forces a string (`type:"string"`, a string `const`, or an
    /// all-string `enum`, see `ref_sub_forces_string`) but the terminal is
    /// non-string: that conjunction is the empty language, so the caller drops
    /// the branch instead of routing it to `other_subs` where `compile_ref`
    /// would re-materialize the (sibling-dropped) non-string target and
    /// over-accept it. The string-forcing check spans the WHOLE chain, not just
    /// the outer `sub`, so an intermediate `$defs` node such as
    /// `{"$ref":<integer def>,"type":"string"}` is caught (issue #473). This
    /// lookup never touches `self.depth` and never registers a
    /// rule; `compile_ref` / `compile_schema` remain the authoritative recursion
    /// guard (issue #343) for the `other_subs` path.
    fn ref_string_contribution(&self, sub: &'a Value) -> Result<RefStr, SchemaError> {
        let mut narrowing: Option<Vec<String>> = None;
        let mut chain_forces_string = ref_sub_forces_string(sub);
        intersect_narrowing(&mut narrowing, node_string_narrowing(sub)?);
        let mut current = sub;
        for _ in 0..=MAX_SCHEMA_DEPTH {
            let Some(ref_str) = current.get("$ref").and_then(Value::as_str) else {
                return Ok(combine_ref_str(
                    string_class_of(current)?,
                    narrowing,
                    chain_forces_string,
                ));
            };
            let Ok(name) = extract_ref_name(ref_str) else {
                return Ok(RefStr::NotString);
            };
            let Some(next) = self.defs.get(name).copied() else {
                return Ok(RefStr::NotString);
            };
            current = next;
            chain_forces_string |= ref_sub_forces_string(current);
            intersect_narrowing(&mut narrowing, node_string_narrowing(current)?);
        }
        Ok(RefStr::NotString)
    }

    /// Compile an `object` schema.
    fn compile_object(&mut self, schema: &'a Value) -> Result<Vec<Alt>, SchemaError> {
        let properties_map = schema.get("properties").and_then(Value::as_object);
        // issue #343 analogue / issue #478: cap property cardinality on the raw
        // map BEFORE `.iter().collect::<Vec<_>>()` below, which otherwise
        // iterates and allocates proportional to untrusted cardinality before
        // any check runs. `build_object_tail` recurses once per property, so an
        // object schema with an absurd number of keys overflows the stack at
        // compile time. Reject at the parse boundary, mirroring the
        // `MAX_ARRAY_CARDINALITY` guard.
        if let Some(m) = properties_map
            && m.len() > MAX_OBJECT_PROPERTIES
        {
            return Err(SchemaError(format!(
                "object property count ({}) exceeds the supported limit ({MAX_OBJECT_PROPERTIES})",
                m.len()
            )));
        }
        let properties = properties_map.map(|m| m.iter().collect::<Vec<_>>());

        let required_arr = schema.get("required").and_then(Value::as_array);
        // Same DoS class: `required` cardinality is untrusted independent of
        // `properties` (a schema can declare a huge `required` array against a
        // tiny or empty `properties` map), and the filter_map/collect below is
        // proportional to it — bound it before collecting too.
        if let Some(a) = required_arr
            && a.len() > MAX_OBJECT_PROPERTIES
        {
            return Err(SchemaError(format!(
                "object required count ({}) exceeds the supported limit ({MAX_OBJECT_PROPERTIES})",
                a.len()
            )));
        }
        let required: Vec<&str> = required_arr
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();

        let ws_id = self.builder.rule_id("ws").unwrap();
        match properties {
            None => {
                // No properties: empty object `{}`, tolerating interior ws (`{ }`).
                Ok(vec![empty_object_as_alt_with_ws(ws_id)])
            }
            Some(props) => {
                // Unique per-object suffix so distinct object schemas sharing a
                // property key don't alias each other's value/pair rules.
                let obj_idx = self.object_counter;
                self.object_counter += 1;

                // Compile value and pair rules for each property.
                //
                // props_info entries: (pair_id, val_id, key_bytes, is_required)
                //   pair_id   — rule for the full `"key" ws ":" ws value` pair
                //   val_id    — rule for the value schema alone
                //   key_bytes — raw bytes of the JSON-encoded key, e.g. b"\"r\""
                //               (includes surrounding quotes, needed for inlining
                //               in the started=true optional emit alternative)
                //   is_req    — whether this property appears in `required`
                let mut props_info: Vec<(usize, usize, Vec<u8>, bool)> =
                    Vec::with_capacity(props.len());
                // Each key is individually capped by `guard_literal_bytes` below,
                // and `MAX_OBJECT_PROPERTIES` bounds the count, but the product
                // (up to the count cap of near-cap keys) still feeds the per-byte
                // `json_string_literal` Symbol expansion and the retained
                // `key_bytes` vectors. Bound the cumulative key bytes too,
                // mirroring the enum/anyOf cumulative guards (issue #474).
                // saturating_add keeps the running sum from wrapping past the cap.
                let mut key_bytes_total: usize = 0;
                for (key, val_schema) in &props {
                    let key_str: &str = key.as_str();
                    let is_req = required.contains(&key_str);

                    guard_literal_bytes(key_str)?;
                    key_bytes_total = key_bytes_total.saturating_add(key_str.len());
                    if key_bytes_total > MAX_STRING_LITERAL_BYTES {
                        return Err(SchemaError(format!(
                            "object property key cumulative byte length exceeds the supported limit ({MAX_STRING_LITERAL_BYTES})"
                        )));
                    }
                    // key_bytes: raw bytes of the JSON-encoded key string.
                    // Used to inline the key as terminals in started=true tail
                    // rules so trailing-comma over-acceptance is prevented for a
                    // single trailing optional (the common case); the >=2
                    // trailing-optional case is closed by the #353 per-frame
                    // byte-consumption guard in pda::try_next_alt. See
                    // build_object_tail for the exact boundary.
                    let key_bytes = serde_json::to_string(key_str)
                        .map_err(|e| SchemaError(format!("cannot JSON-encode property key: {e}")))?
                        .into_bytes();

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

                    props_info.push((pair_id, val_id, key_bytes, is_req));
                }

                // Reorder: required properties first, optional properties after.
                //
                // Within each group the original declaration order (alphabetical
                // from the BTreeMap-backed serde_json Value::as_object) is
                // preserved.  The sort is stable.
                //
                // Why this matters for PDA safety:
                //   In the `started=false` phase, every property alternative in
                //   the tail chain starts with the opening `"` of the property
                //   key.  If an OPTIONAL property P_i appears before a required
                //   property P_j, the PDA commits to P_i's `"` and then fails
                //   when the key character doesn't match (mid-alt mismatch), but
                //   by that point `"` has been consumed and cannot be replayed for
                //   P_j's opening `"` → input `{"pj_key":...}` is over-rejected.
                //
                //   By placing ALL required properties first, the started=false
                //   phase is a single-alternative chain (one `pair_i` per required
                //   prop, no branching) — the PDA cannot take a wrong turn.  Only
                //   after all required properties have been emitted do optional
                //   ones appear (in started=true mode), where the leading `","`
                //   separator provides clean first-byte discrimination.
                //
                //   This reordering changes the GENERATED PROPERTY ORDER (required
                //   fields precede optional ones in the output JSON) but is
                //   semantically valid: JSON objects are unordered by specification.
                props_info.sort_by_key(|&(_, _, _, is_req)| if is_req { 0usize } else { 1 });

                // Build right-nested tail-chain rules (issue #355).
                // tail(0, false) is the entry point: no property emitted yet.
                let tail_id = self.build_object_tail(obj_idx, &props_info, 0, false, ws_id);

                // Object rule: `{` ws tail(0, false) ws `}`
                let main_alt = vec![
                    Symbol::Terminal(b'{'),
                    Symbol::NonTerminal(ws_id),
                    Symbol::NonTerminal(tail_id),
                    Symbol::NonTerminal(ws_id),
                    Symbol::Terminal(b'}'),
                ];
                Ok(vec![main_alt])
            }
        }
    }

    /// Build a right-nested tail rule for object properties (issue #355).
    ///
    /// Constructs a grammar rule for "the remaining properties starting at index
    /// `i`" in `props` (each entry is `(pair_rule_id, is_required)`).
    ///
    /// `started`: whether any property has already been emitted — a compile-time
    /// boolean.  Distinct rules are generated for the two reachable variants.
    ///
    /// # Grammar shape
    ///
    /// - Base case (`i == props.len()`): epsilon
    /// - P_i REQUIRED, started=false: `pair_i  tail(i+1, true)`
    /// - P_i REQUIRED, started=true:  `","  ws  pair_i  tail(i+1, true)`
    /// - P_i OPTIONAL, started=false: `(pair_i  tail(i+1, true)) | tail(i+1, false)`
    /// - P_i OPTIONAL, started=true:  `("," ws pair_i tail(i+1, true)) | tail(i+1, true)`
    ///
    /// The `","` separator is placed as the **first** terminal in the started=true
    /// alternatives — there is no leading `ws` before the comma.  This is
    /// intentional: the PDA switches alternatives only when `sym_pos == 0` (no
    /// bytes consumed yet in the current alternative).  A leading nullable `ws`
    /// would advance `sym_pos` past 0 without consuming bytes, making it
    /// impossible to fall back to the skip alternative.  Compact JSON (no
    /// whitespace before commas) is the only constrained-decoding target that
    /// matters in practice, and the llama.cpp `json-schema-to-grammar` reference
    /// has the same property.
    ///
    /// # PDA safety boundary (issue #353)
    ///
    /// **Single trailing optional** (P_i is optional and all properties after it
    /// are also optional, AND P_i is the last optional before the base-ε):
    /// The inlined key bytes prevent trailing-comma over-acceptance in the common
    /// one-remaining-optional case.  When `}` arrives after `","`, the mismatch
    /// fires at sym_pos≥2 (past the comma and into the inlined key) → propagates
    /// to THIS RULE'S parent, not to the skip alt of THIS rule.  If the parent is
    /// a single-alternative required chain, there is no further alt → correctly
    /// rejects.  This is what fixes the #355 bug reproducer (`{r:req, o:opt}`).
    ///
    /// **Two or more trailing optionals** (an optional P_i followed by another
    /// optional P_j): When `}` arrives after `","` in P_i's emit alt, the mismatch
    /// propagates to THIS rule's parent.  That parent frame has already consumed
    /// the emitted property bytes, so the per-frame byte-consumption guard in
    /// `pda::try_next_alt` (issue #353) refuses to switch it to its skip
    /// alternative: a committed frame cannot be re-interpreted when there is no
    /// input to rewind.  Result: `{"a":1,}` correctly rejects for
    /// `{a:opt, b:opt}`.  (Before the #353 guard the parent's skip alt was
    /// reachable despite the consumed comma, so this case was over-accepted; the
    /// old flat-sequence grammar had the same over-acceptance via
    /// mandatory-comma + ε opt_b.)
    ///
    /// **Interleaved optional** (an optional P_i followed by a required P_j):
    /// Both the emit path (`"," ...`) and the skip path (`tail(i+1, true)` which
    /// itself starts with `","` for the required P_j) share the `","` prefix.
    /// The PDA commits to alt-0 and cannot backtrack, so it over-rejects inputs
    /// that skip P_i.  The emitted CFG is correct; this is the no-rewind
    /// single-stack PDA's shared-prefix limitation and the safe direction for
    /// constrained decoding (a valid member becomes unreachable, but no invalid
    /// output is ever emitted).  A complete fix needs ambiguity-preserving
    /// matching (a trie/NFA form or parallel active stacks), out of scope here.
    fn build_object_tail(
        &mut self,
        obj_idx: usize,
        props: &[(usize, usize, Vec<u8>, bool)], // (pair_id, val_id, key_bytes, is_required)
        i: usize,
        started: bool,
        ws_id: usize,
    ) -> usize {
        let started_char = if started { 's' } else { 'f' };
        let rule_name = format!("obj_{obj_idx}_tail_{i}_{started_char}");

        // Memoization: if this (obj_idx, i, started) variant was already built,
        // return its id without rebuilding.
        if let Some(id) = self.builder.rule_id(&rule_name) {
            return id;
        }

        // Base case: all properties processed → epsilon.
        if i == props.len() {
            let id = self.builder.reserve(&rule_name);
            self.builder.set_alts(id, vec![vec![]]); // epsilon alternative
            return id;
        }

        // Reserve first so that any recursive call that would reference *this*
        // rule (none in practice — the recursion always increments i — but
        // reserve-before-set is the established pattern in this file).
        let id = self.builder.reserve(&rule_name);

        let (pair_id, val_id, ref key_bytes, is_required) = props[i];

        let alts: Vec<Alt> = if is_required {
            // Required: one alternative (no choice).
            let next_id = self.build_object_tail(obj_idx, props, i + 1, true, ws_id);
            if started {
                // "," ws pair_i tail(i+1, true)
                // Leading "," is a terminal at sym_pos==0 of this alternative.
                vec![vec![
                    Symbol::Terminal(b','),
                    Symbol::NonTerminal(ws_id),
                    Symbol::NonTerminal(pair_id),
                    Symbol::NonTerminal(next_id),
                ]]
            } else {
                // pair_i tail(i+1, true)
                vec![vec![
                    Symbol::NonTerminal(pair_id),
                    Symbol::NonTerminal(next_id),
                ]]
            }
        } else {
            // Optional: emit alt | skip alt.
            // next_started_id: tail after emitting P_i (started=true regardless).
            // next_same_id:    tail after skipping P_i (started unchanged).
            let next_started_id = self.build_object_tail(obj_idx, props, i + 1, true, ws_id);
            let next_same_id = self.build_object_tail(obj_idx, props, i + 1, started, ws_id);

            let emit_alt: Alt = if started {
                // INLINED key for started=true optionals: instead of using
                // NonTerminal(pair_id), we inline the key bytes as terminals.
                //
                // Why inlining shifts where trailing-comma errors surface:
                //   With NonTerminal(pair_id): pair_id fails at its own
                //   sym_pos=0 (e.g. `}` vs expected `"`).  try_next_alt
                //   propagates to THIS tail rule and switches it from the emit
                //   alternative to the skip alternative (ε), even though ","
                //   was already consumed in a previous advance_byte call.
                //   Result: `{"r":1,}` is over-accepted.
                //
                //   With inlined key terminals: after "," (sym_pos 0→1) and ws
                //   (nullable, advances sym_pos without consuming bytes), the
                //   opening `"` of the key is a terminal at sym_pos≥2 IN THIS
                //   RULE.  When `}` arrives instead, sym_pos≥2 ≠ 0 →
                //   mid-alt mismatch → propagates to THIS RULE'S PARENT.
                //
                //   If the parent is a single-alternative required-chain tail
                //   (no skip alt), trailing-comma correctly rejects.  This is
                //   the #355 fix for `{r:req, o:opt}`.
                //
                //   If the parent is itself an optional-tail rule with a skip
                //   alt (>=2 trailing optionals), the parent frame has already
                //   consumed the emitted property bytes, so the #353 per-frame
                //   byte-consumption guard refuses to switch it to that skip alt
                //   and the trailing comma correctly rejects.  See doc above.
                //
                // "," ws "<key>" ws ":" ws val_i tail(i+1, true)
                let mut alt: Alt = vec![Symbol::Terminal(b','), Symbol::NonTerminal(ws_id)];
                for &byte in key_bytes.iter() {
                    alt.push(Symbol::Terminal(byte));
                }
                alt.push(Symbol::NonTerminal(ws_id));
                alt.push(Symbol::Terminal(b':'));
                alt.push(Symbol::NonTerminal(ws_id));
                alt.push(Symbol::NonTerminal(val_id));
                alt.push(Symbol::NonTerminal(next_started_id));
                alt
            } else {
                // pair_i tail(i+1, true)
                // For started=false the first byte of pair_id is `"` at
                // sym_pos=0 of this alternative.  If no key matches (absent
                // optional property), try_next_alt correctly switches to the
                // skip alternative.
                vec![
                    Symbol::NonTerminal(pair_id),
                    Symbol::NonTerminal(next_started_id),
                ]
            };
            // Skip: proceed without emitting P_i, maintaining the same started flag.
            let skip_alt: Alt = vec![Symbol::NonTerminal(next_same_id)];
            vec![emit_alt, skip_alt]
        };

        self.builder.set_alts(id, alts);
        id
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
        if let Some(max) = max_items
            && max > MAX_ARRAY_CARDINALITY
        {
            return Err(SchemaError(format!(
                "array schema maxItems ({max}) exceeds the supported limit ({MAX_ARRAY_CARDINALITY})"
            )));
        }

        // issue #310 finding #2: validate cardinality constraints up front.
        if let Some(max) = max_items
            && max < min_items
        {
            return Err(SchemaError(format!(
                "array schema maxItems ({max}) < minItems ({min_items})"
            )));
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

                // DoS bound (issue #474): an unbounded `prefixItems` array with no
                // `maxItems` check drives one `compile_schema` call and one named
                // rule per positional entry, making allocation and recursion work
                // proportional to the attacker-controlled array length. Reject
                // before Vec::with_capacity so no per-item allocation occurs for
                // over-cap inputs.
                if p > MAX_PREFIX_ITEMS {
                    return Err(SchemaError(format!(
                        "prefixItems length ({p}) exceeds the supported limit ({MAX_PREFIX_ITEMS})"
                    )));
                }

                // Cardinality validation against the fixed prefix length.
                if let Some(m) = max_items
                    && m < p
                {
                    return Err(SchemaError(format!(
                        "array maxItems ({m}) < prefixItems length ({p})"
                    )));
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

        // issue #310 #2: honor cardinality even with no `items` schema.
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
            // DoS bound (issue #474): reject oversized enum arrays by their RAW
            // cardinality before any narrowing happens below. A sibling `const`
            // (next) can shrink the surviving set to a single value, so a guard
            // placed AFTER that narrowing would never see the true input size.
            // Checking `values.len()` here, before the filter_map/collect and
            // before the const retain, bounds every downstream allocation sized
            // by this array regardless of how far the const narrows it.
            if values.len() > MAX_STRING_LITERALS {
                return Err(SchemaError(format!(
                    "string enum literal count ({}) exceeds the supported limit ({MAX_STRING_LITERALS})",
                    values.len()
                )));
            }
            let mut str_values: Vec<&str> = values.iter().filter_map(|v| v.as_str()).collect();
            // A sibling `const` is a single-value enum, so the string language is
            // the intersection `{const} ∩ enum`. Narrow to the const when it is a
            // string; a non-string const is unsatisfiable under `type:"string"`,
            // emptying the set. (`compile_schema_inner` dispatches `enum` before
            // `const`, so without this the const would be silently ignored.)
            if let Some(c) = schema.get("const") {
                match c.as_str() {
                    Some(cs) => str_values.retain(|s| *s == cs),
                    None => str_values.clear(),
                }
            }
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
                // issue #310 finding #7: JSON-encode each string value so that
                // special characters (e.g. `"`, `\`) are properly escaped in the
                // grammar, matching what valid JSON actually contains.
                // serde_json::to_string produces `"value"` with surrounding quotes;
                // strip them and emit the inner escaped bytes.
                //
                // The closing `"` is appended to each inner-byte sequence before
                // building the trie, making every sequence prefix-free (two distinct
                // JSON strings can never have one as a proper prefix of the other
                // once the closing `"` is included).  The trie ensures every branch
                // diverges at its first byte, so the no-rewind single-stack PDA
                // always picks the correct alternative at sym_pos == 0 without
                // needing to backtrack across a shared prefix (e.g. ["foo","food"]).

                // DoS bound: check the total encoded byte budget before
                // allocating the full seqs Vec so that an over-budget input
                // under the count cap cannot force a large transient copy.
                // serde_json::to_string(s) returns `"<escaped>"` (len = encoded
                // content + 2 quotes); we strip the opening quote and keep the
                // closing `"` terminator, so each sequence contributes
                // json_repr.len() - 1 bytes. saturating_add prevents an overflow
                // in the running sum from wrapping past the cap.
                let mut pre_byte_total: usize = 0;
                for s in &str_values {
                    let json_repr = serde_json::to_string(s)
                        .map_err(|e| SchemaError(format!("cannot JSON-encode enum value: {e}")))?;
                    pre_byte_total = pre_byte_total.saturating_add(json_repr.len() - 1);
                    if pre_byte_total > MAX_STRING_LITERAL_BYTES {
                        return Err(SchemaError(format!(
                            "string literal encoded byte length exceeds the supported limit ({MAX_STRING_LITERAL_BYTES})"
                        )));
                    }
                }
                let mut seqs: Vec<Vec<u8>> = Vec::with_capacity(str_values.len());
                for s in &str_values {
                    let json_repr = serde_json::to_string(s)
                        .map_err(|e| SchemaError(format!("cannot JSON-encode enum value: {e}")))?;
                    // json_repr is e.g. `"a\"b"` — strip the surrounding `"`.
                    let inner = &json_repr[1..json_repr.len() - 1];
                    let mut seq: Vec<u8> = inner.bytes().collect();
                    // Append the closing `"` as a prefix-free terminator.
                    seq.push(b'"');
                    seqs.push(seq);
                }
                let trie_root_id = self.compile_trie_literals(&seqs)?;
                // Opening `"` factored out; closing `"` is inside the trie leaves.
                return Ok(vec![vec![
                    Symbol::Terminal(b'"'),
                    Symbol::NonTerminal(trie_root_id),
                ]]);
            }
            // No surviving string member — a non-string `enum`, or a `const`
            // that conflicts with every member — so the language is empty. Emit
            // no alternative; falling through to `json_string` below would
            // over-accept arbitrary strings (issue #472).
            return Ok(Vec::new());
        }
        // Default: any JSON string via built-in rule (no `enum` constraint).
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

    /// Compile a set of prefix-free byte sequences into a trie of grammar
    /// rules. Consumes exactly one `self.enum_counter` slot (the value `n`) and
    /// reserves one builder rule per trie node: `str_enum_{n}` for the root and
    /// `str_enum_{n}_{k}` for each inner node.
    ///
    /// Callers must ensure the sequences are prefix-free.  For
    /// `compile_string_type` this is achieved by appending the closing `"`
    /// byte to each inner-byte sequence before calling.  For the
    /// `anyOf`/`oneOf` handler the full-quoted JSON representations are
    /// inherently prefix-free for distinct string values.
    fn compile_trie_literals(&mut self, seqs: &[Vec<u8>]) -> Result<usize, SchemaError> {
        if seqs.len() > MAX_STRING_LITERALS {
            return Err(SchemaError(format!(
                "string literal count ({}) exceeds the supported limit ({MAX_STRING_LITERALS})",
                seqs.len()
            )));
        }
        let total_encoded_bytes: usize = seqs.iter().map(Vec::len).sum();
        if total_encoded_bytes > MAX_STRING_LITERAL_BYTES {
            return Err(SchemaError(format!(
                "string literal encoded byte length ({total_encoded_bytes}) exceeds the supported limit ({MAX_STRING_LITERAL_BYTES})"
            )));
        }
        let n = self.enum_counter;
        self.enum_counter += 1;
        let mut node_counter = 0usize;
        build_trie_node(&mut self.builder, seqs, n, &mut node_counter, 0)
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
    );

    // Hex digit for \uXXXX escapes: 0-9, A-F, a-f.
    let hex_id = b.reserve("json_hex_digit");
    let hex_alts: Vec<Alt> = (b'0'..=b'9')
        .chain(b'A'..=b'F')
        .chain(b'a'..=b'f')
        .map(|byte| vec![Symbol::Terminal(byte)])
        .collect();
    b.set_alts(hex_id, hex_alts);

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
    b.set_alts(escape_id, escape_alts);

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
    );
    let ed_second_id = b.reserve("json_utf8_ed_second");
    b.set_alts(
        ed_second_id,
        (0x80u8..=0x9F)
            .map(|byte| vec![Symbol::Terminal(byte)])
            .collect(),
    );
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
    );
    let f4_second_id = b.reserve("json_utf8_f4_second");
    b.set_alts(
        f4_second_id,
        (0x80u8..=0x8F)
            .map(|byte| vec![Symbol::Terminal(byte)])
            .collect(),
    );
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
fn build_trie_node(
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

    builder.set_alts(id, alts);
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
fn compile_enum(values: &[Value]) -> Result<Vec<Alt>, SchemaError> {
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
fn compile_const(v: &Value) -> Result<Vec<Alt>, SchemaError> {
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
enum StrClass {
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
fn string_class_of(sub: &Value) -> Result<Option<StrClass>, SchemaError> {
    let Some(obj) = sub.as_object() else {
        return Ok(None);
    };
    // A `$ref` alongside `type`/`const`/`enum` is NOT the conjunctive
    // intersection this classifier otherwise models: `compile_schema_inner`
    // resolves `$ref` FIRST and returns before reading any sibling keyword (see
    // `compile_ref`), so the branch's real compiled language is the resolved
    // target's and the sibling is dropped. Folding a sibling `const`/`enum`
    // STRING here would hoist a literal the resolved target never accepts — e.g.
    // `{"$ref":<integer def>,"const":"y"}` compiles to `json_integer` yet the
    // fold would make the grammar accept `"y"` (an over-accept). Return `None` so
    // `compile_any_of` reclassifies via `ref_string_contribution`, which
    // intersects the resolved target's string class with the `const`/`enum`
    // narrowing dropped along the chain: a non-string target leaves the branch in
    // `other_subs` to compile for real, and a string target hoists only the
    // narrowed set — never widening the string entry (issue #473).
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
enum RefStr {
    /// The terminal may accept a non-string value and no chain node forced a
    /// string: contribute no string hoist and leave the branch in `other_subs`
    /// for `compile_schema` to handle (the pre-#473 behavior). Also returned for
    /// an unresolvable/cyclic chain, so the pre-existing `$ref`-not-found error
    /// still surfaces on that path.
    NotString,
    /// A chain node forces a string (`type:"string"`, a string `const`, or an
    /// all-string `enum`) but the terminal is non-string: the conjunction is the
    /// EMPTY language, so the caller drops the branch entirely (it contributes
    /// nothing, and must NOT go to `other_subs` where the sibling-dropped
    /// non-string target would be re-materialized and over-accepted).
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
fn combine_ref_str(
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
fn node_string_narrowing(node: &Value) -> Result<Option<Vec<String>>, SchemaError> {
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
fn intersect_narrowing(acc: &mut Option<Vec<String>>, new: Option<Vec<String>>) {
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
/// language, so `compile_any_of` drops the branch rather than materializing the
/// sibling-dropped non-string target (which would over-accept). A pure `$ref`,
/// or one whose keyword admits a non-string — a numeric `const`, an all-numeric
/// or mixed `enum`, or a `type` array that permits any non-string type
/// (`["string","null"]`, `["integer"]`, …) — returns `false` and keeps its
/// non-string target reachable. Those non-string-forcing cases stay in the
/// pre-existing `$ref`-drops-siblings over-approximation (identical to
/// `origin/main`), the same tolerated class as a dropped numeric `enum`.
fn ref_sub_forces_string(sub: &Value) -> bool {
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
fn flatten_any_of_branches<'v>(
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

/// Fold `values` into the running `literals`/`byte_total` accumulators shared
/// by every string-class-merging path in `compile_any_of` — a direct
/// `StrClass::Literals` branch, a resolved `$ref` target (issue #473 shape
/// 1), or a mixed-enum's string members (issue #473 shape 5). Enforces
/// `MAX_STRING_LITERALS` and `MAX_STRING_LITERAL_BYTES` incrementally (issue
/// #474/#478) so no caller path can accumulate literals past either cap by
/// deferring to a later check.
fn fold_literals_into(
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
fn fold_string_members(sub: &Value) -> Result<Vec<String>, SchemaError> {
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
fn guard_literal_bytes(s: &str) -> Result<(), SchemaError> {
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
fn guard_schema_string_bytes(v: &Value) -> Result<(), SchemaError> {
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
fn json_string_literal(key: &str) -> Result<Alt, SchemaError> {
    let json_repr = serde_json::to_string(key)
        .map_err(|e| SchemaError(format!("cannot JSON-encode property key: {e}")))?;
    guard_literal_bytes(&json_repr)?;
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
/// This is the primary entry point used by `GrammarEngine::new`. The schema-wide
/// byte guard lives in `compile_json_schema` so it covers direct callers too.
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

    /// Regression for issue #931: the built-in `json_string` rule used to
    /// accept any raw byte (including the C0 control range) unescaped, and
    /// any `\` followed by any byte as a valid escape. Both gaps let the
    /// grammar accept output a strict RFC 8259 parser rejects. Verified red
    /// on the unfixed code: the raw 0x07 case, the `\q` case, and the
    /// truncated `\u12"` case all accepted before this fix (the escape
    /// alternative was `'\\' + Symbol::AnyByte`, and the plain-byte
    /// alternative ranged over all of `0u8..=255`).
    #[test]
    fn string_rejects_raw_control_byte() {
        let g = compile_ok(r#"{"type":"string"}"#);
        // Raw BEL (0x07) unescaped inside the quotes must be rejected.
        assert!(rejects(&g, b"\"\x07\""));
    }

    #[test]
    fn string_rejects_invalid_escape() {
        let g = compile_ok(r#"{"type":"string"}"#);
        // `\q` is not in the RFC 8259 escape set.
        assert!(rejects(&g, b"\"\\q\""));
    }

    #[test]
    fn string_rejects_truncated_unicode_escape() {
        let g = compile_ok(r#"{"type":"string"}"#);
        // `\u12` followed directly by the closing quote is only two hex
        // digits; `\u` requires exactly four.
        assert!(rejects(&g, b"\"\\u12\""));
    }

    #[test]
    fn string_accepts_plain_ascii() {
        let g = compile_ok(r#"{"type":"string"}"#);
        assert!(accepts(&g, b"\"A\""));
    }

    #[test]
    fn string_accepts_multibyte_utf8() {
        let g = compile_ok(r#"{"type":"string"}"#);
        // "好" (U+597D, 3-byte), "é" (U+00E9, 2-byte), "😀" (U+1F600, 4-byte).
        assert!(accepts(&g, "\"好\"".as_bytes()));
        assert!(accepts(&g, "\"é\"".as_bytes()));
        assert!(accepts(&g, "\"😀\"".as_bytes()));
    }

    #[test]
    fn string_rejects_lone_continuation_byte() {
        let g = compile_ok(r#"{"type":"string"}"#);
        // 0x80 alone is a UTF-8 continuation byte with no lead byte.
        assert!(rejects(&g, b"\"\x80\""));
    }

    #[test]
    fn string_rejects_overlong_encoding() {
        let g = compile_ok(r#"{"type":"string"}"#);
        // 0xC0 0xAF is the canonical overlong encoding of '/' (U+002F);
        // 0xC0/0xC1 are excluded from the valid 2-byte lead range entirely.
        assert!(rejects(&g, b"\"\xC0\xAF\""));
    }

    /// Regression for issue #931: exercises the E0/ED/F0/F4 restricted
    /// second-byte sub-rules (`e0_second_id`/`ed_second_id`/`f0_second_id`/
    /// `f4_second_id`) end-to-end through the same PDA path constrained
    /// decoding uses (`simulate_token` via `accepts`/`rejects`), plus the
    /// control-byte and escape-body boundaries in the same string grammar.
    #[test]
    fn string_utf8_boundary_and_control_escape_table() {
        let g = compile_ok(r#"{"type":"string"}"#);

        // Minimal/maximal well-formed forms for each restricted lead byte.
        let accept_cases: &[(&str, &[u8])] = &[
            ("2-byte min", &[0xC2, 0x80]),
            ("2-byte max", &[0xDF, 0xBF]),
            ("3-byte E0 min", &[0xE0, 0xA0, 0x80]),
            ("3-byte ED max", &[0xED, 0x9F, 0xBF]),
            ("4-byte F0 min", &[0xF0, 0x90, 0x80, 0x80]),
            ("4-byte F4 max", &[0xF4, 0x8F, 0xBF, 0xBF]),
        ];
        for (name, bytes) in accept_cases {
            let mut input = vec![b'"'];
            input.extend_from_slice(bytes);
            input.push(b'"');
            assert!(
                accepts(&g, &input),
                "expected ACCEPT for {name}: {bytes:02X?}"
            );
        }

        // Boundary violations: overlong encodings, a UTF-16 surrogate, and a
        // code point above the U+10FFFF Unicode maximum.
        let reject_cases: &[(&str, &[u8])] = &[
            ("E0 overlong", &[0xE0, 0x80, 0x80]),
            ("F0 overlong", &[0xF0, 0x80, 0x80, 0x80]),
            ("ED surrogate", &[0xED, 0xA0, 0x80]),
            ("F4 above U+10FFFF", &[0xF4, 0x90, 0x80, 0x80]),
            ("C0 overlong 2-byte", &[0xC0, 0x80]),
            ("lone continuation byte", &[0x80]),
        ];
        for (name, bytes) in reject_cases {
            let mut input = vec![b'"'];
            input.extend_from_slice(bytes);
            input.push(b'"');
            assert!(
                rejects(&g, &input),
                "expected REJECT for {name}: {bytes:02X?}"
            );
        }

        // Raw control bytes 0x00-0x1F must be escaped; 0x20 (space) is the
        // first legal unescaped byte.
        for byte in 0x00u8..=0x1F {
            assert!(
                rejects(&g, &[b'"', byte, b'"']),
                "expected REJECT for raw control byte {byte:#04x}"
            );
        }
        assert!(
            accepts(&g, &[b'"', 0x20, b'"']),
            "expected ACCEPT for raw space (0x20)"
        );

        // Escaped controls (valid \u escapes) are accepted.
        assert!(accepts(&g, b"\"\\u0000\""));
        assert!(accepts(&g, b"\"\\u001F\""));

        // Every one-byte escape body outside the RFC 8259 set is rejected.
        let legal_escape_bodies: &[u8] = b"\"\\/bfnrtu";
        for byte in 0x00u8..=0xFF {
            if legal_escape_bodies.contains(&byte) {
                continue;
            }
            assert!(
                rejects(&g, &[b'"', b'\\', byte, b'"']),
                "expected REJECT for illegal escape body {byte:#04x}"
            );
        }
    }

    #[test]
    fn string_accepts_legal_escapes() {
        let g = compile_ok(r#"{"type":"string"}"#);
        assert!(accepts(&g, b"\"\\\"\""));
        assert!(accepts(&g, b"\"\\\\\""));
        assert!(accepts(&g, b"\"\\/\""));
        assert!(accepts(&g, b"\"\\b\""));
        assert!(accepts(&g, b"\"\\f\""));
        assert!(accepts(&g, b"\"\\n\""));
        assert!(accepts(&g, b"\"\\r\""));
        assert!(accepts(&g, b"\"\\t\""));
        assert!(accepts(&g, b"\"\\u0041\""));
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

    // -----------------------------------------------------------------------
    // Trie-compiled shared-prefix literal alternatives (issue #XXX)
    //
    // The single-stack no-rewind PDA can only switch alternatives at sym_pos==0.
    // A flat encoding of ["foo","food"] commits to "foo" and cannot backtrack
    // when it then sees "d" — "food" is over-rejected. The fix compiles string
    // enum / anyOf-const alternatives into a trie so that every branch diverges
    // on its first byte (the closing `"` is pulled into the trie leaves to make
    // sequences prefix-free).
    // -----------------------------------------------------------------------

    /// string enum ["foo","food"]: both accepted; "foe"/"fooo"/"fo" rejected.
    #[test]
    fn string_enum_shared_prefix_foo_food() {
        let g = compile_ok(r#"{"type":"string","enum":["foo","food"]}"#);
        assert!(accepts(&g, b"\"foo\""), "\"foo\" must be accepted");
        assert!(accepts(&g, b"\"food\""), "\"food\" must be accepted");
        assert!(rejects(&g, b"\"foe\""), "\"foe\" must be rejected");
        assert!(rejects(&g, b"\"fooo\""), "\"fooo\" must be rejected");
        assert!(rejects(&g, b"\"fo\""), "\"fo\" must be rejected");
    }

    /// Bare enum WITHOUT `"type":"string"` must also route through the trie via
    /// the all-strings predicate in `compile_schema_inner`'s enum branch
    /// (`type_is_string || all_strings`), NOT the flat `compile_enum` path —
    /// otherwise shared-prefix members are over-rejected again. This is the
    /// exact schema the prior known-limitation test used; it locks the routing
    /// so a future narrowing to `type_is_string` only cannot silently regress.
    #[test]
    fn string_enum_bare_no_type_shared_prefix() {
        let g = compile_ok(r#"{"enum":["foo","food"]}"#);
        assert!(accepts(&g, b"\"foo\""), "\"foo\" must be accepted");
        assert!(accepts(&g, b"\"food\""), "\"food\" must be accepted");
        assert!(rejects(&g, b"\"foe\""), "\"foe\" must be rejected");
    }

    /// anyOf[const "abc", const "abd"]: both accepted; "abe" rejected.
    #[test]
    fn anyof_const_shared_prefix_abc_abd() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"const":"abd"}]}"#);
        assert!(accepts(&g, b"\"abc\""), "\"abc\" must be accepted");
        assert!(accepts(&g, b"\"abd\""), "\"abd\" must be accepted");
        assert!(rejects(&g, b"\"abe\""), "\"abe\" must be rejected");
        assert!(rejects(&g, b"\"ab\""), "\"ab\" must be rejected");
        assert!(rejects(&g, b"\"abcd\""), "\"abcd\" must be rejected");
    }

    /// Regression: a broad `{"type":"string"}` branch alongside shared-prefix
    /// string consts must stay reachable. The broad branch compiles to
    /// `[NonTerminal(json_string)]`, whose FIRST set contains `"`; #471's
    /// unconditional const-trie hoist made the no-rewind PDA consume the opening
    /// quote and then fail to fall through, wrongly rejecting an arbitrary string
    /// such as `"zzz"`. A present broad string collapses the whole string class
    /// to `json_string`, which subsumes the consts. Mutation guard: reverting to
    /// the unconditional hoist reintroduces the over-rejection and fails here.
    #[test]
    fn anyof_broad_string_with_shared_prefix_consts() {
        let g = compile_ok(r#"{"anyOf":[{"type":"string"},{"const":"abc"},{"const":"abd"}]}"#);
        assert!(
            accepts(&g, b"\"zzz\""),
            "broad string branch must accept an arbitrary string"
        );
        assert!(accepts(&g, b"\"abc\""), "\"abc\" must be accepted");
        assert!(accepts(&g, b"\"abd\""), "\"abd\" must be accepted");
        assert!(rejects(&g, b"123"), "a non-string must still be rejected");
    }

    /// A non-string sibling that cannot begin with `"` (here `integer`) does not
    /// block the shared-prefix trie: `"abc"`/`"abd"` are disambiguated by the
    /// combined trie and the integer branch stays reachable (a non-`"` input
    /// diverges from the trie at sym_pos == 0). Mutation guard: emitting the
    /// branches flat instead of one trie over-rejects `"abd"`.
    #[test]
    fn anyof_shared_prefix_consts_with_integer() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"const":"abd"},{"type":"integer"}]}"#);
        assert!(accepts(&g, b"\"abc\""), "\"abc\" must be accepted");
        assert!(
            accepts(&g, b"\"abd\""),
            "\"abd\" must be accepted via the trie"
        );
        assert!(
            accepts(&g, b"123"),
            "the integer branch must remain reachable"
        );
        assert!(
            rejects(&g, b"\"abe\""),
            "a non-member string must be rejected"
        );
    }

    /// Broad string LAST (issue #310 / PR #472): the broad `{"type":"string"}`
    /// is listed AFTER the shared-prefix consts, so schema order cannot be
    /// relied on to keep it reachable. Classifying the whole string class up
    /// front and collapsing it to `json_string` makes position irrelevant — an
    /// arbitrary string that diverges from the consts (`"abe"`) is still
    /// accepted. Mutation guard: an order-sensitive fix (hoisting the const trie
    /// whenever the broad branch is not first) over-rejects `"abe"` here.
    #[test]
    fn anyof_broad_string_after_consts() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"const":"abd"},{"type":"string"}]}"#);
        assert!(
            accepts(&g, b"\"abe\""),
            "broad string (listed last) must accept a string that diverges from the consts"
        );
        assert!(accepts(&g, b"\"zzz\""), "arbitrary string must be accepted");
        assert!(accepts(&g, b"\"abc\""), "\"abc\" must be accepted");
        assert!(accepts(&g, b"\"abd\""), "\"abd\" must be accepted");
        assert!(rejects(&g, b"123"), "a non-string must still be rejected");
    }

    /// A narrower string sibling — a string `enum` rather than a broad string —
    /// alongside shared-prefix consts (issue #310 / PR #472). With no broad
    /// branch present, every string literal (the two consts and the two enum
    /// members) is unioned into ONE trie, so all four diverge inside the quoted
    /// region and stay reachable. Mutation guard: keeping the branches flat (or
    /// hoisting only the consts) over-rejects `"abd"`; widening the class to
    /// `json_string` would over-accept `"abg"`.
    #[test]
    fn anyof_narrower_string_enum_sibling() {
        let g = compile_ok(
            r#"{"anyOf":[{"const":"abc"},{"const":"abd"},{"type":"string","enum":["abe","abf"]}]}"#,
        );
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" must be accepted");
        assert!(
            accepts(&g, b"\"abd\""),
            "const \"abd\" must be accepted via the combined trie"
        );
        assert!(
            accepts(&g, b"\"abe\""),
            "enum member \"abe\" must be accepted"
        );
        assert!(
            accepts(&g, b"\"abf\""),
            "enum member \"abf\" must be accepted"
        );
        assert!(
            rejects(&g, b"\"abg\""),
            "a string in no branch must be rejected (no over-accept)"
        );
    }

    /// A non-string `enum` sibling (`[1, 2]`) must classify as a non-string
    /// branch, not fold into the string literal trie (issue #310 / PR #472).
    /// The consts form the trie; the numeric enum stays reachable because it
    /// cannot begin with `"`. Mutation guard: misclassifying the numeric enum as
    /// a string literal set drops the `1`/`2` branch; keeping the branches flat
    /// over-rejects `"abd"`.
    #[test]
    fn anyof_numeric_enum_sibling() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"const":"abd"},{"enum":[1,2]}]}"#);
        assert!(
            accepts(&g, b"\"abd\""),
            "\"abd\" must be accepted via the trie"
        );
        assert!(accepts(&g, b"1"), "numeric enum member 1 must be reachable");
        assert!(accepts(&g, b"2"), "numeric enum member 2 must be reachable");
        assert!(
            rejects(&g, b"\"abe\""),
            "a non-member string must be rejected"
        );
    }

    /// A typed mixed `enum` sibling `{"type":"string","enum":["abe",1]}`: the
    /// `type:"string"` makes the integer member unsatisfiable, so the branch's
    /// language is exactly {"abe"} and `string_class_of` folds it into the
    /// literal trie (issue #310 / PR #472). The trie
    /// then accepts both the const "abc" and the folded "abe". Mutation guard:
    /// dropping the typed-mixed fold (classifying the enum as an "other") strands
    /// "abe" behind the const trie and over-rejects it.
    #[test]
    fn anyof_typed_mixed_enum_folded() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"type":"string","enum":["abe",1]}]}"#);
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" must be accepted");
        assert!(
            accepts(&g, b"\"abe\""),
            "the type:string enum's string member \"abe\" must be folded and accepted"
        );
        assert!(
            rejects(&g, b"\"abd\""),
            "a string in no branch must be rejected (no over-accept)"
        );
        assert!(
            rejects(&g, b"\"zzz\""),
            "the dead integer member must not widen the branch to any-string"
        );
        assert!(
            rejects(&g, b"1"),
            "the integer enum member is unsatisfiable under type:string"
        );
    }

    /// A `{"type":"string","enum":[1,2]}` branch is unsatisfiable: an instance
    /// must be a string AND equal a numeric enum member, so it accepts no string.
    /// `string_class_of` must classify it as the empty literal set (a dead
    /// branch), NOT `None` — `None` routes it through `compile_string_type`'s
    /// `json_string` fallback and over-accepts arbitrary strings (issue #472,
    /// finding 2). Mutation guard: returning `None` accepts "zzz".
    #[test]
    fn anyof_typed_enum_no_string_members_rejected() {
        let g = compile_ok(r#"{"anyOf":[{"type":"integer"},{"type":"string","enum":[1,2]}]}"#);
        assert!(accepts(&g, b"7"), "the integer branch must stay reachable");
        assert!(
            rejects(&g, b"\"zzz\""),
            "the unsatisfiable type:string + numeric-enum branch must not widen to any-string"
        );
    }

    /// `const` and `enum` are conjunctive, so `{"type":"string","const":"x","enum":["y"]}`
    /// accepts no value (nothing equals both "x" and "y"). `string_class_of` must
    /// fold the const only when a present enum contains it, classifying this
    /// conflicting branch as the empty literal set (issue #472,
    /// finding 1). Mutation guard: folding the const before checking the enum
    /// accepts "x"; folding the enum before checking the const accepts "y".
    #[test]
    fn anyof_const_conflicting_enum_rejected() {
        let g = compile_ok(
            r#"{"anyOf":[{"type":"integer"},{"type":"string","const":"x","enum":["y"]}]}"#,
        );
        assert!(accepts(&g, b"7"), "the integer branch must stay reachable");
        assert!(
            rejects(&g, b"\"x\""),
            "const \"x\" conflicts with enum [\"y\"] — empty intersection, reject"
        );
        assert!(
            rejects(&g, b"\"y\""),
            "enum member \"y\" conflicts with const \"x\" — empty intersection, reject"
        );
    }

    /// `{"type":"string","const":"x","enum":["x","y"]}`: the const narrows the
    /// enum to the intersection {"x"}, so only "x" is accepted, NOT "y" (issue
    /// #472). Mutation guard: ignoring the const and folding the
    /// whole enum accepts "y".
    #[test]
    fn anyof_const_compatible_enum_folds_to_const() {
        let g = compile_ok(
            r#"{"anyOf":[{"type":"integer"},{"type":"string","const":"x","enum":["x","y"]}]}"#,
        );
        assert!(accepts(&g, b"7"), "the integer branch must stay reachable");
        assert!(
            accepts(&g, b"\"x\""),
            "const \"x\" is in the enum — accepted"
        );
        assert!(
            rejects(&g, b"\"y\""),
            "\"y\" is in the enum but excluded by const \"x\" — reject"
        );
    }

    /// Standalone `{"type":"string","enum":[1,2]}` is unsatisfiable;
    /// `compile_string_type` must emit the empty language, not the `json_string`
    /// fallback (standalone path). Mutation guard: the
    /// `json_string` fallback accepts "zzz".
    #[test]
    fn standalone_typed_enum_no_string_members_rejects_all() {
        let g = compile_ok(r#"{"type":"string","enum":[1,2]}"#);
        assert!(
            rejects(&g, b"\"zzz\""),
            "no string satisfies type:string + enum [1,2]"
        );
        assert!(rejects(&g, b"1"), "the numeric member is not a string");
    }

    /// Standalone const/enum intersection through `compile_string_type`, which
    /// `compile_schema_inner` reaches via the `enum` dispatch — without the
    /// intersection a sibling `const` would be silently ignored (standalone
    /// path). Mutation guard: dropping the const-narrowing
    /// accepts "y" in both cases.
    #[test]
    fn standalone_string_const_enum_intersection() {
        let conflict = compile_ok(r#"{"type":"string","const":"x","enum":["y"]}"#);
        assert!(
            rejects(&conflict, b"\"x\""),
            "const conflicts with enum — reject \"x\""
        );
        assert!(
            rejects(&conflict, b"\"y\""),
            "const conflicts with enum — reject \"y\""
        );
        let compatible = compile_ok(r#"{"type":"string","const":"x","enum":["x","y"]}"#);
        assert!(
            accepts(&compatible, b"\"x\""),
            "const \"x\" is in the enum — accept"
        );
        assert!(
            rejects(&compatible, b"\"y\""),
            "\"y\" excluded by const \"x\" — reject"
        );
    }

    /// Edge cases for typed string enums: an empty `enum` is the empty language
    /// (reject all), and a duplicate member folds to a single trie path without
    /// error (issue #472).
    #[test]
    fn typed_string_enum_empty_and_duplicate_members() {
        let empty = compile_ok(r#"{"type":"string","enum":[]}"#);
        assert!(rejects(&empty, b"\"zzz\""), "empty enum accepts nothing");
        let dups = compile_ok(r#"{"type":"string","enum":["dup","dup"]}"#);
        assert!(
            accepts(&dups, b"\"dup\""),
            "a duplicated member still accepts once"
        );
        assert!(rejects(&dups, b"\"other\""), "no over-accept from dedup");
        let in_anyof = compile_ok(r#"{"anyOf":[{"type":"integer"},{"type":"string","enum":[]}]}"#);
        assert!(accepts(&in_anyof, b"7"), "integer branch reachable");
        assert!(
            rejects(&in_anyof, b"\"zzz\""),
            "empty typed enum contributes no string"
        );
    }

    // -----------------------------------------------------------------------
    // Issue #473 — Tier 1 fixes: a `$ref` to a string schema, a pure nested
    // `anyOf`/`oneOf`, and an untyped mixed `enum` now fold into the hoisted
    // string entry instead of being shadowed by it. Tier 2 (an untyped `{}`
    // and a `{"type":[...]}` union) remains a documented, pinned limitation
    // — fixing it needs parallel-stack matching, not a compile-time fold.
    // -----------------------------------------------------------------------

    /// FIXED (issue #473, shape 1): a `$ref` to a broad-string rule, listed
    /// beside string consts, previously had its quoted inputs shadowed by the
    /// hoisted const trie (the no-rewind PDA commits to the trie once it
    /// consumes the opening `"`). `compile_any_of` now recovers the `$ref`
    /// branch's string language via `ref_string_contribution` (terminal string
    /// class ∩ chain narrowing), so a `$ref`-to-broad-string sibling with no
    /// narrowing widens the whole class to `json_string`, exactly as a literal
    /// broad string would. This schema was previously pinned as a known
    /// limitation; it now asserts the fix.
    ///
    /// Mutation guard: reverting the `$ref` reclassification in
    /// `compile_any_of` (treating every `None` as a plain "other" branch
    /// again) reintroduces the shadow and rejects "zzz".
    #[test]
    fn anyof_ref_to_broad_string_folds_and_accepts() {
        let g = compile_ok(
            r##"{"$defs":{"S":{"type":"string"}},"anyOf":[{"const":"abc"},{"const":"abd"},{"$ref":"#/$defs/S"}]}"##,
        );
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
        assert!(accepts(&g, b"\"abd\""), "const \"abd\" still accepted");
        assert!(
            accepts(&g, b"\"zzz\""),
            "issue #473 fix: $ref->broad-string sibling must now be reachable"
        );
        assert!(rejects(&g, b"123"), "a non-string must still be rejected");
    }

    /// FIXED (issue #473, shape 1, narrow variant): a `$ref` to a schema that
    /// reduces to a fixed LITERAL set (not a broad string) folds its literal
    /// into the SAME trie as the sibling const, rather than widening to any
    /// string. Proves the fold narrows correctly.
    ///
    /// Mutation guard: if the fold incorrectly treated a resolved
    /// `StrClass::Literals` target as `Broad`, "abd" (in neither literal set)
    /// would be wrongly accepted.
    #[test]
    fn anyof_ref_to_string_literal_folds_into_trie() {
        let g = compile_ok(
            r##"{"$defs":{"S":{"const":"zzz"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/S"}]}"##,
        );
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" accepted");
        assert!(
            accepts(&g, b"\"zzz\""),
            "issue #473 fix: $ref->const literal must be folded into the trie"
        );
        assert!(
            rejects(&g, b"\"abd\""),
            "a string in neither literal must be rejected (no over-accept)"
        );
    }

    /// A `$ref` to a NON-string schema must stay an "other" branch, unchanged
    /// — the design explicitly requires that a `$ref` whose target is
    /// non-string keeps its prior behavior (issue #473). The ref'd integer
    /// branch stays reachable and the const trie is unaffected.
    ///
    /// Mutation guard: if reclassification incorrectly widened a non-string
    /// ref target to accept strings, "zzz" would be wrongly accepted here.
    #[test]
    fn anyof_ref_to_non_string_stays_other() {
        let g = compile_ok(
            r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/N"}]}"##,
        );
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
        assert!(
            accepts(&g, b"7"),
            "the ref'd integer branch stays reachable"
        );
        assert!(
            rejects(&g, b"\"zzz\""),
            "a non-member string must still be rejected"
        );
    }

    /// FIXED (issue #473, chain-collect intersection): a branch that is `$ref`
    /// to a NON-string target AND carries a STRING-FORCING sibling (a string
    /// `const`, `type:"string"`, or an all-string `enum`) is the EMPTY language
    /// — no value is both a string and an integer — so the branch must accept
    /// NOTHING. `origin/main` OVER-ACCEPTED the sibling string here (it
    /// mis-hoisted the sibling into the shared string entry, accepting "y" /
    /// "zzz"), while rejecting the number. `ref_string_contribution` resolves
    /// the string part to `RefStr::NotString`, and `ref_sub_forces_string`
    /// makes `compile_any_of` DROP the branch rather than materialize the
    /// sibling-dropped integer target.
    ///
    /// Mutation guard: reverting the drop (routing `RefStr::NotString` straight
    /// to `other_subs` again) reopens a number over-accept — `7` would be
    /// wrongly accepted on every sub-case below. Removing the string guard in
    /// `ref_string_contribution` reopens the STRING over-accept — "y"/"zzz"
    /// would be wrongly accepted.
    #[test]
    fn anyof_ref_to_non_string_with_string_forcing_sibling_drops() {
        // string `const` sibling
        let g = compile_ok(
            r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/N","const":"y"}]}"##,
        );
        assert!(
            accepts(&g, b"\"abc\""),
            "the peer const \"abc\" still accepted"
        );
        assert!(
            rejects(&g, b"\"y\""),
            "issue #473: a string const sibling on a $ref->integer branch must not over-accept the string"
        );
        assert!(
            rejects(&g, b"7"),
            "the string ∧ integer conjunction is empty — the number must be rejected too (no over-accept)"
        );

        // `type:"string"` sibling
        let g = compile_ok(
            r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/N","type":"string"}]}"##,
        );
        assert!(
            rejects(&g, b"\"zzz\""),
            "type:string sibling on a $ref->integer branch must not over-accept"
        );
        assert!(
            rejects(&g, b"7"),
            "empty conjunction rejects the number too"
        );

        // all-string `enum` sibling
        let g = compile_ok(
            r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/N","enum":["y","z"]}]}"##,
        );
        assert!(
            rejects(&g, b"\"y\""),
            "all-string enum sibling on a $ref->integer branch must not over-accept"
        );
        assert!(
            rejects(&g, b"7"),
            "empty conjunction rejects the number too"
        );
    }

    /// FIXED (issue #473): the string-forcing keyword sits on an
    /// INTERMEDIATE `$defs` node, not the outer `anyOf` sub. `S` is
    /// `{$ref: N-integer, type/const/enum-string}`, so `S`'s language is the
    /// empty conjunction (integer ∧ string); the outer branch `{$ref: S}` is a
    /// PURE `$ref` with no string-forcing keyword of its own. An outer-sub-only
    /// `ref_sub_forces_string(sub)` check misses the forcing and routes the
    /// branch to `other_subs`, where `compile_ref` materializes `N` and
    /// over-accepts the integer `7`. The chain-level `chain_forces_string` state
    /// (accumulated across every walked node) classifies the branch `RefStr::Dead`
    /// and drops it, so both the string AND `7` are rejected.
    ///
    /// Mutation guard: reverting to an outer-sub-only forcing check (or dropping
    /// the `RefStr::Dead` arm to `other_subs`) reopens the number over-accept —
    /// `7` would be wrongly accepted on every sub-case below.
    #[test]
    fn anyof_ref_chain_intermediate_string_forcing_node_drops() {
        // intermediate `type:"string"`
        let g = compile_ok(
            r##"{"$defs":{"S":{"$ref":"#/$defs/N","type":"string"},"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/S"}]}"##,
        );
        assert!(
            accepts(&g, b"\"ok\""),
            "the peer const \"ok\" still accepted"
        );
        assert!(
            rejects(&g, b"\"zzz\""),
            "issue #473: an intermediate type:string node on a $ref->integer chain must not over-accept the string"
        );
        assert!(
            rejects(&g, b"7"),
            "the intermediate string ∧ integer conjunction is empty — the number must be rejected too"
        );

        // intermediate string `const`
        let g = compile_ok(
            r##"{"$defs":{"S":{"$ref":"#/$defs/N","const":"y"},"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/S"}]}"##,
        );
        assert!(
            rejects(&g, b"\"y\""),
            "intermediate string const on a $ref->integer chain must not over-accept"
        );
        assert!(
            rejects(&g, b"7"),
            "empty conjunction rejects the number too"
        );

        // intermediate all-string `enum`
        let g = compile_ok(
            r##"{"$defs":{"S":{"$ref":"#/$defs/N","enum":["y","z"]},"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/S"}]}"##,
        );
        assert!(
            rejects(&g, b"\"y\""),
            "intermediate all-string enum on a $ref->integer chain must not over-accept"
        );
        assert!(
            rejects(&g, b"7"),
            "empty conjunction rejects the number too"
        );
    }

    /// FIXED (issue #473): draft-2020-12 §6.1.1 lets `type`
    /// be an ARRAY of type names. A string-ONLY type array (`["string"]`, or any
    /// non-empty all-`"string"` array) is string-forcing, exactly like the scalar
    /// `type:"string"`. On a `$ref`->integer terminal — DIRECT sibling or an
    /// intermediate `$defs` node — the conjunction is empty, so the branch must be
    /// dropped (`RefStr::Dead`), rejecting both the string AND `7`. A `type` array
    /// that admits any non-string type (`["string","null"]`, `["integer"]`) is
    /// NOT string-forcing and stays in the pre-existing `$ref`-drops-siblings
    /// approximation (verified identical to `origin/main`), so it is intentionally
    /// not covered here.
    ///
    /// Mutation guard: reverting `ref_sub_forces_string` to recognize only the
    /// scalar `type:"string"` (dropping the array arm) reopens the number
    /// over-accept — `7` would be wrongly accepted on both sub-cases below.
    #[test]
    fn anyof_ref_type_array_string_only_forces_drop() {
        // DIRECT sibling `type:["string"]`
        let g = compile_ok(
            r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/N","type":["string"]}]}"##,
        );
        assert!(
            accepts(&g, b"\"ok\""),
            "the peer const \"ok\" still accepted"
        );
        assert!(
            rejects(&g, b"\"zzz\""),
            "a string-only type array on a $ref->integer branch must not over-accept the string"
        );
        assert!(
            rejects(&g, b"7"),
            "the string ∧ integer conjunction is empty — the number must be rejected too"
        );

        // INTERMEDIATE `$defs` node with `type:["string"]`
        let g = compile_ok(
            r##"{"$defs":{"S":{"$ref":"#/$defs/N","type":["string"]},"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/S"}]}"##,
        );
        assert!(
            rejects(&g, b"\"zzz\""),
            "an intermediate string-only type array must not over-accept"
        );
        assert!(
            rejects(&g, b"7"),
            "empty conjunction rejects the number too"
        );
    }

    /// A `type` array that admits a NON-string type is not string-forcing, so a
    /// `$ref`->integer branch keeps its integer target reachable (`7` accepted),
    /// exactly as with a numeric `enum` sibling and matching `origin/main`. This
    /// pins `ref_sub_forces_string` to NOT over-fire on a mixed/other-type `type`
    /// array (which would silence a legitimate integer branch).
    ///
    /// Mutation guard: if `ref_sub_forces_string` returned true for
    /// `["string","integer"]` (treating any array containing `"string"` as
    /// string-forcing), `7` would be wrongly REJECTED here.
    #[test]
    fn anyof_ref_type_array_with_nonstring_member_stays_reachable() {
        let g = compile_ok(
            r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/N","type":["string","integer"]}]}"##,
        );
        assert!(
            accepts(&g, b"\"ok\""),
            "the peer const \"ok\" still accepted"
        );
        assert!(
            accepts(&g, b"7"),
            "a type array admitting integer is not string-forcing — the integer branch stays reachable"
        );
    }

    /// A NON-string-forcing sibling (an all-numeric `enum`) on a `$ref`->integer
    /// branch does NOT empty the conjunction — the number is admissible — so the
    /// integer target stays reachable, matching `origin/main` and the
    /// pre-existing `$ref`-drops-sibling approximation (the enum values
    /// themselves are not enforced; the whole integer type is accepted, an
    /// over-reject-free continuation of prior behavior). The point is that
    /// `ref_sub_forces_string` must NOT over-fire on a numeric enum.
    ///
    /// Mutation guard: if `ref_sub_forces_string` returned true for an
    /// all-numeric `enum` (dropping the branch), `7` would be wrongly REJECTED
    /// here — a regression that silences a legitimate integer branch.
    #[test]
    fn anyof_ref_to_integer_with_numeric_enum_sibling_stays_reachable() {
        let g = compile_ok(
            r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/N","enum":[1,2]}]}"##,
        );
        assert!(
            accepts(&g, b"\"abc\""),
            "the peer const \"abc\" still accepted"
        );
        assert!(
            accepts(&g, b"7"),
            "a numeric-enum sibling is not string-forcing — the integer branch stays reachable"
        );
        assert!(
            rejects(&g, b"\"z\""),
            "a non-member string is still rejected"
        );
    }

    /// FIXED (issue #473, chain-collect intersection): a DIRECT string `enum`
    /// sibling on a branch whose `$ref` resolves to a BROAD string narrows the
    /// branch to that enum. `compile_ref` drops the sibling `enum` (ref-first),
    /// so `origin/main`'s branch was the broad target; the fix intersects the
    /// terminal's broad string class with the sibling narrowing, folding exactly
    /// `{"abc"}` into the shared trie.
    ///
    /// Mutation guard: dropping the sibling narrowing (returning the terminal's
    /// broad class unintersected) would accept "zzz" — a re-widening over-accept.
    #[test]
    fn anyof_ref_to_broad_string_with_enum_sibling_narrows() {
        let g = compile_ok(
            r##"{"$defs":{"T":{"type":"string"}},"anyOf":[{"const":"foo"},{"$ref":"#/$defs/T","enum":["abc"]}]}"##,
        );
        assert!(
            accepts(&g, b"\"foo\""),
            "the peer const \"foo\" still accepted"
        );
        assert!(
            accepts(&g, b"\"abc\""),
            "issue #473: the sibling enum member is reachable (terminal-broad intersect the enum)"
        );
        assert!(
            rejects(&g, b"\"zzz\""),
            "the enum-narrowed branch must reject a non-member string (no over-accept)"
        );
    }

    /// FIXED (issue #473, chain-collect intersection): the narrowing is
    /// collected along the WHOLE `$ref` chain, not just the direct sibling. Here
    /// the intermediate `$defs` node `S` states `enum:["abc"]` and `$ref`s `T`
    /// (a broad string); the outer branch `$ref`s `S`. `ref_string_contribution`
    /// walks sub -> S -> T, intersecting `S`'s `{"abc"}` narrowing with `T`'s
    /// broad class, folding `{"abc"}`.
    ///
    /// Mutation guard: if the chain walk stopped collecting narrowing at the
    /// first hop (or reset it per hop), the intermediate `enum` would be lost and
    /// "zzz" wrongly accepted.
    #[test]
    fn anyof_ref_chain_collects_intermediate_enum_narrowing() {
        let g = compile_ok(
            r##"{"$defs":{"S":{"$ref":"#/$defs/T","enum":["abc"]},"T":{"type":"string"}},"anyOf":[{"const":"foo"},{"$ref":"#/$defs/S"}]}"##,
        );
        assert!(
            accepts(&g, b"\"foo\""),
            "the peer const \"foo\" still accepted"
        );
        assert!(
            accepts(&g, b"\"abc\""),
            "issue #473: an enum on an intermediate chain node narrows the folded set"
        );
        assert!(
            rejects(&g, b"\"zzz\""),
            "the chain-collected narrowing must reject a non-member string (no over-accept)"
        );
    }

    /// An unresolvable `$ref` (unknown definition name) inside an `anyOf`
    /// must still surface the pre-existing "$ref not found" compile error —
    /// the classification-time resolution attempt (issue #473) must not
    /// swallow or change this error, only ever supply an ADDITIONAL fold
    /// opportunity when resolution succeeds.
    ///
    /// Mutation guard: if `ref_string_contribution` propagated an error instead
    /// of returning `RefStr::NotString` on a failed lookup, this would fail with
    /// the wrong error (or panic) instead of the expected message.
    #[test]
    fn anyof_ref_to_missing_def_still_errors() {
        let schema_json = r##"{"anyOf":[{"const":"abc"},{"$ref":"#/$defs/Missing"}]}"##;
        let v: Value = serde_json::from_str(schema_json).unwrap();
        let err = compile(&v).expect_err("an unresolvable $ref must still fail to compile");
        assert!(
            err.0.contains("$ref not found"),
            "expected the pre-existing $ref-not-found error, got: {}",
            err.0
        );
    }

    /// FIXED (issue #473, shape 2): a nested `anyOf` containing a broad
    /// string (`{"anyOf":[{"anyOf":[{"type":"string"}]}]}`) previously
    /// classified as an "other" sibling because `string_class_of` does not
    /// recurse into nested unions, shadowing its quoted inputs behind the
    /// hoisted const trie. `compile_any_of` now flattens a PURE nested union
    /// (`flatten_any_of_branches`) before classification, so the inner broad
    /// string classifies exactly like a top-level branch. Previously pinned
    /// as a known limitation; now asserts the fix.
    ///
    /// Mutation guard: reverting the flatten step (classifying the raw
    /// top-level `any_of` again) reintroduces the shadow and rejects "zzz".
    #[test]
    fn anyof_nested_anyof_string_folds_and_accepts() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"anyOf":[{"type":"string"}]}]}"#);
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
        assert!(
            accepts(&g, b"\"zzz\""),
            "issue #473 fix: nested-anyOf broad string must now be reachable"
        );
    }

    /// DOCUMENTED LIMITATION (issue #473, Tier 2): a nested `oneOf` is NOT
    /// flattened, unlike a nested `anyOf`. `oneOf` is exclusive, and a
    /// no-rewind PDA cannot enforce "exactly one branch matches", so merging
    /// its branches into the parent's OR-hoist would over-accept any overlap
    /// (see `anyof_nested_oneof_overlap_no_over_accept`). The conservative,
    /// provably-safe choice leaves every nested `oneOf` unflattened, so its
    /// broad string stays shadowed here even in this single-branch,
    /// non-overlapping case. Matches `origin/main`; over-rejection is safe.
    ///
    /// Mutation guard: restoring `oneOf` to `as_pure_nested_union` flips this
    /// to accept "zzz" and breaks the over-accept guard test below.
    #[test]
    fn anyof_nested_oneof_not_flattened_known_limitation() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"oneOf":[{"type":"string"}]}]}"#);
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
        assert!(
            rejects(&g, b"\"zzz\""),
            "KNOWN LIMITATION #473: a nested oneOf is deliberately not flattened (XOR unenforceable)"
        );
    }

    /// OVER-ACCEPT GUARD (issue #473) — the reason a nested `oneOf` must not be
    /// flattened. The inner `oneOf` has TWO overlapping branches: the broad
    /// `type:string` and `const:"zzz"` both match "zzz", so under JSON Schema
    /// `oneOf` (exactly one) the inner union FAILS for "zzz" and the parent
    /// `anyOf` rejects it. Flattening the `oneOf` into the parent's string
    /// hoist would let the broad string swallow "zzz" and wrongly accept it.
    /// `origin/main` rejects "zzz"; this must too.
    ///
    /// Mutation guard: restoring `oneOf` to `as_pure_nested_union` makes this
    /// accept "zzz" (a genuine over-acceptance), failing this test.
    #[test]
    fn anyof_nested_oneof_overlap_no_over_accept() {
        let g = compile_ok(
            r#"{"anyOf":[{"const":"abc"},{"oneOf":[{"type":"string"},{"const":"zzz"}]}]}"#,
        );
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
        assert!(
            rejects(&g, b"\"zzz\""),
            "issue #473: overlapping nested oneOf must not over-accept the shared value \"zzz\""
        );
    }

    /// A nested union with an EXTRA sibling key (`"description"`) is NOT a
    /// PURE nested union, so `flatten_any_of_branches` deliberately leaves it
    /// unflattened (`as_pure_nested_union` only recognizes an object whose
    /// ONLY key is `anyOf` / `oneOf`). This remains a narrower, documented
    /// limitation: the nested broad string stays shadowed. Not flattening is
    /// always safe (it falls through to the pre-existing `other_subs` path),
    /// so this pins a deliberate scope boundary rather than a bug.
    #[test]
    fn anyof_nested_anyof_with_sibling_key_stays_unflattened() {
        let g = compile_ok(
            r#"{"anyOf":[{"const":"abc"},{"anyOf":[{"type":"string"}],"description":"nested"}]}"#,
        );
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
        assert!(
            rejects(&g, b"\"zzz\""),
            "a nested union with a sibling key is deliberately not flattened"
        );
    }

    /// FIXED (issue #473, shape 5): an untyped `enum` mixing a string and a
    /// non-string member (`{"enum":["abe",1]}`, no `type` keyword) previously
    /// classified as `None` and was shadowed entirely — including its string
    /// member "abe" — behind the hoisted const trie. `fold_string_members`
    /// now folds "abe" into the trie while the branch stays in `other_subs`
    /// too, keeping `1` reachable via its disjoint first byte.
    ///
    /// Mutation guard: reverting the shape-5 fold strands "abe" behind the
    /// const trie (rejected); reverting keeping the branch in `other_subs`
    /// loses reachability of `1`.
    #[test]
    fn anyof_untyped_mixed_enum_folds_string_member() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"enum":["abe",1]}]}"#);
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
        assert!(
            accepts(&g, b"\"abe\""),
            "issue #473 fix: untyped mixed-enum string member must now be reachable"
        );
        assert!(
            accepts(&g, b"1"),
            "the non-string enum member 1 stays reachable"
        );
        assert!(
            rejects(&g, b"\"abd\""),
            "a string in no branch must still be rejected (no over-accept)"
        );
    }

    /// Shape 5 with MULTIPLE string members and multiple non-string members:
    /// every string member folds into the trie, and every non-string member
    /// stays reachable via `other_subs`.
    #[test]
    fn anyof_untyped_mixed_enum_multiple_members_all_reachable() {
        let g = compile_ok(r#"{"anyOf":[{"enum":["abe","abf",1,null]}]}"#);
        assert!(accepts(&g, b"\"abe\""), "string member \"abe\" reachable");
        assert!(accepts(&g, b"\"abf\""), "string member \"abf\" reachable");
        assert!(accepts(&g, b"1"), "non-string member 1 reachable");
        assert!(accepts(&g, b"null"), "non-string member null reachable");
        assert!(
            rejects(&g, b"\"abg\""),
            "a string not in the enum must be rejected (no over-accept)"
        );
    }

    /// DOCUMENTED LIMITATION (issue #473, Tier 2 — out of scope for this
    /// fix): an untyped `{}` sibling accepts any JSON value, but its string
    /// inputs are shadowed by the hoisted const trie. Matches `origin/main`;
    /// pins current behavior pending a parallel-stack matcher.
    #[test]
    fn anyof_untyped_sibling_shadowed_known_limitation() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{}]}"#);
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
        assert!(
            rejects(&g, b"\"zzz\""),
            "KNOWN LIMITATION #473: untyped empty-schema string inputs shadowed by the const trie"
        );
    }

    /// DOCUMENTED LIMITATION (issue #473, Tier 2 — out of scope for this
    /// fix): a `{"type":["string","number"]}` sibling accepts both strings
    /// and numbers, but this compiler does not special-case a `type` ARRAY —
    /// `compile_schema_inner` dispatches on `Value::as_str`, which is `None`
    /// for a JSON array (same as an absent `type`), so it falls to the
    /// untyped `any_value_alts` path, identically to `{}` — and its STRING
    /// inputs are shadowed by the hoisted const trie exactly like the
    /// untyped `{}` case above. There is no fixed literal set to fold here
    /// (the branch's string language is unbounded), so fixing this needs
    /// parallel-stack / NFA matching, not a compile-time fold. Pins current
    /// behavior so a future engine change flips it intentionally.
    #[test]
    fn anyof_mixed_type_array_sibling_shadowed_known_limitation() {
        let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"type":["string","number"]}]}"#);
        assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
        assert!(
            accepts(&g, b"42"),
            "a non-string alternative from the mixed-type branch stays reachable"
        );
        assert!(
            rejects(&g, b"\"zzz\""),
            "KNOWN LIMITATION #473: mixed-type-array sibling's string inputs shadowed by the const trie"
        );
    }

    /// DoS bound (issue #473 extends issue #474 finding 2): a small
    /// top-level `anyOf` (2 branches) whose SECOND branch is a nested pure
    /// union containing MAX_ANYOF_BRANCHES sibling branches must still be
    /// rejected by the branch-count cap — proving `flatten_any_of_branches`
    /// enforces MAX_ANYOF_BRANCHES on the FLATTENED total, not the raw
    /// top-level count (which is only 2 here and would sail under the cap on
    /// its own).
    ///
    /// Mutation guard: fires with "anyOf/oneOf branch count". Reverting the
    /// flatten-time incremental check (bounding only the raw top-level
    /// count, as the pre-issue-473 code did) lets this schema compile
    /// successfully instead of being rejected.
    #[test]
    fn anyof_nested_union_flatten_bypass_rejected_by_branch_cap() {
        let over = MAX_ANYOF_BRANCHES + 1;
        let inner_branches: String = (0..over)
            .map(|_| r#"{"type":"boolean"}"#.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let nested_union = format!("{{\"anyOf\":[{inner_branches}]}}");
        let schema_json = format!("{{\"anyOf\":[{{\"const\":\"abc\"}},{nested_union}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err(
            "flattened branch count over the cap must be rejected even though the raw top-level count is only 2",
        );
        assert!(
            err.0.contains("anyOf/oneOf branch count"),
            "expected the anyOf branch-count cap error, got: {}",
            err.0
        );
    }

    /// DoS bound (issue #473): `flatten_any_of_branches`' own recursion depth
    /// must be capped independent of `MAX_SCHEMA_DEPTH`'s use elsewhere. A
    /// `Value` tree is built directly here (not via `serde_json::from_str`,
    /// so this is not confounded by serde_json's own JSON-text nesting
    /// limit) with `MAX_SCHEMA_DEPTH + 50` nested pure `{"anyOf":[...]}`
    /// wrappers, and must be rejected rather than overflow the native stack.
    ///
    /// Mutation guard: fires with "anyOf/oneOf nesting depth". Removing the
    /// depth check inside `flatten_any_of_branches` either overflows the
    /// stack (a process abort, not a test failure this assertion could
    /// catch) or, if the recursion happens to survive, compiles successfully
    /// instead of being rejected.
    #[test]
    fn anyof_nested_union_flatten_depth_capped() {
        let mut inner = {
            let mut m = serde_json::Map::new();
            m.insert("type".to_string(), Value::String("boolean".to_string()));
            Value::Object(m)
        };
        for _ in 0..(MAX_SCHEMA_DEPTH + 50) {
            let mut m = serde_json::Map::new();
            m.insert("anyOf".to_string(), Value::Array(vec![inner]));
            inner = Value::Object(m);
        }
        let err =
            compile(&inner).expect_err("anyOf/oneOf nesting depth over the cap must be rejected");
        assert!(
            err.0.contains("anyOf/oneOf nesting depth"),
            "expected the flatten depth-cap error, got: {}",
            err.0
        );
    }

    /// Three-way shared prefix ["foo","food","foot"]: all three accepted.
    #[test]
    fn string_enum_three_way_shared_prefix() {
        let g = compile_ok(r#"{"type":"string","enum":["foo","food","foot"]}"#);
        assert!(accepts(&g, b"\"foo\""), "\"foo\" must be accepted");
        assert!(accepts(&g, b"\"food\""), "\"food\" must be accepted");
        assert!(accepts(&g, b"\"foot\""), "\"foot\" must be accepted");
        assert!(rejects(&g, b"\"fooz\""), "\"fooz\" must be rejected");
        assert!(rejects(&g, b"\"fo\""), "\"fo\" must be rejected");
    }

    /// Prefix pair ["a","ab"]: both accepted (quoted → prefix-free).
    #[test]
    fn string_enum_prefix_pair_a_ab() {
        let g = compile_ok(r#"{"type":"string","enum":["a","ab"]}"#);
        assert!(accepts(&g, b"\"a\""), "\"a\" must be accepted");
        assert!(accepts(&g, b"\"ab\""), "\"ab\" must be accepted");
        assert!(rejects(&g, b"\"abc\""), "\"abc\" must be rejected");
        assert!(rejects(&g, b"\"b\""), "\"b\" must be rejected");
    }

    /// Escaped values: schema `["a\\b","a\\c"]` (string values `a\b` and `a\c`,
    /// each containing a literal backslash) share the JSON-encoded byte prefix
    /// `a\\`; the trie must diverge at byte `b` vs `c`.
    ///
    /// The JSON encoding of the literal-backslash string `a\b` is `"a\\b"`,
    /// whose byte stream is [34,97,92,92,98,34].  In a Rust byte literal that
    /// is `b"\"a\\\\b\""` (four backslash chars → two bytes of 0x5C).
    #[test]
    fn string_enum_escaped_values() {
        // Schema: string values a\b and a\c (literal backslash in each).
        // Their JSON encodings "a\\b" and "a\\c" share the byte prefix
        // [97, 92, 92] = 'a', '\', '\' and diverge at 'b' vs 'c'.
        let g = compile_ok(r#"{"type":"string","enum":["a\\b","a\\c"]}"#);
        // JSON "a\\b" = bytes [34,97,92,92,98,34] = Rust b"\"a\\\\b\""
        assert!(
            accepts(&g, b"\"a\\\\b\""),
            "literal-backslash value a\\b must be accepted"
        );
        assert!(
            accepts(&g, b"\"a\\\\c\""),
            "literal-backslash value a\\c must be accepted"
        );
        assert!(
            rejects(&g, b"\"a\\\\d\""),
            "a\\d (not in enum) must be rejected"
        );
        assert!(
            rejects(&g, b"\"a\""),
            "bare a (no backslash) must be rejected"
        );
    }

    /// Single-value enum ["solo"]: accepted normally; no regression.
    #[test]
    fn string_enum_single_value() {
        let g = compile_ok(r#"{"type":"string","enum":["solo"]}"#);
        assert!(accepts(&g, b"\"solo\""), "\"solo\" must be accepted");
        assert!(rejects(&g, b"\"sol\""), "prefix \"sol\" must be rejected");
        assert!(
            rejects(&g, b"\"soloo\""),
            "extension \"soloo\" must be rejected"
        );
    }

    /// Disjoint ["cat","dog"]: distinct first bytes; both accepted.
    #[test]
    fn string_enum_disjoint() {
        let g = compile_ok(r#"{"type":"string","enum":["cat","dog"]}"#);
        assert!(accepts(&g, b"\"cat\""), "\"cat\" must be accepted");
        assert!(accepts(&g, b"\"dog\""), "\"dog\" must be accepted");
        assert!(rejects(&g, b"\"cot\""), "\"cot\" must be rejected");
        assert!(rejects(&g, b"\"dig\""), "\"dig\" must be rejected");
    }

    /// Mixed anyOf[const "abc", {"type":"object"}]: the string "abc" and any
    /// valid object are both accepted; non-member strings are rejected.
    #[test]
    fn anyof_mixed_string_const_and_object() {
        let g = compile_ok(
            r#"{"anyOf":[{"const":"abc"},{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}]}"#,
        );
        assert!(accepts(&g, b"\"abc\""), "const string must be accepted");
        assert!(accepts(&g, b"{\"x\":1}"), "valid object must be accepted");
        assert!(
            rejects(&g, b"\"abd\""),
            "non-member string must be rejected"
        );
    }

    /// A single long enum string compiles flat (no per-byte recursion) and
    /// accepts itself. Regression guard for `build_trie_node` single-child
    /// compression: once a member stops sharing a prefix it is emitted flat, so
    /// member length never drives recursion depth. Reverting the compression
    /// recurses 20_000 deep and aborts the test binary on a stack overflow.
    #[test]
    fn string_enum_long_single_value_compiles() {
        let val = "a".repeat(20_000);
        let g = compile_ok(&format!(r#"{{"type":"string","enum":["{val}"]}}"#));
        let mut input = vec![b'"'];
        input.extend(std::iter::repeat_n(b'a', 20_000));
        input.push(b'"');
        assert!(accepts(&g, &input), "the long enum value must be accepted");
        assert!(rejects(&g, b"\"a\""), "a shorter prefix must be rejected");
    }

    /// Two members sharing a prefix longer than `MAX_TRIE_DEPTH` fail closed
    /// with a `SchemaError` naming the depth bound, rather than overflowing the
    /// stack. Locks the defense-in-depth recursion cap.
    #[test]
    fn string_enum_pathological_shared_prefix_fails_closed() {
        let pfx = "a".repeat(MAX_TRIE_DEPTH + 50);
        let schema = format!(r#"{{"enum":["{pfx}b","{pfx}c"]}}"#);
        let v: Value = serde_json::from_str(&schema).unwrap();
        let err = compile(&v).expect_err("an absurd shared prefix must be rejected, not crash");
        assert!(
            err.0.contains("depth"),
            "error should name the depth limit, got: {err:?}"
        );
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

    // -----------------------------------------------------------------------
    // Issue #355 — mixed required / optional object properties
    // -----------------------------------------------------------------------
    //
    // The old flat-sequence grammar pushed the "," separator unconditionally
    // before every non-first property and then wrapped optional properties in
    // an `opt_pair` rule that could collapse to ε.  When the optional collapsed,
    // the mandatory comma was already committed → the closing `}` was rejected.
    //
    // The new tail-chain construction mirrors `build_bounded_tail` for arrays:
    // each optional property *owns* its leading comma so that the comma is only
    // emitted when the property is present.  The PDA can distinguish the emit
    // path (starts with `","`) from the skip path (reaches ε / `}`) at the
    // first byte — no backtracking needed.

    /// Both properties required — baseline sanity check.
    #[test]
    fn object_both_required() {
        let g = compile_ok(
            r#"{
                "type":"object",
                "properties":{"r":{"type":"integer"},"s":{"type":"integer"}},
                "required":["r","s"]
            }"#,
        );
        assert!(
            accepts(&g, b"{\"r\":1,\"s\":2}"),
            "fully-present object must be accepted"
        );
        assert!(
            rejects(&g, b"{\"r\":1}"),
            "missing required s must be rejected"
        );
        assert!(
            rejects(&g, b"{}"),
            "empty object must be rejected when both required"
        );
    }

    /// Single optional property only — object may be empty or have the property.
    #[test]
    fn object_single_optional_only() {
        let g = compile_ok(
            r#"{
                "type":"object",
                "properties":{"o":{"type":"integer"}}
            }"#,
        );
        assert!(
            accepts(&g, b"{}"),
            "empty object must be accepted when property is optional"
        );
        assert!(
            accepts(&g, b"{\"o\":1}"),
            "object with optional property must be accepted"
        );
        // Trailing comma is invalid JSON — must reject.
        assert!(
            rejects(&g, b"{\"o\":1,}"),
            "trailing comma must be rejected"
        );
    }

    /// Reproducer from issue #355: one required + one trailing optional.
    /// Both `{"r":1}` and `{"r":1,"o":2}` were wrongly rejected before the fix.
    #[test]
    fn object_required_plus_trailing_optional_issue_355() {
        let g = compile_ok(
            r#"{
                "type":"object",
                "properties":{"r":{"type":"integer"},"o":{"type":"integer"}},
                "required":["r"]
            }"#,
        );
        // These two must now be accepted (were both rejected before the fix).
        assert!(
            accepts(&g, b"{\"r\":1}"),
            "required-only object must be accepted (issue #355)"
        );
        assert!(
            accepts(&g, b"{\"r\":1,\"o\":2}"),
            "fully-present object must be accepted (issue #355)"
        );
        // Invalid inputs must remain rejected.
        assert!(rejects(&g, b"{}"), "missing required r must be rejected");
        assert!(
            rejects(&g, b"{\"o\":2}"),
            "object with only optional and no required must be rejected"
        );
        assert!(
            rejects(&g, b"{\"r\":1,}"),
            "trailing comma must be rejected"
        );
        assert!(
            rejects(&g, b"{\"r\":\"bad\"}"),
            "wrong value type for r must be rejected"
        );
    }

    /// Two trailing optionals after a required property.
    #[test]
    fn object_two_trailing_optionals() {
        let g = compile_ok(
            r#"{
                "type":"object",
                "properties":{
                    "r":{"type":"integer"},
                    "o1":{"type":"integer"},
                    "o2":{"type":"integer"}
                },
                "required":["r"]
            }"#,
        );
        // Required only.
        assert!(accepts(&g, b"{\"r\":1}"), "required-only must be accepted");
        // Required + first optional.
        assert!(
            accepts(&g, b"{\"r\":1,\"o1\":2}"),
            "r + o1 must be accepted"
        );
        // All three present.
        assert!(
            accepts(&g, b"{\"r\":1,\"o1\":2,\"o2\":3}"),
            "all three properties must be accepted"
        );
        // Missing required.
        assert!(rejects(&g, b"{}"), "empty object must be rejected");
        assert!(rejects(&g, b"{\"o1\":2}"), "missing r must be rejected");

        // r + o2 only (skip o1): this is the INTERLEAVED optional case.
        // The CFG is correct, but the runtime PDA commits to the emit-o1
        // alternative when it sees "," and cannot backtrack past the o1 key
        // mismatch.  Assert the actual current behaviour so the test is
        // mutation-sensitive and clearly documents the #353 boundary.
        // #353: interleaved optional — PDA over-rejects; CFG is correct.
        assert!(
            rejects(&g, b"{\"r\":1,\"o2\":3}"),
            "#353: PDA over-rejects r+o2 (interleaved optional, skip o1)"
        );
        // Trailing comma boundary (see object_three_props_trailing_comma_boundary
        // for the full explanation):
        // - r only with trailing comma: correctly rejects (inline-key fix).
        // - r+o1 with trailing comma: correctly rejects. The per-frame
        //   byte-consumption guard refuses to fall back to the o2 skip
        //   alternative once the "," has been consumed.
        // - r+o1+o2 with trailing comma: correctly rejects (no optional remaining).
        assert!(
            rejects(&g, b"{\"r\":1,}"),
            "trailing comma after r-only must reject"
        );
        assert!(
            rejects(&g, b"{\"r\":1,\"o1\":2,}"),
            "trailing comma after r+o1 must reject (no fallback to o2 skip alt)"
        );
        assert!(
            rejects(&g, b"{\"r\":1,\"o1\":2,\"o2\":3,}"),
            "trailing comma after all three must reject"
        );
    }

    /// Required-first re-canonicalization (issue #355 reorder).
    ///
    /// The compiler emits object properties in required-first order (required
    /// properties in declaration order, then optional ones), not the
    /// alphabetical declaration order of the source schema.  This is necessary
    /// for PDA safety: with required properties first, the `started=false` phase
    /// is a single-alternative chain, so the no-rewind PDA cannot commit to a
    /// wrong key terminal (see `compile_object` and `build_object_tail`).
    ///
    /// Consequence: the accepted canonical key order is required-first.  An
    /// object written in a different key order is rejected.  This is the safe
    /// direction for constrained decoding: JSON objects are unordered (RFC 8259),
    /// the model is guided to emit the canonical order, and no invalid output is
    /// ever produced.  It also fixes a real over-rejection: the prior flat
    /// grammar made comma separators mandatory around omitted optionals, so a
    /// required-only object was rejected; required-first construction accepts it.
    #[test]
    fn object_required_first_reorder() {
        // Schema declares r required; serde orders keys alphabetically
        // (o1, o2, r), so required-first reordering moves r to the front.
        let g = compile_ok(
            r#"{
                "type":"object",
                "properties":{
                    "o1":{"type":"integer"},
                    "o2":{"type":"integer"},
                    "r":{"type":"integer"}
                },
                "required":["r"]
            }"#,
        );
        // Required-only: accepted (the #355 fix; the old flat grammar rejected
        // this because the inter-property commas were mandatory).
        assert!(
            accepts(&g, b"{\"r\":1}"),
            "required-only object must be accepted (the #355 fix)"
        );
        // Canonical required-first order: accepted.
        assert!(
            accepts(&g, b"{\"r\":1,\"o1\":2,\"o2\":3}"),
            "required-first canonical order must be accepted"
        );
        // Alphabetical order (required key last): rejected. The accepted
        // canonical order is required-first; this documents the intentional
        // re-canonicalization (a different valid order, not a capability loss).
        assert!(
            rejects(&g, b"{\"o1\":2,\"o2\":3,\"r\":1}"),
            "non-canonical (alphabetical) key order is rejected (required-first re-canonicalization)"
        );
    }

    /// No over-acceptance: invalid JSON must still be rejected after the fix.
    #[test]
    fn object_no_over_acceptance() {
        let g = compile_ok(
            r#"{
                "type":"object",
                "properties":{"r":{"type":"integer"},"o":{"type":"integer"}},
                "required":["r"]
            }"#,
        );
        assert!(
            rejects(&g, b"{\"r\":1,\"o\":\"bad\"}"),
            "wrong type for optional must be rejected"
        );
        assert!(
            rejects(&g, b"{\"r\":1,,\"o\":2}"),
            "double comma must be rejected"
        );
        assert!(rejects(&g, b"{,\"r\":1}"), "leading comma must be rejected");
        assert!(rejects(&g, b"\"r\":1"), "missing braces must be rejected");
        assert!(
            rejects(&g, b"{\"r\":1,\"o\":2,}"),
            "trailing comma after optional must be rejected"
        );
    }

    /// Regression: the fully-present case (`{"r":1,"o":2}`) must be accepted.
    /// This was also broken before the fix — the flat grammar made the comma
    /// mandatory before the optional wrapper, so `{"r":1,"o":2}` was accepted
    /// BUT `{"r":1}` was not.  Verify both directions are now correct.
    #[test]
    fn object_fully_present_and_absent_both_correct() {
        let g = compile_ok(
            r#"{
                "type":"object",
                "properties":{"a":{"type":"string"},"b":{"type":"boolean"}},
                "required":["a"]
            }"#,
        );
        assert!(
            accepts(&g, b"{\"a\":\"x\"}"),
            "required-only string must be accepted"
        );
        assert!(
            accepts(&g, b"{\"a\":\"x\",\"b\":true}"),
            "both present must be accepted"
        );
        assert!(
            accepts(&g, b"{\"a\":\"x\",\"b\":false}"),
            "both present (false) must be accepted"
        );
        assert!(
            rejects(&g, b"{\"b\":true}"),
            "missing required a must be rejected"
        );
        assert!(rejects(&g, b"{}"), "empty must be rejected");
    }

    /// Two fully-optional properties — PDA boundary for trailing-comma and
    /// skip-first behaviours.
    ///
    /// Trailing comma after the first optional (`{"a":1,}`) now correctly
    /// rejects: the per-frame byte-consumption guard refuses to fall back to the
    /// second optional's skip alternative once the "," has been consumed.
    /// Skip-first (`{"b":2}` without `a`) is over-rejected: the no-rewind
    /// single-stack matcher cannot reach the second optional's emit branch after
    /// committing to skip the first.  That is the safe direction for constrained
    /// decoding, where a valid member is unreachable but no invalid output is
    /// ever emitted.
    #[test]
    fn object_two_optional_only_pda_boundary() {
        let g = compile_ok(
            r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}"#,
        );
        assert!(
            accepts(&g, b"{}"),
            "empty object must be accepted when all optional"
        );
        assert!(
            accepts(&g, b"{\"a\":1}"),
            "first optional alone must be accepted"
        );
        assert!(
            accepts(&g, b"{\"a\":1,\"b\":2}"),
            "both present must be accepted"
        );
        // Skip-first ({"b":2} without a) is over-rejected (the safe direction).
        assert!(
            rejects(&g, b"{\"b\":2}"),
            "PDA over-rejects skip-first optional (safe direction)"
        );
        // Trailing comma {"a":1,} correctly rejects: the byte-consumption guard
        // refuses to fall back to the second optional's skip alt once the ","
        // has been consumed.
        assert!(
            rejects(&g, b"{\"a\":1,}"),
            "trailing comma after first optional must reject (no fallback to skip alt)"
        );
        // When both properties are present the trailing comma correctly rejects
        // (no further optional to provide an escape route).
        assert!(
            rejects(&g, b"{\"a\":1,\"b\":2,}"),
            "trailing comma after last property must be rejected"
        );
    }

    /// Three-property schema (1 required + 2 trailing optionals) — documents
    /// trailing-comma boundary for the intermediate optional case.
    ///
    /// `{"r":1,"o1":2,}` correctly rejects: the per-frame byte-consumption guard
    /// refuses to fall back to the o2 skip alternative once the "," has been
    /// consumed (the same guard exercised by
    /// `object_two_optional_only_pda_boundary` above).  `{"r":1,}` rejects
    /// because the inline key terminal makes the o1 mismatch fire at
    /// sym_pos >= 2, which propagates to the required chain (no alt) and rejects.
    /// `{"r":1,"o1":2,"o2":3,}` rejects because no optional remains.
    #[test]
    fn object_three_props_trailing_comma_boundary() {
        let g = compile_ok(
            r#"{
            "type":"object",
            "properties":{
                "r":{"type":"integer"},
                "o1":{"type":"integer"},
                "o2":{"type":"integer"}
            },
            "required":["r"]
        }"#,
        );
        // Correct accepts.
        assert!(accepts(&g, b"{\"r\":1}"), "required-only must be accepted");
        assert!(accepts(&g, b"{\"r\":1,\"o1\":2}"), "r+o1 must be accepted");
        assert!(
            accepts(&g, b"{\"r\":1,\"o1\":2,\"o2\":3}"),
            "all three must be accepted"
        );
        // Inline fix: trailing comma after required-only correctly rejects.
        assert!(
            rejects(&g, b"{\"r\":1,}"),
            "trailing comma after required-only must be rejected (inline-key fix)"
        );
        // Trailing comma after intermediate optional correctly rejects.
        assert!(
            rejects(&g, b"{\"r\":1,\"o1\":2,}"),
            "trailing comma after r+o1 must reject (no fallback to o2 skip alt)"
        );
        // Trailing comma after last property correctly rejects.
        assert!(
            rejects(&g, b"{\"r\":1,\"o1\":2,\"o2\":3,}"),
            "trailing comma after last property must be rejected"
        );
    }

    /// Property-count cap (DoS guard): an object schema with more than
    /// `MAX_OBJECT_PROPERTIES` properties is rejected at the parse boundary
    /// rather than recursing once per property in `build_object_tail` and
    /// exhausting the stack at `GrammarEngine::new`. Mirrors the
    /// `MAX_ARRAY_CARDINALITY` guard. The cap fires before any per-property rule
    /// is materialized, so this test is fast and never deep-recurses; it fails
    /// (compile returns `Ok`) when the cap check is reverted.
    #[test]
    fn object_property_count_cap_rejects() {
        let over = MAX_OBJECT_PROPERTIES + 1;
        let mut props = String::new();
        for i in 0..over {
            if i > 0 {
                props.push(',');
            }
            props.push_str(&format!("\"p{i}\":{{\"type\":\"integer\"}}"));
        }
        let schema_json = format!("{{\"type\":\"object\",\"properties\":{{{props}}}}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("over-cap object schema must be rejected");
        assert!(
            err.0.contains("exceeds the supported limit"),
            "expected a property-count cap error, got: {}",
            err.0
        );
    }

    /// A modest object (well under the cap) compiles and accepts a valid value:
    /// the cap does not reject ordinary schemas.
    #[test]
    fn object_under_property_cap_compiles() {
        let mut props = String::new();
        for i in 0..50 {
            if i > 0 {
                props.push(',');
            }
            props.push_str(&format!("\"p{i}\":{{\"type\":\"integer\"}}"));
        }
        let schema_json = format!("{{\"type\":\"object\",\"properties\":{{{props}}}}}");
        let g = compile_ok(&schema_json);
        assert!(
            accepts(&g, b"{}"),
            "empty object (all optional) must accept"
        );
        assert!(accepts(&g, b"{\"p0\":1}"), "single property must accept");
    }

    /// Literal-count cap via typed string-enum: a schema whose enum members
    /// exceed MAX_STRING_LITERALS is rejected at the parse boundary with a
    /// SchemaError naming the count limit. Fails (compile returns Ok) when the
    /// count guard in compile_trie_literals is reverted.
    #[test]
    fn string_literal_count_cap_typed_enum_rejects() {
        let values: String = (0..=MAX_STRING_LITERALS)
            .map(|i| format!("\"v{i}\""))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"type\":\"string\",\"enum\":[{values}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("over-cap string enum must be rejected");
        assert!(
            err.0.contains("exceeds the supported limit"),
            "expected a literal-count cap error, got: {}",
            err.0
        );
    }

    /// Literal-count cap via anyOf const set: the same count guard fires when
    /// an anyOf schema aggregates more than MAX_STRING_LITERALS string consts.
    /// Proves the shared compile_trie_literals chokepoint is inherited by the
    /// anyOf/oneOf entry point, not just compile_string_type.
    #[test]
    fn string_literal_count_cap_any_of_const_rejects() {
        let branches: String = (0..=MAX_STRING_LITERALS)
            .map(|i| format!("{{\"const\":\"c{i}\"}}"))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"anyOf\":[{branches}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("over-cap anyOf const set must be rejected");
        assert!(
            err.0.contains("exceeds the supported limit"),
            "expected a literal-count cap error, got: {}",
            err.0
        );
    }

    /// Byte-budget cap: a schema whose literals' total encoded bytes exceed 1 MiB
    /// (but whose count is under MAX_STRING_LITERALS) is rejected at the parse
    /// boundary. Fails (compile returns Ok) when the byte-budget guard in
    /// compile_trie_literals is reverted.
    #[test]
    fn string_literal_byte_budget_cap_rejects() {
        // Build 2 strings each slightly over 512 KiB so together they clear 1 MiB
        // while staying well under the 4096 count cap.
        let long_val = "a".repeat(600 * 1024);
        let values = format!("\"{long_val}\",\"{long_val}b\"");
        let schema_json = format!("{{\"type\":\"string\",\"enum\":[{values}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("over-byte-budget string enum must be rejected");
        assert!(
            err.0.contains("exceeds the supported limit"),
            "expected a byte-budget cap error, got: {}",
            err.0
        );
    }

    /// Raw-count cap fires before dedup in anyOf: a schema whose anyOf branches
    /// contain more than MAX_STRING_LITERALS entries — even when all entries are
    /// duplicates of a single value — is rejected before deduplication shrinks
    /// the set below the limit. Fails (compile returns Ok) when the raw-count
    /// guard added before sort/dedup in compile_any_of is reverted (MIN-2).
    #[test]
    fn string_literal_count_cap_any_of_duplicates_rejected_before_dedup() {
        // All branches are the same const; after dedup only 1 remains, which is
        // far under MAX_STRING_LITERALS. The raw count (MAX_STRING_LITERALS + 1)
        // must still be rejected because the cap fires before dedup.
        let branches: String = (0..=MAX_STRING_LITERALS)
            .map(|_| r#"{"const":"dup"}"#.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"anyOf\":[{branches}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v)
            .expect_err("anyOf with duplicate branches exceeding raw count cap must be rejected");
        assert!(
            err.0.contains("exceeds the supported limit"),
            "expected a literal-count cap error, got: {}",
            err.0
        );
    }

    /// Happy path: a small string literal set (well under both caps) compiles
    /// successfully. Ensures the new guards do not reject valid schemas.
    #[test]
    fn string_literal_small_set_compiles() {
        let schema = r#"{"type":"string","enum":["alpha","beta","gamma"]}"#;
        let v: Value = serde_json::from_str(schema).unwrap();
        let g = compile(&v).expect("small string enum must compile");
        assert!(accepts(&g, b"\"alpha\""));
        assert!(accepts(&g, b"\"beta\""));
        assert!(rejects(&g, b"\"delta\""));
    }

    // -----------------------------------------------------------------------
    // Issue #474 — early cardinality caps before untrusted-size allocations
    // -----------------------------------------------------------------------

    /// Early string-enum count guard (issue #474, compile_string_type path):
    /// a `{"type":"string","enum":[...]}` with more than MAX_STRING_LITERALS
    /// DISTINCT one-character members whose total encoded bytes stay under
    /// MAX_STRING_LITERAL_BYTES must be rejected by the EARLY guard inserted
    /// before the byte pre-pass, not by the late guard in compile_trie_literals.
    ///
    /// Mutation pin: the early guard fires with "string enum literal count" in
    /// the message. Reverting the early guard lets the byte pre-pass and
    /// Vec::with_capacity run to completion, after which the late
    /// compile_trie_literals guard fires with "string literal count" — a
    /// different message — so the assertion below fails.
    #[test]
    fn string_enum_count_cap_early_guard_fires_before_alloc() {
        // Build MAX_STRING_LITERALS + 1 distinct single-character strings.
        // There are only 128 ASCII printable values, so we use two-char strings
        // to stay distinct. Total encoded bytes ≈ (MAX_STRING_LITERALS + 1) * 4
        // (each "XY" encodes as 4 bytes including quotes, minus 1 for the
        // pre-pass formula) — well under the 1 MiB byte budget.
        let over = MAX_STRING_LITERALS + 1;
        let values: String = (0..over)
            .map(|i| format!("\"x{i:04}\""))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"type\":\"string\",\"enum\":[{values}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v)
            .expect_err("string enum over MAX_STRING_LITERALS must be rejected before allocation");
        assert!(
            err.0.contains("string enum literal count"),
            "early guard must fire with 'string enum literal count', got: {}",
            err.0
        );
    }

    /// prefixItems length cap (issue #474): a tuple array schema whose
    /// `prefixItems` array has more than MAX_PREFIX_ITEMS entries and no
    /// `maxItems` is rejected at the parse boundary before Vec::with_capacity
    /// and before any positional compile_schema call. Mutation pin: reverting
    /// the cap causes compile to either return Ok (memory exhausted silently) or
    /// OOM; the expect_err assertion fails.
    #[test]
    fn prefix_items_count_cap_rejects() {
        // Build MAX_PREFIX_ITEMS + 1 trivial positional schemas.
        let over = MAX_PREFIX_ITEMS + 1;
        let items: String = (0..over)
            .map(|_| r#"{"type":"string"}"#)
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"type\":\"array\",\"prefixItems\":[{items}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v)
            .expect_err("prefixItems over MAX_PREFIX_ITEMS with no maxItems must be rejected");
        assert!(
            err.0.contains("prefixItems length"),
            "expected a prefixItems-length cap error, got: {}",
            err.0
        );
    }

    /// Happy path: a small prefixItems tuple (well under the cap) compiles and
    /// accepts the correct value. Ensures the new cap does not reject valid schemas.
    #[test]
    fn prefix_items_small_tuple_compiles() {
        let schema = r#"{"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}]}"#;
        let v: Value = serde_json::from_str(schema).unwrap();
        let g = compile(&v).expect("small prefixItems tuple must compile");
        assert!(
            accepts(&g, b"[1,\"hi\"]"),
            "a valid tuple value must be accepted"
        );
    }

    // -----------------------------------------------------------------------
    // Issue #478 — adjacent pre-cap allocations (anyOf literal aggregation,
    // object property/required collection)
    // -----------------------------------------------------------------------

    /// Early anyOf string-literal count guard, many-small-branches path
    /// (issue #478): an anyOf schema whose foldable string-const branches
    /// individually stay tiny but together exceed MAX_STRING_LITERALS must be
    /// rejected by the per-extend guard inside compile_any_of's aggregation
    /// loop, not by the post-loop backstop or the late compile_trie_literals
    /// guard.
    ///
    /// Each branch folds 2 literals (a 2-member untyped string `enum`, not a
    /// single `const`) so the BRANCH count stays well under
    /// MAX_ANYOF_BRANCHES (issue #474 finding 2) while the aggregated LITERAL
    /// count still exceeds MAX_STRING_LITERALS — isolating the per-extend
    /// literal guard from the unrelated branch-count guard, which would
    /// otherwise fire first on a one-literal-per-branch construction sized to
    /// `MAX_STRING_LITERALS + 1` branches.
    ///
    /// Mutation pin: the early guard fires with "anyOf string literal count"
    /// in the message. Reverting it lets `literals` accumulate unchecked
    /// through the loop; the post-loop backstop then fires with the
    /// DIFFERENT "string literal count" message (no "anyOf " prefix), so the
    /// assertion below fails.
    #[test]
    fn anyof_string_literal_count_cap_early_guard_fires_before_alloc() {
        let branch_count = (MAX_STRING_LITERALS / 2) + 2;
        let branches: String = (0..branch_count)
            .map(|i| format!("{{\"enum\":[\"c{i}a\",\"c{i}b\"]}}"))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"anyOf\":[{branches}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err =
            compile(&v).expect_err("anyOf branches summing past the literal cap must be rejected");
        assert!(
            err.0.contains("anyOf string literal count"),
            "early per-extend guard must fire with 'anyOf string literal count', got: {}",
            err.0
        );
    }

    /// Early anyOf string-literal count guard, single-oversized-branch path
    /// (issue #478): ONE anyOf branch with a `{"type":"string","enum":[...]}`
    /// over MAX_STRING_LITERALS members must be rejected by the guard inside
    /// `string_class_of`, before it deep-copies every member into an owned
    /// `String`. Distinct from the many-small-branches accumulation path
    /// above: here a single `string_class_of` call would itself materialize
    /// the oversized Vec if unguarded, before compile_any_of's loop body ever
    /// runs again to check the cumulative total.
    ///
    /// Mutation pin: fires with "anyOf string literal count". Reverting just
    /// this guard still leaves the per-extend guard in compile_any_of, which
    /// independently catches the returned oversized Vec on the same
    /// iteration with the same message — so an ALL-STRING enum like this one
    /// does not, by itself, distinguish the two guards. See
    /// `anyof_single_branch_raw_cardinality_bypasses_filtered_count_guard`
    /// below for a test that isolates the `string_class_of` guard alone.
    #[test]
    fn anyof_single_branch_string_enum_count_cap_rejects() {
        let over = MAX_STRING_LITERALS + 1;
        let values: String = (0..over)
            .map(|i| format!("\"v{i}\""))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"anyOf\":[{{\"type\":\"string\",\"enum\":[{values}]}}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("oversized single anyOf branch enum must be rejected");
        assert!(
            err.0.contains("anyOf string literal count"),
            "expected the anyOf string-literal cap error, got: {}",
            err.0
        );
    }

    /// Isolates the `string_class_of`-internal guard from compile_any_of's
    /// per-extend guard (issue #478): a single anyOf branch declares
    /// `{"type":"string","enum":[...]}` with over MAX_STRING_LITERALS
    /// members, but all except ONE are non-string (integers). `type:"string"`
    /// intersects the enum down to its string members, so the FILTERED count
    /// compile_any_of's per-extend guard ever observes is 1 — far under the
    /// cap. Only a guard on the RAW `arr.len()` inside `string_class_of`,
    /// checked before the filter_map/to_string pass, can catch this: the
    /// per-extend guard cannot, because it never sees the discarded members.
    ///
    /// Mutation pin: reverting the `string_class_of`-internal guard makes
    /// this schema compile successfully (`Ok`) instead of being rejected —
    /// a correctness gap, not just a slower rejection — because nothing else
    /// in the call chain re-checks the raw enum cardinality of one branch.
    #[test]
    fn anyof_single_branch_raw_cardinality_bypasses_filtered_count_guard() {
        let over = MAX_STRING_LITERALS + 1;
        let mut members: Vec<String> = (0..over).map(|i| i.to_string()).collect();
        members.push("\"only-string\"".to_string());
        let values = members.join(",");
        let schema_json = format!("{{\"anyOf\":[{{\"type\":\"string\",\"enum\":[{values}]}}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err(
            "oversized raw enum cardinality must be rejected even with only 1 surviving string",
        );
        assert!(
            err.0.contains("anyOf string literal count"),
            "expected the anyOf string-literal cap error, got: {}",
            err.0
        );
    }

    /// Early object-property count guard (issue #478): an object schema with
    /// more than MAX_OBJECT_PROPERTIES properties is rejected by the guard
    /// checked directly on the property map's `.len()` BEFORE
    /// `.iter().collect::<Vec<_>>()`, with the "object property count"
    /// message proving the pre-collect guard fired.
    ///
    /// Mutation pin: reverting the pre-collect guard removes the only
    /// remaining property-count check in `compile_object` (the old post-collect
    /// check was replaced, not duplicated), so compile would have to either
    /// succeed (wrong) or fail elsewhere with a different message.
    #[test]
    fn object_property_count_early_guard_fires_before_collect() {
        let over = MAX_OBJECT_PROPERTIES + 1;
        let mut props = String::new();
        for i in 0..over {
            if i > 0 {
                props.push(',');
            }
            props.push_str(&format!("\"p{i}\":{{\"type\":\"integer\"}}"));
        }
        let schema_json = format!("{{\"type\":\"object\",\"properties\":{{{props}}}}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("over-cap object schema must be rejected");
        assert!(
            err.0.contains("object property count"),
            "expected the pre-collect object-property-count cap error, got: {}",
            err.0
        );
    }

    /// `required` cardinality is bounded independent of `properties` (issue
    /// #478): an object schema whose `required` array exceeds
    /// MAX_OBJECT_PROPERTIES entries is rejected before the
    /// filter_map/collect into `Vec<&str>`, even when `properties` itself is
    /// tiny. Mutation pin: fires with "object required count"; reverting the
    /// guard lets the oversized array collect unchecked (no rejection at
    /// all, since nothing downstream re-checks `required`'s cardinality).
    #[test]
    fn object_required_count_cap_rejects_before_collect() {
        let over = MAX_OBJECT_PROPERTIES + 1;
        let required_list: String = (0..over)
            .map(|i| format!("\"r{i}\""))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!(
            "{{\"type\":\"object\",\"properties\":{{\"p\":{{\"type\":\"integer\"}}}},\"required\":[{required_list}]}}"
        );
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("over-cap required array must be rejected");
        assert!(
            err.0.contains("object required count"),
            "expected the pre-collect required-count cap error, got: {}",
            err.0
        );
    }

    // -----------------------------------------------------------------------
    // Issue #474 (follow-up) — raw enum cardinality must dominate
    // const-narrowing and non-string materialization, not just the
    // already-guarded plain-enum paths above.
    // -----------------------------------------------------------------------

    /// A sibling `const` narrows the surviving string set to a single value,
    /// but the RAW enum array is still attacker-controlled cardinality. The
    /// guard inside `compile_string_type` must fire on `values.len()` before
    /// the const retain runs, not on the post-retain `str_values.len()` (which
    /// would always be 0 or 1 once narrowed, and so could never trip).
    ///
    /// Mutation pin: fires with "string enum literal count". Reverting the
    /// raw-cardinality guard (restoring a check placed after the const
    /// narrowing) lets this schema compile successfully instead of being
    /// rejected, because the narrowed set is always under the cap.
    #[test]
    fn string_const_plus_oversized_enum_rejected_by_raw_guard() {
        let over = MAX_STRING_LITERALS + 1;
        let values: String = (0..over)
            .map(|i| format!("\"v{i}\""))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"type\":\"string\",\"const\":\"v0\",\"enum\":[{values}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err(
            "a const that narrows the enum to one survivor must not bypass the raw cardinality cap",
        );
        assert!(
            err.0.contains("string enum literal count"),
            "expected the string-enum literal-count cap error, got: {}",
            err.0
        );
    }

    /// Same const-narrowing bypass as above, but inside an `anyOf` branch,
    /// exercising the `string_class_of` `(Some(c), Some(arr))` arm rather than
    /// `compile_string_type`. Before the fix this arm scanned
    /// `arr.iter().any(...)` for the const-membership check without ever
    /// bounding `arr.len()`, while the sibling `(None, Some(arr))` arm was
    /// already guarded.
    ///
    /// Mutation pin: fires with "anyOf string literal count". Reverting the
    /// guard hoisted ahead of the `match (const_val, enum_arr)` dispatch lets
    /// this schema compile successfully instead of being rejected.
    #[test]
    fn anyof_string_const_plus_oversized_enum_rejected() {
        let over = MAX_STRING_LITERALS + 1;
        let values: String = (0..over)
            .map(|i| format!("\"v{i}\""))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json =
            format!("{{\"anyOf\":[{{\"type\":\"string\",\"const\":\"v0\",\"enum\":[{values}]}}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err(
            "an anyOf branch's const-narrowed enum must not bypass the raw cardinality cap",
        );
        assert!(
            err.0.contains("anyOf string literal count"),
            "expected the anyOf string-literal cap error, got: {}",
            err.0
        );
    }

    /// A mixed/non-string `enum` (no `type:"string"`, not all-string members)
    /// is routed to `compile_enum`, which allocates one `Alt` per member via
    /// `.map(json_value_to_alt).collect()`. That allocation was previously
    /// unbounded: only the all-string fast path (`compile_string_type`) had a
    /// cardinality guard.
    ///
    /// Mutation pin: fires with "enum value count". Reverting the guard added
    /// at the top of `compile_enum` lets this schema compile successfully
    /// instead of being rejected.
    #[test]
    fn non_string_enum_cardinality_capped() {
        let over = MAX_STRING_LITERALS + 1;
        let values: String = (0..over)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"enum\":[{values}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("a non-string enum over the cap must be rejected");
        assert!(
            err.0.contains("enum value count"),
            "expected the enum-value-count cap error, got: {}",
            err.0
        );
    }

    // -----------------------------------------------------------------------
    // Issue #474 (follow-up) — anyOf branch-count cap and byte
    // budgets that were checked late (or not at all) previously.
    // -----------------------------------------------------------------------

    /// anyOf/oneOf branch-count cap (issue #474, finding 2): a schema whose
    /// `anyOf` array declares more than MAX_ANYOF_BRANCHES entries is
    /// rejected at the top of `compile_any_of`, before the classification
    /// loop runs. anyOf/oneOf branch count was the one cardinality dimension
    /// this file never bounded.
    ///
    /// Mutation pin: fires with "anyOf/oneOf branch count". Reverting the
    /// guard lets `compile_any_of` push one `other_subs` entry (and later one
    /// recursive `compile_schema` call) per branch with no upper bound, so
    /// compile returns `Ok` instead of `Err` and the assertion below fails.
    #[test]
    fn anyof_branch_count_capped() {
        let over = MAX_ANYOF_BRANCHES + 1;
        let branches: String = (0..over)
            .map(|_| r#"{"type":"boolean"}"#.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"anyOf\":[{branches}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("anyOf branch count over the cap must be rejected");
        assert!(
            err.0.contains("anyOf/oneOf branch count"),
            "expected the anyOf branch-count cap error, got: {}",
            err.0
        );
    }

    /// Mixed-enum byte-budget cap (issue #474, finding 3): a non-string enum
    /// (one oversized string member plus a non-string member, so `type` is
    /// absent and members are not all-string, routing to `compile_enum`
    /// rather than `compile_string_type`) whose encoded bytes exceed
    /// MAX_STRING_LITERAL_BYTES is rejected before the per-member
    /// `json_value_to_alt` allocation.
    ///
    /// Backstopped: the schema-wide `guard_schema_string_bytes` pass at the
    /// top of `compile_json_schema` now dominates and rejects this input before
    /// `compile_enum`'s own byte-budget guard runs, so this test is
    /// defense-in-depth for `compile_enum`'s guard, not independently
    /// mutation-sensitive to it — see `schema_defs_name_byte_capped` for the
    /// guard pinned to the entry-point pass.
    ///
    /// Mutation pin: fires with "byte length" (a substring common to both the
    /// entry-point guard's message and `compile_enum`'s own "enum value
    /// encoded byte length" message, so this assertion holds regardless of
    /// which guard catches it).
    #[test]
    fn mixed_enum_byte_budget_capped() {
        // One member well over the 1 MiB byte budget by itself, plus a
        // non-string member so this enum is not all-string and is routed to
        // compile_enum (not compile_string_type's own byte guard).
        let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 16);
        let schema_json = format!("{{\"enum\":[\"{huge}\",1]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("mixed enum over the byte budget must be rejected");
        assert!(
            err.0.contains("byte length"),
            "expected the byte-length cap error, got: {}",
            err.0
        );
    }

    /// Mixed-enum CUMULATIVE byte-budget cap (issue #474): a mixed enum (a
    /// trailing non-string member keeps it routed to `compile_enum` rather than
    /// `compile_string_type`) whose individual string members are each well
    /// under MAX_STRING_LITERAL_BYTES — so the schema-wide entry guard passes
    /// every one — but whose COMBINED encoded bytes exceed the budget is
    /// rejected by `compile_enum`'s incremental byte-budget guard.
    ///
    /// Mutation pin: asserts the incremental guard's exact "enum value encoded
    /// byte length" message. The schema-wide `guard_schema_string_bytes` entry
    /// pass cannot fire here (no single member exceeds the cap), so reverting
    /// `compile_enum`'s incremental guard drops the input to the late
    /// `compile_trie_literals` backstop, whose "string literal encoded byte
    /// length (N)" message lacks the "enum value" substring — making this the
    /// only test that pins the incremental enum guard specifically (its
    /// single-oversized-member sibling `mixed_enum_byte_budget_capped` is now
    /// dominated by the entry guard).
    #[test]
    fn mixed_enum_cumulative_bytes_capped() {
        // Three ~400 KiB string members, each under the 1 MiB per-literal cap
        // (so the entry guard passes each), summing past 1 MiB; a trailing
        // non-string member keeps the enum mixed and routed to compile_enum.
        let chunk = "a".repeat(400 * 1024);
        let members: String = (0..3)
            .map(|i| format!(r#""{chunk}{i}""#))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"enum\":[{members},1]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v)
            .expect_err("mixed enum whose cumulative bytes exceed the budget must be rejected");
        assert!(
            err.0
                .contains("enum value encoded byte length exceeds the supported limit"),
            "expected the incremental enum byte-budget cap error, got: {}",
            err.0
        );
    }

    /// anyOf const cumulative byte-budget cap (issue #474, finding 4): an
    /// `anyOf` whose individual `const` branches are each modest but whose
    /// COMBINED encoded bytes exceed MAX_STRING_LITERAL_BYTES — while the
    /// branch COUNT (3) stays far under MAX_STRING_LITERALS — is rejected by
    /// the incremental byte-budget guard inside `compile_any_of`'s
    /// classification loop, not only by the late backstop in
    /// `compile_trie_literals`.
    ///
    /// Mutation pin: the incremental guard's message has no number between
    /// "byte length" and "exceeds" ("string literal encoded byte length
    /// exceeds the supported limit"); the late `compile_trie_literals`
    /// backstop's message embeds the computed total in between ("byte length
    /// (N) exceeds"). Reverting the incremental guard still leaves the late
    /// backstop to reject (since dedup does not shrink 3 distinct large
    /// strings), but with the DIFFERENT embedded-number message, so the exact
    /// substring assertion below fails.
    #[test]
    fn anyof_const_cumulative_bytes_capped() {
        // Each branch is ~400 KiB alone; three of them sum past 1 MiB while
        // the branch count (3) stays far under MAX_STRING_LITERALS.
        let chunk = "a".repeat(400 * 1024);
        let branches: String = (0..3)
            .map(|i| format!(r#"{{"const":"{chunk}{i}"}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let schema_json = format!("{{\"anyOf\":[{branches}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v)
            .expect_err("anyOf consts whose cumulative bytes exceed the budget must be rejected");
        assert!(
            err.0
                .contains("string literal encoded byte length exceeds the supported limit"),
            "expected the incremental anyOf byte-budget cap error, got: {}",
            err.0
        );
    }

    // -----------------------------------------------------------------------
    // Issue #474 (follow-up) — single-oversized-literal byte cap.
    // The earlier fixes capped cardinality (many literals); this closes the
    // distinct sub-class where ONE literal's own byte length exceeds
    // MAX_STRING_LITERAL_BYTES before any prior guard sees it.
    // -----------------------------------------------------------------------

    /// Top-level const byte cap (issue #474): a top-level `const`
    /// whose byte length exceeds MAX_STRING_LITERAL_BYTES is rejected by the
    /// `guard_literal_bytes` check in `json_value_to_alt`, which has no other
    /// byte guard upstream of it.
    ///
    /// Mutation pin: fires with "byte length". Reverting the guard in
    /// `json_value_to_alt` lets `serde_json::to_string` and the per-byte
    /// `Symbol::Terminal` collect run to completion, so compile returns `Ok`
    /// instead of `Err` and the assertion below fails.
    #[test]
    fn const_literal_byte_capped() {
        let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
        let schema_json = format!("{{\"const\":\"{huge}\"}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("a const literal over the byte cap must be rejected");
        assert!(
            err.0.contains("byte length"),
            "expected the byte-length cap error, got: {}",
            err.0
        );
    }

    /// Object property key byte cap (issue #474): an object property
    /// whose KEY (not value) exceeds MAX_STRING_LITERAL_BYTES is rejected by
    /// the `guard_literal_bytes` checks in `compile_object` (raw key) and
    /// `json_string_literal` (JSON-encoded key), before the key reaches the
    /// `into_bytes` alloc or the per-byte `Symbol::Terminal` expansion.
    ///
    /// Mutation pin: fires with "byte length". Reverting BOTH guards lets the
    /// key flow through unbounded, so compile returns `Ok` instead of `Err`
    /// and the assertion below fails. (Reverting only one guard leaves the
    /// other as a backstop — the pair is what this test pins.)
    #[test]
    fn object_key_byte_capped() {
        let huge_key = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
        let schema_json = format!(
            "{{\"type\":\"object\",\"properties\":{{\"{huge_key}\":{{\"type\":\"integer\"}}}}}}"
        );
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("an object key over the byte cap must be rejected");
        assert!(
            err.0.contains("byte length"),
            "expected the byte-length cap error, got: {}",
            err.0
        );
    }

    /// anyOf const single-byte cap (issue #474): an `anyOf` branch
    /// with a single oversized `const` is rejected by the `guard_literal_bytes`
    /// check in `string_class_of`'s const arm.
    ///
    /// Backstopped: the earlier incremental byte-budget guard inside
    /// `compile_any_of`'s classification loop (Fix 4) also catches this input,
    /// so this test stays green even if this guard alone is reverted.
    /// It is defense-in-depth for the transient ~1x copy at the exact line,
    /// not independently mutation-sensitive — see
    /// `object_key_byte_capped` / `const_literal_byte_capped` for the
    /// mutation-pinned guards.
    #[test]
    fn anyof_const_single_byte_capped() {
        let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
        let schema_json = format!("{{\"anyOf\":[{{\"const\":\"{huge}\"}}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("an anyOf const over the byte cap must be rejected");
        assert!(
            err.0.contains("byte length"),
            "expected the byte-length cap error, got: {}",
            err.0
        );
    }

    /// anyOf enum member single-byte cap (issue #474): an `anyOf`
    /// branch with a single oversized `enum` member is rejected by the
    /// `guard_literal_bytes` checks in `string_class_of`'s enum arms.
    ///
    /// Backstopped: same as `anyof_const_single_byte_capped` — the earlier
    /// incremental byte-budget guard in `compile_any_of` also catches this
    /// input, so this test is defense-in-depth, not independently
    /// mutation-sensitive.
    #[test]
    fn anyof_enum_member_byte_capped() {
        let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
        let schema_json = format!("{{\"anyOf\":[{{\"enum\":[\"{huge}\"]}}]}}");
        let v: Value = serde_json::from_str(&schema_json).unwrap();
        let err = compile(&v).expect_err("an anyOf enum member over the byte cap must be rejected");
        assert!(
            err.0.contains("byte length"),
            "expected the byte-length cap error, got: {}",
            err.0
        );
    }

    // -----------------------------------------------------------------------
    // Issue #474 (entry-point convergent fix) — validate every schema string
    // once at `compile`, before any per-site guard runs. Closes sibling sites
    // (e.g. a `$defs` name) that have no per-site byte guard of their own.
    // -----------------------------------------------------------------------

    /// `$defs` name byte cap (issue #474): a `$defs` map whose KEY (the
    /// definition name itself, not a `properties` key) exceeds
    /// MAX_STRING_LITERAL_BYTES is rejected by the schema-wide
    /// `guard_schema_string_bytes` pass at the top of `compile_json_schema`. Unlike a
    /// `properties` key (`object_key_byte_capped`), a `$defs` name has no
    /// per-site guard: `CompileCtx::new` clones it into the `defs` map
    /// unchecked, and nothing in this schema references it via `$ref`, so no
    /// downstream lookup ever sees it either.
    ///
    /// Mutation pin: fires with "byte length". Reverting the
    /// `guard_schema_string_bytes` call in `compile_json_schema` removes the only
    /// guard that ever inspects this key, so compile returns `Ok` instead of `Err`
    /// and the assertion below fails.
    #[test]
    fn schema_defs_name_byte_capped() {
        let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
        let mut defs = serde_json::Map::new();
        defs.insert(huge, serde_json::json!({"type": "integer"}));
        let mut root = serde_json::Map::new();
        root.insert("$defs".to_string(), Value::Object(defs));
        let v = Value::Object(root);
        let err = compile(&v).expect_err("a $defs name over the byte cap must be rejected");
        assert!(
            err.0.contains("byte length"),
            "expected the byte-length cap error, got: {}",
            err.0
        );
    }

    /// `compile_json_schema` is itself `pub`, so a direct caller bypasses the
    /// `compile` wrapper. The schema-wide byte guard must therefore live in
    /// `compile_json_schema`, not only in `compile`: an oversized `$defs` name
    /// passed straight to `compile_json_schema` must still be rejected before
    /// `CompileCtx::new` clones it into the `defs` map.
    ///
    /// Mutation pin: fires with "byte length". Moving the
    /// `guard_schema_string_bytes` call back into `compile` (leaving
    /// `compile_json_schema` unguarded) lets the oversized key reach
    /// `CompileCtx::new`, so this direct call returns `Ok` and the assertion fails.
    #[test]
    fn compile_json_schema_direct_defs_name_byte_capped() {
        let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
        let mut defs = serde_json::Map::new();
        defs.insert(huge, serde_json::json!({"type": "integer"}));
        let mut root = serde_json::Map::new();
        root.insert("$defs".to_string(), Value::Object(defs));
        let v = Value::Object(root);
        let err = compile_json_schema(&v)
            .expect_err("a $defs name over the byte cap must be rejected at the direct entry");
        assert!(
            err.0.contains("byte length"),
            "expected the byte-length cap error, got: {}",
            err.0
        );
    }

    /// `$defs` name CUMULATIVE byte cap (issue #474): the schema-wide
    /// entry guard is per-string, so many definition names each UNDER the
    /// per-string cap can still sum past it while `CompileCtx::new` clones every
    /// key into the `defs` map (and `$defs` has no cardinality cap). The
    /// cumulative budget in `CompileCtx::new` rejects the aggregate.
    ///
    /// Mutation pin: fires with the distinct "schema definition name cumulative
    /// byte length" message. Each name here is ~400 KiB (under the 1 MiB
    /// per-string cap, so the entry guard passes each one); three of them sum
    /// past the cap. Reverting the cumulative guard in `CompileCtx::new` lets all
    /// three through, so compile returns `Ok` and the assertion below fails.
    #[test]
    fn defs_names_cumulative_bytes_capped() {
        let chunk = "a".repeat(400 * 1024);
        let mut defs = serde_json::Map::new();
        for i in 0..3 {
            defs.insert(
                format!("{chunk}{i}"),
                serde_json::json!({"type": "integer"}),
            );
        }
        let mut root = serde_json::Map::new();
        root.insert("$defs".to_string(), Value::Object(defs));
        let v = Value::Object(root);
        let err =
            compile(&v).expect_err("cumulative $defs name bytes over the budget must be rejected");
        assert!(
            err.0
                .contains("schema definition name cumulative byte length"),
            "expected the cumulative $defs-name cap error, got: {}",
            err.0
        );
    }

    /// Object property key CUMULATIVE byte cap (issue #474): the
    /// per-key `guard_literal_bytes` cap and `MAX_OBJECT_PROPERTIES` count cap
    /// each bound one dimension, but their product (many near-cap keys) still
    /// drives the per-byte `json_string_literal` Symbol expansion and the
    /// retained `key_bytes` vectors. The cumulative budget in the property loop
    /// rejects the aggregate.
    ///
    /// Mutation pin: fires with the distinct "object property key cumulative
    /// byte length" message. Each key here is ~400 KiB (under the per-string
    /// cap, so `guard_literal_bytes` passes each one); three of them sum past
    /// the cap. Reverting the cumulative guard in the property loop lets all
    /// three through, so compile returns `Ok` and the assertion below fails.
    #[test]
    fn object_keys_cumulative_bytes_capped() {
        let chunk = "a".repeat(400 * 1024);
        let mut props = serde_json::Map::new();
        for i in 0..3 {
            props.insert(
                format!("{chunk}{i}"),
                serde_json::json!({"type": "integer"}),
            );
        }
        let mut root = serde_json::Map::new();
        root.insert("type".to_string(), Value::String("object".to_string()));
        root.insert("properties".to_string(), Value::Object(props));
        let v = Value::Object(root);
        let err = compile(&v)
            .expect_err("cumulative object property key bytes over the budget must be rejected");
        assert!(
            err.0.contains("object property key cumulative byte length"),
            "expected the cumulative object-key cap error, got: {}",
            err.0
        );
    }
}
