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
use std::collections::{HashMap, HashSet};

mod schema_compilation;

use schema_compilation::{
    RefStr, StrClass, build_trie_node, check_one_of_exclusivity, combine_ref_str, compile_const,
    compile_enum, empty_object_as_alt_with_ws, extract_ref_name, flatten_any_of_branches,
    fold_literals_into, fold_string_members, guard_literal_bytes, guard_schema_string_bytes,
    intersect_narrowing, intersect_property_schemas, json_string_literal, node_string_narrowing,
    ref_sub_forces_string, register_builtins, schema_type_names, string_class_of,
    unrepresentable_ref_narrowing, value_matches_type,
};

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

/// Keys alongside `$ref` that `compile_ref_with_siblings` (issue #1078) never
/// treats as an assertion needing conjunction or rejection: `$ref` itself;
/// `$defs`/`definitions`, the document's own definitions container (a schema
/// with a root-level `$ref` commonly carries `$defs` in the very same object,
/// see `deep_ref_chain_rejected` in `engine.rs`); and the rest, pure
/// annotations with no grammar meaning.
const REF_IGNORED_SIBLING_KEYS: &[&str] = &[
    "$ref",
    "$defs",
    "definitions",
    "$id",
    "$schema",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// Supported assertion keywords whose conjunction with a `$ref` target cannot
/// be represented by the current grammar compiler.
///
/// Silently compiling the target alone widens the schema, so these siblings
/// must fail closed until their intersections are implemented.
const REF_NARROWING_SIBLING_KEYS: &[&str] = &["const", "enum", "type"];

/// Annotation keys that cannot narrow the resolved target's instance language.
///
/// Structural keys from `REF_IGNORED_SIBLING_KEYS` are deliberately excluded:
/// `$ref` would mean resolution stopped early, while `$defs`, `definitions`,
/// `$id`, and `$schema` describe schema structure rather than emitted values.
const REF_TARGET_ANNOTATION_KEYS: &[&str] = &[
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// Error returned by the JSON Schema compiler.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaError(pub String);

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON Schema compile error: {}", self.0)
    }
}

impl std::error::Error for SchemaError {}

impl From<crate::grammar::pda::BuilderError> for SchemaError {
    fn from(e: crate::grammar::pda::BuilderError) -> Self {
        SchemaError(e.0)
    }
}

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
    ctx.builder.set_alts(root_id, root_alts)?;
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
        register_builtins(&mut builder)?;
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
    ///     to `other_subs`, where the general `$ref` path would reject an
    ///     intersection this specialized fold can represent exactly;
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
                        // Shape 1 (issue #473): a `$ref` branch. The general
                        // `$ref` path rejects unrepresentable `const`/`enum`/`type`
                        // intersections rather than widening them.
                        // `ref_string_contribution` handles the string subset here
                        // before that path: terminal string class ∩ chain narrowing.
                        match self.ref_string_contribution(sub)? {
                            RefStr::Broad => broad_string = true,
                            RefStr::Literals(values) => {
                                // A string terminal has no non-string language, so the
                                // hoisted set fully represents the branch — it must NOT
                                // also go to `other_subs`, where the generic `$ref`
                                // compiler would reject the already-represented
                                // intersection. An empty set is a dead branch (empty
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
                                // integer. The forcing node may be an intermediate
                                // `$defs` link (`{$ref: N, type:"string"}`), not just
                                // the outer sub, so this decision is made across the
                                // whole chain in `ref_string_contribution`. Dropping
                                // the branch is exactly faithful and avoids rejecting a
                                // conjunction this specialized path can prove empty.
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
        // Handle `$ref` siblings before keyword dispatch: object-field
        // conjunctions are merged, while unrepresentable supported narrowing
        // keywords fail closed instead of widening the schema.
        if let Some(ref_str) = schema.get("$ref").and_then(Value::as_str) {
            return self.compile_ref_with_siblings(ref_str, schema);
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
        if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
            return self.compile_any_of(any_of, _path);
        }
        if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
            // `oneOf` requires mutual exclusivity (exactly one branch matches);
            // `compile_any_of` only enforces "at least one" (issue #1077). Full
            // exclusivity would need the no-rewind PDA to track which branch(es)
            // a completed value matches, which this single-stack byte matcher
            // cannot do (same class of limitation as the shared-prefix
            // over-rejection documented in pda.rs). Rather than ship a partial
            // runtime tracker that would sometimes over-reject valid input,
            // catch the one case that IS cheaply and soundly decidable at
            // compile time: branches that reduce to a closed literal value set
            // (`const`/`enum`). See `check_one_of_exclusivity`.
            check_one_of_exclusivity(one_of)?;
            return self.compile_any_of(one_of, _path);
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

    /// Compile a `$ref`, applying `required`/`properties` siblings on the same
    /// schema object as a conjunction with the resolved target instead of
    /// dropping them (issue #1078). This compiler targets JSON Schema draft
    /// 2020-12 (see the module doc and `spec.rs`), under which `$ref` is an
    /// ordinary applicator: sibling keywords apply IN ADDITION to the
    /// reference, not instead of it (draft-07's replace-the-object semantics do
    /// not apply here).
    ///
    /// Scope: `required`/`properties` siblings on a `$ref` to an object-typed
    /// target are merged — the common "extend a base object" pattern.
    /// Supported value/type assertions (`const`, `enum`, and `type`) are
    /// compiled away only when the resolved target's language is provably a
    /// subset of each sibling assertion. Otherwise their conjunction is
    /// rejected: compiling the target alone would silently widen the schema.
    /// Other deferred keywords retain the plain `$ref` fallback.
    ///
    /// Kept deliberately tiny (only the sibling-key scan, no merge locals) and
    /// out of the merge body (`merge_ref_with_siblings`, `#[inline(never)]`):
    /// `compile_schema` re-enters this dispatch once per `$ref` chain link, so
    /// this function's own stack frame size is multiplied by the chain depth
    /// before `MAX_SCHEMA_DEPTH` rejects it (issue #343). The plain,
    /// no-siblings `$ref` — the overwhelmingly common case, including every
    /// link of a long reference chain — must stay as cheap as it was before
    /// this fix. A regression test (`deep_ref_chain_rejected` in
    /// `engine.rs`) exercises a 2000-link chain specifically to catch this: an
    /// earlier version of this fix folded the merge locals into this
    /// function directly and overflowed the test-thread stack at native
    /// recursion depth ~512, well before `MAX_SCHEMA_DEPTH` had a chance to
    /// return its clean rejection.
    fn compile_ref_with_siblings(
        &mut self,
        ref_str: &str,
        schema: &'a Value,
    ) -> Result<Vec<Alt>, SchemaError> {
        if schema.as_object().is_some_and(|m| {
            m.keys()
                .any(|key| REF_NARROWING_SIBLING_KEYS.contains(&key.as_str()))
        }) {
            return self.compile_ref_with_redundant_narrowing(ref_str, schema);
        }

        let mergeable = schema.as_object().is_some_and(|m| {
            let mut saw_sibling = false;
            for k in m.keys() {
                let k = k.as_str();
                // `$ref` itself, the document-structure keywords a schema
                // commonly carries alongside a root-level `$ref` (`$defs`,
                // `definitions`), and pure annotations carry no grammar
                // meaning and must not block the merge or be treated as an
                // unsupported assertion.
                if REF_IGNORED_SIBLING_KEYS.contains(&k) {
                    continue;
                }
                if k != "required" && k != "properties" {
                    return false;
                }
                saw_sibling = true;
            }
            saw_sibling
        });
        if !mergeable {
            return self.compile_ref(ref_str);
        }
        self.merge_ref_with_siblings(ref_str, schema)
    }

    /// Prove every narrowing sibling redundant against the resolved target,
    /// then compile that target alone.
    ///
    /// A target `const`/`enum` value set and its declared scalar `type` are
    /// upper bounds on its emitted language. The proof is available only for
    /// the small target shapes modeled below.
    ///
    /// Kept out of [`Self::compile_ref_with_siblings`] because that function is
    /// re-entered once per `$ref` chain link and must retain a small stack
    /// frame.
    #[inline(never)]
    fn compile_ref_with_redundant_narrowing(
        &mut self,
        ref_str: &str,
        schema: &'a Value,
    ) -> Result<Vec<Alt>, SchemaError> {
        let siblings = schema
            .as_object()
            .ok_or_else(|| SchemaError("schema containing `$ref` must be an object".to_string()))?;
        let narrowing_key = siblings
            .keys()
            .find(|key| REF_NARROWING_SIBLING_KEYS.contains(&key.as_str()))
            .map(String::as_str)
            .ok_or_else(|| SchemaError("missing narrowing `$ref` sibling".to_string()))?;

        // Compiling the target alone drops every sibling, not just the one
        // proven redundant. Mixing this exemption with another assertion such
        // as `required` would therefore silently widen through that assertion.
        if siblings.keys().any(|key| {
            !REF_IGNORED_SIBLING_KEYS.contains(&key.as_str())
                && !REF_NARROWING_SIBLING_KEYS.contains(&key.as_str())
        }) {
            return Err(unrepresentable_ref_narrowing(narrowing_key));
        }

        let target = self.resolve_ref_chain_target(ref_str)?;
        let target_shape_is_modeled = target.as_object().is_some_and(|object| {
            !object.is_empty()
                && object.keys().all(|key| {
                    matches!(key.as_str(), "const" | "enum" | "type")
                        || REF_TARGET_ANNOTATION_KEYS.contains(&key.as_str())
                })
                && !(object.contains_key("const") && object.contains_key("enum"))
                && object.get("enum").is_none_or(Value::is_array)
                && object.get("type").is_none_or(Value::is_string)
        });
        // This is intentionally an allow-list rather than a list of known
        // compiler dispatch keywords. A newly supported keyword must make the
        // proof unavailable by default: rejecting a redundant schema is safer
        // than silently widening constrained decoding through an unmodeled
        // early return.
        if !target_shape_is_modeled {
            return Err(unrepresentable_ref_narrowing(narrowing_key));
        }

        let target_const = target.get("const");
        let target_enum = target.get("enum").and_then(Value::as_array);

        if let Some(Value::String(name)) = target.get("type") {
            if target_const.is_some_and(|value| !value_matches_type(value, name)) {
                return Err(unrepresentable_ref_narrowing(narrowing_key));
            }
            if target_enum
                .is_some_and(|values| values.iter().any(|value| !value_matches_type(value, name)))
            {
                return Err(unrepresentable_ref_narrowing(narrowing_key));
            }
        }

        let target_values: Option<Vec<&Value>> = if let Some(value) = target.get("const") {
            // `Value` equality distinguishes 1 from 1.0 more strictly than
            // JSON Schema. That can refuse a proof, but cannot create one.
            Some(vec![value])
        } else {
            target
                .get("enum")
                .and_then(Value::as_array)
                .filter(|values| !values.is_empty())
                .map(|values| values.iter().collect())
        };

        for key in REF_NARROWING_SIBLING_KEYS {
            let Some(sibling) = siblings.get(*key) else {
                continue;
            };
            let redundant = match *key {
                "const" => target_values
                    .as_ref()
                    .is_some_and(|values| values.iter().all(|value| *value == sibling)),
                "enum" => sibling.as_array().is_some_and(|allowed| {
                    target_values
                        .as_ref()
                        .is_some_and(|values| values.iter().all(|value| allowed.contains(value)))
                }),
                "type" => {
                    let sibling_types = schema_type_names(sibling)?;
                    match target.get("type") {
                        Some(Value::String(name)) => sibling_types.contains(&name.as_str()),
                        // The normal compiler does not currently materialize
                        // target-side type arrays, so compiling that target
                        // alone would widen even without considering the
                        // sibling. Do not use such an unrepresented bound.
                        Some(_) => false,
                        None => target_values.as_ref().is_some_and(|values| {
                            values.iter().all(|value| {
                                sibling_types
                                    .iter()
                                    .any(|name| value_matches_type(value, name))
                            })
                        }),
                    }
                }
                _ => unreachable!("narrowing keys are a closed constant"),
            };
            if !redundant {
                return Err(unrepresentable_ref_narrowing(key));
            }
        }

        self.compile_ref(ref_str)
    }

    /// The actual `required`/`properties` conjunction merge for
    /// [`Self::compile_ref_with_siblings`], split out (and `#[inline(never)]`)
    /// so its locals do not enlarge that function's per-recursion stack frame
    /// — see the doc comment there.
    #[inline(never)]
    fn merge_ref_with_siblings(
        &mut self,
        ref_str: &str,
        schema: &'a Value,
    ) -> Result<Vec<Alt>, SchemaError> {
        let target: &'a Value = self.resolve_ref_chain_target(ref_str)?;

        let target_type = target.get("type").and_then(Value::as_str);

        if let Some(t) = target_type
            && t != "object"
        {
            // `required` and `properties` constrain object instances only, so
            // against a target pinned to some other type they are vacuous:
            // dropping them cannot widen the language. Compile the target
            // alone.
            //
            // This is deliberately keyed on an EXPLICIT non-object `type`, not
            // on the absence of `"object"`. A target with no `type` (or a
            // `type` array) still matches object instances, so its siblings are
            // NOT vacuous — that case falls through to the representability
            // check below and is rejected rather than silently dropped.
            return self.compile_ref(ref_str);
        }

        // The merge reconstructs the target from `properties`/`required` alone
        // (the `compile_object_fields` call that ends this function), so it is
        // only faithful when those are the only assertions the target carries.
        // A target that also constrains itself comes back WIDER than it went
        // in, because `compile_schema_inner` dispatches `enum`/`const` ahead of
        // object compilation and returns immediately: the plain `$ref` path
        // compiles
        // `{"type":"object","enum":[{"x":1}],"properties":{"x":{"type":"integer"}}}`
        // to exactly `{"x":1}`, while a merge that sees only the property and
        // the required key admits `{"x":2}`. Dropping a sibling over-accepts by
        // one keyword; discarding a target assertion over-accepts by the whole
        // referenced schema, so this falls back to the plain path rather than
        // merging.
        //
        // Positive allowlist, not a denylist of known assertions: a keyword the
        // compiler gains later then lands in the safe path by default instead of
        // being silently discarded by a merge that has never heard of it.
        // `$ref` is deliberately NOT allowed here even though it is an ignorable
        // *sibling* — on the target it means an indirection
        // `resolve_ref_chain_target` declined to follow, and compiling the
        // target's properties while ignoring it would widen in the same way.
        let target_is_representable = target.as_object().is_some_and(|m| {
            m.keys().all(|k| {
                let k = k.as_str();
                (k != "$ref" && REF_IGNORED_SIBLING_KEYS.contains(&k))
                    || k == "type"
                    || k == "properties"
                    || k == "required"
            })
        });
        if !target_is_representable || target_type != Some("object") {
            // Falling back to the plain `$ref` path here is NOT safe, and the
            // tempting argument for it is subtly wrong in a way worth naming.
            //
            // That argument runs: for a target carrying `enum`/`const` — which
            // `compile_schema_inner` dispatches ahead of object compilation —
            // the plain path compiles the target's own narrow language, so
            // dropping the sibling cannot widen past the referenced schema.
            // Every step is true and the conclusion still does not follow. The
            // schema's language is the CONJUNCTION of target and siblings,
            // which is narrower than the target alone; "no wider than the
            // target" is therefore not the bound that matters. With
            // `$defs.Base = {"type":"object","enum":[{},{"x":1}]}` and a
            // `required: ["x"]` sibling, the schema admits only `{"x":1}`,
            // while the target alone also admits `{}`.
            //
            // For a keyword this compiler does not support at all
            // (`additionalProperties`, `minProperties`, `patternProperties`,
            // ...) the plain path is worse still: it IGNORES the keyword, so
            // the fallback keeps nothing from the target AND drops the
            // sibling. An untyped target is the same shape once more — it
            // compiles to "any JSON value", which admits every object the
            // siblings exist to exclude.
            //
            // So every non-representable target is rejected, with no carve
            // out. A constrained-decoding grammar that accepts strings the
            // schema forbids fails at the caller's parser, after the tokens
            // are spent; a schema this compiler declines to compile fails
            // here, loudly, at compile time. Filtering a finite `enum` by the
            // siblings would be a real conjunction and is the natural way to
            // lift this, but it needs instance validation this compiler does
            // not have, so it is deliberately left undone rather than
            // approximated.
            return Err(SchemaError(format!(
                "`$ref` to {ref_str} carries `required`/`properties` siblings that must \
                 be applied as a conjunction, but the target is not expressible as one: \
                 it is {}. Inline the intersection at the reference site.",
                match target_type {
                    Some("object") =>
                        "an object carrying assertions beyond \
                                       `properties`/`required` that this compiler cannot \
                                       reproduce",
                    Some(_) => unreachable!("explicit non-object types returned above"),
                    None =>
                        "untyped, so it also matches instances the siblings cannot \
                             constrain",
                }
            )));
        }

        // Merge `properties`. A key declared on both sides must satisfy BOTH
        // schemas — draft 2020-12 applies `$ref` siblings as an intersection,
        // not as a JSON Merge Patch where the closer declaration wins. Only
        // intersections this compiler can represent by reusing one of the two
        // operand schemas are merged; anything else is an explicit error
        // rather than a silent widening to whichever side was picked.
        let target_props = target.get("properties").and_then(Value::as_object);
        let mut merged_props: Vec<(&'a String, &'a Value)> = Vec::new();
        let mut seen_keys: HashSet<&str> = HashSet::new();
        if let Some(m) = schema.get("properties").and_then(Value::as_object) {
            for (k, v) in m {
                let effective = match target_props.and_then(|t| t.get(k.as_str())) {
                    Some(t) => intersect_property_schemas(k, v, t)?,
                    None => v,
                };
                seen_keys.insert(k.as_str());
                merged_props.push((k, effective));
            }
        }
        if let Some(m) = target_props {
            for (k, v) in m {
                if seen_keys.insert(k.as_str()) {
                    merged_props.push((k, v));
                }
            }
        }
        if merged_props.len() > MAX_OBJECT_PROPERTIES {
            return Err(SchemaError(format!(
                "object property count ({}) exceeds the supported limit ({MAX_OBJECT_PROPERTIES})",
                merged_props.len()
            )));
        }
        let properties = if merged_props.is_empty() {
            None
        } else {
            Some(merged_props)
        };

        // Merge `required`: union, deduplicated, sibling first.
        let mut required: Vec<&str> = Vec::new();
        let mut seen_req: HashSet<&str> = HashSet::new();
        for arr in [
            schema.get("required").and_then(Value::as_array),
            target.get("required").and_then(Value::as_array),
        ]
        .into_iter()
        .flatten()
        {
            if arr.len() > MAX_OBJECT_PROPERTIES {
                return Err(SchemaError(format!(
                    "object required count ({}) exceeds the supported limit ({MAX_OBJECT_PROPERTIES})",
                    arr.len()
                )));
            }
            for v in arr {
                if let Some(s) = v.as_str()
                    && seen_req.insert(s)
                {
                    required.push(s);
                    if required.len() > MAX_OBJECT_PROPERTIES {
                        return Err(SchemaError(format!(
                            "object required count ({}) exceeds the supported limit ({MAX_OBJECT_PROPERTIES})",
                            required.len()
                        )));
                    }
                }
            }
        }

        self.compile_object_fields(properties, required)
    }

    /// Resolve `ref_str` through any chain of pure `$ref` indirections to the
    /// definition that actually carries the schema, for the sibling merge in
    /// [`Self::merge_ref_with_siblings`].
    ///
    /// Stopping at the first link instead would look at `{"$ref": ...}`, find
    /// no `"type": "object"`, and fall through to the plain `compile_ref` —
    /// which resolves the same chain but drops the caller's `required` /
    /// `properties` siblings on the way, so the compiled grammar would accept
    /// documents the sibling forbids.
    ///
    /// A chain link that carries any keyword of its own beyond `$ref` and the
    /// annotation/document-structure keys is an explicit unsupported-schema
    /// error: merging the caller's siblings onto the resolved target would
    /// discard that link's own assertions, which is the same silent widening
    /// this resolution exists to prevent. The visited set bounds the walk, so a
    /// cyclic chain reports the cycle instead of looping.
    fn resolve_ref_chain_target(&self, ref_str: &str) -> Result<&'a Value, SchemaError> {
        let mut current = ref_str;
        let mut visited: HashSet<&str> = HashSet::new();
        loop {
            let name = extract_ref_name(current)?;
            let target: &'a Value = self
                .defs
                .get(name)
                .copied()
                .ok_or_else(|| SchemaError(format!("$ref not found: {current}")))?;
            let Some(next) = target.get("$ref").and_then(Value::as_str) else {
                return Ok(target);
            };
            if !visited.insert(name) {
                return Err(SchemaError(format!("$ref chain cycles through {current}")));
            }
            if let Some(m) = target.as_object() {
                for k in m.keys() {
                    if !REF_IGNORED_SIBLING_KEYS.contains(&k.as_str()) {
                        return Err(SchemaError(format!(
                            "unsupported schema: $ref chain link {current} carries keyword `{k}` alongside `$ref`, which cannot be merged with the referring schema's siblings"
                        )));
                    }
                }
            }
            current = next;
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
        self.builder.set_alts(id, alts)?;
        Ok(vec![vec![Symbol::NonTerminal(id)]])
    }

    /// Compute the string-language contribution of a `$ref`-bearing `anyOf`
    /// branch as the resolved terminal's string class INTERSECTED with every
    /// `const`/`enum` narrowing along the `$ref` chain (issue #473).
    ///
    /// The general `$ref` path cannot represent
    /// `{"$ref":<string def>,"enum":["a"]}` and fails closed rather than
    /// compiling the broad target. This specialized `anyOf` path can represent
    /// that conjunction: it walks `sub` and each intermediate `$defs` node,
    /// collects the string set any `const`/`enum` permits, classifies the
    /// terminal, and returns the intersection.
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
    /// the branch instead of rejecting a case it can resolve exactly. The
    /// string-forcing check spans the WHOLE chain, not just the outer `sub`, so
    /// an intermediate `$defs` node such as
    /// `{"$ref":<integer def>,"type":"string"}` is caught (issue #473). This
    /// lookup never touches `self.depth` and never registers a rule;
    /// `compile_ref` / `compile_schema` remain the authoritative recursion guard
    /// (issue #343) for the `other_subs` path.
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

        self.compile_object_fields(properties, required)
    }

    /// Compile an object schema from already-extracted `properties`/`required`,
    /// shared by [`Self::compile_object`] and the `$ref`+sibling merge in
    /// [`Self::compile_ref_with_siblings`] (issue #1078), which must combine a
    /// `$ref` target's fields with a sibling's before compiling — the two
    /// sources live in different `&'a Value` documents, so the merge happens on
    /// these already-borrowed collections rather than on a synthesized `Value`.
    fn compile_object_fields(
        &mut self,
        properties: Option<Vec<(&'a String, &'a Value)>>,
        required: Vec<&'a str>,
    ) -> Result<Vec<Alt>, SchemaError> {
        #[expect(
            clippy::unwrap_used,
            reason = "the ws rule is reserved when the builder is constructed"
        )]
        let ws_id = self.builder.rule_id("ws").unwrap();
        match properties {
            None if required.is_empty() => {
                // No properties: empty object `{}`, tolerating interior ws (`{ }`).
                Ok(vec![empty_object_as_alt_with_ws(ws_id)])
            }
            properties => {
                let mut props: Vec<(&'a str, Option<&'a Value>)> = properties
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(key, schema)| (key.as_str(), Some(schema)))
                    .collect();
                let mut seen_keys: HashSet<&str> = props.iter().map(|(key, _)| *key).collect();
                for name in required.iter().copied() {
                    if seen_keys.insert(name) {
                        if props.len() >= MAX_OBJECT_PROPERTIES {
                            return Err(SchemaError(format!(
                                "object effective property count ({}) exceeds the supported limit ({MAX_OBJECT_PROPERTIES})",
                                props.len() + 1
                            )));
                        }
                        props.push((name, None));
                    }
                }

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
                for (key_str, val_schema) in &props {
                    let key_str = *key_str;
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
                        let val_alts = match val_schema {
                            Some(schema) => self.compile_schema(schema, &[])?,
                            None => self.any_value_alts(),
                        };
                        self.builder.set_alts(id, val_alts)?;
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
                        self.builder.set_alts(id, vec![pair_alt])?;
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
                let tail_id = self.build_object_tail(obj_idx, &props_info, 0, false, ws_id)?;

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
    ) -> Result<usize, SchemaError> {
        let started_char = if started { 's' } else { 'f' };
        let rule_name = format!("obj_{obj_idx}_tail_{i}_{started_char}");

        // Memoization: if this (obj_idx, i, started) variant was already built,
        // return its id without rebuilding.
        if let Some(id) = self.builder.rule_id(&rule_name) {
            return Ok(id);
        }

        // Base case: all properties processed → epsilon.
        if i == props.len() {
            let id = self.builder.reserve(&rule_name);
            self.builder.set_alts(id, vec![vec![]])?; // epsilon alternative
            return Ok(id);
        }

        // Reserve first so that any recursive call that would reference *this*
        // rule (none in practice — the recursion always increments i — but
        // reserve-before-set is the established pattern in this file).
        let id = self.builder.reserve(&rule_name);

        let (pair_id, val_id, ref key_bytes, is_required) = props[i];

        let alts: Vec<Alt> = if is_required {
            // Required: one alternative (no choice).
            let next_id = self.build_object_tail(obj_idx, props, i + 1, true, ws_id)?;
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
            let next_started_id = self.build_object_tail(obj_idx, props, i + 1, true, ws_id)?;
            let next_same_id = self.build_object_tail(obj_idx, props, i + 1, started, ws_id)?;

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

        self.builder.set_alts(id, alts)?;
        Ok(id)
    }

    /// Compile an `array` schema.
    fn compile_array(&mut self, schema: &'a Value) -> Result<Vec<Alt>, SchemaError> {
        #[expect(
            clippy::unwrap_used,
            reason = "the ws rule is reserved when the builder is constructed"
        )]
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
                    self.builder.set_alts(pos_id, pos_alts)?;
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
                    self.builder.set_alts(item_id, item_alts)?;

                    // depth=1: leading-comma continuation after the fixed prefix.
                    let slack = max_items.map(|m| m - p);
                    let tail_id = self.build_bounded_tail(n, 1, slack, item_id, ws_id)?;
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
            self.builder.set_alts(item_id, item_alts)?;

            let alt = self.build_cardinality_array_alt(n, item_id, ws_id, min_items, max_items)?;
            return Ok(vec![alt]);
        }

        // No items schema and no prefixItems: any array `[]` or `[v, ...]`.
        let any_val_rule = "any_json_value";
        let any_id = if let Some(id) = self.builder.rule_id(any_val_rule) {
            id
        } else {
            let id = self.builder.reserve(any_val_rule);
            let any_alts = self.any_value_alts();
            self.builder.set_alts(id, any_alts)?;
            id
        };

        // issue #310 #2: honor cardinality even with no `items` schema.
        if min_items > 0 || max_items.is_some() {
            let n = self.array_counter;
            self.array_counter += 1;
            let alt = self.build_cardinality_array_alt(n, any_id, ws_id, min_items, max_items)?;
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
            self.builder.set_alts(id, tail_alts)?;
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
            )?;
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
    ) -> Result<Alt, SchemaError> {
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
                    let body_id = self.build_bounded_tail(n, 0, slack, item_id, ws_id)?;
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
                    let tail_id = self.build_bounded_tail(n, 1, slack, item_id, ws_id)?;
                    alt.push(Symbol::NonTerminal(tail_id));
                }
            }
        }

        alt.push(Symbol::Terminal(b']'));
        Ok(alt)
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
    ) -> Result<usize, SchemaError> {
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
                        .set_alts(cont_id, vec![cont_alt, vec![] /* epsilon */])?;
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
                    .set_alts(id, vec![body_alt, vec![] /* epsilon */])?;
                Ok(id)
            }
            Some(0) => {
                // No slack: epsilon rule (no additional items allowed).
                let rule_name = format!("arr_{arr_n}_opt_{depth}_0");
                let id = self.builder.reserve(&rule_name);
                self.builder.set_alts(id, vec![vec![] /* epsilon */])?;
                Ok(id)
            }
            Some(k) => {
                // Bounded tail: nested-optional chain of depth k.
                // opt_{depth}_{k} = (leading_comma? item ws opt_{depth+1}_{k-1}) | ε
                let rule_name = format!("arr_{arr_n}_opt_{depth}_{k}");
                let id = self.builder.reserve(&rule_name);
                // Recursively build the inner optional (one fewer slot).
                let inner_id =
                    self.build_bounded_tail(arr_n, depth + 1, Some(k - 1), item_id, ws_id)?;
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
                    .set_alts(id, vec![body_alt, vec![] /* epsilon */])?;
                Ok(id)
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
mod tests;
