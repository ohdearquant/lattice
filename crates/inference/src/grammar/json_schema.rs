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

// ---------------------------------------------------------------------------
// Built-in primitive rules
// ---------------------------------------------------------------------------

/// Register primitive grammar rules into the builder.
fn register_builtins(b: &mut GrammarBuilder) -> Result<(), SchemaError> {
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
enum RefStr {
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
/// language, so `compile_any_of` drops the branch exactly. A pure `$ref`, or
/// one whose keyword admits a non-string — a numeric `const`, an all-numeric
/// or mixed `enum`, or a `type` array that permits any non-string type
/// (`["string","null"]`, `["integer"]`, …) — returns `false`; the generic
/// compiler then either compiles the pure reference or rejects the narrowing
/// sibling it cannot represent.
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
fn value_matches_type(v: &Value, t: &str) -> bool {
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
fn schema_type_names(value: &Value) -> Result<Vec<&str>, SchemaError> {
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

fn unrepresentable_ref_narrowing(key: &str) -> SchemaError {
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
fn check_one_of_exclusivity(branches: &[Value]) -> Result<(), SchemaError> {
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
fn intersect_property_schemas<'a>(
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
