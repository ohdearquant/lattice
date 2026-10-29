//! Structured-output v0: strict JSON Schema admission, the reference validator and the
//! compiled-grammar cache (design note: strict JSON Schema, non-streaming only).

use super::{ChatReq, RequestError};
use lattice_inference::grammar::{GrammarEngine, GrammarSpec};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

/// Keywords the v0 structured-output schema subset does not admit.
/// Checked before the narrower per-type validation in
/// `admit_v0_schema_at` so every rejection names the exact offending
/// keyword instead of falling through to a generic message.
/// `enum`/`const` are deferred per the design note's sign-off ruling
/// (Q2): admitted only once an independent evaluator and fuzz corpus
/// prove their type/escaping semantics, which this patch does not add.
const V0_REJECTED_KEYWORDS: &[&str] = &[
    "enum",
    "const",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "$ref",
    "$defs",
    "definitions",
    "anyOf",
    "oneOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "format",
    "prefixItems",
    "minItems",
    "maxItems",
    "uniqueItems",
    "contains",
];

/// A successful admission: the caller's schema, admitted into the v0
/// subset (closed objects, uniform arrays, primitive types, arbitrary
/// nesting of the above -- see the structured-output v0 design note).
/// Rejection is a `RequestError` with code `unsupported_strict_schema`,
/// naming the offending keyword/structural problem, and happens before
/// any worker job is submitted.
pub(super) fn admit_v0_schema(schema: &Value) -> Result<(), String> {
    admit_v0_schema_at(schema, "schema")
}

fn admit_v0_schema_at(node: &Value, path: &str) -> Result<(), String> {
    let Some(obj) = node.as_object() else {
        return Err(format!("{path}: schema must be a JSON object"));
    };

    for kw in V0_REJECTED_KEYWORDS {
        if obj.contains_key(*kw) {
            return Err(format!(
                "{path}: keyword '{kw}' is not supported by the v0 structured-output schema subset"
            ));
        }
    }

    let Some(type_value) = obj.get("type") else {
        return Err(format!("{path}: 'type' is required"));
    };
    let Some(ty) = type_value.as_str() else {
        return Err(format!(
            "{path}: 'type' must be a single string, not a union or other value"
        ));
    };

    match ty {
        "object" => {
            const ALLOWED: &[&str] = &["type", "properties", "required", "additionalProperties"];
            if let Some(unknown) = obj.keys().find(|k| !ALLOWED.contains(&k.as_str())) {
                return Err(format!(
                    "{path}: unknown keyword '{unknown}' is not supported for an object schema"
                ));
            }
            if !matches!(obj.get("additionalProperties"), Some(Value::Bool(false))) {
                return Err(format!(
                    "{path}: 'additionalProperties' must be present and set to exactly 'false'"
                ));
            }
            let Some(props) = obj.get("properties").and_then(Value::as_object) else {
                return Err(format!(
                    "{path}: 'properties' is required and must be an object"
                ));
            };
            let Some(required) = obj.get("required").and_then(Value::as_array) else {
                return Err(format!(
                    "{path}: 'required' is required and must be an array"
                ));
            };
            let mut required_names = std::collections::BTreeSet::new();
            for entry in required {
                let Some(name) = entry.as_str() else {
                    return Err(format!("{path}: 'required' entries must be strings"));
                };
                required_names.insert(name);
            }
            let prop_names: std::collections::BTreeSet<&str> =
                props.keys().map(String::as_str).collect();
            if required_names != prop_names {
                return Err(format!(
                    "{path}: 'required' must list every property exactly once (closed strict object)"
                ));
            }
            for (name, prop_schema) in props {
                admit_v0_schema_at(prop_schema, &format!("{path}.{name}"))?;
            }
            Ok(())
        }
        "array" => {
            const ALLOWED: &[&str] = &["type", "items"];
            if let Some(unknown) = obj.keys().find(|k| !ALLOWED.contains(&k.as_str())) {
                return Err(format!(
                    "{path}: unknown keyword '{unknown}' is not supported for an array schema"
                ));
            }
            let Some(items) = obj.get("items") else {
                return Err(format!("{path}: 'items' is required for an array schema"));
            };
            admit_v0_schema_at(items, &format!("{path}[]"))
        }
        "string" | "number" | "integer" | "boolean" | "null" => {
            const ALLOWED: &[&str] = &["type"];
            if let Some(unknown) = obj.keys().find(|k| !ALLOWED.contains(&k.as_str())) {
                return Err(format!(
                    "{path}: unknown keyword '{unknown}' is not supported for type '{ty}'"
                ));
            }
            Ok(())
        }
        other => Err(format!("{path}: unsupported type '{other}'")),
    }
}

/// Independent post-generation conformance check (structured-output v0
/// design note's "independent validation" requirement): validates the
/// raw generated JSON text against an admitted v0 schema by walking
/// `content`'s bytes directly, in lockstep with the schema, WITHOUT
/// reusing any of `lattice_inference::grammar::json_schema`'s compiler
/// code AND without ever going through `serde_json::Value` -- so a bug
/// shared between admission-time compilation and this check cannot
/// silently agree on a wrong answer.
///
/// A correctness note on `serde_json::Value`: (without the
/// `arbitrary_precision` feature, which the workspace does not enable)
/// parses any JSON number into `u64`/`i64`/`f64` before this function
/// would ever see it, so a grammar-conforming integer literal outside
/// `u64`/`i64` range (e.g. `18446744073709551616`) got silently rounded
/// to the nearest representable `f64`, and `is_i64()`/`is_u64()` then
/// rejected it -- a false `validation_failed` 500 for output the
/// compiler itself admitted. Preferred this raw-text walker over
/// turning on `arbitrary_precision`: that feature changes every
/// `Value` built anywhere in this binary, not just here, and isn't
/// already enabled anywhere in the workspace.
///
/// Assumes `schema` already passed `admit_v0_schema` -- only the v0
/// subset's shapes are handled.
pub(super) fn v0_validate_json(content: &str, schema: &Value) -> bool {
    let mut pos = 0usize;
    if !v0_validate_at(content, &mut pos, schema) {
        return false;
    }
    v0_skip_ws(content, &mut pos);
    pos == content.len()
}

fn v0_skip_ws(s: &str, pos: &mut usize) {
    let b = s.as_bytes();
    while matches!(b.get(*pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        *pos += 1;
    }
}

fn v0_validate_at(s: &str, pos: &mut usize, schema: &Value) -> bool {
    v0_skip_ws(s, pos);
    let Some(obj) = schema.as_object() else {
        return false;
    };
    let Some(ty) = obj.get("type").and_then(Value::as_str) else {
        return false;
    };
    match ty {
        "object" => v0_validate_object(s, pos, obj),
        "array" => v0_validate_array(s, pos, obj),
        "string" => v0_parse_string(s, pos).is_some(),
        "boolean" => v0_parse_literal(s, pos, "true") || v0_parse_literal(s, pos, "false"),
        "null" => v0_parse_literal(s, pos, "null"),
        "number" => v0_parse_number_lexeme(s, pos).is_some(),
        "integer" => {
            v0_parse_number_lexeme(s, pos).is_some_and(|lexeme| v0_is_integer_lexeme(&lexeme))
        }
        _ => false,
    }
}

fn v0_parse_literal(s: &str, pos: &mut usize, lit: &str) -> bool {
    if s.as_bytes().get(*pos..*pos + lit.len()) == Some(lit.as_bytes()) {
        *pos += lit.len();
        true
    } else {
        false
    }
}

/// Parses a JSON string starting at `s[*pos]` (which must be `"`),
/// returning its unescaped content and advancing `*pos` past the
/// closing quote.
fn v0_parse_string(s: &str, pos: &mut usize) -> Option<String> {
    let b = s.as_bytes();
    if b.get(*pos) != Some(&b'"') {
        return None;
    }
    *pos += 1;
    let mut out = String::new();
    loop {
        match b.get(*pos)? {
            b'"' => {
                *pos += 1;
                return Some(out);
            }
            b'\\' => {
                *pos += 1;
                match *b.get(*pos)? {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hex = s.get(*pos + 1..*pos + 5)?;
                        let code = u32::from_str_radix(hex, 16).ok()?;
                        *pos += 5; // consume 'u' + 4 hex digits
                        if (0xD800..=0xDBFF).contains(&code) {
                            // High surrogate: the compiled grammar
                            // (json_schema.rs escape_id) admits any
                            // four-hex `\uXXXX` independently of
                            // pairing, so a lone half must still
                            // validate here (validator ⊇ grammar
                            // language). Combine with an immediately
                            // following low-surrogate escape when
                            // present; a lone half is represented as
                            // U+FFFD since `char`/`String` cannot hold
                            // an unpaired surrogate scalar.
                            if s.as_bytes().get(*pos) == Some(&b'\\')
                                && s.as_bytes().get(*pos + 1) == Some(&b'u')
                                && let Some(low) = s
                                    .get(*pos + 2..*pos + 6)
                                    .and_then(|h| u32::from_str_radix(h, 16).ok())
                                    .filter(|low| (0xDC00..=0xDFFF).contains(low))
                            {
                                let combined = 0x10000 + (code - 0xD800) * 0x400 + (low - 0xDC00);
                                out.push(char::from_u32(combined)?);
                                *pos += 6;
                                continue;
                            }
                            out.push('\u{FFFD}');
                            continue;
                        }
                        if (0xDC00..=0xDFFF).contains(&code) {
                            // Lone low surrogate: same grammar-parity
                            // rationale as the high-surrogate case
                            // above.
                            out.push('\u{FFFD}');
                            continue;
                        }
                        out.push(char::from_u32(code)?);
                        continue;
                    }
                    _ => return None,
                }
                *pos += 1;
            }
            _ => {
                // Not a quote or a backslash: copy one UTF-8 scalar
                // verbatim, O(1) since `s[*pos..]` is a valid-UTF-8
                // slice already and `chars().next()` decodes lazily.
                // RFC 8259 §7 requires U+0000-U+001F to be escaped;
                // the compiled grammar's string-body alternatives
                // (json_schema.rs, unescaped-ASCII range starts at
                // 0x20) reject them unescaped too, so an unescaped
                // raw control scalar here must fail the same way.
                let ch = s[*pos..].chars().next()?;
                if (ch as u32) < 0x20 {
                    return None;
                }
                out.push(ch);
                *pos += ch.len_utf8();
            }
        }
    }
}

/// Consumes one JSON number token starting at `s[*pos]` per the
/// standard JSON number grammar (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`)
/// and returns its exact source lexeme, unconverted -- the whole point
/// of this function existing instead of `serde_json::Value::as_f64`.
fn v0_parse_number_lexeme(s: &str, pos: &mut usize) -> Option<String> {
    let b = s.as_bytes();
    let start = *pos;
    if b.get(*pos) == Some(&b'-') {
        *pos += 1;
    }
    match b.get(*pos) {
        Some(b'0') => *pos += 1,
        Some(c) if c.is_ascii_digit() => {
            while matches!(b.get(*pos), Some(c) if c.is_ascii_digit()) {
                *pos += 1;
            }
        }
        _ => {
            *pos = start;
            return None;
        }
    }
    if b.get(*pos) == Some(&b'.') {
        let frac_start = *pos + 1;
        let mut probe = frac_start;
        while matches!(b.get(probe), Some(c) if c.is_ascii_digit()) {
            probe += 1;
        }
        if probe > frac_start {
            *pos = probe;
        }
    }
    if matches!(b.get(*pos), Some(b'e' | b'E')) {
        let mut probe = *pos + 1;
        if matches!(b.get(probe), Some(b'+' | b'-')) {
            probe += 1;
        }
        let exp_digits_start = probe;
        while matches!(b.get(probe), Some(c) if c.is_ascii_digit()) {
            probe += 1;
        }
        if probe > exp_digits_start {
            *pos = probe;
        }
    }
    Some(s[start..*pos].to_string())
}

/// JSON Schema's `integer` is any number with a zero fractional part
/// (2020-12 §6.1.1) -- `1.0` is a valid integer, `1.5` is not. The v0
/// compiler's own `json_integer` grammar production never emits a
/// fraction or exponent at all, so this is stricter than necessary for
/// grammar-conformant output and exists to independently reject a
/// non-conformant generation on its own terms, not the compiler's.
fn v0_is_integer_lexeme(lexeme: &str) -> bool {
    if lexeme.contains(['e', 'E']) {
        return false;
    }
    match lexeme.split_once('.') {
        None => true,
        Some((_, frac)) => !frac.is_empty() && frac.bytes().all(|c| c == b'0'),
    }
}

fn v0_validate_object(s: &str, pos: &mut usize, obj: &serde_json::Map<String, Value>) -> bool {
    let Some(props) = obj.get("properties").and_then(Value::as_object) else {
        return false;
    };
    if s.as_bytes().get(*pos) != Some(&b'{') {
        return false;
    }
    *pos += 1;
    v0_skip_ws(s, pos);
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    if s.as_bytes().get(*pos) == Some(&b'}') {
        *pos += 1;
        return props.is_empty();
    }
    loop {
        v0_skip_ws(s, pos);
        let Some(key) = v0_parse_string(s, pos) else {
            return false;
        };
        v0_skip_ws(s, pos);
        if s.as_bytes().get(*pos) != Some(&b':') {
            return false;
        }
        *pos += 1;
        let Some(prop_schema) = props.get(&key) else {
            return false;
        };
        if !seen.insert(key) {
            return false;
        }
        if !v0_validate_at(s, pos, prop_schema) {
            return false;
        }
        v0_skip_ws(s, pos);
        match s.as_bytes().get(*pos) {
            Some(b',') => *pos += 1,
            Some(b'}') => {
                *pos += 1;
                break;
            }
            _ => return false,
        }
    }
    seen.len() == props.len()
}

fn v0_validate_array(s: &str, pos: &mut usize, obj: &serde_json::Map<String, Value>) -> bool {
    let Some(items_schema) = obj.get("items") else {
        return false;
    };
    if s.as_bytes().get(*pos) != Some(&b'[') {
        return false;
    }
    *pos += 1;
    v0_skip_ws(s, pos);
    if s.as_bytes().get(*pos) == Some(&b']') {
        *pos += 1;
        return true;
    }
    loop {
        if !v0_validate_at(s, pos, items_schema) {
            return false;
        }
        v0_skip_ws(s, pos);
        match s.as_bytes().get(*pos) {
            Some(b',') => *pos += 1,
            Some(b']') => {
                *pos += 1;
                break;
            }
            _ => return false,
        }
    }
    true
}

/// Serializes `schema` (already admitted by `admit_v0_schema`) into a
/// key that is identical for two schemas differing only in object
/// property order. Deliberately does NOT rely on `serde_json::Value`'s
/// own `Map` ordering: whether that `Map` is `BTreeMap`-backed (sorted)
/// or insertion-order-backed depends on whether the `preserve_order`
/// Cargo feature is unified into the build anywhere in the dependency
/// graph -- verified empirically for the current lockfile (no crate
/// enables it, so `serde_json::to_string` already sorts keys), but that
/// is a whole-workspace build property that a future unrelated
/// dependency bump could silently flip. This walks the value itself and
/// sorts object keys explicitly, so the cache key's determinism does not
/// depend on that build-wide feature-unification detail.
pub(super) fn canonical_schema_key(schema: &Value) -> String {
    fn write_canonical(v: &Value, out: &mut String) {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(&n.to_string()),
            Value::String(s) => {
                out.push_str(&serde_json::to_string(s).unwrap_or_default());
            }
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_canonical(item, out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, k) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(k).unwrap_or_default());
                    out.push(':');
                    write_canonical(&map[*k], out);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    write_canonical(schema, &mut out);
    out
}

/// Cache key for [`GrammarCache::get_or_compile`].
///
/// The `GrammarEngine` a lookup resolves to was compiled from *two*
/// inputs -- the admitted schema (`canonical_schema_key`) and the
/// vocabulary handed to `GrammarEngine::new` -- but only the schema used
/// to be reflected in the key. Keyed on the schema alone, a cache cannot
/// tell two different vocabularies compiled against the same schema
/// apart: the second vocabulary's lookup would hit the first
/// vocabulary's entry and silently serve a grammar built for the wrong
/// tokenizer.
///
/// This key holds a clone of the `vocab_bytes` `Arc` itself, not just its
/// address: an address alone names an allocation only
/// for as long as that exact allocation is alive, and the whole point of
/// a *stale* cache entry is that its vocabulary may no longer be alive
/// anywhere else -- a later, unrelated `vocab_bytes` allocation (a
/// reloaded model, a swapped tokenizer) can then legitimately land at the
/// freed address and collide with the stale entry's key. Storing the
/// `Arc` inside the key -- used as a `HashMap` key via [`PartialEq`]/
/// [`Eq`]/[`Hash`] impls below keyed on [`Arc::ptr_eq`] and the pointer
/// address, never on vocabulary *contents* -- keeps that allocation
/// alive for as long as the entry (or in-flight compile slot) sits in
/// the cache, so the address it names cannot be freed and reused while
/// a stale entry could still alias it. Comparing/hashing by pointer
/// rather than by content keeps this the same O(1)-per-request cost as
/// before: no hashing of a (possibly ~250K-entry) vocabulary on every
/// admitted request.
#[derive(Clone)]
pub(super) struct GrammarCacheKey {
    vocab_bytes: Arc<Vec<Vec<u8>>>,
    schema_key: String,
}

impl PartialEq for GrammarCacheKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.vocab_bytes, &other.vocab_bytes) && self.schema_key == other.schema_key
    }
}

impl Eq for GrammarCacheKey {}

impl std::hash::Hash for GrammarCacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.vocab_bytes) as *const () as usize).hash(state);
        self.schema_key.hash(state);
    }
}

pub(super) fn grammar_cache_key(
    vocab_bytes: &Arc<Vec<Vec<u8>>>,
    schema: &Value,
) -> GrammarCacheKey {
    GrammarCacheKey {
        vocab_bytes: Arc::clone(vocab_bytes),
        schema_key: canonical_schema_key(schema),
    }
}

/// Stage 2 design note: shipped default capacity for the LRU compiled-
/// grammar cache.
pub(super) const GRAMMAR_CACHE_CAPACITY: usize = 32;

/// [`GrammarCache::get_or_compile`]'s error shape distinguishes a legitimate schema-compile
/// failure -- the caller's fault, an ordinary 400
/// `unsupported_strict_schema`, exactly the behavior before this change
/// -- from the cache's own internal machinery breaking (a poisoned
/// lock, a vanished compiling-slot entry, or the compile closure
/// panicking instead of returning `Err`), which is never the caller's
/// fault and must surface as a bounded 500 instead of a panic that
/// tears down the request or a condvar nobody ever notifies again.
#[derive(Debug, Clone)]
pub(super) enum CacheError {
    Compile(String),
    Internal(String),
}

/// The state shared by a single in-flight compile: waiters block on
/// `cv` until `result` is populated by the one thread that actually
/// calls the compile closure (structured-output v0 design note,
/// stage-2 single-flight requirement).
struct CompileSlot {
    result: Mutex<Option<Result<Arc<GrammarEngine>, CacheError>>>,
    cv: Condvar,
}

struct GrammarCacheInner {
    capacity: usize,
    /// Compiled, ready-to-serve entries. Recency order tracked in
    /// `order` (front = least-recently-used, back = most-recently-used);
    /// `ready` itself has no ordering, so eviction always consults
    /// `order`, never `ready`'s iteration order.
    ready: HashMap<GrammarCacheKey, Arc<GrammarEngine>>,
    order: VecDeque<GrammarCacheKey>,
    /// Keys currently being compiled by exactly one thread; every other
    /// concurrent request for the same key waits on the listed slot
    /// instead of starting a second, redundant compile.
    compiling: HashMap<GrammarCacheKey, Arc<CompileSlot>>,
}

/// Bounded LRU cache of compiled `GrammarEngine`s keyed by
/// [`grammar_cache_key`], with single-flight compilation (structured-
/// output v0 design note §"Schema-to-grammar compilation and cache
/// placement"). Only successfully admitted-and-compiled schemas are
/// cached -- a rejected or over-budget schema is never inserted, so a
/// later, corrected request for the same key compiles fresh rather than
/// replaying a cached failure.
pub(super) struct GrammarCache {
    inner: Mutex<GrammarCacheInner>,
    /// Count of compile closures actually executed (as opposed to
    /// served from cache or a single-flight wait). Test-only signal for
    /// asserting single-flight collapses N concurrent identical
    /// requests into exactly 1 compile; harmless to keep in production
    /// as a cheap counter.
    compile_count: std::sync::atomic::AtomicUsize,
}

impl GrammarCache {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(GrammarCacheInner {
                capacity,
                ready: HashMap::new(),
                order: VecDeque::new(),
                compiling: HashMap::new(),
            }),
            compile_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    #[cfg(test)]
    pub(super) fn compile_count(&self) -> usize {
        self.compile_count.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Returns the cached engine for `key`, compiling it via `compile`
    /// if absent. Concurrent callers with the same `key` share one
    /// compile: all but the first block on that first call's result.
    ///
    /// Never panics on a poisoned lock or a broken internal invariant:
    /// every lock is recovered via
    /// `unwrap_or_else(PoisonError::into_inner)` instead of `expect`,
    /// and `compile` runs under `catch_unwind` so a panicking compile
    /// still frees its slot and wakes every waiter with
    /// `CacheError::Internal` instead of leaving them blocked on a
    /// condvar nobody will ever notify again.
    pub(super) fn get_or_compile(
        &self,
        key: GrammarCacheKey,
        compile: impl FnOnce() -> Result<GrammarEngine, String> + std::panic::UnwindSafe,
    ) -> Result<Arc<GrammarEngine>, CacheError> {
        use std::sync::PoisonError;

        enum Action {
            Hit(Arc<GrammarEngine>),
            Wait(Arc<CompileSlot>),
            Compile,
        }
        let action = {
            let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(engine) = inner.ready.get(&key).cloned() {
                if let Some(pos) = inner.order.iter().position(|k| k == &key) {
                    inner.order.remove(pos);
                }
                inner.order.push_back(key.clone());
                Action::Hit(engine)
            } else if let Some(slot) = inner.compiling.get(&key).cloned() {
                Action::Wait(slot)
            } else {
                let slot = Arc::new(CompileSlot {
                    result: Mutex::new(None),
                    cv: Condvar::new(),
                });
                inner.compiling.insert(key.clone(), slot);
                Action::Compile
            }
        };
        match action {
            Action::Hit(engine) => Ok(engine),
            Action::Wait(slot) => {
                let guard = slot.result.lock().unwrap_or_else(PoisonError::into_inner);
                let guard = slot
                    .cv
                    .wait_while(guard, |r| r.is_none())
                    .unwrap_or_else(PoisonError::into_inner);
                guard.clone().unwrap_or_else(|| {
                    Err(CacheError::Internal(
                        "compile slot signaled with no result".to_string(),
                    ))
                })
            }
            Action::Compile => {
                self.compile_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let result = match std::panic::catch_unwind(compile) {
                    Ok(Ok(engine)) => Ok(Arc::new(engine)),
                    Ok(Err(message)) => Err(CacheError::Compile(message)),
                    Err(_) => Err(CacheError::Internal(
                        "grammar compilation panicked".to_string(),
                    )),
                };
                let slot = {
                    let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
                    let slot = inner.compiling.remove(&key);
                    if let Ok(engine) = &result {
                        inner.ready.insert(key.clone(), Arc::clone(engine));
                        inner.order.push_back(key.clone());
                        while inner.ready.len() > inner.capacity {
                            if let Some(lru_key) = inner.order.pop_front() {
                                inner.ready.remove(&lru_key);
                            } else {
                                break;
                            }
                        }
                    }
                    slot
                };
                // `slot` is only ever absent if this thread's own
                // `Action::Compile` insert (above) never happened --
                // structurally unreachable, but not worth an `expect`:
                // there is simply no waiter to notify in that case.
                if let Some(slot) = slot {
                    let mut slot_result =
                        slot.result.lock().unwrap_or_else(PoisonError::into_inner);
                    *slot_result = Some(result.clone());
                    slot.cv.notify_all();
                }
                result
            }
        }
    }
}

/// A structured (`response_format.json_schema`, `strict: true`) request,
/// admitted and compiled before any worker job is submitted.
pub(super) struct StructuredRequest {
    pub(super) schema: Value,
    pub(super) engine: Arc<GrammarEngine>,
}

/// Parses and admits `req.response_format` (structured-output v0 design
/// note, §"Accepted request profile"). Returns `Ok(None)` for an
/// ordinary text request, `Ok(Some(..))` for an admitted+compiled
/// strict structured request, or `Err` for any request-shape or schema
/// problem -- always before a worker job is submitted, matching the
/// design note's fail-closed admission contract.
pub(super) fn admit_structured_request(
    req: &ChatReq,
    stream: bool,
    vocab_bytes: &Arc<Vec<Vec<u8>>>,
    grammar_cache: &Arc<GrammarCache>,
) -> Result<Option<StructuredRequest>, RequestError> {
    let Some(fmt) = &req.response_format else {
        return Ok(None);
    };
    if fmt.r#type != "json_schema" {
        return Ok(None);
    }
    // Non-streaming only (design note §"Terminal and streaming
    // behavior"): an SSE consumer could observe syntactically
    // incomplete JSON at every intermediate token, so v0 rejects the
    // combination outright rather than hiding partial bytes.
    if stream {
        return Err(RequestError::bad_request(
            "response_format.json_schema requires a non-streaming request \
             (stream must be false or omitted)",
            "unsupported_feature",
        ));
    }
    let Some(json_schema) = &fmt.json_schema else {
        return Err(RequestError::bad_request(
            "response_format.json_schema is required when response_format.type is 'json_schema'",
            "invalid_request",
        ));
    };
    let name_ok = json_schema
        .name
        .as_deref()
        .is_some_and(|n| !n.trim().is_empty());
    if !name_ok {
        return Err(RequestError::bad_request(
            "response_format.json_schema.name must be a nonempty string",
            "invalid_request",
        ));
    }
    // Missing or false `strict` is rejected rather than silently
    // treated as best-effort JSON (design note §"Accepted request
    // profile", item 3).
    if json_schema.strict != Some(true) {
        return Err(RequestError::bad_request(
            "response_format.json_schema.strict must be true; \
             best-effort JSON mode is not supported",
            "unsupported_feature",
        ));
    }
    let Some(schema_raw) = &json_schema.schema else {
        return Err(RequestError::bad_request(
            "response_format.json_schema.schema is required",
            "invalid_request",
        ));
    };
    // Materialized here, not during the outer `ChatReq` deserialization:
    // `schema` is a `RawValue` span until this exact point, so a request
    // that fails the cheaper `name`/`strict`/`stream` checks above never
    // pays this parse.
    let schema: Value = serde_json::from_str(schema_raw.get()).map_err(|err| {
        RequestError::bad_request(
            format!("response_format.json_schema.schema is invalid: {err}"),
            "invalid_request",
        )
    })?;
    let schema = &schema;
    if let Err(message) = admit_v0_schema(schema) {
        return Err(RequestError::bad_request(
            message,
            "unsupported_strict_schema",
        ));
    }
    // Compile before enqueueing (design note §"Schema-to-grammar
    // compilation and cache placement"), via the bounded single-flight
    // cache (stage 2): a schema already compiled against this same
    // vocabulary by an earlier request is served without touching the
    // vocabulary clone or the compiler at all; a schema currently being
    // compiled by a concurrent request is waited on instead of
    // independently recompiled. The key names both the schema and the
    // vocabulary (#1557; see `grammar_cache_key`) so a differently
    // loaded vocabulary can never resolve to a grammar compiled for a
    // different one.
    let key = grammar_cache_key(vocab_bytes, schema);
    let schema_for_compile = schema.clone();
    let vocab_bytes = Arc::clone(vocab_bytes);
    let engine = grammar_cache
        .get_or_compile(key, move || {
            let spec = GrammarSpec::JsonSchema(schema_for_compile);
            let engine = GrammarEngine::new(&spec, (*vocab_bytes).clone())
                .map_err(|e| format!("schema failed to compile: {e}"))?;
            if engine.exceeds_state_budget() {
                return Err(
                    "schema exceeds the strict-output complexity budget (256 precomputed \
                     grammar states); v0 never falls back to the full-vocabulary simulation \
                     path"
                        .to_string(),
                );
            }
            Ok(engine)
        })
        .map_err(|err| match err {
            CacheError::Compile(message) => {
                RequestError::bad_request(message, "unsupported_strict_schema")
            }
            CacheError::Internal(message) => RequestError::server_error(message, "internal_error"),
        })?;
    Ok(Some(StructuredRequest {
        schema: schema.clone(),
        engine,
    }))
}
