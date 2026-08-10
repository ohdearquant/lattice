//! Measurement harness for issue #1342 (GrammarEngine::new doc-comment
//! timing figures).
//!
//! Pure-library measurement: no Metal dependency, no generation. It measures
//! two independently-costed phases of grammar-constrained decoding setup, at
//! two schema depths, over a 248,320-token vocabulary (Qwen3's real
//! tokenizer when available at `GRAMTIME_TOKENIZER_JSON`, else a synthetic
//! stand-in — see `real_or_synthetic` in the printed output):
//!
//!   1. `GrammarEngine::new`'s `VocabPartition::build` phase
//!      (`partition_build_ns`, `BuildProfile`) — the O(|states| x vocab_size
//!      x max_token_len) cost the doc comment describes.
//!   2. `ByteTrie::build` (trie.rs:99) — the deferred over-cap masking
//!      structure, O(vocab_size x max_token_len) with **no** `|states|`
//!      factor. It is called directly here (not via `GrammarEngine`'s
//!      `OnceLock`) because `ByteTrie::build` takes only `vocab_bytes` — no
//!      grammar or state — so timing it directly measures exactly the cost
//!      `GrammarEngine::mask_by_trie` (engine.rs:610) pays on first use,
//!      without needing to drive a live decode into an over-cap state.
//!
//! `deep_schema` is the issue #734 profiling repro (also used by
//! `gramperf_profile.rs`): 4 levels of nested objects, 3 arrays, 6
//! string-enum fields. `simple_schema` was tuned against the real
//! tokenizer's own `probe_reachable_states` count (not assumed) to land in
//! the low tens of states — see its doc comment for why that ruled out
//! *any* `{"type":"object", ...}` schema, however small: against the real
//! 248,320-token vocabulary, every object-rooted schema tried here,
//! including `{"type":"object","properties":{}}`, landed at 400+ states
//! before `VocabPartition::build`'s internal cap even applies. That
//! appears to come from the object grammar's whitespace/structural
//! handling interacting with the vocabulary's sheer token diversity, not
//! from any schema content — a finding distinct from (and worth reporting
//! alongside) the `|states|` question this harness was written to settle.
//!
//! Output: `RESULT key=value ...` lines, one block per measurement,
//! consumed by hand for the REPORT.md this binary was written to produce.
//! Not a bench-gate harness (no Criterion, no `make bench-compare`).

use lattice_inference::grammar::engine::{last_build_profile, probe_reachable_states};
use lattice_inference::grammar::trie::ByteTrie;
use lattice_inference::grammar::vocab_partition::MAX_GRAMMAR_STATES;
use lattice_inference::grammar::{GrammarEngine, GrammarSpec};
use std::time::Instant;

/// Qwen3's tokenizer vocabulary size, per the doc comment under measurement.
/// Override with `GRAMTIME_VOCAB_SIZE` for a fast correctness smoke test;
/// the reported figures are only meaningful at the real default.
const DEFAULT_VOCAB_SIZE: usize = 248_320;
/// Repetitions per measured quantity, for spread reporting. Override with
/// `GRAMTIME_REPS`.
const DEFAULT_REPS: usize = 10;
/// Upper bound for the uncapped-ish state probe (`probe_reachable_states`);
/// bounded because probe cost scales ~linearly with states explored, and
/// this only needs to confirm truncation past `MAX_GRAMMAR_STATES` (256),
/// not find a precise uncapped count. Override with `GRAMTIME_PROBE_CAP`.
const DEFAULT_PROBE_CAP: usize = 512;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// A bare 3-member string enum, no object wrapper: `probe_reachable_states`
/// against the real Qwen3 vocabulary puts this at 13 states (checked, not
/// assumed — see module docs), comfortably in the "roughly 10-20" low-state
/// range this harness needs for a meaningful contrast with `deep_schema`.
///
/// Every `{"type":"object", ...}` variant tried while tuning this — down to
/// `{"type":"object","properties":{}}` with zero fields — measured 400+
/// states against the real vocabulary, indistinguishable from
/// `deep_schema`'s count. Wrapping this same enum in an object reproduces
/// that: `{"type":"object","properties":{"kind":{"enum":[...]}}}` also
/// lands at 400+. So the object *skeleton* (the leading `{`, or its
/// whitespace handling) is what drives the state count against a real
/// vocabulary, not the enum, not nesting depth, and not this schema's
/// content — which is why the only way to get a genuine low-state
/// comparison point here is to drop the object wrapper entirely.
fn simple_schema() -> serde_json::Value {
    serde_json::json!({"type": "string", "enum": ["yes", "no", "maybe"]})
}

/// Issue #734 profiling repro schema (also used by `gramperf_profile.rs`):
/// 4 levels of nested objects, 3 array fields, 6 string-enum fields (6
/// members each). Known from that issue to approach/exceed
/// `MAX_GRAMMAR_STATES`.
fn deep_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "level3": {
                                "type": "object",
                                "properties": {
                                    "level4": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "enum": ["active", "inactive", "pending", "archived", "deleted", "draft"]},
                                            "value": {"type": "integer"}
                                        },
                                        "required": ["status", "value"]
                                    }
                                },
                                "required": ["level4"]
                            },
                            "category": {"type": "string", "enum": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]}
                        },
                        "required": ["level3", "category"]
                    },
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["level2", "tags"]
            },
            "items": {"type": "array", "items": {"type": "integer"}},
            "flags": {"type": "array", "items": {"type": "boolean"}},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent", "critical", "none"]},
            "region": {"type": "string", "enum": ["us", "eu", "apac", "latam", "mea", "other"]},
            "mode": {"type": "string", "enum": ["sync", "async", "batch", "stream", "manual", "auto"]},
            "role": {"type": "string", "enum": ["admin", "user", "guest", "owner", "viewer", "editor"]}
        },
        "required": ["level1", "items", "flags", "priority", "region", "mode", "role"]
    })
}

/// Deterministic xorshift64 PRNG — reproducible across runs, no external
/// `rand` dependency (not in this crate's Cargo.toml).
struct Xorshift64(u64);

impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Build a synthetic vocabulary of `n` tokens with a length distribution
/// comparable to a real byte-level BPE vocab (e.g. Qwen3's):
///
/// - Tokens 0..256: the byte-level fallback alphabet (one token per byte
///   value 0x00..=0xFF). Every real byte-level BPE vocab includes this base
///   layer so any input is representable; without it, grammar-state
///   enumeration here could get stuck at the initial state for schemas
///   whose required first byte (e.g. `{`) has no token, which would make
///   the measurement unrepresentative rather than merely synthetic.
/// - A few dozen literal tokens for the JSON-Schema vocabulary used by
///   `simple_schema`/`deep_schema` (property names, enum members, JSON
///   punctuation) — real tokenizers give frequent whole words/punctuation
///   clusters dedicated tokens, so this keeps state enumeration from
///   degenerating into single-byte stepping for the exact strings these
///   schemas need.
/// - The remainder: pseudo-random multi-byte filler tokens, skewed short
///   (weights below) to match the short-token-heavy shape of real
///   byte-level BPE vocabularies, with a long tail out to 20 bytes.
fn synthetic_vocab(n: usize) -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = Vec::with_capacity(n);

    for b in 0u32..256 {
        v.push(vec![b as u8]);
    }

    const LITERALS: &[&str] = &[
        "{", "}", "[", "]", ":", ",", "\"", ": ", ", ", "\": ", "\",", "\"}", "{\"", "level1",
        "level2", "level3", "level4", "status", "value", "category", "tags", "items", "flags",
        "priority", "region", "mode", "role", "active", "inactive", "pending", "archived",
        "deleted", "draft", "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "low", "medium",
        "high", "urgent", "critical", "none", "us", "eu", "apac", "latam", "mea", "other", "sync",
        "async", "batch", "stream", "manual", "auto", "admin", "user", "guest", "owner", "viewer",
        "editor", "true", "false", "null",
    ];
    for lit in LITERALS {
        v.push(lit.as_bytes().to_vec());
    }

    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    while v.len() < n {
        let r = rng.next_u64();
        let bucket = r % 100;
        let len: usize = if bucket < 40 {
            1 + (r >> 8) as usize % 2 // 1-2 bytes: 40%
        } else if bucket < 70 {
            3 + (r >> 8) as usize % 2 // 3-4 bytes: 30%
        } else if bucket < 90 {
            5 + (r >> 8) as usize % 4 // 5-8 bytes: 20%
        } else {
            9 + (r >> 8) as usize % 12 // 9-20 bytes: 10%
        };
        // Printable ASCII (33..=126, 94 values: upper/lower letters, digits,
        // punctuation) rather than lowercase-only: the schemas below spell
        // their JSON keys and enum members in plain lowercase English words,
        // so an a-z-only filler alphabet would make accidental byte-prefix
        // collisions between filler tokens and schema literals ~3.6x more
        // likely than a real BPE vocab's far more diverse merge-token
        // content — enough to visibly inflate PDA state counts through the
        // engine's known single-stack common-prefix limitation (see
        // grammar/mod.rs "Stability and known limitations"), which would
        // corrupt the simple-vs-deep comparison this harness exists to run.
        let mut bytes = Vec::with_capacity(len);
        for _ in 0..len {
            let byte_r = rng.next_u64();
            bytes.push(33u8 + (byte_r % 94) as u8);
        }
        v.push(bytes);
    }
    v.truncate(n);
    v
}

fn stats(mut xs: Vec<u64>) -> (u64, u64, f64, u64) {
    xs.sort_unstable();
    let min = xs[0];
    let max = xs[xs.len() - 1];
    let mean = xs.iter().sum::<u64>() as f64 / xs.len() as f64;
    let median = xs[xs.len() / 2];
    (min, max, mean, median)
}

fn measure_schema(
    label: &str,
    schema: serde_json::Value,
    vocab: &[Vec<u8>],
    reps: usize,
    probe_cap: usize,
) {
    let spec = GrammarSpec::JsonSchema(schema);

    let probe_t0 = Instant::now();
    let probed_states =
        probe_reachable_states(&spec, vocab, probe_cap).expect("schema must compile");
    let probe_ns = probe_t0.elapsed().as_nanos() as u64;
    let probe_truncated = probed_states >= probe_cap;

    if std::env::var("GRAMTIME_PROBE_ONLY").is_ok() {
        println!(
            "RESULT kind=probe_only label={label} probed_states={probed_states} \
             probe_truncated={probe_truncated} probe_ns={probe_ns}"
        );
        return;
    }

    let mut partition_ns = Vec::with_capacity(reps);
    let mut bfs_ns = Vec::with_capacity(reps);
    // `BuildProfile::reachable_states` is `enumerate_grammar_states`'s raw
    // `visited.len()`, which can overshoot `MAX_GRAMMAR_STATES`: the BFS
    // only rechecks its while-condition between states, so expanding one
    // state can push several states past the cap in a single inner-loop
    // pass before the outer loop notices (engine.rs's own doc comment on
    // `BuildProfile::reachable_states` flags this — "capped at the
    // `max_states` argument" is the target, not a hard per-push limit).
    // `VocabPartition::build` (vocab_partition.rs:91) then reduces this to
    // `effective_states = num_states.min(MAX_GRAMMAR_STATES)` before doing
    // any work, so that — not the raw BFS count — is the true divisor for
    // ns-per-(state, token) pair.
    let mut reachable_raw = 0usize;
    let mut exceeds_budget = false;
    let harness_t0 = Instant::now();
    for rep in 0..reps {
        let engine =
            GrammarEngine::new(&spec, vocab.to_vec()).expect("schema must build an engine");
        let bp = last_build_profile();
        partition_ns.push(bp.partition_build_ns);
        bfs_ns.push(bp.bfs_ns);
        reachable_raw = bp.reachable_states;
        exceeds_budget = engine.exceeds_state_budget();
        // Per-rep line (not just the aggregate below) so a monotonic drift
        // (thermal throttling under this loop's own sustained load) can be
        // told apart from random spikes (external contention on a shared
        // machine) after the fact, instead of only seeing min/max/mean.
        println!(
            "RESULT kind=schema_rep label={label} rep={rep} \
             elapsed_since_harness_start_ns={} partition_build_ns={} bfs_ns={}",
            harness_t0.elapsed().as_nanos() as u64,
            bp.partition_build_ns,
            bp.bfs_ns,
        );
    }
    let effective_states = reachable_raw.min(MAX_GRAMMAR_STATES);

    let (p_min, p_max, p_mean, p_median) = stats(partition_ns);
    let (b_min, b_max, b_mean, b_median) = stats(bfs_ns);
    let ns_per_pair = p_mean / (effective_states as f64 * vocab.len() as f64);

    println!(
        "RESULT kind=schema label={label} probed_states={probed_states} \
         probe_truncated={probe_truncated} probe_ns={probe_ns} \
         reachable_states_raw={reachable_raw} effective_states={effective_states} \
         exceeds_state_budget={exceeds_budget} \
         vocab_size={} reps={reps} \
         partition_build_ns_min={p_min} partition_build_ns_max={p_max} \
         partition_build_ns_mean={p_mean:.1} partition_build_ns_median={p_median} \
         bfs_ns_min={b_min} bfs_ns_max={b_max} bfs_ns_mean={b_mean:.1} bfs_ns_median={b_median} \
         ns_per_state_token_pair={ns_per_pair:.6}",
        vocab.len(),
    );
}

/// Default location of a real Qwen3.5 tokenizer, if present on this
/// machine. Override with `GRAMTIME_TOKENIZER_JSON`; set it to an empty
/// string to force the synthetic fallback.
fn default_tokenizer_path() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(std::path::PathBuf::from(home).join(".lattice/models/qwen3.5-0.8b/tokenizer.json"))
}

fn load_vocab(vocab_size: usize) -> (Vec<Vec<u8>>, &'static str, String) {
    let path = match std::env::var("GRAMTIME_TOKENIZER_JSON") {
        Ok(p) if p.is_empty() => None,
        Ok(p) => Some(std::path::PathBuf::from(p)),
        Err(_) => default_tokenizer_path(),
    };
    if let Some(path) = path
        && path.is_file()
    {
        match lattice_inference::BpeTokenizer::from_tokenizer_json(&path) {
            Ok(tok) => match tok.vocab_bytes(vocab_size) {
                Ok(vocab) => {
                    return (vocab, "real", path.display().to_string());
                }
                Err(e) => {
                    eprintln!(
                        "[gramtime] real tokenizer at {} could not fill vocab_size={vocab_size}: {e} — falling back to synthetic",
                        path.display()
                    );
                }
            },
            Err(e) => {
                eprintln!(
                    "[gramtime] failed to load tokenizer at {}: {e} — falling back to synthetic",
                    path.display()
                );
            }
        }
    } else {
        eprintln!("[gramtime] no real tokenizer found — falling back to synthetic");
    }
    (synthetic_vocab(vocab_size), "synthetic", String::new())
}

fn main() {
    let vocab_size = env_usize("GRAMTIME_VOCAB_SIZE", DEFAULT_VOCAB_SIZE);
    let reps = env_usize("GRAMTIME_REPS", DEFAULT_REPS);
    let probe_cap = env_usize("GRAMTIME_PROBE_CAP", DEFAULT_PROBE_CAP);

    eprintln!("[gramtime] loading vocab: {vocab_size} tokens");
    let (vocab, real_or_synthetic, source) = load_vocab(vocab_size);
    println!(
        "RESULT kind=vocab vocab_size={} real_or_synthetic={real_or_synthetic} source={source:?}",
        vocab.len()
    );

    measure_schema("simple", simple_schema(), &vocab, reps, probe_cap);
    measure_schema("deep", deep_schema(), &vocab, reps, probe_cap);

    // ByteTrie::build depends only on vocab_bytes (trie.rs:99) — no grammar,
    // no state — so timing it directly, independent of any schema, measures
    // exactly what GrammarEngine::mask_by_trie's OnceLock builds on first
    // use for any over-cap engine.
    let mut trie_ns = Vec::with_capacity(reps);
    for rep in 0..reps {
        let t0 = Instant::now();
        let trie = ByteTrie::build(&vocab);
        let ns = t0.elapsed().as_nanos() as u64;
        trie_ns.push(ns);
        std::hint::black_box(&trie);
        println!("RESULT kind=trie_build_rep rep={rep} trie_build_ns={ns}");
    }
    let (t_min, t_max, t_mean, t_median) = stats(trie_ns);
    println!(
        "RESULT kind=trie_build vocab_size={} reps={reps} \
         trie_build_ns_min={t_min} trie_build_ns_max={t_max} \
         trie_build_ns_mean={t_mean:.1} trie_build_ns_median={t_median}",
        vocab.len(),
    );
}
