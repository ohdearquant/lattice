//! Measurement harness for issue #1342 (GrammarEngine::new doc-comment
//! timing figures).
//!
//! Pure-library measurement: no Metal dependency, no generation. It measures
//! two independently-costed phases of grammar-constrained decoding setup, at
//! two schema depths, over a 248,320-token vocabulary (Qwen3's real
//! tokenizer when available at `GRAMTIME_TOKENIZER_JSON`, else a synthetic
//! stand-in — see `real_or_synthetic` in the printed output):
//!
//!   1. `GrammarEngine::new`'s partition phase (`partition_build_ns`,
//!      `BuildProfile`) — the per-state trie walk, which scales with
//!      |states| and excludes the trie build itself.
//!   2. `ByteTrie::build` — the structure `new` builds once and shares
//!      between the partition walk and the over-cap masking path,
//!      O(vocab_size x max_token_len) with **no** `|states|` factor. It is
//!      called directly here because it takes only `vocab_bytes` — no
//!      grammar or state — so timing it directly measures exactly the cost
//!      `GrammarEngine::new` pays for it (`GrammarEngine::trie_build_ns`).
//!
//! With `bench-internals` and `GRAMTIME_MEMBERSHIP_PROFILE=1`, it also
//! reports raw BFS membership-scan intervals, separate empty-clock controls
//! and actual comparison counts. This mode requires an explicit real tokenizer
//! file, and its profiled traversal includes instrumentation overhead.
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
//! consumed by hand to write up the findings. Not a bench-gate harness
//! (no Criterion, no `make bench-compare`).

#[cfg(feature = "bench-internals")]
use lattice_inference::grammar::engine::{BfsDedupProfile, GrammarError, profile_reachable_states};
use lattice_inference::grammar::engine::{last_build_profile, probe_reachable_states};
use lattice_inference::grammar::trie::ByteTrie;
use lattice_inference::grammar::vocab_partition::MAX_GRAMMAR_STATES;
use lattice_inference::grammar::{GrammarEngine, GrammarSpec};
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::path::PathBuf;
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

fn membership_requested(
    value: Option<&OsStr>,
    available: bool,
    probe_only: bool,
) -> Result<bool, String> {
    let Some(value) = value else {
        return Ok(false);
    };
    if value != "1" {
        return Err("GRAMTIME_MEMBERSHIP_PROFILE must be absent or exactly 1".into());
    }
    if !available {
        return Err("GRAMTIME_MEMBERSHIP_PROFILE=1 requires bench-internals".into());
    }
    if probe_only {
        return Err("GRAMTIME_MEMBERSHIP_PROFILE=1 conflicts with GRAMTIME_PROBE_ONLY".into());
    }
    Ok(true)
}

fn profile_usize(value: Option<&OsStr>, name: &str, default: usize) -> Result<usize, String> {
    let parsed = match value {
        None => default,
        Some(value) => value
            .to_str()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| format!("{name} must be a positive integer"))?,
    };
    if parsed == 0 {
        return Err(format!("{name} must be a positive integer"));
    }
    Ok(parsed)
}

struct ProfileInput {
    vocab: Vec<Vec<u8>>,
    path: PathBuf,
    sha256: String,
}

fn load_profile_input(path: Option<&OsStr>, vocab_size: usize) -> Result<ProfileInput, String> {
    let path = path
        .filter(|path| !path.is_empty())
        .ok_or("diagnostic mode requires explicit GRAMTIME_TOKENIZER_JSON")?;
    let path = PathBuf::from(path)
        .canonicalize()
        .map_err(|error| format!("diagnostic tokenizer path: {error}"))?;
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("diagnostic tokenizer {}: {error}", path.display()))?;
    let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    let tokenizer = lattice_inference::BpeTokenizer::from_tokenizer_json_str(&text)
        .map_err(|error| format!("diagnostic tokenizer {}: {error}", path.display()))?;
    let vocab = tokenizer
        .vocab_bytes(vocab_size)
        .map_err(|error| format!("diagnostic vocabulary: {error}"))?;
    Ok(ProfileInput {
        vocab,
        path,
        sha256,
    })
}

fn membership_enabled_record(input: &ProfileInput, reps: usize, probe_cap: usize) -> String {
    format!(
        "GRAMTIME_MEMBERSHIP_PROFILE_ENABLED {}",
        serde_json::json!({
            "schema": 1,
            "feature": "bench-internals",
            "real_or_synthetic": "real",
            "tokenizer_path": input.path,
            "tokenizer_sha256": input.sha256,
            "vocab_size": input.vocab.len(),
            "reps": reps,
            "probe_cap": probe_cap,
            "schemas": ["simple", "deep"],
            "clock_overhead_subtracted": false,
        })
    )
}

fn membership_complete_record(reps: usize, completed: [usize; 2]) -> Result<String, String> {
    if reps == 0 || completed != [reps, reps] {
        return Err("membership profiling did not complete every schema repetition".into());
    }
    Ok(format!(
        "GRAMTIME_MEMBERSHIP_PROFILE_COMPLETE {}",
        serde_json::json!({
            "schema": 1,
            "schemas": ["simple", "deep"],
            "reps": reps,
            "completed_reps": completed,
            "clock_overhead_subtracted": false,
        })
    ))
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

#[cfg(feature = "bench-internals")]
fn measure_dedup(
    spec: &GrammarSpec,
    vocab: &[Vec<u8>],
    expected_states: usize,
) -> Result<(usize, BfsDedupProfile), GrammarError> {
    let (states, profile) = profile_reachable_states(spec, vocab, MAX_GRAMMAR_STATES)?;
    if states != expected_states {
        return Err(GrammarError(format!(
            "profiled BFS found {states} states, unprofiled BFS found {expected_states}"
        )));
    }
    if profile.membership_checks == 0 {
        return Err(GrammarError(
            "BFS dedup instrument did not exercise a membership scan".into(),
        ));
    }
    Ok((states, profile))
}

#[cfg(feature = "bench-internals")]
fn dedup_rep_record(
    label: &str,
    rep: usize,
    unprofiled_states: usize,
    profiled_states: usize,
    profile: BfsDedupProfile,
) -> String {
    format!(
        "RESULT kind=bfs_dedup_rep label={label} rep={rep} \
         unprofiled_states_raw={unprofiled_states} profiled_states_raw={profiled_states} \
         effective_states={} profiled_bfs_ns={} dedup_ns={} empty_interval_ns={} \
         membership_checks={} state_comparisons={} duplicate_states={}",
        profiled_states.min(MAX_GRAMMAR_STATES),
        profile.bfs_ns,
        profile.dedup_ns,
        profile.empty_interval_ns,
        profile.membership_checks,
        profile.state_comparisons,
        profile.duplicate_states,
    )
}

#[cfg(feature = "bench-internals")]
struct DedupMeasurements {
    profiled_bfs_ns: Vec<u64>,
    dedup_ns: Vec<u64>,
    empty_interval_ns: Vec<u64>,
}

#[cfg(feature = "bench-internals")]
impl DedupMeasurements {
    fn new(reps: usize) -> Self {
        Self {
            profiled_bfs_ns: Vec::with_capacity(reps),
            dedup_ns: Vec::with_capacity(reps),
            empty_interval_ns: Vec::with_capacity(reps),
        }
    }
}

fn measure_schema(
    label: &str,
    schema: serde_json::Value,
    vocab: &[Vec<u8>],
    reps: usize,
    probe_cap: usize,
    profile_membership: bool,
    probe_only: bool,
) -> Result<usize, String> {
    #[cfg(not(feature = "bench-internals"))]
    if profile_membership {
        return Err("membership profiling requires bench-internals".into());
    }
    if profile_membership && probe_only {
        return Err("membership profiling cannot use probe-only mode".into());
    }
    let spec = GrammarSpec::JsonSchema(schema);

    let probe_t0 = Instant::now();
    let probed_states =
        probe_reachable_states(&spec, vocab, probe_cap).expect("schema must compile");
    let probe_ns = probe_t0.elapsed().as_nanos() as u64;
    let probe_truncated = probed_states >= probe_cap;

    if probe_only {
        println!(
            "RESULT kind=probe_only label={label} probed_states={probed_states} \
             probe_truncated={probe_truncated} probe_ns={probe_ns}"
        );
        return Ok(0);
    }

    let mut partition_ns = Vec::with_capacity(reps);
    let mut bfs_ns = Vec::with_capacity(reps);
    #[cfg(feature = "bench-internals")]
    let mut dedup = profile_membership.then(|| DedupMeasurements::new(reps));
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
        #[cfg(feature = "bench-internals")]
        if let Some(measurements) = &mut dedup {
            let (states, profile) =
                measure_dedup(&spec, vocab, bp.reachable_states).map_err(|error| error.0)?;
            measurements.profiled_bfs_ns.push(profile.bfs_ns);
            measurements.dedup_ns.push(profile.dedup_ns);
            measurements
                .empty_interval_ns
                .push(profile.empty_interval_ns);
            println!(
                "{}",
                dedup_rep_record(label, rep, bp.reachable_states, states, profile)
            );
        }
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
    #[cfg(feature = "bench-internals")]
    if let Some(measurements) = dedup {
        let completed = measurements.profiled_bfs_ns.len();
        let (pb_min, pb_max, pb_mean, pb_median) = stats(measurements.profiled_bfs_ns);
        let (d_min, d_max, d_mean, d_median) = stats(measurements.dedup_ns);
        let (e_min, e_max, e_mean, e_median) = stats(measurements.empty_interval_ns);
        println!(
            "RESULT kind=bfs_dedup label={label} reachable_states_raw={reachable_raw} \
             effective_states={effective_states} vocab_size={} reps={reps} \
             profiled_bfs_ns_min={pb_min} profiled_bfs_ns_max={pb_max} \
             profiled_bfs_ns_mean={pb_mean:.1} profiled_bfs_ns_median={pb_median} \
             dedup_ns_min={d_min} dedup_ns_max={d_max} \
             dedup_ns_mean={d_mean:.1} dedup_ns_median={d_median} \
             empty_interval_ns_min={e_min} empty_interval_ns_max={e_max} \
             empty_interval_ns_mean={e_mean:.1} empty_interval_ns_median={e_median} \
             clock_overhead_subtracted=false",
            vocab.len(),
        );
        return Ok(completed);
    }
    Ok(0)
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

fn run() -> Result<(), String> {
    let profile_membership = membership_requested(
        std::env::var_os("GRAMTIME_MEMBERSHIP_PROFILE").as_deref(),
        cfg!(feature = "bench-internals"),
        std::env::var_os("GRAMTIME_PROBE_ONLY").is_some(),
    )?;
    let setting = |name, default| {
        if profile_membership {
            profile_usize(std::env::var_os(name).as_deref(), name, default)
        } else {
            Ok(env_usize(name, default))
        }
    };
    let vocab_size = setting("GRAMTIME_VOCAB_SIZE", DEFAULT_VOCAB_SIZE)?;
    let reps = setting("GRAMTIME_REPS", DEFAULT_REPS)?;
    let probe_cap = setting("GRAMTIME_PROBE_CAP", DEFAULT_PROBE_CAP)?;

    eprintln!("[gramtime] loading vocab: {vocab_size} tokens");
    let (vocab, real_or_synthetic, source, enabled_record) = if profile_membership {
        let input = load_profile_input(
            std::env::var_os("GRAMTIME_TOKENIZER_JSON").as_deref(),
            vocab_size,
        )?;
        let record = membership_enabled_record(&input, reps, probe_cap);
        (
            input.vocab,
            "real",
            input.path.display().to_string(),
            Some(record),
        )
    } else {
        let (vocab, kind, source) = load_vocab(vocab_size);
        (vocab, kind, source, None)
    };
    println!(
        "RESULT kind=vocab vocab_size={} real_or_synthetic={real_or_synthetic} source={source:?}",
        vocab.len()
    );
    if let Some(record) = enabled_record {
        println!("{record}");
    }

    let simple_completed = measure_schema(
        "simple",
        simple_schema(),
        &vocab,
        reps,
        probe_cap,
        profile_membership,
        std::env::var("GRAMTIME_PROBE_ONLY").is_ok(),
    )?;
    let deep_completed = measure_schema(
        "deep",
        deep_schema(),
        &vocab,
        reps,
        probe_cap,
        profile_membership,
        std::env::var("GRAMTIME_PROBE_ONLY").is_ok(),
    )?;

    // ByteTrie::build depends only on vocab_bytes — no grammar, no state — so
    // timing it directly, independent of any schema, measures exactly what
    // GrammarEngine::new builds once for any engine.
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
    if profile_membership {
        println!(
            "{}",
            membership_complete_record(reps, [simple_completed, deep_completed])?
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("[gramtime] {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER: &str = r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}}"#;
    const TOKENIZER_SHA256: &str =
        "aec3c8efa003f166de201b117c1cc0081b39bed7c7443fc1e5e9e26f099b42df";

    #[test]
    fn membership_request_selection_is_explicit() {
        for available in [false, true] {
            for probe_only in [false, true] {
                assert!(!membership_requested(None, available, probe_only).unwrap());
            }
        }
        assert!(membership_requested(Some(OsStr::new("1")), true, false).unwrap());
        assert_eq!(
            membership_requested(Some(OsStr::new("1")), false, false).unwrap_err(),
            "GRAMTIME_MEMBERSHIP_PROFILE=1 requires bench-internals"
        );
        assert_eq!(
            membership_requested(Some(OsStr::new("1")), true, true).unwrap_err(),
            "GRAMTIME_MEMBERSHIP_PROFILE=1 conflicts with GRAMTIME_PROBE_ONLY"
        );
        for value in ["", "0", "true", " 1", "1 ", "2"] {
            assert_eq!(
                membership_requested(Some(OsStr::new(value)), true, false).unwrap_err(),
                "GRAMTIME_MEMBERSHIP_PROFILE must be absent or exactly 1"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(membership_requested(Some(OsStr::from_bytes(&[0xff])), true, false).is_err());
        }
    }

    #[test]
    fn membership_settings_require_positive_integers() {
        assert_eq!(profile_usize(None, "SETTING", 7).unwrap(), 7);
        assert_eq!(
            profile_usize(Some(OsStr::new("3")), "SETTING", 7).unwrap(),
            3
        );
        for value in ["", "0", "-1", "3.0", " 3", "3 ", "184467440737095516160"] {
            assert_eq!(
                profile_usize(Some(OsStr::new(value)), "SETTING", 7).unwrap_err(),
                "SETTING must be a positive integer"
            );
        }
        assert!(profile_usize(None, "SETTING", 0).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(profile_usize(Some(OsStr::from_bytes(&[0xff])), "SETTING", 7).is_err());
        }
    }

    #[test]
    fn membership_input_refuses_missing_invalid_and_undersized_vocab() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        assert!(load_profile_input(None, 2).is_err());
        assert!(load_profile_input(Some(OsStr::new("")), 2).is_err());
        assert!(load_profile_input(Some(path.as_os_str()), 2).is_err());
        std::fs::write(&path, "not JSON").unwrap();
        assert!(load_profile_input(Some(path.as_os_str()), 2).is_err());
        std::fs::write(&path, TOKENIZER).unwrap();
        assert!(load_profile_input(Some(path.as_os_str()), 1).is_err());
        assert_eq!(
            load_profile_input(Some(path.as_os_str()), 2).unwrap().vocab,
            [b"a".to_vec(), b"b".to_vec()]
        );
    }

    #[test]
    fn membership_input_binding_uses_loaded_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        std::fs::write(&path, TOKENIZER).unwrap();
        let input = load_profile_input(Some(path.as_os_str()), 2).unwrap();
        assert_eq!(input.path, path.canonicalize().unwrap());
        assert_eq!(input.sha256, TOKENIZER_SHA256);
        assert_eq!(input.vocab, [b"a".to_vec(), b"b".to_vec()]);
        let record = membership_enabled_record(&input, 3, 512);
        let fields: serde_json::Value = serde_json::from_str(
            record
                .strip_prefix("GRAMTIME_MEMBERSHIP_PROFILE_ENABLED ")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            fields,
            serde_json::json!({
                "schema": 1,
                "feature": "bench-internals",
                "real_or_synthetic": "real",
                "tokenizer_path": input.path,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "vocab_size": 2,
                "reps": 3,
                "probe_cap": 512,
                "schemas": ["simple", "deep"],
                "clock_overhead_subtracted": false,
            })
        );
        std::fs::write(&path, format!("{TOKENIZER}\n")).unwrap();
        let changed = load_profile_input(Some(path.as_os_str()), 2).unwrap();
        assert_eq!(changed.vocab, input.vocab);
        assert_ne!(changed.sha256, input.sha256);
    }

    #[test]
    fn membership_completion_requires_both_schema_populations() {
        for (reps, completed) in [
            (0, [0, 0]),
            (2, [0, 0]),
            (2, [2, 1]),
            (2, [1, 2]),
            (2, [3, 2]),
        ] {
            assert_eq!(
                membership_complete_record(reps, completed).unwrap_err(),
                "membership profiling did not complete every schema repetition"
            );
        }
        let record = membership_complete_record(2, [2, 2]).unwrap();
        let fields: serde_json::Value = serde_json::from_str(
            record
                .strip_prefix("GRAMTIME_MEMBERSHIP_PROFILE_COMPLETE ")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            fields,
            serde_json::json!({
                "schema": 1,
                "schemas": ["simple", "deep"],
                "reps": 2,
                "completed_reps": [2, 2],
                "clock_overhead_subtracted": false,
            })
        );
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn dedup_report_keeps_raw_time_and_clock_control_separate() {
        let profile = BfsDedupProfile {
            bfs_ns: 31,
            dedup_ns: 7,
            empty_interval_ns: 9,
            membership_checks: 2,
            state_comparisons: 3,
            duplicate_states: 1,
        };
        assert_eq!(
            dedup_rep_record("deep", 1, 257, 257, profile),
            "RESULT kind=bfs_dedup_rep label=deep rep=1 unprofiled_states_raw=257 profiled_states_raw=257 effective_states=256 profiled_bfs_ns=31 dedup_ns=7 empty_interval_ns=9 membership_checks=2 state_comparisons=3 duplicate_states=1"
        );
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn dedup_measurement_requires_real_scan_and_matching_state_count() {
        let spec = GrammarSpec::Gbnf("root ::= \"a\"".into());
        let vocab = vec![b"a".to_vec(), b"a".to_vec()];
        let expected = probe_reachable_states(&spec, &vocab, MAX_GRAMMAR_STATES).unwrap();
        assert_eq!(expected, 2);
        let (actual, profile) = measure_dedup(&spec, &vocab, expected).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(profile.membership_checks, 2);
        assert_eq!(profile.state_comparisons, 3);
        assert_eq!(profile.duplicate_states, 1);
        assert_eq!(
            measure_dedup(&spec, &vocab, expected + 1).unwrap_err().0,
            "profiled BFS found 2 states, unprofiled BFS found 3"
        );
        assert_eq!(
            measure_dedup(&spec, &[b"z".to_vec()], 1).unwrap_err().0,
            "BFS dedup instrument did not exercise a membership scan"
        );
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn membership_schema_route_requires_actual_profile_work() {
        let vocab = synthetic_vocab(300);
        let simple =
            measure_schema("simple", simple_schema(), &vocab, 2, 512, true, false).unwrap();
        let deep = measure_schema("deep", deep_schema(), &vocab, 2, 512, true, false).unwrap();
        assert_eq!([simple, deep], [2, 2]);
        assert!(membership_complete_record(2, [simple, deep]).is_ok());
        assert_eq!(
            measure_schema("simple", simple_schema(), &vocab, 1, 512, false, false).unwrap(),
            0
        );
        assert_eq!(
            measure_schema("simple", serde_json::Value::Null, &[], 1, 512, true, true).unwrap_err(),
            "membership profiling cannot use probe-only mode"
        );
    }

    #[cfg(not(feature = "bench-internals"))]
    #[test]
    fn membership_unavailable_route_refuses_before_schema_work() {
        assert_eq!(
            measure_schema("simple", serde_json::Value::Null, &[], 1, 512, true, false)
                .unwrap_err(),
            "membership profiling requires bench-internals"
        );
    }
}
