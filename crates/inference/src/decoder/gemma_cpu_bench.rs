//! Manual real-checkpoint measurement of the production Gemma CPU session.
//!
//! Run the exact ignored test in release mode with one test thread, under
//! `scripts/bench-command.sh --durable`. The model loads once outside sample
//! timers. Setup includes cloning prompt IDs and constructing fresh session
//! cache/scratch; lazy allocations inside prefill remain charged to prefill.
//! Selection, evidence scans and destruction are outside phase timers.
//!
//! `LATTICE_GEMMA4_BENCH_CORPUS` optionally names JSON with the shape
//! `{"cases":[{"name":"short","input_ids":[2,818]},{"name":"long","input_ids":[...]}]}`.
//! IDs include any desired BOS/chat-template tokens; no text is re-tokenized.
//! Without it, the short case is the committed golden's IDs and the long case
//! cycles its non-BOS IDs to `LATTICE_GEMMA4_BENCH_LONG_TOKENS` (default 2512).
//! That default long prompt is synthetic, not the issue's real prompt.
//! `LATTICE_GEMMA4_BENCH_RUNS` is 1..=5 (default 3), after one discarded warmup
//! per case. `LATTICE_GEMMA4_MODEL_DIR` names the real checkpoint; absence or
//! invalid configuration fails. `LATTICE_GEMMA4_BENCH_REVISION`, if supplied,
//! is reported as caller-declared provenance, not a verified weight hash.
//!
//! Two real select/accept/decode transitions follow each prefill. This is a
//! fixed-count session diagnostic: EOS does not stop it and it does not time
//! the full driver's detokenization or output protocol. Fingerprints identify
//! exact outputs for repeat controls; they are not a cross-backend tolerance
//! or a substitute for checkpoint parity tests.

use super::*;
use crate::GenerateConfig;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const MAX_PROMPT_TOKENS: usize = 16_384;
const DECODE_STEPS: usize = 2;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    input_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Golden {
    input_ids: Vec<u32>,
}

#[derive(Debug, PartialEq)]
struct Evidence {
    logits_sha256: String,
    logits_sum: f64,
    logits_l1: f64,
    cache_sha256: String,
    cache_values: usize,
    evaluated_len: usize,
}

struct Sample {
    setup: Duration,
    prefill: Duration,
    decode: [Duration; DECODE_STEPS],
    selected: [u32; DECODE_STEPS],
    phases: Vec<Evidence>,
}

fn bounded_env(name: &str, default: usize, minimum: usize, maximum: usize) -> usize {
    let value = match std::env::var(name) {
        Ok(raw) => raw
            .parse::<usize>()
            .expect("benchmark bound must be an integer"),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("{name}: {error}"),
    };
    assert!((minimum..=maximum).contains(&value), "{name} out of bounds");
    value
}

fn model_dir() -> PathBuf {
    let raw = std::env::var("LATTICE_GEMMA4_MODEL_DIR")
        .unwrap_or_else(|_| "~/.lattice/models/gemma-4-e2b-it".to_string());
    if let Some(rest) = raw.strip_prefix("~/") {
        PathBuf::from(std::env::var("HOME").expect("HOME is required for ~/ model paths"))
            .join(rest)
    } else {
        PathBuf::from(raw)
    }
}

fn hash_file(path: &Path) -> String {
    format!(
        "{:x}",
        Sha256::digest(std::fs::read(path).expect("read benchmark input"))
    )
}

fn token_hash(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in ids {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn corpus() -> (Corpus, String) {
    if let Some(path) = std::env::var_os("LATTICE_GEMMA4_BENCH_CORPUS") {
        let path = PathBuf::from(path);
        let bytes = std::fs::read(&path).expect("read exact token-ID corpus");
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let corpus = serde_json::from_slice(&bytes).expect("strict token-ID corpus JSON");
        return (
            corpus,
            format!("external_token_ids sha256={hash} path={path:?}"),
        );
    }
    let golden: Golden = serde_json::from_str(include_str!(
        "../../tests/fixtures/gemma4/stage5/e2e_golden.json"
    ))
    .expect("committed Gemma golden JSON");
    assert!(golden.input_ids.len() > 1);
    let length = bounded_env(
        "LATTICE_GEMMA4_BENCH_LONG_TOKENS",
        2512,
        65,
        MAX_PROMPT_TOKENS,
    );
    let mut long = Vec::with_capacity(length);
    long.push(golden.input_ids[0]);
    long.extend(
        golden.input_ids[1..]
            .iter()
            .copied()
            .cycle()
            .take(length - 1),
    );
    (
        Corpus {
            cases: vec![
                Case {
                    name: "short_golden".into(),
                    input_ids: golden.input_ids,
                },
                Case {
                    name: "long_synthetic_repeated_tokens".into(),
                    input_ids: long,
                },
            ],
        },
        "default_golden_and_synthetic_repeated_tokens".into(),
    )
}

fn evidence(session: &GemmaCpuSession<'_>) -> Evidence {
    let logits = &session.scratch.logits[..session.model.config.vocab_size];
    assert!(!logits.is_empty());
    let mut logits_hash = Sha256::new();
    let mut logits_sum = 0.0_f64;
    let mut logits_l1 = 0.0_f64;
    for &value in logits {
        assert!(value.is_finite(), "non-finite benchmark logit");
        logits_hash.update(value.to_bits().to_le_bytes());
        logits_sum += f64::from(value);
        logits_l1 += f64::from(value).abs();
    }
    assert!(logits_sum.is_finite() && logits_l1.is_finite() && logits_l1 > 0.0);
    let mut cache_hash = Sha256::new();
    let mut cache_values = 0usize;
    for layer in 0..session.model.config.num_hidden_layers {
        let slot = session.cache.layer_slot(layer).expect("cache layer slot");
        let length = session.cache.seq_len(layer).expect("cache sequence length");
        assert!(length > 0, "prefill must publish every layer's donor");
        cache_hash.update((layer as u64).to_le_bytes());
        cache_hash.update((slot as u64).to_le_bytes());
        cache_hash.update((length as u64).to_le_bytes());
        for values in [
            session.cache.k_view(layer).expect("cache K view"),
            session.cache.v_view(layer).expect("cache V view"),
        ] {
            cache_hash.update((values.len() as u64).to_le_bytes());
            cache_values = cache_values
                .checked_add(values.len())
                .expect("cache census");
            for &value in values {
                assert!(value.is_finite(), "non-finite benchmark cache value");
                cache_hash.update(value.to_bits().to_le_bytes());
            }
        }
    }
    assert!(cache_values > 0);
    Evidence {
        logits_sha256: format!("{:x}", logits_hash.finalize()),
        logits_sum,
        logits_l1,
        cache_sha256: format!("{:x}", cache_hash.finalize()),
        cache_values,
        evaluated_len: session.position,
    }
}

fn sample(model: &Gemma4Model, ids: &[u32]) -> Sample {
    let maximum = ids
        .len()
        .checked_add(DECODE_STEPS)
        .expect("request capacity");
    let start = Instant::now();
    let mut session = GemmaCpuSession::new(model, ids.to_vec(), 0.0, Some(7), maximum)
        .expect("fresh Gemma CPU session");
    let setup = start.elapsed();
    let cancel = || false;
    let start = Instant::now();
    let stamp = session
        .prefill(&cancel)
        .expect("production Gemma CPU prefill");
    let prefill = start.elapsed();
    assert_eq!(stamp.evaluated_len, ids.len());
    assert_eq!(stamp.prediction, None);
    let mut phases = vec![evidence(&session)];
    let config = GenerateConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        max_new_tokens: DECODE_STEPS,
        ..Default::default()
    };
    let mut history = ids.to_vec();
    let mut selected = [0; DECODE_STEPS];
    let mut decode = [Duration::ZERO; DECODE_STEPS];
    for index in 0..DECODE_STEPS {
        let outcome = session
            .select(&SelectionRequest {
                config: &config,
                history: &history,
                grammar_mask: None,
            })
            .expect("greedy session selection");
        let SelectOutcome::Candidate(candidate) = outcome else {
            panic!("unconstrained selection must produce a candidate")
        };
        let accepted = AcceptedToken {
            final_id: candidate.candidate_id,
            prediction: candidate.prediction,
        };
        selected[index] = accepted.final_id;
        history.push(accepted.final_id);
        let start = Instant::now();
        let stamp = session
            .decode(&accepted, &cancel)
            .expect("production Gemma CPU decode");
        decode[index] = start.elapsed();
        assert_eq!(stamp.evaluated_len, ids.len() + index + 1);
        assert_eq!(stamp.prediction, None);
        assert!(!session.ledger.is_live(accepted.prediction));
        phases.push(evidence(&session));
    }
    session
        .finish(FinishDisposition::Reusable)
        .expect("finish benchmark session");
    Sample {
        setup,
        prefill,
        decode,
        selected,
        phases,
    }
}

#[test]
#[ignore = "manual release benchmark; requires a real Gemma checkpoint and reserved measurement window"]
fn bench_gemma_cpu_prefill_and_two_decode_steps() {
    // Ordinary debug test builds must compile this ignored measurement, while
    // an explicit invocation must refuse timings from that profile.
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "run this manual benchmark with --release"
    );
    let runs = bounded_env("LATTICE_GEMMA4_BENCH_RUNS", 3, 1, 5);
    let (corpus, source) = corpus();
    assert!(
        (2..=8).contains(&corpus.cases.len()),
        "need 2..=8 corpus cases"
    );
    assert!(corpus.cases.iter().any(|case| case.input_ids.len() < 64));
    assert!(corpus.cases.iter().any(|case| case.input_ids.len() > 64));
    let dir = model_dir()
        .canonicalize()
        .expect("existing Gemma checkpoint directory");
    let start = Instant::now();
    let model = Gemma4Model::from_safetensors(&dir).expect("load real Gemma checkpoint");
    let load = start.elapsed();
    let mut names = std::collections::HashSet::new();
    for case in &corpus.cases {
        assert!(
            !case.name.is_empty() && names.insert(&case.name),
            "unique nonempty case names"
        );
        assert!((1..=MAX_PROMPT_TOKENS).contains(&case.input_ids.len()));
        assert!(
            case.input_ids
                .iter()
                .all(|&id| (id as usize) < model.config.vocab_size)
        );
        assert!(case.input_ids.len() + DECODE_STEPS <= model.config.max_position_embeddings);
    }
    println!(
        "GEMMA_PREFILL_BENCH model_dir={dir:?} load_secs={:.6} config_sha256={} \
         corpus={source:?} runs={runs} warmups_per_case=1 decode_steps={DECODE_STEPS} \
         harness_sha256={:x} arch={} os={} caller_declared_revision={:?}",
        load.as_secs_f64(),
        hash_file(&dir.join("config.json")),
        Sha256::digest(include_bytes!("gemma_cpu_bench.rs")),
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::env::var("LATTICE_GEMMA4_BENCH_REVISION").ok(),
    );
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ] {
        println!(
            "GEMMA_PREFILL_THREADS variable={name} configured={:?}",
            std::env::var(name).ok()
        );
    }
    for case in &corpus.cases {
        let hash = token_hash(&case.input_ids);
        println!(
            "GEMMA_PREFILL_CASE name={:?} tokens={} token_ids_sha256={hash}",
            case.name,
            case.input_ids.len()
        );
        let warmup = sample(&model, &case.input_ids);
        for run in 1..=runs {
            let result = sample(&model, &case.input_ids);
            assert_eq!(result.selected, warmup.selected, "repeat token drift");
            assert_eq!(result.phases, warmup.phases, "repeat logit/cache drift");
            println!(
                "GEMMA_PREFILL_SAMPLE case={:?} run={run} tokens={} setup_secs={:.6} \
                 prefill_secs={:.6} decode1_secs={:.6} decode2_secs={:.6} selected={:?}",
                case.name,
                case.input_ids.len(),
                result.setup.as_secs_f64(),
                result.prefill.as_secs_f64(),
                result.decode[0].as_secs_f64(),
                result.decode[1].as_secs_f64(),
                result.selected,
            );
            for (phase, evidence) in result.phases.iter().enumerate() {
                println!(
                    "GEMMA_PREFILL_EVIDENCE case={:?} run={run} phase={phase} evidence={evidence:?}",
                    case.name
                );
            }
        }
    }
    println!(
        "GEMMA_PREFILL_COMPLETE cases={} samples={} decode_steps_per_sample={DECODE_STEPS}",
        corpus.cases.len(),
        corpus.cases.len() * runs
    );
}
