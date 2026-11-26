//! `/v1/systemone` CPU scoring cost: the shared-state prefill against per-row
//! scoring, on real requests, with run-to-run identity.
//!
//! Each request is prepared with the functions the route calls
//! (`serve::systemone::prepare_request` against the checkpoint's own
//! `decider_config.json`, tokenizer and label table), then its rows are scored
//! three ways, back to back:
//!
//!   shared   what the shared CPU host does: the request's common prefix is
//!            prefilled once (`Qwen35Model::prefill_scoring_prefix`) when the
//!            request has more than one row and they share a prefix, and each
//!            row that continues it is scored from a copy
//!            (`score_option_letters_after_prefix`); any other row is scored
//!            alone (`score_option_letters_batched`). A model the batched
//!            prefill does not run (mixture-of-experts) builds no prefix, and
//!            every row is scored alone, as the host does.
//!   perrow   every row scored alone (`score_option_letters_batched`), the
//!            path before the shared prefix.
//!   perrow2  the same as `perrow`, run again: the control for the spread two
//!            identical arms show on this machine.
//!
//! The three arms of one request run adjacent in time, and their order rotates
//! with the round, so slow drift lands on every arm. Every arm's logits are
//! kept: each row's logits are compared across rounds bit for bit (is the arm
//! run-to-run identical?) and between `shared` and `perrow` (largest absolute
//! difference).
//!
//! The model is loaded at the window `lattice serve` serves by default
//! (`from_safetensors_with_max_context` with `DEFAULT_SERVE_MAX_CONTEXT`, or the
//! checkpoint's own window when smaller). The route's hang-up checks and
//! admission are not exercised.
//!
//! Env:
//!   LATTICE_MODEL_DIR       decider checkpoint directory (required)
//!   BENCH_REQUESTS          JSON Lines file, one request per line with `state`
//!                           and `questions` (required)
//!   BENCH_LIMIT             requests measured, taken in file order (default 4)
//!   BENCH_MAX_ROW_TOKENS    skip a request whose rows sum to more tokens than
//!                           this (default 16384); skipped requests are printed
//!   BENCH_ROUNDS            measured rounds after one untimed warmup round
//!                           (default 3); each arm takes each position in a
//!                           request's order equally often only when this is
//!                           a multiple of 3
//!   BENCH_MIN_ROWS          skip a request with fewer rows than this (default 1;
//!                           2 keeps only requests the shared prefill can act on)
//!
//! Output:
//!   LOAD load_ms=<f> max_context=<n>
//!   SKIP line=<n> reason=<...>
//!   REQ req=<i> line=<n> rows=<n> shared_prefix=<n> row_tokens=<n>
//!     input_tokens=<n> (shared_prefix is 0 for a one-row request, which the
//!     shared arm scores alone)
//!   RESULT round=<r> req=<i> arm=<shared|perrow|perrow2> order=<k> ms=<f>
//!     start_unix_ms=<n>
//!   CHECK req=<i> shared_identical=<0|1> perrow_identical=<0|1>
//!     perrow2_identical=<0|1> perrow_vs_perrow2_identical=<0|1>
//!     shared_vs_perrow_max_abs=<f, NaN when either arm gave a NaN logit>
//!     argmax_disagree=<rows that pick a different option under `shared` than
//!     under `perrow`>/<rows> shared_bits=<hex> perrow_bits=<hex> (FNV-1a over
//!     the logit bits of round 0, the untimed warmup, to compare two builds bit
//!     for bit)
//!   END requests=<n> rounds=<n>

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use lattice_inference::InferenceError;
use lattice_inference::model::qwen35::{Qwen35Model, shared_scoring_prefix_len};
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use lattice_serve::serve::systemone::{load_decider_config, prepare_request, resolve_labels};
use lattice_serve::serving_provider::DEFAULT_SERVE_MAX_CONTEXT;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Shared,
    PerRow,
    PerRow2,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::PerRow => "perrow",
            Self::PerRow2 => "perrow2",
        }
    }
}

struct Row {
    ids: Vec<u32>,
    labels: Vec<u32>,
}

struct Request {
    line: usize,
    input_tokens: usize,
    rows: Vec<Row>,
}

type Logits = Vec<Vec<f32>>;

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|error| format!("{name}={value:?}: {error}")),
        Err(_) => Ok(default),
    }
}

fn score(model: &Qwen35Model, rows: &[Row], arm: Arm) -> Result<Logits, String> {
    let prefix = if arm == Arm::Shared {
        let pairs: Vec<(&[u32], &[u32])> = rows
            .iter()
            .map(|row| (row.ids.as_slice(), row.labels.as_slice()))
            .collect();
        let shared = shared_scoring_prefix_len(&pairs);
        match rows.first() {
            Some(first) if rows.len() > 1 && shared > 0 => {
                match model.prefill_scoring_prefix(&first.ids[..shared]) {
                    Ok(prefix) => Some(prefix),
                    Err(InferenceError::UnsupportedModel(_)) => None,
                    Err(error) => return Err(format!("prefix: {error}")),
                }
            }
            _ => None,
        }
    } else {
        None
    };
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let scores = match &prefix {
                Some(prefix)
                    if row.ids.len() > prefix.ids().len() && row.ids.starts_with(prefix.ids()) =>
                {
                    model.score_option_letters_after_prefix(prefix, &row.ids, &row.labels)
                }
                _ => model.score_option_letters_batched(&row.ids, &row.labels),
            };
            scores
                .map(|scores| scores.logits)
                .map_err(|error| format!("row {index}: {error}"))
        })
        .collect()
}

fn bits(logits: &Logits) -> Vec<Vec<u32>> {
    logits
        .iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

fn max_abs(a: &Logits, b: &Logits) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0, |largest, difference| {
            if largest.is_nan() || difference.is_nan() {
                f32::NAN
            } else {
                largest.max(difference)
            }
        })
}

fn bits_hash(logits: &Logits) -> u64 {
    logits
        .iter()
        .flatten()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn argmax(row: &[f32]) -> Option<usize> {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
}

fn load_requests(
    path: &Path,
    tokenizer: &BpeTokenizer,
    model_dir: &Path,
    limit: usize,
    max_row_tokens: usize,
    min_rows: usize,
) -> Result<Vec<Request>, String> {
    let config = load_decider_config(model_dir).map_err(|error| error.to_string())?;
    let labels =
        resolve_labels(tokenizer, config.max_options).map_err(|error| error.to_string())?;
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let mut requests = Vec::new();
    for (line, raw) in text.lines().enumerate() {
        if requests.len() == limit {
            break;
        }
        if raw.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(raw).map_err(|error| format!("line {line}: {error}"))?;
        let body = serde_json::json!({
            "model": "default",
            "state": value["state"],
            "questions": value["questions"],
        })
        .to_string();
        let prepared = match prepare_request(tokenizer, &labels, &config, &body) {
            Ok(prepared) => prepared,
            Err(error) => {
                println!("SKIP line={line} reason=refused:{error}");
                continue;
            }
        };
        let row_tokens: usize = prepared.prompt.rows.iter().map(|row| row.ids.len()).sum();
        if row_tokens > max_row_tokens {
            println!("SKIP line={line} reason=row_tokens:{row_tokens}");
            continue;
        }
        if prepared.prompt.rows.len() < min_rows {
            println!(
                "SKIP line={line} reason=rows:{}",
                prepared.prompt.rows.len()
            );
            continue;
        }
        let mut rows = Vec::with_capacity(prepared.prompt.rows.len());
        for row in &prepared.prompt.rows {
            let Some(row_labels) = prepared.label_ids.get(..row.nopts) else {
                return Err(format!("line {line}: a row has more options than labels"));
            };
            rows.push(Row {
                ids: row.ids.clone(),
                labels: row_labels.to_vec(),
            });
        }
        requests.push(Request {
            line,
            input_tokens: prepared.input_tokens,
            rows,
        });
    }
    Ok(requests)
}

fn run() -> Result<(), String> {
    let model_dir = PathBuf::from(
        std::env::var("LATTICE_MODEL_DIR").map_err(|_| "LATTICE_MODEL_DIR is required")?,
    );
    let requests_path =
        PathBuf::from(std::env::var("BENCH_REQUESTS").map_err(|_| "BENCH_REQUESTS is required")?);
    let limit = env_usize("BENCH_LIMIT", 4)?;
    let max_row_tokens = env_usize("BENCH_MAX_ROW_TOKENS", 16384)?;
    let rounds = env_usize("BENCH_ROUNDS", 3)?;
    let min_rows = env_usize("BENCH_MIN_ROWS", 1)?;
    if limit == 0 || rounds == 0 {
        return Err("BENCH_LIMIT and BENCH_ROUNDS must be at least 1".to_string());
    }

    let started = Instant::now();
    let model =
        Qwen35Model::from_safetensors_with_max_context(&model_dir, DEFAULT_SERVE_MAX_CONTEXT)
            .map_err(|error| format!("load: {error}"))?;
    println!(
        "LOAD load_ms={:.3} max_context={}",
        started.elapsed().as_secs_f64() * 1e3,
        model.max_context()
    );
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)
        .map_err(|error| format!("could not load {}: {error}", tokenizer_path.display()))?
        .with_max_seq_len(usize::MAX);

    let requests = load_requests(
        &requests_path,
        &tokenizer,
        &model_dir,
        limit,
        max_row_tokens,
        min_rows,
    )?;
    if requests.is_empty() {
        return Err("no request was admitted".to_string());
    }
    for (index, request) in requests.iter().enumerate() {
        let pairs: Vec<(&[u32], &[u32])> = request
            .rows
            .iter()
            .map(|row| (row.ids.as_slice(), row.labels.as_slice()))
            .collect();
        let shared_prefix = if request.rows.len() > 1 {
            shared_scoring_prefix_len(&pairs)
        } else {
            0
        };
        println!(
            "REQ req={index} line={} rows={} shared_prefix={shared_prefix} row_tokens={} \
             input_tokens={}",
            request.line,
            request.rows.len(),
            request.rows.iter().map(|row| row.ids.len()).sum::<usize>(),
            request.input_tokens,
        );
    }

    let arms = [Arm::Shared, Arm::PerRow, Arm::PerRow2];
    let mut first: Vec<[Option<Logits>; 3]> = requests.iter().map(|_| [None, None, None]).collect();
    let mut identical: Vec<[bool; 3]> = requests.iter().map(|_| [true; 3]).collect();
    for round in 0..=rounds {
        for (index, request) in requests.iter().enumerate() {
            for order in 0..arms.len() {
                let slot = (order + round + index) % arms.len();
                let arm = arms[slot];
                let start_unix_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |elapsed| elapsed.as_millis());
                let started = Instant::now();
                let logits = score(&model, &request.rows, arm)
                    .map_err(|error| format!("req {index} {}: {error}", arm.name()))?;
                let ms = started.elapsed().as_secs_f64() * 1e3;
                if round > 0 {
                    println!(
                        "RESULT round={round} req={index} arm={} order={order} ms={ms:.3} \
                         start_unix_ms={start_unix_ms}",
                        arm.name()
                    );
                }
                match &first[index][slot] {
                    Some(earlier) => identical[index][slot] &= bits(earlier) == bits(&logits),
                    None => first[index][slot] = Some(logits),
                }
            }
        }
    }
    for (index, kept) in first.iter().enumerate() {
        let [Some(shared), Some(perrow), Some(perrow2)] = kept else {
            return Err(format!("req {index}: an arm kept no logits"));
        };
        let disagree = shared
            .iter()
            .zip(perrow)
            .filter(|(a, b)| argmax(a) != argmax(b))
            .count();
        println!(
            "CHECK req={index} shared_identical={} perrow_identical={} perrow2_identical={} \
             perrow_vs_perrow2_identical={} shared_vs_perrow_max_abs={:e} \
             argmax_disagree={disagree}/{} shared_bits={:016x} perrow_bits={:016x}",
            u8::from(identical[index][0]),
            u8::from(identical[index][1]),
            u8::from(identical[index][2]),
            u8::from(bits(perrow) == bits(perrow2)),
            max_abs(shared, perrow),
            perrow.len(),
            bits_hash(shared),
            bits_hash(perrow),
        );
    }
    println!("END requests={} rounds={rounds}", requests.len());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("bench_systemone_scoring: {error}");
        std::process::exit(1);
    }
}
