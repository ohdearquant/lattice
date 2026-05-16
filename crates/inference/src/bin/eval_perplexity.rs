//! Perplexity evaluator for Qwen3.5 models on a UTF-8 text corpus.
//!
//! Steps 4a + 4b of ADR-044.
//!
//! Three measurement modes, mutually exclusive:
//! - `--model-dir <PATH>`: CPU forward path on a BF16/F16/F32 safetensors
//!   checkpoint via [`Qwen35Model::from_safetensors`] (the baseline shipped in
//!   step 4a).
//! - `--q4-dir <PATH>`: Metal Q4 forward path via
//!   [`MetalQwen35State::from_q4_dir`] on a directory produced by
//!   `bin/quantize_q4` (unrotated 4-bit weights).
//! - `--q4-dir <Q4> --quarot-q4-dir <QUAROT_Q4>`: dual-Q4 measurement —
//!   runs perplexity on both an unrotated `quantize_q4` directory and a
//!   `quantize_quarot` directory, prints both reports, then the
//!   `quarot - unrotated` PPL delta and the ADR-044 acceptance gate
//!   verdict (< 0.5 PPL by default; override with `--delta-threshold`).
//!   This is the actual step-4b acceptance measurement.
//! - `--quarot-q4-dir <PATH>` alone: runs only the QuaRot-Q4 forward path
//!   (rarely useful — typically you want the unrotated baseline alongside).
//!
//! # Usage
//!
//! ```text
//! # CPU baseline (step 4a)
//! cargo run --release --features f16 --bin eval_perplexity -- \
//!   --model-dir ~/.lattice/models/qwen3.5-0.8b \
//!   --corpus-file wiki.test.raw \
//!   --window 512 --stride 256
//!
//! # Metal Q4 single (step 4b)
//! cargo run --release --features f16,metal-gpu --bin eval_perplexity -- \
//!   --q4-dir ~/.lattice/models/qwen3.5-0.8b-q4 \
//!   --tokenizer-dir ~/.lattice/models/qwen3.5-0.8b \
//!   --corpus-file wiki.test.raw
//!
//! # Step-4b acceptance: rotated-Q4 vs unrotated-Q4 delta
//! cargo run --release --features f16,metal-gpu --bin eval_perplexity -- \
//!   --q4-dir        ~/.lattice/models/qwen3.5-0.8b-q4 \
//!   --quarot-q4-dir ~/.lattice/models/qwen3.5-0.8b-q4-quarot \
//!   --tokenizer-dir ~/.lattice/models/qwen3.5-0.8b \
//!   --corpus-file wiki.test.raw
//! ```
//!
//! Flags:
//! - `--model-dir <PATH>`: CPU mode. Directory with `config.json` + safetensors
//!   + `tokenizer.json`. Loaded via [`Qwen35Model::from_safetensors`].
//! - `--q4-dir <PATH>`: Metal Q4 mode. Directory with `.q4` / `.f16` /
//!   `config.json` / `quantize_index.json` produced by `bin/quantize_q4`.
//! - `--quarot-q4-dir <PATH>`: Metal Q4 mode on a `bin/quantize_quarot`
//!   output directory (rotated 4-bit weights, same file layout).
//! - `--tokenizer-dir <PATH>`: Metal modes only. Directory containing
//!   `tokenizer.json`. Both `quantize_q4` and `quantize_quarot` ship the
//!   model weights but NOT the BPE tokenizer, so this typically points at
//!   the source safetensors directory.
//! - `--corpus-file <PATH>`: UTF-8 text file. Tokenized end-to-end with
//!   the model's BPE tokenizer.
//! - `--window <USIZE>`: context window length in tokens. Default `512`.
//! - `--stride <USIZE>`: tokens advanced between windows. Default `256`.
//! - `--max-tokens <USIZE>`: cap total tokens scored (after tokenization).
//!   Useful for smoke runs on a long corpus. Default: no cap.
//! - `--max-cache-len <USIZE>`: Metal modes only. KV-cache capacity passed
//!   to `MetalQwen35State::from_q4_dir`. Must be `>= window`. Default:
//!   `max(window, 4096)`.
//! - `--delta-threshold <F64>`: dual-Q4 mode only. PPL delta threshold for
//!   the ADR-044 acceptance gate. Default `0.5`. Exit code `1` if the
//!   measured `quarot - unrotated` delta meets or exceeds this value.
//! - `-h, --help`: print usage.
//!
//! Exit codes:
//! - `0`: PPL computed; in dual-Q4 mode, delta < threshold (acceptance pass).
//! - `1`: error (missing file, parse failure, tokenization error) OR
//!   dual-Q4 mode with delta >= threshold (acceptance fail).
//!
//! The harness mirrors HuggingFace's fixed-length-model recipe: each non-
//! first global token is scored exactly once. After the first window, every
//! newly scored target has at least `window - stride` and at most
//! `window - 1` preceding in-window tokens; the first window ramps from 1
//! prior token (target 1) up to `window - 1`. Context never crosses window
//! boundaries.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use lattice_inference::error::InferenceError;
use lattice_inference::forward::metal_qwen35::MetalQwen35State;
use lattice_inference::model::qwen35::{PerplexityConfig, PerplexityReport, Qwen35Model};
use lattice_inference::model::qwen35_config::Qwen35Config;
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use lattice_inference::tokenizer::common::Tokenizer;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir: Option<PathBuf> = None;
    let mut q4_dir: Option<PathBuf> = None;
    let mut quarot_q4_dir: Option<PathBuf> = None;
    let mut tokenizer_dir: Option<PathBuf> = None;
    let mut corpus_file: Option<PathBuf> = None;
    let mut window: Option<usize> = None;
    let mut stride: Option<usize> = None;
    let mut max_tokens: Option<usize> = None;
    let mut max_cache_len: Option<usize> = None;
    let mut delta_threshold: Option<f64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--model-dir requires an argument");
                };
                model_dir = Some(PathBuf::from(v));
            }
            "--q4-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--q4-dir requires an argument");
                };
                q4_dir = Some(PathBuf::from(v));
            }
            "--quarot-q4-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--quarot-q4-dir requires an argument");
                };
                quarot_q4_dir = Some(PathBuf::from(v));
            }
            "--tokenizer-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--tokenizer-dir requires an argument");
                };
                tokenizer_dir = Some(PathBuf::from(v));
            }
            "--corpus-file" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--corpus-file requires an argument");
                };
                corpus_file = Some(PathBuf::from(v));
            }
            "--window" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--window requires an argument");
                };
                window = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return usage(&format!("--window: invalid usize: {e}")),
                });
            }
            "--stride" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--stride requires an argument");
                };
                stride = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return usage(&format!("--stride: invalid usize: {e}")),
                });
            }
            "--max-tokens" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--max-tokens requires an argument");
                };
                max_tokens = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return usage(&format!("--max-tokens: invalid usize: {e}")),
                });
            }
            "--max-cache-len" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--max-cache-len requires an argument");
                };
                max_cache_len = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return usage(&format!("--max-cache-len: invalid usize: {e}")),
                });
            }
            "--delta-threshold" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--delta-threshold requires an argument");
                };
                delta_threshold = Some(match v.parse::<f64>() {
                    Ok(n) => n,
                    Err(e) => return usage(&format!("--delta-threshold: invalid f64: {e}")),
                });
            }
            "--help" | "-h" => {
                eprintln!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => return usage(&format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let Some(corpus_file) = corpus_file else {
        return usage("--corpus-file is required");
    };

    let metal_paths_used = q4_dir.is_some() || quarot_q4_dir.is_some();
    if model_dir.is_some() && metal_paths_used {
        return usage("--model-dir is mutually exclusive with --q4-dir / --quarot-q4-dir");
    }
    if !metal_paths_used && model_dir.is_none() {
        return usage("one of --model-dir, --q4-dir, or --quarot-q4-dir is required");
    }
    if metal_paths_used && tokenizer_dir.is_none() {
        return usage("--tokenizer-dir is required when using --q4-dir or --quarot-q4-dir");
    }

    let cfg = PerplexityConfig {
        window: window.unwrap_or(512),
        stride: stride.unwrap_or(256),
    };

    let resolved_max_cache_len = max_cache_len.unwrap_or_else(|| cfg.window.max(4096));
    if metal_paths_used && resolved_max_cache_len < cfg.window {
        return usage(&format!(
            "--max-cache-len ({resolved_max_cache_len}) must be >= --window ({}); the Metal KV cache must fit a full window",
            cfg.window
        ));
    }

    eprintln!("=== eval_perplexity ===");
    if let Some(ref p) = model_dir {
        eprintln!("Model dir (CPU):  {}", p.display());
    }
    if let Some(ref p) = q4_dir {
        eprintln!("Q4 dir (Metal):   {}", p.display());
    }
    if let Some(ref p) = quarot_q4_dir {
        eprintln!("QuaRot-Q4 dir:    {}", p.display());
    }
    if let Some(ref p) = tokenizer_dir {
        eprintln!("Tokenizer dir:    {}", p.display());
    }
    eprintln!("Corpus:           {}", corpus_file.display());
    eprintln!("Window:           {}", cfg.window);
    eprintln!("Stride:           {}", cfg.stride);
    if metal_paths_used {
        eprintln!("Max cache len:    {resolved_max_cache_len}");
    }
    if let Some(cap) = max_tokens {
        eprintln!("Max tokens:       {cap}");
    }
    eprintln!();

    let corpus_text = match std::fs::read_to_string(&corpus_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "ERROR: failed to read corpus file {}: {e}",
                corpus_file.display()
            );
            return ExitCode::FAILURE;
        }
    };

    // -----------------------------------------------------------------------
    // Mode 1: CPU forward path on a safetensors checkpoint (step 4a).
    // -----------------------------------------------------------------------
    if let Some(model_dir) = model_dir {
        let t_load = Instant::now();
        let model = match Qwen35Model::from_safetensors(&model_dir) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("ERROR: failed to load model: {e}");
                return ExitCode::FAILURE;
            }
        };
        eprintln!("Model loaded in {}ms", t_load.elapsed().as_millis());

        let tokens = match tokenize_with(model.tokenizer(), &corpus_text, max_tokens, &corpus_file)
        {
            Ok(t) => t,
            Err(code) => return code,
        };

        let t_ppl = Instant::now();
        let report = match model.compute_perplexity(&tokens, &cfg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("ERROR: {e}");
                return ExitCode::FAILURE;
            }
        };
        let elapsed = t_ppl.elapsed();
        print_report("CPU safetensors", &report, elapsed.as_secs_f64());
        return ExitCode::SUCCESS;
    }

    // -----------------------------------------------------------------------
    // Modes 2 + 3: Metal Q4 forward path (single or dual delta).
    // -----------------------------------------------------------------------
    let tokenizer_dir = tokenizer_dir.expect("checked above");
    let tokenizer_path = tokenizer_dir.join("tokenizer.json");
    let tokenizer = match BpeTokenizer::from_tokenizer_json(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "ERROR: failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            );
            return ExitCode::FAILURE;
        }
    };
    let tokens = match tokenize_with(&tokenizer, &corpus_text, max_tokens, &corpus_file) {
        Ok(t) => t,
        Err(code) => return code,
    };

    let unrotated_report = if let Some(dir) = q4_dir.as_deref() {
        let cfg_loaded = match load_cfg_for_q4(dir) {
            Ok(c) => c,
            Err(code) => return code,
        };
        match run_metal_q4(
            dir,
            &tokenizer_path,
            &cfg_loaded,
            resolved_max_cache_len,
            &tokens,
            &cfg,
            "unrotated Q4",
        ) {
            Ok(r) => Some(r),
            Err(code) => return code,
        }
    } else {
        None
    };

    let quarot_report = if let Some(dir) = quarot_q4_dir.as_deref() {
        let cfg_loaded = match load_cfg_for_q4(dir) {
            Ok(c) => c,
            Err(code) => return code,
        };
        match run_metal_q4(
            dir,
            &tokenizer_path,
            &cfg_loaded,
            resolved_max_cache_len,
            &tokens,
            &cfg,
            "QuaRot Q4",
        ) {
            Ok(r) => Some(r),
            Err(code) => return code,
        }
    } else {
        None
    };

    // Dual mode — compute delta and verdict.
    if let (Some(u), Some(q)) = (&unrotated_report, &quarot_report) {
        let threshold = delta_threshold.unwrap_or(0.5);
        let delta = q.ppl - u.ppl;
        println!();
        println!("=== Acceptance Gate (ADR-044 step 4) ===");
        println!("Unrotated Q4 PPL: {:.6}", u.ppl);
        println!("QuaRot Q4 PPL:    {:.6}", q.ppl);
        println!("PPL delta:        {delta:+.6}  (quarot - unrotated)");
        println!("Threshold:        < {threshold:.6}");
        if delta < threshold {
            println!("Verdict:          PASS");
            return ExitCode::SUCCESS;
        } else {
            println!("Verdict:          FAIL (delta >= threshold)");
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}

fn tokenize_with(
    tokenizer: &BpeTokenizer,
    text: &str,
    max_tokens: Option<usize>,
    corpus_file: &std::path::Path,
) -> Result<Vec<u32>, ExitCode> {
    // `BpeTokenizer::from_tokenizer_json` builds with a default
    // `max_seq_len = 4_096`, which silently truncates any corpus
    // longer than that to ~4 K tokens at tokenize time. For PPL
    // evaluation we strode-walk the corpus in `--window`-sized
    // slices through the harness, so the tokenizer cap must NOT
    // bound the corpus. Bump it to a byte-level upper bound on
    // the token count (byte-level BPE emits ≤ 1 token per UTF-8
    // byte after the byte-encoder maps every byte to a token).
    // The pad-to-max-seq-len allocation is temporary — the call
    // site slices off `..real_length` immediately and drops the
    // padded buffer.
    let bumped = tokenizer.with_max_seq_len(text.len().saturating_add(64));
    let t_tok = Instant::now();
    let tokenized = bumped.tokenize(text);
    let mut tokens: Vec<u32> = tokenized.input_ids[..tokenized.real_length].to_vec();
    if let Some(cap) = max_tokens {
        if tokens.len() > cap {
            tokens.truncate(cap);
        }
    }
    eprintln!(
        "Tokenized {} → {} tokens in {}ms",
        corpus_file.display(),
        tokens.len(),
        t_tok.elapsed().as_millis()
    );
    Ok(tokens)
}

fn load_cfg_for_q4(dir: &std::path::Path) -> Result<Qwen35Config, ExitCode> {
    let config_path = dir.join("config.json");
    if !config_path.exists() {
        eprintln!("ERROR: Q4 dir {} is missing config.json", dir.display());
        return Err(ExitCode::FAILURE);
    }
    Qwen35Config::from_config_json(&config_path).map_err(|e| {
        eprintln!("ERROR: failed to parse {}: {e}", config_path.display());
        ExitCode::FAILURE
    })
}

fn run_metal_q4(
    q4_dir: &std::path::Path,
    tokenizer_path: &std::path::Path,
    cfg_loaded: &Qwen35Config,
    max_cache_len: usize,
    tokens: &[u32],
    ppl_cfg: &PerplexityConfig,
    label: &str,
) -> Result<PerplexityReport, ExitCode> {
    let t_load = Instant::now();
    let mut state =
        match MetalQwen35State::from_q4_dir(q4_dir, tokenizer_path, cfg_loaded, max_cache_len) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "ERROR: failed to load {label} from {}: {e}",
                    q4_dir.display()
                );
                return Err(ExitCode::FAILURE);
            }
        };
    eprintln!("[{label}] loaded in {}ms", t_load.elapsed().as_millis());

    let t_ppl = Instant::now();
    let report = match state.compute_perplexity(tokens, ppl_cfg) {
        Ok(r) => r,
        Err(InferenceError::Inference(msg)) => {
            eprintln!("ERROR ({label}): {msg}");
            return Err(ExitCode::FAILURE);
        }
        Err(e) => {
            eprintln!("ERROR ({label}): {e}");
            return Err(ExitCode::FAILURE);
        }
    };
    let elapsed = t_ppl.elapsed();
    print_report(label, &report, elapsed.as_secs_f64());
    Ok(report)
}

fn print_report(label: &str, report: &PerplexityReport, secs: f64) {
    println!();
    println!("=== Perplexity Report ({label}) ===");
    println!("PPL:                {:.6}", report.ppl);
    println!("Mean NLL (nats):    {:.6}", report.mean_nll);
    println!("Total NLL (nats):   {:.6}", report.total_nll);
    println!("Tokens scored:      {}", report.num_tokens_scored);
    println!("Windows:            {}", report.num_windows);
    println!("Window / Stride:    {} / {}", report.window, report.stride);
    let toks_per_sec = if secs > 0.0 {
        report.num_tokens_scored as f64 / secs
    } else {
        0.0
    };
    println!("Wall time:          {secs:.2}s ({toks_per_sec:.1} tok/s)");
}

fn usage(msg: &str) -> ExitCode {
    eprintln!("ERROR: {msg}\n");
    eprintln!("{USAGE}");
    ExitCode::FAILURE
}

const USAGE: &str = "\
usage: eval_perplexity [MODE-FLAGS] --corpus-file <PATH> [OPTIONS]

Compute strided sliding-window perplexity of a Qwen3.5 model on a UTF-8
text corpus (ADR-044 steps 4a + 4b). Three measurement modes:

  CPU safetensors (step 4a):
    --model-dir <PATH>     Directory with config.json + safetensors + tokenizer.json.

  Metal Q4 single (step 4b):
    --q4-dir <PATH>        bin/quantize_q4 output dir (unrotated 4-bit weights).
    --tokenizer-dir <PATH> Source model dir holding tokenizer.json.

  Metal Q4 acceptance gate (step 4b delta):
    --q4-dir <Q4>          bin/quantize_q4 output dir (unrotated baseline).
    --quarot-q4-dir <QR>   bin/quantize_quarot output dir (rotated 4-bit weights).
    --tokenizer-dir <PATH> Source model dir holding tokenizer.json.

    Prints both PPL reports and the quarot-unrotated delta. Exits non-zero
    if delta >= --delta-threshold (default 0.5 — the ADR-044 acceptance
    gate). The single-tokenizer assumption requires both Q4 dirs to come
    from the same source safetensors checkpoint.

required (in addition to mode flags):
  --corpus-file <PATH>     UTF-8 text file to score.

options:
  --window <USIZE>         Context window in tokens. Default 512.
  --stride <USIZE>         Tokens advanced per window. Default 256.
  --max-tokens <USIZE>     Cap total tokens after tokenization (for smoke runs).
  --max-cache-len <USIZE>  Metal modes only. KV-cache capacity passed to
                           from_q4_dir. Must be >= --window. Default max(window, 4096).
  --delta-threshold <F64>  Dual-Q4 mode only. PPL delta acceptance threshold.
                           Default 0.5. Exit 1 if measured delta >= threshold.
  -h, --help               Print this help and exit.

The harness mirrors HuggingFace's fixed-length-model recipe: each non-
first global token is scored exactly once. After the first window, every
newly scored target has at least `window - stride` and at most
`window - 1` preceding in-window tokens; the first window ramps from 1
prior token (target 1) up to `window - 1`. Context never crosses window
boundaries.
";
