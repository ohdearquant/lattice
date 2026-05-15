//! Perplexity evaluator for Qwen3.5 models on a UTF-8 text corpus.
//!
//! Step 4a of ADR-044. CPU forward path only — runs against the
//! safetensors checkpoint loaded by [`Qwen35Model::from_safetensors`].
//! The Q4 / QuaRot-Q4 measurement path (Metal) ships in 4b.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin eval_perplexity -- \
//!   --model-dir ~/.lattice/models/qwen3.5-0.8b \
//!   --corpus-file wiki.test.raw \
//!   --window 512 \
//!   --stride 256
//! ```
//!
//! Flags:
//! - `--model-dir <PATH>`: directory with `config.json` + safetensors +
//!   `tokenizer.json`. Loaded via [`Qwen35Model::from_safetensors`].
//! - `--corpus-file <PATH>`: UTF-8 text file. Tokenized end-to-end with the
//!   model's BPE tokenizer.
//! - `--window <USIZE>`: context window length in tokens. Default `512`.
//! - `--stride <USIZE>`: tokens advanced between windows. Default `256`.
//! - `--max-tokens <USIZE>`: cap total tokens scored (after tokenization).
//!   Useful for smoke runs on a long corpus. Default: no cap.
//! - `-h, --help`: print usage.
//!
//! Exit codes:
//! - `0`: PPL computed successfully; report printed to stdout.
//! - `1`: error (missing file, parse failure, tokenization error, etc.).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use lattice_inference::model::qwen35::{PerplexityConfig, Qwen35Model};
use lattice_inference::tokenizer::common::Tokenizer;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir: Option<PathBuf> = None;
    let mut corpus_file: Option<PathBuf> = None;
    let mut window: Option<usize> = None;
    let mut stride: Option<usize> = None;
    let mut max_tokens: Option<usize> = None;

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
            "--help" | "-h" => {
                eprintln!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => return usage(&format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let Some(model_dir) = model_dir else {
        return usage("--model-dir is required");
    };
    let Some(corpus_file) = corpus_file else {
        return usage("--corpus-file is required");
    };

    let cfg = PerplexityConfig {
        window: window.unwrap_or(512),
        stride: stride.unwrap_or(256),
    };

    eprintln!("=== eval_perplexity ===");
    eprintln!("Model dir:   {}", model_dir.display());
    eprintln!("Corpus:      {}", corpus_file.display());
    eprintln!("Window:      {}", cfg.window);
    eprintln!("Stride:      {}", cfg.stride);
    if let Some(cap) = max_tokens {
        eprintln!("Max tokens:  {cap}");
    }
    eprintln!();

    let t_load = Instant::now();
    let model = match Qwen35Model::from_safetensors(&model_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: failed to load model: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("Model loaded in {}ms", t_load.elapsed().as_millis());

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

    let t_tok = Instant::now();
    let tokenized = model.tokenizer().tokenize(&corpus_text);
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

    let t_ppl = Instant::now();
    let report = match model.compute_perplexity(&tokens, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERROR: {e}");
            return ExitCode::FAILURE;
        }
    };
    let elapsed = t_ppl.elapsed();

    println!();
    println!("=== Perplexity Report ===");
    println!("PPL:                {:.6}", report.ppl);
    println!("Mean NLL (nats):    {:.6}", report.mean_nll);
    println!("Total NLL (nats):   {:.6}", report.total_nll);
    println!("Tokens scored:      {}", report.num_tokens_scored);
    println!("Windows:            {}", report.num_windows);
    println!("Window / Stride:    {} / {}", report.window, report.stride);
    let secs = elapsed.as_secs_f64();
    let toks_per_sec = if secs > 0.0 {
        report.num_tokens_scored as f64 / secs
    } else {
        0.0
    };
    println!("Wall time:          {secs:.2}s ({toks_per_sec:.1} tok/s)");

    ExitCode::SUCCESS
}

fn usage(msg: &str) -> ExitCode {
    eprintln!("ERROR: {msg}\n");
    eprintln!("{USAGE}");
    ExitCode::FAILURE
}

const USAGE: &str = "\
usage: eval_perplexity --model-dir <PATH> --corpus-file <PATH> [OPTIONS]

Compute strided sliding-window perplexity of a Qwen3.5 safetensors
model on a UTF-8 text corpus (ADR-044 step 4a).

required:
  --model-dir <PATH>    Directory with config.json + safetensors + tokenizer.json.
  --corpus-file <PATH>  UTF-8 text file to score.

options:
  --window <USIZE>      Context window in tokens. Default 512.
  --stride <USIZE>      Tokens advanced per window. Default 256.
  --max-tokens <USIZE>  Cap total tokens after tokenization (for smoke runs).
  -h, --help            Print this help and exit.

The harness mirrors HuggingFace's fixed-length-model recipe: each non-
first global token is scored exactly once. After the first window, every
newly scored target has at least `window - stride` and at most
`window - 1` preceding in-window tokens; the first window ramps from 1
prior token (target 1) up to `window - 1`. Context never crosses window
boundaries.
";
