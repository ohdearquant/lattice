//! Perplexity evaluator for Qwen3.5 models on a UTF-8 text corpus.
//!
//! ADR-044 step 4 (sub-steps 4a/4b/4c).
//!
//! Four measurement modes, mutually exclusive:
//! - `--model-dir <PATH>`: CPU forward path on a BF16/F16/F32 safetensors
//!   checkpoint via [`Qwen35Model::from_safetensors`] (the baseline shipped in
//!   step 4a).
//! - `--metal-model-dir <PATH>`: Metal GPU forward path on a BF16/F16/F32
//!   safetensors checkpoint (same directory layout as `--model-dir`), via
//!   [`MetalQwen35State::new`] on weights loaded through
//!   [`Qwen35Model::from_safetensors`]. This is the full-precision
//!   configuration actually served over Metal — the CPU path above
//!   approximates it but does not exercise the Metal kernels (f16
//!   accumulation, kernel-specific rounding) that drift can hide in.
//!   Carries its own tokenizer from the checkpoint directory, same as
//!   `--model-dir`; does not take `--tokenizer-dir`.
//! - `--q4-dir <PATH>`: Metal Q4 forward path via
//!   [`MetalQwen35State::from_q4_dir`] on a directory produced by
//!   `bin/quantize_q4` (unrotated 4-bit weights).
//! - `--q4-dir <Q4> --quarot-q4-dir <QUAROT_Q4>`: dual-Q4 measurement —
//!   runs perplexity on both an unrotated `quantize_q4` directory and a
//!   `quantize_quarot` directory, prints both reports, then the
//!   `quarot - unrotated` PPL delta and the ADR-044 acceptance gate
//!   verdict (< 0.5 PPL by default; override with `--delta-threshold`).
//!   This is the ADR-044 step-4 acceptance measurement — it is also the
//!   ONLY thing that can advance a `quantize_quarot` artifact's promotion
//!   marker (#1103) out of `unpromoted`: this mode records the measured
//!   delta/verdict into `<quarot-q4-dir>/quantize_index.json`'s
//!   `promotion` field (`promoted` on PASS, `rejected` on FAIL).
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
//! # Metal BF16 safetensors (full-precision Metal serving path)
//! cargo run --release --features f16,metal-gpu --bin eval_perplexity -- \
//!   --metal-model-dir ~/.lattice/models/qwen3.5-0.8b \
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
//! - `--metal-model-dir <PATH>`: Metal BF16/F16/F32 mode. Same directory
//!   layout as `--model-dir` (`config.json` + safetensors + `tokenizer.json`).
//!   Loaded via [`Qwen35Model::from_safetensors`], then its weights and config
//!   are handed to [`MetalQwen35State::new`]. Requires the `metal-gpu` feature
//!   on macOS to actually run; without it, state construction fails at
//!   runtime with a capability error, same as the other Metal modes.
//! - `--q4-dir <PATH>`: Metal Q4 mode. Directory with `.q4` / `.f16` /
//!   `config.json` / `quantize_index.json` produced by `bin/quantize_q4`.
//! - `--quarot-q4-dir <PATH>`: Metal Q4 mode on a `bin/quantize_quarot`
//!   output directory (rotated 4-bit weights, same file layout).
//! - `--tokenizer-dir <PATH>`: Metal **Q4** modes only (`--q4-dir` /
//!   `--quarot-q4-dir`). Directory containing `tokenizer.json`. Both
//!   `quantize_q4` and `quantize_quarot` ship the model weights but NOT the
//!   BPE tokenizer, so this typically points at the source safetensors
//!   directory. `--model-dir` and `--metal-model-dir` carry their own
//!   tokenizer and ignore this flag.
//! - `--corpus-file <PATH>`: UTF-8 text file. Tokenized end-to-end with
//!   the model's BPE tokenizer.
//! - `--window <USIZE>`: context window length in tokens. Default `512`.
//! - `--stride <USIZE>`: tokens advanced between windows. Default `256`.
//! - `--max-tokens <USIZE>`: cap total tokens scored (after tokenization).
//!   Useful for smoke runs on a long corpus. Default: no cap.
//! - `--max-cache-len <USIZE>`: Metal modes only (`--metal-model-dir`,
//!   `--q4-dir`, `--quarot-q4-dir`). KV-cache capacity passed to the Metal
//!   state constructor. Must be `>= window`. Default: `max(window, 4096)`.
//! - `--delta-threshold <F64>`: dual-Q4 mode only. PPL delta threshold for
//!   the ADR-044 acceptance gate. Default `0.5`. Exit code `1` if the
//!   measured `quarot - unrotated` delta meets or exceeds this value.
//! - `--random-lora-rank <N>`: Metal modes only (`--metal-model-dir`,
//!   `--q4-dir`, `--quarot-q4-dir`). Generate a random synthetic
//!   LoRA adapter at rank N and load it, exercising the full
//!   Metal+QuaRot+LoRA code path end-to-end.
//! - `--quarot-seed <N>`: u64 seed for QuaRot counter-rotation and random
//!   A/B matrix generation. Passed as `Some(seed)` to `load_lora_adapter`.
//! - `--lora-scale <F>`: LoRA scale factor. Default `1.0`.
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
use lattice_inference::forward::metal_qwen35::{LoraLayerData, MetalQwen35State};
use lattice_inference::model::qwen35::{PerplexityConfig, PerplexityReport, Qwen35Model};
use lattice_inference::model::qwen35_config::Qwen35Config;
use lattice_inference::quant::quarot::convert::record_ppl_gate_result;
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use lattice_inference::tokenizer::common::Tokenizer;

/// Emit a `@@lattice {"ev":"perplexity",...}` structured event line to stdout.
fn emit_perplexity_event(label: &str, report: &PerplexityReport, elapsed_ms: u128) {
    let obj = serde_json::json!({
        "ev": "perplexity",
        "label": label,
        "ppl": report.ppl,
        "nll": report.mean_nll,
        "tokens": report.num_tokens_scored,
        "windows": report.num_windows,
        "ms": elapsed_ms,
    });
    println!("@@lattice {obj}");
}

/// Outcome of [`parse_args`]: either the caller asked for `--help`/`-h`
/// (print usage, exit success), or a fully validated set of arguments.
#[derive(Debug)]
enum ArgsOutcome {
    Help,
    Args(Box<ParsedArgs>),
}

/// Parsed and validated CLI arguments. Everything mode-selection and
/// mutual-exclusion related has already been checked by the time this is
/// constructed — `main` only has to dispatch on which `Option` fields are
/// `Some`.
#[derive(Debug)]
struct ParsedArgs {
    model_dir: Option<PathBuf>,
    metal_model_dir: Option<PathBuf>,
    q4_dir: Option<PathBuf>,
    quarot_q4_dir: Option<PathBuf>,
    tokenizer_dir: Option<PathBuf>,
    corpus_file: PathBuf,
    cfg: PerplexityConfig,
    max_tokens: Option<usize>,
    resolved_max_cache_len: usize,
    delta_threshold: Option<f64>,
    random_lora_rank: Option<usize>,
    quarot_seed: Option<u64>,
    lora_scale: Option<f32>,
    emit_json: bool,
    json_label: Option<String>,
}

/// Parse argv into [`ArgsOutcome`], or an error message describing the
/// first invalid flag / value / mode combination encountered. Pure and
/// side-effect-free (no I/O, no process exit) so the validation rules —
/// mode mutual exclusion, which flags require `--tokenizer-dir`, which
/// require a Metal mode — are unit-testable without a model checkpoint.
fn parse_args(args: &[String]) -> Result<ArgsOutcome, String> {
    let mut model_dir: Option<PathBuf> = None;
    let mut metal_model_dir: Option<PathBuf> = None;
    let mut q4_dir: Option<PathBuf> = None;
    let mut quarot_q4_dir: Option<PathBuf> = None;
    let mut tokenizer_dir: Option<PathBuf> = None;
    let mut corpus_file: Option<PathBuf> = None;
    let mut window: Option<usize> = None;
    let mut stride: Option<usize> = None;
    let mut max_tokens: Option<usize> = None;
    let mut max_cache_len: Option<usize> = None;
    let mut delta_threshold: Option<f64> = None;
    let mut random_lora_rank: Option<usize> = None;
    let mut quarot_seed: Option<u64> = None;
    let mut lora_scale: Option<f32> = None;
    let mut emit_json: bool = false;
    let mut json_label: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--model-dir requires an argument".to_string());
                };
                model_dir = Some(PathBuf::from(v));
            }
            "--metal-model-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--metal-model-dir requires an argument".to_string());
                };
                metal_model_dir = Some(PathBuf::from(v));
            }
            "--q4-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--q4-dir requires an argument".to_string());
                };
                q4_dir = Some(PathBuf::from(v));
            }
            "--quarot-q4-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--quarot-q4-dir requires an argument".to_string());
                };
                quarot_q4_dir = Some(PathBuf::from(v));
            }
            "--tokenizer-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--tokenizer-dir requires an argument".to_string());
                };
                tokenizer_dir = Some(PathBuf::from(v));
            }
            "--corpus-file" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--corpus-file requires an argument".to_string());
                };
                corpus_file = Some(PathBuf::from(v));
            }
            "--window" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--window requires an argument".to_string());
                };
                window = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return Err(format!("--window: invalid usize: {e}")),
                });
            }
            "--stride" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--stride requires an argument".to_string());
                };
                stride = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return Err(format!("--stride: invalid usize: {e}")),
                });
            }
            "--max-tokens" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--max-tokens requires an argument".to_string());
                };
                max_tokens = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return Err(format!("--max-tokens: invalid usize: {e}")),
                });
            }
            "--max-cache-len" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--max-cache-len requires an argument".to_string());
                };
                max_cache_len = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return Err(format!("--max-cache-len: invalid usize: {e}")),
                });
            }
            "--delta-threshold" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--delta-threshold requires an argument".to_string());
                };
                delta_threshold = Some(match v.parse::<f64>() {
                    Ok(n) => n,
                    Err(e) => return Err(format!("--delta-threshold: invalid f64: {e}")),
                });
            }
            "--random-lora-rank" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--random-lora-rank requires an argument".to_string());
                };
                random_lora_rank = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return Err(format!("--random-lora-rank: invalid usize: {e}")),
                });
            }
            "--quarot-seed" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--quarot-seed requires an argument".to_string());
                };
                quarot_seed = Some(match v.parse::<u64>() {
                    Ok(n) => n,
                    Err(e) => return Err(format!("--quarot-seed: invalid u64: {e}")),
                });
            }
            "--lora-scale" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--lora-scale requires an argument".to_string());
                };
                lora_scale = Some(match v.parse::<f32>() {
                    Ok(n) => n,
                    Err(e) => return Err(format!("--lora-scale: invalid f32: {e}")),
                });
            }
            "--json" => {
                emit_json = true;
            }
            "--label" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Err("--label requires an argument".to_string());
                };
                json_label = Some(v.clone());
            }
            "--help" | "-h" => return Ok(ArgsOutcome::Help),
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let Some(corpus_file) = corpus_file else {
        return Err("--corpus-file is required".to_string());
    };

    // ------------------------------------------------------------------
    // Mode mutual-exclusion / requirement rules. `metal_paths_used` covers
    // the two Q4 modes (they share the "requires --tokenizer-dir" rule,
    // since neither `quantize_q4` nor `quantize_quarot` output ships a
    // tokenizer); `metal_model_used` is the new BF16-on-Metal mode, which
    // carries its own tokenizer like `--model-dir` and so is deliberately
    // NOT part of `metal_paths_used`. `any_metal_used` is every mode that
    // constructs a `MetalQwen35State` and therefore shares the Metal
    // KV-cache-size rule and the `--random-lora-rank` requirement.
    // ------------------------------------------------------------------
    let metal_paths_used = q4_dir.is_some() || quarot_q4_dir.is_some();
    let metal_model_used = metal_model_dir.is_some();
    let any_metal_used = metal_paths_used || metal_model_used;

    if model_dir.is_some() && metal_paths_used {
        return Err(
            "--model-dir is mutually exclusive with --q4-dir / --quarot-q4-dir".to_string(),
        );
    }
    if model_dir.is_some() && metal_model_used {
        return Err("--model-dir is mutually exclusive with --metal-model-dir".to_string());
    }
    if metal_model_used && metal_paths_used {
        return Err(
            "--metal-model-dir is mutually exclusive with --q4-dir / --quarot-q4-dir".to_string(),
        );
    }
    if !any_metal_used && model_dir.is_none() {
        return Err(
            "one of --model-dir, --metal-model-dir, --q4-dir, or --quarot-q4-dir is required"
                .to_string(),
        );
    }
    if metal_paths_used && tokenizer_dir.is_none() {
        return Err(
            "--tokenizer-dir is required when using --q4-dir or --quarot-q4-dir".to_string(),
        );
    }
    if random_lora_rank.is_some() && !any_metal_used {
        return Err(
            "--random-lora-rank requires --metal-model-dir, --q4-dir, or --quarot-q4-dir (Metal mode only)"
                .to_string(),
        );
    }

    let cfg = PerplexityConfig {
        window: window.unwrap_or(512),
        stride: stride.unwrap_or(256),
    };

    let resolved_max_cache_len = max_cache_len.unwrap_or_else(|| cfg.window.max(4096));
    if any_metal_used && resolved_max_cache_len < cfg.window {
        return Err(format!(
            "--max-cache-len ({resolved_max_cache_len}) must be >= --window ({}); the Metal KV cache must fit a full window",
            cfg.window
        ));
    }

    Ok(ArgsOutcome::Args(Box::new(ParsedArgs {
        model_dir,
        metal_model_dir,
        q4_dir,
        quarot_q4_dir,
        tokenizer_dir,
        corpus_file,
        cfg,
        max_tokens,
        resolved_max_cache_len,
        delta_threshold,
        random_lora_rank,
        quarot_seed,
        lora_scale,
        emit_json,
        json_label,
    })))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let ParsedArgs {
        model_dir,
        metal_model_dir,
        q4_dir,
        quarot_q4_dir,
        tokenizer_dir,
        corpus_file,
        cfg,
        max_tokens,
        resolved_max_cache_len,
        delta_threshold,
        random_lora_rank,
        quarot_seed,
        lora_scale,
        emit_json,
        json_label,
    } = match parse_args(&args) {
        Ok(ArgsOutcome::Help) => {
            eprintln!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Ok(ArgsOutcome::Args(parsed)) => *parsed,
        Err(msg) => return usage(&msg),
    };

    let metal_paths_used = q4_dir.is_some() || quarot_q4_dir.is_some();
    let any_metal_used = metal_paths_used || metal_model_dir.is_some();

    eprintln!("=== eval_perplexity ===");
    if let Some(ref p) = model_dir {
        eprintln!("Model dir (CPU):  {}", p.display());
    }
    if let Some(ref p) = metal_model_dir {
        eprintln!("Metal model dir:  {}", p.display());
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
    if any_metal_used {
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
        if emit_json {
            let label = json_label.as_deref().unwrap_or("bf16");
            emit_perplexity_event(label, &report, elapsed.as_millis());
        }
        return ExitCode::SUCCESS;
    }

    // -----------------------------------------------------------------------
    // Mode 2: Metal GPU forward path on a BF16/F16/F32 safetensors checkpoint
    // (the full-precision configuration served over Metal).
    // -----------------------------------------------------------------------
    if let Some(metal_model_dir) = metal_model_dir {
        return match run_metal_model_dir(
            &metal_model_dir,
            resolved_max_cache_len,
            max_tokens,
            &corpus_text,
            &corpus_file,
            &cfg,
            random_lora_rank,
            quarot_seed,
            lora_scale.unwrap_or(1.0),
            emit_json,
            json_label.as_deref().or(Some("metal-bf16")),
        ) {
            Ok(_) => ExitCode::SUCCESS,
            Err(code) => code,
        };
    }

    // -----------------------------------------------------------------------
    // Modes 3 + 4: Metal Q4 forward path (single or dual delta).
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
            random_lora_rank,
            quarot_seed,
            lora_scale.unwrap_or(1.0),
            emit_json,
            json_label.as_deref().or(Some("q4")),
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
            random_lora_rank,
            quarot_seed,
            lora_scale.unwrap_or(1.0),
            emit_json,
            json_label.as_deref().or(Some("quarot")),
        ) {
            Ok(r) => Some(r),
            Err(code) => return code,
        }
    } else {
        None
    };

    // Dual mode — compute delta and verdict, then durably record the
    // result against the QuaRot artifact's own promotion marker (#1103).
    // This IS the ADR-044 step-4 acceptance measurement `quantize_quarot`
    // itself cannot run (no baseline dir or corpus) — recording here closes
    // the gap between "converter exits 0" and "quality gate ran".
    if let (Some(u), Some(q)) = (&unrotated_report, &quarot_report) {
        let threshold = delta_threshold.unwrap_or(0.5);
        let delta = q.ppl - u.ppl;
        println!();
        println!("=== Acceptance Gate (ADR-044 step 4) ===");
        println!("Unrotated Q4 PPL: {:.6}", u.ppl);
        println!("QuaRot Q4 PPL:    {:.6}", q.ppl);
        println!("PPL delta:        {delta:+.6}  (quarot - unrotated)");
        println!("Threshold:        < {threshold:.6}");

        let quarot_dir = quarot_q4_dir.as_deref().expect("quarot_report is Some");
        let record = match record_ppl_gate_result(quarot_dir, u.ppl, q.ppl, threshold) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "ERROR: PPL delta was measured but could not be recorded against {}: {e}",
                    quarot_dir.display()
                );
                return ExitCode::FAILURE;
            }
        };
        println!(
            "Promotion:        {:?} (recorded to {}/quantize_index.json)",
            record.state,
            quarot_dir.display()
        );

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
    if let Some(cap) = max_tokens
        && tokens.len() > cap
    {
        tokens.truncate(cap);
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
    Qwen35Config::from_model_dir(dir).map_err(|e| {
        eprintln!("ERROR: {e}");
        ExitCode::FAILURE
    })
}

#[allow(clippy::too_many_arguments)]
fn run_metal_q4(
    q4_dir: &std::path::Path,
    tokenizer_path: &std::path::Path,
    cfg_loaded: &Qwen35Config,
    max_cache_len: usize,
    tokens: &[u32],
    ppl_cfg: &PerplexityConfig,
    label: &str,
    random_lora_rank: Option<usize>,
    quarot_seed: Option<u64>,
    lora_scale: f32,
    emit_json: bool,
    json_label: Option<&str>,
) -> Result<PerplexityReport, ExitCode> {
    // `measurement` is only declared under cfg(macos, metal-gpu) (see
    // src/measurement.rs), and this binary — unlike the bench_* harnesses —
    // has no top-level cfg gate, so it must build on every platform/feature
    // set. Gate the guard the same way its own dependency is gated.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();
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

    run_on_metal_state(
        &mut state,
        cfg_loaded,
        tokens,
        ppl_cfg,
        label,
        random_lora_rank,
        quarot_seed,
        lora_scale,
        emit_json,
        json_label,
    )
}

/// Load a BF16/F16/F32 safetensors checkpoint (the same directory layout
/// `--model-dir` reads) and run it through the Metal GPU forward path
/// instead of the CPU path. Reuses [`Qwen35Model::from_safetensors`] for
/// loading (weights, config, and tokenizer all come from there — this mode
/// carries its own tokenizer exactly like `--model-dir`, unlike the Q4
/// modes) and [`run_on_metal_state`] for the LoRA-attach + perplexity +
/// report sequence, which is identical to the Q4 modes because
/// `MetalQwen35State::compute_perplexity` is loading-path-agnostic — it
/// drives the same shared `run_strided_perplexity` window walk regardless
/// of whether the state came from `new` or `from_q4_dir`.
#[allow(clippy::too_many_arguments)]
fn run_metal_model_dir(
    metal_model_dir: &std::path::Path,
    max_cache_len: usize,
    max_tokens: Option<usize>,
    corpus_text: &str,
    corpus_file: &std::path::Path,
    ppl_cfg: &PerplexityConfig,
    random_lora_rank: Option<usize>,
    quarot_seed: Option<u64>,
    lora_scale: f32,
    emit_json: bool,
    json_label: Option<&str>,
) -> Result<PerplexityReport, ExitCode> {
    let label = "Metal BF16 safetensors";

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    let t_load = Instant::now();
    let model = match Qwen35Model::from_safetensors(metal_model_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: failed to load model: {e}");
            return Err(ExitCode::FAILURE);
        }
    };
    eprintln!("Model loaded in {}ms", t_load.elapsed().as_millis());

    let tokens = tokenize_with(model.tokenizer(), corpus_text, max_tokens, corpus_file)?;

    let t_state = Instant::now();
    let mut state = match MetalQwen35State::new(model.weights(), model.config(), max_cache_len) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "ERROR: failed to construct Metal state from {}: {e}",
                metal_model_dir.display()
            );
            return Err(ExitCode::FAILURE);
        }
    };
    eprintln!(
        "[{label}] Metal state constructed in {}ms",
        t_state.elapsed().as_millis()
    );

    run_on_metal_state(
        &mut state,
        model.config(),
        &tokens,
        ppl_cfg,
        label,
        random_lora_rank,
        quarot_seed,
        lora_scale,
        emit_json,
        json_label,
    )
}

/// Shared tail for every Metal mode once a [`MetalQwen35State`] has been
/// constructed (regardless of whether it came from `from_q4_dir` or `new`):
/// optionally attach a random synthetic LoRA adapter, then run and report
/// perplexity via the state's own `compute_perplexity`.
#[allow(clippy::too_many_arguments)]
fn run_on_metal_state(
    state: &mut MetalQwen35State,
    cfg_loaded: &Qwen35Config,
    tokens: &[u32],
    ppl_cfg: &PerplexityConfig,
    label: &str,
    random_lora_rank: Option<usize>,
    quarot_seed: Option<u64>,
    lora_scale: f32,
    emit_json: bool,
    json_label: Option<&str>,
) -> Result<PerplexityReport, ExitCode> {
    if let Some(rank) = random_lora_rank {
        let layers = generate_random_lora_layers(cfg_loaded, rank, quarot_seed.unwrap_or(0));
        let module_count = layers.len();
        match state.load_lora_adapter(layers, lora_scale, quarot_seed) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("ERROR ({label}): failed to load random LoRA adapter: {e}");
                return Err(ExitCode::FAILURE);
            }
        }
        eprintln!(
            "[{label}] loaded random LoRA adapter: rank={rank}, modules={module_count}, quarot_seed={quarot_seed:?}"
        );
    }

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
    if emit_json {
        let ev_label = json_label.unwrap_or(label);
        emit_perplexity_event(ev_label, &report, elapsed.as_millis());
    }
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

fn generate_random_lora_layers(cfg: &Qwen35Config, rank: usize, seed: u64) -> Vec<LoraLayerData> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let mut layers = Vec::new();
    let mut rng_state = seed;

    let mut next_f32 = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 11) as u32 as f32 / u32::MAX as f32) - 0.5
    };

    for layer_idx in 0..cfg.num_hidden_layers {
        let is_full = cfg.is_full_attention(layer_idx);

        let attn_modules: Vec<(&str, usize, usize)> = if is_full {
            vec![
                ("q_proj", hidden, 2 * cfg.full_q_dim()),
                ("k_proj", hidden, cfg.full_kv_dim()),
                ("v_proj", hidden, cfg.full_kv_dim()),
                ("o_proj", cfg.full_q_dim(), hidden),
            ]
        } else {
            vec![
                ("in_proj_qkv", hidden, cfg.linear_qkv_dim()),
                ("in_proj_z", hidden, cfg.linear_output_dim()),
                ("out_proj", cfg.linear_output_dim(), hidden),
            ]
        };

        let mlp_modules: Vec<(&str, usize, usize)> = vec![
            ("gate_proj", hidden, inter),
            ("up_proj", hidden, inter),
            ("down_proj", inter, hidden),
        ];

        for (module, d_in, d_out) in attn_modules.into_iter().chain(mlp_modules.into_iter()) {
            let a: Vec<f32> = (0..rank * d_in).map(|_| next_f32() * 0.02).collect();
            let b: Vec<f32> = (0..d_out * rank).map(|_| next_f32() * 0.02).collect();
            layers.push(LoraLayerData {
                layer_idx,
                module: module.to_string(),
                a,
                b,
                rank,
                d_in,
                d_out,
            });
        }
    }

    layers
}

fn usage(msg: &str) -> ExitCode {
    eprintln!("ERROR: {msg}\n");
    eprintln!("{USAGE}");
    ExitCode::FAILURE
}

const USAGE: &str = "\
usage: eval_perplexity [MODE-FLAGS] --corpus-file <PATH> [OPTIONS]

Compute strided sliding-window perplexity of a Qwen3.5 model on a UTF-8
text corpus (ADR-044 step 4). Four measurement modes:

  CPU safetensors (step 4a):
    --model-dir <PATH>     Directory with config.json + safetensors + tokenizer.json.

  Metal BF16 safetensors (full-precision Metal serving path):
    --metal-model-dir <PATH>
                           Same directory layout as --model-dir. Carries its
                           own tokenizer; does not take --tokenizer-dir.

  Metal Q4 single (step 4b):
    --q4-dir <PATH>        bin/quantize_q4 output dir (unrotated 4-bit weights).
    --tokenizer-dir <PATH> Source model dir holding tokenizer.json.

  Metal Q4 acceptance gate (step 4 delta):
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
  --max-cache-len <USIZE>  Metal modes only (--metal-model-dir, --q4-dir,
                           --quarot-q4-dir). KV-cache capacity passed to the
                           Metal state constructor. Must be >= --window.
                           Default max(window, 4096).
  --delta-threshold <F64>  Dual-Q4 mode only. PPL delta acceptance threshold.
                           Default 0.5. Exit 1 if measured delta >= threshold.
  --random-lora-rank <N>   Metal modes only (--metal-model-dir, --q4-dir,
                           --quarot-q4-dir). Generate a random synthetic LoRA
                           adapter at rank N for all supported modules on all
                           layers and load it via load_lora_adapter. Exercises
                           the full Metal+QuaRot+LoRA code path end-to-end.
  --quarot-seed <N>        u64 seed passed as quarot_seed to load_lora_adapter.
                           Also seeds the random A/B matrix generation. Default:
                           omitted (None passed to load_lora_adapter, seed 0 for
                           matrix generation).
  --lora-scale <F>         LoRA scale factor (alpha/rank). Default 1.0.
  --json                   Emit a structured @@lattice line to stdout for each target
                           evaluated (machine-readable output for the macOS app).
  --label <STRING>         Label to use in the --json event (overrides per-target default).
                           Defaults: bf16 for --model-dir, metal-bf16 for
                           --metal-model-dir, q4 for --q4-dir, quarot for
                           --quarot-q4-dir.
  -h, --help               Print this help and exit.

The harness mirrors HuggingFace's fixed-length-model recipe: each non-
first global token is scored exactly once. After the first window, every
newly scored target has at least `window - stride` and at most
`window - 1` preceding in-window tokens; the first window ramps from 1
prior token (target 1) up to `window - 1`. Context never crosses window
boundaries.
";

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn tokenize_with_uncaps_long_corpus() {
        let tok_path = Path::new(concat!(
            env!("HOME"),
            "/.lattice/models/qwen3.5-0.8b/tokenizer.json"
        ));
        if !tok_path.exists() {
            eprintln!(
                "SKIP: tokenizer not at {}; need Qwen3.5-0.8B locally",
                tok_path.display()
            );
            return;
        }
        let tokenizer = BpeTokenizer::from_tokenizer_json(tok_path).unwrap();
        assert_eq!(
            tokenizer.max_seq_len(),
            4096,
            "test assumes BPE default max_seq_len is 4096"
        );

        let long_text = "a ".repeat(6000);
        let fake_path = Path::new("/tmp/test_corpus.txt");
        let tokens = tokenize_with(&tokenizer, &long_text, None, fake_path).unwrap();

        assert!(
            tokens.len() > 4096,
            "tokenize_with must not be capped at BPE default max_seq_len 4096; got {} tokens",
            tokens.len()
        );
    }

    #[test]
    fn tokenize_with_respects_max_tokens_after_uncap() {
        let tok_path = Path::new(concat!(
            env!("HOME"),
            "/.lattice/models/qwen3.5-0.8b/tokenizer.json"
        ));
        if !tok_path.exists() {
            return;
        }
        let tokenizer = BpeTokenizer::from_tokenizer_json(tok_path).unwrap();
        let long_text = "a ".repeat(6000);
        let fake_path = Path::new("/tmp/test_corpus.txt");
        let tokens = tokenize_with(&tokenizer, &long_text, Some(100), fake_path).unwrap();

        assert_eq!(
            tokens.len(),
            100,
            "--max-tokens cap must still apply after uncap"
        );
    }

    // -- #923: load_cfg_for_q4 is fail-closed on a missing config.json ----
    //
    // This binary was already fail-closed before the fix (unlike the other
    // three loaders named in the issue), just via its own inline check
    // duplicating the same policy. It now goes through the single shared
    // `Qwen35Config::from_model_dir` helper instead of its own copy.
    // Reverting this call site back to the inline duplicate makes this test
    // fail only if that duplicate ever silently regresses to a preset --
    // the real value here is deleting the duplicate, verified by keeping
    // the observable behavior (error on missing config.json) pinned.

    #[test]
    fn load_cfg_for_q4_errors_on_missing_config_json() {
        let dir = std::env::temp_dir().join(format!(
            "lattice_eval_ppl_q4_no_config_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("test setup: create model dir");
        // Deliberately no config.json.

        let result = load_cfg_for_q4(&dir);
        assert!(
            result.is_err(),
            "a Q4 dir with no config.json must be a hard error, not a guessed preset"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // -- #1327: --metal-model-dir argument parsing and mode validation ----
    //
    // `parse_args` is the pure, side-effect-free extraction of the CLI's
    // parsing + mode-validation rules (previously inline in `main`), added
    // specifically so these rules are testable without a model checkpoint.

    fn argv(pieces: &[&str]) -> Vec<String> {
        std::iter::once("eval_perplexity".to_string())
            .chain(pieces.iter().map(std::string::ToString::to_string))
            .collect()
    }

    fn expect_args(pieces: &[&str]) -> ParsedArgs {
        match parse_args(&argv(pieces)) {
            Ok(ArgsOutcome::Args(p)) => *p,
            Ok(ArgsOutcome::Help) => panic!("expected Args, got Help for {pieces:?}"),
            Err(e) => panic!("expected Ok(Args) for {pieces:?}, got Err({e:?})"),
        }
    }

    fn expect_err(pieces: &[&str]) -> String {
        match parse_args(&argv(pieces)) {
            Err(e) => e,
            Ok(outcome) => panic!("expected Err for {pieces:?}, got Ok({outcome:?})"),
        }
    }

    #[test]
    fn metal_model_dir_parses_and_reaches_its_own_mode() {
        let p = expect_args(&[
            "--metal-model-dir",
            "/tmp/does-not-need-to-exist",
            "--corpus-file",
            "/tmp/corpus.txt",
        ]);
        assert_eq!(
            p.metal_model_dir,
            Some(PathBuf::from("/tmp/does-not-need-to-exist"))
        );
        assert_eq!(
            p.model_dir, None,
            "--metal-model-dir must not set model_dir"
        );
        assert_eq!(p.q4_dir, None);
        assert_eq!(p.quarot_q4_dir, None);
        assert_eq!(
            p.tokenizer_dir, None,
            "--metal-model-dir carries its own tokenizer; --tokenizer-dir was never passed"
        );
    }

    #[test]
    fn metal_model_dir_does_not_require_tokenizer_dir() {
        // Unlike --q4-dir / --quarot-q4-dir, --metal-model-dir loads its
        // tokenizer from the checkpoint directory itself (same as
        // --model-dir), so omitting --tokenizer-dir must be valid.
        expect_args(&["--metal-model-dir", "/tmp/m", "--corpus-file", "/tmp/c.txt"]);
    }

    #[test]
    fn metal_model_dir_accepts_random_lora_rank() {
        // --random-lora-rank requires "Metal mode"; --metal-model-dir builds
        // a MetalQwen35State exactly like the Q4 modes do, via the same
        // load_lora_adapter API, so it must count as Metal mode too.
        let p = expect_args(&[
            "--metal-model-dir",
            "/tmp/m",
            "--corpus-file",
            "/tmp/c.txt",
            "--random-lora-rank",
            "8",
        ]);
        assert_eq!(p.random_lora_rank, Some(8));
    }

    // -- pre-existing rules: still rejected, original message text --------

    #[test]
    fn model_dir_and_q4_dir_still_mutually_exclusive_with_original_message() {
        let e = expect_err(&[
            "--model-dir",
            "/tmp/m",
            "--q4-dir",
            "/tmp/q",
            "--tokenizer-dir",
            "/tmp/t",
            "--corpus-file",
            "/tmp/c.txt",
        ]);
        assert_eq!(
            e, "--model-dir is mutually exclusive with --q4-dir / --quarot-q4-dir",
            "pre-existing rule's message must be byte-for-byte unchanged"
        );
    }

    #[test]
    fn q4_dir_without_tokenizer_dir_still_rejected_with_original_message() {
        let e = expect_err(&["--q4-dir", "/tmp/q", "--corpus-file", "/tmp/c.txt"]);
        assert_eq!(
            e, "--tokenizer-dir is required when using --q4-dir or --quarot-q4-dir",
            "pre-existing rule's message must be byte-for-byte unchanged"
        );
    }

    #[test]
    fn no_mode_flag_still_rejected() {
        let e = expect_err(&["--corpus-file", "/tmp/c.txt"]);
        assert!(
            e.contains("--metal-model-dir"),
            "the 'one of ...' message must enumerate the new mode too: {e:?}"
        );
        assert!(
            e.contains("--model-dir") && e.contains("--q4-dir") && e.contains("--quarot-q4-dir")
        );
    }

    #[test]
    fn random_lora_rank_without_any_metal_mode_still_rejected() {
        let e = expect_err(&[
            "--model-dir",
            "/tmp/m",
            "--corpus-file",
            "/tmp/c.txt",
            "--random-lora-rank",
            "4",
        ]);
        assert!(
            e.contains("--random-lora-rank"),
            "message must still name the offending flag: {e:?}"
        );
        assert!(
            e.contains("--metal-model-dir"),
            "message must be widened to mention the new Metal mode: {e:?}"
        );
    }

    // -- new rules: rejected, message names the actual conflict -----------

    #[test]
    fn metal_model_dir_and_model_dir_are_mutually_exclusive() {
        let e = expect_err(&[
            "--metal-model-dir",
            "/tmp/m",
            "--model-dir",
            "/tmp/m2",
            "--corpus-file",
            "/tmp/c.txt",
        ]);
        assert_eq!(
            e,
            "--model-dir is mutually exclusive with --metal-model-dir"
        );
    }

    #[test]
    fn metal_model_dir_and_q4_dir_are_mutually_exclusive() {
        let e = expect_err(&[
            "--metal-model-dir",
            "/tmp/m",
            "--q4-dir",
            "/tmp/q",
            "--tokenizer-dir",
            "/tmp/t",
            "--corpus-file",
            "/tmp/c.txt",
        ]);
        assert_eq!(
            e,
            "--metal-model-dir is mutually exclusive with --q4-dir / --quarot-q4-dir"
        );
    }

    #[test]
    fn metal_model_dir_and_quarot_q4_dir_are_mutually_exclusive() {
        let e = expect_err(&[
            "--metal-model-dir",
            "/tmp/m",
            "--quarot-q4-dir",
            "/tmp/q",
            "--tokenizer-dir",
            "/tmp/t",
            "--corpus-file",
            "/tmp/c.txt",
        ]);
        assert_eq!(
            e,
            "--metal-model-dir is mutually exclusive with --q4-dir / --quarot-q4-dir"
        );
    }

    #[test]
    fn metal_model_dir_max_cache_len_below_window_is_rejected() {
        let e = expect_err(&[
            "--metal-model-dir",
            "/tmp/m",
            "--corpus-file",
            "/tmp/c.txt",
            "--window",
            "1024",
            "--max-cache-len",
            "512",
        ]);
        assert!(
            e.contains("--max-cache-len") && e.contains("--window"),
            "message must name the actual conflict: {e:?}"
        );
    }
}
