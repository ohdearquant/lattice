//! QuaRot offline converter: BF16/F16/F32 SafeTensors → Hadamard-rotated Q4_0 `.q4` files.
//!
//! Step 3c-5 of ADR-044. Thin argparse wrapper over
//! [`lattice_inference::quant::quarot::convert::convert_quarot_qwen35`],
//! which owns the actual rotation + fusion + materialize_lm_head +
//! forward-equivalence-refuse-on-fail pipeline.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin quantize_quarot -- \
//!   --model-dir ~/.lattice/models/qwen3.5-0.8b \
//!   --output-dir ~/.lattice/models/qwen3.5-0.8b-quarot-q4 \
//!   --seed 0xC0FFEE
//! ```
//!
//! Flags:
//! - `--model-dir <PATH>`: input directory containing `config.json` and
//!   safetensors (single file or sharded — auto-detected).
//! - `--output-dir <PATH>`: target directory; created if absent. Output
//!   layout matches `bin/quantize_q4` (per-tensor `.q4` / `.f16` files +
//!   `quantize_index.json` + mutated `config.json`).
//! - `--seed <U64>`: residual-stream Hadamard rotation seed. Accepts
//!   decimal or `0x...` hex. Required (no default — converted artifacts
//!   are not interchangeable across seeds, so the choice should be
//!   recorded explicitly).
//! - `--tolerance <F64>`: forward-equivalence tolerance. Default `1e-5`
//!   per ADR-044 §"Step 3c contract".
//! - `--num-probe-tokens <USIZE>`: chain-probe sample size. Default `4`.
//! - `--dry-run`: run the pipeline + gate but skip every disk write.
//!   Useful for CI sanity passes against a real safetensors source.
//!
//! Exit codes:
//! - `0`: conversion succeeded; output dir is complete (or dry-run
//!   verified the pipeline).
//! - `1`: error (refuse-on-fail, missing file, parse failure, etc.).
//!   Standard error contains the diagnostic; output dir is left empty
//!   when the forward-equivalence gate refused.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use lattice_inference::quant::quarot::convert::{
    ConversionOptions, ConversionReport, convert_quarot_qwen35,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut seed: Option<u64> = None;
    let mut tolerance: Option<f64> = None;
    let mut num_probe_tokens: Option<usize> = None;
    let mut dry_run = false;

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
            "--output-dir" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--output-dir requires an argument");
                };
                output_dir = Some(PathBuf::from(v));
            }
            "--seed" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--seed requires an argument");
                };
                seed = Some(match parse_u64_flex(v) {
                    Ok(n) => n,
                    Err(msg) => return usage(&format!("--seed: {msg}")),
                });
            }
            "--tolerance" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--tolerance requires an argument");
                };
                tolerance = Some(match v.parse::<f64>() {
                    Ok(n) => n,
                    Err(e) => return usage(&format!("--tolerance: invalid f64: {e}")),
                });
            }
            "--num-probe-tokens" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage("--num-probe-tokens requires an argument");
                };
                num_probe_tokens = Some(match v.parse::<usize>() {
                    Ok(n) => n,
                    Err(e) => return usage(&format!("--num-probe-tokens: invalid usize: {e}")),
                });
            }
            "--dry-run" => {
                dry_run = true;
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
    let Some(output_dir) = output_dir else {
        return usage("--output-dir is required");
    };
    let Some(seed) = seed else {
        return usage("--seed is required (rotation determinism)");
    };

    let opts = ConversionOptions {
        rotation_seed: seed,
        tolerance: tolerance.unwrap_or(1e-5),
        num_probe_tokens: num_probe_tokens.unwrap_or(4),
        dry_run,
    };

    eprintln!("=== quantize_quarot: QuaRot Q4_0 converter ===");
    eprintln!("Model dir:   {}", model_dir.display());
    eprintln!("Output dir:  {}", output_dir.display());
    eprintln!("Seed:        0x{seed:016x}");
    eprintln!("Tolerance:   {}", opts.tolerance);
    eprintln!("Probe toks:  {}", opts.num_probe_tokens);
    eprintln!(
        "Mode:        {}",
        if opts.dry_run {
            "DRY RUN (no files written)"
        } else {
            "WRITE"
        }
    );
    eprintln!();

    let start = Instant::now();
    let report = match convert_quarot_qwen35(&model_dir, &output_dir, &opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERROR: {e}");
            return ExitCode::FAILURE;
        }
    };
    let elapsed = start.elapsed();

    print_report(&report, elapsed.as_secs_f64());
    ExitCode::SUCCESS
}

fn print_report(report: &ConversionReport, elapsed_secs: f64) {
    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("Tied input:        {}", report.was_tied);
    eprintln!("Quantized (Q4_0):  {}", report.planned_quantized);
    eprintln!("Kept (F16):        {}", report.kept_f16);
    eprintln!(
        "Input bytes:       {:.2} MB",
        report.total_bytes_in as f64 / 1_048_576.0
    );
    eprintln!(
        "Output bytes:      {:.2} MB",
        report.total_bytes_out as f64 / 1_048_576.0
    );
    if report.total_bytes_in > 0 {
        let ratio = report.total_bytes_out as f64 / report.total_bytes_in as f64;
        eprintln!(
            "Compression:       {:.2}x ({:.1}%)",
            1.0 / ratio.max(f64::MIN_POSITIVE),
            ratio * 100.0
        );
    }
    eprintln!(
        "Forward-equiv:     max_abs={:.3e}, mean_abs={:.3e} (tol={:.0e}, probes={:?})",
        report.forward_equivalence.max_abs_error,
        report.forward_equivalence.mean_abs_error,
        report.forward_equivalence.tolerance,
        report.forward_equivalence.probe_tokens
    );
    eprintln!("Wall time:         {elapsed_secs:.1}s");
}

fn usage(msg: &str) -> ExitCode {
    eprintln!("ERROR: {msg}\n");
    eprintln!("{USAGE}");
    ExitCode::FAILURE
}

const USAGE: &str = "\
usage: quantize_quarot --model-dir <PATH> --output-dir <PATH> --seed <U64> [OPTIONS]

QuaRot Q4_0 converter for Qwen3.5 (ADR-044 step 3c).

required:
  --model-dir <PATH>         Input directory with config.json + safetensors.
  --output-dir <PATH>        Output directory (created if absent).
  --seed <U64>               Hadamard rotation seed (decimal or 0x... hex).

options:
  --tolerance <F64>          Forward-equivalence tolerance. Default 1e-5.
  --num-probe-tokens <USIZE> Chain-probe sample size. Default 4.
  --dry-run                  Run pipeline + gate; skip disk writes.
  -h, --help                 Print this help and exit.

The converter refuses to write any output if the forward-equivalence
gate fails (delta > tolerance) — this protects against silently shipping
a model whose logits diverged during conversion.
";

fn parse_u64_flex(s: &str) -> Result<u64, String> {
    let trimmed = s.trim();
    if let Some(rest) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(rest, 16).map_err(|e| format!("invalid hex: {e}"))
    } else {
        trimmed
            .parse::<u64>()
            .map_err(|e| format!("invalid decimal: {e}"))
    }
}
