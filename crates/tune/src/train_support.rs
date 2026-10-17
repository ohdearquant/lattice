//! Support shared by the gradient-training executables and the layer-23
//! golden bridge test.
//!
//! It owns JSONL sample loading, raw argument lookup, path defaults, and the
//! fail-closed trust-but-verify comparison. Each binary owns its typed options
//! and usage contract. See `docs/design.md` for the CLI data and TBV flow.
//!
//! Library module, not a binary-embedded one: `full_driver` is compiled once
//! here and linked by every consumer, instead of being recompiled per binary.

use std::io::BufRead;
use std::path::{Path, PathBuf};

use lattice_inference::tokenizer::Tokenizer;

pub mod full_driver;

/// One training sample with prompt/completion split.
pub struct Sample {
    pub tokens: Vec<u32>,
    pub completion_start: usize,
}

/// Load up to `max_samples` valid `prompt`/`completion` JSONL rows.
///
/// Each sample stores full token IDs and the completion boundary; invalid or
/// over-length rows are skipped. See [`docs/design.md`](../../../docs/design.md#shared-jsonl-samples) for acceptance rules.
pub fn load_jsonl(
    path: &Path,
    tokenizer: &dyn Tokenizer,
    eos_token_id: u32,
    seq_len: usize,
    max_samples: usize,
) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        if out.len() >= max_samples {
            break;
        }
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let prompt = v["prompt"].as_str().unwrap_or("").to_string();
        let completion = v["completion"].as_str().unwrap_or("").to_string();
        if prompt.is_empty() || completion.is_empty() {
            continue;
        }
        let prompt_tok = tokenizer.tokenize(&prompt);
        let completion_tok = tokenizer.tokenize(&completion);
        let mut tokens = prompt_tok.input_ids[..prompt_tok.real_length].to_vec();
        let completion_start = tokens.len();
        tokens.extend_from_slice(&completion_tok.input_ids[..completion_tok.real_length]);
        tokens.push(eos_token_id);
        let total = tokens.len();
        if total < 2 || total > seq_len {
            continue;
        }
        if completion_start == 0 || completion_tok.real_length == 0 {
            continue;
        }
        out.push(Sample {
            tokens,
            completion_start,
        });
    }
    Ok(out)
}

/// Default model directory: `$HOME/.lattice/models/qwen3.5-0.8b`, falling
/// back to `.` if `$HOME` is unset.
pub fn default_model_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".lattice")
        .join("models")
        .join("qwen3.5-0.8b")
}

/// Default dataset directory: `data/lora-train`.
pub fn default_data_dir() -> PathBuf {
    PathBuf::from("data/lora-train")
}

/// Provides the shared permissive lookup semantics for gradient-training arguments.
/// See [`docs/design.md`](../../../docs/design.md#argview) for compatibility and validation boundaries.
pub struct ArgView<'a> {
    args: &'a [String],
}

impl<'a> ArgView<'a> {
    pub fn new(args: &'a [String]) -> Self {
        Self { args }
    }

    /// The value following the first occurrence of `flag`, or `None` if the
    /// flag is absent or has no following argument.
    pub fn arg(&self, flag: &str) -> Option<String> {
        self.args
            .iter()
            .position(|a| a == flag)
            .and_then(|i| self.args.get(i + 1))
            .cloned()
    }

    /// Whether `flag` appears anywhere in argv.
    pub fn flag(&self, flag: &str) -> bool {
        self.args.iter().any(|a| a == flag)
    }
}

/// The single owner of the TBV (trust-but-verify) tolerance: the masked NLL
/// from a bin's assembled forward chain must agree with the real model's own
/// `compute_token_nlls` within this absolute difference, or the run must
/// fail closed rather than merely print a diagnostic.
pub const TBV_MAX_ABS_DIFF: f32 = 1e-2;

/// A passed TBV check: the two NLLs compared and their absolute difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TbvObservation {
    pub reference_nll: f32,
    pub candidate_nll: f32,
    pub diff: f32,
}

/// Compare candidate and reference masked NLL values and return their observation.
///
/// Fails closed for non-finite values or a difference above [`TBV_MAX_ABS_DIFF`].
/// See [`docs/design.md`](../../../docs/design.md#verify_tbv) for gate behavior and diagnostics.
pub fn verify_tbv(
    context: &str,
    reference_nll: f32,
    candidate_nll: f32,
) -> Result<TbvObservation, String> {
    if !reference_nll.is_finite() {
        return Err(format!(
            "TBV failed ({context}): reference NLL is not finite ({reference_nll})"
        ));
    }
    if !candidate_nll.is_finite() {
        return Err(format!(
            "TBV failed ({context}): candidate NLL is not finite ({candidate_nll})"
        ));
    }
    let diff = (reference_nll - candidate_nll).abs();
    if !diff.is_finite() {
        return Err(format!(
            "TBV failed ({context}): |diff| is not finite ({diff})"
        ));
    }
    if diff > TBV_MAX_ABS_DIFF {
        return Err(format!(
            "TBV failed ({context}): diverges from real model by {diff:.3e} (> {TBV_MAX_ABS_DIFF:.0e})"
        ));
    }
    Ok(TbvObservation {
        reference_nll,
        candidate_nll,
        diff,
    })
}

#[cfg(test)]
mod tests;
