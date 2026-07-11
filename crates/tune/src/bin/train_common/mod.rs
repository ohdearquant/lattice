//! Binary-private support shared by `train_grad`, `train_grad_full`, and
//! `train_grad_layer23` (issue #845): the loader, the CLI arg-lookup
//! semantics, and the fail-closed TBV (trust-but-verify) check.
//!
//! Deliberately narrow. Each bin keeps its own typed config and usage text —
//! defaults and supported flags genuinely differ between bins (see the flag
//! table in issue #845) — so this module holds only the pieces that were
//! byte-identical across all three: `Sample`/`load_jsonl`, the
//! `parse_arg`/`parse_flag` lookup semantics (as `ArgView`), the shared
//! path defaults, and `verify_tbv`.

use std::io::BufRead;
use std::path::{Path, PathBuf};

use lattice_inference::tokenizer::Tokenizer;

/// One training sample with prompt/completion split.
pub struct Sample {
    pub tokens: Vec<u32>,
    pub completion_start: usize,
}

/// Load `prompt`/`completion` JSONL rows, tokenizing full = prompt+completion
/// and recording where the completion starts. Rows that are empty, missing a
/// field, or whose tokenized length falls outside `(1, seq_len]` (or whose
/// completion is empty after tokenization) are skipped. Stops once
/// `max_samples` rows have been collected.
pub fn load_jsonl(
    path: &Path,
    tokenizer: &dyn Tokenizer,
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
        let mut full = prompt.clone();
        full.push_str(&completion);
        let prompt_tok = tokenizer.tokenize(&prompt);
        let full_tok = tokenizer.tokenize(&full);
        let prompt_ids: Vec<u32> = prompt_tok.input_ids[..prompt_tok.real_length].to_vec();
        let full_ids: Vec<u32> = full_tok.input_ids[..full_tok.real_length].to_vec();
        let total = full_ids.len();
        if total < 2 || total > seq_len {
            continue;
        }
        let completion_start = prompt_ids.len();
        if completion_start == 0 || completion_start >= total {
            continue;
        }
        out.push(Sample {
            tokens: full_ids,
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

/// A read-only view over `std::env::args()`-style argv, preserving the exact
/// lookup semantics every bin's inline `parse_arg`/`parse_flag` had: first
/// matching flag wins, a missing/invalid value is the caller's problem (this
/// type only ever returns the raw string or `None`), presence flags are
/// presence-only, and unknown flags are silently ignored. Parser strictness
/// is explicitly out of scope for this migration (issue #845 non-goals) —
/// this type must not add validation beyond what the two free functions did.
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

/// Compare a candidate (cached/assembled-chain) masked NLL against the
/// reference (real model `compute_token_nlls`) masked NLL. Fails closed:
/// non-finite reference, non-finite candidate, non-finite difference, or a
/// difference exceeding [`TBV_MAX_ABS_DIFF`] all return `Err` rather than
/// merely being printed. `context` is folded into the error message to
/// identify which check failed (e.g. "train_grad cache check (sample 0)").
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
mod tests {
    use super::*;

    #[test]
    fn arg_view_first_match_wins() {
        let args: Vec<String> = ["bin", "--x", "1", "--x", "2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let av = ArgView::new(&args);
        assert_eq!(av.arg("--x"), Some("1".to_string()));
    }

    #[test]
    fn arg_view_missing_flag_is_none() {
        let args: Vec<String> = ["bin"].iter().map(|s| s.to_string()).collect();
        let av = ArgView::new(&args);
        assert_eq!(av.arg("--x"), None);
    }

    #[test]
    fn arg_view_dangling_flag_is_none() {
        // Flag present but no following token — same as today's inline
        // parse_arg: `.get(i + 1)` returns None.
        let args: Vec<String> = ["bin", "--x"].iter().map(|s| s.to_string()).collect();
        let av = ArgView::new(&args);
        assert_eq!(av.arg("--x"), None);
    }

    #[test]
    fn arg_view_presence_flag() {
        let args: Vec<String> = ["bin", "--save"].iter().map(|s| s.to_string()).collect();
        let av = ArgView::new(&args);
        assert!(av.flag("--save"));
        assert!(!av.flag("--gradcheck"));
    }

    #[test]
    fn arg_view_unknown_flags_ignored() {
        let args: Vec<String> = ["bin", "--totally-unknown", "x"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let av = ArgView::new(&args);
        assert_eq!(av.arg("--model-dir"), None);
        assert!(!av.flag("--gradcheck"));
    }

    #[test]
    fn verify_tbv_passes_within_tolerance() {
        let obs = verify_tbv("ctx", 1.0, 1.0 + TBV_MAX_ABS_DIFF * 0.5).unwrap();
        assert!(obs.diff < TBV_MAX_ABS_DIFF);
    }

    #[test]
    fn verify_tbv_passes_exactly_at_boundary() {
        // diff > TBV_MAX_ABS_DIFF is the failure condition, so diff ==
        // TBV_MAX_ABS_DIFF (not strictly greater) must still pass. The pair
        // (TBV_MAX_ABS_DIFF, 0.0) subtracts to the constant EXACTLY in f32,
        // so this test sits on the boundary bit-for-bit; a `1.0` vs
        // `1.0 + TBV_MAX_ABS_DIFF` pair rounds to a diff slightly below the
        // constant and would not catch a `>=` mutation.
        let obs = verify_tbv("ctx", TBV_MAX_ABS_DIFF, 0.0).unwrap();
        assert!(obs.diff == TBV_MAX_ABS_DIFF);
    }

    #[test]
    fn verify_tbv_fails_just_above_boundary() {
        let err = verify_tbv("ctx", 1.0, 1.0 + TBV_MAX_ABS_DIFF + 1e-6).unwrap_err();
        assert!(err.contains("TBV failed"));
        assert!(err.contains("ctx"));
    }

    #[test]
    fn verify_tbv_fails_on_reference_nan() {
        let err = verify_tbv("ctx", f32::NAN, 1.0).unwrap_err();
        assert!(err.contains("reference NLL is not finite"));
    }

    #[test]
    fn verify_tbv_fails_on_candidate_nan() {
        let err = verify_tbv("ctx", 1.0, f32::NAN).unwrap_err();
        assert!(err.contains("candidate NLL is not finite"));
    }

    #[test]
    fn verify_tbv_fails_on_reference_inf() {
        let err = verify_tbv("ctx", f32::INFINITY, 1.0).unwrap_err();
        assert!(err.contains("reference NLL is not finite"));
    }

    #[test]
    fn verify_tbv_fails_on_candidate_inf() {
        let err = verify_tbv("ctx", 1.0, f32::NEG_INFINITY).unwrap_err();
        assert!(err.contains("candidate NLL is not finite"));
    }

    #[test]
    fn verify_tbv_fails_on_difference_overflow_to_inf() {
        // Both inputs are individually finite (so the reference/candidate
        // finite-guards pass), but their subtraction overflows f32 range —
        // f32::MAX - f32::MIN = 2 * f32::MAX -> +inf. Exercises the
        // dedicated diff-is-finite guard, distinct from the reference- and
        // candidate-finite guards above.
        let err = verify_tbv("ctx", f32::MAX, f32::MIN).unwrap_err();
        assert!(
            err.contains("|diff| is not finite"),
            "expected the diff-specific guard message, got: {err}"
        );
    }
}
