//! Canonical model-directory format detector (ADR-080 amendment, #829).
//!
//! **Stability**: unstable, internal-binaries-only. This module exists so
//! internal binaries in `src/bin/` -- including the three serving/chat
//! binaries (`lattice`, `lattice_serve`, `chat_metal`) and the three
//! benchmark binaries (`bench_decode_ab`, `bench_decode_slopefit`,
//! `bench_logit_dump`) -- which are separate Cargo binary targets and
//! therefore cannot see one another's `pub(crate)` items -- can share a
//! single model-directory format decision instead of each carrying its own
//! copy. It is public but unstable and internal-use: `pub mod model_format` is
//! externally nameable, but its compatibility follows the crate's
//! Experimental stability policy rather than semver-relevant deprecation
//! cycles, and it may change shape at any time; downstream consumers of this
//! crate should go through [`crate::model`] and [`crate::forward`] instead.
//! See the crate-level `Stability tier: Experimental` note in
//! `lib.rs` and `docs/adr/ADR-080-consolidation-duplicated-contracts.md`
//! (implementation record amendment) for the full rationale.
//!
//! Before this module existed, the same detection logic (and the same
//! precedence rule -- a `model.safetensors`/`model.safetensors.index.json`
//! file always wins over a `.q4` tensor file, and an unreadable or
//! non-matching directory fails closed to [`ModelFormat::Unknown`](crate::model_format::ModelFormat::Unknown)) was
//! implemented independently in `bin/lattice.rs` (`backend::detect_format`,
//! enum-valued, with its own unit tests), `bin/lattice_serve.rs`
//! (`detect_q4`, bool-valued), and `bin/chat_metal.rs` (`is_q4_dir`,
//! bool-valued, plus a fourth, partial decision site that only re-checked
//! safetensors absence inside the non-Q4 branch). All three binaries now
//! match on this single [`ModelFormat`](crate::model_format::ModelFormat) enum; none re-checks sentinel files
//! after the match.

use std::path::Path;

/// The on-disk format of a model directory, decided before any tensor I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelFormat {
    /// `model.safetensors` or `model.safetensors.index.json` present.
    Safetensors,
    /// No safetensors file, but at least one `*.q4` tensor file present
    /// (the output of `quantize_q4`).
    Q4,
    /// Neither a safetensors file nor a `.q4` file was found.
    Unknown,
}

/// Detect whether `dir` is a safetensors model directory, a native Q4
/// quantized directory, or neither.
///
/// A directory is Q4 when it has no safetensors file and contains at least
/// one file whose name ends in `.q4`. Safetensors presence always wins over
/// `.q4` presence (a directory containing both resolves to `Safetensors`),
/// and an unreadable directory (`read_dir` failure) or one matching neither
/// pattern resolves to `Unknown` -- fail-closed, never a silent guess.
pub fn detect_format(dir: &Path) -> ModelFormat {
    if dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists() {
        return ModelFormat::Safetensors;
    }
    let has_q4_file = std::fs::read_dir(dir)
        .ok()
        .and_then(|mut entries| {
            entries.find(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|e| e.file_name().to_str().map(|n| n.ends_with(".q4")))
                    .unwrap_or(false)
            })
        })
        .is_some();
    if has_q4_file {
        ModelFormat::Q4
    } else {
        ModelFormat::Unknown
    }
}

/// The model family a serving binary routes a checkpoint directory to.
///
/// Decided from the directory's `config.json`, not from tensor files: a Gemma
/// 4 checkpoint and a Qwen3.5 checkpoint are both `model.safetensors`
/// directories, so [`detect_format`] cannot tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelFamily {
    /// Qwen3.5. Also the family of every directory that is not positively
    /// identified as another one, which is what `lattice serve` has always
    /// done: the Qwen loader reports whatever is wrong with such a directory.
    Qwen35,
    /// Gemma 4, positively identified by a top-level `model_type` of
    /// `gemma4` in `config.json`. Whether the checkpoint is a supported
    /// Gemma 4 configuration is the Gemma loader's decision, not this one's.
    Gemma4,
}

impl ModelFamily {
    /// The spelling used in route markers and log lines.
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen35",
            Self::Gemma4 => "gemma4",
        }
    }
}

/// Top-level `config.json` `model_type` that identifies a Gemma 4 checkpoint.
/// The nested `text_config`, `audio_config` and `vision_config` objects carry
/// their own `model_type` values, which are not read here.
const GEMMA4_MODEL_TYPE: &str = "gemma4";

/// Decide which family `dir` belongs to from its `config.json`.
///
/// Returns [`ModelFamily::Gemma4`] only when the file is readable JSON whose
/// top-level `model_type` is `gemma4`. A missing, unreadable, oversized or
/// malformed `config.json`, and any other `model_type`, resolve to
/// [`ModelFamily::Qwen35`], so a directory the Gemma route does not claim is
/// handled exactly as it was before this detector existed.
pub fn detect_family(dir: &Path) -> ModelFamily {
    let is_gemma4 = crate::model::config_file::read_config_json_bounded(
        &dir.join("config.json"),
        "config.json",
    )
    .ok()
    .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
    .and_then(|root| {
        root.get("model_type")
            .and_then(serde_json::Value::as_str)
            .map(|model_type| model_type == GEMMA4_MODEL_TYPE)
    })
    .unwrap_or(false);
    if is_gemma4 {
        ModelFamily::Gemma4
    } else {
        ModelFamily::Qwen35
    }
}

/// Error message shown when a Q4 directory is passed to a binary that was
/// built without the `metal-gpu` feature. Q4 inference only runs on the
/// Metal GPU forward pass; there is no CPU fallback for `.q4` tensors, so
/// this is a hard, fail-closed error rather than a silent degrade.
pub fn metal_gpu_required_message(dir: &Path) -> String {
    format!(
        "model directory '{}' is a native Q4 quantized checkpoint, which requires \
         the Metal GPU forward pass. This binary was built without the `metal-gpu` \
         feature. Rebuild with `--features \"f16 metal-gpu\"` (macOS only), or point \
         --model at a safetensors directory instead.",
        dir.display()
    )
}

/// Error message for a directory that is neither safetensors nor Q4.
pub fn unrecognized_format_message(dir: &Path) -> String {
    format!(
        "'{}' is not a recognized model directory: no model.safetensors, \
         model.safetensors.index.json, or *.q4 tensor files were found",
        dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tempdir(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "lattice-model-format-test-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("create tempdir");
        dir
    }

    #[test]
    fn detect_format_safetensors_file() {
        let dir = tempdir("safetensors-file");
        fs::write(dir.join("model.safetensors"), b"stub").unwrap();
        assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_format_safetensors_index_only() {
        let dir = tempdir("safetensors-index");
        fs::write(dir.join("model.safetensors.index.json"), b"{}").unwrap();
        assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_format_q4_dir() {
        let dir = tempdir("q4");
        fs::write(dir.join("model_layers_0_weight.q4"), b"stub").unwrap();
        fs::write(dir.join("config.json"), b"{}").unwrap();
        assert_eq!(detect_format(&dir), ModelFormat::Q4);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_format_prefers_safetensors_over_q4_files() {
        // A directory that (unusually) has both a safetensors file and a
        // stray .q4 file must resolve as Safetensors — the safetensors
        // loader path is untouched and takes priority.
        let dir = tempdir("mixed");
        fs::write(dir.join("model.safetensors"), b"stub").unwrap();
        fs::write(dir.join("leftover.q4"), b"stub").unwrap();
        assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_format_prefers_safetensors_index_over_q4_files() {
        // A directory that has an index.json (sharded safetensors) and a
        // stray .q4 file must also resolve as Safetensors. This is the
        // precedence the three benchmark binaries' old inline heuristic got
        // wrong: `!model.safetensors.exists() && has_q4_file` never checked
        // index.json, so a directory with an index file plus .q4 files used
        // to route those binaries to the Q4 loader. The canonical detector
        // checks both sentinel files before falling through to .q4, so this
        // case now resolves to Safetensors everywhere.
        let dir = tempdir("mixed-index");
        fs::write(dir.join("model.safetensors.index.json"), b"{}").unwrap();
        fs::write(dir.join("leftover.q4"), b"stub").unwrap();
        assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_format_empty_dir_is_unknown() {
        let dir = tempdir("empty");
        assert_eq!(detect_format(&dir), ModelFormat::Unknown);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_format_unrelated_files_is_unknown() {
        let dir = tempdir("unrelated");
        fs::write(dir.join("readme.txt"), b"hello").unwrap();
        fs::write(dir.join("config.json"), b"{}").unwrap();
        assert_eq!(detect_format(&dir), ModelFormat::Unknown);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_format_unreadable_dir_is_unknown() {
        // A directory that does not exist at all: `read_dir` fails, and the
        // fail-closed path (`.ok()` -> `None` -> `is_some() == false`) must
        // resolve to `Unknown` rather than panicking or defaulting to a
        // recognized format.
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "lattice-model-format-test-does-not-exist-{}",
            std::process::id()
        ));
        assert_eq!(detect_format(&dir), ModelFormat::Unknown);
    }

    #[test]
    fn metal_gpu_required_message_mentions_rebuild_flags() {
        let msg = metal_gpu_required_message(Path::new("/tmp/some-q4-dir"));
        assert!(msg.contains("metal-gpu"));
        assert!(msg.contains("--features"));
    }

    #[test]
    fn unrecognized_format_message_mentions_expected_files() {
        let msg = unrecognized_format_message(Path::new("/tmp/bogus"));
        assert!(msg.contains("model.safetensors"));
        assert!(msg.contains(".q4"));
    }

    #[test]
    fn detect_family_names_gemma4_only_from_the_top_level_model_type() {
        let dir = tempdir("family-gemma4");
        fs::write(
            dir.join("config.json"),
            br#"{"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}"#,
        )
        .unwrap();
        assert_eq!(detect_family(&dir), ModelFamily::Gemma4);
        assert_eq!(ModelFamily::Gemma4.name(), "gemma4");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_family_ignores_nested_model_type_values() {
        // A Qwen3.5 vision-language config nests other `model_type` values;
        // only the top-level key decides.
        let dir = tempdir("family-nested");
        fs::write(
            dir.join("config.json"),
            br#"{"model_type": "qwen3_5", "text_config": {"model_type": "gemma4"}}"#,
        )
        .unwrap();
        assert_eq!(detect_family(&dir), ModelFamily::Qwen35);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_family_falls_back_to_qwen35_for_anything_unclaimed() {
        let missing = tempdir("family-missing");
        assert_eq!(detect_family(&missing), ModelFamily::Qwen35);
        let malformed = tempdir("family-malformed");
        fs::write(malformed.join("config.json"), b"{not json").unwrap();
        assert_eq!(detect_family(&malformed), ModelFamily::Qwen35);
        let other = tempdir("family-other");
        fs::write(other.join("config.json"), br#"{"model_type": 4}"#).unwrap();
        assert_eq!(detect_family(&other), ModelFamily::Qwen35);
        let absent = std::env::temp_dir().join(format!(
            "lattice-model-format-test-family-absent-{}",
            std::process::id()
        ));
        assert_eq!(detect_family(&absent), ModelFamily::Qwen35);
        for dir in [missing, malformed, other] {
            fs::remove_dir_all(&dir).ok();
        }
    }

    // Fail-closed without metal-gpu: a Q4 directory must never silently
    // fall back to the CPU safetensors loader. This test only compiles
    // (and only means anything) when the binary is built WITHOUT the
    // metal-gpu feature — it asserts that the code path this binary
    // would take for a Q4 directory is the explicit error message above,
    // never `Qwen35Model::from_safetensors`.
    #[cfg(not(feature = "metal-gpu"))]
    #[test]
    fn q4_dir_without_metal_gpu_feature_fails_closed() {
        let dir = tempdir("q4-no-metal");
        fs::write(dir.join("model_layers_0_weight.q4"), b"stub").unwrap();
        assert_eq!(detect_format(&dir), ModelFormat::Q4);
        // Scope: detection + message content only. This proves a Q4 dir
        // classifies as `ModelFormat::Q4` (so a caller's `match` takes the
        // fail-closed branch, never a safetensors load) and that the error
        // names the rebuild flags. It does not drive any binary's `main`
        // itself — those exit the process, which a unit test cannot cross.
        let msg = metal_gpu_required_message(&dir);
        assert!(msg.contains("metal-gpu"));
        fs::remove_dir_all(&dir).ok();
    }
}
