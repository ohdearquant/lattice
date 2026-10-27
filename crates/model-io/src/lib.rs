//! Model-file input and output for Lattice inference crates.
//!
//! This crate holds the model cache directory ([`default_cache_dir`]), checksum-verified model
//! downloads ([`download::ensure_model_files`]) and the by-value reader for boolean `LATTICE_*`
//! environment switches ([`env_switch_enabled`]). Network access is compiled in only with the
//! `download` feature and is refused at run time when `LATTICE_OFFLINE` is set.
//! It also owns the shared [`multimodal::MultimodalInput`] prompt record.
//!
//! `lattice-inference` re-exports [`download::ensure_model_files`] at its original path, so
//! `lattice_inference::download::ensure_model_files` and `lattice_model_io::download::ensure_model_files`
//! name the same function.

use lattice_inference_core::InferenceError;

/// Model-file ensure/download flow, canonical model names and checksum verification.
pub mod download;
/// Bounded image decoding and shared vision errors.
pub mod image;
/// Shared multimodal prompt layout.
pub mod multimodal;

/// Checkpoint-directory resolver for checkpoint-gated tests in the model crates.
/// Not part of the supported API.
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod test_support;

use std::path::PathBuf;

/// Default model cache directory.
pub fn default_cache_dir() -> Result<PathBuf, InferenceError> {
    if let Ok(path) = std::env::var("LATTICE_MODEL_CACHE") {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var("HOME").map_err(|_| {
        InferenceError::ModelNotFound(
            "unable to determine cache directory; HOME and LATTICE_MODEL_CACHE are unset".into(),
        )
    })?;
    Ok(PathBuf::from(home).join(".lattice").join("models"))
}

/// Reads a boolean engine switch from the environment, by VALUE.
///
/// These switches used to be selected by presence alone, so `LATTICE_MTP_BATCH=0`
/// turned the batch verifier on while every document and comment spells the flag
/// `=1`. An operator writing `=0` to mean "run the default arm" measured the
/// non-default arm twice and saw a clean-looking negative.
///
/// A value spelling "off" now disables the switch: empty, `0`, `false`, `no` or
/// `off`, in any case, with surrounding whitespace ignored. Every other value
/// enables it, so `=1`, `=true` and `=yes` all keep working, and so does any value
/// a caller was already passing to mean "on". An absent variable is off, and a
/// value that is not valid Unicode cannot be one of the off spellings, so it
/// enables, which is what presence-checking did.
///
/// One `var_os` lookup and a few ASCII comparisons against short literals: no
/// allocation, because one of these sits inside a per-round decode loop.
///
/// Not gated to the Metal build: the offline-download gate (`download.rs`) and the
/// CPU-fallback switch (`lattice-inference`'s `model/qwen.rs`) call this from code paths that compile on
/// every platform and feature set, so it has to be available everywhere too.
#[inline]
pub fn env_switch_enabled(name: &str) -> bool {
    match std::env::var_os(name) {
        None => false,
        Some(raw) => match raw.to_str() {
            Some(text) => switch_value_enabled(Some(text)),
            None => true,
        },
    }
}

/// The value half of [`env_switch_enabled`], separated so it is testable without
/// mutating the process environment, which no test can do without racing every
/// other test in the binary.
fn switch_value_enabled(value: Option<&str>) -> bool {
    const OFF: [&str; 4] = ["0", "false", "no", "off"];
    match value {
        None => false,
        Some(raw) => {
            let text = raw.trim();
            !(text.is_empty() || OFF.iter().any(|off| text.eq_ignore_ascii_case(off)))
        }
    }
}

#[cfg(test)]
mod env_switch_tests {
    use super::switch_value_enabled;

    #[test]
    fn absent_is_off() {
        assert!(!switch_value_enabled(None));
    }

    #[test]
    fn documented_on_spellings_enable() {
        for raw in ["1", "true", "TRUE", "yes", "on", " 1 ", "2", "batch"] {
            assert!(switch_value_enabled(Some(raw)), "{raw:?} should enable");
        }
    }

    #[test]
    fn off_spellings_disable_which_presence_checking_could_not() {
        for raw in ["0", "false", "FALSE", "no", "off", "", "  ", " 0 "] {
            assert!(!switch_value_enabled(Some(raw)), "{raw:?} should disable");
        }
    }
}
