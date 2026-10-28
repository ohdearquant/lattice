//! Successful-load evidence for directory-backed BERT models.

use super::{BertModel, infer_config_from_safetensors, parse_config_json_str};
use lattice_inference_core::InferenceError;
use lattice_tokenizer::common::load_tokenizer_with_report;
use lattice_weights::SafetensorsFile;
use lattice_weights::consumed::{
    ConsumedBytes, ConsumedSafetensors, DigestRequest, WeightDigestMode,
};
use std::borrow::Cow;
use std::path::Path;

pub(super) fn config_text<'a>(
    dir: &Path,
    snapshot: Option<&'a [u8]>,
) -> Result<Option<Cow<'a, str>>, InferenceError> {
    if let Some(bytes) = snapshot {
        let text = std::str::from_utf8(bytes).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "stream did not contain valid UTF-8",
            )
        })?;
        return Ok(Some(Cow::Borrowed(text)));
    }
    let path = dir.join("config.json");
    if path.exists() {
        Ok(Some(Cow::Owned(std::fs::read_to_string(path)?)))
    } else {
        Ok(None)
    }
}

/// Inputs retained only after BERT weight assembly succeeds.
#[derive(Clone, Debug)]
pub struct BertLoadReport {
    config: ConsumedBytes,
    tokenizer: Vec<(String, ConsumedBytes)>,
    tokenizer_format: &'static str,
    weights: ConsumedSafetensors,
    tokenizer_settings: Vec<(String, lattice_tokenizer::common::TokenizerPolicyValue)>,
}

impl BertLoadReport {
    /// Exact consumed config-file digest and size.
    pub fn config(&self) -> &ConsumedBytes {
        &self.config
    }
    /// Ordered reads made by tokenizer parsing and effective-length resolution.
    pub fn tokenizer(&self) -> &[(String, ConsumedBytes)] {
        &self.tokenizer
    }
    /// Actual selected tokenizer parser, including fallback format.
    pub fn tokenizer_format(&self) -> &'static str {
        self.tokenizer_format
    }
    /// Resolved tokenizer selectors, including special-token defaults and length caps.
    pub fn tokenizer_settings(
        &self,
    ) -> &[(String, lattice_tokenizer::common::TokenizerPolicyValue)] {
        &self.tokenizer_settings
    }
    /// Evidence from the same parsed safetensors backing used for weight assembly.
    pub fn weights(&self) -> &ConsumedSafetensors {
        &self.weights
    }
}

impl BertModel {
    /// Load with Full consumed-weight evidence by default.
    pub fn from_directory_with_report(dir: &Path) -> Result<Self, InferenceError> {
        Self::from_directory_with_digest_mode(dir, WeightDigestMode::Full, &mut |_| {})
    }

    /// Load and retain consumed-input evidence in the explicitly selected mode.
    ///
    /// Header omits tensor-data digests and cannot detect header-preserving
    /// weight-value edits. Missing explicit config yields no authoritative report.
    /// The observer receives actual digest requests, independently of model reads.
    pub fn from_directory_with_digest_mode(
        dir: &Path,
        mode: WeightDigestMode,
        observe: &mut dyn FnMut(DigestRequest),
    ) -> Result<Self, InferenceError> {
        let (tokenizer, tokenizer_inputs) = load_tokenizer_with_report(dir)?;
        let model_path = dir.join("model.safetensors");
        if !model_path.exists() {
            let sharded = dir.join("model.safetensors.index.json");
            if sharded.exists() {
                return Err(InferenceError::UnsupportedModel(format!(
                    "sharded safetensors are not supported yet: {}",
                    sharded.display()
                )));
            }
            return Err(InferenceError::ModelNotFound(format!(
                "missing model.safetensors in {}",
                dir.display()
            )));
        }
        let (safetensors, weights) =
            SafetensorsFile::open_with_report(&model_path, "model.safetensors", mode, observe)?;
        let safetensors = Box::new(safetensors);
        let snapshot = tokenizer_inputs
            .inputs
            .iter()
            .rev()
            .find(|input| input.logical_identifier == "config.json")
            .map(|input| input.bytes.as_slice());
        let config_text = config_text(dir, snapshot)?;
        let config = match &config_text {
            Some(text) => parse_config_json_str(text)?,
            None => infer_config_from_safetensors(&safetensors)?,
        };
        let mut model = Self::assemble(config, tokenizer, safetensors)?;
        if let Some(text) = config_text {
            let config = ConsumedBytes::capture(text.as_bytes(), "config.json", observe);
            let tokenizer = tokenizer_inputs
                .inputs
                .into_iter()
                .enumerate()
                .map(|(index, input)| {
                    let name = format!("read.{index}/{}", input.logical_identifier);
                    let report = ConsumedBytes::capture(&input.bytes, &name, observe);
                    (name, report)
                })
                .collect();
            model.load_report = Some(BertLoadReport {
                config,
                tokenizer,
                tokenizer_format: tokenizer_inputs.format,
                weights,
                tokenizer_settings: tokenizer_inputs.settings,
            });
        }
        Ok(model)
    }

    /// Acquire a model through the ordinary cache/download path with selected evidence.
    pub fn from_pretrained_with_digest_mode(
        model_name: &str,
        mode: WeightDigestMode,
    ) -> Result<Self, InferenceError> {
        let cache_dir = lattice_model_io::default_cache_dir()?;
        let model_dir = lattice_model_io::download::ensure_model_files(model_name, &cache_dir)?;
        Self::from_directory_with_digest_mode(&model_dir, mode, &mut |_| {})
    }

    /// The successful directory-load report, or no authoritative input origins.
    ///
    /// Ordinary and preloaded constructors remain unidentified. This report is
    /// retained with the model; callers must resolve current output policy too.
    pub fn load_report(&self) -> Option<&BertLoadReport> {
        self.load_report.as_ref()
    }
}

#[cfg(test)]
mod config_snapshot_tests {
    use super::*;

    #[test]
    fn snapshot_is_used_even_if_the_file_changes_or_disappears() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, b"new").unwrap();
        assert_eq!(
            config_text(dir.path(), Some(b"old")).unwrap().as_deref(),
            Some("old")
        );
        assert_eq!(
            config_text(dir.path(), None).unwrap().as_deref(),
            Some("new")
        );
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            config_text(dir.path(), Some(b"old")).unwrap().as_deref(),
            Some("old")
        );
        assert!(config_text(dir.path(), None).unwrap().is_none());
    }

    #[test]
    fn invalid_utf8_snapshot_preserves_io_error_kind() {
        let dir = tempfile::tempdir().unwrap();
        let error = config_text(dir.path(), Some(b"\xff")).unwrap_err();
        assert!(
            matches!(error, InferenceError::Io(ref error) if error.kind() == std::io::ErrorKind::InvalidData)
        );
        std::fs::write(dir.path().join("config.json"), b"\xff").unwrap();
        let error = config_text(dir.path(), None).unwrap_err();
        assert!(
            matches!(error, InferenceError::Io(ref error) if error.kind() == std::io::ErrorKind::InvalidData)
        );
    }
}
