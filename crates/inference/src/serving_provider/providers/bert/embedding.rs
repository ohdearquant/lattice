use crate::error::InferenceError;
use crate::model::BertModel;
use crate::pool::{BertPooling, default_bert_pooling_for_model_name};
use crate::serving_embeddings::{
    EmbeddingRepresentation, TextEmbeddingHandle, TextEmbeddingService, TokenCounts,
};
use crate::serving_provider::TextEmbeddingLoadPlan;
use std::path::PathBuf;
use std::sync::Arc;

const BERT_LOADER_FORMAT: &str = "bert-safetensors";

pub(super) struct BertTextEmbeddingPlan {
    pub(super) directory: PathBuf,
    pub(super) pooling_override: Option<String>,
}

impl TextEmbeddingLoadPlan for BertTextEmbeddingPlan {
    fn load(self: Box<Self>) -> Result<TextEmbeddingHandle, String> {
        if !self.directory.exists() {
            return Err(format!(
                "embedding model directory not found: {}",
                self.directory.display()
            ));
        }
        eprintln!(
            "[lattice_serve] loading embedding model from {} ...",
            self.directory.display()
        );
        let mut model = BertModel::from_directory(&self.directory).map_err(|error| {
            format!(
                "embedding model load failed ({}): {error}",
                self.directory.display()
            )
        })?;
        let model_id = self
            .directory
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("embedding");
        let mut pooling = default_bert_pooling_for_model_name(model_id);
        match self.pooling_override.as_deref() {
            None => {}
            Some("mean") => pooling = BertPooling::Mean,
            Some("cls") => pooling = BertPooling::CLS,
            Some(other) => {
                return Err(format!(
                    "invalid --embedding-pooling '{other}'; expected 'mean' or 'cls'"
                ));
            }
        };
        model.set_pooling(pooling);
        let representation = EmbeddingRepresentation {
            loader_format: BERT_LOADER_FORMAT,
            dimensions: model.dimensions(),
        };
        eprintln!(
            "[lattice_serve] embedding model '{model_id}' ready (dimensions={})",
            representation.dimensions
        );
        Ok(TextEmbeddingHandle::new(Arc::new(
            BertTextEmbeddingService {
                model,
                representation,
            },
        )))
    }
}

struct BertTextEmbeddingService {
    model: BertModel,
    representation: EmbeddingRepresentation,
}

impl TextEmbeddingService for BertTextEmbeddingService {
    fn representation(&self) -> EmbeddingRepresentation {
        self.representation
    }

    fn max_context(&self) -> usize {
        self.model.config().max_position_embeddings
    }

    fn count_batch(&self, input: &[String]) -> Vec<TokenCounts> {
        let references: Vec<&str> = input.iter().map(String::as_str).collect();
        self.model
            .tokenizer()
            .tokenize_batch(&references)
            .iter()
            .map(|tokens| TokenCounts {
                pre_truncation: tokens.pre_truncation_len,
                embedded: tokens.real_length,
            })
            .collect()
    }

    fn encode_batch(&self, input: &[String]) -> Result<Vec<Vec<f32>>, InferenceError> {
        let references: Vec<&str> = input.iter().map(String::as_str).collect();
        self.model.encode_batch(&references)
    }
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::bert_pooling_fixture::{DIMENSIONS, INPUTS, MAX_POSITIONS, write_checkpoint};
    use crate::serving_provider::plan_text_embeddings;
    use std::fs;

    fn fixture(parent: &std::path::Path, name: &str) -> PathBuf {
        let path = parent.join(name);
        write_checkpoint(&path).expect("deterministic checkpoint must be written");
        path
    }

    fn input_strings() -> Vec<String> {
        INPUTS.iter().map(|input| (*input).to_owned()).collect()
    }

    #[test]
    fn text_service_delegates_counts_context_and_batch_encoding() {
        let temp = tempfile::tempdir().expect("temporary directory must be created");
        let path = fixture(temp.path(), "bge-service-fixture");
        let mut reference = BertModel::from_directory(&path).expect("fixture must load");
        reference.set_pooling(BertPooling::CLS);
        let service = plan_text_embeddings(&path, None)
            .expect("fixture must select a text provider")
            .load()
            .expect("fixture service must load");
        let mut inputs = input_strings();
        inputs.push("hello ".repeat(MAX_POSITIONS));
        let references: Vec<&str> = inputs.iter().map(String::as_str).collect();
        let expected_counts = reference.tokenizer().tokenize_batch(&references);
        let actual_counts = service.count_batch(&inputs);

        assert_eq!(actual_counts.len(), expected_counts.len());
        for (actual, expected) in actual_counts.iter().zip(&expected_counts) {
            assert_eq!(
                *actual,
                TokenCounts {
                    pre_truncation: expected.pre_truncation_len,
                    embedded: expected.real_length,
                }
            );
        }
        assert!(actual_counts.last().unwrap().pre_truncation > MAX_POSITIONS);
        assert_eq!(actual_counts.last().unwrap().embedded, MAX_POSITIONS);
        assert_eq!(
            service.max_context(),
            reference.config().max_position_embeddings
        );
        assert_eq!(service.representation().loader_format, BERT_LOADER_FORMAT);
        assert_eq!(service.representation().dimensions, DIMENSIONS);
        assert_eq!(
            service
                .encode_batch(&inputs)
                .expect("service batch must encode"),
            reference
                .encode_batch(&references)
                .expect("reference batch must encode")
        );
    }

    #[test]
    fn text_plan_uses_basename_pooling_then_applies_override() {
        let temp = tempfile::tempdir().expect("temporary directory must be created");
        let cases = [
            ("bge-default-fixture", None, BertPooling::CLS),
            ("all-minilm-default-fixture", None, BertPooling::Mean),
            ("bge-override-fixture", Some("mean"), BertPooling::Mean),
        ];
        let inputs = input_strings();
        let references: Vec<&str> = inputs.iter().map(String::as_str).collect();

        for (name, override_value, expected_pooling) in cases {
            let path = fixture(temp.path(), name);
            let mut reference = BertModel::from_directory(&path).expect("fixture must load");
            reference.set_pooling(expected_pooling);
            let service = plan_text_embeddings(&path, override_value.map(str::to_owned))
                .expect("fixture must select a text provider")
                .load()
                .expect("fixture service must load");
            assert_eq!(
                service
                    .encode_batch(&inputs)
                    .expect("service batch must encode"),
                reference
                    .encode_batch(&references)
                    .expect("reference batch must encode")
            );
        }
    }

    #[test]
    fn text_plan_reports_load_errors_before_invalid_pooling() {
        let temp = tempfile::tempdir().expect("temporary directory must be created");
        let missing = temp.path().join("missing");
        let missing_error = plan_text_embeddings(&missing, Some("invalid".to_owned()))
            .expect("missing paths use the legacy BERT fallback")
            .load()
            .expect_err("missing path must fail before pooling validation");
        assert_eq!(
            missing_error,
            format!("embedding model directory not found: {}", missing.display())
        );

        let malformed = temp.path().join("malformed");
        fs::create_dir(&malformed).expect("malformed directory must be created");
        let malformed_error = plan_text_embeddings(&malformed, Some("invalid".to_owned()))
            .expect("malformed paths use the legacy BERT fallback")
            .load()
            .expect_err("checkpoint load must fail before pooling validation");
        assert!(malformed_error.starts_with(&format!(
            "embedding model load failed ({}): ",
            malformed.display()
        )));
    }
}
