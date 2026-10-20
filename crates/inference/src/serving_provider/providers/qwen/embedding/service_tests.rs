use crate::forward::cpu_f16::PoolingStrategy;
use crate::serve::embeddings::test_support::tiny_embedding_model;
use crate::serving_embeddings::{EmbeddingRepresentation, PooledEmbeddingHandle};
use crate::tokenizer::common::Tokenizer as _;
use std::sync::Arc;

#[test]
fn pooled_handle_matches_legacy_model_operations() {
    let model = Arc::new(tiny_embedding_model());
    let handle = PooledEmbeddingHandle::new(Arc::clone(&model));
    let text = "ab";
    let pooling = PoolingStrategy::MeanVisualTokens;

    let expected_embedding = model.embed_text(text, pooling).expect("legacy embedding");
    let actual_embedding = handle.embed_text(text, pooling).expect("service embedding");
    assert_eq!(actual_embedding, expected_embedding);

    let tokenized = model.tokenizer.tokenize(text);
    assert_eq!(
        handle.text_counts(text),
        crate::serving_embeddings::TokenCounts {
            pre_truncation: tokenized.pre_truncation_len,
            embedded: tokenized.real_length,
        }
    );

    let expected_representation = EmbeddingRepresentation {
        loader_format: crate::serve::embeddings::QWEN35_F16_DECODER_LOADER,
        dimensions: model.dimensions(),
    };
    assert_eq!(handle.representation(), expected_representation);
}
