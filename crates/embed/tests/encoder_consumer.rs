//! A consumer outside `lattice-inference` loading a BERT checkpoint, wrapping it in the facade's
//! encoder adapter and sharing it as `Arc<dyn Encoder>`.

use lattice_inference::bert_pooling_fixture::{
    DIMENSIONS, INPUTS, MAX_POSITIONS, write_checkpoint,
};
use lattice_inference::encoder::{BertEncoder, Encoder, Normalization, Pooling, Provenance};
use lattice_inference::{BertModel, BertPooling};
use std::num::NonZeroUsize;
use std::sync::Arc;

fn load(pooling: BertPooling) -> (tempfile::TempDir, Arc<BertModel>) {
    let parent = tempfile::tempdir().expect("a temporary directory");
    let dir = parent.path().join("checkpoint");
    write_checkpoint(&dir).expect("the synthetic checkpoint writes");
    let mut model = BertModel::from_directory(&dir).expect("the synthetic checkpoint loads");
    model.set_pooling(pooling);
    (parent, Arc::new(model))
}

#[test]
fn the_bert_adapter_returns_the_models_vectors_and_counts_on_every_thread() {
    let (_dir, model) = load(BertPooling::CLS);
    let encoder: Arc<dyn Encoder> =
        Arc::new(BertEncoder::new(Arc::clone(&model)).expect("the adapter wraps the model"));
    assert_eq!(encoder.output_dimension().get(), DIMENSIONS);
    assert_eq!(encoder.identity().output_dimension().get(), DIMENSIONS);
    assert_eq!(encoder.identity().pooling(), Pooling::FirstToken);
    assert_eq!(encoder.identity().normalization(), Normalization::L2);
    assert_eq!(encoder.identity().checkpoint(), &Provenance::Unavailable);
    assert!(!encoder.identity().has_pinned_provenance());
    assert_eq!(
        encoder
            .capabilities()
            .max_input_tokens()
            .map(NonZeroUsize::get),
        Some(MAX_POSITIONS)
    );

    let long = vec!["hello world"; 20].join(" ");
    let inputs = [INPUTS[0], INPUTS[1], INPUTS[2], long.as_str()];
    let expected = model.encode_batch(&inputs).expect("the model encodes");
    let reversed: Vec<&str> = inputs.iter().rev().copied().collect();
    let expected_reversed = model.encode_batch(&reversed).expect("the model encodes");

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|thread| {
                let encoder = Arc::clone(&encoder);
                let model = Arc::clone(&model);
                let (batch, want) = if thread % 2 == 0 {
                    (&inputs[..], &expected)
                } else {
                    (&reversed[..], &expected_reversed)
                };
                scope.spawn(move || {
                    let outputs = encoder.encode_texts(batch).expect("the adapter encodes");
                    assert_eq!(outputs.len(), batch.len());
                    for ((text, output), vector) in batch.iter().zip(&outputs).zip(want) {
                        assert_eq!(output.vector(), vector.as_slice(), "{text}");
                        let counts = model.embedding_token_counts(text);
                        assert_eq!(
                            (
                                output.usage().before_truncation(),
                                output.usage().embedded()
                            ),
                            (counts.before_truncation, counts.embedded),
                            "{text}"
                        );
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("an encoding thread");
        }
    });

    let outputs = encoder.encode_texts(&inputs).expect("the adapter encodes");
    let last = outputs.last().expect("four outputs").usage();
    assert_eq!(last.embedded(), MAX_POSITIONS);
    assert!(last.truncated() > 0, "the long input is cut to the limit");
    assert_eq!(outputs[0].usage().truncated(), 0);
    assert!(encoder.encode_texts(&[]).expect("an empty call").is_empty());
}

#[test]
fn the_bert_adapter_reports_the_pooling_the_model_was_set_to() {
    let (_dir, model) = load(BertPooling::Mean);
    let encoder = BertEncoder::new(model).expect("the adapter wraps the model");
    assert_eq!(encoder.identity().pooling(), Pooling::Mean);
}
