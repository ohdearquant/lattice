//! An encoder written outside `lattice-arch`, built only from the public constructors and shared
//! as `Arc<dyn Encoder>` across threads.

use lattice_arch::encoder::{
    EncodedText, Encoder, EncoderCapabilities, EncoderIdentity, Normalization, Pooling, Provenance,
    TokenUsage,
};
use lattice_inference_core::InferenceError;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

const DIM: usize = 4;
const LIMIT: usize = 3;
const REFUSED: &str = "refused";

/// Embeds the first `LIMIT` whitespace tokens: element 0 is the token count, element 1 the byte
/// length of the embedded prefix, the rest a fixed tag.
struct WordEncoder {
    capabilities: EncoderCapabilities,
    identity: EncoderIdentity,
}

impl WordEncoder {
    fn new() -> Self {
        let dimension = NonZeroUsize::new(DIM).expect("a nonzero dimension");
        Self {
            capabilities: EncoderCapabilities::text(NonZeroUsize::new(LIMIT)),
            identity: EncoderIdentity::new(Pooling::Mean, Normalization::None, dimension)
                .with_checkpoint(Provenance::Pinned("weights-rev-1".into()))
                .with_tokenizer(Provenance::Pinned("tokenizer-rev-1".into()))
                .with_prompt_convention("raw")
                .with_variant("test words"),
        }
    }
}

impl Encoder for WordEncoder {
    fn capabilities(&self) -> &EncoderCapabilities {
        &self.capabilities
    }

    fn output_dimension(&self) -> NonZeroUsize {
        self.identity.output_dimension()
    }

    fn identity(&self) -> &EncoderIdentity {
        &self.identity
    }

    fn encode_texts(&self, prepared: &[&str]) -> Result<Vec<EncodedText>, InferenceError> {
        if prepared.contains(&REFUSED) {
            return Err(InferenceError::InvalidInput("a refused input".into()));
        }
        let limit = self
            .capabilities
            .max_input_tokens()
            .map_or(usize::MAX, NonZeroUsize::get);
        prepared
            .iter()
            .map(|text| {
                let words: Vec<&str> = text.split_whitespace().collect();
                let kept = &words[..words.len().min(limit)];
                let usage = TokenUsage::new(words.len(), kept.len())
                    .ok_or_else(|| InferenceError::Inference("impossible counts".into()))?;
                let bytes: usize = kept.iter().map(|w| w.len()).sum();
                Ok(EncodedText::new(
                    vec![kept.len() as f32, bytes as f32, 1.0, -1.0],
                    usage,
                ))
            })
            .collect()
    }
}

/// What a consumer checks before trusting a handle's output.
fn check_batch(encoder: &dyn Encoder, inputs: &[&str], outputs: &[EncodedText]) {
    assert_eq!(outputs.len(), inputs.len(), "one output per input");
    let dimension = encoder.output_dimension();
    assert_eq!(dimension, encoder.identity().output_dimension());
    for output in outputs {
        assert_eq!(output.vector().len(), dimension.get());
        if let Some(limit) = encoder.capabilities().max_input_tokens() {
            assert!(output.usage().embedded() <= limit.get());
        }
    }
}

fn hash_of(identity: &EncoderIdentity) -> u64 {
    let mut hasher = DefaultHasher::new();
    identity.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn a_shared_handle_keeps_input_order_on_every_thread() {
    let encoder: Arc<dyn Encoder> = Arc::new(WordEncoder::new());
    let batches: [&[&str]; 4] = [
        &["a", "bb cc", "ddd eee fff"],
        &["one two", "x"],
        &["first second third fourth", "", "z y"],
        &["solo"],
    ];
    std::thread::scope(|scope| {
        let handles: Vec<_> = batches
            .iter()
            .map(|batch| {
                let encoder = Arc::clone(&encoder);
                scope.spawn(move || {
                    let outputs = encoder.encode_texts(batch).expect("the batch encodes");
                    check_batch(encoder.as_ref(), batch, &outputs);
                    let embedded: Vec<usize> =
                        outputs.iter().map(|o| o.usage().embedded()).collect();
                    let expected: Vec<usize> = batch
                        .iter()
                        .map(|t| t.split_whitespace().count().min(LIMIT))
                        .collect();
                    assert_eq!(embedded, expected, "outputs follow input order");
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("an encoding thread");
        }
    });
}

#[test]
fn input_past_the_limit_is_truncated_and_reported() {
    let encoder = WordEncoder::new();
    let outputs = encoder
        .encode_texts(&["w1 w2 w3 w4 w5", "w1 w2"])
        .expect("both inputs encode");
    let usages: Vec<(usize, usize, usize)> = outputs
        .iter()
        .map(|o| {
            let u = o.usage();
            (u.before_truncation(), u.embedded(), u.truncated())
        })
        .collect();
    assert_eq!(usages, vec![(5, LIMIT, 2), (2, 2, 0)]);
}

#[test]
fn an_error_covers_the_whole_call_and_an_empty_call_returns_nothing() {
    let encoder = WordEncoder::new();
    assert!(matches!(
        encoder.encode_texts(&["fine", REFUSED, "also fine"]),
        Err(InferenceError::InvalidInput(_))
    ));
    assert_eq!(
        encoder.encode_texts(&[]).expect("an empty call"),
        Vec::new()
    );
}

#[test]
fn every_identity_field_changes_the_identity() {
    let base = WordEncoder::new().identity().clone();
    assert!(base.has_pinned_provenance());
    assert_eq!(WordEncoder::new().identity(), &base);
    assert_eq!(hash_of(WordEncoder::new().identity()), hash_of(&base));
    let other_dimension = NonZeroUsize::new(DIM + 1).expect("a nonzero dimension");
    let rebuilt = |pooling, normalization, dimension| {
        EncoderIdentity::new(pooling, normalization, dimension)
            .with_checkpoint(base.checkpoint().clone())
            .with_tokenizer(base.tokenizer().clone())
            .with_prompt_convention(base.prompt_convention())
            .with_variant(base.variant())
    };
    assert_eq!(
        rebuilt(
            base.pooling(),
            base.normalization(),
            base.output_dimension()
        ),
        base
    );
    let variants = [
        base.clone()
            .with_checkpoint(Provenance::Pinned("weights-rev-2".into())),
        base.clone().with_checkpoint(Provenance::Unavailable),
        base.clone()
            .with_tokenizer(Provenance::Pinned("tokenizer-rev-2".into())),
        base.clone().with_prompt_convention("query-v2"),
        base.clone().with_variant("test words f16"),
        rebuilt(
            Pooling::LastToken,
            base.normalization(),
            base.output_dimension(),
        ),
        rebuilt(base.pooling(), Normalization::L2, base.output_dimension()),
        rebuilt(base.pooling(), base.normalization(), other_dimension),
    ];
    for (index, changed) in variants.iter().enumerate() {
        assert_ne!(changed, &base, "variant {index} must differ from the base");
    }
    assert!(!variants[1].has_pinned_provenance());
}
