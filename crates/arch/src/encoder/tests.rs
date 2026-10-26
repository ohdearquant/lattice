use super::*;

fn dim(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("a nonzero test dimension")
}

#[test]
fn token_usage_refuses_more_embedded_than_tokenized() {
    assert_eq!(TokenUsage::new(4, 5), None);
    let exact = TokenUsage::new(5, 5).expect("equal counts are a usage");
    assert_eq!(exact.truncated(), 0);
    let cut = TokenUsage::new(9, 5).expect("a truncated usage");
    assert_eq!(
        (cut.before_truncation(), cut.embedded(), cut.truncated()),
        (9, 5, 4)
    );
    assert_eq!(TokenUsage::new(0, 0).map(|u| u.truncated()), Some(0));
}

#[test]
fn encoded_text_keeps_its_vector_and_usage() {
    let usage = TokenUsage::new(3, 3).expect("a usage");
    let encoded = EncodedText::new(vec![0.5, -0.5], usage);
    assert_eq!(encoded.vector(), &[0.5, -0.5]);
    assert_eq!(encoded.usage(), usage);
    assert_eq!(encoded.into_vector(), vec![0.5, -0.5]);
}

#[test]
fn capabilities_report_the_limit_they_were_built_with() {
    assert_eq!(
        EncoderCapabilities::text(Some(dim(512))).max_input_tokens(),
        Some(dim(512))
    );
    assert_eq!(EncoderCapabilities::text(None).max_input_tokens(), None);
}

#[test]
fn a_new_identity_has_unavailable_provenance_and_the_raw_convention() {
    let identity = EncoderIdentity::new(Pooling::Mean, Normalization::L2, dim(8));
    assert_eq!(identity.checkpoint(), &Provenance::Unavailable);
    assert_eq!(identity.tokenizer(), &Provenance::Unavailable);
    assert_eq!(identity.prompt_convention(), "raw");
    assert_eq!(identity.variant(), "");
    assert_eq!(identity.pooling(), Pooling::Mean);
    assert_eq!(identity.normalization(), Normalization::L2);
    assert_eq!(identity.output_dimension(), dim(8));
    assert!(!identity.has_pinned_provenance());
}

#[test]
fn pinned_provenance_needs_both_the_checkpoint_and_the_tokenizer() {
    let base = EncoderIdentity::new(Pooling::LastToken, Normalization::L2, dim(8));
    let weights_only = base
        .clone()
        .with_checkpoint(Provenance::Pinned("rev-a".into()));
    let tokenizer_only = base
        .clone()
        .with_tokenizer(Provenance::Pinned("tok-a".into()));
    let both = weights_only
        .clone()
        .with_tokenizer(Provenance::Pinned("tok-a".into()));
    assert!(!weights_only.has_pinned_provenance());
    assert!(!tokenizer_only.has_pinned_provenance());
    assert!(both.has_pinned_provenance());
}
