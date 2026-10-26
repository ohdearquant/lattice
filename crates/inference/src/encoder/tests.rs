use super::*;

fn dim(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("a nonzero test dimension")
}

fn counts(before_truncation: usize, embedded: usize) -> EmbeddingTokenCounts {
    EmbeddingTokenCounts {
        before_truncation,
        embedded,
    }
}

#[test]
fn usage_refuses_counts_no_truncation_produces() {
    assert!(matches!(
        usage(counts(3, 4)),
        Err(InferenceError::Inference(_))
    ));
    let cut = usage(counts(9, 4)).expect("a truncated usage");
    assert_eq!((cut.before_truncation(), cut.embedded()), (9, 4));
}

#[test]
fn pair_refuses_a_batch_of_the_wrong_size_or_width() {
    let u = usage(counts(2, 2)).expect("a usage");
    let paired = pair(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![u, u], dim(2))
        .expect("two vectors of width 2");
    assert_eq!(paired[1].vector(), &[0.0, 1.0]);
    assert!(matches!(
        pair(vec![vec![1.0, 0.0]], vec![u, u], dim(2)),
        Err(InferenceError::Inference(_))
    ));
    assert!(matches!(
        pair(vec![vec![1.0, 0.0], vec![1.0]], vec![u, u], dim(2)),
        Err(InferenceError::Inference(_))
    ));
}
