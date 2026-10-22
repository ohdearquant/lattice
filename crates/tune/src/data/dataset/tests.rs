use super::*;
use crate::data::IntentLabels;

fn make_example(context_size: usize) -> TrainingExample {
    TrainingExample::new(
        vec![vec![0.1, 0.2, 0.3]; context_size],
        vec![0.4, 0.5, 0.6],
        IntentLabels::continuation(0.8),
    )
}

#[test]
fn test_dataset_creation() {
    let examples: Vec<TrainingExample> = (0..10).map(|_| make_example(3)).collect();
    let dataset = Dataset::from_examples(examples);

    assert_eq!(dataset.len(), 10);
    assert!(!dataset.is_empty());
}

#[test]
fn test_dataset_batching() {
    let examples: Vec<TrainingExample> = (0..10).map(|_| make_example(3)).collect();
    let mut dataset = Dataset::from_examples(examples);
    dataset
        .set_config(DatasetConfig::with_batch_size(3).shuffle(false))
        .unwrap();

    assert_eq!(dataset.num_batches(), 4); // 10 / 3 = 3.33 -> 4 batches

    let batches: Vec<Batch> = dataset.batches().collect();
    assert_eq!(batches.len(), 4);
    assert_eq!(batches[0].len(), 3);
    assert_eq!(batches[1].len(), 3);
    assert_eq!(batches[2].len(), 3);
    assert_eq!(batches[3].len(), 1); // Incomplete final batch
}

#[test]
fn test_dataset_drop_last() {
    let examples: Vec<TrainingExample> = (0..10).map(|_| make_example(3)).collect();
    let mut dataset = Dataset::from_examples(examples);
    dataset
        .set_config(
            DatasetConfig::with_batch_size(3)
                .shuffle(false)
                .drop_last(true),
        )
        .unwrap();

    assert_eq!(dataset.num_batches(), 3); // Drop incomplete batch

    let batches: Vec<Batch> = dataset.batches().collect();
    assert_eq!(batches.len(), 3);
}

#[test]
fn test_dataset_shuffle() {
    let examples: Vec<TrainingExample> = (0..100).map(|_| make_example(3)).collect();
    let mut dataset1 = Dataset::from_examples(examples.clone());
    let mut dataset2 = Dataset::from_examples(examples);

    dataset1
        .set_config(DatasetConfig::with_batch_size(10).seed(42))
        .unwrap();
    dataset2
        .set_config(DatasetConfig::with_batch_size(10).seed(42))
        .unwrap();

    // Same seed should produce same shuffle
    let batches1: Vec<Batch> = dataset1.batches().collect();
    let batches2: Vec<Batch> = dataset2.batches().collect();

    for (b1, b2) in batches1.iter().zip(batches2.iter()) {
        for (e1, e2) in b1.examples.iter().zip(b2.examples.iter()) {
            assert_eq!(e1.id, e2.id);
        }
    }
}

#[test]
fn test_dataset_stats() {
    let examples: Vec<TrainingExample> = vec![
        make_example(2),
        make_example(3),
        make_example(4),
        make_example(5),
    ];
    let dataset = Dataset::from_examples(examples);
    let stats = dataset.stats();

    assert_eq!(stats.num_examples, 4);
    assert_eq!(stats.embedding_dim, 3);
    assert_eq!(stats.min_context_size, 2);
    assert_eq!(stats.max_context_size, 5);
    assert!((stats.avg_context_size - 3.5).abs() < 0.01);
}

#[test]
fn test_dataset_split() {
    let examples: Vec<TrainingExample> = (0..100).map(|_| make_example(3)).collect();
    let dataset = Dataset::from_examples(examples);

    let (train, val) = dataset.split(0.8).unwrap();
    assert_eq!(train.len(), 80);
    assert_eq!(val.len(), 20);
}

#[test]
fn test_batch_methods() {
    let examples: Vec<TrainingExample> = (0..5).map(|_| make_example(3)).collect();
    let batch = Batch {
        examples,
        batch_idx: 0,
        total_batches: 1,
    };

    assert_eq!(batch.len(), 5);
    assert!(!batch.is_empty());

    let embeddings = batch.message_embeddings();
    assert_eq!(embeddings.len(), 5);
    assert_eq!(embeddings[0].len(), 3);

    let labels = batch.labels();
    assert_eq!(labels.len(), 5);
    assert_eq!(labels[0].len(), 6);
}
