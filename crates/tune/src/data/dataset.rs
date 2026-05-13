//! Dataset loading and batching

use super::TrainingExample;
use crate::error::{Result, TuneError};
use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for dataset operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DatasetConfig {
    /// Batch size for training
    pub batch_size: usize,

    /// Whether to shuffle data each epoch
    pub shuffle: bool,

    /// Random seed for shuffling (None = use system entropy)
    pub seed: Option<u64>,

    /// Drop incomplete final batch
    pub drop_last: bool,

    /// Minimum context size required (filter out shorter examples)
    pub min_context_size: usize,

    /// Maximum context size (truncate longer sequences)
    pub max_context_size: Option<usize>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            seed: None,
            drop_last: false,
            min_context_size: 1,
            max_context_size: None,
        }
    }
}

impl DatasetConfig {
    /// Create a new config with specified batch size
    pub fn with_batch_size(batch_size: usize) -> Self {
        Self {
            batch_size,
            ..Default::default()
        }
    }

    /// Set shuffle behavior
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set drop_last behavior
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(TuneError::InvalidConfig(
                "batch_size must be > 0".to_string(),
            ));
        }
        if let Some(max) = self.max_context_size {
            if max < self.min_context_size {
                return Err(TuneError::InvalidConfig(format!(
                    "max_context_size ({}) must be >= min_context_size ({})",
                    max, self.min_context_size
                )));
            }
        }
        Ok(())
    }
}

/// A batch of training examples
#[derive(Debug, Clone)]
pub struct Batch {
    /// Examples in this batch
    pub examples: Vec<TrainingExample>,

    /// Batch index within the epoch
    pub batch_idx: usize,

    /// Total number of batches in the epoch
    pub total_batches: usize,
}

impl Batch {
    /// Create a batch from examples
    pub fn from_examples(examples: Vec<TrainingExample>, batch_idx: usize) -> Self {
        Self {
            examples,
            batch_idx,
            total_batches: 1, // Single batch by default
        }
    }

    /// Get the batch size
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get all message embeddings as a 2D matrix (batch_size x embedding_dim)
    pub fn message_embeddings(&self) -> Vec<Vec<f32>> {
        self.examples
            .iter()
            .map(|e| e.message_embedding.clone())
            .collect()
    }

    /// Get all label vectors as a 2D matrix (batch_size x num_classes)
    pub fn labels(&self) -> Vec<Vec<f32>> {
        self.examples.iter().map(|e| e.labels.to_vec()).collect()
    }
}

/// Dataset statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DatasetStats {
    /// Total number of examples
    pub num_examples: usize,

    /// Embedding dimension
    pub embedding_dim: usize,

    /// Average context size
    pub avg_context_size: f32,

    /// Min context size
    pub min_context_size: usize,

    /// Max context size
    pub max_context_size: usize,

    /// Label distribution (count per class)
    pub label_distribution: Vec<usize>,
}

/// A collection of training examples with batching support
#[derive(Debug, Clone)]
pub struct Dataset {
    /// All examples in the dataset
    examples: Vec<TrainingExample>,

    /// Configuration for batching
    config: DatasetConfig,

    /// Current index for iteration
    current_idx: usize,

    /// Shuffled indices for current epoch
    indices: Vec<usize>,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
            config: DatasetConfig::default(),
            current_idx: 0,
            indices: Vec::new(),
        }
    }

    /// Create a dataset from examples
    pub fn from_examples(examples: Vec<TrainingExample>) -> Self {
        let indices: Vec<usize> = (0..examples.len()).collect();
        Self {
            examples,
            config: DatasetConfig::default(),
            current_idx: 0,
            indices,
        }
    }

    /// Create a dataset with specific configuration
    pub fn with_config(examples: Vec<TrainingExample>, config: DatasetConfig) -> Result<Self> {
        config.validate()?;

        // Filter examples based on context size requirements
        let filtered: Vec<TrainingExample> = examples
            .into_iter()
            .filter(|e| {
                e.context_size() >= config.min_context_size
                    && config
                        .max_context_size
                        .is_none_or(|max| e.context_size() <= max)
            })
            .collect();

        let indices: Vec<usize> = (0..filtered.len()).collect();

        Ok(Self {
            examples: filtered,
            config,
            current_idx: 0,
            indices,
        })
    }

    /// Add an example to the dataset
    pub fn add(&mut self, example: TrainingExample) {
        self.examples.push(example);
        self.indices.push(self.indices.len());
    }

    /// Get the number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get an example by ID
    pub fn get(&self, id: &Uuid) -> Option<&TrainingExample> {
        self.examples.iter().find(|e| e.id == *id)
    }

    /// Get an example by index
    pub fn get_idx(&self, idx: usize) -> Option<&TrainingExample> {
        self.examples.get(idx)
    }

    /// Get all examples
    pub fn examples(&self) -> &[TrainingExample] {
        &self.examples
    }

    /// Get mutable access to all examples
    pub fn examples_mut(&mut self) -> &mut Vec<TrainingExample> {
        &mut self.examples
    }

    /// Set the dataset configuration
    pub fn set_config(&mut self, config: DatasetConfig) -> Result<()> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Get the current configuration
    pub fn config(&self) -> &DatasetConfig {
        &self.config
    }

    /// Calculate the number of batches per epoch
    pub fn num_batches(&self) -> usize {
        if self.examples.is_empty() {
            return 0;
        }

        let n = self.examples.len();
        let batch_size = self.config.batch_size;

        if self.config.drop_last {
            n / batch_size
        } else {
            n.div_ceil(batch_size)
        }
    }

    /// Reset iteration and optionally shuffle for a new epoch
    pub fn reset_epoch(&mut self) {
        self.current_idx = 0;
        self.indices = (0..self.examples.len()).collect();

        if self.config.shuffle && !self.examples.is_empty() {
            self.shuffle_indices();
        }
    }

    /// Shuffle indices using simple Fisher-Yates
    fn shuffle_indices(&mut self) {
        // Simple LCG for deterministic shuffling when seed is provided
        let mut state = self.config.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });

        for i in (1..self.indices.len()).rev() {
            // LCG: state = (a * state + c) mod m
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (state as usize) % (i + 1);
            self.indices.swap(i, j);
        }
    }

    /// Get the next batch, returns None when epoch is complete
    pub fn next_batch(&mut self) -> Option<Batch> {
        if self.current_idx >= self.examples.len() {
            return None;
        }

        let batch_size = self.config.batch_size;
        let end_idx = (self.current_idx + batch_size).min(self.examples.len());

        // Skip incomplete final batch if drop_last is true
        if self.config.drop_last && (end_idx - self.current_idx) < batch_size {
            self.current_idx = self.examples.len();
            return None;
        }

        let batch_idx = self.current_idx / batch_size;
        let total_batches = self.num_batches();

        let examples: Vec<TrainingExample> = self.indices[self.current_idx..end_idx]
            .iter()
            .map(|&i| self.examples[i].clone())
            .collect();

        self.current_idx = end_idx;

        Some(Batch {
            examples,
            batch_idx,
            total_batches,
        })
    }

    /// Iterate over batches (consumes one epoch)
    pub fn batches(&mut self) -> BatchIterator<'_> {
        self.reset_epoch();
        BatchIterator { dataset: self }
    }

    /// Compute dataset statistics
    pub fn stats(&self) -> DatasetStats {
        if self.examples.is_empty() {
            return DatasetStats::default();
        }

        let num_examples = self.examples.len();
        let embedding_dim = self.examples[0].embedding_dim();

        let context_sizes: Vec<usize> = self
            .examples
            .iter()
            .map(TrainingExample::context_size)
            .collect();
        let avg_context_size = context_sizes.iter().sum::<usize>() as f32 / num_examples as f32;
        let min_context_size = *context_sizes.iter().min().unwrap_or(&0);
        let max_context_size = *context_sizes.iter().max().unwrap_or(&0);

        // Count dominant labels
        let mut label_distribution = vec![0usize; 6];
        for example in &self.examples {
            let probs = example.labels.to_vec();
            let dominant_idx = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);
            label_distribution[dominant_idx] += 1;
        }

        DatasetStats {
            num_examples,
            embedding_dim,
            avg_context_size,
            min_context_size,
            max_context_size,
            label_distribution,
        }
    }

    /// Split dataset into train/validation sets
    pub fn split(&self, train_ratio: f32) -> Result<(Dataset, Dataset)> {
        if !(0.0..=1.0).contains(&train_ratio) {
            return Err(TuneError::InvalidConfig(
                "train_ratio must be between 0.0 and 1.0".to_string(),
            ));
        }

        let split_idx = (self.examples.len() as f32 * train_ratio) as usize;
        let (train_examples, val_examples) = self.examples.split_at(split_idx);

        let train = Dataset::from_examples(train_examples.to_vec());
        let val = Dataset::from_examples(val_examples.to_vec());

        Ok((train, val))
    }
}

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over batches in a dataset
pub struct BatchIterator<'a> {
    dataset: &'a mut Dataset,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        self.dataset.next_batch()
    }
}

#[cfg(test)]
mod tests {
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
}
