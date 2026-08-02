//! Frozen-baseline embedding drift comparison.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{EmbedError, EmbeddingModel, EmbeddingService, NativeEmbeddingService, Result};

/// Maximum permitted per-vector value of `1 - cosine`.
pub const MAX_COSINE_DRIFT: f32 = 1e-3;

/// Frozen texts and embeddings for one model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineFixture {
    /// Canonical embedding model name.
    pub model: String,
    /// Inputs used to generate the frozen vectors.
    pub texts: Vec<String>,
    /// Frozen vectors in the same order as [`Self::texts`].
    pub embeddings: Vec<Vec<f32>>,
}

/// Result of attempting one model comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelDriftOutcome {
    /// The model ran and every current vector was compared with its baseline.
    Checked {
        /// Largest `1 - cosine` value observed.
        max_one_minus_cos: f32,
        /// Index of the vector with the largest drift.
        worst_index: usize,
    },
    /// The model could not run because its checkpoint was not provisioned.
    WeightsAbsent {
        /// Canonical model name.
        model: String,
    },
    /// The requested model has no frozen fixture.
    NoBaseline {
        /// Canonical model name.
        model: String,
    },
}

/// Load every JSON fixture in a directory in filename order.
pub fn load_baselines(dir: &Path) -> Result<Vec<BaselineFixture>> {
    let entries = std::fs::read_dir(dir).map_err(|error| {
        EmbedError::InvalidInput(format!(
            "failed to read baseline directory {}: {error}",
            dir.display()
        ))
    })?;
    let mut paths = entries
        .map(|entry| {
            entry.map(|entry| entry.path()).map_err(|error| {
                EmbedError::InvalidInput(format!(
                    "failed to enumerate baseline directory {}: {error}",
                    dir.display()
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    paths.retain(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "json"));
    paths.sort();

    paths
        .into_iter()
        .map(|path| {
            let bytes = std::fs::read(&path).map_err(|error| {
                EmbedError::InvalidInput(format!(
                    "failed to read baseline fixture {}: {error}",
                    path.display()
                ))
            })?;
            serde_json::from_slice(&bytes).map_err(|error| {
                EmbedError::InvalidInput(format!(
                    "failed to parse baseline fixture {}: {error}",
                    path.display()
                ))
            })
        })
        .collect()
}

/// Compare current embeddings with one frozen fixture.
pub fn compare_embeddings(
    baseline: &BaselineFixture,
    current: &[Vec<f32>],
) -> Result<ModelDriftOutcome> {
    if baseline.texts.len() != baseline.embeddings.len() {
        return Err(EmbedError::InvalidInput(format!(
            "baseline {} texts and embeddings counts differ: {} != {}",
            baseline.model,
            baseline.texts.len(),
            baseline.embeddings.len()
        )));
    }
    if baseline.embeddings.is_empty() {
        return Err(EmbedError::InvalidInput(format!(
            "baseline {} contains no embeddings",
            baseline.model
        )));
    }
    if current.len() != baseline.embeddings.len() {
        return Err(EmbedError::DimensionMismatch {
            expected: baseline.embeddings.len(),
            actual: current.len(),
        });
    }

    let mut max_one_minus_cos = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (expected, actual)) in baseline.embeddings.iter().zip(current.iter()).enumerate() {
        if expected.len() != actual.len() {
            return Err(EmbedError::DimensionMismatch {
                expected: expected.len(),
                actual: actual.len(),
            });
        }
        if expected.is_empty() {
            return Err(EmbedError::InvalidInput(format!(
                "baseline {} vector {index} is empty",
                baseline.model
            )));
        }
        let drift = 1.0 - cosine_f32(expected, actual);
        if !drift.is_finite() {
            return Err(EmbedError::InvalidInput(format!(
                "baseline {} vector {index} produced non-finite cosine drift",
                baseline.model
            )));
        }
        if drift > max_one_minus_cos {
            max_one_minus_cos = drift;
            worst_index = index;
        }
    }

    Ok(ModelDriftOutcome::Checked {
        max_one_minus_cos,
        worst_index,
    })
}

/// Run one fixture against the currently provisioned model checkpoint.
pub async fn check_baseline(baseline: &BaselineFixture) -> Result<ModelDriftOutcome> {
    let model = EmbeddingModel::from_str(&baseline.model).map_err(EmbedError::InvalidInput)?;
    let weights = model_weights_path(model)?;
    if !weights.is_file() {
        return Ok(ModelDriftOutcome::WeightsAbsent {
            model: baseline.model.clone(),
        });
    }

    let service = NativeEmbeddingService::with_model(model);
    let current = service.embed(&baseline.texts, model).await?;
    compare_embeddings(baseline, &current)
}

/// Regenerate one fixture from a provisioned model checkpoint and input corpus.
pub async fn generate_baseline(model: EmbeddingModel, texts: &[String]) -> Result<BaselineFixture> {
    let weights = model_weights_path(model)?;
    if !weights.is_file() {
        return Err(EmbedError::ModelNotLoaded(format!(
            "weights missing for {model} at {}",
            weights.display()
        )));
    }
    if texts.is_empty() {
        return Err(EmbedError::InvalidInput(
            "cannot generate a baseline from an empty text corpus".to_string(),
        ));
    }

    let service = NativeEmbeddingService::with_model(model);
    let embeddings = service.embed(texts, model).await?;
    Ok(BaselineFixture {
        model: model.to_string(),
        texts: texts.to_vec(),
        embeddings,
    })
}

fn model_weights_path(model: EmbeddingModel) -> Result<PathBuf> {
    if !model.is_local() {
        return Err(EmbedError::UnsupportedModel(model.to_string()));
    }
    let root = match std::env::var_os("LATTICE_MODEL_CACHE") {
        Some(path) => PathBuf::from(path),
        None => {
            let home = std::env::var_os("HOME").ok_or_else(|| {
                EmbedError::ModelNotLoaded(
                    "HOME and LATTICE_MODEL_CACHE are both unset".to_string(),
                )
            })?;
            PathBuf::from(home).join(".lattice").join("models")
        }
    };
    Ok(root.join(model.to_string()).join("model.safetensors"))
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|value| value * value).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|value| value * value).sum::<f32>().sqrt();
    let denominator = norm_a * norm_b;
    if denominator < 1e-12 {
        1.0
    } else {
        (dot / denominator).clamp(-1.0, 1.0)
    }
}
