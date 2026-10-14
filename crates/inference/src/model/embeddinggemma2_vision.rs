//! Deferred EmbeddingGemma 2 vision checkpoint loading.

#![allow(
    dead_code,
    reason = "the image forward path consumes this internal cache separately"
)]

use crate::error::InferenceError;
use crate::weights::{SafetensorsFile, TensorSource};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisionConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch_size: usize,
    pub(crate) position_embedding_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f64,
    pub(crate) text_hidden_size: usize,
}

#[derive(Deserialize)]
struct RawRope {
    rope_type: String,
    rope_theta: f64,
}

#[derive(Deserialize)]
struct RawVisionConfig {
    model_type: String,
    dtype: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    global_head_dim: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
    position_embedding_size: usize,
    default_output_length: usize,
    rms_norm_eps: f64,
    hidden_activation: String,
    attention_bias: bool,
    standardize: bool,
    use_clipped_linears: bool,
    rope_parameters: RawRope,
}

impl VisionConfig {
    fn from_config_json_str(json: &str, text_hidden_size: usize) -> Result<Self, InferenceError> {
        let root: serde_json::Value = serde_json::from_str(json).map_err(|error| {
            InferenceError::Inference(format!("invalid EmbeddingGemma 2 config.json: {error}"))
        })?;
        let vision = root.get("vision_config").ok_or_else(|| {
            InferenceError::Inference("EmbeddingGemma 2 config: missing vision_config".into())
        })?;
        let raw: RawVisionConfig = serde_json::from_value(vision.clone()).map_err(|error| {
            InferenceError::Inference(format!("invalid EmbeddingGemma 2 vision_config: {error}"))
        })?;
        let unsupported = |field: &str, actual: String, expected: &str| {
            InferenceError::UnsupportedModel(format!(
                "EmbeddingGemma 2 vision_config: {field}={actual}, expected {expected}"
            ))
        };
        for (field, actual, expected) in [
            ("model_type", raw.model_type.as_str(), "gemma4_vision"),
            ("dtype", raw.dtype.as_str(), "bfloat16"),
            (
                "hidden_activation",
                raw.hidden_activation.as_str(),
                "gelu_pytorch_tanh",
            ),
            (
                "rope_parameters.rope_type",
                raw.rope_parameters.rope_type.as_str(),
                "axial",
            ),
        ] {
            if actual != expected {
                return Err(unsupported(field, format!("{actual:?}"), expected));
            }
        }
        for (field, actual, expected) in [
            ("hidden_size", raw.hidden_size, 768),
            ("intermediate_size", raw.intermediate_size, 3072),
            ("num_hidden_layers", raw.num_hidden_layers, 16),
            ("num_attention_heads", raw.num_attention_heads, 12),
            ("num_key_value_heads", raw.num_key_value_heads, 12),
            ("head_dim", raw.head_dim, 64),
            ("global_head_dim", raw.global_head_dim, 64),
            ("patch_size", raw.patch_size, 16),
            ("pooling_kernel_size", raw.pooling_kernel_size, 3),
            (
                "position_embedding_size",
                raw.position_embedding_size,
                10240,
            ),
            ("default_output_length", raw.default_output_length, 280),
            ("text_config.hidden_size", text_hidden_size, 512),
        ] {
            if actual != expected {
                return Err(unsupported(
                    field,
                    actual.to_string(),
                    &expected.to_string(),
                ));
            }
        }
        for (field, enabled) in [
            ("attention_bias", raw.attention_bias),
            ("standardize", raw.standardize),
            ("use_clipped_linears", raw.use_clipped_linears),
        ] {
            if enabled {
                return Err(unsupported(field, "true".into(), "false"));
            }
        }
        for (field, actual, expected) in [
            ("rms_norm_eps", raw.rms_norm_eps, 1e-6),
            (
                "rope_parameters.rope_theta",
                raw.rope_parameters.rope_theta,
                100.0,
            ),
        ] {
            if !actual.is_finite() || actual != expected {
                return Err(unsupported(
                    field,
                    actual.to_string(),
                    &expected.to_string(),
                ));
            }
        }
        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            patch_size: raw.patch_size,
            position_embedding_size: raw.position_embedding_size,
            rms_norm_eps: raw.rms_norm_eps as f32,
            rope_theta: raw.rope_parameters.rope_theta,
            text_hidden_size,
        })
    }
}

#[derive(Debug, Clone)]
struct TensorSpec {
    name: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct TensorMetadata {
    name: String,
    shape: Vec<usize>,
    dtype: Option<String>,
}

fn is_vision_tensor(name: &str) -> bool {
    name.starts_with("vision_tower.") || name.starts_with("embed_vision.")
}

fn checked_product(values: &[usize], name: &str) -> Result<usize, InferenceError> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!("tensor {name}: shape product overflow"))
        })
    })
}

fn inventory(cfg: &VisionConfig) -> Result<Vec<TensorSpec>, InferenceError> {
    let hidden = cfg.hidden_size;
    let ff = cfg.intermediate_size;
    let q = checked_product(
        &[cfg.num_attention_heads, cfg.head_dim],
        "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
    )?;
    let kv = checked_product(
        &[cfg.num_key_value_heads, cfg.head_dim],
        "vision_tower.encoder.layers.0.self_attn.k_proj.linear.weight",
    )?;
    let patch = checked_product(
        &[3, cfg.patch_size, cfg.patch_size],
        "vision_tower.patch_embedder.input_proj.weight",
    )?;
    let mut specs = Vec::new();
    for index in 0..cfg.num_hidden_layers {
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![hidden]),
            ("post_attention_layernorm.weight", vec![hidden]),
            ("pre_feedforward_layernorm.weight", vec![hidden]),
            ("post_feedforward_layernorm.weight", vec![hidden]),
            ("self_attn.q_norm.weight", vec![cfg.head_dim]),
            ("self_attn.k_norm.weight", vec![cfg.head_dim]),
            ("self_attn.q_proj.linear.weight", vec![q, hidden]),
            ("self_attn.k_proj.linear.weight", vec![kv, hidden]),
            ("self_attn.v_proj.linear.weight", vec![kv, hidden]),
            ("self_attn.o_proj.linear.weight", vec![hidden, q]),
            ("mlp.gate_proj.linear.weight", vec![ff, hidden]),
            ("mlp.up_proj.linear.weight", vec![ff, hidden]),
            ("mlp.down_proj.linear.weight", vec![hidden, ff]),
        ] {
            specs.push(TensorSpec {
                name: format!("vision_tower.encoder.layers.{index}.{suffix}"),
                shape,
            });
        }
    }
    for (name, shape) in [
        (
            "vision_tower.patch_embedder.input_proj.weight",
            vec![hidden, patch],
        ),
        (
            "vision_tower.patch_embedder.position_embedding_table",
            vec![2, cfg.position_embedding_size, hidden],
        ),
        (
            "embed_vision.embedding_projection.weight",
            vec![cfg.text_hidden_size, hidden],
        ),
    ] {
        specs.push(TensorSpec {
            name: name.into(),
            shape,
        });
    }
    for spec in &specs {
        checked_product(&spec.shape, &spec.name)?;
    }
    Ok(specs)
}

fn metadata(file: &SafetensorsFile) -> Result<Vec<TensorMetadata>, InferenceError> {
    file.tensor_names()
        .into_iter()
        .filter(|name| is_vision_tensor(name))
        .map(|name| {
            let shape = file
                .tensor_shape(name)
                .ok_or_else(|| InferenceError::MissingTensor(name.into()))?;
            Ok(TensorMetadata {
                name: name.into(),
                shape: shape.to_vec(),
                dtype: file.tensor_dtype(name).map(str::to_string),
            })
        })
        .collect()
}

fn validate_metadata(records: &[TensorMetadata], cfg: &VisionConfig) -> Result<(), InferenceError> {
    let specs = inventory(cfg)?;
    let expected: BTreeMap<&str, &[usize]> = specs
        .iter()
        .map(|spec| (spec.name.as_str(), spec.shape.as_slice()))
        .collect();
    let mut actual = BTreeMap::new();
    for record in records
        .iter()
        .filter(|record| is_vision_tensor(&record.name))
    {
        if actual.insert(record.name.as_str(), record).is_some() {
            return Err(InferenceError::InvalidSafetensors(format!(
                "duplicate vision tensor {}",
                record.name
            )));
        }
    }
    for &name in actual.keys() {
        if !expected.contains_key(name) {
            return Err(InferenceError::InvalidSafetensors(format!(
                "unexpected vision tensor {name}"
            )));
        }
    }
    for spec in &specs {
        let record = actual
            .get(spec.name.as_str())
            .ok_or_else(|| InferenceError::MissingTensor(spec.name.clone()))?;
        if record.dtype.as_deref() != Some("BF16") {
            return Err(InferenceError::InvalidSafetensors(format!(
                "vision tensor {} has dtype {:?}, expected BF16",
                spec.name, record.dtype
            )));
        }
        if record.shape != spec.shape {
            return Err(InferenceError::ShapeMismatch {
                name: spec.name.clone(),
                expected: spec.shape.clone(),
                actual: record.shape.clone(),
            });
        }
    }
    Ok(())
}

pub(crate) struct VisionLayerWeights {
    pub(crate) input_layernorm: Vec<f32>,
    pub(crate) post_attention_layernorm: Vec<f32>,
    pub(crate) pre_feedforward_layernorm: Vec<f32>,
    pub(crate) post_feedforward_layernorm: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) q_proj: Vec<f32>,
    pub(crate) k_proj: Vec<f32>,
    pub(crate) v_proj: Vec<f32>,
    pub(crate) o_proj: Vec<f32>,
    pub(crate) gate_proj: Vec<f32>,
    pub(crate) up_proj: Vec<f32>,
    pub(crate) down_proj: Vec<f32>,
}

pub(crate) struct VisionWeights {
    pub(crate) config: VisionConfig,
    pub(crate) layers: Vec<VisionLayerWeights>,
    pub(crate) input_proj: Vec<f32>,
    pub(crate) position_embedding_table: Vec<f32>,
    pub(crate) embedding_projection: Vec<f32>,
}

fn load_tensor<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    if source.tensor_dtype(name)?.as_deref() != Some("BF16") {
        return Err(InferenceError::InvalidSafetensors(format!(
            "vision tensor {name}: expected BF16 dtype"
        )));
    }
    let declared = source
        .tensor_shape(name)?
        .ok_or_else(|| InferenceError::MissingTensor(name.into()))?;
    if declared != expected {
        return Err(InferenceError::ShapeMismatch {
            name: name.into(),
            expected: expected.to_vec(),
            actual: declared,
        });
    }
    let (values, actual) = source.get_f32_tensor_owned(name)?;
    if actual != expected {
        return Err(InferenceError::ShapeMismatch {
            name: name.into(),
            expected: expected.to_vec(),
            actual,
        });
    }
    let count = checked_product(expected, name)?;
    if values.len() != count {
        return Err(InferenceError::InvalidSafetensors(format!(
            "vision tensor {name}: {} values for expected {count}",
            values.len()
        )));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(InferenceError::InvalidSafetensors(format!(
            "vision tensor {name}: nonfinite f32 value at index {index}"
        )));
    }
    Ok(values)
}

fn load_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: VisionConfig,
    records: &[TensorMetadata],
) -> Result<VisionWeights, InferenceError> {
    validate_metadata(records, &cfg)?;
    let hidden = cfg.hidden_size;
    let q = checked_product(
        &[cfg.num_attention_heads, cfg.head_dim],
        "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
    )?;
    let kv = checked_product(
        &[cfg.num_key_value_heads, cfg.head_dim],
        "vision_tower.encoder.layers.0.self_attn.k_proj.linear.weight",
    )?;
    let patch = checked_product(
        &[3, cfg.patch_size, cfg.patch_size],
        "vision_tower.patch_embedder.input_proj.weight",
    )?;
    let input_proj = load_tensor(
        source,
        "vision_tower.patch_embedder.input_proj.weight",
        &[hidden, patch],
    )?;
    let position_embedding_table = load_tensor(
        source,
        "vision_tower.patch_embedder.position_embedding_table",
        &[2, cfg.position_embedding_size, hidden],
    )?;
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for index in 0..cfg.num_hidden_layers {
        let prefix = format!("vision_tower.encoder.layers.{index}.");
        let mut get = |suffix: &str, shape: &[usize]| {
            load_tensor(source, &format!("{prefix}{suffix}"), shape)
        };
        layers.push(VisionLayerWeights {
            input_layernorm: get("input_layernorm.weight", &[hidden])?,
            post_attention_layernorm: get("post_attention_layernorm.weight", &[hidden])?,
            pre_feedforward_layernorm: get("pre_feedforward_layernorm.weight", &[hidden])?,
            post_feedforward_layernorm: get("post_feedforward_layernorm.weight", &[hidden])?,
            q_norm: get("self_attn.q_norm.weight", &[cfg.head_dim])?,
            k_norm: get("self_attn.k_norm.weight", &[cfg.head_dim])?,
            q_proj: get("self_attn.q_proj.linear.weight", &[q, hidden])?,
            k_proj: get("self_attn.k_proj.linear.weight", &[kv, hidden])?,
            v_proj: get("self_attn.v_proj.linear.weight", &[kv, hidden])?,
            o_proj: get("self_attn.o_proj.linear.weight", &[hidden, q])?,
            gate_proj: get(
                "mlp.gate_proj.linear.weight",
                &[cfg.intermediate_size, hidden],
            )?,
            up_proj: get(
                "mlp.up_proj.linear.weight",
                &[cfg.intermediate_size, hidden],
            )?,
            down_proj: get(
                "mlp.down_proj.linear.weight",
                &[hidden, cfg.intermediate_size],
            )?,
        });
    }
    let embedding_projection = load_tensor(
        source,
        "embed_vision.embedding_projection.weight",
        &[cfg.text_hidden_size, hidden],
    )?;
    Ok(VisionWeights {
        config: cfg,
        layers,
        input_proj,
        position_embedding_table,
        embedding_projection,
    })
}

struct SuccessCache<T> {
    value: Mutex<Option<Arc<T>>>,
}

impl<T> Default for SuccessCache<T> {
    fn default() -> Self {
        Self {
            value: Mutex::new(None),
        }
    }
}

impl<T> SuccessCache<T> {
    fn get_or_try_init(
        &self,
        initialize: impl FnOnce() -> Result<T, InferenceError>,
    ) -> Result<Arc<T>, InferenceError> {
        let mut value = self.value.lock().map_err(|_| {
            InferenceError::Inference("EmbeddingGemma 2 vision weight cache lock poisoned".into())
        })?;
        if let Some(loaded) = value.as_ref() {
            return Ok(Arc::clone(loaded));
        }
        let loaded = Arc::new(initialize()?);
        *value = Some(Arc::clone(&loaded));
        Ok(loaded)
    }
}

struct VisionCheckpoint {
    config_json: String,
    file: Mutex<File>,
    path: PathBuf,
    text_hidden_size: usize,
}

pub(super) struct LazyVision {
    checkpoint: Option<VisionCheckpoint>,
    weights: SuccessCache<VisionWeights>,
}

impl LazyVision {
    pub(super) fn from_checkpoint(
        config_json: String,
        file: File,
        path: PathBuf,
        text_hidden_size: usize,
    ) -> Self {
        Self {
            checkpoint: Some(VisionCheckpoint {
                config_json,
                file: Mutex::new(file),
                path,
                text_hidden_size,
            }),
            weights: SuccessCache::default(),
        }
    }

    pub(super) fn get(&self) -> Result<Arc<VisionWeights>, InferenceError> {
        self.weights.get_or_try_init(|| {
            let checkpoint = self.checkpoint.as_ref().ok_or_else(|| {
                InferenceError::ModelNotFound(
                    "EmbeddingGemma 2 vision checkpoint is unavailable".into(),
                )
            })?;
            let cfg = VisionConfig::from_config_json_str(
                &checkpoint.config_json,
                checkpoint.text_hidden_size,
            )?;
            let descriptor = checkpoint
                .file
                .lock()
                .map_err(|_| {
                    InferenceError::Inference(
                        "EmbeddingGemma 2 vision checkpoint lock poisoned".into(),
                    )
                })?
                .try_clone()
                .map_err(InferenceError::Io)?;
            let mut file = SafetensorsFile::from_open_file(descriptor, &checkpoint.path)?;
            let records = metadata(&file)?;
            load_weights(&mut file, cfg, &records)
        })
    }

    #[cfg(test)]
    pub(super) fn unavailable() -> Self {
        Self {
            checkpoint: None,
            weights: SuccessCache::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn is_loaded(&self) -> bool {
        self.weights.value.lock().is_ok_and(|value| value.is_some())
    }
}

mod forward;

pub(crate) use forward::VisionStages;

/// Test seam: a vision configuration with the tiny dimensions the forward tests use.
///
/// The production parser rejects anything but the released 768/16-layer geometry, so
/// these dimensions are only reachable from this crate's own tests.
#[cfg(test)]
pub(crate) fn tiny_test_config() -> VisionConfig {
    VisionConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 8,
        patch_size: 16,
        position_embedding_size: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 100.0,
        text_hidden_size: 8,
    }
}

/// Test seam: deterministic weights for [`tiny_test_config`].
#[cfg(test)]
pub(crate) fn tiny_test_weights() -> VisionWeights {
    let cfg = tiny_test_config();
    let h = cfg.hidden_size;
    let d = cfg.head_dim;
    let ff = cfg.intermediate_size;
    let patch = 3 * cfg.patch_size * cfg.patch_size;
    let pattern = |len: usize, seed: usize| -> Vec<f32> {
        (0..len)
            .map(|i| {
                let raw = ((seed + 1) * 97 + (i + 1) * 31) % 41;
                (raw as f32 - 20.0) / 40.0
            })
            .collect()
    };
    let layers = (0..cfg.num_hidden_layers)
        .map(|layer| VisionLayerWeights {
            input_layernorm: vec![1.0; h],
            post_attention_layernorm: vec![1.0; h],
            pre_feedforward_layernorm: vec![1.0; h],
            post_feedforward_layernorm: vec![1.0; h],
            q_norm: vec![1.0; d],
            k_norm: vec![1.0; d],
            q_proj: pattern(h * h, layer * 13 + 1),
            k_proj: pattern(h * h, layer * 13 + 2),
            v_proj: pattern(h * h, layer * 13 + 3),
            o_proj: pattern(h * h, layer * 13 + 4),
            gate_proj: pattern(ff * h, layer * 13 + 5),
            up_proj: pattern(ff * h, layer * 13 + 6),
            down_proj: pattern(h * ff, layer * 13 + 7),
        })
        .collect();
    VisionWeights {
        config: cfg.clone(),
        layers,
        input_proj: pattern(h * patch, 3),
        position_embedding_table: pattern(2 * cfg.position_embedding_size * h, 4),
        embedding_projection: pattern(cfg.text_hidden_size * h, 5),
    }
}

/// Test seam: a loader that hands back [`tiny_test_weights`] without touching a file.
#[cfg(test)]
pub(super) fn tiny_test_loader() -> LazyVision {
    LazyVision {
        checkpoint: None,
        weights: SuccessCache {
            value: Mutex::new(Some(Arc::new(tiny_test_weights()))),
        },
    }
}

#[cfg(test)]
mod tests;
