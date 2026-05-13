use crate::error::InferenceError;
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }
}

#[derive(Debug)]
struct TensorMeta {
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
    converted_f32: OnceLock<Box<[f32]>>,
}

/// **Unstable**: 2D tensor view over memory-mapped safetensors data; field layout may change.
///
/// A 2D tensor (matrix) backed by memory-mapped data.
#[derive(Debug, Clone, Copy)]
pub struct Tensor2D<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

/// **Unstable**: 1D tensor view over memory-mapped safetensors data; field layout may change.
///
/// A 1D tensor (vector/bias) backed by memory-mapped data.
#[derive(Debug, Clone, Copy)]
pub struct Tensor1D<'a> {
    pub data: &'a [f32],
    pub len: usize,
}

/// **Unstable**: per-layer BERT transformer weights; field set may change with model variants.
///
/// All weights for one transformer layer.
#[derive(Debug, Clone)]
pub struct TransformerLayerWeights<'a> {
    pub query_weight: Tensor2D<'a>,
    pub query_bias: Tensor1D<'a>,
    pub key_weight: Tensor2D<'a>,
    pub key_bias: Tensor1D<'a>,
    pub value_weight: Tensor2D<'a>,
    pub value_bias: Tensor1D<'a>,
    pub attn_output_weight: Tensor2D<'a>,
    pub attn_output_bias: Tensor1D<'a>,
    pub attn_layer_norm_weight: Tensor1D<'a>,
    pub attn_layer_norm_bias: Tensor1D<'a>,
    pub ffn_intermediate_weight: Tensor2D<'a>,
    pub ffn_intermediate_bias: Tensor1D<'a>,
    pub ffn_output_weight: Tensor2D<'a>,
    pub ffn_output_bias: Tensor1D<'a>,
    pub ffn_layer_norm_weight: Tensor1D<'a>,
    pub ffn_layer_norm_bias: Tensor1D<'a>,
}

/// **Unstable**: full BERT model weight bundle; structure may change with model variants.
///
/// All weights for the BERT model.
#[derive(Debug, Clone)]
pub struct BertWeights<'a> {
    pub word_embeddings: Tensor2D<'a>,
    pub position_embeddings: Tensor2D<'a>,
    pub token_type_embeddings: Tensor2D<'a>,
    pub embedding_layer_norm_weight: Tensor1D<'a>,
    pub embedding_layer_norm_bias: Tensor1D<'a>,
    pub layers: Vec<TransformerLayerWeights<'a>>,
    pub pooler_weight: Tensor2D<'a>,
    pub pooler_bias: Tensor1D<'a>,
}

/// Owned classifier head weights for a `BertForSequenceClassification` checkpoint.
///
/// Copied at load time so the struct is `'static` and independent of the mmap.
#[derive(Debug, Clone)]
pub struct CrossEncoderWeights {
    /// Classifier weight row; length equals `hidden_size`.
    pub classifier_weight: Vec<f32>,
    /// Classifier bias scalar.
    pub classifier_bias: f32,
}

impl CrossEncoderWeights {
    /// Compute the raw logit: `weight · pooled + bias`.
    pub fn logit(&self, pooled: &[f32]) -> f32 {
        debug_assert_eq!(self.classifier_weight.len(), pooled.len());
        self.classifier_weight
            .iter()
            .zip(pooled.iter())
            .map(|(w, v)| w * v)
            .sum::<f32>()
            + self.classifier_bias
    }
}

/// **Unstable**: memory-mapped safetensors file reader; internal representation may change.
///
/// Memory-mapped safetensors file.
#[derive(Debug)]
pub struct SafetensorsFile {
    mmap: Mmap,
    /// Byte offset where tensor data starts (8 + header_length).
    data_offset: usize,
    /// Parsed tensor metadata.
    tensors: HashMap<String, TensorMeta>,
}

impl SafetensorsFile {
    /// **Unstable**: open a safetensors file; error type and header parsing may change.
    ///
    /// Open and parse a safetensors file.
    pub fn open(path: &Path) -> Result<Self, InferenceError> {
        let file = File::open(path).map_err(|e| {
            InferenceError::InvalidSafetensors(format!("failed to open {}: {e}", path.display()))
        })?;

        // SAFETY: The file descriptor remains alive until the mmap is created,
        // and the returned Mmap owns the mapping independently of the File.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            InferenceError::InvalidSafetensors(format!("failed to mmap {}: {e}", path.display()))
        })?;

        if mmap.len() < 8 {
            return Err(InferenceError::InvalidSafetensors(
                "file too small to contain safetensors header".into(),
            ));
        }

        let header_len = u64::from_le_bytes(
            mmap[0..8]
                .try_into()
                .map_err(|_| InferenceError::InvalidSafetensors("invalid header length".into()))?,
        ) as usize;
        let data_offset = 8usize
            .checked_add(header_len)
            .ok_or_else(|| InferenceError::InvalidSafetensors("header length overflow".into()))?;

        if data_offset > mmap.len() {
            return Err(InferenceError::InvalidSafetensors(format!(
                "header extends past end of file: header_end={}, file_len={}",
                data_offset,
                mmap.len()
            )));
        }

        let header = std::str::from_utf8(&mmap[8..data_offset]).map_err(|e| {
            InferenceError::InvalidSafetensors(format!("header is not valid UTF-8: {e}"))
        })?;
        let tensors = parse_safetensors_header(header)?;

        let data_len = mmap.len() - data_offset;
        for (name, meta) in &tensors {
            if meta.start > meta.end {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name} has invalid offsets [{}, {})",
                    meta.start, meta.end
                )));
            }
            if meta.end > data_len {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name} points past data section: end={}, data_len={}",
                    meta.end, data_len
                )));
            }
            let numel = meta.shape.iter().try_fold(1usize, |acc, &dim| {
                acc.checked_mul(dim).ok_or_else(|| {
                    InferenceError::InvalidSafetensors(format!(
                        "tensor {name} shape {:?} overflows usize",
                        meta.shape
                    ))
                })
            })?;
            let expected = numel.checked_mul(meta.dtype.size_bytes()).ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {name} byte length overflows usize"
                ))
            })?;
            let actual = meta.end - meta.start;
            if actual != expected {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "tensor {name} byte length mismatch for dtype {} and shape {:?}: \
                     expected {expected}, got {actual}",
                    meta.dtype.name(),
                    meta.shape
                )));
            }
        }

        Ok(Self {
            mmap,
            data_offset,
            tensors,
        })
    }

    /// **Unstable**: query tensor shape by name; return type may change.
    ///
    /// Return a tensor shape if present.
    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|m| m.shape.as_slice())
    }

    /// **Unstable**: list all tensor names in the file; return type may change.
    ///
    /// Return a list of tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// **Unstable**: check whether a named tensor exists; method may be merged into tensor_shape.
    ///
    /// Whether the file contains a tensor.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// **Unstable**: load a named tensor as f32 slice; dtype conversion strategy may change.
    ///
    /// Get a tensor by name as f32 slice.
    pub fn get_f32_tensor(&self, name: &str) -> Result<(&[f32], &[usize]), InferenceError> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?;

        let start = self
            .data_offset
            .checked_add(meta.start)
            .ok_or_else(|| InferenceError::InvalidSafetensors("tensor start overflow".into()))?;
        let end = self
            .data_offset
            .checked_add(meta.end)
            .ok_or_else(|| InferenceError::InvalidSafetensors("tensor end overflow".into()))?;
        let bytes = &self.mmap[start..end];

        let slice: &[f32] = match meta.dtype {
            DType::F32 => {
                // SAFETY: `open()` validated that bytes.len() is exactly the declared
                // F32 element count times 4. On little-endian targets, aligned mmap
                // storage can be viewed directly; otherwise use the owned LE conversion
                // cache below.
                #[cfg(target_endian = "little")]
                if bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) == 0 {
                    bytes_to_f32_slice(bytes)
                } else {
                    meta.converted_f32
                        .get_or_init(|| copy_bytes_to_f32_owned(bytes).into_boxed_slice())
                        .as_ref()
                }
                #[cfg(not(target_endian = "little"))]
                {
                    meta.converted_f32
                        .get_or_init(|| copy_bytes_to_f32_owned(bytes).into_boxed_slice())
                        .as_ref()
                }
            }
            DType::F16 => {
                #[cfg(feature = "f16")]
                {
                    meta.converted_f32
                        .get_or_init(|| convert_f16_bytes_to_f32(bytes).into_boxed_slice())
                        .as_ref()
                }
                #[cfg(not(feature = "f16"))]
                {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "tensor {name} is F16 but lattice-inference was built without the f16 feature"
                    )));
                }
            }
            DType::BF16 => {
                #[cfg(feature = "f16")]
                {
                    meta.converted_f32
                        .get_or_init(|| convert_bf16_bytes_to_f32(bytes).into_boxed_slice())
                        .as_ref()
                }
                #[cfg(not(feature = "f16"))]
                {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "tensor {name} is BF16 but lattice-inference was built without the f16 feature"
                    )));
                }
            }
        };

        Ok((slice, meta.shape.as_slice()))
    }

    /// **Unstable**: load BERT weights; tensor name conventions and signature may change.
    ///
    /// Load all BERT weights from the safetensors file.
    pub fn load_bert_weights(
        &self,
        num_layers: usize,
        hidden_size: usize,
    ) -> Result<BertWeights<'_>, InferenceError> {
        let word_shape = self.require_shape("embeddings.word_embeddings.weight")?;
        let position_shape = self.require_shape("embeddings.position_embeddings.weight")?;
        let token_type_shape = self.require_shape("embeddings.token_type_embeddings.weight")?;

        if word_shape.len() != 2 || word_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "embeddings.word_embeddings.weight".into(),
                expected: vec![word_shape[0], hidden_size],
                actual: word_shape.to_vec(),
            });
        }
        if position_shape.len() != 2 || position_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "embeddings.position_embeddings.weight".into(),
                expected: vec![position_shape[0], hidden_size],
                actual: position_shape.to_vec(),
            });
        }
        if token_type_shape.len() != 2 || token_type_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "embeddings.token_type_embeddings.weight".into(),
                expected: vec![token_type_shape[0], hidden_size],
                actual: token_type_shape.to_vec(),
            });
        }

        let word_embeddings = self.tensor2d(
            "embeddings.word_embeddings.weight",
            word_shape[0],
            word_shape[1],
        )?;
        let position_embeddings = self.tensor2d(
            "embeddings.position_embeddings.weight",
            position_shape[0],
            position_shape[1],
        )?;
        let token_type_embeddings = self.tensor2d(
            "embeddings.token_type_embeddings.weight",
            token_type_shape[0],
            token_type_shape[1],
        )?;
        let embedding_layer_norm_weight =
            self.tensor1d("embeddings.LayerNorm.weight", hidden_size)?;
        let embedding_layer_norm_bias = self.tensor1d("embeddings.LayerNorm.bias", hidden_size)?;

        let intermediate_shape = self.require_shape("encoder.layer.0.intermediate.dense.weight")?;
        if intermediate_shape.len() != 2 || intermediate_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "encoder.layer.0.intermediate.dense.weight".into(),
                expected: vec![intermediate_shape[0], hidden_size],
                actual: intermediate_shape.to_vec(),
            });
        }
        let intermediate_size = intermediate_shape[0];

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("encoder.layer.{i}");
            layers.push(TransformerLayerWeights {
                query_weight: self.tensor2d(
                    &format!("{prefix}.attention.self.query.weight"),
                    hidden_size,
                    hidden_size,
                )?,
                query_bias: self
                    .tensor1d(&format!("{prefix}.attention.self.query.bias"), hidden_size)?,
                key_weight: self.tensor2d(
                    &format!("{prefix}.attention.self.key.weight"),
                    hidden_size,
                    hidden_size,
                )?,
                key_bias: self
                    .tensor1d(&format!("{prefix}.attention.self.key.bias"), hidden_size)?,
                value_weight: self.tensor2d(
                    &format!("{prefix}.attention.self.value.weight"),
                    hidden_size,
                    hidden_size,
                )?,
                value_bias: self
                    .tensor1d(&format!("{prefix}.attention.self.value.bias"), hidden_size)?,
                attn_output_weight: self.tensor2d(
                    &format!("{prefix}.attention.output.dense.weight"),
                    hidden_size,
                    hidden_size,
                )?,
                attn_output_bias: self.tensor1d(
                    &format!("{prefix}.attention.output.dense.bias"),
                    hidden_size,
                )?,
                attn_layer_norm_weight: self.tensor1d(
                    &format!("{prefix}.attention.output.LayerNorm.weight"),
                    hidden_size,
                )?,
                attn_layer_norm_bias: self.tensor1d(
                    &format!("{prefix}.attention.output.LayerNorm.bias"),
                    hidden_size,
                )?,
                ffn_intermediate_weight: self.tensor2d(
                    &format!("{prefix}.intermediate.dense.weight"),
                    intermediate_size,
                    hidden_size,
                )?,
                ffn_intermediate_bias: self.tensor1d(
                    &format!("{prefix}.intermediate.dense.bias"),
                    intermediate_size,
                )?,
                ffn_output_weight: self.tensor2d(
                    &format!("{prefix}.output.dense.weight"),
                    hidden_size,
                    intermediate_size,
                )?,
                ffn_output_bias: self
                    .tensor1d(&format!("{prefix}.output.dense.bias"), hidden_size)?,
                ffn_layer_norm_weight: self
                    .tensor1d(&format!("{prefix}.output.LayerNorm.weight"), hidden_size)?,
                ffn_layer_norm_bias: self
                    .tensor1d(&format!("{prefix}.output.LayerNorm.bias"), hidden_size)?,
            });
        }

        let pooler_weight =
            self.tensor2d_optional("pooler.dense.weight", hidden_size, hidden_size)?;
        let pooler_bias = self.tensor1d_optional("pooler.dense.bias", hidden_size)?;

        Ok(BertWeights {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embedding_layer_norm_weight,
            embedding_layer_norm_bias,
            layers,
            pooler_weight,
            pooler_bias,
        })
    }

    /// Load classifier head weights from a `BertForSequenceClassification` safetensors file.
    ///
    /// Accepts `classifier.weight` shaped `[1, hidden_size]` or `[hidden_size]` and
    /// `classifier.bias` shaped `[1]`.
    pub fn load_cross_encoder_weights(
        &self,
        hidden_size: usize,
    ) -> Result<CrossEncoderWeights, InferenceError> {
        let (weight, weight_shape) = self.get_f32_tensor("classifier.weight")?;
        let classifier_weight = match weight_shape {
            [1, cols] if *cols == hidden_size => weight.to_vec(),
            [cols] if *cols == hidden_size => weight.to_vec(),
            _ => {
                return Err(InferenceError::ShapeMismatch {
                    name: "classifier.weight".to_string(),
                    expected: vec![1, hidden_size],
                    actual: weight_shape.to_vec(),
                });
            }
        };

        let (bias, bias_shape) = self.get_f32_tensor("classifier.bias")?;
        if bias_shape != [1usize] {
            return Err(InferenceError::ShapeMismatch {
                name: "classifier.bias".to_string(),
                expected: vec![1],
                actual: bias_shape.to_vec(),
            });
        }
        let classifier_bias = bias[0];

        Ok(CrossEncoderWeights {
            classifier_weight,
            classifier_bias,
        })
    }

    fn require_shape(&self, name: &str) -> Result<&[usize], InferenceError> {
        self.tensor_shape(name)
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    fn tensor1d(&self, name: &str, len: usize) -> Result<Tensor1D<'_>, InferenceError> {
        let (data, shape) = self.get_f32_tensor(name)?;
        if shape != [len] {
            return Err(InferenceError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![len],
                actual: shape.to_vec(),
            });
        }
        Ok(Tensor1D { data, len })
    }

    fn tensor2d(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor2D<'_>, InferenceError> {
        let (data, shape) = self.get_f32_tensor(name)?;
        if shape != [rows, cols] {
            return Err(InferenceError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![rows, cols],
                actual: shape.to_vec(),
            });
        }
        Ok(Tensor2D { data, rows, cols })
    }

    fn tensor1d_optional(&self, name: &str, len: usize) -> Result<Tensor1D<'_>, InferenceError> {
        if self.has_tensor(name) {
            self.tensor1d(name, len)
        } else {
            Ok(Tensor1D { data: &[], len: 0 })
        }
    }

    fn tensor2d_optional(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor2D<'_>, InferenceError> {
        if self.has_tensor(name) {
            self.tensor2d(name, rows, cols)
        } else {
            Ok(Tensor2D {
                data: &[],
                rows: 0,
                cols: 0,
            })
        }
    }
}

/// **Unstable**: per-layer Qwen3 decoder weights; field set may change with model variants.
///
/// Weights for one Qwen3 decoder layer (GQA + SwiGLU + QK-norm).
#[derive(Debug, Clone)]
pub struct QwenLayerWeights<'a> {
    // Self-attention (GQA: Q has num_heads, K/V have num_kv_heads)
    pub q_proj_weight: Tensor2D<'a>,
    pub k_proj_weight: Tensor2D<'a>,
    pub v_proj_weight: Tensor2D<'a>,
    pub o_proj_weight: Tensor2D<'a>,
    // QK normalization (per-head RMSNorm, shape [head_dim])
    pub q_norm_weight: Tensor1D<'a>,
    pub k_norm_weight: Tensor1D<'a>,
    // Pre-attention RMSNorm
    pub input_layernorm_weight: Tensor1D<'a>,
    // SwiGLU FFN (no bias)
    pub gate_proj_weight: Tensor2D<'a>,
    pub up_proj_weight: Tensor2D<'a>,
    pub down_proj_weight: Tensor2D<'a>,
    // Pre-FFN RMSNorm
    pub post_attention_layernorm_weight: Tensor1D<'a>,

    /// Fused QKV weight: [q_dim + 2*kv_dim, hidden_size] — rows of W_q, W_k, W_v concatenated.
    /// Enables a single GEMM call instead of 3 separate projections.
    pub fused_qkv: Vec<f32>,
    /// Output dimension of fused QKV: q_dim + 2 * kv_dim.
    pub qkv_out_dim: usize,

    /// Fused gate+up weight: [2*intermediate_size, hidden_size] — rows of W_gate, W_up concatenated.
    /// Enables a single GEMM call instead of 2 separate projections.
    pub fused_gate_up: Vec<f32>,
    /// Output dimension of fused gate+up: 2 * intermediate_size.
    pub gate_up_out_dim: usize,
}

/// **Unstable**: full Qwen3 model weight bundle; structure may change with model variants.
///
/// All weights for a Qwen3 model.
#[derive(Debug, Clone)]
pub struct QwenWeights<'a> {
    pub embed_tokens: Tensor2D<'a>,
    pub norm_weight: Tensor1D<'a>,
    pub layers: Vec<QwenLayerWeights<'a>>,
}

impl SafetensorsFile {
    /// **Unstable**: load Qwen3 weights; tensor name conventions and fused-weight strategy may change.
    ///
    /// Load all Qwen3 weights from the safetensors file.
    pub fn load_qwen_weights(
        &self,
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Result<QwenWeights<'_>, InferenceError> {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let embed_shape = self.require_shape("embed_tokens.weight")?;
        if embed_shape.len() != 2 || embed_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "embed_tokens.weight".into(),
                expected: vec![embed_shape[0], hidden_size],
                actual: embed_shape.to_vec(),
            });
        }

        let embed_tokens = self.tensor2d("embed_tokens.weight", embed_shape[0], hidden_size)?;
        let norm_weight = self.tensor1d("norm.weight", hidden_size)?;

        let qkv_out_dim = q_dim + 2 * kv_dim;
        let gate_up_out_dim = 2 * intermediate_size;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let p = format!("layers.{i}");

            // Load individual weight tensors first.
            let q_proj_weight =
                self.tensor2d(&format!("{p}.self_attn.q_proj.weight"), q_dim, hidden_size)?;
            let k_proj_weight =
                self.tensor2d(&format!("{p}.self_attn.k_proj.weight"), kv_dim, hidden_size)?;
            let v_proj_weight =
                self.tensor2d(&format!("{p}.self_attn.v_proj.weight"), kv_dim, hidden_size)?;
            let gate_proj_weight = self.tensor2d(
                &format!("{p}.mlp.gate_proj.weight"),
                intermediate_size,
                hidden_size,
            )?;
            let up_proj_weight = self.tensor2d(
                &format!("{p}.mlp.up_proj.weight"),
                intermediate_size,
                hidden_size,
            )?;

            // Build fused QKV weight: [q_dim + 2*kv_dim, hidden_size]
            // Rows of W_q, W_k, W_v concatenated vertically.
            let mut fused_qkv = Vec::with_capacity(qkv_out_dim * hidden_size);
            fused_qkv.extend_from_slice(q_proj_weight.data); // [q_dim * hidden_size]
            fused_qkv.extend_from_slice(k_proj_weight.data); // [kv_dim * hidden_size]
            fused_qkv.extend_from_slice(v_proj_weight.data); // [kv_dim * hidden_size]

            // Build fused gate+up weight: [2*intermediate_size, hidden_size]
            // Rows of W_gate, W_up concatenated vertically.
            let mut fused_gate_up = Vec::with_capacity(gate_up_out_dim * hidden_size);
            fused_gate_up.extend_from_slice(gate_proj_weight.data); // [intermediate * hidden]
            fused_gate_up.extend_from_slice(up_proj_weight.data); // [intermediate * hidden]

            layers.push(QwenLayerWeights {
                q_proj_weight,
                k_proj_weight,
                v_proj_weight,
                o_proj_weight: self.tensor2d(
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden_size,
                    q_dim,
                )?,
                q_norm_weight: self.tensor1d(&format!("{p}.self_attn.q_norm.weight"), head_dim)?,
                k_norm_weight: self.tensor1d(&format!("{p}.self_attn.k_norm.weight"), head_dim)?,
                input_layernorm_weight: self
                    .tensor1d(&format!("{p}.input_layernorm.weight"), hidden_size)?,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight: self.tensor2d(
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden_size,
                    intermediate_size,
                )?,
                post_attention_layernorm_weight: self
                    .tensor1d(&format!("{p}.post_attention_layernorm.weight"), hidden_size)?,
                fused_qkv,
                qkv_out_dim,
                fused_gate_up,
                gate_up_out_dim,
            });
        }

        Ok(QwenWeights {
            embed_tokens,
            norm_weight,
            layers,
        })
    }
}

/// Reinterpret a byte slice as a f32 slice (zero-copy).
fn bytes_to_f32_slice(bytes: &[u8]) -> &[f32] {
    assert!(bytes.len() % 4 == 0);
    assert!(bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) == 0);
    // SAFETY: The caller guarantees that `bytes` is 4-byte aligned, its length is
    // a multiple of 4, and the backing storage outlives the returned slice.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

fn copy_bytes_to_f32_owned(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 4, 0);
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

#[cfg(feature = "f16")]
fn convert_f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 2, 0);
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
    }
    out
}

#[cfg(feature = "f16")]
fn convert_bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 2, 0);
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
    }
    out
}

#[cfg(feature = "f16")]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(feature = "f16")]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let f32_bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, _) => {
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (mant << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, _) => (sign << 31) | 0x7f80_0000 | (frac << 13),
        _ => (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
    };

    f32::from_bits(f32_bits)
}

/// Owned backing store for Qwen weights loaded from a sharded checkpoint.
///
/// Each field is the raw `Vec<f32>` that a `Tensor2D` or `Tensor1D` slice
/// would point into.  The struct is heap-allocated and pinned behind a `Box`
/// so the slices remain valid for the lifetime of the `QwenWeights<'_>` that
/// borrows from it.
#[derive(Debug)]
pub struct ShardedQwenBacking {
    pub embed_tokens: Vec<f32>,
    pub norm_weight: Vec<f32>,
    // Per-layer fields; indexed by layer index.
    pub q_proj: Vec<Vec<f32>>,
    pub k_proj: Vec<Vec<f32>>,
    pub v_proj: Vec<Vec<f32>>,
    pub o_proj: Vec<Vec<f32>>,
    pub q_norm: Vec<Vec<f32>>,
    pub k_norm: Vec<Vec<f32>>,
    pub input_ln: Vec<Vec<f32>>,
    pub gate_proj: Vec<Vec<f32>>,
    pub up_proj: Vec<Vec<f32>>,
    pub down_proj: Vec<Vec<f32>>,
    pub post_ln: Vec<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Sharded safetensors index + TensorSource trait
// ---------------------------------------------------------------------------

/// Parsed `model.safetensors.index.json` from a HuggingFace sharded checkpoint.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SafetensorsIndex {
    #[serde(default)]
    pub metadata: Value,
    /// Maps tensor name → shard filename (e.g. `"model-00001-of-00026.safetensors"`).
    pub weight_map: HashMap<String, String>,
}

/// Owned tensor with f32 data and shape, returned by `load_sharded`.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Uniform interface for single-file and sharded tensor sources.
pub trait TensorSource {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError>;
    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError>;
    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError>;
}

impl TensorSource for SafetensorsFile {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
        Ok(SafetensorsFile::has_tensor(self, name))
    }

    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        Ok(SafetensorsFile::tensor_shape(self, name).map(<[usize]>::to_vec))
    }

    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        let (data, shape) = self.get_f32_tensor(name)?;
        Ok((data.to_vec(), shape.to_vec()))
    }
}

/// Lazy-opening reader for sharded safetensors checkpoints.
///
/// Opens shard files on demand (first access) to avoid mapping 26 files upfront.
#[derive(Debug)]
pub struct ShardedSafetensors {
    root: PathBuf,
    index: SafetensorsIndex,
    shards: HashMap<String, SafetensorsFile>,
}

/// Parse `model.safetensors.index.json` from a model directory.
pub fn parse_index(model_dir: &Path) -> Result<SafetensorsIndex, InferenceError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let json = std::fs::read_to_string(&index_path).map_err(InferenceError::Io)?;
    serde_json::from_str(&json).map_err(|e| {
        InferenceError::InvalidSafetensors(format!("failed to parse {}: {e}", index_path.display()))
    })
}

/// Resolve a tensor name to its shard filename via the weight map.
pub fn resolve_shard(
    index: &SafetensorsIndex,
    tensor_name: &str,
) -> Result<PathBuf, InferenceError> {
    let shard_file = index
        .weight_map
        .get(tensor_name)
        .ok_or_else(|| InferenceError::MissingTensor(tensor_name.to_string()))?;
    Ok(PathBuf::from(shard_file))
}

/// Eagerly load all tensors from a sharded checkpoint into an owned map.
///
/// Groups tensor names by shard file to open each shard file exactly once.
/// For large checkpoints (e.g. 26-shard Qwen3.6), prefer `ShardedSafetensors`
/// for lower peak memory during model loading.
pub fn load_sharded(model_dir: &Path) -> Result<HashMap<String, Tensor>, InferenceError> {
    let index = parse_index(model_dir)?;
    let mut by_shard: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();

    for (tensor_name, shard_file) in &index.weight_map {
        by_shard
            .entry(shard_file.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    let mut tensors = HashMap::with_capacity(index.weight_map.len());
    for (shard_file, tensor_names) in by_shard {
        let shard_path = model_dir.join(&shard_file);
        let shard = SafetensorsFile::open(&shard_path)?;
        for tensor_name in tensor_names {
            let (data, shape) = shard.get_f32_tensor(&tensor_name)?;
            tensors.insert(
                tensor_name,
                Tensor {
                    data: data.to_vec(),
                    shape: shape.to_vec(),
                },
            );
        }
    }

    Ok(tensors)
}

impl ShardedSafetensors {
    /// Parse `model.safetensors.index.json` and set up the lazy shard reader.
    pub fn open_index(index_path: &Path) -> Result<Self, InferenceError> {
        let root = index_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let index = parse_index(&root)?;
        Ok(Self {
            root,
            index,
            shards: HashMap::new(),
        })
    }

    /// Return a reference to the parsed index.
    pub fn index(&self) -> &SafetensorsIndex {
        &self.index
    }

    /// Resolve a tensor name to `(shard_path, tensor_name)`.
    pub fn resolve_weight(&self, tensor_name: &str) -> Result<(PathBuf, String), InferenceError> {
        let shard_file = self
            .index
            .weight_map
            .get(tensor_name)
            .ok_or_else(|| InferenceError::MissingTensor(tensor_name.to_string()))?;
        Ok((self.root.join(shard_file), tensor_name.to_string()))
    }

    fn shard_file_for(&self, name: &str) -> Result<String, InferenceError> {
        self.index
            .weight_map
            .get(name)
            .cloned()
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }

    fn open_shard(&mut self, shard_file: &str) -> Result<&SafetensorsFile, InferenceError> {
        if !self.shards.contains_key(shard_file) {
            let shard_path = self.root.join(shard_file);
            let shard = SafetensorsFile::open(&shard_path)?;
            self.shards.insert(shard_file.to_string(), shard);
        }
        self.shards.get(shard_file).ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!("failed to cache shard {shard_file}"))
        })
    }
}

impl TensorSource for ShardedSafetensors {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
        Ok(self.index.weight_map.contains_key(name))
    }

    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        let shard_file = self.shard_file_for(name)?;
        let shard = self.open_shard(&shard_file)?;
        Ok(shard.tensor_shape(name).map(<[usize]>::to_vec))
    }

    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        let shard_file = self.shard_file_for(name)?;
        let shard = self.open_shard(&shard_file)?;
        let (data, shape) = shard.get_f32_tensor(name)?;
        Ok((data.to_vec(), shape.to_vec()))
    }
}

impl ShardedSafetensors {
    /// Load all Qwen3 weights from a sharded checkpoint into an owned backing store.
    ///
    /// Returns `(backing, weights)` where `weights` borrows from `backing`.
    /// The caller MUST store `backing` in a `Box` and ensure `weights` does not
    /// outlive it.  In `QwenModel` the `SafetensorsStorage::Sharded` variant
    /// owns the `Box<ShardedQwenBacking>` and is dropped after `weights`.
    pub fn load_qwen_weights_owned(
        &mut self,
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Result<(Box<ShardedQwenBacking>, QwenWeights<'static>), InferenceError> {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = q_dim + 2 * kv_dim;
        let gate_up_out_dim = 2 * intermediate_size;

        // --- Load global tensors ---
        let (embed_data, embed_shape) = self.get_f32_tensor_owned("embed_tokens.weight")?;
        if embed_shape.len() != 2 || embed_shape[1] != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "embed_tokens.weight".into(),
                expected: vec![embed_shape[0], hidden_size],
                actual: embed_shape,
            });
        }
        let embed_vocab = embed_shape[0];

        let (norm_data, norm_shape) = self.get_f32_tensor_owned("norm.weight")?;
        if norm_shape != [hidden_size] {
            return Err(InferenceError::ShapeMismatch {
                name: "norm.weight".into(),
                expected: vec![hidden_size],
                actual: norm_shape,
            });
        }

        // --- Load per-layer tensors ---
        let mut q_proj_vecs = Vec::with_capacity(num_layers);
        let mut k_proj_vecs = Vec::with_capacity(num_layers);
        let mut v_proj_vecs = Vec::with_capacity(num_layers);
        let mut o_proj_vecs = Vec::with_capacity(num_layers);
        let mut q_norm_vecs = Vec::with_capacity(num_layers);
        let mut k_norm_vecs = Vec::with_capacity(num_layers);
        let mut input_ln_vecs = Vec::with_capacity(num_layers);
        let mut gate_proj_vecs = Vec::with_capacity(num_layers);
        let mut up_proj_vecs = Vec::with_capacity(num_layers);
        let mut down_proj_vecs = Vec::with_capacity(num_layers);
        let mut post_ln_vecs = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let p = format!("layers.{i}");

            let (q, qs) = self.get_f32_tensor_owned(&format!("{p}.self_attn.q_proj.weight"))?;
            if qs != [q_dim, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.q_proj.weight"),
                    expected: vec![q_dim, hidden_size],
                    actual: qs,
                });
            }
            q_proj_vecs.push(q);

            let (k, ks) = self.get_f32_tensor_owned(&format!("{p}.self_attn.k_proj.weight"))?;
            if ks != [kv_dim, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.k_proj.weight"),
                    expected: vec![kv_dim, hidden_size],
                    actual: ks,
                });
            }
            k_proj_vecs.push(k);

            let (v, vs) = self.get_f32_tensor_owned(&format!("{p}.self_attn.v_proj.weight"))?;
            if vs != [kv_dim, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.v_proj.weight"),
                    expected: vec![kv_dim, hidden_size],
                    actual: vs,
                });
            }
            v_proj_vecs.push(v);

            let (o, os) = self.get_f32_tensor_owned(&format!("{p}.self_attn.o_proj.weight"))?;
            if os != [hidden_size, q_dim] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.o_proj.weight"),
                    expected: vec![hidden_size, q_dim],
                    actual: os,
                });
            }
            o_proj_vecs.push(o);

            let (qn, qns) = self.get_f32_tensor_owned(&format!("{p}.self_attn.q_norm.weight"))?;
            if qns != [head_dim] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.q_norm.weight"),
                    expected: vec![head_dim],
                    actual: qns,
                });
            }
            q_norm_vecs.push(qn);

            let (kn, kns) = self.get_f32_tensor_owned(&format!("{p}.self_attn.k_norm.weight"))?;
            if kns != [head_dim] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.self_attn.k_norm.weight"),
                    expected: vec![head_dim],
                    actual: kns,
                });
            }
            k_norm_vecs.push(kn);

            let (iln, ilns) = self.get_f32_tensor_owned(&format!("{p}.input_layernorm.weight"))?;
            if ilns != [hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.input_layernorm.weight"),
                    expected: vec![hidden_size],
                    actual: ilns,
                });
            }
            input_ln_vecs.push(iln);

            let (g, gs) = self.get_f32_tensor_owned(&format!("{p}.mlp.gate_proj.weight"))?;
            if gs != [intermediate_size, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.mlp.gate_proj.weight"),
                    expected: vec![intermediate_size, hidden_size],
                    actual: gs,
                });
            }
            gate_proj_vecs.push(g);

            let (u, us) = self.get_f32_tensor_owned(&format!("{p}.mlp.up_proj.weight"))?;
            if us != [intermediate_size, hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.mlp.up_proj.weight"),
                    expected: vec![intermediate_size, hidden_size],
                    actual: us,
                });
            }
            up_proj_vecs.push(u);

            let (d, ds) = self.get_f32_tensor_owned(&format!("{p}.mlp.down_proj.weight"))?;
            if ds != [hidden_size, intermediate_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.mlp.down_proj.weight"),
                    expected: vec![hidden_size, intermediate_size],
                    actual: ds,
                });
            }
            down_proj_vecs.push(d);

            let (pln, plns) =
                self.get_f32_tensor_owned(&format!("{p}.post_attention_layernorm.weight"))?;
            if plns != [hidden_size] {
                return Err(InferenceError::ShapeMismatch {
                    name: format!("{p}.post_attention_layernorm.weight"),
                    expected: vec![hidden_size],
                    actual: plns,
                });
            }
            post_ln_vecs.push(pln);
        }

        // --- Assemble owned backing ---
        let backing = Box::new(ShardedQwenBacking {
            embed_tokens: embed_data,
            norm_weight: norm_data,
            q_proj: q_proj_vecs,
            k_proj: k_proj_vecs,
            v_proj: v_proj_vecs,
            o_proj: o_proj_vecs,
            q_norm: q_norm_vecs,
            k_norm: k_norm_vecs,
            input_ln: input_ln_vecs,
            gate_proj: gate_proj_vecs,
            up_proj: up_proj_vecs,
            down_proj: down_proj_vecs,
            post_ln: post_ln_vecs,
        });

        // --- Build QwenWeights borrowing from the backing ---
        // SAFETY: The backing is heap-allocated in a Box and will be stored alongside
        // the QwenWeights<'static> in SafetensorsStorage::Sharded inside QwenModel.
        // Field drop order (RFC 1857) guarantees backing outlives weights: `weights`
        // is declared before `_safetensors` in QwenModel, so `_safetensors` (which
        // owns the backing) is dropped last.
        let weights: QwenWeights<'static> = {
            #[allow(clippy::explicit_auto_deref)]
            let b: &ShardedQwenBacking = &*backing;
            // SAFETY: We extend the lifetime of references into `b` to 'static.
            // This is safe because `backing` lives in a Box that is co-located
            // with the QwenWeights inside QwenModel and is dropped after it.
            let b: &'static ShardedQwenBacking = unsafe { &*(b as *const ShardedQwenBacking) };

            let embed_tokens = Tensor2D {
                data: &b.embed_tokens,
                rows: embed_vocab,
                cols: hidden_size,
            };
            let norm_weight = Tensor1D {
                data: &b.norm_weight,
                len: hidden_size,
            };

            let mut layers = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                // Build fused QKV: [q_dim + 2*kv_dim, hidden_size]
                let mut fused_qkv = Vec::with_capacity(qkv_out_dim * hidden_size);
                fused_qkv.extend_from_slice(&b.q_proj[i]);
                fused_qkv.extend_from_slice(&b.k_proj[i]);
                fused_qkv.extend_from_slice(&b.v_proj[i]);

                // Build fused gate+up: [2*intermediate_size, hidden_size]
                let mut fused_gate_up = Vec::with_capacity(gate_up_out_dim * hidden_size);
                fused_gate_up.extend_from_slice(&b.gate_proj[i]);
                fused_gate_up.extend_from_slice(&b.up_proj[i]);

                layers.push(QwenLayerWeights {
                    q_proj_weight: Tensor2D {
                        data: &b.q_proj[i],
                        rows: q_dim,
                        cols: hidden_size,
                    },
                    k_proj_weight: Tensor2D {
                        data: &b.k_proj[i],
                        rows: kv_dim,
                        cols: hidden_size,
                    },
                    v_proj_weight: Tensor2D {
                        data: &b.v_proj[i],
                        rows: kv_dim,
                        cols: hidden_size,
                    },
                    o_proj_weight: Tensor2D {
                        data: &b.o_proj[i],
                        rows: hidden_size,
                        cols: q_dim,
                    },
                    q_norm_weight: Tensor1D {
                        data: &b.q_norm[i],
                        len: head_dim,
                    },
                    k_norm_weight: Tensor1D {
                        data: &b.k_norm[i],
                        len: head_dim,
                    },
                    input_layernorm_weight: Tensor1D {
                        data: &b.input_ln[i],
                        len: hidden_size,
                    },
                    gate_proj_weight: Tensor2D {
                        data: &b.gate_proj[i],
                        rows: intermediate_size,
                        cols: hidden_size,
                    },
                    up_proj_weight: Tensor2D {
                        data: &b.up_proj[i],
                        rows: intermediate_size,
                        cols: hidden_size,
                    },
                    down_proj_weight: Tensor2D {
                        data: &b.down_proj[i],
                        rows: hidden_size,
                        cols: intermediate_size,
                    },
                    post_attention_layernorm_weight: Tensor1D {
                        data: &b.post_ln[i],
                        len: hidden_size,
                    },
                    fused_qkv,
                    qkv_out_dim,
                    fused_gate_up,
                    gate_up_out_dim,
                });
            }

            QwenWeights {
                embed_tokens,
                norm_weight,
                layers,
            }
        };

        Ok((backing, weights))
    }
}

fn parse_safetensors_header(json: &str) -> Result<HashMap<String, TensorMeta>, InferenceError> {
    let mut parser = JsonParser::new(json);
    parser.skip_ws();
    parser.expect(b'{')?;

    let mut tensors = HashMap::new();

    loop {
        parser.skip_ws();
        match parser.peek() {
            Some(b'}') => {
                parser.bump();
                break;
            }
            Some(_) => {}
            None => {
                return Err(InferenceError::InvalidSafetensors(
                    "unexpected end of header while parsing top-level object".into(),
                ));
            }
        }

        let key = parser.parse_string()?;
        parser.skip_ws();
        parser.expect(b':')?;
        parser.skip_ws();

        if key == "__metadata__" {
            parser.skip_value()?;
        } else if let Some(meta) = parser.parse_tensor_meta()? {
            tensors.insert(key, meta);
        }

        parser.skip_ws();
        match parser.peek() {
            Some(b',') => {
                parser.bump();
            }
            Some(b'}') => {
                parser.bump();
                break;
            }
            Some(other) => {
                return Err(InferenceError::InvalidSafetensors(format!(
                    "expected ',' or '}}' in header, found byte {other}"
                )));
            }
            None => {
                return Err(InferenceError::InvalidSafetensors(
                    "unexpected end of header after tensor entry".into(),
                ));
            }
        }
    }

    Ok(tensors)
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), InferenceError> {
        match self.bump() {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(InferenceError::InvalidSafetensors(format!(
                "expected byte {:?}, found byte {:?} at position {}",
                expected as char,
                actual as char,
                self.pos.saturating_sub(1)
            ))),
            None => Err(InferenceError::InvalidSafetensors(format!(
                "expected byte {:?}, found end of input",
                expected as char
            ))),
        }
    }

    fn parse_string(&mut self) -> Result<String, InferenceError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = self.bump().ok_or_else(|| {
                InferenceError::InvalidSafetensors("unterminated string in header".into())
            })?;
            match byte {
                b'"' => break,
                b'\\' => {
                    let esc = self.bump().ok_or_else(|| {
                        InferenceError::InvalidSafetensors(
                            "unterminated escape sequence in header".into(),
                        )
                    })?;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = self.take_n(4)?;
                            let hex_str = std::str::from_utf8(hex).map_err(|e| {
                                InferenceError::InvalidSafetensors(format!(
                                    "invalid unicode escape in header: {e}"
                                ))
                            })?;
                            let value = u16::from_str_radix(hex_str, 16).map_err(|e| {
                                InferenceError::InvalidSafetensors(format!(
                                    "invalid unicode escape value {hex_str}: {e}"
                                ))
                            })?;
                            let ch = char::from_u32(value as u32).ok_or_else(|| {
                                InferenceError::InvalidSafetensors(format!(
                                    "invalid unicode scalar value {value}"
                                ))
                            })?;
                            out.push(ch);
                        }
                        other => {
                            return Err(InferenceError::InvalidSafetensors(format!(
                                "unsupported escape sequence \\{}",
                                other as char
                            )));
                        }
                    }
                }
                other => out.push(other as char),
            }
        }
        Ok(out)
    }

    fn take_n(&mut self, n: usize) -> Result<&'a [u8], InferenceError> {
        if self.pos + n > self.bytes.len() {
            return Err(InferenceError::InvalidSafetensors(
                "unexpected end of input while reading header".into(),
            ));
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn parse_usize(&mut self) -> Result<usize, InferenceError> {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(InferenceError::InvalidSafetensors(format!(
                "expected unsigned integer at byte {}",
                self.pos
            )));
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| {
            InferenceError::InvalidSafetensors(format!("invalid number token in header: {e}"))
        })?;
        s.parse::<usize>().map_err(|e| {
            InferenceError::InvalidSafetensors(format!("invalid usize value {s}: {e}"))
        })
    }

    fn parse_usize_array(&mut self) -> Result<Vec<usize>, InferenceError> {
        self.expect(b'[')?;
        self.skip_ws();
        let mut values = Vec::new();
        if matches!(self.peek(), Some(b']')) {
            self.bump();
            return Ok(values);
        }
        loop {
            self.skip_ws();
            values.push(self.parse_usize()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                    self.skip_ws();
                }
                Some(b']') => {
                    self.bump();
                    break;
                }
                Some(other) => {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "expected ',' or ']' in array, found byte {other}"
                    )));
                }
                None => {
                    return Err(InferenceError::InvalidSafetensors(
                        "unexpected end of input in array".into(),
                    ));
                }
            }
        }
        Ok(values)
    }

    fn skip_value(&mut self) -> Result<(), InferenceError> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => {
                self.parse_string()?;
                Ok(())
            }
            Some(b'{') => {
                self.bump();
                self.skip_ws();
                if matches!(self.peek(), Some(b'}')) {
                    self.bump();
                    return Ok(());
                }
                loop {
                    self.skip_ws();
                    self.parse_string()?;
                    self.skip_ws();
                    self.expect(b':')?;
                    self.skip_ws();
                    self.skip_value()?;
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => {
                            self.bump();
                        }
                        Some(b'}') => {
                            self.bump();
                            break;
                        }
                        Some(other) => {
                            return Err(InferenceError::InvalidSafetensors(format!(
                                "expected ',' or '}}' while skipping object, found {other}"
                            )));
                        }
                        None => {
                            return Err(InferenceError::InvalidSafetensors(
                                "unexpected end of input while skipping object".into(),
                            ));
                        }
                    }
                }
                Ok(())
            }
            Some(b'[') => {
                self.bump();
                self.skip_ws();
                if matches!(self.peek(), Some(b']')) {
                    self.bump();
                    return Ok(());
                }
                loop {
                    self.skip_ws();
                    self.skip_value()?;
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => {
                            self.bump();
                        }
                        Some(b']') => {
                            self.bump();
                            break;
                        }
                        Some(other) => {
                            return Err(InferenceError::InvalidSafetensors(format!(
                                "expected ',' or ']' while skipping array, found {other}"
                            )));
                        }
                        None => {
                            return Err(InferenceError::InvalidSafetensors(
                                "unexpected end of input while skipping array".into(),
                            ));
                        }
                    }
                }
                Ok(())
            }
            Some(b't') => self.skip_literal(b"true"),
            Some(b'f') => self.skip_literal(b"false"),
            Some(b'n') => self.skip_literal(b"null"),
            Some(b'-' | b'0'..=b'9') => {
                self.skip_number();
                Ok(())
            }
            Some(other) => Err(InferenceError::InvalidSafetensors(format!(
                "unsupported JSON value starting with byte {other}"
            ))),
            None => Err(InferenceError::InvalidSafetensors(
                "unexpected end of input while skipping value".into(),
            )),
        }
    }

    fn skip_literal(&mut self, literal: &[u8]) -> Result<(), InferenceError> {
        if self.pos + literal.len() > self.bytes.len()
            || &self.bytes[self.pos..self.pos + literal.len()] != literal
        {
            return Err(InferenceError::InvalidSafetensors(format!(
                "expected literal {:?}",
                std::str::from_utf8(literal).unwrap_or("<literal>")
            )));
        }
        self.pos += literal.len();
        Ok(())
    }

    fn skip_number(&mut self) {
        while matches!(
            self.peek(),
            Some(b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        ) {
            self.pos += 1;
        }
    }

    fn parse_tensor_meta(&mut self) -> Result<Option<TensorMeta>, InferenceError> {
        self.expect(b'{')?;
        self.skip_ws();
        let mut dtype = None;
        let mut shape = None;
        let mut data_offsets = None;

        if matches!(self.peek(), Some(b'}')) {
            self.bump();
        } else {
            loop {
                self.skip_ws();
                let key = self.parse_string()?;
                self.skip_ws();
                self.expect(b':')?;
                self.skip_ws();

                match key.as_str() {
                    "dtype" => {
                        let dtype_str = self.parse_string()?;
                        dtype = match dtype_str.as_str() {
                            "F32" => Some(DType::F32),
                            "F16" => Some(DType::F16),
                            "BF16" => Some(DType::BF16),
                            _ => None, // Non-float tensor (I64, I32, etc.) — skip
                        };
                    }
                    "shape" => shape = Some(self.parse_usize_array()?),
                    "data_offsets" => {
                        let arr = self.parse_usize_array()?;
                        if arr.len() != 2 {
                            return Err(InferenceError::InvalidSafetensors(format!(
                                "data_offsets must have length 2, got {}",
                                arr.len()
                            )));
                        }
                        data_offsets = Some((arr[0], arr[1]));
                    }
                    _ => self.skip_value()?,
                }

                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.bump();
                    }
                    Some(b'}') => {
                        self.bump();
                        break;
                    }
                    Some(other) => {
                        return Err(InferenceError::InvalidSafetensors(format!(
                            "expected ',' or '}}' in tensor object, found {other}"
                        )));
                    }
                    None => {
                        return Err(InferenceError::InvalidSafetensors(
                            "unexpected end of input in tensor object".into(),
                        ));
                    }
                }
            }
        }

        let Some(dtype) = dtype else {
            return Ok(None); // Non-float tensor (I64, I32, etc.) — skip
        };
        let shape = shape.ok_or_else(|| {
            InferenceError::InvalidSafetensors("tensor entry missing shape".into())
        })?;
        let (start, end) = data_offsets.ok_or_else(|| {
            InferenceError::InvalidSafetensors("tensor entry missing data_offsets".into())
        })?;

        Ok(Some(TensorMeta {
            dtype,
            shape,
            start,
            end,
            converted_f32: OnceLock::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("invariant: system time is after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "{}_{}_{}.safetensors",
            name,
            std::process::id(),
            nanos
        ))
    }

    #[test]
    fn test_parse_small_safetensors_file() {
        let path = temp_path("lattice_weights_test");
        let header = r#"{
            "__metadata__": {"format": "pt"},
            "vec": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "mat": {"dtype": "F32", "shape": [2, 2], "data_offsets": [8, 24]}
        }"#
        .replace(['\n', ' '], "");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let mut file = File::create(&path).expect("test setup: create safetensors file");
        file.write_all(&bytes)
            .expect("test setup: write safetensors bytes");
        drop(file);

        let st = SafetensorsFile::open(&path).expect("test setup: parse safetensors file");
        let (vec_data, vec_shape) = st
            .get_f32_tensor("vec")
            .expect("test setup: vec tensor exists");
        let (mat_data, mat_shape) = st
            .get_f32_tensor("mat")
            .expect("test setup: mat tensor exists");

        assert_eq!(vec_shape, &[2]);
        assert_eq!(mat_shape, &[2, 2]);
        assert_eq!(vec_data, &[1.0, 2.0]);
        assert_eq!(mat_data, &[3.0, 4.0, 5.0, 6.0]);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_rejects_shape_byte_length_mismatch() {
        let path = temp_path("lattice_weights_bad_shape");
        let header = r#"{
            "bad": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}
        }"#
        .replace(['\n', ' '], "");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in [1.0f32, 2.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let mut file = File::create(&path).expect("test setup: create safetensors file");
        file.write_all(&bytes)
            .expect("test setup: write safetensors bytes");
        drop(file);

        let err = SafetensorsFile::open(&path)
            .expect_err("shape byte length mismatch should be rejected at open");
        assert!(
            err.to_string().contains("byte length mismatch"),
            "unexpected error: {err}"
        );

        fs::remove_file(&path).ok();
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("invariant: system time is after UNIX_EPOCH")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{}_{}_{}", name, std::process::id(), nanos));
        fs::create_dir_all(&path).expect("test setup: create temp dir");
        path
    }

    fn write_single_f32_tensor(path: &std::path::Path, name: &str, values: &[f32]) {
        let byte_len = values.len() * std::mem::size_of::<f32>();
        let header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[{}],"data_offsets":[0,{byte_len}]}}}}"#,
            values.len()
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut file = File::create(path).expect("test setup: create shard");
        file.write_all(&bytes).expect("test setup: write shard");
    }

    #[test]
    fn test_parse_safetensors_index_and_resolve_weight_map() {
        let dir = temp_dir("lattice_sharded_weights_test");
        let shard_a = dir.join("model-00001-of-00002.safetensors");
        let shard_b = dir.join("model-00002-of-00002.safetensors");
        write_single_f32_tensor(&shard_a, "tensor.a", &[1.0, 2.0]);
        write_single_f32_tensor(&shard_b, "tensor.b", &[3.0, 4.0, 5.0]);

        let index_path = dir.join("model.safetensors.index.json");
        fs::write(
            &index_path,
            r#"{
                "metadata": {"total_size": 20.0},
                "weight_map": {
                    "tensor.a": "model-00001-of-00002.safetensors",
                    "tensor.b": "model-00002-of-00002.safetensors"
                }
            }"#,
        )
        .expect("test setup: write index");

        let mut st = ShardedSafetensors::open_index(&index_path).expect("index parses");

        let (path_a, name_a) = st.resolve_weight("tensor.a").expect("tensor a resolves");
        assert_eq!(
            path_a.file_name().unwrap().to_string_lossy(),
            "model-00001-of-00002.safetensors"
        );
        assert_eq!(name_a, "tensor.a");

        let (a, a_shape) = st.get_f32_tensor_owned("tensor.a").expect("tensor a loads");
        let (b, b_shape) = st.get_f32_tensor_owned("tensor.b").expect("tensor b loads");
        assert_eq!(a_shape, vec![2]);
        assert_eq!(b_shape, vec![3]);
        assert_eq!(a, vec![1.0, 2.0]);
        assert_eq!(b, vec![3.0, 4.0, 5.0]);

        assert!(st.has_tensor("tensor.a").unwrap());
        assert!(!st.has_tensor("tensor.c").unwrap());

        // Test parse_index free function.
        let index = parse_index(&dir).expect("parse_index succeeds");
        assert_eq!(index.weight_map.len(), 2);

        // Test resolve_shard free function.
        let shard_a_path = resolve_shard(&index, "tensor.a").expect("resolve_shard tensor.a");
        assert_eq!(
            shard_a_path.to_string_lossy(),
            "model-00001-of-00002.safetensors"
        );
        assert!(resolve_shard(&index, "tensor.missing").is_err());

        // Test load_sharded free function.
        let loaded = load_sharded(&dir).expect("load_sharded succeeds");
        assert_eq!(loaded.len(), 2);
        let ta = loaded.get("tensor.a").expect("tensor.a in loaded map");
        let tb = loaded.get("tensor.b").expect("tensor.b in loaded map");
        assert_eq!(ta.data, vec![1.0_f32, 2.0]);
        assert_eq!(ta.shape, vec![2]);
        assert_eq!(tb.data, vec![3.0_f32, 4.0, 5.0]);
        assert_eq!(tb.shape, vec![3]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_sharded_multi_tensor_per_shard() {
        let dir = temp_dir("lattice_multi_tensor_shard_test");
        let shard = dir.join("model-00001-of-00001.safetensors");

        // Write one shard file containing two named tensors.
        let x_byte_count = 2_usize * std::mem::size_of::<f32>(); // 8
        let y_byte_count = 3_usize * std::mem::size_of::<f32>(); // 12
        let xy_end = x_byte_count + y_byte_count;
        let header = format!(
            r#"{{"tensor.x":{{"dtype":"F32","shape":[2],"data_offsets":[0,{x_byte_count}]}},"tensor.y":{{"dtype":"F32","shape":[3],"data_offsets":[{x_byte_count},{xy_end}]}}}}"#
        );
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for v in [1.0_f32, 2.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in [3.0_f32, 4.0, 5.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&shard, &bytes).expect("test setup: write two-tensor shard");

        let index_path = dir.join("model.safetensors.index.json");
        fs::write(
            &index_path,
            r#"{"weight_map":{"tensor.x":"model-00001-of-00001.safetensors","tensor.y":"model-00001-of-00001.safetensors"}}"#,
        )
        .expect("test setup: write index");

        let loaded = load_sharded(&dir).expect("load_sharded with multi-tensor shard succeeds");
        assert_eq!(loaded.len(), 2);
        let tx = loaded.get("tensor.x").expect("tensor.x in loaded map");
        let ty = loaded.get("tensor.y").expect("tensor.y in loaded map");
        assert_eq!(tx.data, vec![1.0_f32, 2.0]);
        assert_eq!(tx.shape, vec![2]);
        assert_eq!(ty.data, vec![3.0_f32, 4.0, 5.0]);
        assert_eq!(ty.shape, vec![3]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_sharded_missing_shard_returns_err() {
        let dir = temp_dir("lattice_missing_shard_test");

        // Index references a shard file that is never written to disk.
        let index_path = dir.join("model.safetensors.index.json");
        fs::write(
            &index_path,
            r#"{"weight_map":{"tensor.a":"model-00001-of-00001.safetensors"}}"#,
        )
        .expect("test setup: write index");

        let result = load_sharded(&dir);
        assert!(
            result.is_err(),
            "load_sharded must return Err when indexed shard file is absent"
        );

        fs::remove_dir_all(&dir).ok();
    }

    // --- CrossEncoderWeights tests ---

    #[test]
    fn test_cross_encoder_weights_logit_known_value() {
        let weights = CrossEncoderWeights {
            classifier_weight: vec![1.0f32, 2.0, -1.0],
            classifier_bias: 0.5,
        };
        let pooled = [2.0f32, 1.0, 3.0];
        // logit = 1.0*2.0 + 2.0*1.0 + (-1.0)*3.0 + 0.5 = 2.0 + 2.0 - 3.0 + 0.5 = 1.5
        let logit = weights.logit(&pooled);
        assert!(
            (logit - 1.5f32).abs() < 1e-6,
            "expected logit=1.5, got {logit}"
        );
    }

    fn write_classifier_safetensors(
        path: &std::path::Path,
        weight_shape_header: &str,
        weight_values: &[f32],
        bias_value: f32,
        weight_end: usize,
    ) {
        let bias_end = weight_end + 4; // one f32
        let header = format!(
            r#"{{"classifier.weight":{{{},"data_offsets":[0,{weight_end}]}},"classifier.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{weight_end},{bias_end}]}}}}"#,
            weight_shape_header
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for v in weight_values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&bias_value.to_le_bytes());
        let mut file = File::create(path).expect("test setup: create safetensors");
        file.write_all(&bytes)
            .expect("test setup: write safetensors");
    }

    #[test]
    fn test_load_cross_encoder_weights_2d_shape() {
        // classifier.weight [1, 3] — standard HF BertForSequenceClassification shape
        let path = temp_path("lattice_ce_weights_2d");
        write_classifier_safetensors(
            &path,
            r#""dtype":"F32","shape":[1,3]"#,
            &[1.0f32, 2.0, 3.0],
            0.5,
            12,
        );
        let st = SafetensorsFile::open(&path).unwrap();
        let weights = st.load_cross_encoder_weights(3).unwrap();
        assert_eq!(weights.classifier_weight, vec![1.0f32, 2.0, 3.0]);
        assert!(
            (weights.classifier_bias - 0.5f32).abs() < 1e-6,
            "bias mismatch: {}",
            weights.classifier_bias
        );
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_cross_encoder_weights_1d_shape() {
        // classifier.weight [3] — accepted as converted-checkpoint fallback
        let path = temp_path("lattice_ce_weights_1d");
        write_classifier_safetensors(
            &path,
            r#""dtype":"F32","shape":[3]"#,
            &[4.0f32, 5.0, 6.0],
            1.0,
            12,
        );
        let st = SafetensorsFile::open(&path).unwrap();
        let weights = st.load_cross_encoder_weights(3).unwrap();
        assert_eq!(weights.classifier_weight, vec![4.0f32, 5.0, 6.0]);
        assert!(
            (weights.classifier_bias - 1.0f32).abs() < 1e-6,
            "bias mismatch: {}",
            weights.classifier_bias
        );
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_cross_encoder_weights_shape_mismatch_error() {
        // classifier.weight [1, 4] with hidden_size=3 must yield ShapeMismatch
        let path = temp_path("lattice_ce_weights_mismatch");
        write_classifier_safetensors(
            &path,
            r#""dtype":"F32","shape":[1,4]"#,
            &[1.0f32, 2.0, 3.0, 4.0],
            0.0,
            16,
        );
        let st = SafetensorsFile::open(&path).unwrap();
        let err = st
            .load_cross_encoder_weights(3)
            .expect_err("shape mismatch should be an error");
        assert!(
            matches!(err, InferenceError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err:?}"
        );
        fs::remove_file(&path).ok();
    }
}
