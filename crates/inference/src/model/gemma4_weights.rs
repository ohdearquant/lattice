//! Gemma 4 E2B weight storage (ADR-082 stage 5).
//!
//! Mirrors the qwen35 weight-storage split (`qwen35::weights`): a per-layer
//! struct plus a top-level container. Per ADR-082 Amendment 1, K/V-shared
//! layers carry `None` for `k_proj`/`v_proj`/`k_norm`. Their checkpoint
//! tensors exist but are tolerate-and-skipped at load time
//! (`gemma4_loading`) and never wired into the forward pass; the forward
//! pass instead resolves those layers' K/V through
//! [`crate::model::gemma4_cache::Gemma4KvCache`]'s donor-slot indirection.

use crate::error::InferenceError;
use crate::weights::SafetensorsFile;
use crate::weights::half_bits::bf16_bits_to_f32;

/// Per-layer Gemma 4 weights.
pub(crate) struct Gemma4LayerWeights {
    pub(crate) input_layernorm: Vec<f32>,            // [hidden]
    pub(crate) post_attention_layernorm: Vec<f32>,   // [hidden]
    pub(crate) pre_feedforward_layernorm: Vec<f32>,  // [hidden]
    pub(crate) post_feedforward_layernorm: Vec<f32>, // [hidden]
    pub(crate) post_per_layer_input_norm: Vec<f32>,  // [hidden]
    /// `layer_scalar` checkpoint tensor, shape `[1]`, stored unwrapped.
    pub(crate) layer_scalar: f32,
    pub(crate) per_layer_input_gate: Vec<f32>, // [per_layer_dim, hidden]
    pub(crate) per_layer_projection: Vec<f32>, // [hidden, per_layer_dim]

    pub(crate) q_proj: Vec<f32>, // [num_attention_heads * head_w, hidden]
    pub(crate) o_proj: Vec<f32>, // [hidden, num_attention_heads * head_w]
    pub(crate) q_norm: Vec<f32>, // [head_w]

    /// `None` on KV-shared layers (ADR-082 Amendment 1): those layers have
    /// no `k_proj`/`v_proj`/`k_norm` weights loaded, and read their donor's
    /// K/V via [`crate::model::gemma4_cache::Gemma4KvCache`] instead.
    pub(crate) k_proj: Option<Vec<f32>>, // [kv_dim, hidden]
    pub(crate) v_proj: Option<Vec<f32>>, // [kv_dim, hidden]
    pub(crate) k_norm: Option<Vec<f32>>, // [head_w]

    pub(crate) gate_proj: Vec<f32>, // [mlp_dim, hidden]
    pub(crate) up_proj: Vec<f32>,   // [mlp_dim, hidden]
    pub(crate) down_proj: Vec<f32>, // [hidden, mlp_dim]
}

/// **Unstable**: Gemma 4 E2B weight storage; layout tied to checkpoint format.
pub(crate) struct Gemma4Weights {
    pub(crate) embed_tokens: Vec<f32>,               // [vocab, hidden]
    pub(crate) norm: Vec<f32>,                       // [hidden]
    pub(crate) per_layer_model_projection: Vec<f32>, // [num_hidden_layers * per_layer_dim, hidden]
    pub(crate) per_layer_projection_norm: Vec<f32>,  // [per_layer_dim]
    pub(crate) layers: Vec<Gemma4LayerWeights>,
}

/// The per-layer embedding table (`embed_tokens_per_layer`,
/// `[vocab, num_hidden_layers * per_layer_dim]`), read in place from the
/// checkpoint's own bf16 bytes instead of being widened to f32.
///
/// The table holds 2.35B parameters, so an f32 copy alone is 8.75 GiB, yet the
/// forward pass reads exactly one row per token. The model keeps the
/// [`SafetensorsFile`] (and so its mapping) for its whole lifetime and widens
/// only the requested row; widening bf16 to f32 is exact, so the values match
/// an f32-materialized table bit for bit.
pub(crate) struct PerLayerEmbeddings {
    file: SafetensorsFile,
    tensor_name: String,
    rows: usize,
    row_len: usize,
}

impl PerLayerEmbeddings {
    /// Takes ownership of `file` and refuses a table that is missing, is not
    /// `[rows, row_len]`, is not BF16, or holds a non-finite value.
    pub(super) fn new(
        file: SafetensorsFile,
        tensor_name: String,
        rows: usize,
        row_len: usize,
    ) -> Result<Self, InferenceError> {
        let expected = [rows, row_len];
        match file.tensor_shape(&tensor_name) {
            Some(actual) if actual == expected => {}
            Some(actual) => {
                return Err(InferenceError::ShapeMismatch {
                    name: tensor_name,
                    expected: expected.to_vec(),
                    actual: actual.to_vec(),
                });
            }
            None => return Err(InferenceError::MissingTensor(tensor_name)),
        }
        file.bf16_payload(&tensor_name)?;
        Ok(Self {
            file,
            tensor_name,
            rows,
            row_len,
        })
    }

    /// Row `row` widened to f32 and multiplied by `scale`.
    pub(crate) fn scaled_row(&self, row: usize, scale: f32) -> Result<Vec<f32>, InferenceError> {
        if row >= self.rows {
            return Err(InferenceError::InvalidInput(format!(
                "gemma4 per-layer embedding row {row} out of range ({} rows)",
                self.rows
            )));
        }
        let payload = self.file.bf16_payload(&self.tensor_name)?;
        let row_bytes = self.row_len * 2;
        let bytes = payload
            .get(row * row_bytes..(row + 1) * row_bytes)
            .ok_or_else(|| {
                InferenceError::InvalidSafetensors(format!(
                    "tensor {} payload is shorter than its declared {} rows",
                    self.tensor_name, self.rows
                ))
            })?;
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| bf16_bits_to_f32(u16::from_le_bytes([pair[0], pair[1]])) * scale)
            .collect())
    }
}

#[cfg(test)]
pub(crate) fn synthetic_safetensors_bytes(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
    let mut entries = Vec::new();
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = payload.len();
        payload.extend_from_slice(bytes);
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        entries.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{start},{}]}}"#,
            dims.join(","),
            payload.len()
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    out
}

#[cfg(test)]
pub(crate) fn per_layer_embeddings_from_bits(
    rows: usize,
    row_len: usize,
    bits: &[u16],
) -> PerLayerEmbeddings {
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let file = SafetensorsFile::from_bytes(synthetic_safetensors_bytes(&[(
        "embed_tokens_per_layer.weight",
        "BF16",
        &[rows, row_len],
        &bytes,
    )]))
    .expect("synthetic per-layer table is a valid safetensors buffer");
    PerLayerEmbeddings::new(
        file,
        "embed_tokens_per_layer.weight".to_string(),
        rows,
        row_len,
    )
    .expect("synthetic per-layer table is finite BF16 of the declared shape")
}
