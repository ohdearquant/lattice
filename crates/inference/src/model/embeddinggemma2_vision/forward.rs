//! EmbeddingGemma 2 vision tower forward, CPU f32 over the valid patches.
//!
//! Grounded in the reference source `transformers/models/gemma4/modeling_gemma4.py`
//! (patch embedder 587-629, pooler 632-696, MLP 699-718, axial RoPE 726-777,
//! attention 905-968, encoder layer 974-1024, vision model 1977-2044) and
//! `models/embedding_gemma2/modeling_embedding_gemma2.py` (bridge 565-591).
//! Only the valid patches are ever materialized: padded reference rows are
//! excluded as attention keys and pooling only accumulates valid rows, so the
//! valid-row outputs are the same function this computes.

use super::{VisionConfig, VisionWeights};
use crate::error::InferenceError;
use crate::forward::cpu::{matmul_bt, rms_norm};
use crate::model::embeddinggemma2::bidirectional_attention;
use crate::model::embeddinggemma2_image::PreparedImage;
use crate::model::gemma4_ops::{
    gemma4_geglu_mlp, gemma4_qk_norm_v_unscaled, vision_axial_rope_apply,
    vision_axial_rope_inv_freq,
};

const POOLING_KERNEL: usize = 3;

/// One requested-prefix of the vision tower, kept at the boundary a reference capture uses.
pub(crate) struct VisionStages {
    pub(crate) patch: Vec<f32>,
    pub(crate) layer_0: Vec<f32>,
    pub(crate) layer_8: Vec<f32>,
    pub(crate) layer_15: Vec<f32>,
    pub(crate) encoder: Vec<f32>,
    pub(crate) pooler_values: Vec<f32>,
    pub(crate) pooler_occupancy: Vec<bool>,
    pub(crate) valid: Vec<f32>,
    pub(crate) projected: Vec<f32>,
    pub(crate) layer_0_q_normed: Vec<f32>,
    pub(crate) layer_0_k_normed: Vec<f32>,
    pub(crate) layer_0_q_rotated: Vec<f32>,
    pub(crate) layer_0_k_rotated: Vec<f32>,
    pub(crate) hidden_size: usize,
    pub(crate) text_hidden: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
}

pub(crate) struct VisionEncoding {
    pub(crate) stages: VisionStages,
    pub(crate) soft_rows: usize,
}

fn vision_error(message: impl Into<String>) -> InferenceError {
    InferenceError::Inference(format!("EmbeddingGemma 2 vision: {}", message.into()))
}

fn stage_is_captured(index: usize) -> bool {
    index == 0 || index == 8 || index == 15
}

impl VisionWeights {
    fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Patch embedder: `2 * (p - 0.5)`, the bias-free 768-to-768 projection and the
    /// learned `table[0][x] + table[1][y]` position sum, over the valid rows.
    pub(crate) fn patch_embed(&self, prepared: &PreparedImage) -> Result<Vec<f32>, InferenceError> {
        let cfg = self.config();
        let hidden = cfg.hidden_size;
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        let valid = prepared.valid_count();
        if valid == 0 {
            return Err(vision_error("a positive valid patch count"));
        }
        let patch_values = prepared.patch_values();
        if patch_values.len() != valid * patch_dim {
            return Err(vision_error(format!(
                "patch values must hold {valid}x{patch_dim} values, got {}",
                patch_values.len()
            )));
        }
        if self.input_proj.len() != hidden * patch_dim {
            return Err(vision_error("input projection has the wrong shape"));
        }
        let mut centered = vec![0f32; valid * patch_dim];
        for (out, &value) in centered.iter_mut().zip(patch_values) {
            *out = 2.0 * (value - 0.5);
        }
        let mut hidden_states = vec![0f32; valid * hidden];
        matmul_bt(
            &centered,
            &self.input_proj,
            &mut hidden_states,
            valid,
            patch_dim,
            hidden,
        );
        let size = cfg.position_embedding_size;
        let table = &self.position_embedding_table;
        if table.len() != 2 * size * hidden {
            return Err(vision_error("position embedding table has the wrong shape"));
        }
        let positions = prepared.positions();
        if positions.len() != valid {
            return Err(vision_error(format!(
                "position count must be {valid}, got {}",
                positions.len()
            )));
        }
        for (row, &[x, y]) in positions.iter().enumerate() {
            if x >= size || y >= size {
                return Err(vision_error(format!(
                    "patch position ({x},{y}) is outside the {size}-entry position table"
                )));
            }
            let row_out = &mut hidden_states[row * hidden..(row + 1) * hidden];
            let x_row = &table[x * hidden..(x + 1) * hidden];
            let y_row = &table[size * hidden + y * hidden..size * hidden + (y + 1) * hidden];
            for (out, (&x_value, &y_value)) in row_out.iter_mut().zip(x_row.iter().zip(y_row)) {
                *out += x_value + y_value;
            }
        }
        Ok(hidden_states)
    }

    /// One encoder layer in place at `index`; the arithmetic is identical at every index.
    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &self,
        index: usize,
        hidden_states: &mut [f32],
        tokens: usize,
        positions: &[[usize; 2]],
        inv_freq: &[f32],
        capture: bool,
        captures: &mut VisionStages,
    ) -> Result<(), InferenceError> {
        let cfg = self.config();
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let eps = cfg.rms_norm_eps;
        let layer = self
            .layers
            .get(index)
            .ok_or_else(|| vision_error(format!("missing weights for vision layer {index}")))?;

        let mut x = hidden_states.to_vec();
        rms_norm(&mut x, &layer.input_layernorm, hidden, eps);

        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let mut q = vec![0f32; tokens * q_dim];
        let mut k = vec![0f32; tokens * kv_dim];
        let mut v = vec![0f32; tokens * kv_dim];
        matmul_bt(&x, &layer.q_proj, &mut q, tokens, hidden, q_dim);
        matmul_bt(&x, &layer.k_proj, &mut k, tokens, hidden, kv_dim);
        matmul_bt(&x, &layer.v_proj, &mut v, tokens, hidden, kv_dim);
        gemma4_qk_norm_v_unscaled(
            &mut q,
            &mut k,
            &mut v,
            &layer.q_norm,
            &layer.k_norm,
            head_dim,
            eps,
        );
        if capture {
            captures.layer_0_q_normed = q.clone();
            captures.layer_0_k_normed = k.clone();
        }
        vision_axial_rope_apply(&mut q, positions, inv_freq, heads, head_dim);
        vision_axial_rope_apply(&mut k, positions, inv_freq, kv_heads, head_dim);
        if capture {
            captures.layer_0_q_rotated = q.clone();
            captures.layer_0_k_rotated = k.clone();
        }

        let context = bidirectional_attention(&q, &k, &v, tokens, heads, kv_heads, head_dim, None);
        let mut attention = vec![0f32; tokens * hidden];
        matmul_bt(
            &context,
            &layer.o_proj,
            &mut attention,
            tokens,
            q_dim,
            hidden,
        );
        rms_norm(&mut attention, &layer.post_attention_layernorm, hidden, eps);
        for (h, a) in hidden_states.iter_mut().zip(&attention) {
            *h += *a;
        }

        let mut feed_forward = hidden_states.to_vec();
        rms_norm(
            &mut feed_forward,
            &layer.pre_feedforward_layernorm,
            hidden,
            eps,
        );
        let mut gate = vec![0f32; tokens * cfg.intermediate_size];
        let mut up = vec![0f32; tokens * cfg.intermediate_size];
        let mut mlp = vec![0f32; tokens * hidden];
        gemma4_geglu_mlp(
            &feed_forward,
            &layer.gate_proj,
            &layer.up_proj,
            &layer.down_proj,
            tokens,
            hidden,
            cfg.intermediate_size,
            &mut gate,
            &mut up,
            &mut mlp,
        );
        rms_norm(&mut mlp, &layer.post_feedforward_layernorm, hidden, eps);
        for (h, m) in hidden_states.iter_mut().zip(&mlp) {
            *h += *m;
        }

        Ok(())
    }

    /// Full tower over the valid patches, capturing the named reference stages.
    pub(crate) fn tower_forward(
        &self,
        patch: &[f32],
        positions: &[[usize; 2]],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<VisionStages, InferenceError> {
        let cfg = self.config();
        let hidden = cfg.hidden_size;
        let tokens = positions.len();
        if tokens == 0 || patch.len() != tokens * hidden {
            return Err(vision_error(format!(
                "patch rows must be {tokens}x{hidden}, got {}",
                patch.len()
            )));
        }
        if cfg.num_hidden_layers != self.layers.len() {
            return Err(vision_error(
                "the layer count does not match the loaded weights",
            ));
        }
        let inv_freq = vision_axial_rope_inv_freq(cfg.head_dim, cfg.rope_theta);
        let mut captures = VisionStages {
            patch: patch.to_vec(),
            layer_0: Vec::new(),
            layer_8: Vec::new(),
            layer_15: Vec::new(),
            encoder: Vec::new(),
            pooler_values: Vec::new(),
            pooler_occupancy: Vec::new(),
            valid: Vec::new(),
            projected: Vec::new(),
            layer_0_q_normed: Vec::new(),
            layer_0_k_normed: Vec::new(),
            layer_0_q_rotated: Vec::new(),
            layer_0_k_rotated: Vec::new(),
            hidden_size: cfg.hidden_size,
            text_hidden: cfg.text_hidden_size,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
        };
        let mut hidden_states = patch.to_vec();
        for index in 0..cfg.num_hidden_layers {
            if should_cancel() {
                return Err(vision_error("cancelled during the vision tower"));
            }
            self.run_layer(
                index,
                &mut hidden_states,
                tokens,
                positions,
                &inv_freq,
                index == 0,
                &mut captures,
            )?;
            if stage_is_captured(index) {
                match index {
                    0 => captures.layer_0 = hidden_states.clone(),
                    8 => captures.layer_8 = hidden_states.clone(),
                    _ => captures.layer_15 = hidden_states.clone(),
                }
            }
        }
        captures.encoder = hidden_states.clone();
        Ok(captures)
    }

    /// Coordinate pooling at kernel 3 and the `sqrt(hidden)` scale, keeping
    /// exactly the occupied rows. Grid sides are multiples of three, so every
    /// bucket holds only valid rows and the valid-only form needs no padded-row
    /// bucket.
    pub(crate) fn pool(
        &self,
        encoder: &[f32],
        positions: &[[usize; 2]],
    ) -> Result<(Vec<f32>, Vec<bool>), InferenceError> {
        let cfg = self.config();
        let hidden = cfg.hidden_size;
        let tokens = positions.len();
        if tokens == 0 || encoder.len() != tokens * hidden {
            return Err(vision_error(format!(
                "encoder rows must be {tokens}x{hidden}, got {}",
                encoder.len()
            )));
        }
        let mut grid_w = 0usize;
        let mut grid_h = 0usize;
        for &[x, y] in positions {
            grid_w = grid_w.max(x + 1);
            grid_h = grid_h.max(y + 1);
        }
        if grid_w == 0 || grid_h == 0 {
            return Err(vision_error("patch grid is empty"));
        }
        if !grid_w.is_multiple_of(POOLING_KERNEL) || !grid_h.is_multiple_of(POOLING_KERNEL) {
            return Err(vision_error("patch grid sides must be multiples of 3"));
        }
        let buckets_w = grid_w / POOLING_KERNEL;
        let buckets_h = grid_h / POOLING_KERNEL;
        let soft_rows = buckets_w * buckets_h;
        let mut sums = vec![0f32; soft_rows * hidden];
        let mut counts = vec![0usize; soft_rows];
        for (row, &[x, y]) in positions.iter().enumerate() {
            if x >= grid_w || y >= grid_h {
                return Err(vision_error("patch position is outside its grid"));
            }
            let bucket = (x / POOLING_KERNEL) + buckets_w * (y / POOLING_KERNEL);
            counts[bucket] += 1;
            let source = &encoder[row * hidden..(row + 1) * hidden];
            let target = &mut sums[bucket * hidden..(bucket + 1) * hidden];
            for (accumulator, &value) in target.iter_mut().zip(source) {
                *accumulator += value;
            }
        }
        let scale = (hidden as f64).sqrt() as f32;
        let mut values = vec![0f32; soft_rows * hidden];
        let mut occupancy = vec![false; soft_rows];
        for bucket in 0..soft_rows {
            let count = counts[bucket];
            if count == 0 {
                continue;
            }
            occupancy[bucket] = true;
            let inverse = 1.0 / count as f32;
            let source = &sums[bucket * hidden..(bucket + 1) * hidden];
            let target = &mut values[bucket * hidden..(bucket + 1) * hidden];
            for (out, &value) in target.iter_mut().zip(source) {
                *out = value * inverse * scale;
            }
        }
        Ok((values, occupancy))
    }

    /// Bridge: scale-free RMSNorm at the vision epsilon, then the bias-free
    /// 768-to-`text_hidden` projection.
    pub(crate) fn bridge(&self, values: &[f32], rows: usize) -> Result<Vec<f32>, InferenceError> {
        let cfg = self.config();
        let hidden = cfg.hidden_size;
        let text_hidden = cfg.text_hidden_size;
        if rows == 0 || values.len() != rows * hidden {
            return Err(vision_error(format!(
                "pooled rows must be {rows}x{hidden}, got {}",
                values.len()
            )));
        }
        if self.embedding_projection.len() != text_hidden * hidden {
            return Err(vision_error("embedding projection has the wrong shape"));
        }
        let ones = vec![1.0f32; hidden];
        let mut normed = values.to_vec();
        rms_norm(&mut normed, &ones, hidden, cfg.rms_norm_eps);
        let mut projected = vec![0f32; rows * text_hidden];
        matmul_bt(
            &normed,
            &self.embedding_projection,
            &mut projected,
            rows,
            hidden,
            text_hidden,
        );
        Ok(projected)
    }

    /// Full image encoding: patch embed, tower, pooling, bridge.
    pub(crate) fn encode(
        &self,
        prepared: &PreparedImage,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<VisionEncoding, InferenceError> {
        if should_cancel() {
            return Err(vision_error("cancelled before the patch embed"));
        }
        let patch = self.patch_embed(prepared)?;
        let positions = prepared.positions();
        let stages = self.tower_forward(&patch, positions, should_cancel)?;
        if should_cancel() {
            return Err(vision_error("cancelled before pooling"));
        }
        let (pooler_values, pooler_occupancy) = self.pool(&stages.encoder, positions)?;
        let soft_rows = prepared.soft_rows();
        if pooler_values.len() != soft_rows * self.config().hidden_size {
            return Err(vision_error(
                "pooled rows do not match the prepared soft-row count",
            ));
        }
        let projected = self.bridge(&pooler_values, soft_rows)?;
        Ok(self.finish_encoding(
            stages,
            pooler_values,
            pooler_occupancy,
            projected,
            soft_rows,
        ))
    }

    /// Fills the pooled and bridged tiers of `stages` and pairs them with the soft-row count.
    /// Shared by the whole-CPU path ([`VisionWeights::encode`]) and any backend that produces the
    /// same tower output, so the two package an encoding identically.
    pub(crate) fn finish_encoding(
        &self,
        mut stages: VisionStages,
        pooler_values: Vec<f32>,
        pooler_occupancy: Vec<bool>,
        projected: Vec<f32>,
        soft_rows: usize,
    ) -> VisionEncoding {
        stages.pooler_values = pooler_values.clone();
        stages.pooler_occupancy = pooler_occupancy;
        stages.valid = pooler_values;
        stages.projected = projected;
        VisionEncoding { stages, soft_rows }
    }
}
