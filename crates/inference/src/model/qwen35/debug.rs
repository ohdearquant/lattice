//! Qwen3.5 debug methods and debug statistics.
use super::cache::{ForwardScratch, KvCache};
use super::model::Qwen35Model;
use super::norm::qwen35_rms_norm;
use super::weights::{AttentionWeights, FeedForwardWeights};
use crate::attention::gdn::GatedDeltaNetState;
#[cfg(test)]
use crate::attention::gdn::GatedDeltaNetWeights;
#[cfg(test)]
use crate::attention::gdn_fused::GatedDeltaNetFusedScratch;
use crate::attention::gdn_fused::gated_delta_net_step_fused;
use crate::error::InferenceError;
use crate::forward::cpu::matmul_bt;
#[cfg(test)]
use crate::quant::quarot::hadamard::RandomizedHadamard;

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct ForwardTrace {
    pub(crate) mixer_residuals: Vec<Vec<f32>>,
    pub(crate) block_residuals: Vec<Vec<f32>>,
    pub(crate) logits: Vec<f32>,
    pub(crate) gdn_layer0: Option<GdnMixerTrace>,
}

/// Test-only capture of the first GDN mixer's first-divergence boundary.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct GdnMixerTrace {
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) z: Vec<f32>,
    pub(crate) a: Vec<f32>,
    pub(crate) b: Vec<f32>,
    pub(crate) conv: Vec<f32>,
    pub(crate) state: Vec<f32>,
    pub(crate) gated: Vec<f32>,
    pub(crate) out_proj: Vec<f32>,
}

/// Test-only residual-basis controls for the QuaRot FP16 localization ladder.
#[cfg(test)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ResidualBoundaryControl {
    pub(crate) rotate_embedding_output: bool,
    pub(crate) original_rmsnorm_boundaries: bool,
    pub(crate) original_gqa_boundaries: bool,
    pub(crate) original_gdn_boundaries: bool,
    pub(crate) original_mlp_boundaries: bool,
    pub(crate) rotate_endpoint_input: bool,
    pub(crate) unrotate_endpoint_input: bool,
}

impl Qwen35Model {
    /// **Unstable**: debug single-token forward pass returning raw logits.
    pub fn forward_single_token_debug(&self, token_id: u32) -> Vec<f32> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();
        scratch.ensure_capacity(cfg, 1);

        let embed_start = token_id as usize * hidden;
        scratch.hidden[..hidden]
            .copy_from_slice(&self.weights.embed_tokens[embed_start..embed_start + hidden]);

        let h = &scratch.hidden[..hidden];
        let (hmean, hstd) = debug_stats(h);
        eprintln!("  [embed] mean={hmean:.6}, std={hstd:.6}");

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        for layer_i in 0..cfg.num_hidden_layers {
            let (hmean, hstd) = self.forward_debug_layer(
                layer_i,
                &mut linear_idx,
                &mut full_idx,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
                hidden,
            );
            let layer_type = if cfg.is_full_attention(layer_i) {
                "full"
            } else {
                "linear"
            };
            let h = &scratch.hidden[..hidden];
            let (hmean2, hstd2) = debug_stats(h);
            eprintln!(
                "  [L{layer_i:02} {layer_type}] attn: mean={hmean:.6} std={hstd:.6} | ffn: mean={hmean2:.6} std={hstd2:.6}"
            );
        }

        qwen35_rms_norm(
            &mut scratch.hidden[..hidden],
            &self.weights.final_norm,
            hidden,
            cfg.rms_norm_eps,
        );
        matmul_bt(
            &scratch.hidden[..hidden],
            &self.weights.embed_tokens,
            &mut scratch.logits[..cfg.vocab_size],
            1,
            hidden,
            cfg.vocab_size,
        );
        scratch.logits[..cfg.vocab_size].to_vec()
    }

    fn forward_debug_layer(
        &self,
        layer_i: usize,
        linear_idx: &mut usize,
        full_idx: &mut usize,
        gdn_states: &mut [GatedDeltaNetState],
        kv_cache: &mut KvCache,
        scratch: &mut ForwardScratch,
        hidden: usize,
    ) -> (f32, f32) {
        let cfg = &self.config;
        let (attn_weights, common) = &self.weights.layers[layer_i];
        scratch.residual[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
        qwen35_rms_norm(
            &mut scratch.hidden[..hidden],
            &common.input_layernorm,
            hidden,
            cfg.rms_norm_eps,
        );

        match attn_weights {
            AttentionWeights::Linear(gdn_w) => {
                gated_delta_net_step_fused(
                    &scratch.hidden[..hidden],
                    &mut gdn_states[*linear_idx],
                    gdn_w,
                    cfg,
                    &mut scratch.gdn_scratch,
                    &mut scratch.attn_out[..hidden],
                    self.lora.as_ref(),
                    layer_i,
                );
                *linear_idx += 1;
            }
            AttentionWeights::Full(full_w) => {
                scratch.attn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
                self.full_attention_step_from_attn_out(
                    full_w, *full_idx, layer_i, 0, kv_cache, scratch, hidden,
                );
                *full_idx += 1;
            }
        }

        for i in 0..hidden {
            scratch.hidden[i] = scratch.residual[i] + scratch.attn_out[i];
        }

        let (hmean, hstd) = debug_stats(&scratch.hidden[..hidden]);

        scratch.residual[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
        qwen35_rms_norm(
            &mut scratch.hidden[..hidden],
            &common.post_attention_layernorm,
            hidden,
            cfg.rms_norm_eps,
        );
        scratch.ffn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
        match &common.ffn {
            FeedForwardWeights::Dense(dense) => {
                self.dense_ffn_step_from_ffn_out(dense, layer_i, scratch, hidden);
            }
            FeedForwardWeights::Moe(moe) => {
                super::moe::moe_ffn_step(moe, scratch, hidden);
            }
        }
        for i in 0..hidden {
            scratch.hidden[i] = scratch.residual[i] + scratch.ffn_out[i];
        }

        (hmean, hstd)
    }

    /// **Unstable**: debug prompt forward pass returning final logits.
    ///
    /// Errors if `prompt_ids` is empty or if `prompt_ids.len() > self.max_context()`.
    /// The precomputed RoPE table only covers positions `0..max_context()`; without
    /// this guard a long prompt would panic on an out-of-range slice index.
    pub fn forward_prompt_debug(&self, prompt_ids: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.config;
        if prompt_ids.is_empty() {
            return Err(InferenceError::Inference(
                "forward_prompt_debug: prompt_ids must not be empty".into(),
            ));
        }
        let max_context = self.max_context();
        if prompt_ids.len() > max_context {
            return Err(InferenceError::Inference(format!(
                "forward_prompt_debug: prompt_ids.len() ({}) exceeds RoPE table capacity ({}); \
                 load the model with a larger context table or use a shorter prompt",
                prompt_ids.len(),
                max_context,
            )));
        }
        let _hidden = cfg.hidden_size;
        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();

        for (pos, &token_id) in prompt_ids.iter().enumerate() {
            self.forward_step(token_id, pos, &mut gdn_states, &mut kv_cache, &mut scratch);
            if pos < prompt_ids.len() - 1 {
                kv_cache.seq_len += 1;
            }
        }
        Ok(scratch.logits[..cfg.vocab_size].to_vec())
    }

    #[cfg(test)]
    pub(crate) fn forward_prompt_residual_trace_debug(
        &self,
        prompt_ids: &[u32],
    ) -> Result<ForwardTrace, InferenceError> {
        let rotation = RandomizedHadamard::new(0, self.config.hidden_size)?;
        self.forward_prompt_residual_trace_debug_with_boundary_control(
            prompt_ids,
            ResidualBoundaryControl::default(),
            &rotation,
        )
    }

    /// Runs the test-only trace path with selected original-basis boundaries.
    #[cfg(test)]
    pub(crate) fn forward_prompt_residual_trace_debug_with_boundary_control(
        &self,
        prompt_ids: &[u32],
        control: ResidualBoundaryControl,
        rotation: &RandomizedHadamard,
    ) -> Result<ForwardTrace, InferenceError> {
        if prompt_ids.is_empty() {
            return Err(InferenceError::Inference(
                "forward_prompt_residual_trace_debug: prompt_ids must not be empty".into(),
            ));
        }

        let cfg = &self.config;
        if prompt_ids.len() > self.max_context() {
            return Err(InferenceError::Inference(format!(
                "forward_prompt_residual_trace_debug: prompt_ids.len() ({}) exceeds RoPE table capacity ({})",
                prompt_ids.len(),
                self.max_context()
            )));
        }

        let hidden = cfg.hidden_size;
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();
        let mut mixer_trace = Vec::with_capacity(cfg.num_hidden_layers);
        let mut block_trace = Vec::with_capacity(cfg.num_hidden_layers);
        let mut gdn_layer0 = None;

        for (position, &token_id) in prompt_ids.iter().enumerate() {
            scratch.ensure_capacity(cfg, kv_cache.seq_len + 1);
            let embed_start = token_id as usize * hidden;
            scratch.hidden[..hidden]
                .copy_from_slice(&self.weights.embed_tokens[embed_start..embed_start + hidden]);
            if control.rotate_embedding_output {
                rotation.apply(&mut scratch.hidden[..hidden])?;
            }

            let mut linear_idx = 0usize;
            let mut full_idx = 0usize;
            for layer_i in 0..cfg.num_hidden_layers {
                let (attn_weights, common) = &self.weights.layers[layer_i];
                scratch.residual[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
                let gdn_state_idx = match attn_weights {
                    AttentionWeights::Linear(_) => Some(linear_idx),
                    AttentionWeights::Full(_) => None,
                };
                let original_attention_boundary = match attn_weights {
                    AttentionWeights::Linear(_) => control.original_gdn_boundaries,
                    AttentionWeights::Full(_) => control.original_gqa_boundaries,
                };
                if original_attention_boundary {
                    scratch.hidden[..hidden].copy_from_slice(&scratch.residual[..hidden]);
                    rotation.apply_inverse(&mut scratch.hidden[..hidden])?;
                    qwen35_rms_norm(
                        &mut scratch.hidden[..hidden],
                        &common.input_layernorm,
                        hidden,
                        cfg.rms_norm_eps,
                    );
                    self.run_attention_layer(
                        layer_i,
                        attn_weights,
                        &mut linear_idx,
                        &mut full_idx,
                        position,
                        &mut gdn_states,
                        &mut kv_cache,
                        &mut scratch,
                        hidden,
                    );
                    rotation.apply(&mut scratch.attn_out[..hidden])?;
                } else if control.original_rmsnorm_boundaries {
                    scratch.hidden[..hidden].copy_from_slice(&scratch.residual[..hidden]);
                    rotation.apply_inverse(&mut scratch.hidden[..hidden])?;
                    qwen35_rms_norm(
                        &mut scratch.hidden[..hidden],
                        &common.input_layernorm,
                        hidden,
                        cfg.rms_norm_eps,
                    );
                    rotation.apply(&mut scratch.hidden[..hidden])?;
                    self.run_attention_layer(
                        layer_i,
                        attn_weights,
                        &mut linear_idx,
                        &mut full_idx,
                        position,
                        &mut gdn_states,
                        &mut kv_cache,
                        &mut scratch,
                        hidden,
                    );
                } else {
                    qwen35_rms_norm(
                        &mut scratch.hidden[..hidden],
                        &common.input_layernorm,
                        hidden,
                        cfg.rms_norm_eps,
                    );
                    self.run_attention_layer(
                        layer_i,
                        attn_weights,
                        &mut linear_idx,
                        &mut full_idx,
                        position,
                        &mut gdn_states,
                        &mut kv_cache,
                        &mut scratch,
                        hidden,
                    );
                }
                if position + 1 == prompt_ids.len()
                    && layer_i == 0
                    && let (Some(gdn_w), Some(state_idx)) = (
                        match attn_weights {
                            AttentionWeights::Linear(weights) => Some(weights),
                            AttentionWeights::Full(_) => None,
                        },
                        gdn_state_idx,
                    )
                {
                    gdn_layer0 = Some(capture_gdn_mixer_trace(
                        &scratch.hidden[..hidden],
                        &gdn_states[state_idx],
                        gdn_w,
                        cfg,
                        &scratch.gdn_scratch,
                        &scratch.attn_out[..hidden],
                        self.lora.as_ref(),
                        layer_i,
                    ));
                }
                for i in 0..hidden {
                    scratch.hidden[i] = scratch.residual[i] + scratch.attn_out[i];
                }
                if position + 1 == prompt_ids.len() {
                    mixer_trace.push(scratch.hidden[..hidden].to_vec());
                }

                scratch.residual[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
                if control.original_mlp_boundaries {
                    scratch.hidden[..hidden].copy_from_slice(&scratch.residual[..hidden]);
                    rotation.apply_inverse(&mut scratch.hidden[..hidden])?;
                    qwen35_rms_norm(
                        &mut scratch.hidden[..hidden],
                        &common.post_attention_layernorm,
                        hidden,
                        cfg.rms_norm_eps,
                    );
                    scratch.ffn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
                    self.run_ffn_layer(layer_i, common, &mut scratch, hidden);
                    rotation.apply(&mut scratch.ffn_out[..hidden])?;
                } else {
                    if control.original_rmsnorm_boundaries {
                        scratch.hidden[..hidden].copy_from_slice(&scratch.residual[..hidden]);
                        rotation.apply_inverse(&mut scratch.hidden[..hidden])?;
                        qwen35_rms_norm(
                            &mut scratch.hidden[..hidden],
                            &common.post_attention_layernorm,
                            hidden,
                            cfg.rms_norm_eps,
                        );
                        rotation.apply(&mut scratch.hidden[..hidden])?;
                    } else {
                        qwen35_rms_norm(
                            &mut scratch.hidden[..hidden],
                            &common.post_attention_layernorm,
                            hidden,
                            cfg.rms_norm_eps,
                        );
                    }
                    scratch.ffn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
                    self.run_ffn_layer(layer_i, common, &mut scratch, hidden);
                }
                for i in 0..hidden {
                    scratch.hidden[i] = scratch.residual[i] + scratch.ffn_out[i];
                }

                if position + 1 == prompt_ids.len() {
                    block_trace.push(scratch.hidden[..hidden].to_vec());
                }
            }

            if control.rotate_endpoint_input {
                rotation.apply(&mut scratch.hidden[..hidden])?;
            }
            if control.unrotate_endpoint_input {
                rotation.apply_inverse(&mut scratch.hidden[..hidden])?;
            }
            qwen35_rms_norm(
                &mut scratch.hidden[..hidden],
                &self.weights.final_norm,
                hidden,
                cfg.rms_norm_eps,
            );
            let logits_weight = self.weights.logits_weight();
            matmul_bt(
                &scratch.hidden[..hidden],
                logits_weight,
                &mut scratch.logits[..cfg.vocab_size],
                1,
                hidden,
                cfg.vocab_size,
            );
            if position + 1 < prompt_ids.len() {
                kv_cache.seq_len += 1;
            }
        }

        Ok(ForwardTrace {
            mixer_residuals: mixer_trace,
            block_residuals: block_trace,
            logits: scratch.logits[..cfg.vocab_size].to_vec(),
            gdn_layer0,
        })
    }
}

#[cfg(test)]
fn capture_gdn_mixer_trace(
    input: &[f32],
    state: &GatedDeltaNetState,
    weights: &GatedDeltaNetWeights,
    cfg: &crate::model::qwen35_config::Qwen35Config,
    scratch: &GatedDeltaNetFusedScratch,
    output: &[f32],
    lora: &dyn crate::lora_hook::LoraHook,
    layer_idx: usize,
) -> GdnMixerTrace {
    let hidden = cfg.hidden_size;
    let num_heads = cfg.linear_num_key_heads;
    let key_dim = cfg.linear_key_head_dim;
    let q_total = num_heads * key_dim;
    let k_total = q_total;
    let v_offset = q_total + k_total;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let value_heads = cfg.linear_num_value_heads();

    let mut b = vec![0.0; value_heads];
    matmul_bt(input, &weights.in_proj_b, &mut b, 1, hidden, value_heads);
    lora.apply(layer_idx, "in_proj_b", input, &mut b);

    GdnMixerTrace {
        q: scratch.qkv_proj[..q_total].to_vec(),
        k: scratch.qkv_proj[q_total..v_offset].to_vec(),
        v: scratch.qkv_proj[v_offset..qkv_dim].to_vec(),
        z: scratch.z_proj[..output_dim].to_vec(),
        a: scratch.alpha_proj[..value_heads].to_vec(),
        b,
        conv: scratch.conv_output[..qkv_dim].to_vec(),
        state: state.s_matrices.clone(),
        gated: scratch.gated_norm_buf[..output_dim].to_vec(),
        out_proj: output[..hidden].to_vec(),
    }
}

fn debug_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let std = (x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
    (mean, std)
}
