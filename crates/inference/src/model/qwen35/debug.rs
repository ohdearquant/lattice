use super::cache::{ForwardScratch, KvCache};
use super::model::Qwen35Model;
use super::norm::qwen35_rms_norm;
use super::weights::{AttentionWeights, FeedForwardWeights};
use crate::attention::gdn::GatedDeltaNetState;
use crate::attention::gdn_fused::gated_delta_net_step_fused;
use crate::forward::cpu::matmul_bt;

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
    pub fn forward_prompt_debug(&self, prompt_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.config;
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
        scratch.logits[..cfg.vocab_size].to_vec()
    }
}

fn debug_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let std = (x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
    (mean, std)
}
