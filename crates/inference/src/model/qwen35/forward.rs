use super::cache::{ForwardScratch, KvCache};
use super::model::Qwen35Model;
use super::moe::moe_ffn_step;
use super::norm::qwen35_rms_norm;
use super::weights::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
    FullAttentionLayerWeights,
};
use crate::attention::gated::apply_sigmoid_gate;
use crate::attention::gdn::GatedDeltaNetState;
use crate::attention::gdn_fused::gated_delta_net_step_fused;
use crate::forward::cpu::{elementwise_mul, matmul_bt, silu_inplace};

impl Qwen35Model {
    /// Single-token forward pass. Writes logits into scratch.logits.
    pub(crate) fn forward_step(
        &self,
        token_id: u32,
        position: usize,
        gdn_states: &mut [GatedDeltaNetState],
        kv_cache: &mut KvCache,
        scratch: &mut ForwardScratch,
    ) {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;

        scratch.ensure_capacity(cfg, kv_cache.seq_len + 1);

        let embed_start = token_id as usize * hidden;
        scratch.hidden[..hidden]
            .copy_from_slice(&self.weights.embed_tokens[embed_start..embed_start + hidden]);

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        for layer_i in 0..cfg.num_hidden_layers {
            let (attn_weights, common) = &self.weights.layers[layer_i];

            scratch.residual[..hidden].copy_from_slice(&scratch.hidden[..hidden]);

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
                gdn_states,
                kv_cache,
                scratch,
                hidden,
            );

            for i in 0..hidden {
                scratch.hidden[i] = scratch.residual[i] + scratch.attn_out[i];
            }

            scratch.residual[..hidden].copy_from_slice(&scratch.hidden[..hidden]);

            qwen35_rms_norm(
                &mut scratch.hidden[..hidden],
                &common.post_attention_layernorm,
                hidden,
                cfg.rms_norm_eps,
            );

            scratch.ffn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
            self.run_ffn_layer(layer_i, common, scratch, hidden);

            for i in 0..hidden {
                scratch.hidden[i] = scratch.residual[i] + scratch.ffn_out[i];
            }
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
    }

    fn run_attention_layer(
        &self,
        layer_i: usize,
        attn_weights: &AttentionWeights,
        linear_idx: &mut usize,
        full_idx: &mut usize,
        position: usize,
        gdn_states: &mut [GatedDeltaNetState],
        kv_cache: &mut KvCache,
        scratch: &mut ForwardScratch,
        hidden: usize,
    ) {
        match attn_weights {
            AttentionWeights::Linear(gdn_w) => {
                gated_delta_net_step_fused(
                    &scratch.hidden[..hidden],
                    &mut gdn_states[*linear_idx],
                    gdn_w,
                    &self.config,
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
                    full_w, *full_idx, layer_i, position, kv_cache, scratch, hidden,
                );
                *full_idx += 1;
            }
        }
    }

    fn run_ffn_layer(
        &self,
        layer_i: usize,
        common: &CommonLayerWeights,
        scratch: &mut ForwardScratch,
        hidden: usize,
    ) {
        match &common.ffn {
            FeedForwardWeights::Dense(dense) => {
                self.dense_ffn_step_from_ffn_out(dense, layer_i, scratch, hidden);
            }
            FeedForwardWeights::Moe(moe) => {
                moe_ffn_step(moe, scratch, hidden);
            }
        }
    }

    /// Full GQA attention for a single token (decode step).
    /// Input is read from scratch.attn_out[..hidden], output written back to scratch.attn_out[..hidden].
    pub(super) fn full_attention_step_from_attn_out(
        &self,
        weights: &FullAttentionLayerWeights,
        cache_idx: usize,
        layer_idx: usize,
        position: usize,
        kv_cache: &mut KvCache,
        scratch: &mut ForwardScratch,
        hidden: usize,
    ) {
        let cfg = &self.config;
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();
        let head_dim = cfg.head_dim;
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let rope_dim = cfg.rope_dim();

        self.project_qkv(
            weights,
            layer_idx,
            scratch,
            hidden,
            q_dim,
            kv_dim,
            head_dim,
            num_q_heads,
            num_kv_heads,
        );

        self.normalize_and_rope(
            weights,
            position,
            scratch,
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_dim,
            kv_dim,
        );

        kv_cache.append_kv(
            cache_idx,
            &scratch.k_buf[..kv_dim],
            &scratch.v_buf[..kv_dim],
        );
        let cur_seq_len = kv_cache.seq_len + 1;

        self.compute_attention_context(
            kv_cache,
            cache_idx,
            scratch,
            cur_seq_len,
            q_dim,
            kv_dim,
            head_dim,
            num_q_heads,
            num_kv_heads,
        );

        apply_sigmoid_gate(&mut scratch.context[..q_dim], &scratch.gate_z[..q_dim]);

        matmul_bt(
            &scratch.context[..q_dim],
            &weights.o_proj,
            &mut scratch.attn_out[..hidden],
            1,
            q_dim,
            hidden,
        );
        self.lora.apply(
            layer_idx,
            "o_proj",
            &scratch.context[..q_dim],
            &mut scratch.attn_out[..hidden],
        );
    }

    fn project_qkv(
        &self,
        weights: &FullAttentionLayerWeights,
        layer_idx: usize,
        scratch: &mut ForwardScratch,
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) {
        let q_proj_dim = 2 * q_dim;
        scratch.ensure_decode_capacity(hidden, q_proj_dim, q_dim, self.config.intermediate_size);
        scratch.input_tmp[..hidden].copy_from_slice(&scratch.attn_out[..hidden]);
        let input = &scratch.input_tmp[..hidden];
        let q_and_gate = &mut scratch.q_and_gate[..q_proj_dim];
        matmul_bt(input, &weights.q_proj, q_and_gate, 1, hidden, q_proj_dim);
        self.lora.apply(layer_idx, "q_proj", input, q_and_gate);

        scratch.split_q_and_gate(num_q_heads, head_dim);

        let input = &scratch.input_tmp[..hidden];
        matmul_bt(
            input,
            &weights.k_proj,
            &mut scratch.k_buf[..kv_dim],
            1,
            hidden,
            kv_dim,
        );
        self.lora
            .apply(layer_idx, "k_proj", input, &mut scratch.k_buf[..kv_dim]);
        matmul_bt(
            input,
            &weights.v_proj,
            &mut scratch.v_buf[..kv_dim],
            1,
            hidden,
            kv_dim,
        );
        self.lora
            .apply(layer_idx, "v_proj", input, &mut scratch.v_buf[..kv_dim]);

        let cfg = &self.config;
        for h in 0..num_q_heads {
            let start = h * head_dim;
            qwen35_rms_norm(
                &mut scratch.q_buf[start..start + head_dim],
                &weights.q_norm,
                head_dim,
                cfg.rms_norm_eps,
            );
        }
        for h in 0..num_kv_heads {
            let start = h * head_dim;
            qwen35_rms_norm(
                &mut scratch.k_buf[start..start + head_dim],
                &weights.k_norm,
                head_dim,
                cfg.rms_norm_eps,
            );
        }
    }

    fn normalize_and_rope(
        &self,
        _weights: &FullAttentionLayerWeights,
        position: usize,
        scratch: &mut ForwardScratch,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        rope_dim: usize,
        _kv_dim: usize,
    ) {
        for h in 0..num_q_heads {
            let start = h * head_dim;
            self.apply_partial_rope(
                &mut scratch.q_buf[start..start + head_dim],
                position,
                rope_dim,
            );
        }
        for h in 0..num_kv_heads {
            let start = h * head_dim;
            self.apply_partial_rope(
                &mut scratch.k_buf[start..start + head_dim],
                position,
                rope_dim,
            );
        }
    }

    fn compute_attention_context(
        &self,
        kv_cache: &KvCache,
        cache_idx: usize,
        scratch: &mut ForwardScratch,
        cur_seq_len: usize,
        q_dim: usize,
        kv_dim: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) {
        let groups = num_q_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let k_cache = &kv_cache.k[cache_idx];
        let v_cache = &kv_cache.v[cache_idx];

        for qh in 0..num_q_heads {
            let kvh = qh / groups;
            let q_off = qh * head_dim;
            let q = &scratch.q_buf[q_off..q_off + head_dim];

            let scores_start = qh * cur_seq_len;
            let mut max_score = f32::NEG_INFINITY;

            for t in 0..cur_seq_len {
                let k_off = t * kv_dim + kvh * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[d] * k_cache[k_off + d];
                }
                let s = dot * scale;
                scratch.scores[scores_start + t] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            let mut sum_exp = 0.0f32;
            for t in 0..cur_seq_len {
                let e = (scratch.scores[scores_start + t] - max_score).exp();
                scratch.scores[scores_start + t] = e;
                sum_exp += e;
            }
            let inv_sum = 1.0 / sum_exp;
            for t in 0..cur_seq_len {
                scratch.scores[scores_start + t] *= inv_sum;
            }

            let ctx_off = qh * head_dim;
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for t in 0..cur_seq_len {
                    let v_off = t * kv_dim + kvh * head_dim;
                    sum += scratch.scores[scores_start + t] * v_cache[v_off + d];
                }
                scratch.context[ctx_off + d] = sum;
            }
        }
        let _ = q_dim; // used by caller for gate_z
    }

    /// Apply partial RoPE with INTERLEAVED pairing.
    /// Qwen3.5 uses mrope_interleaved=true: pairs are (0,1), (2,3), (4,5), ...
    /// Only the first `rope_dim` dimensions are rotated.
    pub(crate) fn apply_partial_rope(
        &self,
        head_vec: &mut [f32],
        position: usize,
        rope_dim: usize,
    ) {
        let half = rope_dim / 2;
        let base = position * half;
        for i in 0..half {
            let cos_val = self.rope.cos_at(base + i);
            let sin_val = self.rope.sin_at(base + i);
            let x0 = head_vec[2 * i];
            let x1 = head_vec[2 * i + 1];
            head_vec[2 * i] = x0 * cos_val - x1 * sin_val;
            head_vec[2 * i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }

    /// Dense SwiGLU FFN step.
    /// Input is read from scratch.ffn_out[..hidden], output written back to scratch.ffn_out[..hidden].
    pub(super) fn dense_ffn_step_from_ffn_out(
        &self,
        dense: &DenseFfnWeights,
        layer_idx: usize,
        scratch: &mut ForwardScratch,
        hidden: usize,
    ) {
        let inter = self.config.intermediate_size;

        scratch.ensure_decode_capacity(
            hidden,
            2 * self.config.full_q_dim(),
            self.config.full_q_dim(),
            inter,
        );
        scratch.input_tmp[..hidden].copy_from_slice(&scratch.ffn_out[..hidden]);
        let input = &scratch.input_tmp[..hidden];

        matmul_bt(
            input,
            &dense.gate_proj,
            &mut scratch.gate_buf[..inter],
            1,
            hidden,
            inter,
        );
        self.lora.apply(
            layer_idx,
            "gate_proj",
            input,
            &mut scratch.gate_buf[..inter],
        );
        matmul_bt(
            input,
            &dense.up_proj,
            &mut scratch.up_buf[..inter],
            1,
            hidden,
            inter,
        );
        self.lora
            .apply(layer_idx, "up_proj", input, &mut scratch.up_buf[..inter]);

        silu_inplace(&mut scratch.gate_buf[..inter]);
        elementwise_mul(&mut scratch.gate_buf[..inter], &scratch.up_buf[..inter]);

        scratch.down_input[..inter].copy_from_slice(&scratch.gate_buf[..inter]);
        let down_input = &scratch.down_input[..inter];
        matmul_bt(
            down_input,
            &dense.down_proj,
            &mut scratch.ffn_out[..hidden],
            1,
            inter,
            hidden,
        );
        self.lora.apply(
            layer_idx,
            "down_proj",
            down_input,
            &mut scratch.ffn_out[..hidden],
        );
    }
}
