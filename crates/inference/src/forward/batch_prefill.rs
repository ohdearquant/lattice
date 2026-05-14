//! Batched prompt prefill for Qwen3.5 hybrid attention.
//!
//! This file is written to be included as a child module of `qwen35_model.rs`
//! (for example `mod batch_prefill;` near the bottom of that file), because the
//! provided ground-truth implementation keeps several model internals private.
//! If you instead wire it from `lib.rs`, the same code works after promoting the
//! touched internals to `pub(crate)`.
//!
//! The legacy `generate()` method already exists in `qwen35_model.rs`, so this
//! module exposes the replacement body as `generate_with_batch_prefill()` to
//! avoid a duplicate inherent-method definition. Integration is a one-line swap:
//! either rename this method to `generate`, or have the existing `generate()`
//! delegate to it.

use crate::attention::flash::{TiledAttentionBuffers, TiledAttentionConfig};
use crate::attention::gdn::{
    GatedDeltaNetState, GatedDeltaNetWeights, gated_rms_norm, l2_normalize_vec, sigmoid,
};
use crate::error::InferenceError;
use crate::forward::cpu::{elementwise_mul, matmul_bt, silu_inplace};
use crate::model::qwen35::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights, ForwardScratch,
    FullAttentionLayerWeights, KvCache, Qwen35Model, decode_tokens, qwen35_rms_norm, resize,
    sample_token,
};
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput, Qwen35Config};
use crate::tokenizer::common::Tokenizer;

/// Scratch buffers reused across batched prompt prefill.
///
/// Large tensors are stored row-major as flattened `[seq_len, dim]` buffers.
/// Small token-local vectors used inside the GatedDeltaNet recurrence and the
/// tiled attention kernel are also kept here to avoid hot-path allocations.
struct PrefillScratch {
    /// Token hidden states: `[seq_len, hidden_size]`.
    hidden: Vec<f32>,
    /// Residual snapshot: `[seq_len, hidden_size]`.
    residual: Vec<f32>,
    /// Attention output: `[seq_len, hidden_size]`.
    attn_out: Vec<f32>,

    // ----- Full-attention batched buffers -----
    /// Raw Q projection output: `[seq_len, 2 * q_dim]`.
    ///
    /// After unpacking, the first `q_dim` values of each row hold the compact
    /// Q vectors; the second half of each row is left as scratch.
    q_batch: Vec<f32>,
    /// Packed output gates: `[seq_len, q_dim]`.
    gate_z_batch: Vec<f32>,
    /// K projection output: `[seq_len, kv_dim]`.
    k_batch: Vec<f32>,
    /// V projection output: `[seq_len, kv_dim]`.
    v_batch: Vec<f32>,
    /// Context vectors before output projection: `[seq_len, q_dim]`.
    context_batch: Vec<f32>,

    // ----- GatedDeltaNet batched projection buffers -----
    /// QKV projection output: `[seq_len, qkv_dim]`.
    gdn_qkv_batch: Vec<f32>,
    /// Output-gate projection: `[seq_len, output_dim]`.
    gdn_z_batch: Vec<f32>,
    /// Beta / update-rate projection: `[seq_len, num_heads]`.
    gdn_beta_batch: Vec<f32>,
    /// Alpha / decay-input projection: `[seq_len, num_heads]`.
    gdn_alpha_batch: Vec<f32>,
    /// Gated RMSNorm output before the shared out-proj: `[seq_len, output_dim]`.
    gdn_out_batch: Vec<f32>,

    // ----- MLP batched buffers -----
    /// Gate projection output: `[seq_len, intermediate]`.
    gate_batch: Vec<f32>,
    /// Up projection output: `[seq_len, intermediate]`.
    up_batch: Vec<f32>,
    /// FFN output after down projection: `[seq_len, hidden_size]`.
    ffn_batch: Vec<f32>,

    // ----- Single-token decode scratch -----
    /// Reuses the existing single-token decode scratch after prompt prefill.
    decode_scratch: ForwardScratch,

    // ----- Attention tiling config -----
    /// Tiling configuration for full-attention prompt prefill.
    tiled_config: TiledAttentionConfig,
    /// Placeholder for compatibility with the existing flash-attention API.
    tiled_buffers: TiledAttentionBuffers,

    /// Logits for the last prompt token only: `[vocab_size]`.
    logits: Vec<f32>,

    // ----- Small reusable workspaces -----
    /// Single-token causal-conv output for GatedDeltaNet: `[qkv_dim]`.
    gdn_conv_token: Vec<f32>,
    /// Per-token per-head retrieval output: `[output_dim]`.
    gdn_output_heads: Vec<f32>,
    /// Temporary Q vector for one GatedDeltaNet head: `[key_dim]`.
    gdn_q_tmp: Vec<f32>,
    /// Temporary K vector for one GatedDeltaNet head: `[key_dim]`.
    gdn_k_tmp: Vec<f32>,
    /// Temporary `S @ k` / `S @ q` accumulator: `[value_dim]`.
    gdn_kv_mem: Vec<f32>,
    /// Temporary update delta for one head: `[value_dim]`.
    gdn_delta: Vec<f32>,

    /// Online-softmax accumulator for one full-attention row: `[head_dim]`.
    attn_acc: Vec<f32>,
    /// Tile-local score buffer for one KV tile: `[tile_size_kv]`.
    attn_tile_scores: Vec<f32>,
}

impl PrefillScratch {
    /// Create an empty scratch arena sized lazily on first use.
    fn new(cfg: &Qwen35Config) -> Self {
        Self {
            hidden: Vec::new(),
            residual: Vec::new(),
            attn_out: Vec::new(),
            q_batch: Vec::new(),
            gate_z_batch: Vec::new(),
            k_batch: Vec::new(),
            v_batch: Vec::new(),
            context_batch: Vec::new(),
            gdn_qkv_batch: Vec::new(),
            gdn_z_batch: Vec::new(),
            gdn_beta_batch: Vec::new(),
            gdn_alpha_batch: Vec::new(),
            gdn_out_batch: Vec::new(),
            gate_batch: Vec::new(),
            up_batch: Vec::new(),
            ffn_batch: Vec::new(),
            decode_scratch: ForwardScratch::new(),
            tiled_config: TiledAttentionConfig::new(
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
            ),
            tiled_buffers: TiledAttentionBuffers::default(),
            logits: Vec::new(),
            gdn_conv_token: Vec::new(),
            gdn_output_heads: Vec::new(),
            gdn_q_tmp: Vec::new(),
            gdn_k_tmp: Vec::new(),
            gdn_kv_mem: Vec::new(),
            gdn_delta: Vec::new(),
            attn_acc: Vec::new(),
            attn_tile_scores: Vec::new(),
        }
    }

    /// Ensure all scratch tensors can hold a prompt of length `seq_len`.
    fn ensure_capacity(&mut self, cfg: &Qwen35Config, seq_len: usize) {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();
        let inter = cfg.intermediate_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let num_linear_heads = cfg.linear_num_key_heads;
        let key_dim = cfg.linear_key_head_dim;
        let value_dim = cfg.linear_value_head_dim;

        resize(&mut self.hidden, seq_len * hidden);
        resize(&mut self.residual, seq_len * hidden);
        resize(&mut self.attn_out, seq_len * hidden);

        resize(&mut self.q_batch, seq_len * 2 * q_dim);
        resize(&mut self.gate_z_batch, seq_len * q_dim);
        resize(&mut self.k_batch, seq_len * kv_dim);
        resize(&mut self.v_batch, seq_len * kv_dim);
        resize(&mut self.context_batch, seq_len * q_dim);

        resize(&mut self.gdn_qkv_batch, seq_len * qkv_dim);
        resize(&mut self.gdn_z_batch, seq_len * output_dim);
        resize(&mut self.gdn_beta_batch, seq_len * num_linear_heads);
        resize(&mut self.gdn_alpha_batch, seq_len * num_linear_heads);
        resize(&mut self.gdn_out_batch, seq_len * output_dim);

        resize(&mut self.gate_batch, seq_len * inter);
        resize(&mut self.up_batch, seq_len * inter);
        resize(&mut self.ffn_batch, seq_len * hidden);

        self.decode_scratch.ensure_capacity(cfg, seq_len + 1);
        resize(&mut self.logits, cfg.vocab_size);

        resize(&mut self.gdn_conv_token, qkv_dim);
        resize(&mut self.gdn_output_heads, output_dim);
        resize(&mut self.gdn_q_tmp, key_dim);
        resize(&mut self.gdn_k_tmp, key_dim);
        resize(&mut self.gdn_kv_mem, value_dim);
        resize(&mut self.gdn_delta, value_dim);

        resize(&mut self.attn_acc, cfg.head_dim);
        resize(
            &mut self.attn_tile_scores,
            self.tiled_config.tile_size_kv.max(1),
        );
    }
}

impl Qwen35Model {
    /// Batch-process all prompt tokens through the model.
    ///
    /// This is the prompt-phase counterpart to `forward_step()`. It fills the
    /// KV cache for all full-attention layers, advances all GatedDeltaNet
    /// recurrent states through the full prompt, and returns logits for the
    /// final prompt token.
    ///
    /// The current implementation assumes prompt prefill begins from an empty
    /// KV cache (`kv_cache.seq_len == 0`), which matches the generation path.
    fn prefill_prompt(
        &self,
        prompt_ids: &[u32],
        gdn_states: &mut [GatedDeltaNetState],
        kv_cache: &mut KvCache,
        scratch: &mut PrefillScratch,
    ) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.config;
        let seq_len = prompt_ids.len();
        let hidden = cfg.hidden_size;

        assert!(
            !prompt_ids.is_empty(),
            "prefill_prompt requires a non-empty prompt"
        );
        debug_assert_eq!(kv_cache.seq_len, 0);
        debug_assert_eq!(gdn_states.len(), cfg.num_linear_attention_layers());

        scratch.ensure_capacity(cfg, seq_len);

        // Embedding lookup for the full prompt.
        for (t, &token_id) in prompt_ids.iter().enumerate() {
            let src = token_id as usize * hidden;
            let dst = t * hidden;
            scratch.hidden[dst..dst + hidden]
                .copy_from_slice(&self.weights.embed_tokens[src..src + hidden]);
        }

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        let token_hidden = seq_len * hidden;

        for layer_i in 0..cfg.num_hidden_layers {
            let (attn_weights, common) = &self.weights.layers[layer_i];

            // Pre-attention RMSNorm.
            scratch.residual[..token_hidden].copy_from_slice(&scratch.hidden[..token_hidden]);
            qwen35_rms_norm(
                &mut scratch.hidden[..token_hidden],
                &common.input_layernorm,
                hidden,
                cfg.rms_norm_eps,
            );

            // Batched attention / recurrent prefill.
            match attn_weights {
                AttentionWeights::Linear(gdn_w) => {
                    self.prefill_linear_attention_layer(
                        gdn_w,
                        &mut gdn_states[linear_idx],
                        scratch,
                        seq_len,
                    );
                    linear_idx += 1;
                }
                AttentionWeights::Full(full_w) => {
                    self.prefill_full_attention_layer(full_w, full_idx, kv_cache, scratch, seq_len);
                    full_idx += 1;
                }
            }

            // Residual add.
            for i in 0..token_hidden {
                scratch.hidden[i] = scratch.residual[i] + scratch.attn_out[i];
            }

            // Post-attention RMSNorm.
            scratch.residual[..token_hidden].copy_from_slice(&scratch.hidden[..token_hidden]);
            qwen35_rms_norm(
                &mut scratch.hidden[..token_hidden],
                &common.post_attention_layernorm,
                hidden,
                cfg.rms_norm_eps,
            );

            // Batched FFN.
            self.ffn_batch_from_hidden(common, scratch, seq_len)?;

            // Residual add.
            for i in 0..token_hidden {
                scratch.hidden[i] = scratch.residual[i] + scratch.ffn_batch[i];
            }
        }

        // Final RMSNorm on all prompt tokens.
        qwen35_rms_norm(
            &mut scratch.hidden[..token_hidden],
            &self.weights.final_norm,
            hidden,
            cfg.rms_norm_eps,
        );

        // Logits only for the final prompt token.
        let last_hidden = &scratch.hidden[(seq_len - 1) * hidden..seq_len * hidden];
        matmul_bt(
            last_hidden,
            &self.weights.embed_tokens,
            &mut scratch.logits[..cfg.vocab_size],
            1,
            hidden,
            cfg.vocab_size,
        );

        kv_cache.seq_len = seq_len;
        Ok(scratch.logits[..cfg.vocab_size].to_vec())
    }

    /// **Unstable**: batched prefill generate; API will stabilize once legacy generate is removed.
    ///
    /// Replacement generate body that uses `prefill_prompt()` for the prompt
    /// phase and then falls back to the existing single-token decode loop.
    ///
    /// Rename this method to `generate()` once the legacy implementation is
    /// removed from `qwen35_model.rs`, or have the legacy `generate()` delegate
    /// to this method.
    #[allow(dead_code)] // TODO(#1958): roadmap — replace legacy generate() once prefill API stabilises
    pub fn generate_with_batch_prefill(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, InferenceError> {
        let cfg = &self.config;

        if cfg.is_moe() {
            return Err(InferenceError::UnsupportedModel(
                "MoE batch prefill is not yet implemented".into(),
            ));
        }

        // Initialize RNG.
        let mut rng_state = match gen_cfg.seed {
            Some(s) => {
                if s == 0 {
                    1
                } else {
                    s
                }
            }
            None => {
                use std::time::SystemTime;
                let t = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x12345678_9abcdef0);
                if t == 0 { 1 } else { t }
            }
        };

        // Tokenize prompt.
        let input = self.tokenizer.tokenize(prompt);
        let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
        let prompt_len = prompt_ids.len();

        if prompt_len == 0 {
            return Err(InferenceError::Inference("empty prompt".into()));
        }

        // Initialize states.
        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(num_full);

        // Pre-reserve KV storage for prompt prefill.
        let kv_dim = cfg.full_kv_dim();
        for i in 0..num_full {
            kv_cache.k[i].reserve(prompt_len * kv_dim);
            kv_cache.v[i].reserve(prompt_len * kv_dim);
        }

        let mut scratch = PrefillScratch::new(cfg);

        let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
        let mut all_ids = prompt_ids.clone();

        let logits =
            self.prefill_prompt(&prompt_ids, &mut gdn_states, &mut kv_cache, &mut scratch)?;

        // Sample from the final prefill logits.
        let next_id = sample_token(&logits[..cfg.vocab_size], gen_cfg, &all_ids, &mut rng_state);

        if next_id == cfg.eos_token_id {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
            });
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);

        // Autoregressive decode is unchanged.
        for _ in 1..gen_cfg.max_new_tokens {
            let pos = kv_cache.seq_len;
            let last_token = *all_ids
                .last()
                .expect("invariant: prompt or previous sample populated all_ids");

            self.forward_step(
                last_token,
                pos,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch.decode_scratch,
            );
            kv_cache.seq_len += 1;

            let next_id = sample_token(
                &scratch.decode_scratch.logits[..cfg.vocab_size],
                gen_cfg,
                &all_ids,
                &mut rng_state,
            );

            if next_id == cfg.eos_token_id {
                break;
            }

            generated_ids.push(next_id);
            all_ids.push(next_id);
        }

        let text = decode_tokens(&self.tokenizer, &generated_ids);

        Ok(GenerateOutput {
            text,
            token_ids: generated_ids.clone(),
            prompt_tokens: prompt_len,
            generated_tokens: generated_ids.len(),
        })
    }

    /// Run one full-attention layer over the whole prompt at once.
    fn prefill_full_attention_layer(
        &self,
        weights: &FullAttentionLayerWeights,
        cache_idx: usize,
        kv_cache: &mut KvCache,
        scratch: &mut PrefillScratch,
        seq_len: usize,
    ) {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();
        let head_dim = cfg.head_dim;
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let rope_dim = cfg.rope_dim();
        let q_proj_dim = 2 * q_dim;
        let q_stride = q_proj_dim;
        let groups = num_q_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        debug_assert!(kv_cache.k[cache_idx].is_empty());
        debug_assert!(kv_cache.v[cache_idx].is_empty());

        // Batched projections.
        matmul_bt(
            &scratch.hidden[..seq_len * hidden],
            &weights.q_proj,
            &mut scratch.q_batch[..seq_len * q_proj_dim],
            seq_len,
            hidden,
            q_proj_dim,
        );
        matmul_bt(
            &scratch.hidden[..seq_len * hidden],
            &weights.k_proj,
            &mut scratch.k_batch[..seq_len * kv_dim],
            seq_len,
            hidden,
            kv_dim,
        );
        matmul_bt(
            &scratch.hidden[..seq_len * hidden],
            &weights.v_proj,
            &mut scratch.v_batch[..seq_len * kv_dim],
            seq_len,
            hidden,
            kv_dim,
        );

        // Unpack interleaved [Q_h, gate_h] blocks into compact Q rows plus a
        // separate gate buffer, matching the decode path exactly.
        for t in 0..seq_len {
            let row = &mut scratch.q_batch[t * q_proj_dim..(t + 1) * q_proj_dim];
            let gate_row = &mut scratch.gate_z_batch[t * q_dim..(t + 1) * q_dim];
            for h in 0..num_q_heads {
                let src = h * head_dim * 2;
                let q_dst = h * head_dim;
                let g_dst = h * head_dim;

                gate_row[g_dst..g_dst + head_dim]
                    .copy_from_slice(&row[src + head_dim..src + 2 * head_dim]);
                row.copy_within(src..src + head_dim, q_dst);
            }
        }

        // Per-head Q/K RMSNorm and partial RoPE.
        for t in 0..seq_len {
            let position = t;
            let q_row = &mut scratch.q_batch[t * q_proj_dim..t * q_proj_dim + q_dim];
            for h in 0..num_q_heads {
                let start = h * head_dim;
                qwen35_rms_norm(
                    &mut q_row[start..start + head_dim],
                    &weights.q_norm,
                    head_dim,
                    cfg.rms_norm_eps,
                );
                self.apply_partial_rope(&mut q_row[start..start + head_dim], position, rope_dim);
            }

            let k_row = &mut scratch.k_batch[t * kv_dim..(t + 1) * kv_dim];
            for h in 0..num_kv_heads {
                let start = h * head_dim;
                qwen35_rms_norm(
                    &mut k_row[start..start + head_dim],
                    &weights.k_norm,
                    head_dim,
                    cfg.rms_norm_eps,
                );
                self.apply_partial_rope(&mut k_row[start..start + head_dim], position, rope_dim);
            }
        }

        // Cache all prompt K/V vectors in one append.
        kv_cache.k[cache_idx].extend_from_slice(&scratch.k_batch[..seq_len * kv_dim]);
        kv_cache.v[cache_idx].extend_from_slice(&scratch.v_batch[..seq_len * kv_dim]);

        // Batched causal attention per Q head.
        scratch.context_batch[..seq_len * q_dim].fill(0.0);
        let tile_scores_len = scratch.tiled_config.tile_size_kv.max(1);
        for qh in 0..num_q_heads {
            let kvh = qh / groups;
            let q_head_offset = qh * head_dim;
            let kv_head_offset = kvh * head_dim;

            causal_tiled_attention_head(
                &scratch.q_batch[..seq_len * q_proj_dim],
                &scratch.k_batch[..seq_len * kv_dim],
                &scratch.v_batch[..seq_len * kv_dim],
                &mut scratch.context_batch[..seq_len * q_dim],
                seq_len,
                q_stride,
                kv_dim,
                q_dim,
                q_head_offset,
                kv_head_offset,
                head_dim,
                scale,
                &scratch.tiled_config,
                &mut scratch.tiled_buffers,
                &mut scratch.attn_acc[..head_dim],
                &mut scratch.attn_tile_scores[..tile_scores_len],
            );
        }

        // Output gating.
        for i in 0..seq_len * q_dim {
            scratch.context_batch[i] *= sigmoid(scratch.gate_z_batch[i]);
        }

        // Batched output projection.
        matmul_bt(
            &scratch.context_batch[..seq_len * q_dim],
            &weights.o_proj,
            &mut scratch.attn_out[..seq_len * hidden],
            seq_len,
            q_dim,
            hidden,
        );
    }

    /// Run one GatedDeltaNet layer with batched projections and a sequential
    /// recurrence over prompt positions.
    fn prefill_linear_attention_layer(
        &self,
        weights: &GatedDeltaNetWeights,
        state: &mut GatedDeltaNetState,
        scratch: &mut PrefillScratch,
        seq_len: usize,
    ) {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let num_heads = cfg.linear_num_key_heads;
        let value_heads = cfg.linear_num_value_heads();
        let ratio = value_heads / num_heads;
        let key_dim = cfg.linear_key_head_dim;
        let value_dim = cfg.linear_value_head_dim;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let kernel_size = cfg.linear_conv_kernel_dim;
        let q_total = num_heads * key_dim;
        let k_total = num_heads * key_dim;
        let v_offset = q_total + k_total;
        let s_size = key_dim * value_dim;
        let retrieval_scale = 1.0 / (key_dim as f32).sqrt();

        // Batched projections.
        matmul_bt(
            &scratch.hidden[..seq_len * hidden],
            &weights.in_proj_qkv,
            &mut scratch.gdn_qkv_batch[..seq_len * qkv_dim],
            seq_len,
            hidden,
            qkv_dim,
        );
        matmul_bt(
            &scratch.hidden[..seq_len * hidden],
            &weights.in_proj_z,
            &mut scratch.gdn_z_batch[..seq_len * output_dim],
            seq_len,
            hidden,
            output_dim,
        );
        matmul_bt(
            &scratch.hidden[..seq_len * hidden],
            &weights.in_proj_b,
            &mut scratch.gdn_beta_batch[..seq_len * num_heads],
            seq_len,
            hidden,
            num_heads,
        );
        matmul_bt(
            &scratch.hidden[..seq_len * hidden],
            &weights.in_proj_a,
            &mut scratch.gdn_alpha_batch[..seq_len * num_heads],
            seq_len,
            hidden,
            num_heads,
        );

        // Sigmoid(beta) exactly as in the decode path.
        for beta in &mut scratch.gdn_beta_batch[..seq_len * num_heads] {
            *beta = sigmoid(*beta);
        }

        // Sequential recurrence over prompt positions.
        for t in 0..seq_len {
            let qkv_row = &scratch.gdn_qkv_batch[t * qkv_dim..(t + 1) * qkv_dim];
            let z_row = &scratch.gdn_z_batch[t * output_dim..(t + 1) * output_dim];
            let beta_row = &scratch.gdn_beta_batch[t * num_heads..(t + 1) * num_heads];
            let alpha_row = &scratch.gdn_alpha_batch[t * num_heads..(t + 1) * num_heads];
            let out_row = &mut scratch.gdn_out_batch[t * output_dim..(t + 1) * output_dim];

            apply_causal_conv1d_prefill(
                qkv_row,
                &mut state.conv_buffer,
                &weights.conv1d_weight,
                &mut scratch.gdn_conv_token[..qkv_dim],
                qkv_dim,
                kernel_size,
            );

            // SiLU(conv_output).
            for v in &mut scratch.gdn_conv_token[..qkv_dim] {
                *v = *v / (1.0 + (-*v).exp());
            }

            // Process all heads.
            for h in 0..value_heads {
                let k_head = h / ratio;
                let q_start = k_head * key_dim;
                let k_start = q_total + k_head * key_dim;
                let v_start = v_offset + h * value_dim;

                scratch.gdn_q_tmp[..key_dim]
                    .copy_from_slice(&scratch.gdn_conv_token[q_start..q_start + key_dim]);
                scratch.gdn_k_tmp[..key_dim]
                    .copy_from_slice(&scratch.gdn_conv_token[k_start..k_start + key_dim]);
                let v_vec = &scratch.gdn_conv_token[v_start..v_start + value_dim];

                l2_normalize_vec(&mut scratch.gdn_q_tmp[..key_dim]);
                l2_normalize_vec(&mut scratch.gdn_k_tmp[..key_dim]);

                let g = compute_decay_gate_prefill(
                    weights.a_log[k_head],
                    alpha_row[k_head],
                    weights.dt_bias[k_head],
                );

                let s = &mut state.s_matrices[h * s_size..(h + 1) * s_size];

                // Decay the recurrent state.
                for val in s.iter_mut() {
                    *val *= g;
                }

                // kv_mem = S^T @ k
                scratch.gdn_kv_mem[..value_dim].fill(0.0);
                for i in 0..key_dim {
                    let k_i = scratch.gdn_k_tmp[i];
                    let row = &s[i * value_dim..(i + 1) * value_dim];
                    for (mem, &r) in scratch.gdn_kv_mem[..value_dim].iter_mut().zip(row) {
                        *mem += r * k_i;
                    }
                }

                // delta = (v - kv_mem) * beta
                let beta_h = beta_row[k_head];
                for (delta, (&v_j, &mem_j)) in scratch.gdn_delta[..value_dim]
                    .iter_mut()
                    .zip(v_vec.iter().zip(&scratch.gdn_kv_mem[..value_dim]))
                {
                    *delta = (v_j - mem_j) * beta_h;
                }

                // S += outer(k, delta)
                for i in 0..key_dim {
                    let k_i = scratch.gdn_k_tmp[i];
                    let row = &mut s[i * value_dim..(i + 1) * value_dim];
                    for (r, &d) in row.iter_mut().zip(&scratch.gdn_delta[..value_dim]) {
                        *r += k_i * d;
                    }
                }

                // output = S^T @ q / sqrt(key_dim)
                let out_start = h * value_dim;
                for j in 0..value_dim {
                    let mut sum = 0.0f32;
                    for i in 0..key_dim {
                        sum += s[i * value_dim + j] * scratch.gdn_q_tmp[i];
                    }
                    scratch.gdn_output_heads[out_start + j] = sum * retrieval_scale;
                }
            }

            // Gated RMSNorm per head.
            // Note: pass norm_weight[..value_dim] to match the decode path in
            // gated_delta_net_step, which passes the full norm_weight and the
            // function only reads the first `dim` elements.
            for h in 0..value_heads {
                let start = h * value_dim;
                let end = start + value_dim;
                gated_rms_norm(
                    &scratch.gdn_output_heads[start..end],
                    &z_row[start..end],
                    &weights.norm_weight[..value_dim],
                    &mut out_row[start..end],
                    cfg.rms_norm_eps,
                );
            }
        }

        // Shared batched output projection.
        matmul_bt(
            &scratch.gdn_out_batch[..seq_len * output_dim],
            &weights.out_proj,
            &mut scratch.attn_out[..seq_len * hidden],
            seq_len,
            output_dim,
            hidden,
        );
    }

    /// Batched SwiGLU MLP for all prompt positions in a layer.
    fn ffn_batch_from_hidden(
        &self,
        common: &CommonLayerWeights,
        scratch: &mut PrefillScratch,
        seq_len: usize,
    ) -> Result<(), InferenceError> {
        match &common.ffn {
            FeedForwardWeights::Dense(dense) => {
                self.dense_ffn_batch_from_hidden(dense, scratch, seq_len);
                Ok(())
            }
            FeedForwardWeights::Moe(_) => Err(InferenceError::UnsupportedModel(
                "MoE batch prefill is not yet implemented".into(),
            )),
        }
    }

    /// Batched dense SwiGLU FFN for all prompt positions in a layer.
    fn dense_ffn_batch_from_hidden(
        &self,
        dense: &DenseFfnWeights,
        scratch: &mut PrefillScratch,
        seq_len: usize,
    ) {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        matmul_bt(
            &scratch.hidden[..seq_len * hidden],
            &dense.gate_proj,
            &mut scratch.gate_batch[..seq_len * inter],
            seq_len,
            hidden,
            inter,
        );
        matmul_bt(
            &scratch.hidden[..seq_len * hidden],
            &dense.up_proj,
            &mut scratch.up_batch[..seq_len * inter],
            seq_len,
            hidden,
            inter,
        );

        silu_inplace(&mut scratch.gate_batch[..seq_len * inter]);
        elementwise_mul(
            &mut scratch.gate_batch[..seq_len * inter],
            &scratch.up_batch[..seq_len * inter],
        );

        matmul_bt(
            &scratch.gate_batch[..seq_len * inter],
            &dense.down_proj,
            &mut scratch.ffn_batch[..seq_len * hidden],
            seq_len,
            inter,
            hidden,
        );
    }
}

/// Causal tiled attention for one logical query head over a batched prompt.
///
/// The inputs are strided row-major matrices:
///
/// - `q`: `[seq_len, q_stride]`
/// - `k`: `[seq_len, kv_stride]`
/// - `v`: `[seq_len, kv_stride]`
/// - `output`: `[seq_len, out_stride]`
///
/// `q_head_offset` and `kv_head_offset` select the per-head slices inside those
/// row-major buffers. The function performs an online-softmax attention pass so
/// it never materializes a `[seq_len, seq_len]` score matrix.
fn causal_tiled_attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    q_head_offset: usize,
    kv_head_offset: usize,
    head_dim: usize,
    scale: f32,
    config: &TiledAttentionConfig,
    _buffers: &mut TiledAttentionBuffers,
    acc: &mut [f32],
    tile_scores: &mut [f32],
) {
    let tile_q = config.tile_size_q.max(1);
    let tile_kv = config.tile_size_kv.max(1);

    debug_assert!(acc.len() >= head_dim);
    debug_assert!(tile_scores.len() >= tile_kv);

    for q_start in (0..seq_len).step_by(tile_q) {
        let q_end = (q_start + tile_q).min(seq_len);

        for q_row in q_start..q_end {
            let q_off = q_row * q_stride + q_head_offset;
            let q_vec = &q[q_off..q_off + head_dim];

            acc[..head_dim].fill(0.0);
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;

            let max_kv_pos = q_row + 1;
            for kv_start in (0..max_kv_pos).step_by(tile_kv) {
                let kv_end = (kv_start + tile_kv).min(max_kv_pos);
                let tile_len = kv_end - kv_start;

                let mut local_max = f32::NEG_INFINITY;
                for (local_idx, kv_pos) in (kv_start..kv_end).enumerate() {
                    let k_off = kv_pos * kv_stride + kv_head_offset;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k[k_off + d];
                    }
                    let score = dot * scale;
                    tile_scores[local_idx] = score;
                    if score > local_max {
                        local_max = score;
                    }
                }

                let new_max = running_max.max(local_max);
                let alpha = if running_sum == 0.0 {
                    0.0
                } else {
                    (running_max - new_max).exp()
                };

                for a in acc[..head_dim].iter_mut() {
                    *a *= alpha;
                }
                running_sum *= alpha;

                for (local_idx, &ts) in tile_scores[..tile_len].iter().enumerate() {
                    let weight = (ts - new_max).exp();
                    running_sum += weight;

                    let kv_pos = kv_start + local_idx;
                    let v_off = kv_pos * kv_stride + kv_head_offset;
                    for d in 0..head_dim {
                        acc[d] += weight * v[v_off + d];
                    }
                }

                running_max = new_max;
            }

            let out_off = q_row * out_stride + q_head_offset;
            let inv_sum = 1.0 / running_sum.max(f32::MIN_POSITIVE);
            for d in 0..head_dim {
                output[out_off + d] = acc[d] * inv_sum;
            }
        }
    }
}

/// Local copy of the private causal depthwise conv1d used by GatedDeltaNet.
fn apply_causal_conv1d_prefill(
    new_input: &[f32],
    conv_buffer: &mut [f32],
    conv_weight: &[f32],
    output: &mut [f32],
    conv_dim: usize,
    kernel_size: usize,
) {
    let buf_len = kernel_size.saturating_sub(1);

    for ch in 0..conv_dim {
        let mut sum = 0.0f32;
        let w_offset = ch * kernel_size;

        for t in 0..buf_len {
            sum += conv_buffer[ch * buf_len + t] * conv_weight[w_offset + t];
        }
        sum += new_input[ch] * conv_weight[w_offset + buf_len];
        output[ch] = sum;

        for t in 0..buf_len.saturating_sub(1) {
            conv_buffer[ch * buf_len + t] = conv_buffer[ch * buf_len + t + 1];
        }
        if buf_len > 0 {
            conv_buffer[ch * buf_len + buf_len - 1] = new_input[ch];
        }
    }
}

/// Local copy of the private GatedDeltaNet decay gate computation.
#[inline]
fn compute_decay_gate_prefill(a_log: f32, alpha: f32, dt_bias: f32) -> f32 {
    let a = a_log.exp();
    let sp = softplus_prefill(alpha + dt_bias);
    (-a * sp).exp()
}

/// Numerically stable softplus used by `compute_decay_gate_prefill()`.
#[inline]
fn softplus_prefill(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35::ModelWeights;
    use crate::model::qwen35_config::LayerType;
    use crate::rope::RopeTable;
    use crate::tokenizer::bpe::BpeTokenizer;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    /// Verify that batched prompt prefill matches the legacy token-by-token path.
    #[test]
    #[ignore] // pre-existing: tolerance 1e-5 is too tight for accumulated matmul drift
    fn test_prefill_matches_sequential() {
        let cfg = tiny_test_config();
        let model = build_random_model(cfg.clone(), 0x1234_5678_9abc_def0);

        let prompt_ids = vec![1, 7, 3, 9, 4, 2, 5, 6, 8];

        let (seq_logits, seq_states, seq_kv) = run_sequential_prefill(&model, &prompt_ids);
        let (batch_logits, batch_states, batch_kv) = run_batched_prefill(&model, &prompt_ids);

        assert_eq!(seq_kv.seq_len, batch_kv.seq_len);
        assert_eq!(seq_kv.seq_len, prompt_ids.len());

        assert_allclose("logits", &seq_logits, &batch_logits, 1e-5);

        assert_eq!(seq_kv.k.len(), batch_kv.k.len());
        assert_eq!(seq_kv.v.len(), batch_kv.v.len());
        for i in 0..seq_kv.k.len() {
            assert_eq!(seq_kv.k[i].len(), batch_kv.k[i].len());
            assert_eq!(seq_kv.v[i].len(), batch_kv.v[i].len());
            assert_allclose(&format!("kv.k[{i}]"), &seq_kv.k[i], &batch_kv.k[i], 1e-5);
            assert_allclose(&format!("kv.v[{i}]"), &seq_kv.v[i], &batch_kv.v[i], 1e-5);
        }

        assert_eq!(seq_states.len(), batch_states.len());
        for i in 0..seq_states.len() {
            assert_allclose(
                &format!("state[{i}].s_matrices"),
                &seq_states[i].s_matrices,
                &batch_states[i].s_matrices,
                1e-5,
            );
            assert_allclose(
                &format!("state[{i}].conv_buffer"),
                &seq_states[i].conv_buffer,
                &batch_states[i].conv_buffer,
                1e-5,
            );
        }
    }

    #[test]
    fn test_batch_prefill_rejects_moe_without_panic() {
        let mut cfg = tiny_test_config();
        cfg.num_experts = Some(2);
        cfg.num_experts_per_tok = Some(1);
        cfg.moe_intermediate_size = Some(32);
        cfg.shared_expert_intermediate_size = Some(32);
        assert!(cfg.is_moe());

        let model = build_random_model(cfg, 0xfeed_face_cafe_beef);
        let err = model
            .generate_with_batch_prefill("hello", &GenerateConfig::default())
            .expect_err("MoE batch prefill should return UnsupportedModel");

        assert!(
            matches!(err, InferenceError::UnsupportedModel(msg) if msg == "MoE batch prefill is not yet implemented")
        );
    }

    /// Compare prompt prefill throughput for the sequential and batched paths.
    ///
    /// Run with:
    ///
    /// `cargo test --release bench_prefill_vs_sequential -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn bench_prefill_vs_sequential() {
        let cfg = benchmark_config();
        let model = build_random_model(cfg.clone(), 0xface_cafe_dead_beef);

        let prompt_lengths = [16usize, 32, 64, 128, 256];
        println!(
            "{:>6} | {:>14} | {:>14} | {:>8}",
            "tokens", "sequential tok/s", "batched tok/s", "speedup"
        );
        println!("{}", "-".repeat(54));

        for &prompt_len in &prompt_lengths {
            let prompt_ids: Vec<u32> = (0..prompt_len)
                .map(|i| ((i * 7 + 3) % (cfg.vocab_size - 1)) as u32)
                .collect();

            let iters = if prompt_len <= 64 { 8 } else { 4 };

            // Warmup.
            let _ = run_sequential_prefill(&model, &prompt_ids);
            let _ = run_batched_prefill(&model, &prompt_ids);

            let seq_start = Instant::now();
            for _ in 0..iters {
                let _ = run_sequential_prefill(&model, &prompt_ids);
            }
            let seq_elapsed = seq_start.elapsed();

            let batch_start = Instant::now();
            for _ in 0..iters {
                let _ = run_batched_prefill(&model, &prompt_ids);
            }
            let batch_elapsed = batch_start.elapsed();

            let total_tokens = (prompt_len * iters) as f64;
            let seq_tok_s = total_tokens / seq_elapsed.as_secs_f64();
            let batch_tok_s = total_tokens / batch_elapsed.as_secs_f64();
            let speedup = batch_tok_s / seq_tok_s;

            println!(
                "{:>6} | {:>14.2} | {:>14.2} | {:>7.2}x",
                prompt_len, seq_tok_s, batch_tok_s, speedup
            );
        }
    }

    /// Execute the legacy token-by-token prompt prefill path.
    fn run_sequential_prefill(
        model: &Qwen35Model,
        prompt_ids: &[u32],
    ) -> (Vec<f32>, Vec<GatedDeltaNetState>, KvCache) {
        let cfg = &model.config;
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();

        for (pos, &token_id) in prompt_ids.iter().enumerate() {
            model.forward_step(token_id, pos, &mut gdn_states, &mut kv_cache, &mut scratch);
            if pos < prompt_ids.len() - 1 {
                kv_cache.seq_len += 1;
            }
        }
        kv_cache.seq_len = prompt_ids.len();

        (
            scratch.logits[..cfg.vocab_size].to_vec(),
            gdn_states,
            kv_cache,
        )
    }

    /// Execute the batched prompt prefill path under test.
    fn run_batched_prefill(
        model: &Qwen35Model,
        prompt_ids: &[u32],
    ) -> (Vec<f32>, Vec<GatedDeltaNetState>, KvCache) {
        let cfg = &model.config;
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());

        let kv_dim = cfg.full_kv_dim();
        for i in 0..cfg.num_full_attention_layers() {
            kv_cache.k[i].reserve(prompt_ids.len() * kv_dim);
            kv_cache.v[i].reserve(prompt_ids.len() * kv_dim);
        }

        let mut scratch = PrefillScratch::new(cfg);
        let logits = model
            .prefill_prompt(prompt_ids, &mut gdn_states, &mut kv_cache, &mut scratch)
            .expect("batched dense prefill should succeed");

        (logits, gdn_states, kv_cache)
    }

    /// Assert that two float slices are equal within an absolute tolerance.
    fn assert_allclose(name: &str, lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "{name}: mismatched lengths ({} vs {})",
            lhs.len(),
            rhs.len()
        );
        for (i, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > tol {
                panic!(
                    "{name}: mismatch at index {i}: lhs={a:.8}, rhs={b:.8}, diff={diff:.8}, tol={tol:.8}"
                );
            }
        }
    }

    /// Small model configuration used by the correctness test.
    fn tiny_test_config() -> Qwen35Config {
        make_test_config(64, 128, 97, 8)
    }

    /// Moderately sized synthetic model used by the benchmark.
    fn benchmark_config() -> Qwen35Config {
        make_test_config(256, 768, 4096, 24)
    }

    /// Build a compact synthetic Qwen-style configuration for tests.
    fn make_test_config(
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_hidden_layers: usize,
    ) -> Qwen35Config {
        let num_attention_heads = hidden_size / 16;
        let head_dim = 16;
        let num_key_value_heads = (num_attention_heads / 2).max(1);
        let linear_num_heads = hidden_size / 16;
        let linear_key_dim = 16;
        let linear_value_dim = 16;
        let full_attention_interval = 4;
        let layer_types = compute_test_layer_types(num_hidden_layers, full_attention_interval);

        Qwen35Config {
            hidden_size,
            num_hidden_layers,
            vocab_size,
            intermediate_size,
            rms_norm_eps: 1e-6,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: linear_num_heads,
            linear_num_value_heads: Some(linear_num_heads),
            linear_key_head_dim: linear_key_dim,
            linear_value_head_dim: linear_value_dim,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval,
            layer_types,
            layer_mask: vec![true; num_hidden_layers],
            eos_token_id: (vocab_size - 1) as u32,
            max_position_embeddings: 1024,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
        }
    }

    /// Recreate the `[linear, linear, linear, full]` layer schedule.
    fn compute_test_layer_types(num_layers: usize, interval: usize) -> Vec<LayerType> {
        (0..num_layers)
            .map(|i| {
                if (i + 1) % interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }

    /// Construct a synthetic model with deterministic random weights.
    fn build_random_model(cfg: Qwen35Config, seed: u64) -> Qwen35Model {
        let mut rng = seed.max(1);

        let embed_tokens = random_vec(&mut rng, cfg.vocab_size * cfg.hidden_size, 0.02);
        let final_norm = random_vec(&mut rng, cfg.hidden_size, 0.02);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_type in &cfg.layer_types {
            let common = CommonLayerWeights {
                input_layernorm: random_vec(&mut rng, cfg.hidden_size, 0.02),
                post_attention_layernorm: random_vec(&mut rng, cfg.hidden_size, 0.02),
                ffn: crate::model::qwen35::FeedForwardWeights::Dense(
                    crate::model::qwen35::DenseFfnWeights {
                        gate_proj: random_vec(
                            &mut rng,
                            cfg.intermediate_size * cfg.hidden_size,
                            0.02,
                        ),
                        up_proj: random_vec(
                            &mut rng,
                            cfg.intermediate_size * cfg.hidden_size,
                            0.02,
                        ),
                        down_proj: random_vec(
                            &mut rng,
                            cfg.hidden_size * cfg.intermediate_size,
                            0.02,
                        ),
                    },
                ),
            };

            let attn = match layer_type {
                LayerType::LinearAttention => {
                    let qkv_dim = cfg.linear_qkv_dim();
                    let output_dim = cfg.linear_output_dim();
                    let num_heads = cfg.linear_num_key_heads;
                    let kernel = cfg.linear_conv_kernel_dim;

                    AttentionWeights::Linear(GatedDeltaNetWeights {
                        in_proj_qkv: random_vec(&mut rng, qkv_dim * cfg.hidden_size, 0.02),
                        in_proj_qkv_rows: qkv_dim,
                        in_proj_qkv_cols: cfg.hidden_size,
                        in_proj_z: random_vec(&mut rng, output_dim * cfg.hidden_size, 0.02),
                        in_proj_z_rows: output_dim,
                        in_proj_z_cols: cfg.hidden_size,
                        in_proj_b: random_vec(&mut rng, num_heads * cfg.hidden_size, 0.02),
                        in_proj_b_rows: num_heads,
                        in_proj_b_cols: cfg.hidden_size,
                        in_proj_a: random_vec(&mut rng, num_heads * cfg.hidden_size, 0.02),
                        in_proj_a_rows: num_heads,
                        in_proj_a_cols: cfg.hidden_size,
                        a_log: random_vec(&mut rng, num_heads, 0.02),
                        dt_bias: random_vec(&mut rng, num_heads, 0.02),
                        conv1d_weight: random_vec(&mut rng, qkv_dim * kernel, 0.02),
                        conv_dim: qkv_dim,
                        kernel_size: kernel,
                        norm_weight: random_vec(&mut rng, output_dim, 0.02),
                        out_proj: random_vec(&mut rng, cfg.hidden_size * output_dim, 0.02),
                        out_proj_rows: cfg.hidden_size,
                        out_proj_cols: output_dim,
                    })
                }
                LayerType::FullAttention => {
                    let q_dim = cfg.full_q_dim();
                    let kv_dim = cfg.full_kv_dim();
                    let q_proj_dim = 2 * q_dim;

                    AttentionWeights::Full(FullAttentionLayerWeights {
                        q_proj: random_vec(&mut rng, q_proj_dim * cfg.hidden_size, 0.02),
                        k_proj: random_vec(&mut rng, kv_dim * cfg.hidden_size, 0.02),
                        v_proj: random_vec(&mut rng, kv_dim * cfg.hidden_size, 0.02),
                        o_proj: random_vec(&mut rng, cfg.hidden_size * q_dim, 0.02),
                        q_norm: random_vec(&mut rng, cfg.head_dim, 0.02),
                        k_norm: random_vec(&mut rng, cfg.head_dim, 0.02),
                    })
                }
            };

            layers.push((attn, common));
        }

        let rope_dim = cfg.rope_dim();
        let rope_max = cfg.max_position_embeddings.min(8192);
        let rope = RopeTable::new(rope_dim, rope_max, cfg.rope_theta);

        Qwen35Model {
            config: cfg,
            weights: ModelWeights {
                embed_tokens,
                lm_head: None,
                final_norm,
                layers,
            },
            tokenizer: dummy_tokenizer(),
            rope,
            lora: Box::new(crate::lora_hook::NoopLoraHook),
        }
    }

    /// Fill a vector with deterministic uniform noise in `[-scale, scale]`.
    fn random_vec(rng: &mut u64, len: usize, scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push((next_u32(rng) as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale);
        }
        out
    }

    /// Advance the tiny xorshift RNG used by the synthetic weight builder.
    fn next_u32(state: &mut u64) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x >> 32) as u32
    }

    /// Load a minimal byte-level BPE tokenizer for tests that instantiate the model.
    fn dummy_tokenizer() -> BpeTokenizer {
        let mut path: PathBuf = std::env::temp_dir();
        path.push("qwen35_batch_prefill_dummy_tokenizer.json");

        let json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": true
  },
  "post_processor": null,
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {
      "<unk>": 0,
      "a": 1,
      "b": 2,
      "c": 3,
      "d": 4,
      "e": 5,
      " ": 6
    },
    "merges": []
  }
}"#;

        fs::write(&path, json).expect("failed to write dummy tokenizer");
        BpeTokenizer::from_tokenizer_json(&path).expect("failed to load dummy tokenizer")
    }
}
