//! Gemma 4 E2B text-only CPU forward + greedy generate (ADR-082 stage 5).
//!
//! Assembles the already-merged stages: config/loader preflight (#996),
//! text math kernels with per-op HF goldens (#999, [`super::gemma4_ops`]),
//! and the donor-slot shared-KV cache (#1002, [`super::gemma4_cache`]).
//! Single-token-at-a-time forward (`forward_step`), called in a loop for
//! both prompt prefill and greedy decode -- the same shape as
//! `qwen35::forward::forward_step`. This is CPU-only, f32 throughout, no
//! Metal wiring (ADR-082's stage ladder gates Metal on CPU parity landing
//! first).
//!
//! Local+global softmax attention, **not** GDN: 28 `sliding_attention`
//! layers (window 512, 8x256 Q / 1x256 KV) interleaved with 7
//! `full_attention` layers at indices 4, 9, 14, 19, 24, 29, 34 (8x512 Q /
//! 1x512 KV) -- ADR-082 G3/G4. Attention uses **no** `1/sqrt(head_dim)`
//! score scaling (`Gemma4TextAttention.__init__` sets `self.scaling = 1.0`
//! unconditionally, verified directly against the pinned reference source
//! rather than assumed) -- a materially different convention from this
//! crate's Qwen3.5 GQA attention, so that kernel is not reused here.

use super::gemma4_cache::Gemma4KvCache;
use super::gemma4_config::Gemma4Config;
use super::gemma4_loading::load_weights;
use super::gemma4_ops::{
    gemma4_apply_rope, gemma4_geglu_mlp, gemma4_gelu_tanh, gemma4_logit_softcap, gemma4_rms_norm,
    gemma4_rope_cos_sin, gemma4_rope_inv_freq, gemma4_scaled_embedding,
};
use super::gemma4_weights::Gemma4Weights;
use crate::error::InferenceError;
use crate::forward::cpu::{elementwise_mul, matmul_bt, rms_norm};
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
use crate::weights::SafetensorsFile;
use std::path::Path;

/// Per-layer captured hidden-state trace: `(layer_idx, hidden_state)` pairs,
/// in the order layers were visited.
type LayerProbeTrace = Vec<(usize, Vec<f32>)>;

/// Reusable per-token forward-pass buffers (mirrors
/// `qwen35::cache::ForwardScratch`): every `Vec` [`Gemma4Model::forward_step`]
/// previously allocated fresh inside its 35-layer loop (or once per call for
/// the vocab-sized logits buffer) now lives here, sized once at
/// [`Self::new`] and reused across every token of a generation. Callers own
/// one instance per generation (or per test) and pass it `&mut` into every
/// `forward_step` call.
pub(crate) struct Gemma4Scratch {
    hidden: Vec<f32>,
    residual: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
    context: Vec<f32>,
    attn_out: Vec<f32>,
    residual2: Vec<f32>,
    normed2: Vec<f32>,
    ffn_out: Vec<f32>,
    residual3: Vec<f32>,
    gate: Vec<f32>,
    proj: Vec<f32>,
    /// MLP gate/up projections, sized once to the widest
    /// [`Gemma4Config::mlp_intermediate_size`] across all layers (the
    /// double-wide KV-shared layers) and reused by every layer's
    /// [`gemma4_geglu_mlp`] call instead of allocating fresh per layer.
    mlp_gate: Vec<f32>,
    mlp_up: Vec<f32>,
    /// All-ones weight for the unscaled V RMSNorm (G4), sized to the widest
    /// attention head width across sliding/global layers and reused every
    /// layer instead of allocating a fresh `vec![1.0; head_w]` per call.
    v_norm_ones: Vec<f32>,
    /// Post-softcap logits, `[vocab_size]`. `forward_step` writes here
    /// instead of returning an owned vector; callers that need an owned
    /// copy across the generation boundary clone out of this once, not on
    /// every token.
    pub(crate) logits: Vec<f32>,
}

impl Gemma4Scratch {
    /// Allocate every buffer at its call-independent maximum size (head
    /// width varies sliding-vs-global, so `q`/`k`/`v`/`context` are sized to
    /// the wider of the two head dims). `scores` is grown lazily in
    /// [`Gemma4Model::forward_step`] since its length tracks the live KV
    /// sequence length, not a config constant.
    pub(crate) fn new(cfg: &Gemma4Config) -> Self {
        let hidden_size = cfg.hidden_size;
        let widest_head = cfg.head_dim.max(cfg.global_head_dim);
        let q_dim_max = cfg.num_attention_heads * widest_head;
        let kv_dim_max = cfg.num_key_value_heads * widest_head;
        let widest_mlp_intermediate = (0..cfg.num_hidden_layers)
            .map(|layer_idx| cfg.mlp_intermediate_size(layer_idx))
            .max()
            .unwrap_or(cfg.intermediate_size);
        Self {
            hidden: vec![0f32; hidden_size],
            residual: vec![0f32; hidden_size],
            normed: vec![0f32; hidden_size],
            q: vec![0f32; q_dim_max],
            k: vec![0f32; kv_dim_max],
            v: vec![0f32; kv_dim_max],
            scores: Vec::new(),
            context: vec![0f32; q_dim_max],
            attn_out: vec![0f32; hidden_size],
            residual2: vec![0f32; hidden_size],
            normed2: vec![0f32; hidden_size],
            ffn_out: vec![0f32; hidden_size],
            residual3: vec![0f32; hidden_size],
            gate: vec![0f32; cfg.hidden_size_per_layer_input],
            proj: vec![0f32; hidden_size],
            mlp_gate: vec![0f32; widest_mlp_intermediate],
            mlp_up: vec![0f32; widest_mlp_intermediate],
            v_norm_ones: vec![1.0f32; widest_head],
            logits: vec![0f32; cfg.vocab_size],
        }
    }

    fn ensure_scores_capacity(&mut self, n: usize) {
        if self.scores.len() < n {
            self.scores.resize(n, 0.0);
        }
    }
}

/// **Unstable**: Gemma 4 E2B text-only generation model.
pub struct Gemma4Model {
    pub(crate) config: Gemma4Config,
    pub(crate) weights: Gemma4Weights,
    pub(crate) tokenizer: GemmaBpeTokenizer,
    /// Sliding-layer RoPE inverse-frequency table, length `head_dim / 2`.
    local_inv_freq: Vec<f32>,
    /// Global-layer RoPE inverse-frequency table (proportional, zero-padded
    /// past `partial_rotary_factor * global_head_dim / 2`), length
    /// `global_head_dim / 2`.
    global_inv_freq: Vec<f32>,
}

impl Gemma4Model {
    /// **Unstable**: load a Gemma 4 E2B model from a local safetensors
    /// directory (`config.json`, `tokenizer.json`, `model.safetensors`).
    pub fn from_safetensors(path: &Path) -> Result<Self, InferenceError> {
        let config = Gemma4Config::from_model_dir(path)?;

        let model_path = path.join("model.safetensors");
        if !model_path.exists() {
            return Err(InferenceError::ModelNotFound(format!(
                "missing model.safetensors in {}",
                path.display()
            )));
        }
        let mut source = SafetensorsFile::open(&model_path)?;
        let weights = load_weights(&mut source, &config)?;

        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = GemmaBpeTokenizer::from_tokenizer_json(&tokenizer_path)?;

        let local_inv_freq =
            gemma4_rope_inv_freq(config.head_dim, config.rope_local_base_freq, None);
        let global_inv_freq = gemma4_rope_inv_freq(
            config.global_head_dim,
            config.rope_theta,
            Some(config.partial_rotary_factor),
        );

        Ok(Self {
            config,
            weights,
            tokenizer,
            local_inv_freq,
            global_inv_freq,
        })
    }

    /// **Unstable**: access Gemma 4 configuration.
    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    /// **Unstable**: access the Gemma BPE tokenizer.
    pub fn tokenizer(&self) -> &GemmaBpeTokenizer {
        &self.tokenizer
    }

    /// **Unstable**: allocate a fresh KV cache sized for up to
    /// `max_seq_len` tokens on non-sliding (global) layers.
    pub fn new_cache(&self, max_seq_len: usize) -> Result<Gemma4KvCache, InferenceError> {
        Gemma4KvCache::new(&self.config, max_seq_len)
    }

    /// **Unstable**: single-token forward pass. Writes post-softcap logits
    /// into `scratch.logits` (`[vocab_size]`) rather than returning an owned
    /// vector -- callers that need an owned copy across a generation
    /// boundary clone out of `scratch.logits` once, not on every token (see
    /// [`Gemma4Scratch`]). `capture_layers` names zero-based layer indices
    /// whose post-layer hidden state (after that layer's PLE residual and
    /// `layer_scalar` multiply, before the final norm) should be recorded
    /// into the returned trace, in the order layers are visited --
    /// equivalent to HF's `output_hidden_states=True` trace entry
    /// `hidden_states[layer + 1]` at this token's position.
    ///
    /// # Errors
    /// Propagates any [`Gemma4KvCache`] error (out-of-bounds layer, shared-layer
    /// write, or capacity overflow) and fails closed if `token_id` is out of
    /// the configured vocabulary.
    pub(crate) fn forward_step(
        &self,
        token_id: u32,
        position: usize,
        cache: &mut Gemma4KvCache,
        scratch: &mut Gemma4Scratch,
        capture_layers: &[usize],
    ) -> Result<LayerProbeTrace, InferenceError> {
        let cfg = &self.config;
        let hidden_size = cfg.hidden_size;
        if token_id as usize >= cfg.vocab_size {
            return Err(InferenceError::InvalidInput(format!(
                "gemma4 forward: token_id {token_id} out of range (vocab_size={})",
                cfg.vocab_size
            )));
        }

        let per_layer_dim = cfg.hidden_size_per_layer_input;
        let ple_packed_dim = cfg.num_hidden_layers * per_layer_dim;

        // -- Scaled token embedding (G10a). --
        gemma4_scaled_embedding(
            &[token_id],
            &self.weights.embed_tokens,
            hidden_size,
            &mut scratch.hidden[..hidden_size],
        );

        // -- Per-Layer Embeddings (PLE, G9): token-identity + context. --
        // Computed once per token (not per layer), so left as an owned
        // return value rather than threaded through `Gemma4Scratch` -- the
        // hot-path cost this buffer-reuse pass targets is the 35x-per-layer
        // repetition below, not this once-per-token allocation.
        let per_layer_inputs = self.compute_per_layer_inputs(
            &scratch.hidden[..hidden_size],
            token_id,
            per_layer_dim,
            ple_packed_dim,
        );

        // -- Dual RoPE cos/sin, computed once per token (position-only). --
        let (cos_local, sin_local) = gemma4_rope_cos_sin(&self.local_inv_freq, &[position as u32]);
        let (cos_global, sin_global) =
            gemma4_rope_cos_sin(&self.global_inv_freq, &[position as u32]);

        let mut captured = Vec::with_capacity(capture_layers.len());

        for layer_idx in 0..cfg.num_hidden_layers {
            let lw = &self.weights.layers[layer_idx];
            let is_global = cfg.is_global_layer(layer_idx);
            let is_shared = cfg.is_kv_shared_layer(layer_idx);
            let head_w = cfg.attn_head_dim(layer_idx);
            let num_q_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let q_dim = num_q_heads * head_w;
            let kv_dim = num_kv_heads * head_w;
            let (cos, sin) = if is_global {
                (&cos_global, &sin_global)
            } else {
                (&cos_local, &sin_local)
            };

            // -- Attention block. --
            scratch.residual[..hidden_size].copy_from_slice(&scratch.hidden[..hidden_size]);
            scratch.normed[..hidden_size].copy_from_slice(&scratch.hidden[..hidden_size]);
            gemma4_rms_norm(
                &mut scratch.normed[..hidden_size],
                &lw.input_layernorm,
                hidden_size,
                cfg.rms_norm_eps,
            );

            matmul_bt(
                &scratch.normed[..hidden_size],
                &lw.q_proj,
                &mut scratch.q[..q_dim],
                1,
                hidden_size,
                q_dim,
            );
            for h in 0..num_q_heads {
                let start = h * head_w;
                gemma4_rms_norm(
                    &mut scratch.q[start..start + head_w],
                    &lw.q_norm,
                    head_w,
                    cfg.rms_norm_eps,
                );
            }
            gemma4_apply_rope(&mut scratch.q[..q_dim], cos, sin, 1, num_q_heads, head_w);

            if !is_shared {
                let k_proj = lw.k_proj.as_ref().ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "gemma4 forward: layer {layer_idx} is non-shared but has no k_proj weights"
                    ))
                })?;
                let v_proj = lw.v_proj.as_ref().ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "gemma4 forward: layer {layer_idx} is non-shared but has no v_proj weights"
                    ))
                })?;
                let k_norm = lw.k_norm.as_ref().ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "gemma4 forward: layer {layer_idx} is non-shared but has no k_norm weights"
                    ))
                })?;

                matmul_bt(
                    &scratch.normed[..hidden_size],
                    k_proj,
                    &mut scratch.k[..kv_dim],
                    1,
                    hidden_size,
                    kv_dim,
                );
                matmul_bt(
                    &scratch.normed[..hidden_size],
                    v_proj,
                    &mut scratch.v[..kv_dim],
                    1,
                    hidden_size,
                    kv_dim,
                );

                for h in 0..num_kv_heads {
                    let start = h * head_w;
                    gemma4_rms_norm(
                        &mut scratch.k[start..start + head_w],
                        k_norm,
                        head_w,
                        cfg.rms_norm_eps,
                    );
                }
                gemma4_apply_rope(&mut scratch.k[..kv_dim], cos, sin, 1, num_kv_heads, head_w);
                for h in 0..num_kv_heads {
                    let start = h * head_w;
                    rms_norm(
                        &mut scratch.v[start..start + head_w],
                        &scratch.v_norm_ones[..head_w],
                        head_w,
                        cfg.rms_norm_eps,
                    );
                }

                cache.append_kv(layer_idx, &scratch.k[..kv_dim], &scratch.v[..kv_dim])?;
            }

            let seq_len = cache.seq_len(layer_idx)?;
            let k_view = cache.k_view(layer_idx)?;
            let v_view = cache.v_view(layer_idx)?;

            let groups = num_q_heads / num_kv_heads;
            scratch.ensure_scores_capacity(seq_len);
            for qh in 0..num_q_heads {
                let kvh = qh / groups;
                let q_head = &scratch.q[qh * head_w..(qh + 1) * head_w];
                let scores = &mut scratch.scores[..seq_len];
                for t in 0..seq_len {
                    let k_off = t * kv_dim + kvh * head_w;
                    let mut dot = 0.0f32;
                    for d in 0..head_w {
                        dot += q_head[d] * k_view[k_off + d];
                    }
                    // Gemma4TextAttention.scaling == 1.0 (verified against
                    // the pinned reference source, not the usual
                    // 1/sqrt(head_dim) convention) -- no scale applied here.
                    scores[t] = dot;
                }
                softmax_row_fail_closed(scores);
                let ctx_off = qh * head_w;
                for d in 0..head_w {
                    let mut sum = 0.0f32;
                    for t in 0..seq_len {
                        let v_off = t * kv_dim + kvh * head_w;
                        sum += scratch.scores[t] * v_view[v_off + d];
                    }
                    scratch.context[ctx_off + d] = sum;
                }
            }

            matmul_bt(
                &scratch.context[..q_dim],
                &lw.o_proj,
                &mut scratch.attn_out[..hidden_size],
                1,
                q_dim,
                hidden_size,
            );
            gemma4_rms_norm(
                &mut scratch.attn_out[..hidden_size],
                &lw.post_attention_layernorm,
                hidden_size,
                cfg.rms_norm_eps,
            );
            for i in 0..hidden_size {
                scratch.hidden[i] = scratch.residual[i] + scratch.attn_out[i];
            }

            // -- FFN block. --
            scratch.residual2[..hidden_size].copy_from_slice(&scratch.hidden[..hidden_size]);
            scratch.normed2[..hidden_size].copy_from_slice(&scratch.hidden[..hidden_size]);
            gemma4_rms_norm(
                &mut scratch.normed2[..hidden_size],
                &lw.pre_feedforward_layernorm,
                hidden_size,
                cfg.rms_norm_eps,
            );
            let mlp_dim = cfg.mlp_intermediate_size(layer_idx);
            gemma4_geglu_mlp(
                &scratch.normed2[..hidden_size],
                &lw.gate_proj,
                &lw.up_proj,
                &lw.down_proj,
                1,
                hidden_size,
                mlp_dim,
                &mut scratch.mlp_gate[..mlp_dim],
                &mut scratch.mlp_up[..mlp_dim],
                &mut scratch.ffn_out[..hidden_size],
            );
            gemma4_rms_norm(
                &mut scratch.ffn_out[..hidden_size],
                &lw.post_feedforward_layernorm,
                hidden_size,
                cfg.rms_norm_eps,
            );
            for i in 0..hidden_size {
                scratch.hidden[i] = scratch.residual2[i] + scratch.ffn_out[i];
            }

            // -- Per-layer-embedding residual gate (G9). --
            scratch.residual3[..hidden_size].copy_from_slice(&scratch.hidden[..hidden_size]);
            matmul_bt(
                &scratch.hidden[..hidden_size],
                &lw.per_layer_input_gate,
                &mut scratch.gate[..per_layer_dim],
                1,
                hidden_size,
                per_layer_dim,
            );
            gemma4_gelu_tanh(&mut scratch.gate[..per_layer_dim]);
            let this_layer_input =
                &per_layer_inputs[layer_idx * per_layer_dim..(layer_idx + 1) * per_layer_dim];
            elementwise_mul(&mut scratch.gate[..per_layer_dim], this_layer_input);
            matmul_bt(
                &scratch.gate[..per_layer_dim],
                &lw.per_layer_projection,
                &mut scratch.proj[..hidden_size],
                1,
                per_layer_dim,
                hidden_size,
            );
            gemma4_rms_norm(
                &mut scratch.proj[..hidden_size],
                &lw.post_per_layer_input_norm,
                hidden_size,
                cfg.rms_norm_eps,
            );
            for i in 0..hidden_size {
                scratch.hidden[i] = scratch.residual3[i] + scratch.proj[i];
            }

            for v in scratch.hidden[..hidden_size].iter_mut() {
                *v *= lw.layer_scalar;
            }

            if capture_layers.contains(&layer_idx) {
                captured.push((layer_idx, scratch.hidden[..hidden_size].to_vec()));
            }
        }

        gemma4_rms_norm(
            &mut scratch.hidden[..hidden_size],
            &self.weights.norm,
            hidden_size,
            cfg.rms_norm_eps,
        );

        matmul_bt(
            &scratch.hidden[..hidden_size],
            &self.weights.embed_tokens,
            &mut scratch.logits[..cfg.vocab_size],
            1,
            hidden_size,
            cfg.vocab_size,
        );
        gemma4_logit_softcap(
            &mut scratch.logits[..cfg.vocab_size],
            cfg.final_logit_softcapping,
        );

        Ok(captured)
    }

    /// PLE token-identity (`embed_tokens_per_layer`, scaled by
    /// `sqrt(hidden_size_per_layer_input)`) combined with the context
    /// projection (`per_layer_model_projection(embed) * hidden_size^-0.5`,
    /// per-layer-normalized), per `Gemma4TextModel.project_per_layer_inputs`
    /// (`modeling_gemma4.py:1798-1821`): `(context + identity) / sqrt(2)`.
    /// Returns a packed `[num_hidden_layers * per_layer_dim]` buffer.
    fn compute_per_layer_inputs(
        &self,
        scaled_embed: &[f32],
        token_id: u32,
        per_layer_dim: usize,
        ple_packed_dim: usize,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hidden_size = cfg.hidden_size;

        let id_scale = (per_layer_dim as f32).sqrt();
        let row_start = token_id as usize * ple_packed_dim;
        let mut identity: Vec<f32> = self.weights.embed_tokens_per_layer
            [row_start..row_start + ple_packed_dim]
            .iter()
            .map(|&v| v * id_scale)
            .collect();

        let mut ctx = vec![0f32; ple_packed_dim];
        matmul_bt(
            scaled_embed,
            &self.weights.per_layer_model_projection,
            &mut ctx,
            1,
            hidden_size,
            ple_packed_dim,
        );
        let ctx_scale = 1.0 / (hidden_size as f32).sqrt();
        for v in ctx.iter_mut() {
            *v *= ctx_scale;
        }
        for layer in 0..cfg.num_hidden_layers {
            let start = layer * per_layer_dim;
            gemma4_rms_norm(
                &mut ctx[start..start + per_layer_dim],
                &self.weights.per_layer_projection_norm,
                per_layer_dim,
                cfg.rms_norm_eps,
            );
        }

        let combine_scale = std::f32::consts::FRAC_1_SQRT_2;
        for i in 0..ple_packed_dim {
            identity[i] = (ctx[i] + identity[i]) * combine_scale;
        }
        identity
    }

    /// **Unstable**: greedy-decode `max_new_tokens` continuation tokens for
    /// `prompt_ids` (already tokenized, BOS included by the caller). Runs
    /// prefill and decode through the same single-token [`Self::forward_step`]
    /// loop.
    ///
    /// # Errors
    /// Propagates [`Self::forward_step`] errors (invalid token id, cache
    /// bounds/capacity).
    pub fn generate_greedy(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        max_seq_len: usize,
    ) -> Result<Vec<u32>, InferenceError> {
        let mut cache = self.new_cache(max_seq_len)?;
        let mut scratch = Gemma4Scratch::new(&self.config);
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut position = 0usize;

        for &tok in prompt_ids {
            self.forward_step(tok, position, &mut cache, &mut scratch, &[])?;
            position += 1;
        }

        for _ in 0..max_new_tokens {
            let next = argmax(&scratch.logits[..self.config.vocab_size]);
            generated.push(next);
            self.forward_step(next, position, &mut cache, &mut scratch, &[])?;
            position += 1;
        }

        Ok(generated)
    }

    /// **Unstable**: like [`Self::generate_greedy`], but also returns the
    /// per-layer hidden-state trace captured at the LAST prompt position,
    /// before any generated token is produced. `capture_layers` names
    /// zero-based layer indices; the trace records each captured layer's
    /// post-layer hidden state (after that layer's PLE residual and
    /// `layer_scalar` multiply, before the final norm) -- equivalent to
    /// HF's `output_hidden_states=True` trace entry `hidden_states[layer +
    /// 1]` at the prompt's last position.
    ///
    /// # Errors
    /// Propagates any cache error (out-of-bounds layer, shared-layer write,
    /// capacity overflow) or invalid-token-id error from the underlying
    /// per-token forward pass.
    pub fn generate_greedy_with_probe(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        max_seq_len: usize,
        probe_layers: &[usize],
    ) -> Result<(Vec<u32>, Vec<f32>, LayerProbeTrace), InferenceError> {
        let mut cache = self.new_cache(max_seq_len)?;
        let mut scratch = Gemma4Scratch::new(&self.config);
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut position = 0usize;
        let mut probe = Vec::new();

        for (i, &tok) in prompt_ids.iter().enumerate() {
            let is_last = i + 1 == prompt_ids.len();
            let layers: &[usize] = if is_last { probe_layers } else { &[] };
            let captured = self.forward_step(tok, position, &mut cache, &mut scratch, layers)?;
            if is_last {
                probe = captured;
            }
            position += 1;
        }

        // Snapshot the prompt's-last-position logits right after prefill,
        // before any generated token's forward pass overwrites
        // `scratch.logits` -- this is what the HF golden's
        // `final_logits_last_pos_top8` records (logits used to pick the
        // *first* greedy token), not whatever `scratch.logits` holds after
        // the last generated token.
        let final_logits = scratch.logits[..self.config.vocab_size].to_vec();

        for _ in 0..max_new_tokens {
            let next = argmax(&scratch.logits[..self.config.vocab_size]);
            generated.push(next);
            self.forward_step(next, position, &mut cache, &mut scratch, &[])?;
            position += 1;
        }

        Ok((generated, final_logits, probe))
    }
}

/// Row-wise softmax, fail-closed on non-finite input (this repo's softmax
/// bug class: silently propagating NaN/+inf produces plausible-looking but
/// wrong attention weights rather than a loud failure).
fn softmax_row_fail_closed(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        row.fill(0.0);
        if !row.is_empty() {
            row[row.len() - 1] = 1.0;
        }
        return;
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    if !sum.is_finite() || sum <= 0.0 {
        row.fill(0.0);
        if !row.is_empty() {
            row[row.len() - 1] = 1.0;
        }
        return;
    }
    for v in row.iter_mut() {
        *v /= sum;
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Mutation-sensitivity proof for ADR-082 stage 5's declared negative test
/// (mirrors stage 4's "point a shared-KV layer at the wrong owner layer's
/// state" -- the ladder's highest silent-correctness risk). Requires the
/// real checkpoint and the `f16` feature; skips (not fails) when the
/// checkpoint is absent, same convention as
/// `tests/gemma4_e2e_forward_test.rs`.
#[cfg(all(test, feature = "f16"))]
mod donor_mutation_tests {
    use super::*;

    fn model_dir() -> Option<std::path::PathBuf> {
        let raw = std::env::var("LATTICE_GEMMA4_MODEL_DIR")
            .unwrap_or_else(|_| "~/.lattice/models/gemma-4-e2b-it".to_string());
        let path = if let Some(rest) = raw.strip_prefix("~/") {
            std::path::PathBuf::from(std::env::var("HOME").ok()?).join(rest)
        } else {
            std::path::PathBuf::from(raw)
        };
        path.join("model.safetensors").exists().then_some(path)
    }

    /// Runs the same per-token loop as [`Gemma4Model::generate_greedy_with_probe`],
    /// but against a caller-supplied cache instead of one the model builds
    /// internally -- lets the mutation test below install a corrupted donor
    /// map on the cache before the first token, without touching the
    /// model's (correctly loaded) config or weights at all.
    fn run_greedy_with_probe_on_cache(
        model: &Gemma4Model,
        input_ids: &[u32],
        max_new_tokens: usize,
        cache: &mut Gemma4KvCache,
        probe_layers: &[usize],
    ) -> (Vec<u32>, LayerProbeTrace) {
        let mut scratch = Gemma4Scratch::new(&model.config);
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut position = 0usize;
        let mut probe = Vec::new();

        for (i, &tok) in input_ids.iter().enumerate() {
            let is_last = i + 1 == input_ids.len();
            let layers: &[usize] = if is_last { probe_layers } else { &[] };
            let captured = model
                .forward_step(tok, position, cache, &mut scratch, layers)
                .expect("forward_step");
            if is_last {
                probe = captured;
            }
            position += 1;
        }
        for _ in 0..max_new_tokens {
            let next = argmax(&scratch.logits[..model.config.vocab_size]);
            generated.push(next);
            model
                .forward_step(next, position, cache, &mut scratch, &[])
                .expect("forward_step");
            position += 1;
        }
        (generated, probe)
    }

    /// Layer 34 is a shared global layer whose correct donor is layer 14
    /// (the last non-shared global layer -- see
    /// `gemma4_cache::tests::e2b_slot_map_matches_amendment_1`). This test
    /// forces layer 34's resolved slot to layer 9 instead -- a real
    /// non-shared global layer, so the swap doesn't cross attention types
    /// or trip a type-mismatch guard, it just silently reads the *wrong*
    /// owner's K/V, exactly the "off-by-one/wrong donor layer" failure mode
    /// ADR-082's Stage 4 negative test targets -- and proves the forward
    /// output diverges end-to-end, not just the cache's own slot-map unit
    /// tests. The model's config and weights are untouched, so this
    /// isolates the donor-mapping's own correctness contribution: no MLP
    /// width or k/v-projection-presence classification changes, only which
    /// slot layer 34's attention reads from.
    #[test]
    fn wrong_donor_mapping_diverges_layer34_probe_and_greedy_tokens() {
        let Some(dir) = model_dir() else {
            eprintln!("LATTICE_GEMMA4_MUTATION_TEST_SKIPPED reason=missing_checkpoint");
            return;
        };
        let model = Gemma4Model::from_safetensors(&dir).expect("loading real checkpoint");
        let input_ids: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563]; // BOS + "The capital of France is"

        let (baseline_greedy, _, baseline_probe) = model
            .generate_greedy_with_probe(&input_ids, 3, 64, &[34])
            .expect("baseline forward");

        let mut mutated_cache = model.new_cache(64).expect("cache construction");
        assert_eq!(
            mutated_cache.layer_slot(34).unwrap(),
            14,
            "sanity: correct donor before mutation"
        );
        mutated_cache.override_layer_slot_for_test(34, 9);
        assert_eq!(mutated_cache.layer_slot(34).unwrap(), 9);

        let (mutated_greedy, mutated_probe) =
            run_greedy_with_probe_on_cache(&model, &input_ids, 3, &mut mutated_cache, &[34]);

        let baseline_hidden = &baseline_probe[0].1;
        let mutated_hidden = &mutated_probe[0].1;
        let diff = baseline_hidden
            .iter()
            .zip(mutated_hidden.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // The real e2e gate (`gemma4_e2e_forward_test.rs::run_gate`) asserts
        // `diff <= 1e-3` at every probed layer, against measured
        // correct-donor diffs of ~1e-5. A wrong donor must blow through
        // that tolerance by a wide margin -- proving this is a real gate
        // failure under the gate's own numbers, not an arbitrary threshold
        // picked for this test alone.
        const E2E_GATE_TOLERANCE: f32 = 1e-3;
        assert!(
            diff > E2E_GATE_TOLERANCE * 10.0,
            "wrong-donor mutation must blow through the e2e gate's own {E2E_GATE_TOLERANCE} \
             tolerance by a wide margin (got diff {diff}) -- otherwise this test is decorative"
        );
        eprintln!(
            "donor mutation: layer 34 hidden-state max-abs-diff={diff} (gate tolerance \
             {E2E_GATE_TOLERANCE}, correct-donor baseline ~1e-5) -- fails the e2e gate's \
             per-layer probe assertion. greedy tokens baseline={baseline_greedy:?} \
             mutated={mutated_greedy:?} (top-1 margin at this prompt is wide enough, ~5 \
             logit points, that this single-layer perturbation does not always flip argmax; \
             the per-layer probe assertion is the gate this mutation is proven against)."
        );
    }
}
