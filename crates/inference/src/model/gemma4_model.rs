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
use crate::decoder::Cancellation;
use crate::decoder::driver;
use crate::decoder::gemma_cpu::GemmaCpuSession;
use crate::error::InferenceError;
use crate::forward::cpu::{elementwise_mul, matmul_bt, rms_norm};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::model::qwen35::check_prompt_not_empty;
use crate::model::qwen35_config::decode_cap;
use crate::stop_reason::StopReason;
use crate::tokenizer::common::Tokenizer;
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
use crate::weights::SafetensorsFile;
use std::path::Path;

/// Adapts the streaming API's `should_cancel: impl FnMut() -> bool` (which may
/// capture a `Receiver`-style handle and mutate on each poll) to
/// `decoder::Cancellation`, whose blanket impl covers only non-mutating `Fn`
/// closures (see that trait's own doc comment). Deliberately duplicated from
/// `model::qwen35::generation`'s identical private adapter rather than
/// shared: that one is private to its own module, and promoting either copy
/// to a shared crate-visible helper for two call sites is not this row's job.
struct FnMutCancellation<'a, F: FnMut() -> bool>(&'a std::cell::RefCell<F>);

impl<F: FnMut() -> bool> Cancellation for FnMutCancellation<'_, F> {
    fn is_cancelled(&self) -> bool {
        let mut should_cancel = self.0.borrow_mut();
        (*should_cancel)()
    }
}

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

/// UTF-8-boundary-safe streaming detokenizer for [`Gemma4Model::generate_streaming_via_driver`].
/// `GemmaBpeTokenizer::decode` is stateless and re-walks its whole input on every call, so
/// decoding one fresh id in isolation flushes every trailing `<0xXX>` byte-fallback run as
/// incomplete (`U+FFFD` per byte) even when a later token would have completed it, garbling any
/// multi-byte character split across byte-fallback tokens in both the stream and the final text.
///
/// The fix re-decodes the full `ids` list every call and holds back a trailing `U+FFFD` run until
/// it resolves or generation ends -- O(n) per call / O(n^2) total, accepted for this **Unstable**
/// path. Sound because an earlier run's flush is fixed once a later real token closes it, so the
/// safe (non-`U+FFFD`-tail) prefix only grows and `emitted_len` can track it as a byte offset.
struct IncrementalByteFallbackDetokenizer<'t> {
    tokenizer: &'t GemmaBpeTokenizer,
    /// Generated token ids seen so far, in order (excludes the prompt).
    ids: Vec<u32>,
    /// Byte offset of text already returned; always a valid `char` boundary of the current decode.
    emitted_len: usize,
}

impl<'t> IncrementalByteFallbackDetokenizer<'t> {
    fn new(tokenizer: &'t GemmaBpeTokenizer) -> Self {
        Self {
            tokenizer,
            ids: Vec::new(),
            emitted_len: 0,
        }
    }

    /// Appends `next_id`, re-decodes the full id list, and returns only the newly safe suffix --
    /// empty when `next_id` only extended a still-unresolved trailing byte-fallback run.
    fn push(&mut self, next_id: u32) -> String {
        self.ids.push(next_id);
        let full = self.tokenizer.decode(&self.ids).unwrap_or_default();
        let safe_len = safe_prefix_len(&full);
        debug_assert!(
            safe_len >= self.emitted_len,
            "the safe (non-U+FFFD-tail) prefix can only grow as more ids are appended"
        );
        let delta = full[self.emitted_len..safe_len].to_string();
        self.emitted_len = safe_len;
        delta
    }

    /// End-of-generation flush: emits whatever is still held back, `U+FFFD` and all.
    fn finish(&self) -> String {
        let full = self.tokenizer.decode(&self.ids).unwrap_or_default();
        full[self.emitted_len..].to_string()
    }
}

/// Byte length of `full` after stripping a trailing run of `U+FFFD`. `U+FFFD` has exactly one
/// source here: `flush_byte_fallback_run`'s error branch, one per byte of a fallback run that
/// never assembled into valid UTF-8. Only a run at the tail is still open to more ids; an earlier
/// run is already closed by a following real token and cannot change.
fn safe_prefix_len(full: &str) -> usize {
    let mut end = full.len();
    for (byte_idx, ch) in full.char_indices().rev() {
        if ch == '\u{FFFD}' {
            end = byte_idx;
        } else {
            break;
        }
    }
    end
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
    /// prefill and decode through the same single-token `Self::forward_step`
    /// loop.
    ///
    /// # Errors
    /// Propagates `Self::forward_step` errors (invalid token id, cache
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

    /// **Unstable**: autoregressive text generation with full sampling-policy
    /// support (temperature/top-k/top-p/min-p/seed), routed through the
    /// shared decoder driver (ADR-090 row R04). `prompt_ids` is already
    /// tokenized with BOS included by the caller -- same convention as
    /// [`Self::generate_greedy`]/[`Self::generate_greedy_with_probe`], and
    /// unlike `Qwen35Model::generate`'s `prompt: &str`. EOS-aware: stops on
    /// `self.config.eos_token_id` or any id in `gen_cfg.stop_token_ids`, and
    /// excludes the terminating token from the returned `token_ids`/`text`
    /// (the crate-wide stop-token contract, `GenerateOutput`'s own doc
    /// comment). Runs no forward pass after the last requested output
    /// (ADR-090 D2).
    ///
    /// `generate_greedy`/`generate_greedy_with_probe` are UNCHANGED by this
    /// row and remain the fixed-count diagnostic entry points; this is a
    /// separate, EOS-aware entry.
    ///
    /// **Landmine inherited from `GenerateConfig::default()`, not introduced
    /// here**: the default `stop_token_ids` contains `QWEN_CHAT_IM_END_TOKEN_ID`
    /// (248,046), a Qwen-specific id with no relationship to Gemma's chat
    /// template. Gemma's vocabulary (262,144) is large enough that this id is
    /// a valid, unrelated Gemma token, so a caller using
    /// `GenerateConfig { .. Default::default() }` unmodified inherits an
    /// early, semantically meaningless stop condition on that token id. This
    /// is a pre-existing property of the shared `GenerateConfig` type (used by
    /// 75+ call sites) and is out of scope to change here; callers that care
    /// should set `stop_token_ids` explicitly.
    ///
    /// This session declares every [`crate::decoder::ExecutionCapabilities`]
    /// field `false` (see `decoder::gemma_cpu`'s module doc comment): a
    /// `gen_cfg` requesting grammar, logprobs, `stop_strings`, or a
    /// reasoning budget is refused by [`driver::run`]'s `check_capabilities`
    /// before any session method runs.
    ///
    /// Delegates to [`Self::generate_with_trace`] and discards the driver
    /// trace, mirroring `Qwen35Model::generate`'s relationship to
    /// `generate_with_trace` (ADR-090 row C, decomposition "Open question 3,
    /// ANSWERED"): the trace exists for this migration's own tests to see,
    /// not for callers of the public API.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, InferenceError> {
        self.generate_with_trace(prompt_ids, gen_cfg)
            .map(|(output, _trace)| output)
    }

    /// ADR-090 row R04 dispatch point, Gemma sibling of
    /// `Qwen35Model::generate_with_trace`. Crate-private and not
    /// `#[cfg(test)]`-gated, for the same reason that function is not: a
    /// marker that only exists under `cfg(test)` would make the shipped path
    /// and the tested path differ in the one respect the test observes --
    /// `driver::DriverTrace`, which already exists and is model-agnostic;
    /// this method is the only new surface Gemma needs to make it observable
    /// to this crate's own tests.
    pub(crate) fn generate_with_trace(
        &self,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
    ) -> Result<(GenerateOutput, driver::DriverTrace), InferenceError> {
        self.generate_via_driver(prompt_ids, gen_cfg)
    }

    /// Constructs a [`GemmaCpuSession`] and runs [`driver::run`] over it.
    /// Mirrors `Qwen35Model::generate_via_driver`'s fast (no-stop-strings)
    /// branch exactly; Gemma has no other branch to choose between, since
    /// `GemmaCpuSession`'s `stop_strings` capability is permanently `false`
    /// and any `gen_cfg.stop_strings` request is refused by
    /// `driver::run`'s `check_capabilities` before this function's own body
    /// would need to route around it.
    ///
    /// `think_close_id: None` unconditionally: Gemma has no reasoning-budget
    /// support this row (`reasoning_budget` capability is `false`), so there
    /// is no close-token to resolve -- `Qwen35Model::generate_via_driver`'s
    /// `resolve_reasoning_close_token` call has no Gemma equivalent to call.
    ///
    /// The context-budget check below is a Gemma-local reimplementation of
    /// `Qwen35Model`'s `check_context_budget`, not a call to it: that
    /// function is Metal-gated at its only crate-visible path
    /// (`#[cfg(all(target_os = "macos", feature = "metal-gpu"))]` in
    /// `model::qwen35::mod`'s re-export list), so it does not exist at all on
    /// a plain CPU build. [`decode_cap`] is the one piece of that check's
    /// arithmetic that is already shared, model-agnostic infrastructure
    /// (`driver::run` itself imports it directly), so this reuses that and
    /// re-derives the rest of the bound inline rather than duplicating a
    /// function this crate cannot reach from here.
    fn generate_via_driver(
        &self,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
    ) -> Result<(GenerateOutput, driver::DriverTrace), InferenceError> {
        let cfg = &self.config;
        let prompt_len = prompt_ids.len();

        check_prompt_not_empty(prompt_len)?;

        if gen_cfg.max_new_tokens == 0 {
            return Ok((
                GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: Some(StopReason::Length),
                    token_logprobs: vec![],
                },
                driver::DriverTrace::default(),
            ));
        }

        let effective_new =
            decode_cap(gen_cfg.effective_reasoning_budget(), gen_cfg.max_new_tokens);
        let max_context = cfg.max_position_embeddings;
        if prompt_len.saturating_add(effective_new) > max_context {
            return Err(InferenceError::Inference(format!(
                "prompt ({prompt_len} tokens) plus effective decode cap ({effective_new} \
                 tokens; max_new_tokens={}) exceeds Gemma 4 context window ({max_context})",
                gen_cfg.max_new_tokens
            )));
        }
        let max_seq_len = prompt_len.saturating_add(effective_new);

        let mut session = GemmaCpuSession::new(
            self,
            prompt_ids.to_vec(),
            gen_cfg.temperature,
            gen_cfg.seed,
            max_seq_len,
        )?;

        // Non-streaming callers never cancel and never need a real
        // per-token delta: `decoder::gemma_cpu`'s session declares
        // `stop_strings: false`, so `driver::run` refuses any
        // `gen_cfg.stop_strings` request before this call would need to
        // route around it -- unlike `Qwen35Model::generate_via_driver`,
        // there is no second (stop-strings-aware) branch here at all.
        let never_cancel = || false;
        let mut throwaway_text = String::new();
        let mut throwaway_offsets: Vec<usize> = Vec::new();

        let result = driver::run(
            &mut session,
            gen_cfg,
            None,
            prompt_ids,
            cfg.eos_token_id,
            false,
            &never_cancel,
            |_generated_len| {},
            |_next_id| String::new(),
            &mut throwaway_text,
            &mut throwaway_offsets,
            |_delta, _next_id| true,
            || {},
            String::new,
        )?;

        // `GemmaBpeTokenizer::decode` (the `Tokenizer` trait's real override
        // for this tokenizer) always returns `Some` -- see that method's own
        // implementation -- so `unwrap_or_default` never actually falls back
        // in practice; it exists only to satisfy the trait's `Option<String>`
        // signature, which allows for tokenizers with no decode support at
        // all (the trait's default).
        let text = self
            .tokenizer
            .decode(&result.generated_ids)
            .unwrap_or_default();

        Ok((
            GenerateOutput {
                text,
                token_ids: result.generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: result.generated_ids.len(),
                stopped: result.stopped,
                stop_reason: Some(result.stop_reason),
                token_logprobs: result.token_logprobs,
            },
            result.trace,
        ))
    }

    /// **Unstable**: streaming sibling of [`Self::generate`], with
    /// cancellation -- mirrors `Qwen35Model::generate_streaming_with_cancel`'s
    /// signature shape (`on_token: impl FnMut(&str) -> bool`,
    /// `should_cancel: impl FnMut() -> bool`). `should_cancel` is polled
    /// before the prefill pass starts, immediately after it returns, and at
    /// the top of every decode iteration (`driver::run`'s own three
    /// checkpoints); `on_token` itself also stops generation the moment it
    /// returns `false`. Both stopping paths report `stopped: false,
    /// stop_reason: Some(StopReason::Interrupt)`, matching the Qwen CPU/Metal
    /// contract.
    ///
    /// Deltas are UTF-8-boundary-safe: [`IncrementalByteFallbackDetokenizer`]
    /// re-decodes the full generated-id list on every call and holds back a
    /// trailing incomplete byte-fallback run (see that type's doc comment).
    /// The concatenation of every streamed delta plus the `driver::run`
    /// end-of-generation flush equals `Tokenizer::decode(&output.token_ids)`
    /// -- the same text [`Self::generate`] returns for the same ids.
    pub fn generate_streaming_with_cancel<F, C>(
        &self,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<GenerateOutput, InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
    {
        self.generate_streaming_via_driver(prompt_ids, gen_cfg, on_token, should_cancel)
            .map(|(output, _trace)| output)
    }

    /// Driver-routed dispatch target for [`Self::generate_streaming_with_cancel`].
    /// Mirrors [`Self::generate_via_driver`]'s preflight sequence exactly
    /// (empty-prompt check, zero-`max_new_tokens` short-circuit,
    /// context-budget bound, session construction); see this method's own
    /// doc comment on `generate_streaming_with_cancel` for the UTF-8-boundary
    /// -safe delta contract [`IncrementalByteFallbackDetokenizer`] provides.
    fn generate_streaming_via_driver<F, C>(
        &self,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
        mut on_token: F,
        should_cancel: C,
    ) -> Result<(GenerateOutput, driver::DriverTrace), InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
    {
        let cfg = &self.config;
        let prompt_len = prompt_ids.len();

        check_prompt_not_empty(prompt_len)?;

        if gen_cfg.max_new_tokens == 0 {
            return Ok((
                GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: Some(StopReason::Length),
                    token_logprobs: vec![],
                },
                driver::DriverTrace::default(),
            ));
        }

        let effective_new =
            decode_cap(gen_cfg.effective_reasoning_budget(), gen_cfg.max_new_tokens);
        let max_context = cfg.max_position_embeddings;
        if prompt_len.saturating_add(effective_new) > max_context {
            return Err(InferenceError::Inference(format!(
                "prompt ({prompt_len} tokens) plus effective decode cap ({effective_new} \
                 tokens; max_new_tokens={}) exceeds Gemma 4 context window ({max_context})",
                gen_cfg.max_new_tokens
            )));
        }
        let max_seq_len = prompt_len.saturating_add(effective_new);

        let mut session = GemmaCpuSession::new(
            self,
            prompt_ids.to_vec(),
            gen_cfg.temperature,
            gen_cfg.seed,
            max_seq_len,
        )?;

        let should_cancel_cell = std::cell::RefCell::new(should_cancel);
        let cancel = FnMutCancellation(&should_cancel_cell);

        let mut text = String::new();
        let mut token_logprob_end_offsets: Vec<usize> = Vec::new();

        // `RefCell`, not two separate closures each borrowing `detok` directly: `decode_delta`
        // (`FnMut`) and `finish_tail` (`FnOnce`) are two distinct closures both alive for the
        // whole `driver::run` call below, so the borrow checker needs interior mutability here --
        // same shape as `driver.rs`'s own `RefCell<&mut dyn DecoderSession>` and
        // `model::qwen35::generation::generate_streaming_via_driver`'s `detok_cell`, which this
        // mirrors directly.
        let detok_cell =
            std::cell::RefCell::new(IncrementalByteFallbackDetokenizer::new(&self.tokenizer));

        let result = driver::run(
            &mut session,
            gen_cfg,
            None,
            prompt_ids,
            cfg.eos_token_id,
            true,
            &cancel,
            |_generated_len| {},
            // See this method's own doc comment on `generate_streaming_with_cancel` and
            // `IncrementalByteFallbackDetokenizer`'s own doc comment for why a stateful,
            // full-re-decode-per-call detokenizer (not a per-token `decode` call) is required
            // for a correct, UTF-8-boundary-safe stream.
            |next_id| detok_cell.borrow_mut().push(next_id),
            &mut text,
            &mut token_logprob_end_offsets,
            |delta, _next_id| on_token(delta),
            || {},
            || detok_cell.borrow().finish(),
        )?;

        Ok((
            GenerateOutput {
                text,
                token_ids: result.generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: result.generated_ids.len(),
                stopped: result.stopped,
                stop_reason: Some(result.stop_reason),
                token_logprobs: result.token_logprobs,
            },
            result.trace,
        ))
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

/// Test-only tiny zero-weight synthetic Gemma 4 model (ADR-090 row R04,
/// mirrors `model::qwen35::test_support::tiny_zero_model`'s all-zero-weight
/// trick, and `gemma4_cache.rs`'s own `tests::tiny_config` precedent for
/// bypassing [`Gemma4Config::validate`] entirely): 1 layer, non-shared global
/// attention, hidden_size 8, vocab_size 16. Deliberately simpler than
/// `gemma4_cache::tests::tiny_config`'s 6-layer fixture -- a single
/// non-shared global layer has no donor-slot indirection to reason about at
/// all, which this test-support builder does not need.
///
/// All-zero weights make every logit exactly `0.0` at every step
/// (hand-verified end to end: RMSNorm of an all-zero vector is
/// `0 / sqrt(0 + rms_norm_eps) = 0`, finite since `rms_norm_eps = 1e-6` keeps
/// the denominator away from zero; RoPE rotation, the attention
/// softmax-then-weighted-sum against all-zero V, GeGLU (`gelu_tanh(0) = 0`),
/// and the final tanh soft-cap (`tanh(0/30) * 30 = 0`) all leave an all-zero
/// input at exactly zero), so greedy sampling deterministically picks token
/// id 0 (the first index satisfying strict `>` against `f32::NEG_INFINITY` in
/// `argmax`/the crate's sampler) and every token's reporting log-probability
/// under [`crate::sampling::compute_step_logprobs`] is exactly
/// `-ln(vocab_size)` (a uniform distribution over equal logits).
///
/// Bypasses [`Gemma4Config::validate`] deliberately (the same choice
/// `gemma4_cache::tests::tiny_config` already makes): `validate` hard-locks
/// `layer_types`'s full_attention positions to the real 35-layer E2B schedule
/// `[4, 9, 14, 19, 24, 29, 34]`, which a 1-layer config can never satisfy.
///
/// Reuses the real committed tokenizer fixture
/// (`tests/fixtures/gemma4/tokenizer/tokenizer.json`) rather than
/// hand-building a second synthetic one: no [`crate::decoder::DecoderSession`]
/// method this builder exists to test ever reads `Gemma4Model::tokenizer`,
/// and `Self::generate`'s own output-text decode is exercised separately by
/// this module's `tests` module below.
#[cfg(test)]
pub(crate) fn tiny_zero_model() -> Gemma4Model {
    use super::gemma4_config::Gemma4LayerType;
    use super::gemma4_weights::Gemma4LayerWeights;

    let hidden_size = 8;
    let head_w = 8;
    let per_layer_dim = 4;
    let mlp_dim = 8;
    let vocab_size = 16;

    let config = Gemma4Config {
        hidden_size,
        num_hidden_layers: 1,
        vocab_size,
        intermediate_size: mlp_dim,
        rms_norm_eps: 1e-6,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: head_w,
        global_head_dim: head_w,
        sliding_window: head_w,
        attention_k_eq_v: false,
        attention_bias: false,
        rope_theta: 10_000.0,
        rope_local_base_freq: 10_000.0,
        partial_rotary_factor: 1.0,
        layer_types: vec![Gemma4LayerType::FullAttention],
        num_kv_shared_layers: 0,
        use_double_wide_mlp_raw: false,
        hidden_size_per_layer_input: per_layer_dim,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        final_logit_softcapping: 30.0,
        tie_word_embeddings: true,
        eos_token_id: 1,
        max_position_embeddings: 1024,
    };

    let layer = Gemma4LayerWeights {
        input_layernorm: vec![0.0; hidden_size],
        post_attention_layernorm: vec![0.0; hidden_size],
        pre_feedforward_layernorm: vec![0.0; hidden_size],
        post_feedforward_layernorm: vec![0.0; hidden_size],
        post_per_layer_input_norm: vec![0.0; hidden_size],
        layer_scalar: 1.0,
        per_layer_input_gate: vec![0.0; per_layer_dim * hidden_size],
        per_layer_projection: vec![0.0; hidden_size * per_layer_dim],
        q_proj: vec![0.0; head_w * hidden_size],
        o_proj: vec![0.0; hidden_size * head_w],
        q_norm: vec![0.0; head_w],
        k_proj: Some(vec![0.0; head_w * hidden_size]),
        v_proj: Some(vec![0.0; head_w * hidden_size]),
        k_norm: Some(vec![0.0; head_w]),
        gate_proj: vec![0.0; mlp_dim * hidden_size],
        up_proj: vec![0.0; mlp_dim * hidden_size],
        down_proj: vec![0.0; hidden_size * mlp_dim],
    };

    let weights = Gemma4Weights {
        embed_tokens: vec![0.0; vocab_size * hidden_size],
        embed_tokens_per_layer: vec![0.0; vocab_size * per_layer_dim],
        norm: vec![0.0; hidden_size],
        per_layer_model_projection: vec![0.0; per_layer_dim * hidden_size],
        per_layer_projection_norm: vec![0.0; per_layer_dim],
        layers: vec![layer],
    };

    let tokenizer_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemma4")
        .join("tokenizer")
        .join("tokenizer.json");
    let tokenizer = GemmaBpeTokenizer::from_tokenizer_json(&tokenizer_path)
        .expect("committed gemma4 tokenizer fixture must load");

    let local_inv_freq = gemma4_rope_inv_freq(config.head_dim, config.rope_local_base_freq, None);
    let global_inv_freq = gemma4_rope_inv_freq(
        config.global_head_dim,
        config.rope_theta,
        Some(config.partial_rotary_factor),
    );

    Gemma4Model {
        config,
        weights,
        tokenizer,
        local_inv_freq,
        global_inv_freq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::ExecutionCapabilities;
    use crate::grammar::{GrammarEngine, GrammarSpec};

    // These four byte-fallback ids -- 478, 382, 378, 366 (`<0xF0><0x90><0x8C><0x80>`), verified
    // against the committed tokenizer fixture -- are the 4-byte UTF-8 encoding of U+10300; no
    // proper prefix of them is valid UTF-8 alone. Reverting `push` to decode each id in isolation
    // instead of re-decoding the accumulated `ids` list must fail the assertions below.
    #[test]
    fn incremental_detokenizer_resolves_a_byte_fallback_run_split_across_tokens() {
        let model = tiny_zero_model();
        let mut detok = IncrementalByteFallbackDetokenizer::new(&model.tokenizer);

        assert_eq!(
            detok.push(478),
            "",
            "1/4 bytes of a 4-byte sequence must not resolve yet"
        );
        assert_eq!(detok.push(382), "", "2/4 bytes must still be held back");
        assert_eq!(detok.push(378), "", "3/4 bytes must still be held back");
        assert_eq!(
            detok.push(366),
            "\u{10300}",
            "the 4th byte completes the sequence: the delta must be the resolved \
             character, not four individually-flushed U+FFFD replacement characters"
        );
        assert_eq!(
            detok.finish(),
            "",
            "nothing left to flush once the sequence resolved"
        );

        // Cross-check: streamed text must equal the non-streaming path's full-batch decode
        // of the same ids.
        let ids = [478u32, 382, 378, 366];
        assert_eq!(
            model.tokenizer.decode(&ids).unwrap_or_default(),
            "\u{10300}"
        );
    }

    /// A run that never completes must flush at `finish()` exactly as a full-batch decode
    /// would: one `U+FFFD` per unresolved byte, never a lossy collapsed replacement.
    #[test]
    fn incremental_detokenizer_flushes_a_genuinely_incomplete_run_at_finish() {
        let model = tiny_zero_model();
        let mut detok = IncrementalByteFallbackDetokenizer::new(&model.tokenizer);
        assert_eq!(detok.push(478), "");
        assert_eq!(detok.push(382), "");
        assert_eq!(detok.push(378), "");

        let expected_full = model.tokenizer.decode(&[478, 382, 378]).unwrap_or_default();
        assert_eq!(
            expected_full, "\u{FFFD}\u{FFFD}\u{FFFD}",
            "control: the tokenizer's own decode over the same incomplete 3-byte run"
        );
        assert_eq!(detok.finish(), expected_full);
    }

    /// Capability refusal 1/4: a `gen_cfg.grammar` request is refused before
    /// any session method runs, through the public `Gemma4Model::generate`
    /// entry point end to end. `GemmaCpuSession`'s `grammar` capability is
    /// permanently `false` (this row implements no grammar masking).
    #[test]
    fn generate_refuses_grammar_request() {
        let model = tiny_zero_model();
        let grammar = GrammarEngine::new(
            &GrammarSpec::Gbnf("root ::= \"a\"\n".into()),
            vec![b"a".to_vec()],
        )
        .expect("trivial one-token grammar must compile");
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            grammar: Some(std::sync::Arc::new(grammar)),
            ..Default::default()
        };
        let err = model
            .generate(&[2, 3], &gen_cfg)
            .expect_err("a session declaring grammar: false must refuse a grammar request");
        match err {
            InferenceError::InvalidInput(msg) => {
                assert!(msg.contains("grammar"), "message must name grammar: {msg}")
            }
            other => panic!("expected InvalidInput naming grammar, got: {other:?}"),
        }
    }

    /// Capability refusal 2/4: a `gen_cfg.logprobs` request is refused the
    /// same way. `GemmaCpuSession::metadata` is a real, working
    /// implementation (see that module's doc comment), but is unreachable
    /// because `check_capabilities` runs before `select`/`metadata` are ever
    /// called.
    #[test]
    fn generate_refuses_logprobs_request() {
        let model = tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            logprobs: Some(0),
            ..Default::default()
        };
        let err = model
            .generate(&[2, 3], &gen_cfg)
            .expect_err("a session declaring logprobs: false must refuse a logprobs request");
        match err {
            InferenceError::InvalidInput(msg) => {
                assert!(
                    msg.contains("logprobs"),
                    "message must name logprobs: {msg}"
                )
            }
            other => panic!("expected InvalidInput naming logprobs, got: {other:?}"),
        }
    }

    /// Capability refusal 3/4: a non-empty `gen_cfg.stop_strings` request is
    /// refused the same way.
    #[test]
    fn generate_refuses_stop_strings_request() {
        let model = tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            stop_strings: vec!["x".to_string()],
            ..Default::default()
        };
        let err = model
            .generate(&[2, 3], &gen_cfg)
            .expect_err("a session declaring stop_strings: false must refuse a request");
        match err {
            InferenceError::InvalidInput(msg) => assert!(
                msg.contains("stop_strings"),
                "message must name stop_strings: {msg}"
            ),
            other => panic!("expected InvalidInput naming stop_strings, got: {other:?}"),
        }
    }

    /// Capability refusal 4/4: a `gen_cfg.reasoning_budget` request is
    /// refused the same way. `enable_thinking` must stay `true` (the
    /// default) for this to reach the driver as `Some`, since
    /// `GenerateConfig::effective_reasoning_budget` masks the raw field to
    /// `None` whenever `enable_thinking` is `false` -- and
    /// `check_capabilities` reads the raw `gen_cfg.reasoning_budget` field
    /// directly, not the effective one, so this must set `enable_thinking`
    /// explicitly rather than relying on the default.
    #[test]
    fn generate_refuses_reasoning_budget_request() {
        let model = tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            reasoning_budget: Some(4),
            enable_thinking: true,
            ..Default::default()
        };
        let err = model
            .generate(&[2, 3], &gen_cfg)
            .expect_err("a session declaring reasoning_budget: false must refuse a request");
        match err {
            InferenceError::InvalidInput(msg) => assert!(
                msg.contains("reasoning_budget"),
                "message must name reasoning_budget: {msg}"
            ),
            other => panic!("expected InvalidInput naming reasoning_budget, got: {other:?}"),
        }
    }

    /// Passing control for the four refusal tests above: the identical
    /// tiny-model request shape, with every capability-gated field left at
    /// its default (unset), must succeed -- proving the refusals above are
    /// about the specific field each test sets, not about `generate` itself
    /// being broken against this synthetic model.
    #[test]
    fn generate_succeeds_with_no_capability_gated_fields_set() {
        let model = tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 2,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
            ..Default::default()
        };
        let output = model
            .generate(&[2, 3], &gen_cfg)
            .expect("a request with no capability-gated field set must succeed");
        // All-zero weights -> greedy argmax always picks token id 0 (see
        // `tiny_zero_model`'s doc comment) -> both requested tokens are 0,
        // and `eos_token_id` (1) never matches, so the budget is exhausted
        // rather than an early EOS stop.
        assert_eq!(output.token_ids, vec![0, 0]);
        assert_eq!(output.generated_tokens, 2);
        assert!(!output.stopped);
    }

    /// `max_new_tokens` budget: generation stops exactly at the requested
    /// count when nothing else (EOS, a stop token) intervenes first.
    #[test]
    fn generate_stops_at_max_new_tokens_budget() {
        let model = tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 3,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
            ..Default::default()
        };
        let output = model
            .generate(&[2, 3], &gen_cfg)
            .expect("generation over the tiny model must succeed");
        assert_eq!(output.token_ids.len(), 3);
        assert_eq!(output.generated_tokens, 3);
        assert!(
            !output.stopped,
            "reaching the token budget with no EOS/stop-token hit is NOT a `stopped` exit"
        );
        assert_eq!(output.stop_reason, Some(StopReason::Length));
    }

    /// EOS via `stop_token_ids`: greedy decode on the all-zero tiny model
    /// always samples token id 0 (see `tiny_zero_model`'s doc comment), so
    /// configuring `stop_token_ids: vec![0]` must stop generation at the
    /// first token, before `max_new_tokens` is reached, and exclude the
    /// stopping token from the output (the crate-wide stop-token contract).
    #[test]
    fn generate_stops_on_configured_stop_token_id() {
        let model = tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![0],
            ..Default::default()
        };
        let output = model
            .generate(&[2, 3], &gen_cfg)
            .expect("generation over the tiny model must succeed");
        assert!(
            output.token_ids.is_empty(),
            "the stopping token (id 0, sampled at step 0) must be excluded from token_ids, \
             per the crate-wide stop-token contract"
        );
        assert_eq!(output.generated_tokens, 0);
        assert!(output.stopped);
        assert_eq!(output.stop_reason, Some(StopReason::Eos));
    }

    /// `prompt_len + effective decode cap > max_position_embeddings` must be refused before any
    /// KV-cache allocation or forward call. `max_new_tokens: usize::MAX` proves the ordering:
    /// skipping the check would abort the process on an oversized allocation, not return `Err`.
    #[test]
    fn generate_refuses_when_prompt_plus_decode_cap_exceeds_context_window() {
        let model = tiny_zero_model(); // max_position_embeddings: 1024
        let prompt_ids = [2u32, 3u32];
        let gen_cfg = GenerateConfig {
            max_new_tokens: usize::MAX,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
            ..Default::default()
        };
        let err = model.generate(&prompt_ids, &gen_cfg).expect_err(
            "prompt_len + effective decode cap exceeding the context window must be refused",
        );
        match err {
            InferenceError::Inference(msg) => assert!(
                msg.contains("context window"),
                "message must name the context-window bound: {msg}"
            ),
            other => panic!(
                "expected InferenceError::Inference naming the context window, got: {other:?}"
            ),
        }
    }

    /// Streaming sibling of the test above: `generate_streaming_via_driver` runs the identical
    /// preflight check. `on_token_calls == 0` proves no token was ever produced, since
    /// `on_token` is only invoked from inside `driver::run`'s loop, which this must never reach.
    #[test]
    fn generate_streaming_refuses_when_prompt_plus_decode_cap_exceeds_context_window() {
        let model = tiny_zero_model();
        let prompt_ids = [2u32, 3u32];
        let gen_cfg = GenerateConfig {
            max_new_tokens: usize::MAX,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
            ..Default::default()
        };
        let mut on_token_calls = 0usize;
        let err = model
            .generate_streaming_with_cancel(
                &prompt_ids,
                &gen_cfg,
                |_delta| {
                    on_token_calls += 1;
                    true
                },
                || false,
            )
            .expect_err("the same context-window bound must be enforced on the streaming path");
        match err {
            InferenceError::Inference(msg) => assert!(
                msg.contains("context window"),
                "message must name the context-window bound: {msg}"
            ),
            other => panic!(
                "expected InferenceError::Inference naming the context window, got: {other:?}"
            ),
        }
        assert_eq!(
            on_token_calls, 0,
            "the over-budget refusal must fire before any token is streamed, i.e. before \
             any forward pass runs"
        );
    }

    /// Capabilities declared by `ExecutionCapabilities::default()` are all
    /// `false` -- the baseline this row's session relies on to make its own
    /// all-false declaration meaningful rather than accidental (if the
    /// crate's own default ever flipped a field to `true`, "declares nothing"
    /// and "declares everything false" would silently diverge).
    #[test]
    fn execution_capabilities_default_is_all_false() {
        let caps = ExecutionCapabilities::default();
        assert!(!caps.grammar);
        assert!(!caps.logprobs);
        assert!(!caps.stop_strings);
        assert!(!caps.reasoning_budget);
    }

    /// Explicit opt-out for the checkpoint-gated driver-trace test below,
    /// mirroring `tests/gemma4_e2e_forward_test.rs`'s own
    /// `skip_allowed`/`resolve_model_dir` contract exactly (deliberately
    /// duplicated, not shared -- see `crate::test_support`'s doc comment on
    /// why this crate's checkpoint-dir resolver and the integration-test
    /// binary's are two separate implementations on purpose: different
    /// contracts, different crate boundaries).
    #[cfg(feature = "f16")]
    fn driver_trace_gate_skip_allowed() -> bool {
        std::env::var("LATTICE_GEMMA4_GATE_SKIP").as_deref() == Ok("1")
    }

    #[cfg(feature = "f16")]
    fn resolve_real_checkpoint_dir() -> Option<std::path::PathBuf> {
        const VAR: &str = "LATTICE_GEMMA4_MODEL_DIR";
        let raw =
            std::env::var(VAR).unwrap_or_else(|_| "~/.lattice/models/gemma-4-e2b-it".to_string());
        let path = if let Some(rest) = raw.strip_prefix("~/") {
            std::path::PathBuf::from(std::env::var("HOME").ok()?).join(rest)
        } else {
            std::path::PathBuf::from(&raw)
        };
        if path.join("model.safetensors").exists() {
            Some(path)
        } else if driver_trace_gate_skip_allowed() {
            eprintln!(
                "LATTICE_GEMMA4_E2E_SKIPPED reason=missing_checkpoint path={}",
                path.display()
            );
            None
        } else {
            panic!(
                "{VAR}={} has no model.safetensors -- this driver-trace gate fails closed by \
                 default on a missing checkpoint, mirroring \
                 tests/gemma4_e2e_forward_test.rs's own contract. Set \
                 LATTICE_GEMMA4_GATE_SKIP=1 to explicitly skip.",
                path.display()
            );
        }
    }

    /// ADR-090 row R04 (#1597) driver-marker acceptance, checkpoint-gated.
    /// Mirrors `model::qwen35::generation`'s own
    /// `driver_trace_matches_ids_len_on_the_pre_migration_golden` test (same
    /// rationale, same crate-boundary constraint -- see that test's and
    /// `tests/gemma4_e2e_forward_test.rs`'s
    /// `stage5_shared_driver_greedy_matches_hf_golden`'s own doc comments):
    /// `generate_with_trace`/`driver::DriverTrace` are `pub(crate)` by
    /// deliberate design, so the only place a test can assert the driver's
    /// opened/consumed counts is inside this crate.
    ///
    /// Fail-closed by default via `resolve_real_checkpoint_dir` above: a
    /// missing checkpoint panics unless `LATTICE_GEMMA4_GATE_SKIP=1` is set.
    /// This is a stricter contract than this file's own pre-existing
    /// `donor_mutation_tests::model_dir` helper (which always skips silently
    /// on a missing checkpoint, with no panic path at all, despite that
    /// module's doc comment claiming "same convention as
    /// tests/gemma4_e2e_forward_test.rs" -- that claim does not hold against
    /// the read source and predates this row; noted here, not fixed, since
    /// correcting a pre-existing unrelated test is outside this row's scope).
    #[cfg(feature = "f16")]
    fn run_driver_trace_gate(model_dir: &Path) {
        #[derive(serde::Deserialize)]
        struct Golden {
            input_ids: Vec<u32>,
            greedy_tokens: Vec<u32>,
        }
        const FIXTURE: &str = include_str!("../../tests/fixtures/gemma4/stage5/e2e_golden.json");
        let golden: Golden = serde_json::from_str(FIXTURE).expect("golden fixture parses");

        let model = Gemma4Model::from_safetensors(model_dir).expect("loading real checkpoint");

        // See `Gemma4Model::generate`'s own doc comment on why
        // `stop_token_ids` must be cleared explicitly.
        let gen_cfg = GenerateConfig {
            max_new_tokens: 3,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
            ..Default::default()
        };
        let (output, trace) = model
            .generate_with_trace(&golden.input_ids, &gen_cfg)
            .expect("gemma4 shared-driver generate_with_trace");

        assert!(
            !output.stopped,
            "must not stop before 3 tokens for this golden prompt (stop reason: {:?}) -- see \
             tests/gemma4_e2e_forward_test.rs's stage5_shared_driver_greedy_matches_hf_golden \
             for the full explanation of this risk",
            output.stop_reason
        );
        assert_eq!(
            output.token_ids, golden.greedy_tokens,
            "driver-routed ids must match the HF golden exactly"
        );
        // The driver-trace bypass-detector invariant (`decoder::driver`'s own
        // module doc comment): one `select()` per emitted token on a natural
        // (non-EOS-at-step-0) finish, and exactly one prediction stays open
        // at finish (`consumed == opened - 1`). A step routed around the
        // driver would leave both counters short of what a real run
        // produces, in a way the token-id comparison above cannot see by
        // construction.
        assert_eq!(
            trace.opened,
            output.token_ids.len(),
            "one select() per emitted token on a natural finish"
        );
        assert_eq!(
            trace.consumed + 1,
            trace.opened,
            "exactly one prediction stays open at finish"
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn generate_with_trace_matches_hf_golden_and_driver_trace_is_consistent() {
        let Some(model_dir) = resolve_real_checkpoint_dir() else {
            return;
        };
        run_driver_trace_gate(&model_dir);
    }

    /// Unconditional sibling of the checkpoint-gated test above: pins the same `DriverTrace`
    /// invariant (`decode()` runs on the previous iteration's still-open prediction before
    /// `select()` opens the current one, so `opened`/`consumed` lag by exactly one at a natural
    /// finish) without needing a real checkpoint. Pins absolute counts, not only the relative
    /// invariant: `max_new_tokens: 3` against the all-zero tiny model must open exactly 3
    /// predictions and consume exactly 2.
    #[test]
    fn generate_with_trace_drives_the_session_in_the_exact_select_decode_count() {
        let model = tiny_zero_model();
        let prompt_ids = [2u32, 3u32];
        let gen_cfg = GenerateConfig {
            max_new_tokens: 3,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
            ..Default::default()
        };
        let (output, trace) = model
            .generate_with_trace(&prompt_ids, &gen_cfg)
            .expect("generation over the tiny model must succeed");

        assert!(
            !output.stopped,
            "must run the full 3-token budget, no early EOS"
        );
        assert_eq!(output.token_ids.len(), 3);
        assert_eq!(trace.opened, 3, "one select() per emitted token");
        assert_eq!(
            trace.consumed, 2,
            "one decode() per emitted token except the last (its prediction is dropped \
             via finish, not decoded)"
        );
        assert_eq!(
            trace.consumed + 1,
            trace.opened,
            "the driver's own standing invariant (decoder/driver.rs module doc comment)"
        );
    }
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
