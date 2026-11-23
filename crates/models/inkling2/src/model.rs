//! The decoder: embedding, stored layers, the looped core and the output head.

use lattice_arch::inkling2_config::{AttentionSpan, Inkling2TextConfig};
use lattice_inference_core::InferenceError;

use crate::attention::{AttentionInputs, AttentionParams, attention};
use crate::cache::Inkling2Cache;
use crate::conv::static_conv;
use crate::mixture::{NormalizedStates, message_gate};
use crate::moe::{MoeParams, moe};
use crate::ops::{rms_norm_weighted_into, softcap};
use crate::tensor::{DenseMatrix, ExpertBank};
use crate::weights::{Inkling2Weights, KvConvolution, LayerWeights, MessageGateWeights};

/// Messages of one loop layer for every token of a call, in the order query, key, value and
/// MLP; each holds `tokens * hidden` values.
type Messages = [Vec<f32>; 4];

/// The Inkling 2 text decoder on the CPU.
#[derive(Debug, Clone)]
pub struct Inkling2Model {
    config: Inkling2TextConfig,
    weights: Inkling2Weights,
}

impl Inkling2Model {
    /// Builds the decoder after checking every weight against the shape `config` gives it.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidInput`] naming the first weight whose shape, count or
    /// kind does not match the configuration.
    pub fn new(
        config: Inkling2TextConfig,
        weights: Inkling2Weights,
    ) -> Result<Self, InferenceError> {
        validate(&config, &weights)?;
        Ok(Self { config, weights })
    }

    /// The configuration the decoder was built with.
    pub fn config(&self) -> &Inkling2TextConfig {
        &self.config
    }

    /// An empty cache for this decoder.
    pub fn new_cache(&self) -> Inkling2Cache {
        Inkling2Cache::new(&self.config)
    }

    /// Runs `tokens` after the tokens already in `cache` and returns the logits of the last
    /// one: `unpadded_vocab_size` values after the final softcap.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidInput`] for an empty input, a token outside the
    /// unpadded vocabulary, or a cache made for another configuration; the cache is unchanged
    /// by such a refusal. An error from an expert kernel is passed on, and the cache then holds
    /// part of the call and should be discarded.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut Inkling2Cache,
    ) -> Result<Vec<f32>, InferenceError> {
        self.run(tokens, cache, false)
    }

    /// As [`Inkling2Model::forward`], returning the logits of every token:
    /// `tokens.len() * unpadded_vocab_size` values.
    ///
    /// # Errors
    ///
    /// As [`Inkling2Model::forward`].
    pub fn forward_all(
        &self,
        tokens: &[u32],
        cache: &mut Inkling2Cache,
    ) -> Result<Vec<f32>, InferenceError> {
        self.run(tokens, cache, true)
    }

    fn run(
        &self,
        tokens: &[u32],
        cache: &mut Inkling2Cache,
        all: bool,
    ) -> Result<Vec<f32>, InferenceError> {
        self.check_run(tokens, cache)?;
        let c = &self.config;
        let lc = c.loop_config;
        let core = lc.core_end - lc.core_start;
        let mut x = self.embed(tokens);
        let mut blocks = Vec::new();
        for layer in 0..lc.core_start {
            x = self.apply_layer(layer, layer, x, &mut blocks, None, cache)?;
        }
        let pre_loop_x = x.clone();
        let pre_loop_blocks = blocks.clone();
        // The first loop layer opens a block, so the first pass appends the pre-loop state;
        // messages come from what each pass appended after it.
        let source_start = pre_loop_blocks.len() + 1;
        for layer in lc.core_start..lc.core_end {
            x = self.apply_layer(layer, layer, x, &mut blocks, None, cache)?;
        }
        for pass in 1..lc.num_passes {
            let source = MessageSource::new(
                c.hidden_size,
                blocks.get(source_start..).unwrap_or_default(),
                &x,
            );
            x = pre_loop_x.clone();
            blocks = pre_loop_blocks.clone();
            for (j, layer) in (lc.core_start..lc.core_end).enumerate() {
                let messages = source.messages(&self.weights.message_queries[4 * j..4 * j + 4]);
                let slot = c.num_hidden_layers + (pass - 1) * core + j;
                x = self.apply_layer(layer, slot, x, &mut blocks, Some(&messages), cache)?;
            }
        }
        for layer in lc.core_end..c.num_hidden_layers {
            x = self.apply_layer(layer, layer, x, &mut blocks, None, cache)?;
        }
        cache.position += tokens.len();
        Ok(self.head(&x, &blocks, all))
    }

    /// Runs the stored layers once each in order, with no repeated passes.
    #[cfg(test)]
    pub(crate) fn forward_straight(
        &self,
        tokens: &[u32],
        cache: &mut Inkling2Cache,
    ) -> Result<Vec<f32>, InferenceError> {
        self.check_run(tokens, cache)?;
        let mut x = self.embed(tokens);
        let mut blocks = Vec::new();
        for layer in 0..self.config.num_hidden_layers {
            x = self.apply_layer(layer, layer, x, &mut blocks, None, cache)?;
        }
        cache.position += tokens.len();
        Ok(self.head(&x, &blocks, true))
    }

    fn check_run(&self, tokens: &[u32], cache: &Inkling2Cache) -> Result<(), InferenceError> {
        let vocab = self.config.unpadded_vocab_size;
        if tokens.is_empty() {
            return Err(invalid("no tokens to run".to_owned()));
        }
        if let Some(token) = tokens.iter().find(|&&token| token as usize >= vocab) {
            return Err(invalid(format!(
                "token {token} is outside the vocabulary of {vocab} tokens"
            )));
        }
        if !cache.fits(&self.config) {
            return Err(invalid(
                "the cache was made for a different configuration".to_owned(),
            ));
        }
        Ok(())
    }

    /// Runs stored layer `layer` with cache slot `slot`, and appends its input to `blocks` when
    /// the layer opens a block.
    fn apply_layer(
        &self,
        layer: usize,
        slot: usize,
        x: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        messages: Option<&Messages>,
        cache: &mut Inkling2Cache,
    ) -> Result<Vec<f32>, InferenceError> {
        let out = self.layer_forward(layer, slot, &x, blocks, messages, cache)?;
        if self.config.starts_attention_residual_block(layer) {
            blocks.push(x);
        }
        Ok(out)
    }

    fn layer_forward(
        &self,
        layer: usize,
        slot: usize,
        x: &[f32],
        blocks: &[Vec<f32>],
        messages: Option<&Messages>,
        cache: &mut Inkling2Cache,
    ) -> Result<Vec<f32>, InferenceError> {
        let c = &self.config;
        let lw = &self.weights.layers[layer];
        let h = c.hidden_size;
        let eps = c.rms_norm_eps as f32;
        let gated = messages.zip(lw.gates.as_ref());

        let mut query_in = vec![0.0f32; x.len()];
        let mut key_in = vec![0.0f32; x.len()];
        let mut value_in = vec![0.0f32; x.len()];
        let mut scratch = vec![0.0f32; h];
        let norm_q = &lw.attention_norms[0];
        let norm_k = lw.attention_norms.get(1).unwrap_or(norm_q);
        let norm_v = lw.attention_norms.get(2).unwrap_or(norm_q);
        for (t, xt) in x.chunks_exact(h).enumerate() {
            let range = t * h..(t + 1) * h;
            let q = &mut query_in[range.clone()];
            let k = &mut key_in[range.clone()];
            let v = &mut value_in[range.clone()];
            match lw.attention_input_queries.as_slice() {
                [] => {
                    q.copy_from_slice(xt);
                    k.copy_from_slice(xt);
                    v.copy_from_slice(xt);
                }
                [shared] => {
                    token_states(h, blocks, t, &[xt]).attend(shared, q);
                    k.copy_from_slice(q);
                    v.copy_from_slice(q);
                }
                queries => {
                    let states = token_states(h, blocks, t, &[xt]);
                    states.attend(&queries[0], q);
                    states.attend(&queries[1], k);
                    states.attend(&queries[2], v);
                }
            }
            if let Some((m, gates)) = gated {
                for (input, gate, message) in [
                    (&mut *q, &gates.query, &m[0]),
                    (&mut *k, &gates.key, &m[1]),
                    (&mut *v, &gates.value, &m[2]),
                ] {
                    message_gate(gate, eps, input, &message[range.clone()], &mut scratch);
                    input.copy_from_slice(&scratch);
                }
            }
            for (input, norm) in [(q, norm_q), (k, norm_k), (v, norm_v)] {
                rms_norm_weighted_into(input, norm, eps, &mut scratch);
                input.copy_from_slice(&scratch);
            }
        }

        let params = self.attention_params(layer);
        let start = cache.position;
        let cache_slot = &mut cache.slots[slot];
        let mut attended = vec![0.0f32; x.len()];
        attention(
            &params,
            &lw.attention,
            &AttentionInputs {
                query: &query_in,
                key: &key_in,
                value: &value_in,
            },
            start,
            cache_slot,
            &mut attended,
        );
        let mut residual = vec![0.0f32; x.len()];
        static_conv(
            &mut cache_slot.attention_history,
            &lw.post_attention_convolution,
            c.conv_kernel_size,
            &attended,
            &mut residual,
        );
        let boundary = c.starts_attention_residual_block(layer);
        if !boundary {
            for (r, value) in residual.iter_mut().zip(x) {
                *r += value;
            }
        }

        let moe_params = self.moe_params();
        let mut expert_out = vec![0.0f32; x.len()];
        let mut mixed = vec![0.0f32; h];
        for (t, (rt, et)) in residual
            .chunks_exact(h)
            .zip(expert_out.chunks_exact_mut(h))
            .enumerate()
        {
            let range = t * h..(t + 1) * h;
            let states = if boundary {
                token_states(h, blocks, t, &[&x[range.clone()], rt])
            } else {
                token_states(h, blocks, t, &[rt])
            };
            states.attend(&lw.mlp_input_query, &mut mixed);
            if let Some((m, gates)) = gated {
                message_gate(&gates.mlp, eps, &mixed, &m[3][range], &mut scratch);
                mixed.copy_from_slice(&scratch);
            }
            rms_norm_weighted_into(&mixed, &lw.mlp_norm, eps, &mut scratch);
            moe(&moe_params, &lw.moe, &scratch, et)?;
        }
        let mut out = vec![0.0f32; x.len()];
        static_conv(
            &mut cache_slot.mlp_history,
            &lw.post_mlp_convolution,
            c.conv_kernel_size,
            &expert_out,
            &mut out,
        );
        for (o, r) in out.iter_mut().zip(&residual) {
            *o += r;
        }
        Ok(out)
    }

    fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        let h = self.config.hidden_size;
        let eps = self.config.rms_norm_eps as f32;
        let mut x = vec![0.0f32; tokens.len() * h];
        let mut row = vec![0.0f32; h];
        for (&token, out) in tokens.iter().zip(x.chunks_exact_mut(h)) {
            self.weights.embedding.row_into(token as usize, &mut row);
            rms_norm_weighted_into(&row, &self.weights.embedding_norm, eps, out);
        }
        x
    }

    /// Logits of the last token, or of every token when `all` is set.
    fn head(&self, x: &[f32], blocks: &[Vec<f32>], all: bool) -> Vec<f32> {
        let c = &self.config;
        let h = c.hidden_size;
        let vocab = c.unpadded_vocab_size;
        let eps = c.rms_norm_eps as f32;
        let cap = c.final_logit_softcapping as f32;
        let tokens = x.len() / h;
        let first = if all { 0 } else { tokens - 1 };
        let mut logits = vec![0.0f32; (tokens - first) * vocab];
        let mut mixed = vec![0.0f32; h];
        let mut normed = vec![0.0f32; h];
        for (out, t) in logits.chunks_exact_mut(vocab).zip(first..tokens) {
            token_states(h, blocks, t, &[&x[t * h..(t + 1) * h]])
                .attend(&self.weights.output_query, &mut mixed);
            rms_norm_weighted_into(&mixed, &self.weights.final_norm, eps, &mut normed);
            self.weights.unembedding.matvec(&normed, out);
            for value in out.iter_mut() {
                *value = softcap(*value, cap);
            }
        }
        logits
    }

    fn attention_params(&self, layer: usize) -> AttentionParams {
        let c = &self.config;
        let span = c.layer_spans[layer];
        let heads = c.heads(span);
        AttentionParams {
            span,
            heads: heads.num_attention_heads,
            kv_heads: heads.num_key_value_heads,
            head_dim: heads.head_dim,
            hidden: c.hidden_size,
            d_rel: c.d_rel,
            extent: c.relative_extent(span),
            kernel: c.conv_kernel_size,
            window: match span {
                AttentionSpan::Global => None,
                AttentionSpan::Sliding => Some(c.sliding_window_size),
            },
            qk_clip: c.qk_l2_clip_threshold as f32,
            rel_clip: c.rel_l2_clip_threshold as f32,
            eps: c.rms_norm_eps as f32,
            log_alpha: c.log_scaling_alpha as f32,
            log_floor: c.log_scaling_n_floor,
        }
    }

    fn moe_params(&self) -> MoeParams {
        let c = &self.config;
        MoeParams {
            top_k: c.num_experts_per_tok,
            route_scale: c.route_scale as f32,
            gate_cap: c.mlp_gate_softcap as f32,
            up_cap: c.mlp_up_softcap as f32,
            intermediate: c.moe_intermediate_size,
        }
    }
}

/// The rows of `blocks` at token `t` followed by `tail`, normalized for a block mixture.
fn token_states(hidden: usize, blocks: &[Vec<f32>], t: usize, tail: &[&[f32]]) -> NormalizedStates {
    let range = t * hidden..(t + 1) * hidden;
    let mut rows: Vec<&[f32]> = blocks.iter().map(|block| &block[range.clone()]).collect();
    rows.extend_from_slice(tail);
    NormalizedStates::new(hidden, &rows)
}

/// The previous pass, normalized once per token, from which each loop layer reads its messages.
struct MessageSource {
    hidden: usize,
    states: Vec<NormalizedStates>,
}

impl MessageSource {
    fn new(hidden: usize, blocks: &[Vec<f32>], x: &[f32]) -> Self {
        let states = x
            .chunks_exact(hidden)
            .enumerate()
            .map(|(t, xt)| token_states(hidden, blocks, t, &[xt]))
            .collect();
        Self { hidden, states }
    }

    fn messages(&self, queries: &[Vec<f32>]) -> Messages {
        let len = self.states.len() * self.hidden;
        let mut messages: Messages = std::array::from_fn(|_| vec![0.0f32; len]);
        for (message, query) in messages.iter_mut().zip(queries) {
            for (out, states) in message.chunks_exact_mut(self.hidden).zip(&self.states) {
                states.attend(query, out);
            }
        }
        messages
    }
}

fn invalid(message: String) -> InferenceError {
    InferenceError::InvalidInput(message)
}

fn check_len(values: &[f32], len: usize, what: &str) -> Result<(), InferenceError> {
    if values.len() != len {
        return Err(invalid(format!(
            "{what}: expected {len} values, got {}",
            values.len()
        )));
    }
    Ok(())
}

fn check_matrix(
    matrix: &DenseMatrix,
    rows: usize,
    cols: usize,
    what: &str,
) -> Result<(), InferenceError> {
    if matrix.rows() != rows || matrix.cols() != cols {
        return Err(invalid(format!(
            "{what}: expected {rows} x {cols}, got {} x {}",
            matrix.rows(),
            matrix.cols()
        )));
    }
    Ok(())
}

fn check_bank(
    bank: &ExpertBank,
    experts: usize,
    rows: usize,
    cols: usize,
    what: &str,
) -> Result<(), InferenceError> {
    if bank.experts() != experts || bank.rows() != rows || bank.cols() != cols {
        return Err(invalid(format!(
            "{what}: expected {experts} experts of {rows} x {cols}, got {} of {} x {}",
            bank.experts(),
            bank.rows(),
            bank.cols()
        )));
    }
    Ok(())
}

fn check_count<T>(items: &[T], count: usize, what: &str) -> Result<(), InferenceError> {
    if items.len() != count {
        return Err(invalid(format!(
            "{what}: expected {count}, got {}",
            items.len()
        )));
    }
    Ok(())
}

fn validate(c: &Inkling2TextConfig, w: &Inkling2Weights) -> Result<(), InferenceError> {
    let h = c.hidden_size;
    let vocab = c.unpadded_vocab_size;
    for (matrix, what) in [(&w.embedding, "embedding"), (&w.unembedding, "unembedding")] {
        if matrix.rows() < vocab || matrix.cols() != h {
            return Err(invalid(format!(
                "{what}: expected at least {vocab} rows of {h}, got {} x {}",
                matrix.rows(),
                matrix.cols()
            )));
        }
    }
    check_len(&w.embedding_norm, h, "embedding norm")?;
    check_len(&w.final_norm, h, "final norm")?;
    check_len(&w.output_query, h, "output query")?;
    check_count(&w.layers, c.num_hidden_layers, "layers")?;
    let lc = c.loop_config;
    check_count(
        &w.message_queries,
        4 * (lc.core_end - lc.core_start),
        "message queries",
    )?;
    for (index, query) in w.message_queries.iter().enumerate() {
        check_len(query, h, &format!("message query {index}"))?;
    }
    for (layer, lw) in w.layers.iter().enumerate() {
        check_layer(c, layer, lw)?;
    }
    Ok(())
}

fn check_layer(
    c: &Inkling2TextConfig,
    layer: usize,
    lw: &LayerWeights,
) -> Result<(), InferenceError> {
    let part = |name: &str| format!("layer {layer} {name}");
    let span = c.layer_spans[layer];
    let global = span == AttentionSpan::Global;
    let heads = c.heads(span);
    let h = c.hidden_size;
    let kernel = c.conv_kernel_size;
    let q = heads.num_attention_heads * heads.head_dim;
    let kv = heads.num_key_value_heads * heads.head_dim;
    let r = heads.num_attention_heads * c.d_rel;

    let queries = if layer == 0 {
        0
    } else if global {
        3
    } else {
        1
    };
    check_count(
        &lw.attention_input_queries,
        queries,
        &part("attention input queries"),
    )?;
    for query in &lw.attention_input_queries {
        check_len(query, h, &part("attention input query"))?;
    }
    check_len(&lw.mlp_input_query, h, &part("MLP input query"))?;
    check_count(
        &lw.attention_norms,
        if global { 3 } else { 1 },
        &part("attention norms"),
    )?;
    for norm in &lw.attention_norms {
        check_len(norm, h, &part("attention norm"))?;
    }
    check_len(&lw.mlp_norm, h, &part("MLP norm"))?;

    let a = &lw.attention;
    check_matrix(&a.query, q, h, &part("query"))?;
    check_len(&a.query_scale, q, &part("query scale"))?;
    check_matrix(&a.key, kv, h, &part("key"))?;
    check_matrix(&a.value, kv, h, &part("value"))?;
    check_matrix(&a.output_gate, q, h, &part("output gate"))?;
    check_matrix(&a.output, h, q, &part("output"))?;
    check_matrix(&a.sink, heads.num_attention_heads, h, &part("sink"))?;
    check_matrix(&a.relative, r, h, &part("relative projection"))?;
    check_len(&a.relative_scale, r, &part("relative scale"))?;
    check_matrix(
        &a.relative_profiles,
        c.d_rel,
        c.relative_extent(span),
        &part("relative profiles"),
    )?;
    for (convolution, name) in [
        (&a.key_convolution, "key convolution"),
        (&a.value_convolution, "value convolution"),
    ] {
        let what = part(name);
        match (convolution, global) {
            (KvConvolution::Static { taps }, false) => check_len(taps, kv * kernel, &what)?,
            (
                KvConvolution::Dynamic {
                    static_taps,
                    projection,
                },
                true,
            ) => {
                check_len(static_taps, kernel * kv, &what)?;
                check_matrix(projection, heads.num_key_value_heads * kernel, h, &what)?;
            }
            (KvConvolution::Static { .. }, true) => {
                return Err(invalid(format!(
                    "{what}: a full-attention layer needs a dynamic convolution"
                )));
            }
            (KvConvolution::Dynamic { .. }, false) => {
                return Err(invalid(format!(
                    "{what}: a sliding-window layer needs a static convolution"
                )));
            }
        }
    }
    check_len(
        &lw.post_attention_convolution,
        h * kernel,
        &part("post-attention convolution"),
    )?;
    check_len(
        &lw.post_mlp_convolution,
        h * kernel,
        &part("post-MLP convolution"),
    )?;

    let experts = c.n_routed_experts;
    let intermediate = c.moe_intermediate_size;
    check_matrix(&lw.moe.router, experts, h, &part("router"))?;
    check_len(&lw.moe.selection_bias, experts, &part("selection bias"))?;
    check_bank(
        &lw.moe.gate_up,
        experts,
        2 * intermediate,
        h,
        &part("gate and up experts"),
    )?;
    check_bank(
        &lw.moe.down,
        experts,
        h,
        intermediate,
        &part("down experts"),
    )?;

    let lc = c.loop_config;
    let looped = (lc.core_start..lc.core_end).contains(&layer);
    match (&lw.gates, looped) {
        (Some(gates), true) => {
            for (gate, name) in [
                (&gates.query, "query message gate"),
                (&gates.key, "key message gate"),
                (&gates.value, "value message gate"),
                (&gates.mlp, "MLP message gate"),
            ] {
                check_gate(gate, h, lc.gate_rank, &part(name))?;
            }
        }
        (None, false) => {}
        (Some(_), false) => {
            return Err(invalid(part(
                "message gates: only the loop layers have them",
            )));
        }
        (None, true) => return Err(invalid(part("message gates: missing on a loop layer"))),
    }
    Ok(())
}

fn check_gate(
    gate: &MessageGateWeights,
    hidden: usize,
    rank: usize,
    what: &str,
) -> Result<(), InferenceError> {
    check_matrix(&gate.down, rank, 2 * hidden, what)?;
    check_matrix(&gate.forget, hidden, rank, what)?;
    check_matrix(&gate.gain, hidden, rank, what)
}
