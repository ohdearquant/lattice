//! Bounded Gemma prompt chunks with scalar, per-query attention reductions.

use super::*;
use std::ops::Range;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PrefillLayout {
    rows: usize,
    hidden_len: usize,
    q_len: usize,
    kv_len: usize,
    mlp_len: usize,
    gate_len: usize,
    ple_len: usize,
    local_rope_len: usize,
    global_rope_len: usize,
    scores_len: usize,
    v_norm_len: usize,
    logits_len: usize,
}

fn invalid(message: impl Into<String>) -> InferenceError {
    InferenceError::InvalidInput(format!("gemma4 prefill: {}", message.into()))
}

fn checked_len(length: usize, name: &str) -> Result<usize, InferenceError> {
    if length > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(invalid(format!(
            "{name} exceeds the f32 addressable byte limit"
        )));
    }
    Ok(length)
}

fn checked_product(a: usize, b: usize, name: &str) -> Result<usize, InferenceError> {
    let length = a
        .checked_mul(b)
        .ok_or_else(|| invalid(format!("{name} overflows usize")))?;
    checked_len(length, name)
}

impl PrefillLayout {
    fn new(cfg: &Gemma4Config, rows: usize, max_context: usize) -> Result<Self, InferenceError> {
        for (name, value) in [
            ("rows", rows),
            ("hidden_size", cfg.hidden_size),
            ("vocab_size", cfg.vocab_size),
            ("num_hidden_layers", cfg.num_hidden_layers),
            ("intermediate_size", cfg.intermediate_size),
            ("num_attention_heads", cfg.num_attention_heads),
            ("num_key_value_heads", cfg.num_key_value_heads),
            (
                "hidden_size_per_layer_input",
                cfg.hidden_size_per_layer_input,
            ),
        ] {
            if value == 0 {
                return Err(invalid(format!("{name} must be positive")));
            }
        }
        if max_context < rows {
            return Err(invalid("max_context must cover the active chunk"));
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(invalid("query heads must be divisible by KV heads"));
        }
        for (name, width) in [
            ("head_dim", cfg.head_dim),
            ("global_head_dim", cfg.global_head_dim),
        ] {
            if width == 0 || !width.is_multiple_of(2) {
                return Err(invalid(format!("{name} must be positive and even")));
            }
        }
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            return Err(invalid("layer_types length must match num_hidden_layers"));
        }
        if cfg.num_kv_shared_layers > cfg.num_hidden_layers {
            return Err(invalid("shared layer count exceeds the layer count"));
        }
        if cfg.sliding_window == 0
            && (0..cfg.num_hidden_layers).any(|layer| !cfg.is_global_layer(layer))
        {
            return Err(invalid(
                "sliding_window must be positive for sliding layers",
            ));
        }

        let head = cfg.head_dim.max(cfg.global_head_dim);
        let q = checked_product(cfg.num_attention_heads, head, "query width")?;
        let kv = checked_product(cfg.num_key_value_heads, head, "KV width")?;
        let intermediate = if cfg.num_kv_shared_layers > 0 {
            checked_product(cfg.intermediate_size, 2, "shared FFN width")?
        } else {
            cfg.intermediate_size
        };
        let ple = checked_product(
            cfg.num_hidden_layers,
            cfg.hidden_size_per_layer_input,
            "packed PLE width",
        )?;
        for (a, b, name) in [
            (cfg.vocab_size, cfg.hidden_size, "embedding weights"),
            (ple, cfg.hidden_size, "PLE context weights"),
            (q, cfg.hidden_size, "query/output weights"),
            (kv, cfg.hidden_size, "key/value weights"),
            (intermediate, cfg.hidden_size, "FFN weights"),
            (
                cfg.hidden_size_per_layer_input,
                cfg.hidden_size,
                "PLE layer weights",
            ),
            (max_context, kv, "published KV footprint"),
        ] {
            checked_product(a, b, name)?;
        }

        Ok(Self {
            rows,
            hidden_len: checked_product(rows, cfg.hidden_size, "hidden scratch")?,
            q_len: checked_product(rows, q, "query scratch")?,
            kv_len: checked_product(rows, kv, "KV scratch")?,
            mlp_len: checked_product(rows, intermediate, "FFN scratch")?,
            gate_len: checked_product(rows, cfg.hidden_size_per_layer_input, "PLE gate scratch")?,
            ple_len: checked_product(rows, ple, "PLE scratch")?,
            local_rope_len: checked_product(rows, cfg.head_dim, "local RoPE scratch")?,
            global_rope_len: checked_product(rows, cfg.global_head_dim, "global RoPE scratch")?,
            scores_len: checked_len(max_context, "score scratch")?,
            v_norm_len: checked_len(head, "V normalization scratch")?,
            logits_len: checked_len(cfg.vocab_size, "logit scratch")?,
        })
    }
}

#[derive(Default)]
pub(super) struct Gemma4PrefillScratch {
    layout: PrefillLayout,
    hidden: Vec<f32>,
    residual: Vec<f32>,
    normed: Vec<f32>,
    branch_out: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    context: Vec<f32>,
    mlp_gate: Vec<f32>,
    mlp_up: Vec<f32>,
    ple_gate: Vec<f32>,
    ple_inputs: Vec<f32>,
    ple_context: Vec<f32>,
    cos_local: Vec<f32>,
    sin_local: Vec<f32>,
    cos_global: Vec<f32>,
    sin_global: Vec<f32>,
    scores: Vec<f32>,
    v_norm_ones: Vec<f32>,
    final_logits: Vec<f32>,
}

fn grow(values: &mut Vec<f32>, length: usize) {
    if values.len() < length {
        values.resize(length, 0.0);
    }
}

impl Gemma4PrefillScratch {
    pub(super) fn prepare(
        &mut self,
        cfg: &Gemma4Config,
        rows: usize,
        max_context: usize,
    ) -> Result<(), InferenceError> {
        let layout = PrefillLayout::new(cfg, rows, max_context)?;
        grow(&mut self.hidden, layout.hidden_len);
        grow(&mut self.residual, layout.hidden_len);
        grow(&mut self.normed, layout.hidden_len);
        grow(&mut self.branch_out, layout.hidden_len);
        grow(&mut self.q, layout.q_len);
        grow(&mut self.k, layout.kv_len);
        grow(&mut self.v, layout.kv_len);
        grow(&mut self.context, layout.q_len);
        grow(&mut self.mlp_gate, layout.mlp_len);
        grow(&mut self.mlp_up, layout.mlp_len);
        grow(&mut self.ple_gate, layout.gate_len);
        grow(&mut self.ple_inputs, layout.ple_len);
        grow(&mut self.ple_context, layout.ple_len);
        grow(&mut self.cos_local, layout.local_rope_len);
        grow(&mut self.sin_local, layout.local_rope_len);
        grow(&mut self.cos_global, layout.global_rope_len);
        grow(&mut self.sin_global, layout.global_rope_len);
        grow(&mut self.scores, layout.scores_len);
        grow(&mut self.v_norm_ones, layout.v_norm_len);
        grow(&mut self.final_logits, layout.logits_len);
        self.v_norm_ones[..layout.v_norm_len].fill(1.0);
        self.layout = layout;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Projection {
    PerLayerContext,
    Query,
    Key,
    Value,
    AttentionOutput,
    PerLayerGate,
    PerLayerOutput,
    FinalVocabulary,
}

pub(super) trait WorkObserver {
    fn projection(
        &mut self,
        _stage: Projection,
        _layer: Option<usize>,
        _rows: usize,
        _input: usize,
        _output: usize,
    ) {
    }
    fn ffn(&mut self, _layer: usize, _rows: usize, _intermediate: usize) {}
    fn append(&mut self, _layer: usize, _rows: usize) {}
    fn layer_output(&mut self, _layer: usize, _last_hidden: &[f32]) {}
}

struct Noop;
impl WorkObserver for Noop {}

fn project<O: WorkObserver>(
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    rows: usize,
    input_width: usize,
    output_width: usize,
    stage: Projection,
    layer: Option<usize>,
    observer: &mut O,
) {
    matmul_bt(input, weights, output, rows, input_width, output_width);
    observer.projection(stage, layer, rows, input_width, output_width);
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), InferenceError> {
    if actual != expected {
        return Err(InferenceError::ShapeMismatch {
            name: format!("gemma4 prefill {name}"),
            expected: vec![expected],
            actual: vec![actual],
        });
    }
    Ok(())
}

fn validate_model(
    model: &Gemma4Model,
    cache: &Gemma4KvCache,
    output: &Gemma4Scratch,
    position: usize,
) -> Result<(), InferenceError> {
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let ple = cfg.hidden_size_per_layer_input;
    let packed = cfg.num_hidden_layers * ple;
    require_len(
        "layer count",
        model.weights.layers.len(),
        cfg.num_hidden_layers,
    )?;
    require_len(
        "embedding",
        model.weights.embed_tokens.len(),
        cfg.vocab_size * hidden,
    )?;
    require_len("final norm", model.weights.norm.len(), hidden)?;
    require_len(
        "PLE context",
        model.weights.per_layer_model_projection.len(),
        packed * hidden,
    )?;
    require_len(
        "PLE context norm",
        model.weights.per_layer_projection_norm.len(),
        ple,
    )?;
    require_len(
        "local RoPE frequencies",
        model.local_inv_freq.len(),
        cfg.head_dim / 2,
    )?;
    require_len(
        "global RoPE frequencies",
        model.global_inv_freq.len(),
        cfg.global_head_dim / 2,
    )?;
    if output.hidden.len() < hidden || output.logits.len() < cfg.vocab_size {
        return Err(invalid("single-token output scratch is too short"));
    }
    for (layer, weights) in model.weights.layers.iter().enumerate() {
        let head = cfg.attn_head_dim(layer);
        let q = cfg.num_attention_heads * head;
        let kv = cfg.num_key_value_heads * head;
        let intermediate = cfg.mlp_intermediate_size(layer);
        for (name, actual, expected) in [
            ("input norm", weights.input_layernorm.len(), hidden),
            (
                "post-attention norm",
                weights.post_attention_layernorm.len(),
                hidden,
            ),
            (
                "pre-FFN norm",
                weights.pre_feedforward_layernorm.len(),
                hidden,
            ),
            (
                "post-FFN norm",
                weights.post_feedforward_layernorm.len(),
                hidden,
            ),
            (
                "post-PLE norm",
                weights.post_per_layer_input_norm.len(),
                hidden,
            ),
            ("Q projection", weights.q_proj.len(), q * hidden),
            ("O projection", weights.o_proj.len(), hidden * q),
            ("Q norm", weights.q_norm.len(), head),
            ("FFN gate", weights.gate_proj.len(), intermediate * hidden),
            ("FFN up", weights.up_proj.len(), intermediate * hidden),
            ("FFN down", weights.down_proj.len(), hidden * intermediate),
            ("PLE gate", weights.per_layer_input_gate.len(), ple * hidden),
            (
                "PLE output",
                weights.per_layer_projection.len(),
                hidden * ple,
            ),
        ] {
            require_len(name, actual, expected)?;
        }
        if !cfg.is_kv_shared_layer(layer) {
            let k = weights.k_proj.as_ref().ok_or_else(|| {
                InferenceError::Inference(format!(
                    "gemma4 forward: layer {layer} is non-shared but has no k_proj weights"
                ))
            })?;
            let v = weights.v_proj.as_ref().ok_or_else(|| {
                InferenceError::Inference(format!(
                    "gemma4 forward: layer {layer} is non-shared but has no v_proj weights"
                ))
            })?;
            let norm = weights.k_norm.as_ref().ok_or_else(|| {
                InferenceError::Inference(format!(
                    "gemma4 forward: layer {layer} is non-shared but has no k_norm weights"
                ))
            })?;
            require_len("K projection", k.len(), kv * hidden)?;
            require_len("V projection", v.len(), kv * hidden)?;
            require_len("K norm", norm.len(), head)?;
        }
        if cache.is_kv_shared_layer(layer)? != cfg.is_kv_shared_layer(layer) {
            return Err(invalid("cache sharing does not match the model"));
        }
        let length = cache.seq_len(layer)?;
        let minimum = if cfg.is_global_layer(layer) {
            position
        } else {
            position.min(cfg.sliding_window - 1)
        };
        if length < minimum || length > position {
            return Err(invalid("cache prefix does not match the absolute position"));
        }
        require_len("cached K", cache.k_view(layer)?.len(), length * kv)?;
        require_len("cached V", cache.v_view(layer)?.len(), length * kv)?;
    }
    Ok(())
}

fn rope_tables(inv_freq: &[f32], position: usize, rows: usize, cos: &mut [f32], sin: &mut [f32]) {
    let half = inv_freq.len();
    let width = half * 2;
    for row in 0..rows {
        let pos = (position + row) as u32;
        for i in 0..half {
            let angle = pos as f32 * inv_freq[i];
            let (s, c) = angle.sin_cos();
            cos[row * width + i] = c;
            cos[row * width + half + i] = c;
            sin[row * width + i] = s;
            sin[row * width + half + i] = s;
        }
    }
}

fn attention_row(
    query: &[f32],
    k: &[f32],
    v: &[f32],
    kv_dim: usize,
    kv_head_offset: usize,
    visible: Range<usize>,
    scores: &mut [f32],
    context: &mut [f32],
) {
    for (index, token) in visible.clone().enumerate() {
        let offset = token * kv_dim + kv_head_offset;
        let mut dot = 0.0f32;
        for d in 0..query.len() {
            dot += query[d] * k[offset + d];
        }
        scores[index] = dot;
    }
    softmax_row_fail_closed(scores);
    for d in 0..query.len() {
        let mut sum = 0.0f32;
        for (index, token) in visible.clone().enumerate() {
            sum += scores[index] * v[token * kv_dim + kv_head_offset + d];
        }
        context[d] = sum;
    }
}

pub(super) fn forward_chunk(
    model: &Gemma4Model,
    token_ids: &[u32],
    position: usize,
    cache: &mut Gemma4KvCache,
    scratch: &mut Gemma4PrefillScratch,
    output: &mut Gemma4Scratch,
    emit_logits: bool,
    cancel: &dyn Cancellation,
) -> Result<bool, InferenceError> {
    forward_chunk_with_observer(
        model,
        token_ids,
        position,
        cache,
        scratch,
        output,
        emit_logits,
        cancel,
        &mut Noop,
    )
}

pub(super) fn forward_chunk_with_observer<O: WorkObserver>(
    model: &Gemma4Model,
    token_ids: &[u32],
    position: usize,
    cache: &mut Gemma4KvCache,
    scratch: &mut Gemma4PrefillScratch,
    output: &mut Gemma4Scratch,
    emit_logits: bool,
    cancel: &dyn Cancellation,
    observer: &mut O,
) -> Result<bool, InferenceError> {
    if cancel.is_cancelled() {
        return Ok(false);
    }
    let cfg = &model.config;
    let rows = token_ids.len();
    let end_position = position
        .checked_add(rows)
        .ok_or_else(|| invalid("absolute position overflow"))?;
    PrefillLayout::new(cfg, rows, end_position)?;
    if end_position - 1 > u32::MAX as usize {
        return Err(invalid("absolute position exceeds the RoPE u32 range"));
    }
    for &token_id in token_ids {
        if token_id as usize >= cfg.vocab_size {
            return Err(InferenceError::InvalidInput(format!(
                "gemma4 forward: token_id {token_id} out of range (vocab_size={})",
                cfg.vocab_size
            )));
        }
    }
    validate_model(model, cache, output, position)?;
    scratch.prepare(cfg, rows, end_position)?;
    let layout = scratch.layout;
    let rows = layout.rows;
    let hidden = cfg.hidden_size;
    let ple = cfg.hidden_size_per_layer_input;
    let packed = cfg.num_hidden_layers * ple;

    gemma4_scaled_embedding(
        token_ids,
        &model.weights.embed_tokens,
        hidden,
        &mut scratch.hidden[..layout.hidden_len],
    );
    let id_scale = (ple as f32).sqrt();
    for (row, &token_id) in token_ids.iter().enumerate() {
        model.per_layer_embeddings.scaled_row_into(
            token_id as usize,
            id_scale,
            &mut scratch.ple_inputs[row * packed..(row + 1) * packed],
        )?;
    }
    project(
        &scratch.hidden[..layout.hidden_len],
        &model.weights.per_layer_model_projection,
        &mut scratch.ple_context[..layout.ple_len],
        rows,
        hidden,
        packed,
        Projection::PerLayerContext,
        None,
        observer,
    );
    let context_scale = 1.0 / (hidden as f32).sqrt();
    for value in &mut scratch.ple_context[..layout.ple_len] {
        *value *= context_scale;
    }
    for row in scratch.ple_context[..layout.ple_len].chunks_exact_mut(ple) {
        gemma4_rms_norm(
            row,
            &model.weights.per_layer_projection_norm,
            ple,
            cfg.rms_norm_eps,
        );
    }
    for i in 0..layout.ple_len {
        scratch.ple_inputs[i] =
            (scratch.ple_context[i] + scratch.ple_inputs[i]) * std::f32::consts::FRAC_1_SQRT_2;
    }
    rope_tables(
        &model.local_inv_freq,
        position,
        rows,
        &mut scratch.cos_local[..layout.local_rope_len],
        &mut scratch.sin_local[..layout.local_rope_len],
    );
    rope_tables(
        &model.global_inv_freq,
        position,
        rows,
        &mut scratch.cos_global[..layout.global_rope_len],
        &mut scratch.sin_global[..layout.global_rope_len],
    );

    for layer in 0..cfg.num_hidden_layers {
        if cancel.is_cancelled() {
            return Ok(false);
        }
        let weights = &model.weights.layers[layer];
        let head = cfg.attn_head_dim(layer);
        let q_dim = cfg.num_attention_heads * head;
        let kv_dim = cfg.num_key_value_heads * head;
        let q_len = rows * q_dim;
        let kv_len = rows * kv_dim;
        let (cos, sin) = if cfg.is_global_layer(layer) {
            (
                &scratch.cos_global[..layout.global_rope_len],
                &scratch.sin_global[..layout.global_rope_len],
            )
        } else {
            (
                &scratch.cos_local[..layout.local_rope_len],
                &scratch.sin_local[..layout.local_rope_len],
            )
        };
        scratch.residual[..layout.hidden_len].copy_from_slice(&scratch.hidden[..layout.hidden_len]);
        scratch.normed[..layout.hidden_len].copy_from_slice(&scratch.hidden[..layout.hidden_len]);
        for row in scratch.normed[..layout.hidden_len].chunks_exact_mut(hidden) {
            gemma4_rms_norm(row, &weights.input_layernorm, hidden, cfg.rms_norm_eps);
        }
        project(
            &scratch.normed[..layout.hidden_len],
            &weights.q_proj,
            &mut scratch.q[..q_len],
            rows,
            hidden,
            q_dim,
            Projection::Query,
            Some(layer),
            observer,
        );
        for row in scratch.q[..q_len].chunks_exact_mut(head) {
            gemma4_rms_norm(row, &weights.q_norm, head, cfg.rms_norm_eps);
        }
        gemma4_apply_rope(
            &mut scratch.q[..q_len],
            cos,
            sin,
            rows,
            cfg.num_attention_heads,
            head,
        );

        if !cfg.is_kv_shared_layer(layer) {
            let k_proj = weights.k_proj.as_ref().ok_or_else(|| {
                InferenceError::Inference(format!(
                    "gemma4 forward: layer {layer} is non-shared but has no k_proj weights"
                ))
            })?;
            let v_proj = weights.v_proj.as_ref().ok_or_else(|| {
                InferenceError::Inference(format!(
                    "gemma4 forward: layer {layer} is non-shared but has no v_proj weights"
                ))
            })?;
            let k_norm = weights.k_norm.as_ref().ok_or_else(|| {
                InferenceError::Inference(format!(
                    "gemma4 forward: layer {layer} is non-shared but has no k_norm weights"
                ))
            })?;
            project(
                &scratch.normed[..layout.hidden_len],
                k_proj,
                &mut scratch.k[..kv_len],
                rows,
                hidden,
                kv_dim,
                Projection::Key,
                Some(layer),
                observer,
            );
            project(
                &scratch.normed[..layout.hidden_len],
                v_proj,
                &mut scratch.v[..kv_len],
                rows,
                hidden,
                kv_dim,
                Projection::Value,
                Some(layer),
                observer,
            );
            for row in scratch.k[..kv_len].chunks_exact_mut(head) {
                gemma4_rms_norm(row, k_norm, head, cfg.rms_norm_eps);
            }
            gemma4_apply_rope(
                &mut scratch.k[..kv_len],
                cos,
                sin,
                rows,
                cfg.num_key_value_heads,
                head,
            );
            for row in scratch.v[..kv_len].chunks_exact_mut(head) {
                rms_norm(row, &scratch.v_norm_ones[..head], head, cfg.rms_norm_eps);
            }
            cache.append_kv(layer, &scratch.k[..kv_len], &scratch.v[..kv_len])?;
            observer.append(layer, rows);
        }

        let published = cache.seq_len(layer)?;
        let prefix = published
            .checked_sub(rows)
            .ok_or_else(|| invalid("donor cache has not published the complete chunk"))?;
        let expected_prefix = if cfg.is_global_layer(layer) {
            position
        } else {
            position.min(cfg.sliding_window - 1)
        };
        if prefix != expected_prefix {
            return Err(invalid(
                "published donor prefix does not match the absolute position",
            ));
        }
        let k_view = cache.k_view(layer)?;
        let v_view = cache.v_view(layer)?;
        let published_len = checked_product(published, kv_dim, "published donor KV")?;
        require_len("published donor K", k_view.len(), published_len)?;
        require_len("published donor V", v_view.len(), published_len)?;
        let groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        for row in 0..rows {
            if cancel.is_cancelled() {
                return Ok(false);
            }
            let end = prefix + row + 1;
            let begin = if cfg.is_global_layer(layer) {
                0
            } else {
                end.saturating_sub(cfg.sliding_window)
            };
            for qh in 0..cfg.num_attention_heads {
                let offset = row * q_dim + qh * head;
                attention_row(
                    &scratch.q[offset..offset + head],
                    k_view,
                    v_view,
                    kv_dim,
                    (qh / groups) * head,
                    begin..end,
                    &mut scratch.scores[..end - begin],
                    &mut scratch.context[offset..offset + head],
                );
            }
        }

        project(
            &scratch.context[..q_len],
            &weights.o_proj,
            &mut scratch.branch_out[..layout.hidden_len],
            rows,
            q_dim,
            hidden,
            Projection::AttentionOutput,
            Some(layer),
            observer,
        );
        for row in scratch.branch_out[..layout.hidden_len].chunks_exact_mut(hidden) {
            gemma4_rms_norm(
                row,
                &weights.post_attention_layernorm,
                hidden,
                cfg.rms_norm_eps,
            );
        }
        for i in 0..layout.hidden_len {
            scratch.hidden[i] = scratch.residual[i] + scratch.branch_out[i];
        }

        scratch.residual[..layout.hidden_len].copy_from_slice(&scratch.hidden[..layout.hidden_len]);
        scratch.normed[..layout.hidden_len].copy_from_slice(&scratch.hidden[..layout.hidden_len]);
        for row in scratch.normed[..layout.hidden_len].chunks_exact_mut(hidden) {
            gemma4_rms_norm(
                row,
                &weights.pre_feedforward_layernorm,
                hidden,
                cfg.rms_norm_eps,
            );
        }
        let intermediate = cfg.mlp_intermediate_size(layer);
        gemma4_geglu_mlp(
            &scratch.normed[..layout.hidden_len],
            &weights.gate_proj,
            &weights.up_proj,
            &weights.down_proj,
            rows,
            hidden,
            intermediate,
            &mut scratch.mlp_gate[..rows * intermediate],
            &mut scratch.mlp_up[..rows * intermediate],
            &mut scratch.branch_out[..layout.hidden_len],
        );
        observer.ffn(layer, rows, intermediate);
        for row in scratch.branch_out[..layout.hidden_len].chunks_exact_mut(hidden) {
            gemma4_rms_norm(
                row,
                &weights.post_feedforward_layernorm,
                hidden,
                cfg.rms_norm_eps,
            );
        }
        for i in 0..layout.hidden_len {
            scratch.hidden[i] = scratch.residual[i] + scratch.branch_out[i];
        }

        scratch.residual[..layout.hidden_len].copy_from_slice(&scratch.hidden[..layout.hidden_len]);
        project(
            &scratch.hidden[..layout.hidden_len],
            &weights.per_layer_input_gate,
            &mut scratch.ple_gate[..layout.gate_len],
            rows,
            hidden,
            ple,
            Projection::PerLayerGate,
            Some(layer),
            observer,
        );
        gemma4_gelu_tanh(&mut scratch.ple_gate[..layout.gate_len]);
        for row in 0..rows {
            let input_start = (row * cfg.num_hidden_layers + layer) * ple;
            elementwise_mul(
                &mut scratch.ple_gate[row * ple..(row + 1) * ple],
                &scratch.ple_inputs[input_start..input_start + ple],
            );
        }
        project(
            &scratch.ple_gate[..layout.gate_len],
            &weights.per_layer_projection,
            &mut scratch.branch_out[..layout.hidden_len],
            rows,
            ple,
            hidden,
            Projection::PerLayerOutput,
            Some(layer),
            observer,
        );
        for row in scratch.branch_out[..layout.hidden_len].chunks_exact_mut(hidden) {
            gemma4_rms_norm(
                row,
                &weights.post_per_layer_input_norm,
                hidden,
                cfg.rms_norm_eps,
            );
        }
        for i in 0..layout.hidden_len {
            scratch.hidden[i] = scratch.residual[i] + scratch.branch_out[i];
        }
        for value in &mut scratch.hidden[..layout.hidden_len] {
            *value *= weights.layer_scalar;
        }
        observer.layer_output(
            layer,
            &scratch.hidden[(rows - 1) * hidden..layout.hidden_len],
        );
    }

    if cancel.is_cancelled() {
        return Ok(false);
    }
    let last = (rows - 1) * hidden;
    if emit_logits {
        gemma4_rms_norm(
            &mut scratch.hidden[last..layout.hidden_len],
            &model.weights.norm,
            hidden,
            cfg.rms_norm_eps,
        );
        project(
            &scratch.hidden[last..layout.hidden_len],
            &model.weights.embed_tokens,
            &mut scratch.final_logits[..cfg.vocab_size],
            1,
            hidden,
            cfg.vocab_size,
            Projection::FinalVocabulary,
            None,
            observer,
        );
        gemma4_logit_softcap(
            &mut scratch.final_logits[..cfg.vocab_size],
            cfg.final_logit_softcapping,
        );
        if cancel.is_cancelled() {
            return Ok(false);
        }
    }
    output.hidden[..hidden].copy_from_slice(&scratch.hidden[last..layout.hidden_len]);
    if emit_logits {
        output.logits[..cfg.vocab_size].copy_from_slice(&scratch.final_logits[..cfg.vocab_size]);
    }
    Ok(true)
}

#[cfg(test)]
#[path = "gemma4_prefill_tests.rs"]
mod tests;
