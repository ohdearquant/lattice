//! Weights of the text decoder, grouped as the forward reads them.

use crate::tensor::{DenseMatrix, ExpertBank};

/// Key or value convolution of one attention layer.
#[derive(Debug, Clone)]
pub enum KvConvolution {
    /// Sliding layers: depthwise taps `[channels, kernel]`. Tap `i` of a channel multiplies the
    /// input `kernel - 1 - i` tokens back.
    Static {
        /// The taps, `[channels, kernel]`.
        taps: Vec<f32>,
    },
    /// Global layers: static taps `[kernel, channels]` indexed by backward distance (row 0
    /// multiplies the current token), plus per-token taps shared within each key/value head,
    /// projected from the projection's input by `projection` `[kv_heads * kernel, hidden]`.
    Dynamic {
        /// The static taps, `[kernel, channels]`.
        static_taps: Vec<f32>,
        /// Projection of the per-token taps, `[kv_heads * kernel, hidden]`; row `h * kernel + lag`
        /// is head `h` at backward distance `lag`.
        projection: DenseMatrix,
    },
}

/// Attention of one layer. Widths: `q = heads * head_dim`, `kv = kv_heads * head_dim` and
/// `r = heads * d_rel`, with the head geometry of the layer's span.
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Query projection, `[q, hidden]`.
    pub query: DenseMatrix,
    /// Per-head query scale applied before the clip, `[q]`.
    pub query_scale: Vec<f32>,
    /// Key projection, `[kv, hidden]`.
    pub key: DenseMatrix,
    /// Value projection, `[kv, hidden]`.
    pub value: DenseMatrix,
    /// Convolution of the projected keys.
    pub key_convolution: KvConvolution,
    /// Convolution of the projected values.
    pub value_convolution: KvConvolution,
    /// Sigmoid gate on the attention output, `[q, hidden]`.
    pub output_gate: DenseMatrix,
    /// Output projection, `[hidden, q]`.
    pub output: DenseMatrix,
    /// Attention sink logit per head, `[heads, hidden]`.
    pub sink: DenseMatrix,
    /// Projection of the relative-position states, `[r, hidden]`.
    pub relative: DenseMatrix,
    /// Per-head scale of the relative-position states applied before the clip, `[r]`.
    pub relative_scale: Vec<f32>,
    /// Bias profiles per backward distance, `[d_rel, extent]`.
    pub relative_profiles: DenseMatrix,
}

/// One message gate: `out = (1 - tanh(forget z)) * h + tanh(gain z) * m` with
/// `z = down [n(h); n(m)]`.
#[derive(Debug, Clone)]
pub struct MessageGateWeights {
    /// `[rank, 2 * hidden]`, hidden-state half first.
    pub down: DenseMatrix,
    /// `[hidden, rank]`.
    pub forget: DenseMatrix,
    /// `[hidden, rank]`.
    pub gain: DenseMatrix,
}

/// The four message gates of a loop layer.
#[derive(Debug, Clone)]
pub struct LoopGates {
    /// Gate on the query input.
    pub query: MessageGateWeights,
    /// Gate on the key input.
    pub key: MessageGateWeights,
    /// Gate on the value input.
    pub value: MessageGateWeights,
    /// Gate on the MLP input.
    pub mlp: MessageGateWeights,
}

/// Router and experts of one layer.
#[derive(Debug, Clone)]
pub struct MoeWeights {
    /// Router, `[experts, hidden]`.
    pub router: DenseMatrix,
    /// Per-expert bias added to the scores only to choose the experts.
    pub selection_bias: Vec<f32>,
    /// Gate and up projections, `[experts, 2 * intermediate, hidden]`, with gate and up rows
    /// interleaved: row `2j` is the gate and row `2j + 1` the up projection of channel `j`.
    pub gate_up: ExpertBank,
    /// Down projection, `[experts, hidden, intermediate]`.
    pub down: ExpertBank,
}

/// One stored decoder layer.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Queries of the attention-input block mixture: none for layer 0, one for a sliding layer,
    /// three (query, key and value inputs) for a global layer. Each has `hidden` elements.
    pub attention_input_queries: Vec<Vec<f32>>,
    /// Query of the MLP-input block mixture.
    pub mlp_input_query: Vec<f32>,
    /// RMS norm weights of the attention inputs: one shared by the three inputs of a sliding
    /// layer, three (query, key and value) for a global layer.
    pub attention_norms: Vec<Vec<f32>>,
    /// RMS norm weight before the experts.
    pub mlp_norm: Vec<f32>,
    /// Attention.
    pub attention: AttentionWeights,
    /// Taps of the convolution after attention, `[hidden, kernel]` as in
    /// [`KvConvolution::Static`].
    pub post_attention_convolution: Vec<f32>,
    /// Taps of the convolution after the experts, `[hidden, kernel]`.
    pub post_mlp_convolution: Vec<f32>,
    /// Router and experts.
    pub moe: MoeWeights,
    /// Message gates, present exactly on the loop layers.
    pub gates: Option<LoopGates>,
}

/// All weights of the text decoder.
#[derive(Debug, Clone)]
pub struct Inkling2Weights {
    /// Token embedding, `[rows, hidden]` with at least `unpadded_vocab_size` rows.
    pub embedding: DenseMatrix,
    /// RMS norm weight applied to the embedding row.
    pub embedding_norm: Vec<f32>,
    /// The stored layers.
    pub layers: Vec<LayerWeights>,
    /// Message queries of the loop layers, four per layer in the order query, key, value and
    /// MLP: entry `4 * j + i` belongs to loop layer `j`. Each has `hidden` elements.
    pub message_queries: Vec<Vec<f32>>,
    /// Query of the final block mixture.
    pub output_query: Vec<f32>,
    /// RMS norm weight before the unembedding.
    pub final_norm: Vec<f32>,
    /// Unembedding, `[rows, hidden]`; the first `unpadded_vocab_size` rows produce logits.
    pub unembedding: DenseMatrix,
}
