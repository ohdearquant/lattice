use crate::attention::AttentionBuffers;
use crate::forward::cpu::{add_bias, matmul_bt};
use crate::weights::TransformerLayerWeights;
use core::cmp::min;
use core::mem::size_of;

/// L1 data cache budget for tile working set.
///
/// Conservative estimate: 48 KiB. Modern CPUs have at least this much L1d
/// (Apple Silicon: 64 KiB, Intel Ice Lake+: 48 KiB, AMD Zen: 32 KiB).
/// We leave headroom for non-tile data (stack, loop variables, etc.) by
/// targeting 75% of the minimum common L1d size (64 KiB * 0.75 = 48 KiB).
const L1_BUDGET_BYTES: usize = 48 * 1024;
const MASK_VALUE: f32 = -10_000.0;

/// Threshold below which the full materialized scores matrix fits in L1 cache,
/// making tiling overhead not worthwhile. At 32KB L1, sqrt(32768/4) ~ 90 tokens.
const TILED_SEQ_THRESHOLD: usize = 96;

/// **Unstable**: CPU tiling configuration for memory-efficient attention; parameters under active tuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiledAttentionConfig {
    /// Number of query rows per tile.
    pub tile_size_q: usize,
    /// Number of key/value rows per tile.
    pub tile_size_kv: usize,
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of key/value heads.
    pub num_kv_heads: usize,
    /// Per-head dimensionality.
    pub head_dim: usize,
}

impl TiledAttentionConfig {
    /// **Unstable**: construct tiling config with default cache-optimal tile sizes.
    pub fn new(num_q_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let (tile_size_q, tile_size_kv) = Self::optimal_tile_sizes(head_dim);
        Self {
            tile_size_q,
            tile_size_kv,
            num_q_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// **Unstable**: cache-optimal tile size heuristic; may change as hardware targets expand.
    ///
    /// Choose cache-friendly query / KV tile sizes.
    ///
    /// Working set model (conservative, counts all live tiles):
    ///   `2 * Br * D + 2 * Bc * D + Br * Bc` floats
    /// where the `2×` factors account for Q+output tiles (Br side) and K+V
    /// tiles (Bc side), plus the scores tile `Br * Bc`.
    ///
    /// **Strategy**: Maximise `Br * Bc` (tile area) to minimise the total
    /// number of tile operations (`ceil(S/Br) * ceil(S/Bc)`), which directly
    /// determines the number of matmul_bt calls. Among configurations with
    /// equal area, prefer larger `Br` (tile_q) over larger `Bc` (tile_kv)
    /// because the Q-tile output accumulator stays resident while KV tiles
    /// stream through, reducing online-softmax rescaling overhead.
    pub fn optimal_tile_sizes(head_dim: usize) -> (usize, usize) {
        if head_dim == 0 {
            return (1, 1);
        }

        let budget_floats = L1_BUDGET_BYTES / size_of::<f32>();
        let candidates = [64usize, 48, 32, 24, 16, 12, 8, 4, 2, 1];
        let mut best = (1usize, 1usize);
        let mut best_area = 1usize;

        for &br in &candidates {
            for &bc in &candidates {
                let working_set = 2 * br * head_dim + 2 * bc * head_dim + br * bc;
                if working_set > budget_floats {
                    continue;
                }
                let area = br * bc;
                if area > best_area || (area == best_area && br > best.0) {
                    best = (br, bc);
                    best_area = area;
                }
            }
        }

        best
    }

    /// **Unstable**: assert config invariants; panics on violation.
    pub fn validate(&self) {
        assert!(self.tile_size_q > 0, "tile_size_q must be > 0");
        assert!(self.tile_size_kv > 0, "tile_size_kv must be > 0");
        assert!(self.head_dim > 0, "head_dim must be > 0");
        assert!(self.num_q_heads > 0, "num_q_heads must be > 0");
        assert!(self.num_kv_heads > 0, "num_kv_heads must be > 0");
        assert!(
            self.num_q_heads % self.num_kv_heads == 0,
            "num_q_heads ({}) must be divisible by num_kv_heads ({})",
            self.num_q_heads,
            self.num_kv_heads
        );
    }

    /// **Unstable**: diagnostic estimate of tile working-set memory.
    pub fn estimated_working_set_bytes(&self) -> usize {
        size_of::<f32>()
            * (2 * self.tile_size_q * self.head_dim
                + 2 * self.tile_size_kv * self.head_dim
                + self.tile_size_q * self.tile_size_kv)
    }

    /// **Unstable**: diagnostic estimate of score-tile scratch memory.
    pub fn estimated_score_scratch_bytes(&self) -> usize {
        size_of::<f32>() * self.tile_size_q * self.tile_size_kv
    }

    /// **Unstable**: dispatch heuristic; threshold constant subject to tuning.
    ///
    /// Returns `true` when the tiled kernel is expected to outperform the
    /// fully-materialized attention for the given sequence length.
    ///
    /// For short sequences the full `seq_len * seq_len` score matrix fits
    /// comfortably in L1/L2 cache and the tiling bookkeeping (online-softmax
    /// rescaling, tile copies) is pure overhead. In that regime the caller
    /// should use the materialized path instead.
    pub fn should_use_tiled(&self, seq_len: usize) -> bool {
        seq_len >= TILED_SEQ_THRESHOLD
    }
}

/// **Unstable**: reusable scratch buffers for the tiled online-softmax kernel; layout may grow.
#[derive(Debug, Clone, Default)]
pub struct TiledAttentionBuffers {
    /// Query tile `[Br, D]`.
    pub q_tile: Vec<f32>,
    /// Key tile `[Bc, D]`.
    pub k_tile: Vec<f32>,
    /// Value tile `[Bc, D]`.
    pub v_tile: Vec<f32>,
    /// Score / probability tile `[Br, Bc]`.
    pub s_tile: Vec<f32>,
    /// Per-head output accumulator `[seq_len, D]`.
    pub output: Vec<f32>,
    /// Running row-wise max for the active query tile `[Br]`.
    pub row_max: Vec<f32>,
    /// Running row-wise exp sum for the active query tile `[Br]`.
    pub row_sum: Vec<f32>,

    q_head: Vec<f32>,
    k_head: Vec<f32>,
    v_head: Vec<f32>,
    v_tile_t: Vec<f32>,
    weighted_values: Vec<f32>,
    row_max_new: Vec<f32>,
    row_scale: Vec<f32>,
    row_sum_new: Vec<f32>,
}

impl TiledAttentionBuffers {
    /// **Unstable**: allocate buffers for the given max sequence length and config.
    pub fn new(max_seq_len: usize, config: &TiledAttentionConfig) -> Self {
        let mut buffers = Self {
            q_tile: Vec::new(),
            k_tile: Vec::new(),
            v_tile: Vec::new(),
            s_tile: Vec::new(),
            output: Vec::new(),
            row_max: Vec::new(),
            row_sum: Vec::new(),
            q_head: Vec::new(),
            k_head: Vec::new(),
            v_head: Vec::new(),
            v_tile_t: Vec::new(),
            weighted_values: Vec::new(),
            row_max_new: Vec::new(),
            row_scale: Vec::new(),
            row_sum_new: Vec::new(),
        };
        buffers.ensure_capacity(max_seq_len, config);
        buffers
    }

    /// **Unstable**: grow buffers in-place if the new capacity exceeds current allocation.
    pub fn ensure_capacity(&mut self, max_seq_len: usize, config: &TiledAttentionConfig) {
        let br = config.tile_size_q.max(1);
        let bc = config.tile_size_kv.max(1);
        let d = config.head_dim.max(1);
        let seq = max_seq_len.max(1);

        resize_zeroed(&mut self.q_tile, br * d);
        resize_zeroed(&mut self.k_tile, bc * d);
        resize_zeroed(&mut self.v_tile, bc * d);
        resize_zeroed(&mut self.s_tile, br * bc);
        resize_zeroed(&mut self.output, seq * d);
        resize_zeroed(&mut self.row_max, br);
        resize_zeroed(&mut self.row_sum, br);
        resize_zeroed(&mut self.q_head, seq * d);
        resize_zeroed(&mut self.k_head, seq * d);
        resize_zeroed(&mut self.v_head, seq * d);
        resize_zeroed(&mut self.v_tile_t, d * bc);
        resize_zeroed(&mut self.weighted_values, br * d);
        resize_zeroed(&mut self.row_max_new, br);
        resize_zeroed(&mut self.row_scale, br);
        resize_zeroed(&mut self.row_sum_new, br);
    }

    /// **Unstable**: total bytes currently allocated across all scratch buffers.
    pub fn allocated_bytes(&self) -> usize {
        let floats = self.q_tile.len()
            + self.k_tile.len()
            + self.v_tile.len()
            + self.s_tile.len()
            + self.output.len()
            + self.row_max.len()
            + self.row_sum.len()
            + self.q_head.len()
            + self.k_head.len()
            + self.v_head.len()
            + self.v_tile_t.len()
            + self.weighted_values.len()
            + self.row_max_new.len()
            + self.row_scale.len()
            + self.row_sum_new.len();
        floats * size_of::<f32>()
    }
}

fn resize_zeroed(buffer: &mut Vec<f32>, len: usize) {
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
}

#[inline]
fn kv_head_index(q_head: usize, num_q_heads: usize, num_kv_heads: usize) -> usize {
    debug_assert!(num_q_heads % num_kv_heads == 0);
    q_head / (num_q_heads / num_kv_heads)
}

#[inline]
fn extract_head_interleaved(
    src: &[f32],
    seq_len: usize,
    stride: usize,
    head_index: usize,
    head_dim: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(dst.len(), seq_len * head_dim);
    let head_offset = head_index * head_dim;
    for row in 0..seq_len {
        let src_start = row * stride + head_offset;
        let dst_start = row * head_dim;
        dst[dst_start..dst_start + head_dim].copy_from_slice(&src[src_start..src_start + head_dim]);
    }
}

#[inline]
fn write_head_to_interleaved(
    src: &[f32],
    seq_len: usize,
    stride: usize,
    head_index: usize,
    head_dim: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), seq_len * head_dim);
    let head_offset = head_index * head_dim;
    for row in 0..seq_len {
        let src_start = row * head_dim;
        let dst_start = row * stride + head_offset;
        dst[dst_start..dst_start + head_dim].copy_from_slice(&src[src_start..src_start + head_dim]);
    }
}

/// **Unstable**: diagnostic estimation of legacy materialized attention buffer footprint.
///
/// Estimate the bytes allocated by the legacy materialized attention buffers.
///
/// This matches the current `AttentionBuffers::new` layout.
pub fn estimate_materialized_attention_buffer_bytes(
    max_seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
) -> usize {
    let head_dim = hidden_size / num_heads;
    let floats = max_seq_len * hidden_size // q
        + max_seq_len * hidden_size // k
        + max_seq_len * hidden_size // v
        + num_heads * max_seq_len * max_seq_len // scores
        + num_heads * max_seq_len * head_dim // context
        + max_seq_len * hidden_size // concat
        + max_seq_len * intermediate_size // ffn_intermediate
        + max_seq_len * hidden_size // temp
        + max_seq_len * head_dim // q_head
        + max_seq_len * head_dim // k_head
        + head_dim * max_seq_len // v_head_t
        + max_seq_len * max_seq_len // scores_head
        + max_seq_len * head_dim; // context_head
    floats * size_of::<f32>()
}

/// Core tiled online-softmax attention kernel for a single head.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_tiled_attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    attention_mask: &[u32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    tile_size_q: usize,
    tile_size_kv: usize,
    _q_tile: &mut [f32],
    k_tile: &mut [f32],
    _v_tile: &mut [f32],
    s_tile: &mut [f32],
    row_max: &mut [f32],
    row_sum: &mut [f32],
    v_tile_t: &mut [f32],
    weighted_values: &mut [f32],
    row_max_new: &mut [f32],
    row_scale: &mut [f32],
    row_sum_new: &mut [f32],
) {
    output.fill(0.0);

    for q_start in (0..seq_len).step_by(tile_size_q) {
        let q_rows = min(tile_size_q, seq_len - q_start);
        // Q is already contiguous per-head -- use the slice directly instead
        // of copying into q_tile. This saves `q_rows * head_dim` copies per
        // outer iteration.
        let q_slice = &q[q_start * head_dim..(q_start + q_rows) * head_dim];

        let out_tile = &mut output[q_start * head_dim..(q_start + q_rows) * head_dim];
        out_tile.fill(0.0);
        row_max[..q_rows].fill(f32::NEG_INFINITY);
        row_sum[..q_rows].fill(0.0);

        for kv_start in (0..seq_len).step_by(tile_size_kv) {
            let kv_rows = min(tile_size_kv, seq_len - kv_start);

            // Copy K tile and build transposed V tile in one pass.
            // v_tile is NOT needed -- only v_tile_t is consumed by matmul_bt.
            for row in 0..kv_rows {
                let src_start = (kv_start + row) * head_dim;
                let dst_start = row * head_dim;
                k_tile[dst_start..dst_start + head_dim]
                    .copy_from_slice(&k[src_start..src_start + head_dim]);
                let v_src = &v[src_start..src_start + head_dim];
                for d in 0..head_dim {
                    v_tile_t[d * kv_rows + row] = v_src[d];
                }
            }

            let s_tile_used = &mut s_tile[..q_rows * kv_rows];
            matmul_bt(
                q_slice,
                &k_tile[..kv_rows * head_dim],
                s_tile_used,
                q_rows,
                head_dim,
                kv_rows,
            );

            for r in 0..q_rows {
                let row = &mut s_tile_used[r * kv_rows..(r + 1) * kv_rows];
                let mut local_max = f32::NEG_INFINITY;
                for c in 0..kv_rows {
                    let mut value = row[c] * scale;
                    if attention_mask[kv_start + c] == 0 {
                        value = MASK_VALUE;
                    }
                    row[c] = value;
                    local_max = local_max.max(value);
                }

                let prev_max = row_max[r];
                let new_max = prev_max.max(local_max);
                row_max_new[r] = new_max;
                row_scale[r] = if prev_max.is_finite() {
                    (prev_max - new_max).exp()
                } else {
                    0.0
                };

                let mut tile_sum = 0.0f32;
                for value in row.iter_mut() {
                    *value = (*value - new_max).exp();
                    tile_sum += *value;
                }
                row_sum_new[r] = row_sum[r] * row_scale[r] + tile_sum;
            }

            let weighted_values_used = &mut weighted_values[..q_rows * head_dim];
            matmul_bt(
                &s_tile[..q_rows * kv_rows],
                &v_tile_t[..head_dim * kv_rows],
                weighted_values_used,
                q_rows,
                kv_rows,
                head_dim,
            );

            for r in 0..q_rows {
                let alpha = row_scale[r];
                let out_row = &mut out_tile[r * head_dim..(r + 1) * head_dim];
                let weighted_row = &weighted_values_used[r * head_dim..(r + 1) * head_dim];
                if alpha == 0.0 {
                    out_row.copy_from_slice(weighted_row);
                } else {
                    for d in 0..head_dim {
                        out_row[d] = out_row[d] * alpha + weighted_row[d];
                    }
                }
                row_max[r] = row_max_new[r];
                row_sum[r] = row_sum_new[r];
            }
        }

        for r in 0..q_rows {
            let denom = row_sum[r];
            let out_row = &mut out_tile[r * head_dim..(r + 1) * head_dim];
            if denom > 0.0 {
                let inv = 1.0 / denom;
                for value in out_row.iter_mut() {
                    *value *= inv;
                }
            } else {
                out_row.fill(0.0);
            }
        }
    }
}

/// Run the tiled kernel for one head using the q_head/k_head/v_head already
/// stored inside `buffers`. Caller must have called `ensure_capacity` already.
#[inline]
#[allow(clippy::too_many_arguments)]
fn tiled_attention_head_from_internal_buffers(
    attention_mask: &[u32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    config: &TiledAttentionConfig,
    buffers: &mut TiledAttentionBuffers,
) {
    // Capacity is ensured once by the caller (tiled_multi_head_attention_in_place).
    let TiledAttentionBuffers {
        q_tile,
        k_tile,
        v_tile,
        s_tile,
        output,
        row_max,
        row_sum,
        q_head,
        k_head,
        v_head,
        v_tile_t,
        weighted_values,
        row_max_new,
        row_scale,
        row_sum_new,
    } = buffers;

    run_tiled_attention_head(
        &q_head[..seq_len * head_dim],
        &k_head[..seq_len * head_dim],
        &v_head[..seq_len * head_dim],
        &mut output[..seq_len * head_dim],
        attention_mask,
        seq_len,
        head_dim,
        scale,
        config.tile_size_q,
        config.tile_size_kv,
        q_tile,
        k_tile,
        v_tile,
        s_tile,
        row_max,
        row_sum,
        v_tile_t,
        weighted_values,
        row_max_new,
        row_scale,
        row_sum_new,
    );
}

/// **Unstable**: single-head tiled attention with caller-supplied buffers; signature may change.
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    attention_mask: &[u32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    config: &TiledAttentionConfig,
    buffers: &mut TiledAttentionBuffers,
) {
    debug_assert_eq!(q.len(), seq_len * head_dim);
    debug_assert_eq!(k.len(), seq_len * head_dim);
    debug_assert_eq!(v.len(), seq_len * head_dim);
    debug_assert_eq!(output.len(), seq_len * head_dim);
    debug_assert_eq!(attention_mask.len(), seq_len);
    assert_eq!(
        config.head_dim, head_dim,
        "config.head_dim must match head_dim"
    );

    buffers.ensure_capacity(seq_len, config);
    let TiledAttentionBuffers {
        q_tile,
        k_tile,
        v_tile,
        s_tile,
        row_max,
        row_sum,
        v_tile_t,
        weighted_values,
        row_max_new,
        row_scale,
        row_sum_new,
        ..
    } = buffers;

    run_tiled_attention_head(
        q,
        k,
        v,
        output,
        attention_mask,
        seq_len,
        head_dim,
        scale,
        config.tile_size_q,
        config.tile_size_kv,
        q_tile,
        k_tile,
        v_tile,
        s_tile,
        row_max,
        row_sum,
        v_tile_t,
        weighted_values,
        row_max_new,
        row_scale,
        row_sum_new,
    );
}

/// **Unstable**: tiled multi-head attention using online-softmax; GQA support in progress.
///
/// Tiled multi-head attention. This mirrors the existing materialized path but
/// uses the online-softmax tiled kernel instead of materializing `scores`.
#[allow(clippy::too_many_arguments)]
pub fn tiled_multi_head_attention(
    hidden_states: &[f32],
    layer_weights: &TransformerLayerWeights<'_>,
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    buffers: &mut AttentionBuffers,
    tiled_buffers: &mut TiledAttentionBuffers,
    config: &TiledAttentionConfig,
) -> Vec<f32> {
    tiled_multi_head_attention_in_place(
        hidden_states,
        layer_weights,
        attention_mask,
        seq_len,
        hidden_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        buffers,
        tiled_buffers,
        config,
    );
    buffers.temp[..seq_len * hidden_size].to_vec()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tiled_multi_head_attention_in_place(
    hidden_states: &[f32],
    layer_weights: &TransformerLayerWeights<'_>,
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    buffers: &mut AttentionBuffers,
    tiled_buffers: &mut TiledAttentionBuffers,
    config: &TiledAttentionConfig,
) {
    config.validate();
    assert_eq!(
        config.num_q_heads, num_q_heads,
        "config.num_q_heads must match"
    );
    assert_eq!(
        config.num_kv_heads, num_kv_heads,
        "config.num_kv_heads must match"
    );
    assert_eq!(config.head_dim, head_dim, "config.head_dim must match");
    assert_eq!(hidden_states.len(), seq_len * hidden_size);
    assert_eq!(attention_mask.len(), seq_len);
    assert_eq!(hidden_size, num_q_heads * head_dim);
    assert_eq!(num_q_heads % num_kv_heads, 0);

    let q_proj_dim = num_q_heads * head_dim;
    let kv_proj_dim = num_kv_heads * head_dim;
    let used_hidden = seq_len * hidden_size;
    let used_q = seq_len * q_proj_dim;
    let used_kv = seq_len * kv_proj_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    debug_assert_eq!(layer_weights.query_weight.rows, q_proj_dim);
    debug_assert_eq!(layer_weights.query_weight.cols, hidden_size);
    debug_assert_eq!(layer_weights.key_weight.rows, kv_proj_dim);
    debug_assert_eq!(layer_weights.key_weight.cols, hidden_size);
    debug_assert_eq!(layer_weights.value_weight.rows, kv_proj_dim);
    debug_assert_eq!(layer_weights.value_weight.cols, hidden_size);
    debug_assert_eq!(layer_weights.attn_output_weight.rows, hidden_size);
    debug_assert_eq!(layer_weights.attn_output_weight.cols, hidden_size);

    tiled_buffers.ensure_capacity(seq_len, config);

    {
        let q = &mut buffers.q[..used_q];
        matmul_bt(
            hidden_states,
            layer_weights.query_weight.data,
            q,
            seq_len,
            hidden_size,
            q_proj_dim,
        );
        add_bias(q, layer_weights.query_bias.data, q_proj_dim);
    }

    {
        let k = &mut buffers.k[..used_kv];
        matmul_bt(
            hidden_states,
            layer_weights.key_weight.data,
            k,
            seq_len,
            hidden_size,
            kv_proj_dim,
        );
        add_bias(k, layer_weights.key_bias.data, kv_proj_dim);
    }

    {
        let v = &mut buffers.v[..used_kv];
        matmul_bt(
            hidden_states,
            layer_weights.value_weight.data,
            v,
            seq_len,
            hidden_size,
            kv_proj_dim,
        );
        add_bias(v, layer_weights.value_bias.data, kv_proj_dim);
    }

    let q_all = &buffers.q[..used_q];
    let k_all = &buffers.k[..used_kv];
    let v_all = &buffers.v[..used_kv];
    let concat = &mut buffers.concat[..used_hidden];

    for q_head_idx in 0..num_q_heads {
        let kv_head_idx = kv_head_index(q_head_idx, num_q_heads, num_kv_heads);
        extract_head_interleaved(
            q_all,
            seq_len,
            q_proj_dim,
            q_head_idx,
            head_dim,
            &mut tiled_buffers.q_head[..seq_len * head_dim],
        );
        extract_head_interleaved(
            k_all,
            seq_len,
            kv_proj_dim,
            kv_head_idx,
            head_dim,
            &mut tiled_buffers.k_head[..seq_len * head_dim],
        );
        extract_head_interleaved(
            v_all,
            seq_len,
            kv_proj_dim,
            kv_head_idx,
            head_dim,
            &mut tiled_buffers.v_head[..seq_len * head_dim],
        );

        tiled_attention_head_from_internal_buffers(
            attention_mask,
            seq_len,
            head_dim,
            scale,
            config,
            tiled_buffers,
        );

        write_head_to_interleaved(
            &tiled_buffers.output[..seq_len * head_dim],
            seq_len,
            hidden_size,
            q_head_idx,
            head_dim,
            concat,
        );
    }

    let output = &mut buffers.temp[..used_hidden];
    matmul_bt(
        &buffers.concat[..used_hidden],
        layer_weights.attn_output_weight.data,
        output,
        seq_len,
        hidden_size,
        hidden_size,
    );
    add_bias(output, layer_weights.attn_output_bias.data, hidden_size);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{AttentionBuffers, multi_head_attention};
    use crate::forward::cpu::{matmul_bt, softmax_attention};
    use crate::weights::{Tensor1D, Tensor2D, TransformerLayerWeights};

    #[derive(Clone)]
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.state >> 32) as u32
        }

        fn next_f32(&mut self) -> f32 {
            let unit = (self.next_u32() as f32) / (u32::MAX as f32);
            unit - 0.5
        }
    }

    fn random_vec(len: usize, rng: &mut Lcg) -> Vec<f32> {
        (0..len).map(|_| rng.next_f32()).collect()
    }

    fn patterned_mask(seq_len: usize) -> Vec<u32> {
        let mut mask = vec![1u32; seq_len];
        for i in 0..seq_len {
            if i > 0 && (i % 7 == 0 || i % 11 == 0) {
                mask[i] = 0;
            }
        }
        mask[0] = 1;
        mask
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn reference_attention_head(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        attention_mask: &[u32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; seq_len * seq_len];
        matmul_bt(q, k, &mut scores, seq_len, head_dim, seq_len);
        for i in 0..seq_len {
            let row = &mut scores[i * seq_len..(i + 1) * seq_len];
            for j in 0..seq_len {
                row[j] *= scale;
                if attention_mask[j] == 0 {
                    row[j] = MASK_VALUE;
                }
            }
        }
        softmax_attention(&mut scores, seq_len, 1);

        let mut output = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let weight = scores[i * seq_len + j];
                let v_row = &v[j * head_dim..(j + 1) * head_dim];
                let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
                for d in 0..head_dim {
                    out_row[d] += weight * v_row[d];
                }
            }
        }
        output
    }

    fn reference_projected_attention(
        q_all: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        attention_mask: &[u32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let q_proj_dim = num_q_heads * head_dim;
        let kv_proj_dim = num_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut concat = vec![0.0f32; seq_len * q_proj_dim];
        let group_size = num_q_heads / num_kv_heads;

        let mut q_head = vec![0.0f32; seq_len * head_dim];
        let mut k_head = vec![0.0f32; seq_len * head_dim];
        let mut v_head = vec![0.0f32; seq_len * head_dim];

        for qh in 0..num_q_heads {
            let kvh = qh / group_size;
            extract_head_interleaved(q_all, seq_len, q_proj_dim, qh, head_dim, &mut q_head);
            extract_head_interleaved(k_all, seq_len, kv_proj_dim, kvh, head_dim, &mut k_head);
            extract_head_interleaved(v_all, seq_len, kv_proj_dim, kvh, head_dim, &mut v_head);
            let out = reference_attention_head(
                &q_head,
                &k_head,
                &v_head,
                attention_mask,
                seq_len,
                head_dim,
                scale,
            );
            write_head_to_interleaved(&out, seq_len, q_proj_dim, qh, head_dim, &mut concat);
        }

        concat
    }

    struct OwnedLayer {
        hidden_size: usize,
        intermediate_size: usize,
        q_proj_dim: usize,
        kv_proj_dim: usize,
        q_weight: Box<[f32]>,
        q_bias: Box<[f32]>,
        k_weight: Box<[f32]>,
        k_bias: Box<[f32]>,
        v_weight: Box<[f32]>,
        v_bias: Box<[f32]>,
        out_weight: Box<[f32]>,
        out_bias: Box<[f32]>,
        attn_ln_w: Box<[f32]>,
        attn_ln_b: Box<[f32]>,
        ffn_int_w: Box<[f32]>,
        ffn_int_b: Box<[f32]>,
        ffn_out_w: Box<[f32]>,
        ffn_out_b: Box<[f32]>,
        ffn_ln_w: Box<[f32]>,
        ffn_ln_b: Box<[f32]>,
    }

    impl OwnedLayer {
        fn borrowed(&self) -> TransformerLayerWeights<'_> {
            TransformerLayerWeights {
                query_weight: Tensor2D {
                    data: &self.q_weight,
                    rows: self.q_proj_dim,
                    cols: self.hidden_size,
                },
                query_bias: Tensor1D {
                    data: &self.q_bias,
                    len: self.q_proj_dim,
                },
                key_weight: Tensor2D {
                    data: &self.k_weight,
                    rows: self.kv_proj_dim,
                    cols: self.hidden_size,
                },
                key_bias: Tensor1D {
                    data: &self.k_bias,
                    len: self.kv_proj_dim,
                },
                value_weight: Tensor2D {
                    data: &self.v_weight,
                    rows: self.kv_proj_dim,
                    cols: self.hidden_size,
                },
                value_bias: Tensor1D {
                    data: &self.v_bias,
                    len: self.kv_proj_dim,
                },
                attn_output_weight: Tensor2D {
                    data: &self.out_weight,
                    rows: self.hidden_size,
                    cols: self.hidden_size,
                },
                attn_output_bias: Tensor1D {
                    data: &self.out_bias,
                    len: self.hidden_size,
                },
                attn_layer_norm_weight: Tensor1D {
                    data: &self.attn_ln_w,
                    len: self.hidden_size,
                },
                attn_layer_norm_bias: Tensor1D {
                    data: &self.attn_ln_b,
                    len: self.hidden_size,
                },
                ffn_intermediate_weight: Tensor2D {
                    data: &self.ffn_int_w,
                    rows: self.intermediate_size,
                    cols: self.hidden_size,
                },
                ffn_intermediate_bias: Tensor1D {
                    data: &self.ffn_int_b,
                    len: self.intermediate_size,
                },
                ffn_output_weight: Tensor2D {
                    data: &self.ffn_out_w,
                    rows: self.hidden_size,
                    cols: self.intermediate_size,
                },
                ffn_output_bias: Tensor1D {
                    data: &self.ffn_out_b,
                    len: self.hidden_size,
                },
                ffn_layer_norm_weight: Tensor1D {
                    data: &self.ffn_ln_w,
                    len: self.hidden_size,
                },
                ffn_layer_norm_bias: Tensor1D {
                    data: &self.ffn_ln_b,
                    len: self.hidden_size,
                },
            }
        }
    }

    fn build_test_layer(
        hidden_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rng: &mut Lcg,
    ) -> OwnedLayer {
        let q_proj_dim = num_q_heads * head_dim;
        let kv_proj_dim = num_kv_heads * head_dim;

        OwnedLayer {
            hidden_size,
            intermediate_size,
            q_proj_dim,
            kv_proj_dim,
            q_weight: random_vec(q_proj_dim * hidden_size, rng).into_boxed_slice(),
            q_bias: random_vec(q_proj_dim, rng).into_boxed_slice(),
            k_weight: random_vec(kv_proj_dim * hidden_size, rng).into_boxed_slice(),
            k_bias: random_vec(kv_proj_dim, rng).into_boxed_slice(),
            v_weight: random_vec(kv_proj_dim * hidden_size, rng).into_boxed_slice(),
            v_bias: random_vec(kv_proj_dim, rng).into_boxed_slice(),
            out_weight: random_vec(hidden_size * hidden_size, rng).into_boxed_slice(),
            out_bias: random_vec(hidden_size, rng).into_boxed_slice(),
            attn_ln_w: vec![1.0f32; hidden_size].into_boxed_slice(),
            attn_ln_b: vec![0.0f32; hidden_size].into_boxed_slice(),
            ffn_int_w: random_vec(intermediate_size * hidden_size, rng).into_boxed_slice(),
            ffn_int_b: random_vec(intermediate_size, rng).into_boxed_slice(),
            ffn_out_w: random_vec(hidden_size * intermediate_size, rng).into_boxed_slice(),
            ffn_out_b: random_vec(hidden_size, rng).into_boxed_slice(),
            ffn_ln_w: vec![1.0f32; hidden_size].into_boxed_slice(),
            ffn_ln_b: vec![0.0f32; hidden_size].into_boxed_slice(),
        }
    }

    #[test]
    fn test_optimal_tile_sizes_are_nonzero_and_cache_bounded() {
        for head_dim in [32usize, 64, 128, 256] {
            let (br, bc) = TiledAttentionConfig::optimal_tile_sizes(head_dim);
            assert!(br > 0 && bc > 0);
            let working = 2 * br * head_dim + 2 * bc * head_dim + br * bc;
            assert!(working * size_of::<f32>() <= L1_BUDGET_BYTES);
        }
    }

    #[test]
    fn test_tiled_attention_head_matches_reference_across_sequence_lengths() {
        let head_dim = 32usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let config = TiledAttentionConfig::new(1, 1, head_dim);
        let mut tiled_buffers = TiledAttentionBuffers::new(512, &config);
        let mut rng = Lcg::new(0x1234_5678);

        for &seq_len in &[1usize, 16, 32, 64, 128, 255, 512] {
            let q = random_vec(seq_len * head_dim, &mut rng);
            let k = random_vec(seq_len * head_dim, &mut rng);
            let v = random_vec(seq_len * head_dim, &mut rng);
            let mask = patterned_mask(seq_len);
            let mut tiled_out = vec![0.0f32; seq_len * head_dim];
            let ref_out = reference_attention_head(&q, &k, &v, &mask, seq_len, head_dim, scale);
            tiled_attention_head(
                &q,
                &k,
                &v,
                &mut tiled_out,
                &mask,
                seq_len,
                head_dim,
                scale,
                &config,
                &mut tiled_buffers,
            );
            let max_diff = max_abs_diff(&ref_out, &tiled_out);
            assert!(
                max_diff <= 5e-3,
                "seq_len={seq_len}, max_abs_diff={max_diff} (tolerance 5e-3 for SIMD softmax)"
            );
        }
    }

    #[test]
    fn test_tiled_multi_head_attention_matches_materialized_mha() {
        let hidden_size = 64usize;
        let num_heads = 4usize;
        let head_dim = 16usize;
        let intermediate_size = 128usize;
        let seq_len = 67usize;
        let mut rng = Lcg::new(0xfeed_beef);
        let hidden_states = random_vec(seq_len * hidden_size, &mut rng);
        let attention_mask = patterned_mask(seq_len);

        let owned_layer = build_test_layer(
            hidden_size,
            num_heads,
            num_heads,
            head_dim,
            intermediate_size,
            &mut rng,
        );
        let layer = owned_layer.borrowed();

        let mut materialized_buffers =
            AttentionBuffers::new(seq_len, hidden_size, num_heads, intermediate_size);
        let mut tiled_attention_buffers =
            AttentionBuffers::new(seq_len, hidden_size, num_heads, intermediate_size);
        let config = TiledAttentionConfig::new(num_heads, num_heads, head_dim);
        let mut tiled_buffers = TiledAttentionBuffers::new(seq_len, &config);

        let materialized = multi_head_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            hidden_size,
            num_heads,
            head_dim,
            &mut materialized_buffers,
        );
        let tiled = tiled_multi_head_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            hidden_size,
            num_heads,
            num_heads,
            head_dim,
            &mut tiled_attention_buffers,
            &mut tiled_buffers,
            &config,
        );
        let max_diff = max_abs_diff(&materialized, &tiled);
        assert!(
            max_diff <= 0.05,
            "max_abs_diff={max_diff} (tolerance 5e-2: tiled online-softmax vs materialized two-pass softmax have different FP ordering)"
        );
    }

    fn reference_general_projected_attention(
        hidden_states: &[f32],
        layer: &TransformerLayerWeights<'_>,
        attention_mask: &[u32],
        seq_len: usize,
        hidden_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let q_proj_dim = num_q_heads * head_dim;
        let kv_proj_dim = num_kv_heads * head_dim;

        let mut q = vec![0.0f32; seq_len * q_proj_dim];
        let mut k = vec![0.0f32; seq_len * kv_proj_dim];
        let mut v = vec![0.0f32; seq_len * kv_proj_dim];
        matmul_bt(
            hidden_states,
            layer.query_weight.data,
            &mut q,
            seq_len,
            hidden_size,
            q_proj_dim,
        );
        add_bias(&mut q, layer.query_bias.data, q_proj_dim);
        matmul_bt(
            hidden_states,
            layer.key_weight.data,
            &mut k,
            seq_len,
            hidden_size,
            kv_proj_dim,
        );
        add_bias(&mut k, layer.key_bias.data, kv_proj_dim);
        matmul_bt(
            hidden_states,
            layer.value_weight.data,
            &mut v,
            seq_len,
            hidden_size,
            kv_proj_dim,
        );
        add_bias(&mut v, layer.value_bias.data, kv_proj_dim);

        let concat = reference_projected_attention(
            &q,
            &k,
            &v,
            attention_mask,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );

        let mut output = vec![0.0f32; seq_len * hidden_size];
        matmul_bt(
            &concat,
            layer.attn_output_weight.data,
            &mut output,
            seq_len,
            hidden_size,
            hidden_size,
        );
        add_bias(&mut output, layer.attn_output_bias.data, hidden_size);
        output
    }

    #[test]
    fn test_tiled_multi_head_attention_matches_reference_gqa() {
        let hidden_size = 64usize;
        let num_q_heads = 8usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;
        let intermediate_size = 128usize;
        let seq_len = 97usize;
        let mut rng = Lcg::new(0x0bad_f00d);
        let hidden_states = random_vec(seq_len * hidden_size, &mut rng);
        let attention_mask = patterned_mask(seq_len);

        let owned_layer = build_test_layer(
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            &mut rng,
        );
        let layer = owned_layer.borrowed();

        let reference = reference_general_projected_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );

        let mut buffers =
            AttentionBuffers::new(seq_len, hidden_size, num_q_heads, intermediate_size);
        let config = TiledAttentionConfig::new(num_q_heads, num_kv_heads, head_dim);
        let mut tiled_buffers = TiledAttentionBuffers::new(seq_len, &config);
        let tiled = tiled_multi_head_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut buffers,
            &mut tiled_buffers,
            &config,
        );

        let max_diff = max_abs_diff(&reference, &tiled);
        assert!(
            max_diff <= 0.05,
            "max_abs_diff={max_diff} (tolerance 5e-2: tiled online-softmax vs materialized two-pass softmax have different FP ordering)"
        );
    }

    #[test]
    fn test_tiled_multi_head_attention_matches_reference_mqa() {
        let hidden_size = 64usize;
        let num_q_heads = 8usize;
        let num_kv_heads = 1usize;
        let head_dim = 8usize;
        let intermediate_size = 128usize;
        let seq_len = 73usize;
        let mut rng = Lcg::new(0xface_cafe);
        let hidden_states = random_vec(seq_len * hidden_size, &mut rng);
        let attention_mask = patterned_mask(seq_len);

        let owned_layer = build_test_layer(
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            &mut rng,
        );
        let layer = owned_layer.borrowed();

        let reference = reference_general_projected_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
        );

        let mut buffers =
            AttentionBuffers::new(seq_len, hidden_size, num_q_heads, intermediate_size);
        let config = TiledAttentionConfig::new(num_q_heads, num_kv_heads, head_dim);
        let mut tiled_buffers = TiledAttentionBuffers::new(seq_len, &config);
        let tiled = tiled_multi_head_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            &mut buffers,
            &mut tiled_buffers,
            &config,
        );

        let max_diff = max_abs_diff(&reference, &tiled);
        assert!(
            max_diff <= 0.05,
            "max_abs_diff={max_diff} (tolerance 5e-2: tiled online-softmax vs materialized two-pass softmax have different FP ordering)"
        );
    }

    #[test]
    fn test_should_use_tiled_threshold() {
        let config = TiledAttentionConfig::new(12, 12, 32);
        assert!(!config.should_use_tiled(32));
        assert!(!config.should_use_tiled(64));
        assert!(!config.should_use_tiled(95));
        assert!(config.should_use_tiled(96));
        assert!(config.should_use_tiled(128));
        assert!(config.should_use_tiled(512));
    }

    #[test]
    fn test_tile_q_ge_tile_kv() {
        // The new heuristic should prefer tile_q >= tile_kv for all common head dims.
        for head_dim in [32usize, 64, 128] {
            let (br, bc) = TiledAttentionConfig::optimal_tile_sizes(head_dim);
            assert!(
                br >= bc,
                "head_dim={head_dim}: tile_q={br} < tile_kv={bc}, want tile_q >= tile_kv"
            );
        }
    }

    /// Manual benchmark: run with `--nocapture --ignored`.
    /// Compares tiled vs materialized (reference) per-head attention for
    /// bge-small (head_dim=32) and bge-base (head_dim=64) configurations.
    #[test]
    #[ignore]
    fn bench_attention_tiled_vs_materialized() {
        use std::time::Instant;

        let configs: &[(usize, usize, &str)] = &[
            (32, 12, "bge-small (hd=32, 12h)"),
            (64, 12, "bge-base  (hd=64, 12h)"),
        ];

        eprintln!();
        eprintln!(
            "{:30} {:>6} {:>12} {:>12} {:>8}",
            "config", "seqlen", "materialized", "tiled", "speedup"
        );
        eprintln!("{}", "-".repeat(72));

        for &(head_dim, num_heads, label) in configs {
            let hidden_size = num_heads * head_dim;
            let _intermediate_size = hidden_size * 4;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let config = TiledAttentionConfig::new(num_heads, num_heads, head_dim);

            for &seq_len in &[32usize, 64, 128, 256, 512] {
                let mut rng = Lcg::new(0xbe0c + seq_len as u64);
                let q = random_vec(seq_len * head_dim, &mut rng);
                let k = random_vec(seq_len * head_dim, &mut rng);
                let v = random_vec(seq_len * head_dim, &mut rng);
                let mask = vec![1u32; seq_len];

                // Warm up
                let ref_out = reference_attention_head(&q, &k, &v, &mask, seq_len, head_dim, scale);
                let mut tiled_buffers = TiledAttentionBuffers::new(seq_len, &config);
                let mut tiled_out = vec![0.0f32; seq_len * head_dim];
                tiled_attention_head(
                    &q,
                    &k,
                    &v,
                    &mut tiled_out,
                    &mask,
                    seq_len,
                    head_dim,
                    scale,
                    &config,
                    &mut tiled_buffers,
                );

                // Verify correctness
                let diff = max_abs_diff(&ref_out, &tiled_out);
                assert!(diff <= 1e-4, "correctness failed: max_diff={diff}");

                let iters = if seq_len <= 64 {
                    2000
                } else if seq_len <= 256 {
                    500
                } else {
                    200
                };

                // Benchmark materialized
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = std::hint::black_box(reference_attention_head(
                        &q, &k, &v, &mask, seq_len, head_dim, scale,
                    ));
                }
                let mat_ns = start.elapsed().as_nanos() as f64 / iters as f64;

                // Benchmark tiled
                let start = Instant::now();
                for _ in 0..iters {
                    tiled_attention_head(
                        &q,
                        &k,
                        &v,
                        std::hint::black_box(&mut tiled_out),
                        &mask,
                        seq_len,
                        head_dim,
                        scale,
                        &config,
                        &mut tiled_buffers,
                    );
                }
                let tiled_ns = start.elapsed().as_nanos() as f64 / iters as f64;

                let speedup = mat_ns / tiled_ns;
                eprintln!(
                    "{:30} {:>6} {:>10.0} ns {:>10.0} ns {:>7.2}x",
                    label, seq_len, mat_ns, tiled_ns, speedup
                );
            }
        }
    }
}
