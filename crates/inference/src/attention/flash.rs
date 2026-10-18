//! Tiled attention configuration and scratch-buffer support for tiled attention kernels.
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
            self.num_q_heads.is_multiple_of(self.num_kv_heads),
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

        assert_tiled_scratch_no_overflow(br, bc, d, seq);

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

/// Reject tile/sequence shape products that wrap `usize` before scratch allocation.
///
/// `TiledAttentionConfig::{tile_size_q, tile_size_kv, head_dim}` are public, caller-controlled
/// fields, so an arbitrary tile size can wrap a scratch-length product (`tile_q * tile_kv`,
/// `tile_kv * head_dim`, ...) to a small value. `resize_zeroed` would then allocate an undersized
/// buffer that the core kernel later slices as if the unwrapped tile were present, panicking from
/// an internal bound instead of rejecting the impossible config. This is the non-causal sibling of
/// the `flash_causal.rs` tile-scratch overflow guard (#366). Release-active: the wrap is invisible
/// to a debug-only check.
#[inline]
fn assert_tiled_scratch_no_overflow(br: usize, bc: usize, head_dim: usize, seq: usize) {
    assert!(
        br.checked_mul(bc).is_some(),
        "tiled scratch overflow: tile_q * tile_kv"
    );
    assert!(
        br.checked_mul(head_dim).is_some(),
        "tiled scratch overflow: tile_q * head_dim"
    );
    assert!(
        bc.checked_mul(head_dim).is_some(),
        "tiled scratch overflow: tile_kv * head_dim"
    );
    assert!(
        seq.checked_mul(head_dim).is_some(),
        "tiled scratch overflow: max_seq_len * head_dim"
    );
}

/// Reject multi-head projection shape products that wrap `usize` before the release length asserts.
///
/// The release-active `assert_eq!(hidden_size, num_q_heads * head_dim)` and the derived
/// `seq_len * q_proj_dim` / `seq_len * kv_proj_dim` slice lengths in
/// `tiled_multi_head_attention_in_place` are computed with plain `*`. A wrapping
/// `num_q_heads * head_dim` makes the `hidden_size` assert *accept* an impossible head layout, then
/// per-head extraction (`head_index * head_dim`, `src[start..start + head_dim]`) indexes a buffer
/// sized for the wrapped (tiny) shape and reads out of bounds. Mirrors
/// `flash_causal.rs::assert_flash_no_overflow` (#366). Caller guarantees `num_kv_heads != 0` and
/// `num_q_heads % num_kv_heads == 0` (asserted at the call site).
#[inline]
fn assert_flash_tiled_no_overflow(
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    hidden_size: usize,
) {
    assert!(
        num_q_heads.checked_mul(head_dim).is_some(),
        "tiled shape overflow: num_q_heads * head_dim"
    );
    assert!(
        num_kv_heads.checked_mul(head_dim).is_some(),
        "tiled shape overflow: num_kv_heads * head_dim"
    );
    assert!(
        seq_len.checked_mul(hidden_size).is_some(),
        "tiled shape overflow: seq_len * hidden_size"
    );
    assert!(
        seq_len.checked_mul(num_q_heads * head_dim).is_some(),
        "tiled shape overflow: seq_len * q_proj_dim"
    );
    assert!(
        seq_len.checked_mul(num_kv_heads * head_dim).is_some(),
        "tiled shape overflow: seq_len * kv_proj_dim"
    );
}

#[inline]
fn kv_head_index(q_head: usize, num_q_heads: usize, num_kv_heads: usize) -> usize {
    debug_assert!(num_q_heads.is_multiple_of(num_kv_heads));
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
/// This mirrors every field of `AttentionBuffers` (`crate::attention::standard`)
/// as it exists today, including the fused-QKV projection scratch (`qkv`) and
/// the full-layer V-transpose buffer (`v_all_t`) added by #674/#673: it is
/// *not* the field set from before those changes. `estimate_matches_actual_attention_buffer_lengths`
/// below cross-checks this estimate against `AttentionBuffers::new`'s real
/// field lengths so this can't silently rot again the way it did in #678.
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
        + max_seq_len * 3 * hidden_size // qkv
        + max_seq_len * head_dim // q_head
        + max_seq_len * head_dim // k_head
        + hidden_size * max_seq_len // v_all_t
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
                    // Masked keys are excluded structurally with a -inf score so their
                    // softmax weight is exactly zero no matter how low a valid score sits.
                    // A finite sentinel can be *exceeded* by a valid logit below it, which
                    // would make the masked key the row max and hand it dominant
                    // probability (#361).
                    let value = if attention_mask[kv_start + c] == 0 {
                        f32::NEG_INFINITY
                    } else {
                        row[c] * scale
                    };
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
                if new_max.is_finite() {
                    for value in row.iter_mut() {
                        // exp(-inf - finite) = 0 for masked columns; never NaN here
                        // because new_max is finite.
                        *value = (*value - new_max).exp();
                        tile_sum += *value;
                    }
                } else {
                    // Every key seen so far in this row is masked (new_max == -inf), so
                    // `*value - new_max` would be `-inf - -inf = NaN`. Define the masked
                    // probabilities as zero; an all-masked row then carries a zero
                    // row_sum and the final normalization emits a zero output, not NaN.
                    row.fill(0.0);
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
    assert_ne!(num_kv_heads, 0, "num_kv_heads must be > 0");
    assert_flash_tiled_no_overflow(num_q_heads, num_kv_heads, head_dim, seq_len, hidden_size);
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
mod tests;
