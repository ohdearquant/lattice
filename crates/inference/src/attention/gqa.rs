//! Grouped-query attention optimized for Qwen-style GQA.
//!
//! Refactors per-Q-head attention loop into per-KV-head loop:
//! - Iterate once per KV head (8), not once per Q head (16)
//! - Batch all Q heads sharing a KV head into one GEMM
//! - Transpose V once per KV head, batch scores @ V GEMM
//! - Fused scale + causal mask + softmax in one pass
//!
//! For Qwen3-Embedding-0.6B: 16 Q heads / 8 KV heads = groups of 2.
//! This halves attention BLAS calls: 32 → 16 per layer, 896 → 448 per forward.

use crate::forward::cpu::matmul_bt;
#[cfg(target_os = "macos")]
use crate::forward::cpu::sgemm_bt_strided;

/// **Unstable**: GQA head layout configuration; tied to Qwen3 attention shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GqaConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GqaConfig {
    /// **Unstable**: number of query heads per KV head.
    #[inline]
    pub fn groups(self) -> usize {
        debug_assert!(self.num_kv_heads > 0);
        debug_assert_eq!(self.num_heads % self.num_kv_heads, 0);
        self.num_heads / self.num_kv_heads
    }

    /// **Unstable**: total query projection dimension.
    #[inline]
    pub fn q_dim(self) -> usize {
        self.num_heads * self.head_dim
    }

    /// **Unstable**: total KV projection dimension.
    #[inline]
    pub fn kv_dim(self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// **Unstable**: pre-allocated scratch buffers for GQA; buffer set may grow with multi-batch support.
#[derive(Default, Clone, Debug)]
pub struct GqaScratch {
    /// Packed Q rows for one KV group: `[groups * seq_len, head_dim]`.
    q_batch: Vec<f32>,
    /// Batched score rows for one KV group: `[groups * seq_len, seq_len]`.
    scores_batch: Vec<f32>,
    /// Batched context rows for one KV group: `[groups * seq_len, head_dim]`.
    context_batch: Vec<f32>,
    /// Non-macOS fallback: packed K rows for one KV head: `[seq_len, head_dim]`.
    k_head: Vec<f32>,
    /// Transposed V for one KV head: `[head_dim, seq_len]`.
    v_head_t: Vec<f32>,
}

impl GqaScratch {
    /// **Unstable**: resize scratch buffers to hold a given sequence length and config.
    #[inline]
    pub fn reserve_for(&mut self, seq_len: usize, cfg: GqaConfig) {
        let groups = cfg.groups();
        let head_dim = cfg.head_dim;
        self.q_batch.resize(groups * seq_len * head_dim, 0.0);
        self.scores_batch.resize(groups * seq_len * seq_len, 0.0);
        self.context_batch.resize(groups * seq_len * head_dim, 0.0);
        self.k_head.resize(seq_len * head_dim, 0.0);
        self.v_head_t.resize(head_dim * seq_len, 0.0);
    }
}

/// **Unstable**: GQA kernel for Qwen3 models; Metal BLAS path under development.
///
/// Apply grouped-query attention.
///
/// Layouts (same as existing Qwen attention path):
/// - `q_buf`: `[seq_len, q_dim]`, interleaved by head
/// - `k_buf`: `[seq_len, kv_dim]`, interleaved by KV head
/// - `v_buf`: `[seq_len, kv_dim]`, interleaved by KV head
/// - `attn_out`: `[seq_len, q_dim]`, same layout as `q_buf`
///
/// Q and K must already have RoPE applied.
#[inline]
pub fn apply_gqa_attention(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    attn_out: &mut [f32],
    seq_len: usize,
    cfg: GqaConfig,
    scratch: &mut GqaScratch,
) {
    // Hard divisibility guard, matching `decode_attention_scores` and the flash
    // paths. `GqaConfig::groups()` only `debug_assert`s this invariant, which is
    // stripped in release; a non-divisible config would then truncate `groups`,
    // skip the high query heads, and leave them stale in `attn_out` (never zeroed)
    // while the unsafe strided-BLAS path below assumes a validated head layout.
    assert!(cfg.num_kv_heads > 0, "num_kv_heads must be > 0");
    assert_eq!(
        cfg.num_heads % cfg.num_kv_heads,
        0,
        "num_heads must be divisible by num_kv_heads"
    );
    let groups = cfg.groups();
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let head_dim = cfg.head_dim;

    // Release-active length guards, matching the divisibility asserts above and
    // `decode_attention_scores`. These must hold before the macOS unsafe strided-BLAS
    // path below dereferences `k_buf`/`q_buf` pointers; a too-short buffer would
    // otherwise read out of bounds inside Accelerate FFI in release builds, where
    // `debug_assert_eq!` is stripped. This is a safe public API, so the precondition
    // is enforced here rather than left to the caller.
    assert_eq!(q_buf.len(), seq_len * q_dim, "q_buf length mismatch");
    assert_eq!(k_buf.len(), seq_len * kv_dim, "k_buf length mismatch");
    assert_eq!(v_buf.len(), seq_len * kv_dim, "v_buf length mismatch");
    assert_eq!(attn_out.len(), seq_len * q_dim, "attn_out length mismatch");

    if seq_len == 0 {
        return;
    }

    scratch.reserve_for(seq_len, cfg);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for kv_h in 0..cfg.num_kv_heads {
        let q_head_start = kv_h * groups;
        let batch_rows = groups * seq_len;

        // Pack Q heads that share this KV head into contiguous [groups*seq, head_dim].
        extract_q_group(
            q_buf,
            seq_len,
            q_dim,
            head_dim,
            q_head_start,
            groups,
            &mut scratch.q_batch[..batch_rows * head_dim],
        );

        // Q_batch @ K_head^T -> scores_batch [groups*seq, seq]
        #[cfg(target_os = "macos")]
        {
            // On macOS: use strided BLAS to read K directly from k_buf without copy.
            // SAFETY: kv_h < num_kv_heads and head_dim/kv_dim come from the validated
            // config, and `k_buf.len() == seq_len * kv_dim` is asserted (release-active)
            // above, so this offset lands within the first K row of k_buf.
            let k_ptr = unsafe { k_buf.as_ptr().add(kv_h * head_dim) };
            // SAFETY: pointers derive from valid slices whose lengths were asserted
            // above, with strides matching q_batch, k_buf, and scores_batch dimensions.
            unsafe {
                sgemm_bt_strided(
                    scratch.q_batch.as_ptr(),
                    head_dim, // lda = head_dim (contiguous)
                    k_ptr,
                    kv_dim, // ldb = kv_dim (stride between K rows)
                    scratch.scores_batch.as_mut_ptr(),
                    seq_len, // ldc = seq_len (contiguous)
                    batch_rows,
                    seq_len,
                    head_dim,
                );
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Non-macOS: copy K head to contiguous buffer first.
            extract_k_head(
                k_buf,
                seq_len,
                kv_dim,
                head_dim,
                kv_h,
                &mut scratch.k_head[..seq_len * head_dim],
            );
            matmul_bt(
                &scratch.q_batch[..batch_rows * head_dim],
                &scratch.k_head[..seq_len * head_dim],
                &mut scratch.scores_batch[..batch_rows * seq_len],
                batch_rows,
                head_dim,
                seq_len,
            );
        }

        // Scale + causal mask + softmax (fused, handles batched rows).
        apply_scaled_causal_softmax_fused(
            &mut scratch.scores_batch[..batch_rows * seq_len],
            batch_rows,
            seq_len,
            scale,
        );

        // Transpose V for this KV head: [head_dim, seq_len] for matmul_bt trick.
        transpose_v_head(
            v_buf,
            seq_len,
            kv_dim,
            head_dim,
            kv_h,
            &mut scratch.v_head_t[..head_dim * seq_len],
        );

        // scores_batch @ V_T^T -> context_batch [groups*seq, head_dim]
        matmul_bt(
            &scratch.scores_batch[..batch_rows * seq_len],
            &scratch.v_head_t[..head_dim * seq_len],
            &mut scratch.context_batch[..batch_rows * head_dim],
            batch_rows,
            seq_len,
            head_dim,
        );

        // Scatter context back to interleaved attn_out.
        write_context_group(
            &scratch.context_batch[..batch_rows * head_dim],
            attn_out,
            seq_len,
            q_dim,
            head_dim,
            q_head_start,
            groups,
        );
    }
}

// --- Helper functions ---

#[inline]
fn extract_q_group(
    q_buf: &[f32],
    seq_len: usize,
    q_dim: usize,
    head_dim: usize,
    q_head_start: usize,
    groups: usize,
    q_batch: &mut [f32],
) {
    debug_assert_eq!(q_batch.len(), groups * seq_len * head_dim);
    for pos in 0..seq_len {
        let src_group_base = pos * q_dim + q_head_start * head_dim;
        let src_group = &q_buf[src_group_base..src_group_base + groups * head_dim];
        for g in 0..groups {
            let dst_off = (g * seq_len + pos) * head_dim;
            let src_off = g * head_dim;
            q_batch[dst_off..dst_off + head_dim]
                .copy_from_slice(&src_group[src_off..src_off + head_dim]);
        }
    }
}

#[cfg(any(not(target_os = "macos"), test))]
#[inline]
fn extract_k_head(
    k_buf: &[f32],
    seq_len: usize,
    kv_dim: usize,
    head_dim: usize,
    kv_h: usize,
    k_head: &mut [f32],
) {
    debug_assert_eq!(k_head.len(), seq_len * head_dim);
    for pos in 0..seq_len {
        let src_off = pos * kv_dim + kv_h * head_dim;
        let dst_off = pos * head_dim;
        k_head[dst_off..dst_off + head_dim].copy_from_slice(&k_buf[src_off..src_off + head_dim]);
    }
}

#[inline]
fn transpose_v_head(
    v_buf: &[f32],
    seq_len: usize,
    kv_dim: usize,
    head_dim: usize,
    kv_h: usize,
    v_head_t: &mut [f32],
) {
    debug_assert_eq!(v_head_t.len(), head_dim * seq_len);
    for pos in 0..seq_len {
        let src_off = pos * kv_dim + kv_h * head_dim;
        let src = &v_buf[src_off..src_off + head_dim];
        for d in 0..head_dim {
            v_head_t[d * seq_len + pos] = src[d];
        }
    }
}

#[inline]
fn write_context_group(
    context_batch: &[f32],
    attn_out: &mut [f32],
    seq_len: usize,
    q_dim: usize,
    head_dim: usize,
    q_head_start: usize,
    groups: usize,
) {
    debug_assert_eq!(context_batch.len(), groups * seq_len * head_dim);
    for g in 0..groups {
        let head = q_head_start + g;
        let src_base = g * seq_len * head_dim;
        for pos in 0..seq_len {
            let src_off = src_base + pos * head_dim;
            let dst_off = pos * q_dim + head * head_dim;
            attn_out[dst_off..dst_off + head_dim]
                .copy_from_slice(&context_batch[src_off..src_off + head_dim]);
        }
    }
}

/// Scale + causal mask + softmax in one pass over batched rows.
///
/// `batch_rows` must be a multiple of `seq_len` (groups * seq_len).
/// Row i within each group of seq_len rows has causal position `i % seq_len`.
#[inline]
fn apply_scaled_causal_softmax_fused(
    scores: &mut [f32],
    batch_rows: usize,
    seq_len: usize,
    scale: f32,
) {
    debug_assert_eq!(scores.len(), batch_rows * seq_len);
    debug_assert_eq!(batch_rows % seq_len, 0);

    for row_idx in 0..batch_rows {
        let qi = row_idx % seq_len;
        let valid = qi + 1;
        let row = &mut scores[row_idx * seq_len..(row_idx + 1) * seq_len];

        for v in &mut row[..valid] {
            *v *= scale;
        }

        // ADR-080 C1: route the fail-closed final decision through the
        // shared row-finalizer contract (#780). Behavior-preserving: this
        // implementation used real `.exp()` (not an approximation), so a
        // lone NaN score already propagated into `sum` and tripped the old
        // `sum > 0.0 && sum.is_finite()` else-branch; the explicit
        // `any_nan` pre-scan here just short-circuits that same outcome
        // before spending the `exp()` calls, matching `decode.rs`/
        // `flash-causal`'s degenerate-row fallback.
        let (max_val, any_nan) = crate::attention::softmax_row::row_max_and_any_nan(&row[..valid]);
        if crate::attention::softmax_row::row_fails_closed_pre_exp(max_val, any_nan) {
            row.fill(0.0);
            continue;
        }

        let mut sum = 0.0f32;
        for v in &mut row[..valid] {
            *v = (*v - max_val).exp();
            sum += *v;
        }

        crate::attention::softmax_row::finalize_row(&mut row[..valid], sum);
        row[valid..].fill(0.0);
    }
}

#[cfg(test)]
mod tests;
