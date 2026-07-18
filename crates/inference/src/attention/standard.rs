//! Standard attention buffers, multi-head attention, and in-place attention helpers.
use crate::forward::cpu::{add_bias, matmul_bt, softmax_attention};
use crate::lora_hook::{LoraHook, apply_lora_rows};
use crate::weights::TransformerLayerWeights;

/// **Unstable**: pre-allocated buffers for multi-head attention computation; field layout may change.
#[derive(Debug, Clone)]
pub struct AttentionBuffers {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub scores: Vec<f32>,
    pub context: Vec<f32>,
    pub concat: Vec<f32>,
    pub ffn_intermediate: Vec<f32>,
    pub temp: Vec<f32>,

    // Fused Q/K/V projection scratch: `[max_seq_len, 3*hidden_size]`. One
    // `matmul_bt` against the layer's fused QKV weight (built once at load
    // time and threaded in alongside `TransformerLayerWeights`, see
    // `crate::model::bert::LayerFusedQkv`) lands Q/K/V here interleaved per
    // row; the result is split into the contiguous `q`/`k`/`v` buffers above
    // before any per-head work, so the `LoraHook` contract (a plain
    // `[seq_len, hidden_size]` slice per tensor) is preserved exactly.
    qkv: Vec<f32>,

    // Reshape buffers for SIMD matmul in attention scoring and context
    // aggregation.  Allocated once per model lifetime, reused every layer.
    q_head: Vec<f32>,
    k_head: Vec<f32>,
    // Full-layer V transpose `[hidden_size, max_seq_len]`, computed once per
    // layer instead of once per head; each head's `[head_dim, seq_len]` slice
    // is a contiguous sub-range of this buffer.
    v_all_t: Vec<f32>,
    scores_head: Vec<f32>,
    context_head: Vec<f32>,
}

impl AttentionBuffers {
    /// **Unstable**: allocate buffers for a given model shape.
    pub fn new(
        max_seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        Self {
            q: vec![0.0; max_seq_len * hidden_size],
            k: vec![0.0; max_seq_len * hidden_size],
            v: vec![0.0; max_seq_len * hidden_size],
            scores: vec![0.0; num_heads * max_seq_len * max_seq_len],
            context: vec![0.0; num_heads * max_seq_len * head_dim],
            concat: vec![0.0; max_seq_len * hidden_size],
            ffn_intermediate: vec![0.0; max_seq_len * intermediate_size],
            temp: vec![0.0; max_seq_len * hidden_size],

            qkv: vec![0.0; max_seq_len * 3 * hidden_size],

            // Per-head reshape buffers for SIMD matmul
            q_head: vec![0.0; max_seq_len * head_dim],
            k_head: vec![0.0; max_seq_len * head_dim],
            v_all_t: vec![0.0; hidden_size * max_seq_len],
            scores_head: vec![0.0; max_seq_len * max_seq_len],
            context_head: vec![0.0; max_seq_len * head_dim],
        }
    }

    /// Sum of every scratch buffer's element count. Used by
    /// `crate::attention::flash::estimate_materialized_attention_buffer_bytes`'s
    /// own test to keep that estimate honest against this struct's actual
    /// field set (see PR #678 review finding 3).
    #[cfg(test)]
    pub(crate) fn total_scratch_len(&self) -> usize {
        self.q.len()
            + self.k.len()
            + self.v.len()
            + self.scores.len()
            + self.context.len()
            + self.concat.len()
            + self.ffn_intermediate.len()
            + self.temp.len()
            + self.qkv.len()
            + self.q_head.len()
            + self.k_head.len()
            + self.v_all_t.len()
            + self.scores_head.len()
            + self.context_head.len()
    }
}

/// **Unstable**: compute multi-head self-attention and return the output projection.
pub fn multi_head_attention(
    hidden_states: &[f32],
    layer_weights: &TransformerLayerWeights<'_>,
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    buffers: &mut AttentionBuffers,
    lora: &dyn LoraHook,
    layer_idx: usize,
) -> Vec<f32> {
    // This wrapper is a test/bench convenience, not the model hot path (see
    // `crate::model::bert::BertModel::forward_with_hook`, which calls
    // `multi_head_attention_in_place` directly with a fused QKV blob built
    // once at model-load time). Building the fused weight/bias here per call
    // keeps this function's public signature stable across #678 while still
    // exercising the single-fused-matmul code path.
    let fused_qkv_weight: Vec<f32> = layer_weights
        .query_weight
        .data
        .iter()
        .chain(layer_weights.key_weight.data.iter())
        .chain(layer_weights.value_weight.data.iter())
        .copied()
        .collect();
    let fused_qkv_bias: Vec<f32> = layer_weights
        .query_bias
        .data
        .iter()
        .chain(layer_weights.key_bias.data.iter())
        .chain(layer_weights.value_bias.data.iter())
        .copied()
        .collect();
    multi_head_attention_in_place(
        hidden_states,
        layer_weights,
        &fused_qkv_weight,
        &fused_qkv_bias,
        attention_mask,
        seq_len,
        hidden_size,
        num_heads,
        head_dim,
        buffers,
        lora,
        layer_idx,
    );
    buffers.temp[..seq_len * hidden_size].to_vec()
}

/// Release-active precondition guard for the bidirectional MHA shape products.
///
/// `multi_head_attention_in_place` previously checked the entry shapes only with
/// `debug_assert!`, so a release build silently accepted a malformed shape. Two
/// hazards follow from that: (1) `hidden_size != num_heads * head_dim` produces a
/// stale concat layout (the per-head copy loops write only `num_heads * head_dim`
/// lanes of each `hidden_size`-wide row, leaving the rest stale before the output
/// projection consumes them); (2) the local products `seq_len * hidden_size`,
/// `num_heads * seq_len * seq_len`, and `num_heads * seq_len * head_dim` are not
/// dominated by the `matmul_bt` boundary guards and could wrap a 64-bit `usize`
/// for an absurd shape, yielding an undersized scratch slice. This asserts the
/// head-layout invariant and that every product is computed before it wraps.
#[inline]
fn assert_standard_no_overflow(
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
) {
    assert!(num_heads > 0, "standard: num_heads must be non-zero");
    assert!(head_dim > 0, "standard: head_dim must be non-zero");
    assert!(
        num_heads.checked_mul(head_dim).is_some(),
        "standard shape overflow: num_heads * head_dim"
    );
    assert_eq!(
        hidden_size,
        num_heads * head_dim,
        "standard: hidden_size must equal num_heads * head_dim"
    );
    assert!(
        seq_len.checked_mul(hidden_size).is_some(),
        "standard shape overflow: seq_len * hidden_size"
    );
    assert!(
        num_heads.checked_mul(seq_len).is_some(),
        "standard shape overflow: num_heads * seq_len"
    );
    let nh_sl = num_heads * seq_len;
    assert!(
        nh_sl.checked_mul(seq_len).is_some(),
        "standard shape overflow: num_heads * seq_len * seq_len"
    );
    assert!(
        nh_sl.checked_mul(head_dim).is_some(),
        "standard shape overflow: num_heads * seq_len * head_dim"
    );
}

/// Internal in-place attention kernel.
///
/// `fused_qkv_weight`/`fused_qkv_bias` are the layer's Q/K/V weight and bias
/// concatenated vertically (`[3*hidden_size, hidden_size]` / `[3*hidden_size]`),
/// built once per layer at model-load time and threaded in alongside
/// `layer_weights` rather than stored on `TransformerLayerWeights` itself --
/// that struct is a publicly constructible API surface and gaining fields
/// there is a breaking change (#678).
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_head_attention_in_place(
    hidden_states: &[f32],
    layer_weights: &TransformerLayerWeights<'_>,
    fused_qkv_weight: &[f32],
    fused_qkv_bias: &[f32],
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    buffers: &mut AttentionBuffers,
    lora: &dyn LoraHook,
    layer_idx: usize,
) {
    assert_standard_no_overflow(seq_len, hidden_size, num_heads, head_dim);
    assert_eq!(
        hidden_states.len(),
        seq_len * hidden_size,
        "standard: hidden_states length must equal seq_len * hidden_size"
    );
    assert_eq!(
        attention_mask.len(),
        seq_len,
        "standard: attention_mask length must equal seq_len"
    );

    let used_hidden = seq_len * hidden_size;
    let used_scores = num_heads * seq_len * seq_len;

    // Fused Q/K/V projection (#674): one matmul_bt against the layer's
    // [3*hidden, hidden] fused weight, instead of three separate [hidden,
    // hidden] projections. The interleaved [seq_len, 3*hidden] result is then
    // split into the plain contiguous q/k/v buffers below in one pass so each
    // projected row can be passed to LoraHook::apply as a contiguous slice.
    {
        let AttentionBuffers { qkv, q, k, v, .. } = &mut *buffers;
        let qkv = &mut qkv[..seq_len * 3 * hidden_size];
        matmul_bt(
            hidden_states,
            fused_qkv_weight,
            qkv,
            seq_len,
            hidden_size,
            3 * hidden_size,
        );
        add_bias(qkv, fused_qkv_bias, 3 * hidden_size);

        for i in 0..seq_len {
            let src = i * 3 * hidden_size;
            q[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(&qkv[src..src + hidden_size]);
            k[i * hidden_size..(i + 1) * hidden_size]
                .copy_from_slice(&qkv[src + hidden_size..src + 2 * hidden_size]);
            v[i * hidden_size..(i + 1) * hidden_size]
                .copy_from_slice(&qkv[src + 2 * hidden_size..src + 3 * hidden_size]);
        }
    }
    apply_lora_rows(
        lora,
        layer_idx,
        "query",
        hidden_states,
        &mut buffers.q[..used_hidden],
        hidden_size,
        hidden_size,
    );
    apply_lora_rows(
        lora,
        layer_idx,
        "key",
        hidden_states,
        &mut buffers.k[..used_hidden],
        hidden_size,
        hidden_size,
    );
    apply_lora_rows(
        lora,
        layer_idx,
        "value",
        hidden_states,
        &mut buffers.v[..used_hidden],
        hidden_size,
        hidden_size,
    );

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Q*K^T via SIMD matmul_bt.
    //
    // Q and K are stored as [seq_len, hidden_size] with heads interleaved.
    // For each head we reshape into contiguous [seq_len, head_dim] buffers,
    // call matmul_bt (which computes A @ B^T), then scale and write back.
    {
        let (q_buf, rest) = buffers.q.split_at(used_hidden);
        // We need mutable access to scores, q_head, k_head, and scores_head
        // but they are all on `buffers`.  Split borrows through indexing:
        // q is read-only, k is read-only.  The reshape buffers and scores
        // are disjoint fields so we access them via `buffers` directly.
        let _ = rest; // suppress unused

        for h in 0..num_heads {
            let head_offset = h * head_dim;

            // Reshape Q for this head into contiguous q_head[seq_len, head_dim]
            for i in 0..seq_len {
                let src_start = i * hidden_size + head_offset;
                let dst_start = i * head_dim;
                buffers.q_head[dst_start..dst_start + head_dim]
                    .copy_from_slice(&q_buf[src_start..src_start + head_dim]);
            }

            // Reshape K for this head into contiguous k_head[seq_len, head_dim]
            for i in 0..seq_len {
                let src_start = i * hidden_size + head_offset;
                let dst_start = i * head_dim;
                buffers.k_head[dst_start..dst_start + head_dim]
                    .copy_from_slice(&buffers.k[src_start..src_start + head_dim]);
            }

            // matmul_bt: scores_head[seq_len, seq_len] = q_head[seq_len, head_dim] @ k_head[seq_len, head_dim]^T
            let q_head = &buffers.q_head[..seq_len * head_dim];
            let k_head = &buffers.k_head[..seq_len * head_dim];
            let scores_head = &mut buffers.scores_head[..seq_len * seq_len];
            matmul_bt(q_head, k_head, scores_head, seq_len, head_dim, seq_len);

            // Scale and copy into the full scores array at head h's offset
            let scores_offset = h * seq_len * seq_len;
            for (idx, &score) in scores_head.iter().enumerate() {
                buffers.scores[scores_offset + idx] = score * scale;
            }
        }
    }

    {
        let scores = &mut buffers.scores[..used_scores];
        for h in 0..num_heads {
            for i in 0..seq_len {
                let row = &mut scores[(h * seq_len + i) * seq_len..(h * seq_len + i + 1) * seq_len];
                for j in 0..seq_len {
                    if attention_mask[j] == 0 {
                        // Mask structurally with -inf, not a finite sentinel. A finite
                        // sentinel can be *exceeded* by a valid logit that sits below it,
                        // which would make the masked key the softmax row max and hand it
                        // dominant probability (the #361 leakage mode, fixed in flash.rs;
                        // standard.rs is the live materialized CPU path). softmax_attention
                        // zeros an all-masked row via its max-finiteness guard.
                        row[j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        softmax_attention(scores, seq_len, num_heads);
    }

    // Transpose V once per layer (#673 acceptable-minimum): a single
    // [hidden_size, seq_len] transpose instead of `num_heads` separate
    // [head_dim, seq_len] transposes. Total elements moved is identical
    // (hidden_size * seq_len either way); this removes the per-head loop
    // setup/dispatch overhead and gives each head a contiguous sub-range of
    // one buffer instead of re-deriving it per head.
    {
        let AttentionBuffers { v, v_all_t, .. } = &mut *buffers;
        let v_all_t = &mut v_all_t[..hidden_size * seq_len];
        for i in 0..seq_len {
            let row_start = i * hidden_size;
            for d in 0..hidden_size {
                v_all_t[d * seq_len + i] = v[row_start + d];
            }
        }
    }

    // scores*V context aggregation via SIMD matmul_bt, writing directly into
    // `concat`'s final interleaved position (#673): this removes the
    // intermediate `context` buffer and its extra full-hidden-size copy pass
    // that a separate "collect all heads, then interleave into concat" step
    // used to require.
    //
    // For each head we need: context[seq_len, head_dim] = scores[seq_len, seq_len] @ V_head[seq_len, head_dim]
    //
    // matmul_bt computes A @ B^T, so v_all_t's per-head slice (already
    // transposed above) serves directly as B in matmul_bt(scores, v_head_t, ...),
    // giving scores @ v_head_t^T = scores @ V_head.
    {
        let AttentionBuffers {
            scores,
            v_all_t,
            context_head,
            concat,
            ..
        } = &mut *buffers;
        let concat = &mut concat[..used_hidden];
        for h in 0..num_heads {
            let head_offset = h * head_dim;

            let scores_offset = h * seq_len * seq_len;
            let scores_head = &scores[scores_offset..scores_offset + seq_len * seq_len];
            let v_head_t = &v_all_t[head_offset * seq_len..(head_offset + head_dim) * seq_len];
            let context_head = &mut context_head[..seq_len * head_dim];

            // matmul_bt: context_head[seq_len, head_dim] = scores[seq_len, seq_len] @ v_head_t[head_dim, seq_len]^T
            //          = scores @ V_head
            matmul_bt(
                scores_head,
                v_head_t,
                context_head,
                seq_len,
                seq_len,
                head_dim,
            );

            for i in 0..seq_len {
                let dst = i * hidden_size + head_offset;
                concat[dst..dst + head_dim]
                    .copy_from_slice(&context_head[i * head_dim..(i + 1) * head_dim]);
            }
        }
    }

    {
        let concat = &buffers.concat[..used_hidden];
        let output = &mut buffers.temp[..used_hidden];
        matmul_bt(
            concat,
            layer_weights.attn_output_weight.data,
            output,
            seq_len,
            hidden_size,
            hidden_size,
        );
        add_bias(output, layer_weights.attn_output_bias.data, hidden_size);
        apply_lora_rows(
            lora,
            layer_idx,
            "attn_output",
            concat,
            output,
            hidden_size,
            hidden_size,
        );
    }
}

/// Fused batched multi-head attention for a packed (padding-free) `[total, hidden]`
/// tensor (#677).
///
/// This is the batch analogue of [`multi_head_attention_in_place`]: it fuses the
/// position-wise Q/K/V and output projections into single `matmul_bt` calls over
/// every row of the packed batch (bigger GEMMs, fewer BLAS/SIMD dispatches), while the
/// O(seq_len^2) score/softmax/context step -- which cannot be flattened across
/// sequences without letting one sequence's tokens attend into another sequence --
/// runs per-sequence, serially.
///
/// `hidden_states` is the packed `[total, hidden_size]` tensor: sequence `b` occupies
/// rows `cu_seqlens[b]..cu_seqlens[b+1]`, with no padding rows anywhere in between.
/// `cu_seqlens` has `batch + 1` entries (`cu_seqlens[0] == 0`,
/// `cu_seqlens[batch] == total`), the standard varlen cumulative-offset index. There
/// is no `attention_mask` parameter: every packed row is a real token, so there is
/// nothing to mask -- the structural `-inf` masking `multi_head_attention_in_place`
/// needs for its padded single-sequence case does not apply here.
///
/// This loop is deliberately **not** parallelized with rayon/std::thread, even
/// though each sequence's slice is independent. On macOS, `matmul_bt` dispatches to
/// Apple Accelerate (`forward/cpu/blas.rs`), which already runs GEMM across its own
/// multi-threaded AMX worker pool -- including at small M. Wrapping this per-sequence
/// loop in an outer parallel iterator nests a second thread pool on top of that one:
/// measured A/B on this crate's own bench harness, batch=64, all-MiniLM-L6-v2, showed
/// the rayon-parallel version at ~870-900 texts/s versus ~1225-1370 texts/s serial --
/// oversubscription made it slower, not faster, confirming the fused GEMM calls
/// (bigger M) are the actual lever, not manual threading on top of them. Only the
/// non-Accelerate fallback kernels (non-macOS, or the hand-rolled SIMD path) and
/// non-GEMM position-wise ops would be candidates for added threading, and only if a
/// fresh A/B on that specific backend shows a win -- do not assume this decision
/// carries over to a different dispatch path without re-measuring.
///
/// `output` receives the same output-projection result that
/// `multi_head_attention_in_place` writes into `buffers.temp` (bias-added,
/// LoRA-applied); callers add the residual and run `layer_norm` themselves, exactly
/// as the single-sequence path does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_head_attention_batched(
    hidden_states: &[f32],
    layer_weights: &TransformerLayerWeights<'_>,
    fused_qkv_weight: &[f32],
    fused_qkv_bias: &[f32],
    cu_seqlens: &[usize],
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    qkv: &mut [f32],
    concat: &mut [f32],
    output: &mut [f32],
    lora: &dyn LoraHook,
    layer_idx: usize,
) {
    assert!(
        cu_seqlens.len() >= 2,
        "standard: cu_seqlens must have at least 2 entries (batch + 1)"
    );
    assert_eq!(cu_seqlens[0], 0, "standard: cu_seqlens must start at 0");
    let batch = cu_seqlens.len() - 1;
    let total = cu_seqlens[batch];
    assert!(num_heads > 0, "standard: num_heads must be non-zero");
    assert!(head_dim > 0, "standard: head_dim must be non-zero");
    assert_eq!(
        hidden_size,
        num_heads * head_dim,
        "standard: hidden_size must equal num_heads * head_dim"
    );
    assert!(
        total.checked_mul(hidden_size).is_some(),
        "standard: total * hidden_size overflow"
    );
    let used_hidden = total * hidden_size;
    assert_eq!(
        hidden_states.len(),
        used_hidden,
        "standard: hidden_states length must equal total * hidden_size"
    );
    assert!(q.len() >= used_hidden, "standard: q scratch too small");
    assert!(k.len() >= used_hidden, "standard: k scratch too small");
    assert!(v.len() >= used_hidden, "standard: v scratch too small");
    assert!(
        concat.len() >= used_hidden,
        "standard: concat scratch too small"
    );
    assert!(
        output.len() >= used_hidden,
        "standard: output scratch too small"
    );
    assert!(
        qkv.len() >= used_hidden * 3,
        "standard: qkv scratch too small"
    );

    // Fused Q/K/V projection (#674): one matmul_bt call across every row in
    // the packed batch against the layer's [3*hidden, hidden] fused weight,
    // instead of three separate [hidden, hidden] projections. The interleaved
    // [total, 3*hidden] result is split into plain contiguous q/k/v buffers in
    // one pass so each projected row can be passed to LoraHook::apply as a
    // contiguous slice. This step is row-independent: it does not need
    // `cu_seqlens` at all, only the total row count.
    {
        let qkv = &mut qkv[..used_hidden * 3];
        matmul_bt(
            hidden_states,
            fused_qkv_weight,
            qkv,
            total,
            hidden_size,
            3 * hidden_size,
        );
        add_bias(qkv, fused_qkv_bias, 3 * hidden_size);

        for r in 0..total {
            let src = r * 3 * hidden_size;
            q[r * hidden_size..(r + 1) * hidden_size].copy_from_slice(&qkv[src..src + hidden_size]);
            k[r * hidden_size..(r + 1) * hidden_size]
                .copy_from_slice(&qkv[src + hidden_size..src + 2 * hidden_size]);
            v[r * hidden_size..(r + 1) * hidden_size]
                .copy_from_slice(&qkv[src + 2 * hidden_size..src + 3 * hidden_size]);
        }
    }
    apply_lora_rows(
        lora,
        layer_idx,
        "query",
        hidden_states,
        &mut q[..used_hidden],
        hidden_size,
        hidden_size,
    );
    apply_lora_rows(
        lora,
        layer_idx,
        "key",
        hidden_states,
        &mut k[..used_hidden],
        hidden_size,
        hidden_size,
    );
    apply_lora_rows(
        lora,
        layer_idx,
        "value",
        hidden_states,
        &mut v[..used_hidden],
        hidden_size,
        hidden_size,
    );

    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = &q[..used_hidden];
    let k = &k[..used_hidden];
    let v = &v[..used_hidden];
    let concat = &mut concat[..used_hidden];

    // Per-sequence score/softmax/context. This loop runs serially, one sequence
    // at a time; each `cu_seqlens[b]..cu_seqlens[b+1]` region of `concat` is
    // written by exactly one iteration, over disjoint slices of the shared
    // read-only `q`/`k`/`v` buffers. Every row in a sequence's region is real
    // (packing removed padding entirely), so there is no mask to apply here.
    for b in 0..batch {
        let start = cu_seqlens[b];
        let end = cu_seqlens[b + 1];
        assert!(
            end >= start,
            "standard: cu_seqlens must be non-decreasing (segment {b})"
        );
        let seq_len = end - start;
        if seq_len == 0 {
            continue;
        }
        // Per-sequence quadratic-in-seq_len scratch (scores is
        // [num_heads, seq_len, seq_len]): reuse the single-sequence overflow
        // guard, since this is exactly that shape check applied per segment.
        assert_standard_no_overflow(seq_len, hidden_size, num_heads, head_dim);

        let row_start = start * hidden_size;
        let concat_b = &mut concat[row_start..row_start + seq_len * hidden_size];

        let mut q_head = vec![0.0f32; seq_len * head_dim];
        let mut k_head = vec![0.0f32; seq_len * head_dim];
        let mut v_all_t = vec![0.0f32; hidden_size * seq_len];
        let mut scores_head = vec![0.0f32; seq_len * seq_len];
        let mut scores = vec![0.0f32; num_heads * seq_len * seq_len];
        let mut context_head = vec![0.0f32; seq_len * head_dim];

        // Q*K^T via SIMD matmul_bt, one head at a time (mirrors
        // multi_head_attention_in_place's single-sequence loop exactly).
        for h in 0..num_heads {
            let head_offset = h * head_dim;

            for i in 0..seq_len {
                let src_start = row_start + i * hidden_size + head_offset;
                let dst_start = i * head_dim;
                q_head[dst_start..dst_start + head_dim]
                    .copy_from_slice(&q[src_start..src_start + head_dim]);
            }
            for i in 0..seq_len {
                let src_start = row_start + i * hidden_size + head_offset;
                let dst_start = i * head_dim;
                k_head[dst_start..dst_start + head_dim]
                    .copy_from_slice(&k[src_start..src_start + head_dim]);
            }

            matmul_bt(
                &q_head[..seq_len * head_dim],
                &k_head[..seq_len * head_dim],
                &mut scores_head[..seq_len * seq_len],
                seq_len,
                head_dim,
                seq_len,
            );

            let scores_offset = h * seq_len * seq_len;
            for (idx, &score) in scores_head.iter().enumerate() {
                scores[scores_offset + idx] = score * scale;
            }
        }

        // No masking: every row in this sequence's packed region is real, so
        // softmax runs over the raw scaled scores directly (still through the
        // same fail-closed `softmax_attention` kernel as every other path).
        softmax_attention(&mut scores, seq_len, num_heads);

        // Transpose V once for this sequence (#673 acceptable-minimum):
        // one [hidden_size, seq_len] transpose instead of `num_heads`
        // separate [head_dim, seq_len] transposes; identical element
        // count moved, one loop instead of `num_heads` smaller loops.
        for i in 0..seq_len {
            let v_row_start = row_start + i * hidden_size;
            for d in 0..hidden_size {
                v_all_t[d * seq_len + i] = v[v_row_start + d];
            }
        }

        // scores*V context aggregation, writing directly into this
        // sequence's `concat_b` region (#673): removes the intermediate
        // `context` buffer and its extra full-hidden-size copy pass.
        for h in 0..num_heads {
            let head_offset = h * head_dim;

            let scores_offset = h * seq_len * seq_len;
            let scores_head = &scores[scores_offset..scores_offset + seq_len * seq_len];
            let v_head_t = &v_all_t[head_offset * seq_len..(head_offset + head_dim) * seq_len];
            matmul_bt(
                scores_head,
                v_head_t,
                &mut context_head[..seq_len * head_dim],
                seq_len,
                seq_len,
                head_dim,
            );

            for i in 0..seq_len {
                let dst = i * hidden_size + head_offset;
                concat_b[dst..dst + head_dim]
                    .copy_from_slice(&context_head[i * head_dim..(i + 1) * head_dim]);
            }
        }
    }

    // Fused output projection: one matmul_bt call across every row in the batch.
    let concat = &concat[..used_hidden];
    let output = &mut output[..used_hidden];
    matmul_bt(
        concat,
        layer_weights.attn_output_weight.data,
        output,
        total,
        hidden_size,
        hidden_size,
    );
    add_bias(output, layer_weights.attn_output_bias.data, hidden_size);
    apply_lora_rows(
        lora,
        layer_idx,
        "attn_output",
        concat,
        output,
        hidden_size,
        hidden_size,
    );
}

/// Pre-#677 padded batched attention, preserved as a test-only reference.
///
/// This is the padded `[batch, seq_len]` implementation `multi_head_attention_batched`
/// used before the packed/varlen rewrite: every sequence occupies a uniform
/// `seq_len`-row slot with masked padding rows, instead of a `cu_seqlens`-indexed
/// packed region. It exists solely so the parity test can compare the new packed
/// production path against an independently-computed ground truth without
/// reimplementing the old kernel inline in the test module.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_head_attention_batched_padded_reference(
    hidden_states: &[f32],
    layer_weights: &TransformerLayerWeights<'_>,
    fused_qkv_weight: &[f32],
    fused_qkv_bias: &[f32],
    attention_mask: &[u32],
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    qkv: &mut [f32],
    concat: &mut [f32],
    output: &mut [f32],
    lora: &dyn LoraHook,
    layer_idx: usize,
) {
    assert_standard_no_overflow(seq_len, hidden_size, num_heads, head_dim);
    assert!(
        batch.checked_mul(seq_len).is_some(),
        "standard: batch * seq_len overflow"
    );
    let rows = batch * seq_len;
    assert!(
        rows.checked_mul(hidden_size).is_some(),
        "standard: rows * hidden_size overflow"
    );
    let used_hidden = rows * hidden_size;
    assert_eq!(
        hidden_states.len(),
        used_hidden,
        "standard: hidden_states length must equal batch * seq_len * hidden_size"
    );
    assert_eq!(
        attention_mask.len(),
        rows,
        "standard: attention_mask length must equal batch * seq_len"
    );
    assert!(q.len() >= used_hidden, "standard: q scratch too small");
    assert!(k.len() >= used_hidden, "standard: k scratch too small");
    assert!(v.len() >= used_hidden, "standard: v scratch too small");
    assert!(
        concat.len() >= used_hidden,
        "standard: concat scratch too small"
    );
    assert!(
        output.len() >= used_hidden,
        "standard: output scratch too small"
    );
    assert!(
        qkv.len() >= used_hidden * 3,
        "standard: qkv scratch too small"
    );

    {
        let qkv = &mut qkv[..used_hidden * 3];
        matmul_bt(
            hidden_states,
            fused_qkv_weight,
            qkv,
            rows,
            hidden_size,
            3 * hidden_size,
        );
        add_bias(qkv, fused_qkv_bias, 3 * hidden_size);

        for r in 0..rows {
            let src = r * 3 * hidden_size;
            q[r * hidden_size..(r + 1) * hidden_size].copy_from_slice(&qkv[src..src + hidden_size]);
            k[r * hidden_size..(r + 1) * hidden_size]
                .copy_from_slice(&qkv[src + hidden_size..src + 2 * hidden_size]);
            v[r * hidden_size..(r + 1) * hidden_size]
                .copy_from_slice(&qkv[src + 2 * hidden_size..src + 3 * hidden_size]);
        }
    }
    apply_lora_rows(
        lora,
        layer_idx,
        "query",
        hidden_states,
        &mut q[..used_hidden],
        hidden_size,
        hidden_size,
    );
    apply_lora_rows(
        lora,
        layer_idx,
        "key",
        hidden_states,
        &mut k[..used_hidden],
        hidden_size,
        hidden_size,
    );
    apply_lora_rows(
        lora,
        layer_idx,
        "value",
        hidden_states,
        &mut v[..used_hidden],
        hidden_size,
        hidden_size,
    );

    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = &q[..used_hidden];
    let k = &k[..used_hidden];
    let v = &v[..used_hidden];
    let concat = &mut concat[..used_hidden];

    concat
        .chunks_mut(seq_len * hidden_size)
        .enumerate()
        .for_each(|(b, concat_b)| {
            let seq_offset = b * seq_len;
            let row_start = seq_offset * hidden_size;
            let mask_b = &attention_mask[seq_offset..seq_offset + seq_len];

            let mut q_head = vec![0.0f32; seq_len * head_dim];
            let mut k_head = vec![0.0f32; seq_len * head_dim];
            let mut v_all_t = vec![0.0f32; hidden_size * seq_len];
            let mut scores_head = vec![0.0f32; seq_len * seq_len];
            let mut scores = vec![0.0f32; num_heads * seq_len * seq_len];
            let mut context_head = vec![0.0f32; seq_len * head_dim];

            for h in 0..num_heads {
                let head_offset = h * head_dim;

                for i in 0..seq_len {
                    let src_start = row_start + i * hidden_size + head_offset;
                    let dst_start = i * head_dim;
                    q_head[dst_start..dst_start + head_dim]
                        .copy_from_slice(&q[src_start..src_start + head_dim]);
                }
                for i in 0..seq_len {
                    let src_start = row_start + i * hidden_size + head_offset;
                    let dst_start = i * head_dim;
                    k_head[dst_start..dst_start + head_dim]
                        .copy_from_slice(&k[src_start..src_start + head_dim]);
                }

                matmul_bt(
                    &q_head[..seq_len * head_dim],
                    &k_head[..seq_len * head_dim],
                    &mut scores_head[..seq_len * seq_len],
                    seq_len,
                    head_dim,
                    seq_len,
                );

                let scores_offset = h * seq_len * seq_len;
                for (idx, &score) in scores_head.iter().enumerate() {
                    scores[scores_offset + idx] = score * scale;
                }
            }

            for h in 0..num_heads {
                for i in 0..seq_len {
                    let row_off = (h * seq_len + i) * seq_len;
                    let row = &mut scores[row_off..row_off + seq_len];
                    for j in 0..seq_len {
                        if mask_b[j] == 0 {
                            row[j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            softmax_attention(&mut scores, seq_len, num_heads);

            for i in 0..seq_len {
                let v_row_start = row_start + i * hidden_size;
                for d in 0..hidden_size {
                    v_all_t[d * seq_len + i] = v[v_row_start + d];
                }
            }

            for h in 0..num_heads {
                let head_offset = h * head_dim;

                let scores_offset = h * seq_len * seq_len;
                let scores_head = &scores[scores_offset..scores_offset + seq_len * seq_len];
                let v_head_t = &v_all_t[head_offset * seq_len..(head_offset + head_dim) * seq_len];
                matmul_bt(
                    scores_head,
                    v_head_t,
                    &mut context_head[..seq_len * head_dim],
                    seq_len,
                    seq_len,
                    head_dim,
                );

                for i in 0..seq_len {
                    let dst = i * hidden_size + head_offset;
                    concat_b[dst..dst + head_dim]
                        .copy_from_slice(&context_head[i * head_dim..(i + 1) * head_dim]);
                }
            }
        });

    let concat = &concat[..used_hidden];
    let output = &mut output[..used_hidden];
    matmul_bt(
        concat,
        layer_weights.attn_output_weight.data,
        output,
        rows,
        hidden_size,
        hidden_size,
    );
    add_bias(output, layer_weights.attn_output_bias.data, hidden_size);
    apply_lora_rows(
        lora,
        layer_idx,
        "attn_output",
        concat,
        output,
        hidden_size,
        hidden_size,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora_hook::NoopLoraHook;
    use crate::weights::{Tensor1D, Tensor2D, TransformerLayerWeights};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RowShapeHook {
        row_width: usize,
        calls: AtomicUsize,
    }

    impl LoraHook for RowShapeHook {
        fn apply(&self, _layer_idx: usize, _module: &str, x: &[f32], output: &mut [f32]) {
            assert_eq!(x.len(), self.row_width);
            assert_eq!(output.len(), self.row_width);
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Build identity-like weights and run multi_head_attention on a small
    /// 2-token, 2-head, head_dim=2 model to verify the SIMD matmul path
    /// produces numerically correct results.
    #[test]
    fn test_attention_simd_matches_expected() {
        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 2;
        let hidden_size = num_heads * head_dim; // 4

        // hidden_states: 2 tokens, each of dim 4
        let hidden_states = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Use identity weight matrices (4x4) so Q=K=V=hidden_states (before bias).
        let identity_4x4: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 1.0, // row 3
        ];
        let zero_bias_4: Vec<f32> = vec![0.0; 4];

        // Attention layer norm weights: gamma=1, beta=0 (passthrough)
        let ones_4: Vec<f32> = vec![1.0; 4];

        // FFN weights: use identity for intermediate (but size could differ).
        // For this test we only care about the attention part, so make FFN
        // a passthrough too.  intermediate_size = hidden_size for simplicity.
        let intermediate_size = hidden_size;

        let layer = TransformerLayerWeights {
            query_weight: Tensor2D {
                data: &identity_4x4,
                rows: hidden_size,
                cols: hidden_size,
            },
            query_bias: Tensor1D {
                data: &zero_bias_4,
                len: hidden_size,
            },
            key_weight: Tensor2D {
                data: &identity_4x4,
                rows: hidden_size,
                cols: hidden_size,
            },
            key_bias: Tensor1D {
                data: &zero_bias_4,
                len: hidden_size,
            },
            value_weight: Tensor2D {
                data: &identity_4x4,
                rows: hidden_size,
                cols: hidden_size,
            },
            value_bias: Tensor1D {
                data: &zero_bias_4,
                len: hidden_size,
            },
            attn_output_weight: Tensor2D {
                data: &identity_4x4,
                rows: hidden_size,
                cols: hidden_size,
            },
            attn_output_bias: Tensor1D {
                data: &zero_bias_4,
                len: hidden_size,
            },
            attn_layer_norm_weight: Tensor1D {
                data: &ones_4,
                len: hidden_size,
            },
            attn_layer_norm_bias: Tensor1D {
                data: &zero_bias_4,
                len: hidden_size,
            },
            ffn_intermediate_weight: Tensor2D {
                data: &identity_4x4,
                rows: intermediate_size,
                cols: hidden_size,
            },
            ffn_intermediate_bias: Tensor1D {
                data: &zero_bias_4,
                len: intermediate_size,
            },
            ffn_output_weight: Tensor2D {
                data: &identity_4x4,
                rows: hidden_size,
                cols: intermediate_size,
            },
            ffn_output_bias: Tensor1D {
                data: &zero_bias_4,
                len: hidden_size,
            },
            ffn_layer_norm_weight: Tensor1D {
                data: &ones_4,
                len: hidden_size,
            },
            ffn_layer_norm_bias: Tensor1D {
                data: &zero_bias_4,
                len: hidden_size,
            },
        };

        let attention_mask = vec![1u32; seq_len];
        let mut buffers = AttentionBuffers::new(seq_len, hidden_size, num_heads, intermediate_size);

        let result = multi_head_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            hidden_size,
            num_heads,
            head_dim,
            &mut buffers,
            &NoopLoraHook,
            0,
        );

        // With identity Q/K/V weights, zero biases, and mask=all-1:
        //   Q = K = V = hidden_states
        //   Head 0: q_h = [[1,2],[5,6]], k_h = [[1,2],[5,6]]
        //   scores = q @ k^T / sqrt(2) then softmax
        //   context = softmax(scores) @ v_h
        //
        // We don't need exact expected values -- we verify:
        // 1. Output has correct length
        // 2. Values are finite (no NaN/Inf from the SIMD path)
        // 3. Output is deterministic (running twice gives same result)
        assert_eq!(result.len(), seq_len * hidden_size);
        for (i, &val) in result.iter().enumerate() {
            assert!(val.is_finite(), "result[{i}] = {val} is not finite");
        }

        // Run again to verify determinism
        let mut buffers2 =
            AttentionBuffers::new(seq_len, hidden_size, num_heads, intermediate_size);
        let result2 = multi_head_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            hidden_size,
            num_heads,
            head_dim,
            &mut buffers2,
            &NoopLoraHook,
            0,
        );
        assert_eq!(result, result2, "attention must be deterministic");
    }

    /// Verify that masked positions are properly suppressed in attention.
    #[test]
    fn test_attention_mask_suppresses_tokens() {
        let seq_len = 3;
        let num_heads = 1;
        let head_dim = 2;
        let hidden_size = num_heads * head_dim; // 2

        let hidden_states = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let identity_2x2: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let zero_bias_2: Vec<f32> = vec![0.0; 2];
        let ones_2: Vec<f32> = vec![1.0; 2];

        let intermediate_size = hidden_size;

        let layer = TransformerLayerWeights {
            query_weight: Tensor2D {
                data: &identity_2x2,
                rows: hidden_size,
                cols: hidden_size,
            },
            query_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            key_weight: Tensor2D {
                data: &identity_2x2,
                rows: hidden_size,
                cols: hidden_size,
            },
            key_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            value_weight: Tensor2D {
                data: &identity_2x2,
                rows: hidden_size,
                cols: hidden_size,
            },
            value_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            attn_output_weight: Tensor2D {
                data: &identity_2x2,
                rows: hidden_size,
                cols: hidden_size,
            },
            attn_output_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            attn_layer_norm_weight: Tensor1D {
                data: &ones_2,
                len: hidden_size,
            },
            attn_layer_norm_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            ffn_intermediate_weight: Tensor2D {
                data: &identity_2x2,
                rows: intermediate_size,
                cols: hidden_size,
            },
            ffn_intermediate_bias: Tensor1D {
                data: &zero_bias_2,
                len: intermediate_size,
            },
            ffn_output_weight: Tensor2D {
                data: &identity_2x2,
                rows: hidden_size,
                cols: intermediate_size,
            },
            ffn_output_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            ffn_layer_norm_weight: Tensor1D {
                data: &ones_2,
                len: hidden_size,
            },
            ffn_layer_norm_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
        };

        // Mask out the third token
        let mask_all = vec![1u32, 1, 1];
        let mask_partial = vec![1u32, 1, 0];

        let mut buf1 = AttentionBuffers::new(seq_len, hidden_size, num_heads, intermediate_size);
        let mut buf2 = AttentionBuffers::new(seq_len, hidden_size, num_heads, intermediate_size);

        let result_all = multi_head_attention(
            &hidden_states,
            &layer,
            &mask_all,
            seq_len,
            hidden_size,
            num_heads,
            head_dim,
            &mut buf1,
            &NoopLoraHook,
            0,
        );
        let result_masked = multi_head_attention(
            &hidden_states,
            &layer,
            &mask_partial,
            seq_len,
            hidden_size,
            num_heads,
            head_dim,
            &mut buf2,
            &NoopLoraHook,
            0,
        );

        // With different masks, the outputs must differ
        assert_ne!(
            result_all, result_masked,
            "masking a token should change attention output"
        );
        // Both outputs must be finite
        for &v in result_all.iter().chain(result_masked.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn masked_token_value_does_not_leak_when_valid_score_below_sentinel() {
        // #361 live-path (standard.rs) regression. A masked key is excluded with -inf,
        // not a finite sentinel. Construct a row whose only VALID score sits below where
        // the old -10_000 sentinel lived: with the finite sentinel the masked key becomes
        // the softmax row max and its (large) value leaks into the output; with -inf the
        // valid key dominates and the masked value is suppressed. Reverting line 258 to
        // `-10_000.0` makes this fail (row-0 output jumps to the masked token's value).
        let seq_len = 2;
        let num_heads = 1;
        let head_dim = 2;
        let hidden_size = num_heads * head_dim; // 2

        // Token 0 carries a small value; token 1 (which we mask) carries a large value so
        // any leak is unmistakable.
        let hidden_states = vec![1.0, 0.0, 500.0, 500.0];

        // Distinct Q/K projections drive score[0][0] = Q_0·K_0·scale below -10_000:
        // Q_0 = [200,0], K_0 = [-100,0] -> -20000 * (1/sqrt(2)) ≈ -14142.
        let query_w: Vec<f32> = vec![200.0, 0.0, 0.0, 0.0];
        let key_w: Vec<f32> = vec![-100.0, 0.0, 0.0, 0.0];
        let identity_2x2: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let zero_bias_2: Vec<f32> = vec![0.0; 2];
        let ones_2: Vec<f32> = vec![1.0; 2];
        let intermediate_size = hidden_size;

        let layer = TransformerLayerWeights {
            query_weight: Tensor2D {
                data: &query_w,
                rows: hidden_size,
                cols: hidden_size,
            },
            query_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            key_weight: Tensor2D {
                data: &key_w,
                rows: hidden_size,
                cols: hidden_size,
            },
            key_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            value_weight: Tensor2D {
                data: &identity_2x2,
                rows: hidden_size,
                cols: hidden_size,
            },
            value_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            attn_output_weight: Tensor2D {
                data: &identity_2x2,
                rows: hidden_size,
                cols: hidden_size,
            },
            attn_output_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            attn_layer_norm_weight: Tensor1D {
                data: &ones_2,
                len: hidden_size,
            },
            attn_layer_norm_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            ffn_intermediate_weight: Tensor2D {
                data: &identity_2x2,
                rows: intermediate_size,
                cols: hidden_size,
            },
            ffn_intermediate_bias: Tensor1D {
                data: &zero_bias_2,
                len: intermediate_size,
            },
            ffn_output_weight: Tensor2D {
                data: &identity_2x2,
                rows: hidden_size,
                cols: intermediate_size,
            },
            ffn_output_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
            ffn_layer_norm_weight: Tensor1D {
                data: &ones_2,
                len: hidden_size,
            },
            ffn_layer_norm_bias: Tensor1D {
                data: &zero_bias_2,
                len: hidden_size,
            },
        };

        // Mask token 1 (the large-value token) for every query row.
        let mask = vec![1u32, 0];
        let mut buf = AttentionBuffers::new(seq_len, hidden_size, num_heads, intermediate_size);
        let out = multi_head_attention(
            &hidden_states,
            &layer,
            &mask,
            seq_len,
            hidden_size,
            num_heads,
            head_dim,
            &mut buf,
            &NoopLoraHook,
            0,
        );

        assert!(
            out.iter().all(|v| v.is_finite()),
            "output must be finite: {out:?}"
        );
        // Row 0 must reflect the VALID token's value (V_0 = [1,0]), not the masked
        // token's value (V_1 = [500,500]).
        assert!(
            out[0].abs() < 50.0 && out[1].abs() < 50.0,
            "masked token value leaked into row 0 output: {:?} (expected ~[1,0])",
            &out[0..2]
        );
    }

    #[test]
    fn standard_no_overflow_accepts_valid_shape() {
        // hidden_size == num_heads * head_dim, no product wraps.
        assert_standard_no_overflow(8, 64, 8, 8);
    }

    #[test]
    #[should_panic(expected = "hidden_size must equal num_heads * head_dim")]
    fn standard_no_overflow_rejects_layout_mismatch() {
        // hidden_size=4 but num_heads * head_dim = 2: the concat layout would
        // leave lanes 2..4 of every row stale before the output projection.
        assert_standard_no_overflow(1, 4, 1, 2);
    }

    #[test]
    #[should_panic(expected = "num_heads * seq_len * seq_len")]
    fn standard_no_overflow_rejects_wrapping_product() {
        // seq_len=2^32, num_heads=2, head_dim=1, hidden_size=2: every earlier
        // product fits, but num_heads * seq_len * seq_len = 2^65 wraps a 64-bit
        // usize to a small value that would feed an undersized scores slice.
        assert_standard_no_overflow(1usize << 32, 2, 2, 1);
    }

    /// Weights-free parity check between the packed `multi_head_attention_batched`
    /// (#677) and the single-sequence `multi_head_attention` path, run over
    /// synthetic (no model file) inputs so it exercises in default CI.
    ///
    /// Builds two sequences of different real length (2 tokens and 3 tokens),
    /// concatenated with no padding: `cu_seqlens = [0, 2, 5]`. If the
    /// per-sequence offset/length derived from `cu_seqlens` were wrong -- an
    /// off-by-one in `start`/`end`, or a swapped segment -- either the wrong
    /// slice of `q`/`k`/`v` would be attended over, or the wrong sequence's
    /// output would be compared below, and the parity check would fail.
    ///
    /// This check is mutation-sensitive: shifting sequence 1's `cu_seqlens`
    /// entry by `+1` (so its window silently absorbs one of sequence 0's rows)
    /// makes the parity assertion fail.
    ///
    /// Q, K, and V use DISTINCT scaled-identity weights and distinct per-dim
    /// biases (not a shared identity block): with Q == K == V, a swapped or
    /// shifted QKV split offset keeps
    /// the fused split self-consistent with the reference path, since both
    /// sides read the same values regardless of which third of `qkv` each
    /// buffer actually came from. With distinct Q/K/V this test additionally
    /// asserts the produced `q`/`k`/`v` scratch against an INDEPENDENT
    /// reference computed via a separate `matmul_bt` + `add_bias` per tensor
    /// (i.e. not through the fused split at all), which does catch that
    /// class: swapping the Q/K split offsets or shifting the V split by one
    /// hidden block flips the corresponding scratch buffer against a
    /// differently-scaled/biased reference and fails the assertion below.
    #[test]
    fn batched_attention_matches_single_sequence_per_row_packed() {
        let hidden_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let intermediate_size = hidden_size;

        let identity_8x8: Vec<f32> = {
            let mut m = vec![0.0f32; hidden_size * hidden_size];
            for i in 0..hidden_size {
                m[i * hidden_size + i] = 1.0;
            }
            m
        };
        let scaled_identity = |scale: f32| -> Vec<f32> {
            let mut m = vec![0.0f32; hidden_size * hidden_size];
            for i in 0..hidden_size {
                m[i * hidden_size + i] = scale;
            }
            m
        };
        let zero_bias_8: Vec<f32> = vec![0.0; hidden_size];
        let ones_8: Vec<f32> = vec![1.0; hidden_size];

        // Distinct Q/K/V projections: different diagonal scale AND different
        // per-dim bias, so Q != K != V and the split offsets are load-bearing.
        let query_w: Vec<f32> = scaled_identity(1.0);
        let key_w: Vec<f32> = scaled_identity(2.0);
        let value_w: Vec<f32> = scaled_identity(3.0);
        let query_bias_v: Vec<f32> = (0..hidden_size).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let key_bias_v: Vec<f32> = (0..hidden_size).map(|i| 1.0 + i as f32).collect();
        let value_bias_v: Vec<f32> = (0..hidden_size).map(|i| 10.0 + i as f32).collect();

        let mut fused_qkv_weight: Vec<f32> = Vec::with_capacity(3 * hidden_size * hidden_size);
        fused_qkv_weight.extend_from_slice(&query_w);
        fused_qkv_weight.extend_from_slice(&key_w);
        fused_qkv_weight.extend_from_slice(&value_w);
        let mut fused_qkv_bias: Vec<f32> = Vec::with_capacity(3 * hidden_size);
        fused_qkv_bias.extend_from_slice(&query_bias_v);
        fused_qkv_bias.extend_from_slice(&key_bias_v);
        fused_qkv_bias.extend_from_slice(&value_bias_v);

        let layer = TransformerLayerWeights {
            query_weight: Tensor2D {
                data: &query_w,
                rows: hidden_size,
                cols: hidden_size,
            },
            query_bias: Tensor1D {
                data: &query_bias_v,
                len: hidden_size,
            },
            key_weight: Tensor2D {
                data: &key_w,
                rows: hidden_size,
                cols: hidden_size,
            },
            key_bias: Tensor1D {
                data: &key_bias_v,
                len: hidden_size,
            },
            value_weight: Tensor2D {
                data: &value_w,
                rows: hidden_size,
                cols: hidden_size,
            },
            value_bias: Tensor1D {
                data: &value_bias_v,
                len: hidden_size,
            },
            attn_output_weight: Tensor2D {
                data: &identity_8x8,
                rows: hidden_size,
                cols: hidden_size,
            },
            attn_output_bias: Tensor1D {
                data: &zero_bias_8,
                len: hidden_size,
            },
            attn_layer_norm_weight: Tensor1D {
                data: &ones_8,
                len: hidden_size,
            },
            attn_layer_norm_bias: Tensor1D {
                data: &zero_bias_8,
                len: hidden_size,
            },
            ffn_intermediate_weight: Tensor2D {
                data: &identity_8x8,
                rows: intermediate_size,
                cols: hidden_size,
            },
            ffn_intermediate_bias: Tensor1D {
                data: &zero_bias_8,
                len: intermediate_size,
            },
            ffn_output_weight: Tensor2D {
                data: &identity_8x8,
                rows: intermediate_size,
                cols: hidden_size,
            },
            ffn_output_bias: Tensor1D {
                data: &zero_bias_8,
                len: hidden_size,
            },
            ffn_layer_norm_weight: Tensor1D {
                data: &ones_8,
                len: hidden_size,
            },
            ffn_layer_norm_bias: Tensor1D {
                data: &zero_bias_8,
                len: hidden_size,
            },
        };

        // Sequence 0: 2 real tokens, deterministic small values.
        let seq0_real: Vec<f32> = (0..2 * hidden_size).map(|i| 1.0 + i as f32 * 0.1).collect();

        // Sequence 1: 3 real tokens, deterministic small values distinct from
        // sequence 0.
        let seq1_real: Vec<f32> = (0..3 * hidden_size)
            .map(|i| 100.0 + i as f32 * 0.1)
            .collect();

        // Packed input: [seq0 tok0, seq0 tok1, seq1 tok0, seq1 tok1, seq1 tok2],
        // no padding rows anywhere. cu_seqlens marks sequence 0 as rows [0, 2)
        // and sequence 1 as rows [2, 5).
        let mut hidden_states_packed = Vec::with_capacity(5 * hidden_size);
        hidden_states_packed.extend_from_slice(&seq0_real);
        hidden_states_packed.extend_from_slice(&seq1_real);
        let cu_seqlens = vec![0usize, 2, 5];
        let total = *cu_seqlens.last().unwrap();

        let used_hidden = total * hidden_size;
        let mut q = vec![0.0f32; used_hidden];
        let mut k = vec![0.0f32; used_hidden];
        let mut v = vec![0.0f32; used_hidden];
        let mut qkv = vec![0.0f32; 3 * used_hidden];
        let mut concat = vec![0.0f32; used_hidden];
        let mut output = vec![0.0f32; used_hidden];
        let lora = RowShapeHook {
            row_width: hidden_size,
            calls: AtomicUsize::new(0),
        };

        multi_head_attention_batched(
            &hidden_states_packed,
            &layer,
            &fused_qkv_weight,
            &fused_qkv_bias,
            &cu_seqlens,
            hidden_size,
            num_heads,
            head_dim,
            &mut q,
            &mut k,
            &mut v,
            &mut qkv,
            &mut concat,
            &mut output,
            &lora,
            0,
        );

        assert_eq!(lora.calls.load(Ordering::Relaxed), total * 4);

        for out_val in output.iter() {
            assert!(
                out_val.is_finite(),
                "batched output must be finite: {output:?}"
            );
        }

        // Independent reference for the fused split: compute Q/K/V via a
        // separate matmul_bt + add_bias per tensor (bypassing the fused
        // matmul + split entirely) and compare against the scratch buffers
        // multi_head_attention_batched actually produced. This is
        // mutation-sensitive to the split offsets themselves -- swapping the
        // Q/K split range, or shifting the V split by one hidden block,
        // makes one of these three comparisons fail even though Q, K, and V
        // share the same identity-derived shape, because their scale and
        // bias differ.
        let mut expected_q = vec![0.0f32; used_hidden];
        matmul_bt(
            &hidden_states_packed,
            &query_w,
            &mut expected_q,
            total,
            hidden_size,
            hidden_size,
        );
        add_bias(&mut expected_q, &query_bias_v, hidden_size);
        for (i, (&got, &exp)) in q.iter().zip(expected_q.iter()).enumerate() {
            assert!(
                (got - exp).abs() <= 1e-5,
                "q scratch element {i} mismatch: batched={got} independent={exp}"
            );
        }

        let mut expected_k = vec![0.0f32; used_hidden];
        matmul_bt(
            &hidden_states_packed,
            &key_w,
            &mut expected_k,
            total,
            hidden_size,
            hidden_size,
        );
        add_bias(&mut expected_k, &key_bias_v, hidden_size);
        for (i, (&got, &exp)) in k.iter().zip(expected_k.iter()).enumerate() {
            assert!(
                (got - exp).abs() <= 1e-5,
                "k scratch element {i} mismatch: batched={got} independent={exp}"
            );
        }

        let mut expected_v = vec![0.0f32; used_hidden];
        matmul_bt(
            &hidden_states_packed,
            &value_w,
            &mut expected_v,
            total,
            hidden_size,
            hidden_size,
        );
        add_bias(&mut expected_v, &value_bias_v, hidden_size);
        for (i, (&got, &exp)) in v.iter().zip(expected_v.iter()).enumerate() {
            assert!(
                (got - exp).abs() <= 1e-5,
                "v scratch element {i} mismatch: batched={got} independent={exp}"
            );
        }

        // Sequence 0: compare the first 2 (real) rows of the batched output
        // against an independent single-sequence call on the 2 unpadded tokens.
        let mut buf0 = AttentionBuffers::new(2, hidden_size, num_heads, intermediate_size);
        let expected0 = multi_head_attention(
            &seq0_real,
            &layer,
            &[1u32, 1],
            2,
            hidden_size,
            num_heads,
            head_dim,
            &mut buf0,
            &NoopLoraHook,
            0,
        );
        let seq0_row_start = 0;
        let got0 = &output[seq0_row_start..seq0_row_start + 2 * hidden_size];
        for (i, (&g, &e)) in got0.iter().zip(expected0.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6,
                "seq0 row element {i} mismatch: batched={g} single={e}"
            );
        }

        // Sequence 1: compare all 3 (real) rows of the batched output against an
        // independent single-sequence call on the same 3 tokens.
        let mut buf1 = AttentionBuffers::new(3, hidden_size, num_heads, intermediate_size);
        let expected1 = multi_head_attention(
            &seq1_real,
            &layer,
            &[1u32, 1, 1],
            3,
            hidden_size,
            num_heads,
            head_dim,
            &mut buf1,
            &NoopLoraHook,
            0,
        );
        let seq1_row_start = 2 * hidden_size; // cu_seqlens[1] * hidden_size
        let got1 = &output[seq1_row_start..seq1_row_start + 3 * hidden_size];
        for (i, (&g, &e)) in got1.iter().zip(expected1.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6,
                "seq1 row element {i} mismatch: batched={g} single={e}"
            );
        }
    }
}
