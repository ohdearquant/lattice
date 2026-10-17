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
    // before any per-head work, so `apply_lora_rows` can still hand
    // `LoraHook::apply` one row at a time (`x`: one input row, `output`: the
    // corresponding `[hidden_size]` output row) exactly as it does for every
    // other projection.
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
    /// field set (see PR #678).
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
        // Per-sequence overflow guard: hidden_size == num_heads * head_dim and
        // seq_len * hidden_size fit a usize. The score scratch below is now a
        // single head's [seq_len, seq_len] matrix rather than the full
        // [num_heads, seq_len, seq_len] buffer (#1480), but `num_heads >= 1`
        // means `num_heads * seq_len * seq_len` not overflowing (checked here)
        // still dominates the single-head `seq_len * seq_len` product.
        assert_standard_no_overflow(seq_len, hidden_size, num_heads, head_dim);

        let row_start = start * hidden_size;
        let concat_b = &mut concat[row_start..row_start + seq_len * hidden_size];

        let mut q_head = vec![0.0f32; seq_len * head_dim];
        let mut k_head = vec![0.0f32; seq_len * head_dim];
        let mut v_all_t = vec![0.0f32; hidden_size * seq_len];
        // One head's score matrix at a time (#1480): peak quadratic-in-seq_len
        // scratch is bounded by `seq_len^2`, independent of `num_heads`,
        // instead of allocating the full `[num_heads, seq_len, seq_len]`
        // buffer up front.
        let mut scores_head = vec![0.0f32; seq_len * seq_len];
        let mut context_head = vec![0.0f32; seq_len * head_dim];

        // Transpose V once for this sequence (#673 acceptable-minimum), ahead
        // of the fused per-head loop below: one [hidden_size, seq_len]
        // transpose instead of `num_heads` separate [head_dim, seq_len]
        // transposes; identical element count moved, one loop instead of
        // `num_heads` smaller loops. Moving it ahead of the head loop doesn't
        // change what it computes: it reads only from `v` and writes only to
        // `v_all_t`, independent of `scores_head`.
        for i in 0..seq_len {
            let v_row_start = row_start + i * hidden_size;
            for d in 0..hidden_size {
                v_all_t[d * seq_len + i] = v[v_row_start + d];
            }
        }

        // Fused per-head score/softmax/context (#1480): compute one head's
        // scores, scale, softmax, and context aggregation before moving on to
        // the next head, so only one `[seq_len, seq_len]` matrix is live at a
        // time instead of `num_heads` of them. The scale multiply keeps the
        // same op order as before fusion (matmul_bt, then `score * scale`
        // elementwise), and `softmax_attention` is called with `num_heads=1`
        // over this head's own scratch, which is bit-identical to the row
        // range that head occupied inside the old `[num_heads, seq_len,
        // seq_len]` buffer: `softmax_attention` operates per-row within a
        // head with no cross-head coupling (`forward/cpu/softmax.rs`'s
        // scalar, NEON, and AVX2 paths all index rows as
        // `(h * seq_len + s) * seq_len`, so head 0 of a 1-head call reads
        // exactly the row range head `h` occupied before). No masking: every
        // row in this sequence's packed region is real, so softmax runs over
        // the raw scaled scores directly (still through the same fail-closed
        // `softmax_attention` kernel as every other path).
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
            for score in scores_head[..seq_len * seq_len].iter_mut() {
                *score *= scale;
            }

            softmax_attention(&mut scores_head[..seq_len * seq_len], seq_len, 1);

            let v_head_t = &v_all_t[head_offset * seq_len..(head_offset + head_dim) * seq_len];
            matmul_bt(
                &scores_head[..seq_len * seq_len],
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

// -----------------------------------------------------------------------
// Bench-only support module
// -----------------------------------------------------------------------

/// Bench-only re-export of [`multi_head_attention_batched`], which stays
/// `pub(crate)` outside `bench-internals` builds. Same visibility discipline
/// as `weights::bench_support` and `forward::neon_forward::bench_support`:
/// the default public API is unchanged, and only a `--features
/// bench-internals` build can reach the packed-batch attention path directly
/// from a Criterion bench.
#[cfg(feature = "bench-internals")]
pub mod bench_support {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    pub fn multi_head_attention_batched(
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
        super::multi_head_attention_batched(
            hidden_states,
            layer_weights,
            fused_qkv_weight,
            fused_qkv_bias,
            cu_seqlens,
            hidden_size,
            num_heads,
            head_dim,
            q,
            k,
            v,
            qkv,
            concat,
            output,
            lora,
            layer_idx,
        )
    }
}

#[cfg(test)]
mod tests;
