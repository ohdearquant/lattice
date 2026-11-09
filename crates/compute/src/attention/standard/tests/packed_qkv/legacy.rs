use crate::attention::standard::{
    AttentionProjections, BatchedAttentionScratch, assert_standard_no_overflow,
};
use crate::forward::cpu::{add_bias, matmul_bt, softmax_attention};
use crate::lora_hook::{LoraHook, apply_lora_rows};

#[allow(clippy::too_many_arguments)]
pub(super) fn legacy_multi_head_attention_batched(
    hidden_states: &[f32],
    projections: &AttentionProjections<'_>,
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
    mut scratch: Option<BatchedAttentionScratch<'_>>,
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

        // One head's score matrix at a time (#1480): peak quadratic-in-seq_len
        // scratch is bounded by `seq_len^2`, independent of `num_heads`,
        // instead of allocating the full `[num_heads, seq_len, seq_len]`
        // buffer up front.
        let mut owned_q;
        let mut owned_k;
        let mut owned_v;
        let mut owned_scores;
        let mut owned_context;
        let (q_head, k_head, v_all_t, scores_head, context_head) =
            if let Some(scratch) = scratch.as_mut() {
                assert!(scratch.q_head.len() >= seq_len * head_dim);
                assert!(scratch.k_head.len() >= seq_len * head_dim);
                assert!(scratch.v_all_t.len() >= hidden_size * seq_len);
                assert!(scratch.scores_head.len() >= seq_len * seq_len);
                assert!(scratch.context_head.len() >= seq_len * head_dim);
                (
                    &mut scratch.q_head[..seq_len * head_dim],
                    &mut scratch.k_head[..seq_len * head_dim],
                    &mut scratch.v_all_t[..hidden_size * seq_len],
                    &mut scratch.scores_head[..seq_len * seq_len],
                    &mut scratch.context_head[..seq_len * head_dim],
                )
            } else {
                owned_q = vec![0.0f32; seq_len * head_dim];
                owned_k = vec![0.0f32; seq_len * head_dim];
                owned_v = vec![0.0f32; hidden_size * seq_len];
                owned_scores = vec![0.0f32; seq_len * seq_len];
                owned_context = vec![0.0f32; seq_len * head_dim];
                (
                    owned_q.as_mut_slice(),
                    owned_k.as_mut_slice(),
                    owned_v.as_mut_slice(),
                    owned_scores.as_mut_slice(),
                    owned_context.as_mut_slice(),
                )
            };

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
        projections.output_weight.data,
        output,
        total,
        hidden_size,
        hidden_size,
    );
    add_bias(output, projections.output_bias.data, hidden_size);
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
