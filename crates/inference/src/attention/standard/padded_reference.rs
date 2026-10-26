//! Test-only padded batched attention reference for the BERT packed-batch parity test.

use crate::forward::cpu::{add_bias, matmul_bt, softmax_attention};
use crate::lora_hook::{LoraHook, apply_lora_rows};
use crate::weights::TransformerLayerWeights;

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
    lattice_compute::__internal::assert_standard_no_overflow(
        seq_len,
        hidden_size,
        num_heads,
        head_dim,
    );
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
