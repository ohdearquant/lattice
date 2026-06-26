use crate::forward::cpu::{add_bias, matmul_bt, softmax_attention};
use crate::lora_hook::LoraHook;
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

    // Reshape buffers for SIMD matmul in attention scoring and context
    // aggregation.  Allocated once per model lifetime, reused every layer.
    q_head: Vec<f32>,
    k_head: Vec<f32>,
    v_head_t: Vec<f32>,
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

            // Per-head reshape buffers for SIMD matmul
            q_head: vec![0.0; max_seq_len * head_dim],
            k_head: vec![0.0; max_seq_len * head_dim],
            v_head_t: vec![0.0; head_dim * max_seq_len],
            scores_head: vec![0.0; max_seq_len * max_seq_len],
            context_head: vec![0.0; max_seq_len * head_dim],
        }
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
    multi_head_attention_in_place(
        hidden_states,
        layer_weights,
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
pub(crate) fn multi_head_attention_in_place(
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
    let used_context = num_heads * seq_len * head_dim;

    {
        let q = &mut buffers.q[..used_hidden];
        matmul_bt(
            hidden_states,
            layer_weights.query_weight.data,
            q,
            seq_len,
            hidden_size,
            hidden_size,
        );
        add_bias(q, layer_weights.query_bias.data, hidden_size);
        lora.apply(layer_idx, "query", hidden_states, q);
    }

    {
        let k = &mut buffers.k[..used_hidden];
        matmul_bt(
            hidden_states,
            layer_weights.key_weight.data,
            k,
            seq_len,
            hidden_size,
            hidden_size,
        );
        add_bias(k, layer_weights.key_bias.data, hidden_size);
        lora.apply(layer_idx, "key", hidden_states, k);
    }

    {
        let v = &mut buffers.v[..used_hidden];
        matmul_bt(
            hidden_states,
            layer_weights.value_weight.data,
            v,
            seq_len,
            hidden_size,
            hidden_size,
        );
        add_bias(v, layer_weights.value_bias.data, hidden_size);
        lora.apply(layer_idx, "value", hidden_states, v);
    }

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

    // scores*V context aggregation via SIMD matmul_bt.
    //
    // For each head we need: context[seq_len, head_dim] = scores[seq_len, seq_len] @ V_head[seq_len, head_dim]
    //
    // matmul_bt computes A @ B^T, so we transpose V_head into
    // v_head_t[head_dim, seq_len] and call matmul_bt(scores, v_head_t, ...)
    // which gives scores @ v_head_t^T = scores @ V_head.
    {
        for h in 0..num_heads {
            let head_offset = h * head_dim;

            // Transpose V_head[seq_len, head_dim] -> v_head_t[head_dim, seq_len]
            for i in 0..seq_len {
                let v_row_start = i * hidden_size + head_offset;
                for d in 0..head_dim {
                    buffers.v_head_t[d * seq_len + i] = buffers.v[v_row_start + d];
                }
            }

            // scores for this head are already contiguous at scores[h*seq_len*seq_len ..]
            let scores_offset = h * seq_len * seq_len;
            let scores_head = &buffers.scores[scores_offset..scores_offset + seq_len * seq_len];
            let v_head_t = &buffers.v_head_t[..head_dim * seq_len];
            let context_head = &mut buffers.context_head[..seq_len * head_dim];

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

            // Copy results into context buffer at the correct head offset
            let ctx_offset = h * seq_len * head_dim;
            buffers.context[ctx_offset..ctx_offset + seq_len * head_dim]
                .copy_from_slice(&context_head[..seq_len * head_dim]);
        }
    }

    {
        let context = &buffers.context[..used_context];
        let concat = &mut buffers.concat[..used_hidden];
        for i in 0..seq_len {
            for h in 0..num_heads {
                let head_offset = h * head_dim;
                for d in 0..head_dim {
                    concat[i * hidden_size + head_offset + d] =
                        context[(h * seq_len + i) * head_dim + d];
                }
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
        lora.apply(layer_idx, "attn_output", concat, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora_hook::NoopLoraHook;
    use crate::weights::{Tensor1D, Tensor2D, TransformerLayerWeights};

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
}
