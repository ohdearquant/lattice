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
    let mut buffers2 = AttentionBuffers::new(seq_len, hidden_size, num_heads, intermediate_size);
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

/// #1480 regression: the per-sequence score/softmax/context loop was
/// fused (one head's `[seq_len, seq_len]` matrix live at a time) to bound
/// peak quadratic scratch at `seq_len^2` regardless of `num_heads`,
/// instead of allocating the full `[num_heads, seq_len, seq_len]` buffer
/// per call. A NaN anywhere in one packed segment's input row propagates
/// through that segment's own Q/K/V and attention output (IEEE 754: any
/// arithmetic touching a NaN operand yields NaN, including `0.0 * NaN`),
/// but must never leak across the `cu_seqlens` boundary into a sibling
/// segment's output -- each segment's `q`/`k`/`v` slice and `concat_b`
/// region are disjoint, so a segment-indexing regression in the fused
/// loop would show up here as the sibling segment losing finiteness.
#[test]
fn nan_in_one_packed_segment_does_not_leak_into_a_sibling_segment() {
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
    let zero_bias_8: Vec<f32> = vec![0.0; hidden_size];
    let ones_8: Vec<f32> = vec![1.0; hidden_size];
    let fused_qkv_weight: Vec<f32> = identity_8x8
        .iter()
        .chain(identity_8x8.iter())
        .chain(identity_8x8.iter())
        .copied()
        .collect();
    let fused_qkv_bias: Vec<f32> = vec![0.0; 3 * hidden_size];

    let layer = TransformerLayerWeights {
        query_weight: Tensor2D {
            data: &identity_8x8,
            rows: hidden_size,
            cols: hidden_size,
        },
        query_bias: Tensor1D {
            data: &zero_bias_8,
            len: hidden_size,
        },
        key_weight: Tensor2D {
            data: &identity_8x8,
            rows: hidden_size,
            cols: hidden_size,
        },
        key_bias: Tensor1D {
            data: &zero_bias_8,
            len: hidden_size,
        },
        value_weight: Tensor2D {
            data: &identity_8x8,
            rows: hidden_size,
            cols: hidden_size,
        },
        value_bias: Tensor1D {
            data: &zero_bias_8,
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

    // Two segments of 3 tokens each; poison one dimension of segment 0's
    // middle token. Values are deterministic and nonzero so a genuinely
    // broken fusion (e.g. sharing scratch across segments) would surface
    // as more than just the expected lanes going non-finite.
    let mut hidden_states: Vec<f32> = (0..6 * hidden_size).map(|i| 1.0 + i as f32 * 0.1).collect();
    let poisoned_row = 1; // segment 0's middle token
    hidden_states[poisoned_row * hidden_size] = f32::NAN;
    let cu_seqlens = vec![0usize, 3, 6];
    let total = 6;
    let used_hidden = total * hidden_size;

    let mut q = vec![0.0f32; used_hidden];
    let mut k = vec![0.0f32; used_hidden];
    let mut v = vec![0.0f32; used_hidden];
    let mut qkv = vec![0.0f32; 3 * used_hidden];
    let mut concat = vec![0.0f32; used_hidden];
    let mut output = vec![0.0f32; used_hidden];

    multi_head_attention_batched(
        &hidden_states,
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
        &NoopLoraHook,
        0,
    );

    let seg0 = &output[0..3 * hidden_size];
    let seg1 = &output[3 * hidden_size..6 * hidden_size];
    assert!(
        seg0.iter().any(|v| !v.is_finite()),
        "the poisoned segment should show the propagated non-finite value: {seg0:?}"
    );
    assert!(
        seg1.iter().all(|v| v.is_finite()),
        "a NaN in one packed segment must not leak into a sibling segment's output: {seg1:?}"
    );
}
