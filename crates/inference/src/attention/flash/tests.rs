use super::*;
use crate::attention::{AttentionBuffers, multi_head_attention};
use crate::forward::cpu::{matmul_bt, softmax_attention};
use crate::lora_hook::NoopLoraHook;
use crate::weights::{Tensor1D, Tensor2D, TransformerLayerWeights};

/// Keeps `estimate_materialized_attention_buffer_bytes` honest against the
/// real `AttentionBuffers` field set (#678): the helper's
/// formula is hand-maintained and previously drifted (it kept counting a
/// removed `context`/per-head `v_head_t` allocation and omitted the newer
/// `qkv` and full-layer `v_all_t` scratch). Covers the concrete BGE-small
/// and BGE-base shapes used by the bench harness.
#[test]
fn estimate_matches_actual_attention_buffer_lengths() {
    let cases = [
        // (max_seq_len, hidden_size, num_heads, intermediate_size)
        (128, 384, 12, 1536), // BGE-small-en-v1.5
        (128, 768, 12, 3072), // BGE-base-en-v1.5
    ];
    for (max_seq_len, hidden_size, num_heads, intermediate_size) in cases {
        let buffers = AttentionBuffers::new(max_seq_len, hidden_size, num_heads, intermediate_size);
        let expected_bytes = buffers.total_scratch_len() * size_of::<f32>();
        let estimated_bytes = estimate_materialized_attention_buffer_bytes(
            max_seq_len,
            hidden_size,
            num_heads,
            intermediate_size,
        );
        assert_eq!(
            estimated_bytes, expected_bytes,
            "estimate drifted from AttentionBuffers::new for \
                 (max_seq_len={max_seq_len}, hidden_size={hidden_size}, \
                 num_heads={num_heads}, intermediate_size={intermediate_size})"
        );
    }
}

/// Legacy finite mask sentinel, retained only as the reference path's masking
/// convention. The kernel itself now masks with `-inf` (see #361); for the
/// in-distribution inputs these reference tests use, the two agree to within
/// the SIMD-softmax tolerance.
const MASK_VALUE: f32 = -10_000.0;

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

// NaN-honest max|a-b|: a `.fold(0.0, f32::max)` silently drops a NaN/Inf
// operand (IEEE maxNum keeps the non-NaN side), letting a catastrophically
// wrong output read as 0.0 and slip past a `<= tol` gate. Surface it instead.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if !d.is_finite() {
            return d;
        }
        if d > max {
            max = d;
        }
    }
    max
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
        &NoopLoraHook,
        0,
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

    let mut buffers = AttentionBuffers::new(seq_len, hidden_size, num_q_heads, intermediate_size);
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

    let mut buffers = AttentionBuffers::new(seq_len, hidden_size, num_q_heads, intermediate_size);
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
fn masked_key_below_finite_sentinel_does_not_dominate() {
    // #361 trigger. A valid score below the old finite sentinel (-10000) must not
    // let a masked key win the softmax.
    //   seq_len=2, head_dim=1, scale=1
    //   q=[-20000, 0], k=[1, 0], v=[1, 100], mask=[1, 0]  (key 1 masked)
    // Row 0: valid key-0 score = -20000, key 1 masked. The masked key's weight
    // must be exactly 0, so output[0] = v[0] = 1.0. The old finite sentinel made
    // -10000 the row max and output[0] = v[1] = 100.0 (the masked value wins).
    let head_dim = 1usize;
    let seq_len = 2usize;
    let scale = 1.0f32;
    let q = vec![-20000.0f32, 0.0];
    let k = vec![1.0f32, 0.0];
    let v = vec![1.0f32, 100.0];
    let mask = vec![1u32, 0u32];
    let config = TiledAttentionConfig::new(1, 1, head_dim);
    let mut buffers = TiledAttentionBuffers::new(seq_len, &config);
    let mut out = vec![0.0f32; seq_len * head_dim];
    tiled_attention_head(
        &q,
        &k,
        &v,
        &mut out,
        &mask,
        seq_len,
        head_dim,
        scale,
        &config,
        &mut buffers,
    );
    assert!(
        (out[0] - 1.0).abs() < 1e-6,
        "row 0 output={} expected 1.0 (masked key must not dominate; old sentinel gives 100.0)",
        out[0]
    );
    // Row 1 (query 0): key-0 score = 0 (valid), key 1 masked → output = v[0] = 1.0.
    assert!(
        (out[1] - 1.0).abs() < 1e-6,
        "row 1 output={} expected 1.0",
        out[1]
    );
}

#[test]
fn all_masked_row_yields_zero_output_not_nan() {
    // Degenerate guard: if every key is masked, the row max stays -inf. The kernel
    // must define a zero output (zero row_sum → zero-fill), never NaN from
    // exp(-inf - -inf).
    let head_dim = 2usize;
    let seq_len = 2usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q = vec![0.5f32, -0.5, 1.0, 2.0];
    let k = vec![0.3f32, 0.7, -0.2, 0.1];
    let v = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![0u32, 0u32];
    let config = TiledAttentionConfig::new(1, 1, head_dim);
    let mut buffers = TiledAttentionBuffers::new(seq_len, &config);
    let mut out = vec![0.0f32; seq_len * head_dim];
    tiled_attention_head(
        &q,
        &k,
        &v,
        &mut out,
        &mask,
        seq_len,
        head_dim,
        scale,
        &config,
        &mut buffers,
    );
    for (i, &o) in out.iter().enumerate() {
        assert!(o.is_finite(), "output[{i}]={o} must be finite (no NaN)");
        assert_eq!(o, 0.0, "output[{i}]={o} expected 0.0 for an all-masked row");
    }
}

#[test]
fn all_masked_multi_tile_row_yields_zero_output_not_nan() {
    // Single-tile coverage above does not exercise the cross-tile case where the
    // running `new_max` stays -inf across SEVERAL KV tiles. Force tile_size_kv=1
    // over a 3-key all-masked row so the `if new_max.is_finite()` guard is hit on
    // every tile; without it `exp(-inf - -inf)` poisons row_sum to NaN.
    let head_dim = 2usize;
    let seq_len = 3usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q = vec![0.5f32, -0.5, 1.0, 2.0, -1.0, 0.25];
    let k = vec![0.3f32, 0.7, -0.2, 0.1, 0.9, -0.4];
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mask = vec![0u32, 0u32, 0u32];
    let config = TiledAttentionConfig {
        tile_size_q: 1,
        tile_size_kv: 1,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim,
    };
    let mut buffers = TiledAttentionBuffers::new(seq_len, &config);
    let mut out = vec![0.0f32; seq_len * head_dim];
    tiled_attention_head(
        &q,
        &k,
        &v,
        &mut out,
        &mask,
        seq_len,
        head_dim,
        scale,
        &config,
        &mut buffers,
    );
    for (i, &o) in out.iter().enumerate() {
        assert!(o.is_finite(), "output[{i}]={o} must be finite (no NaN)");
        assert_eq!(o, 0.0, "output[{i}]={o} expected 0.0 for an all-masked row");
    }
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
#[ignore = "manual tiled-vs-materialized performance bench; run explicitly, not part of the correctness gate"]
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
                "{label:30} {seq_len:>6} {mat_ns:>10.0} ns {tiled_ns:>10.0} ns {speedup:>7.2}x"
            );
        }
    }
}

#[test]
#[should_panic(expected = "tiled scratch overflow")]
fn tile_config_scratch_overflow_is_rejected() {
    // Public `TiledAttentionConfig` fields let a caller pick a tile size that wraps
    // `tile_kv * head_dim` to 0; the guard must reject it before `ensure_capacity` allocates
    // an undersized scratch buffer. External head shapes are valid.
    let config = TiledAttentionConfig {
        tile_size_q: 2,
        tile_size_kv: 1usize << 63,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: 2,
    };
    let _buffers = TiledAttentionBuffers::new(2, &config);
}

#[test]
#[should_panic(expected = "tiled shape overflow")]
fn entry_shape_overflow_is_rejected_not_silently_accepted() {
    // `num_q_heads * head_dim` wraps to a small value; without the guard the
    // `hidden_size == num_q_heads * head_dim` release assert would accept an impossible head
    // layout and per-head extraction would index out of bounds. Exercised directly because the
    // only call site (`tiled_multi_head_attention_in_place`) needs a fully-populated
    // 16-tensor `TransformerLayerWeights`; the guard placement before the products is verified
    // by inspection/review.
    assert_flash_tiled_no_overflow((1usize << 63) + 1, 1, 2, 2, 2);
}
