use lattice_inference::attention::flash::{
    TiledAttentionBuffers, TiledAttentionConfig, estimate_materialized_attention_buffer_bytes,
    tiled_multi_head_attention,
};
use lattice_inference::attention::standard::{AttentionBuffers, multi_head_attention};
use lattice_inference::weights::{Tensor1D, Tensor2D, TransformerLayerWeights};
use std::hint::black_box;
use std::time::Instant;

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

fn iterations_for(seq_len: usize) -> usize {
    match seq_len {
        0..=64 => 24,
        65..=128 => 12,
        129..=256 => 6,
        _ => 3,
    }
}

fn average_ms<F>(iters: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() * 1_000.0 / iters as f64
}

fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / MB)
    } else {
        format!("{:.1} KB", bytes as f64 / KB)
    }
}

struct ModelCase {
    name: &'static str,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
}

fn run_case(model: &ModelCase, seq_len: usize) {
    let mut rng = Lcg::new((seq_len as u64) ^ ((model.hidden_size as u64) << 32));
    let hidden_states = random_vec(seq_len * model.hidden_size, &mut rng);
    let attention_mask = patterned_mask(seq_len);
    let owned_layer = build_test_layer(
        model.hidden_size,
        model.num_heads,
        model.num_heads,
        model.head_dim,
        model.intermediate_size,
        &mut rng,
    );
    let layer = owned_layer.borrowed();

    let config = TiledAttentionConfig::new(model.num_heads, model.num_heads, model.head_dim);
    let mut materialized_buffers = AttentionBuffers::new(
        seq_len,
        model.hidden_size,
        model.num_heads,
        model.intermediate_size,
    );
    let mut tiled_attention_buffers = AttentionBuffers::new(
        seq_len,
        model.hidden_size,
        model.num_heads,
        model.intermediate_size,
    );
    let mut tiled_buffers = TiledAttentionBuffers::new(seq_len, &config);

    let reference = multi_head_attention(
        &hidden_states,
        &layer,
        &attention_mask,
        seq_len,
        model.hidden_size,
        model.num_heads,
        model.head_dim,
        &mut materialized_buffers,
    );
    let tiled_once = tiled_multi_head_attention(
        &hidden_states,
        &layer,
        &attention_mask,
        seq_len,
        model.hidden_size,
        model.num_heads,
        model.num_heads,
        model.head_dim,
        &mut tiled_attention_buffers,
        &mut tiled_buffers,
        &config,
    );
    let max_diff = max_abs_diff(&reference, &tiled_once);
    assert!(
        max_diff <= 1e-4,
        "{} seq_len={} max_abs_diff={}",
        model.name,
        seq_len,
        max_diff
    );

    // Warm-up.
    let _ = black_box(multi_head_attention(
        &hidden_states,
        &layer,
        &attention_mask,
        seq_len,
        model.hidden_size,
        model.num_heads,
        model.head_dim,
        &mut materialized_buffers,
    ));
    let _ = black_box(tiled_multi_head_attention(
        &hidden_states,
        &layer,
        &attention_mask,
        seq_len,
        model.hidden_size,
        model.num_heads,
        model.num_heads,
        model.head_dim,
        &mut tiled_attention_buffers,
        &mut tiled_buffers,
        &config,
    ));

    let iters = iterations_for(seq_len);
    let materialized_ms = average_ms(iters, || {
        let output = multi_head_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            model.hidden_size,
            model.num_heads,
            model.head_dim,
            &mut materialized_buffers,
        );
        black_box(output);
    });
    let tiled_ms = average_ms(iters, || {
        let output = tiled_multi_head_attention(
            &hidden_states,
            &layer,
            &attention_mask,
            seq_len,
            model.hidden_size,
            model.num_heads,
            model.num_heads,
            model.head_dim,
            &mut tiled_attention_buffers,
            &mut tiled_buffers,
            &config,
        );
        black_box(output);
    });

    let speedup = materialized_ms / tiled_ms;
    let materialized_scores_bytes =
        model.num_heads * seq_len * seq_len * std::mem::size_of::<f32>();
    let tiled_scratch_bytes = tiled_buffers.allocated_bytes();
    let legacy_buffers_bytes = estimate_materialized_attention_buffer_bytes(
        seq_len,
        model.hidden_size,
        model.num_heads,
        model.intermediate_size,
    );

    println!(
        "{:<14} {:>6} {:>4}x{:<4} {:>12.3} {:>12.3} {:>8.2} {:>14} {:>14} {:>14}",
        model.name,
        seq_len,
        config.tile_size_q,
        config.tile_size_kv,
        materialized_ms,
        tiled_ms,
        speedup,
        format_bytes(materialized_scores_bytes),
        format_bytes(tiled_scratch_bytes),
        format_bytes(legacy_buffers_bytes),
    );
}

fn run_benchmark() {
    let models = [
        ModelCase {
            name: "bge-small",
            hidden_size: 384,
            num_heads: 12,
            head_dim: 32,
            intermediate_size: 1536,
        },
        ModelCase {
            name: "bge-base",
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_size: 3072,
        },
    ];
    let seq_lens = [64usize, 128, 256, 512];

    println!(
        "{:<14} {:>6} {:>9} {:>12} {:>12} {:>8} {:>14} {:>14} {:>14}",
        "model",
        "seq",
        "tile",
        "materialized",
        "tiled",
        "speedup",
        "mat_scores",
        "tile_scratch",
        "legacy_bufs",
    );
    println!(
        "{:<14} {:>6} {:>9} {:>12} {:>12} {:>8} {:>14} {:>14} {:>14}",
        "--------------",
        "------",
        "---------",
        "------------",
        "------------",
        "--------",
        "--------------",
        "--------------",
        "--------------",
    );

    for model in &models {
        for &seq_len in &seq_lens {
            run_case(model, seq_len);
        }
    }
}

#[test]
#[ignore]
fn attention_benchmark() {
    run_benchmark();
}
