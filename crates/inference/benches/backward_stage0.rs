use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::attention::gdn::GatedDeltaNetWeights;
use lattice_inference::attention::gdn_backward::{GdnSaved, gdn_backward, gdn_forward_save};
use lattice_inference::backward::attention_gqa::{AttnCache, gqa_backward, gqa_forward_with_cache};
use lattice_inference::backward::ops::{
    cross_entropy_backward, linear_vjp, lora_vjp, rmsnorm_backward, rope_backward, swiglu_backward,
};
use lattice_inference::model::qwen35_config::Qwen35Config;
use std::time::Duration;

const TINY: &str = "tiny";
const QWEN35_08B: &str = "qwen35_0_8b";

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn f32(&mut self, scale: f32) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1_u32 << 24) as f32;
        (2.0 * unit - 1.0) * scale
    }

    fn vector(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.f32(scale)).collect()
    }
}

fn bench_cross_entropy_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_stage0/cross_entropy_backward");
    for (name, seq_len, vocab_size, completion_start) in
        [(TINY, 4, 128, 2), (QWEN35_08B, 8, 248_320, 4)]
    {
        let mut rng = Rng::new(0xCE00 + vocab_size as u64);
        let logits = rng.vector(seq_len * vocab_size, 0.5);
        let targets: Vec<u32> = (0..seq_len)
            .map(|position| ((position * 7919 + 17) % vocab_size) as u32)
            .collect();
        group.throughput(Throughput::Elements(
            ((seq_len - completion_start) * vocab_size) as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                black_box(cross_entropy_backward(
                    black_box(&logits),
                    black_box(&targets),
                    seq_len,
                    vocab_size,
                    completion_start,
                ))
            });
        });
    }
    group.finish();
}

fn bench_linear_vjp(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_stage0/linear_vjp");
    for (name, d_in, d_out) in [(TINY, 64, 192), (QWEN35_08B, 1024, 3584)] {
        let mut rng = Rng::new(0x11EA + d_out as u64);
        let weights = rng.vector(d_out * d_in, 0.02);
        let grad = rng.vector(d_out, 0.1);
        group.throughput(Throughput::Elements((d_out * d_in) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                black_box(linear_vjp(
                    black_box(&weights),
                    black_box(&grad),
                    d_in,
                    d_out,
                ))
            });
        });
    }
    group.finish();
}

fn bench_lora_vjp(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_stage0/lora_vjp");
    for (name, rank, d_in, d_out) in [(TINY, 4, 64, 192), (QWEN35_08B, 16, 1024, 3584)] {
        let mut rng = Rng::new(0x10A0 + d_out as u64);
        let grad = rng.vector(d_out, 0.1);
        let x = rng.vector(d_in, 0.1);
        let h = rng.vector(rank, 0.1);
        let a = rng.vector(rank * d_in, 0.02);
        let b_weights = rng.vector(d_out * rank, 0.02);
        let scale = 1.0 / rank as f32;
        group.throughput(Throughput::Elements(
            (d_out * rank + rank * d_in + d_in) as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                black_box(lora_vjp(
                    black_box(&grad),
                    black_box(&x),
                    black_box(&h),
                    black_box(&a),
                    black_box(&b_weights),
                    rank,
                    d_in,
                    d_out,
                    scale,
                ))
            });
        });
    }
    group.finish();
}

fn bench_rmsnorm_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_stage0/rmsnorm_backward");
    for (name, dim) in [(TINY, 64), (QWEN35_08B, 1024)] {
        let mut rng = Rng::new(0xA11CE + dim as u64);
        let x = rng.vector(dim, 0.5);
        let weights: Vec<f32> = rng.vector(dim, 0.05).into_iter().map(|v| 1.0 + v).collect();
        let grad = rng.vector(dim, 0.1);
        let mean_square = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = (mean_square + 1e-6).sqrt().recip();
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                black_box(rmsnorm_backward(
                    black_box(&x),
                    black_box(&weights),
                    inv_rms,
                    black_box(&grad),
                ))
            });
        });
    }
    group.finish();
}

fn bench_rope_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_stage0/rope_backward");
    for (name, head_dim, rope_dim) in [(TINY, 64, 32), (QWEN35_08B, 256, 64)] {
        let mut rng = Rng::new(0xA0FE + head_dim as u64);
        let grad = rng.vector(head_dim, 0.1);
        let half = rope_dim / 2;
        let cos_vals: Vec<f32> = (0..half).map(|i| (i as f32 * 0.013).cos()).collect();
        let sin_vals: Vec<f32> = (0..half).map(|i| (i as f32 * 0.013).sin()).collect();
        group.throughput(Throughput::Elements(rope_dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                black_box(rope_backward(
                    black_box(&grad),
                    black_box(&cos_vals),
                    black_box(&sin_vals),
                    rope_dim,
                ))
            });
        });
    }
    group.finish();
}

fn bench_swiglu_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_stage0/swiglu_backward");
    for (name, hidden, inter) in [(TINY, 64, 192), (QWEN35_08B, 1024, 3584)] {
        let mut rng = Rng::new(0x5A16 + hidden as u64);
        let dy = rng.vector(hidden, 0.1);
        let gate_pre = rng.vector(inter, 0.5);
        let up_pre = rng.vector(inter, 0.5);
        let w_down = rng.vector(hidden * inter, 0.02);
        let w_gate = rng.vector(inter * hidden, 0.02);
        let w_up = rng.vector(inter * hidden, 0.02);
        group.throughput(Throughput::Elements((3 * hidden * inter) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |bench, _| {
            bench.iter(|| {
                black_box(swiglu_backward(
                    black_box(&dy),
                    black_box(&gate_pre),
                    black_box(&up_pre),
                    black_box(&w_down),
                    black_box(&w_gate),
                    black_box(&w_up),
                    hidden,
                    inter,
                ))
            });
        });
    }
    group.finish();
}

#[derive(Clone, Copy)]
struct GqaShape {
    name: &'static str,
    seq_len: usize,
    hidden: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    lora_rank: usize,
}

struct GqaFixture {
    dy: Vec<f32>,
    cache: AttnCache,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    q_norm_w: Vec<f32>,
    k_norm_w: Vec<f32>,
    lora_a_q: Vec<f32>,
    lora_b_q: Vec<f32>,
    lora_a_v: Vec<f32>,
    lora_b_v: Vec<f32>,
    lora_rank: usize,
    lora_scale: f32,
}

fn make_gqa_fixture(shape: GqaShape) -> GqaFixture {
    let q_dim = shape.num_q_heads * shape.head_dim;
    let kv_dim = shape.num_kv_heads * shape.head_dim;
    let half = shape.rope_dim / 2;
    let mut rng = Rng::new(0x6AA0 + shape.hidden as u64);
    let x = rng.vector(shape.seq_len * shape.hidden, 0.05);
    let w_q = rng.vector(2 * q_dim * shape.hidden, 0.01);
    let w_k = rng.vector(kv_dim * shape.hidden, 0.01);
    let w_v = rng.vector(kv_dim * shape.hidden, 0.01);
    let w_o = rng.vector(shape.hidden * q_dim, 0.01);
    let q_norm_w = vec![0.0; shape.head_dim];
    let k_norm_w = vec![0.0; shape.head_dim];
    let lora_a_q = rng.vector(shape.lora_rank * shape.hidden, 0.01);
    let lora_b_q = rng.vector(2 * q_dim * shape.lora_rank, 0.01);
    let lora_a_v = rng.vector(shape.lora_rank * shape.hidden, 0.01);
    let lora_b_v = rng.vector(kv_dim * shape.lora_rank, 0.01);
    let lora_scale = 1.0 / shape.lora_rank as f32;
    let cos_table: Vec<f32> = (0..shape.seq_len * half)
        .map(|i| {
            let position = i / half;
            let dimension = i % half;
            let theta = position as f32
                / 10_000_000_f32.powf(2.0 * dimension as f32 / shape.rope_dim as f32);
            theta.cos()
        })
        .collect();
    let sin_table: Vec<f32> = (0..shape.seq_len * half)
        .map(|i| {
            let position = i / half;
            let dimension = i % half;
            let theta = position as f32
                / 10_000_000_f32.powf(2.0 * dimension as f32 / shape.rope_dim as f32);
            theta.sin()
        })
        .collect();
    let (_, cache) = gqa_forward_with_cache(
        &x,
        &w_q,
        &w_k,
        &w_v,
        &w_o,
        &q_norm_w,
        &k_norm_w,
        Some(&lora_a_q),
        Some(&lora_b_q),
        Some(&lora_a_v),
        Some(&lora_b_v),
        shape.lora_rank,
        lora_scale,
        shape.seq_len,
        shape.hidden,
        shape.num_q_heads,
        shape.num_kv_heads,
        shape.head_dim,
        shape.rope_dim,
        &cos_table,
        &sin_table,
        1e-6,
    );

    GqaFixture {
        dy: rng.vector(shape.seq_len * shape.hidden, 0.1),
        cache,
        w_q,
        w_k,
        w_v,
        w_o,
        q_norm_w,
        k_norm_w,
        lora_a_q,
        lora_b_q,
        lora_a_v,
        lora_b_v,
        lora_rank: shape.lora_rank,
        lora_scale,
    }
}

fn bench_gqa_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_stage0/gqa_backward");
    for shape in [
        GqaShape {
            name: TINY,
            seq_len: 4,
            hidden: 64,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            rope_dim: 8,
            lora_rank: 4,
        },
        GqaShape {
            name: QWEN35_08B,
            seq_len: 16,
            hidden: 1024,
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 256,
            rope_dim: 64,
            lora_rank: 16,
        },
    ] {
        let fixture = make_gqa_fixture(shape);
        group.throughput(Throughput::Elements((shape.seq_len * shape.hidden) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(shape.name),
            &shape.name,
            |bench, _| {
                bench.iter(|| {
                    black_box(gqa_backward(
                        black_box(&fixture.dy),
                        black_box(&fixture.cache),
                        black_box(&fixture.w_q),
                        black_box(&fixture.w_k),
                        black_box(&fixture.w_v),
                        black_box(&fixture.w_o),
                        black_box(&fixture.q_norm_w),
                        black_box(&fixture.k_norm_w),
                        Some(black_box(&fixture.lora_a_q)),
                        Some(black_box(&fixture.lora_b_q)),
                        Some(black_box(&fixture.lora_a_v)),
                        Some(black_box(&fixture.lora_b_v)),
                        fixture.lora_rank,
                        fixture.lora_scale,
                    ))
                });
            },
        );
    }
    group.finish();
}

#[derive(Clone, Copy)]
struct GdnShape {
    name: &'static str,
    seq_len: usize,
    hidden: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel_size: usize,
    lora_rank: usize,
}

struct GdnFixture {
    grad_outputs: Vec<f32>,
    saved: GdnSaved,
    weights: GatedDeltaNetWeights,
}

fn make_gdn_weights(shape: GdnShape, rng: &mut Rng) -> GatedDeltaNetWeights {
    let qkv_dim = 2 * shape.num_key_heads * shape.key_dim + shape.num_value_heads * shape.value_dim;
    let output_dim = shape.num_value_heads * shape.value_dim;
    GatedDeltaNetWeights {
        in_proj_qkv: rng.vector(qkv_dim * shape.hidden, 0.01),
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: shape.hidden,
        in_proj_z: rng.vector(output_dim * shape.hidden, 0.01),
        in_proj_z_rows: output_dim,
        in_proj_z_cols: shape.hidden,
        in_proj_b: rng.vector(shape.num_value_heads * shape.hidden, 0.01),
        in_proj_b_rows: shape.num_value_heads,
        in_proj_b_cols: shape.hidden,
        in_proj_a: rng.vector(shape.num_value_heads * shape.hidden, 0.01),
        in_proj_a_rows: shape.num_value_heads,
        in_proj_a_cols: shape.hidden,
        a_log: vec![-0.5; shape.num_value_heads],
        dt_bias: vec![0.1; shape.num_value_heads],
        conv1d_weight: rng.vector(qkv_dim * shape.kernel_size, 0.01),
        conv_dim: qkv_dim,
        kernel_size: shape.kernel_size,
        norm_weight: vec![1.0; shape.value_dim],
        out_proj: rng.vector(shape.hidden * output_dim, 0.01),
        out_proj_rows: shape.hidden,
        out_proj_cols: output_dim,
    }
}

fn make_gdn_fixture(shape: GdnShape) -> GdnFixture {
    let qkv_dim = 2 * shape.num_key_heads * shape.key_dim + shape.num_value_heads * shape.value_dim;
    let output_dim = shape.num_value_heads * shape.value_dim;
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.hidden_size = shape.hidden;
    cfg.linear_num_key_heads = shape.num_key_heads;
    cfg.linear_num_value_heads = Some(shape.num_value_heads);
    cfg.linear_key_head_dim = shape.key_dim;
    cfg.linear_value_head_dim = shape.value_dim;
    cfg.linear_conv_kernel_dim = shape.kernel_size;
    let mut rng = Rng::new(0x6D00 + shape.hidden as u64);
    let weights = make_gdn_weights(shape, &mut rng);
    let inputs = rng.vector(shape.seq_len * shape.hidden, 0.05);
    let mut saved = GdnSaved::new(
        shape.seq_len,
        shape.num_key_heads,
        shape.num_value_heads,
        shape.key_dim,
        shape.value_dim,
        shape.hidden,
        qkv_dim,
        output_dim,
        shape.kernel_size,
        1.0 / (shape.key_dim as f32).sqrt(),
        cfg.rms_norm_eps,
    );
    let mut outputs = vec![0.0; shape.seq_len * shape.hidden];
    let lora_a_qkv = rng.vector(shape.lora_rank * shape.hidden, 0.01);
    let lora_b_qkv = rng.vector(qkv_dim * shape.lora_rank, 0.01);
    let lora_a_z = rng.vector(shape.lora_rank * shape.hidden, 0.01);
    let lora_b_z = rng.vector(output_dim * shape.lora_rank, 0.01);
    let lora_a_b = rng.vector(shape.lora_rank * shape.hidden, 0.01);
    let lora_b_b = rng.vector(shape.num_value_heads * shape.lora_rank, 0.01);
    let lora_a_a = rng.vector(shape.lora_rank * shape.hidden, 0.01);
    let lora_b_a = rng.vector(shape.num_value_heads * shape.lora_rank, 0.01);
    let lora_a_out = rng.vector(shape.lora_rank * output_dim, 0.01);
    let lora_b_out = rng.vector(shape.hidden * shape.lora_rank, 0.01);
    gdn_forward_save(
        &inputs,
        &weights,
        &cfg,
        &mut saved,
        &mut outputs,
        Some(&lora_a_qkv),
        Some(&lora_b_qkv),
        Some(&lora_a_z),
        Some(&lora_b_z),
        Some(&lora_a_b),
        Some(&lora_b_b),
        Some(&lora_a_a),
        Some(&lora_b_a),
        Some(&lora_a_out),
        Some(&lora_b_out),
        shape.lora_rank,
        1.0 / shape.lora_rank as f32,
    );

    GdnFixture {
        grad_outputs: rng.vector(shape.seq_len * shape.hidden, 0.1),
        saved,
        weights,
    }
}

fn bench_gdn_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_stage0/gdn_backward");
    for shape in [
        GdnShape {
            name: TINY,
            seq_len: 4,
            hidden: 64,
            num_key_heads: 2,
            num_value_heads: 4,
            key_dim: 16,
            value_dim: 16,
            kernel_size: 4,
            lora_rank: 4,
        },
        GdnShape {
            name: QWEN35_08B,
            seq_len: 8,
            hidden: 1024,
            num_key_heads: 16,
            num_value_heads: 16,
            key_dim: 128,
            value_dim: 128,
            kernel_size: 4,
            lora_rank: 16,
        },
    ] {
        let fixture = make_gdn_fixture(shape);
        group.throughput(Throughput::Elements((shape.seq_len * shape.hidden) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(shape.name),
            &shape.name,
            |bench, _| {
                bench.iter(|| {
                    black_box(gdn_backward(
                        black_box(&fixture.grad_outputs),
                        black_box(&fixture.saved),
                        black_box(&fixture.weights),
                    ))
                });
            },
        );
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .noise_threshold(0.03)
}

criterion_group! {
    name = backward_stage0;
    config = criterion_config();
    targets =
        bench_cross_entropy_backward,
        bench_linear_vjp,
        bench_lora_vjp,
        bench_rmsnorm_backward,
        bench_rope_backward,
        bench_swiglu_backward,
        bench_gqa_backward,
        bench_gdn_backward
}
criterion_main!(backward_stage0);
