use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::attention::decode::{decode_attention, decode_attention_scores};
use lattice_inference::attention::gqa::GqaConfig;

const HEAD_DIM: usize = 128;
const KV_SEQ_LENS: [usize; 3] = [128, 512, 2048];

#[derive(Clone, Copy, Debug)]
struct AttentionCase {
    name: &'static str,
    num_heads: usize,
    num_kv_heads: usize,
}

const CASES: [AttentionCase; 3] = [
    AttentionCase {
        name: "mha_q32_kv32",
        num_heads: 32,
        num_kv_heads: 32,
    },
    AttentionCase {
        name: "mha_q8_kv8",
        num_heads: 8,
        num_kv_heads: 8,
    },
    AttentionCase {
        name: "gqa_q32_kv8",
        num_heads: 32,
        num_kv_heads: 8,
    },
];

/// Deterministic decode-attention fixture for Criterion benchmarks.
///
/// Shapes:
/// - `q`: `[num_heads * HEAD_DIM]`
/// - `k`: `[kv_seq_len, num_kv_heads * HEAD_DIM]`
/// - `v`: `[kv_seq_len, num_kv_heads * HEAD_DIM]`
/// - `scores`: `[num_heads, kv_seq_len]`
/// - `output`: `[num_heads * HEAD_DIM]`
struct DecodeAttentionFixture {
    cfg: GqaConfig,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
    output: Vec<f32>,
    kv_seq_len: usize,
}

fn tensor_value(kind: u32, i: usize) -> f32 {
    let mut x = (i as u32)
        .wrapping_mul(1_664_525)
        .wrapping_add(1_013_904_223 ^ kind.wrapping_mul(0x9E37_79B9));
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    let unit = ((x & 0x00FF_FFFF) as f32) / 16_777_216.0;
    (unit - 0.5) * 0.125
}

fn build_q(num_heads: usize) -> Vec<f32> {
    // Flat shape: [num_heads * HEAD_DIM] (q_seq_len = 1).
    (0..num_heads * HEAD_DIM)
        .map(|i| tensor_value(1, i))
        .collect()
}

fn build_kv(kind: u32, num_kv_heads: usize, kv_seq_len: usize) -> Vec<f32> {
    // Logical shape: [n_kv_heads, kv_seq_len, HEAD_DIM].
    // Flat shape:    [kv_seq_len, n_kv_heads * HEAD_DIM].
    let kv_dim = num_kv_heads * HEAD_DIM;
    let mut buffer = vec![0.0f32; kv_seq_len * kv_dim];
    for kv_h in 0..num_kv_heads {
        for pos in 0..kv_seq_len {
            for d in 0..HEAD_DIM {
                let logical_index = (kv_h * kv_seq_len + pos) * HEAD_DIM + d;
                let flat_index = pos * kv_dim + kv_h * HEAD_DIM + d;
                buffer[flat_index] = tensor_value(kind, logical_index);
            }
        }
    }
    buffer
}

impl DecodeAttentionFixture {
    fn new(case: AttentionCase, kv_seq_len: usize) -> Self {
        let cfg = GqaConfig {
            num_heads: case.num_heads,
            num_kv_heads: case.num_kv_heads,
            head_dim: HEAD_DIM,
        };
        let q = build_q(case.num_heads);
        let k = build_kv(2, case.num_kv_heads, kv_seq_len);
        let v = build_kv(3, case.num_kv_heads, kv_seq_len);
        let scores = vec![0.0f32; case.num_heads * kv_seq_len];
        let output = vec![0.0f32; case.num_heads * HEAD_DIM];

        Self {
            cfg,
            q,
            k,
            v,
            scores,
            output,
            kv_seq_len,
        }
    }

    fn kv_bytes_per_decode_step(&self) -> u64 {
        (2 * self.kv_seq_len * self.cfg.kv_dim() * std::mem::size_of::<f32>()) as u64
    }
}

fn bench_decode_scores(c: &mut Criterion) {
    let mut group = c.benchmark_group("attn_opt_scores");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    for &kv_seq_len in &KV_SEQ_LENS {
        for case in CASES {
            let mut fixture = DecodeAttentionFixture::new(case, kv_seq_len);
            group.throughput(Throughput::Bytes(
                (fixture.k.len() * std::mem::size_of::<f32>()) as u64,
            ));
            group.bench_with_input(
                BenchmarkId::new(case.name, kv_seq_len),
                &kv_seq_len,
                |b, &kv_len| {
                    b.iter(|| {
                        decode_attention_scores(
                            black_box(fixture.q.as_slice()),
                            black_box(fixture.k.as_slice()),
                            black_box(fixture.scores.as_mut_slice()),
                            kv_len,
                            fixture.cfg,
                            kv_len,
                        );
                        black_box(fixture.scores.as_slice());
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_decode_full_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attn_opt_full_attention");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    for &kv_seq_len in &KV_SEQ_LENS {
        for case in CASES {
            let mut fixture = DecodeAttentionFixture::new(case, kv_seq_len);
            group.throughput(Throughput::Bytes(fixture.kv_bytes_per_decode_step()));
            group.bench_with_input(
                BenchmarkId::new(case.name, kv_seq_len),
                &kv_seq_len,
                |b, &kv_len| {
                    b.iter(|| {
                        decode_attention(
                            black_box(fixture.q.as_slice()),
                            black_box(fixture.k.as_slice()),
                            black_box(fixture.v.as_slice()),
                            black_box(fixture.output.as_mut_slice()),
                            black_box(fixture.scores.as_mut_slice()),
                            kv_len,
                            fixture.cfg,
                            kv_len,
                        );
                        black_box(fixture.output.as_slice());
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_decode_scores, bench_decode_full_attention);
criterion_main!(benches);
