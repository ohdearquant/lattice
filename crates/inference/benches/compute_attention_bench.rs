use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::attention::gqa::GqaConfig;
use lattice_inference::generate::bench_support::compute_attention_for_bench;

const BATCH: usize = 1;
const Q_SEQ_LEN: usize = 1;
const NUM_HEADS: usize = 28;
const HEAD_DIM: usize = 128;
const KV_SEQ_LENS: [usize; 4] = [128, 512, 1024, 2048];

#[derive(Clone, Copy, Debug)]
struct AttentionCase {
    name: &'static str,
    num_kv_heads: usize,
    groups: usize,
}

const CASES: [AttentionCase; 2] = [
    AttentionCase {
        name: "gqa_h28_kv4_groups7",
        num_kv_heads: 4,
        groups: 7,
    },
    AttentionCase {
        name: "control_h28_kv28_groups1",
        num_kv_heads: 28,
        groups: 1,
    },
];

#[derive(Debug)]
struct DecodeFixture {
    cfg: GqaConfig,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    output: Vec<f32>,
    scores: Vec<f32>,
    kv_seq_len: usize,
}

impl DecodeFixture {
    fn new(case: AttentionCase, kv_seq_len: usize) -> Self {
        let cfg = GqaConfig {
            num_heads: NUM_HEADS,
            num_kv_heads: case.num_kv_heads,
            head_dim: HEAD_DIM,
        };
        assert_eq!(cfg.groups(), case.groups);

        let q = build_q(NUM_HEADS, HEAD_DIM);
        let k = build_kv(2, case.num_kv_heads, kv_seq_len, HEAD_DIM);
        let v = build_kv(3, case.num_kv_heads, kv_seq_len, HEAD_DIM);
        let output = vec![0.0f32; Q_SEQ_LEN * NUM_HEADS * HEAD_DIM];
        let scores = vec![0.0f32; NUM_HEADS * kv_seq_len];

        Self {
            cfg,
            q,
            k,
            v,
            output,
            scores,
            kv_seq_len,
        }
    }

    fn kv_bytes_per_decode_step(&self) -> u64 {
        ((self.k.len() + self.v.len()) * std::mem::size_of::<f32>()) as u64
    }
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

fn build_q(num_heads: usize, head_dim: usize) -> Vec<f32> {
    // Logical Q shape: [batch = 1, n_heads, q_seq_len = 1, head_dim].
    // compute_attention flat Q shape: [q_seq_len, n_heads * head_dim].
    let mut q = vec![0.0f32; BATCH * num_heads * Q_SEQ_LEN * head_dim];
    for batch in 0..BATCH {
        for h in 0..num_heads {
            for d in 0..head_dim {
                let logical_index = (((batch * num_heads + h) * Q_SEQ_LEN) * head_dim) + d;
                let flat_index = h * head_dim + d;
                debug_assert_eq!(logical_index, flat_index);
                q[flat_index] = tensor_value(1, logical_index);
            }
        }
    }
    q
}

fn build_kv(kind: u32, num_kv_heads: usize, kv_seq_len: usize, head_dim: usize) -> Vec<f32> {
    // Logical K/V shape: [batch = 1, n_kv_heads, kv_seq_len, head_dim].
    // compute_attention flat K/V shape: [kv_seq_len, n_kv_heads * head_dim].
    let mut buffer = vec![0.0f32; BATCH * num_kv_heads * kv_seq_len * head_dim];
    for batch in 0..BATCH {
        for kv_h in 0..num_kv_heads {
            for pos in 0..kv_seq_len {
                for d in 0..head_dim {
                    let logical_index =
                        (((batch * num_kv_heads + kv_h) * kv_seq_len + pos) * head_dim) + d;
                    let flat_index = pos * (num_kv_heads * head_dim) + kv_h * head_dim + d;
                    buffer[flat_index] = tensor_value(kind, logical_index);
                }
            }
        }
    }
    buffer
}

fn bench_compute_attention_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_attention_decode");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    for &kv_seq_len in &KV_SEQ_LENS {
        for case in CASES {
            let mut fixture = DecodeFixture::new(case, kv_seq_len);
            group.throughput(Throughput::Bytes(fixture.kv_bytes_per_decode_step()));
            group.bench_with_input(
                BenchmarkId::new(case.name, kv_seq_len),
                &kv_seq_len,
                |b, &kv_len| {
                    b.iter(|| {
                        compute_attention_for_bench(
                            black_box(fixture.output.as_mut_slice()),
                            black_box(fixture.q.as_slice()),
                            black_box(fixture.k.as_slice()),
                            black_box(fixture.v.as_slice()),
                            Q_SEQ_LEN,
                            kv_len,
                            kv_len - 1,
                            black_box(&fixture.cfg),
                            black_box(fixture.scores.as_mut_slice()),
                            kv_len,
                        );
                        black_box(fixture.output.as_slice());
                    });
                },
            );
            assert_eq!(fixture.kv_seq_len, kv_seq_len);
        }
    }

    group.finish();
}

criterion_group!(benches, bench_compute_attention_decode);
criterion_main!(benches);
