use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::kv_cache::{FlatKVCache, FlatKVCacheConfig};

const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
const KV_SEQ_LENS: [usize; 3] = [512, 1024, 2048];
const THEORETICAL_PEAK_GBPS: f64 = 400.0;

#[derive(Clone, Copy, Debug)]
enum Layout {
    CurrentSequenceMajor,
    AlternativeHeadMajor,
}

impl Layout {
    fn name(self) -> &'static str {
        match self {
            Layout::CurrentSequenceMajor => "current_seq_major",
            Layout::AlternativeHeadMajor => "alternative_head_major",
        }
    }
}

#[derive(Debug)]
struct CurrentFixture {
    cache: FlatKVCache,
    kv_seq_len: usize,
}

impl CurrentFixture {
    fn new(kv_seq_len: usize) -> Self {
        let sequence_major_k = build_sequence_major_kv(1, kv_seq_len);
        let sequence_major_v = build_sequence_major_kv(2, kv_seq_len);

        let cfg = FlatKVCacheConfig::for_qwen3(1, N_KV_HEADS, HEAD_DIM, kv_seq_len);
        let mut cache = FlatKVCache::new(cfg);
        cache.prefill_layer(0, &sequence_major_k, &sequence_major_v, kv_seq_len);
        cache.advance_by(kv_seq_len);

        assert_eq!(cache.seq_len(), kv_seq_len);
        assert_eq!(cache.get_k(0).len(), kv_seq_len * KV_DIM);
        assert_eq!(cache.get_v(0).len(), kv_seq_len * KV_DIM);

        Self { cache, kv_seq_len }
    }

    #[inline]
    fn k(&self) -> &[f32] {
        self.cache.get_k(0)
    }

    #[inline]
    fn v(&self) -> &[f32] {
        self.cache.get_v(0)
    }
}

#[derive(Debug)]
struct HeadMajorFixture {
    k: Vec<f32>,
    v: Vec<f32>,
    kv_seq_len: usize,
}

impl HeadMajorFixture {
    fn new(kv_seq_len: usize) -> Self {
        let sequence_major_k = build_sequence_major_kv(1, kv_seq_len);
        let sequence_major_v = build_sequence_major_kv(2, kv_seq_len);
        let k = sequence_major_to_head_major(&sequence_major_k, kv_seq_len);
        let v = sequence_major_to_head_major(&sequence_major_v, kv_seq_len);

        assert_eq!(k.len(), kv_seq_len * KV_DIM);
        assert_eq!(v.len(), kv_seq_len * KV_DIM);

        Self { k, v, kv_seq_len }
    }
}

fn tensor_value(kind: u32, logical_index: usize) -> f32 {
    let mut x = (logical_index as u32)
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

fn build_sequence_major_kv(kind: u32, kv_seq_len: usize) -> Vec<f32> {
    let mut buffer = vec![0.0f32; kv_seq_len * KV_DIM];

    for kv_h in 0..N_KV_HEADS {
        for pos in 0..kv_seq_len {
            for d in 0..HEAD_DIM {
                let logical_index = (kv_h * kv_seq_len + pos) * HEAD_DIM + d;
                let flat_index = pos * KV_DIM + kv_h * HEAD_DIM + d;
                buffer[flat_index] = tensor_value(kind, logical_index);
            }
        }
    }

    buffer
}

fn sequence_major_to_head_major(sequence_major: &[f32], kv_seq_len: usize) -> Vec<f32> {
    assert_eq!(sequence_major.len(), kv_seq_len * KV_DIM);
    let mut head_major = vec![0.0f32; sequence_major.len()];

    for kv_h in 0..N_KV_HEADS {
        for pos in 0..kv_seq_len {
            for d in 0..HEAD_DIM {
                let src = pos * KV_DIM + kv_h * HEAD_DIM + d;
                let dst = kv_h * kv_seq_len * HEAD_DIM + pos * HEAD_DIM + d;
                head_major[dst] = sequence_major[src];
            }
        }
    }

    head_major
}

fn bytes_per_iter(kv_seq_len: usize) -> u64 {
    (2 * N_KV_HEADS * kv_seq_len * HEAD_DIM * std::mem::size_of::<f32>()) as u64
}

#[allow(dead_code)]
fn achieved_bw_gbps(total_bytes_read: u64, time_ns: f64) -> f64 {
    total_bytes_read as f64 / time_ns
}

#[allow(dead_code)]
fn gap_pct(achieved_bw_gbps: f64) -> f64 {
    (1.0 - achieved_bw_gbps / THEORETICAL_PEAK_GBPS) * 100.0
}

#[allow(dead_code)]
fn alternative_gain_pct(current_bw_gbps: f64, alternative_bw_gbps: f64) -> f64 {
    (alternative_bw_gbps / current_bw_gbps - 1.0) * 100.0
}

#[inline(never)]
fn read_current_decode_order(k: &[f32], v: &[f32], kv_seq_len: usize) -> f32 {
    debug_assert_eq!(k.len(), kv_seq_len * KV_DIM);
    debug_assert_eq!(v.len(), kv_seq_len * KV_DIM);

    let mut acc = 0.0f32;

    for kv_h in 0..N_KV_HEADS {
        for pos in 0..kv_seq_len {
            let k_off = pos * KV_DIM + kv_h * HEAD_DIM;
            let mut dot_proxy = 0.0f32;
            for d in 0..HEAD_DIM {
                dot_proxy += k[k_off + d];
            }
            acc += dot_proxy;
        }
    }

    for kv_h in 0..N_KV_HEADS {
        for pos in 0..kv_seq_len {
            let v_off = pos * KV_DIM + kv_h * HEAD_DIM;
            let weight = 1.0 + ((pos & 15) as f32) * 0.03125;
            let mut value_proxy = 0.0f32;
            for d in 0..HEAD_DIM {
                value_proxy += v[v_off + d] * weight;
            }
            acc += value_proxy;
        }
    }

    acc
}

#[inline(never)]
fn read_head_major_decode_order(k: &[f32], v: &[f32], kv_seq_len: usize) -> f32 {
    debug_assert_eq!(k.len(), kv_seq_len * KV_DIM);
    debug_assert_eq!(v.len(), kv_seq_len * KV_DIM);

    let mut acc = 0.0f32;

    for kv_h in 0..N_KV_HEADS {
        let head_base = kv_h * kv_seq_len * HEAD_DIM;
        for pos in 0..kv_seq_len {
            let k_off = head_base + pos * HEAD_DIM;
            let mut dot_proxy = 0.0f32;
            for d in 0..HEAD_DIM {
                dot_proxy += k[k_off + d];
            }
            acc += dot_proxy;
        }
    }

    for kv_h in 0..N_KV_HEADS {
        let head_base = kv_h * kv_seq_len * HEAD_DIM;
        for pos in 0..kv_seq_len {
            let v_off = head_base + pos * HEAD_DIM;
            let weight = 1.0 + ((pos & 15) as f32) * 0.03125;
            let mut value_proxy = 0.0f32;
            for d in 0..HEAD_DIM {
                value_proxy += v[v_off + d] * weight;
            }
            acc += value_proxy;
        }
    }

    acc
}

fn bench_kv_cache_layout_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_layout_read");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    for &kv_seq_len in &KV_SEQ_LENS {
        group.throughput(Throughput::BytesDecimal(bytes_per_iter(kv_seq_len)));

        let current = CurrentFixture::new(kv_seq_len);
        group.bench_with_input(
            BenchmarkId::new(Layout::CurrentSequenceMajor.name(), kv_seq_len),
            &current,
            |b, fixture| {
                b.iter(|| {
                    let acc = read_current_decode_order(
                        black_box(fixture.k()),
                        black_box(fixture.v()),
                        black_box(fixture.kv_seq_len),
                    );
                    black_box(acc);
                });
            },
        );

        let alternative = HeadMajorFixture::new(kv_seq_len);
        group.bench_with_input(
            BenchmarkId::new(Layout::AlternativeHeadMajor.name(), kv_seq_len),
            &alternative,
            |b, fixture| {
                b.iter(|| {
                    let acc = read_head_major_decode_order(
                        black_box(fixture.k.as_slice()),
                        black_box(fixture.v.as_slice()),
                        black_box(fixture.kv_seq_len),
                    );
                    black_box(acc);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_kv_cache_layout_read);
criterion_main!(benches);
