use super::*;

const SENTINEL: f32 = -12_345.25;
const ATOL: f32 = 3e-5;
const RTOL: f32 = 3e-5;

struct Fixture {
    seq: usize,
    dim: usize,
    q_heads: usize,
    kv_heads: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl Fixture {
    fn new(seq: usize, dim: usize, q_heads: usize, kv_heads: usize, padded: bool) -> Self {
        let q_dim = q_heads * dim;
        let kv_dim = kv_heads * dim;
        let q_stride = if padded { 2 * q_dim + 5 } else { q_dim };
        let kv_stride = kv_dim + if padded { 3 } else { 0 };
        let out_stride = q_dim + if padded { 7 } else { 0 };
        let mut fixture = Self {
            seq,
            dim,
            q_heads,
            kv_heads,
            q_stride,
            kv_stride,
            out_stride,
            q: vec![f32::NAN; seq * q_stride],
            k: vec![f32::NAN; seq * kv_stride],
            v: vec![f32::NAN; seq * kv_stride],
        };
        for row in 0..seq {
            for head in 0..q_heads {
                for d in 0..dim {
                    fixture.q[row * q_stride + head * dim + d] =
                        ((row * 7 + head * 11 + d * 3 + 2) % 29) as f32 / 17.0 - 0.8;
                }
            }
            for head in 0..kv_heads {
                for d in 0..dim {
                    fixture.k[row * kv_stride + head * dim + d] =
                        ((row * 13 + head * 19 + d * 5 + 3) % 31) as f32 / 13.0 - 1.1;
                    fixture.v[row * kv_stride + head * dim + d] =
                        ((row * 17 + head * 23 + d * 7 + 5) % 43) as f32 / 9.0 - 2.2;
                }
            }
        }
        fixture
    }

    fn output(&self) -> Vec<f32> {
        vec![SENTINEL; (self.seq * self.out_stride).max(self.out_stride).max(1)]
    }

    fn offsets(&self, q_head: usize) -> (usize, usize) {
        let groups = self.q_heads / self.kv_heads;
        (q_head * self.dim, (q_head / groups) * self.dim)
    }

    fn run(
        &self,
        q_head: usize,
        scale: f32,
        config: &TiledAttentionConfig,
        buffers: &mut TiledAttentionBuffers,
        output: &mut [f32],
    ) {
        let (q_offset, kv_offset) = self.offsets(q_head);
        causal_attention_head(
            &self.q,
            &self.k,
            &self.v,
            output,
            self.seq,
            self.q_stride,
            self.kv_stride,
            self.out_stride,
            q_offset,
            kv_offset,
            self.dim,
            scale,
            config,
            buffers,
        );
    }

    fn scalar(&self, q_head: usize, scale: f32, config: &TiledAttentionConfig, output: &mut [f32]) {
        let (q_offset, kv_offset) = self.offsets(q_head);
        let mut unused = TiledAttentionBuffers::default();
        let mut acc = vec![0.0; self.dim];
        let mut scores = vec![0.0; config.tile_size_kv.max(1)];
        old_scalar_attention_head(
            &self.q,
            &self.k,
            &self.v,
            output,
            self.seq,
            self.q_stride,
            self.kv_stride,
            self.out_stride,
            q_offset,
            kv_offset,
            self.dim,
            scale,
            config,
            &mut unused,
            &mut acc,
            &mut scores,
        );
    }

    fn f64_reference(&self, q_head: usize, scale: f32, output: &mut [f32]) {
        let (q_offset, kv_offset) = self.offsets(q_head);
        for row in 0..self.seq {
            let mut scores = Vec::with_capacity(row + 1);
            for key in 0..=row {
                let mut dot = 0.0f64;
                for d in 0..self.dim {
                    dot += f64::from(self.q[row * self.q_stride + q_offset + d])
                        * f64::from(self.k[key * self.kv_stride + kv_offset + d]);
                }
                scores.push(dot * f64::from(scale));
            }
            assert!(scores.iter().all(|score| score.is_finite()));
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let denominator: f64 = weights.iter().sum();
            assert!(denominator.is_finite() && denominator > 0.0);
            for d in 0..self.dim {
                let mut numerator = 0.0f64;
                for (key, weight) in weights.iter().enumerate() {
                    numerator += weight * f64::from(self.v[key * self.kv_stride + kv_offset + d]);
                }
                output[row * self.out_stride + q_offset + d] = (numerator / denominator) as f32;
            }
        }
    }
}

fn tile_config(dim: usize, tile_q: usize, tile_kv: usize) -> TiledAttentionConfig {
    TiledAttentionConfig {
        tile_size_q: tile_q,
        tile_size_kv: tile_kv,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: dim,
    }
}

fn buffers(config: &TiledAttentionConfig) -> TiledAttentionBuffers {
    let mut buffers = TiledAttentionBuffers::default();
    prepare_buffers(config, &mut buffers).unwrap();
    buffers
}

fn assert_finite_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite(),
            "nonfinite lane {i}: {a} vs {e}"
        );
        assert!(
            (a - e).abs() <= ATOL + RTOL * e.abs(),
            "lane {i}: {a} vs {e}"
        );
    }
}

fn assert_scalar_classification(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "lane {i}: expected NaN, got {a}");
        } else if e.is_infinite() {
            assert_eq!(a.to_bits(), e.to_bits(), "lane {i}");
        } else {
            assert!(a.is_finite(), "lane {i}: {a} vs finite {e}");
            assert!(
                (a - e).abs() <= ATOL + RTOL * e.abs(),
                "lane {i}: {a} vs {e}"
            );
        }
    }
}

fn assert_row_zero(output: &[f32], fixture: &Fixture, row: usize) {
    for &value in &output[row * fixture.out_stride..row * fixture.out_stride + fixture.dim] {
        assert_eq!(value.to_bits(), 0, "poisoned row {row} must be exact +0");
    }
}

fn one_hot_queries(fixture: &mut Fixture) {
    for row in 0..fixture.seq {
        for d in 0..fixture.dim {
            fixture.q[row * fixture.q_stride + d] = if d == 0 { 1.0 } else { 0.0 };
            fixture.k[row * fixture.kv_stride + d] = if d == 0 { row as f32 / 10.0 } else { 0.0 };
        }
    }
}

fn poison_scratch(buffers: &mut TiledAttentionBuffers) {
    for vector in [
        &mut buffers.q_tile,
        &mut buffers.k_tile,
        &mut buffers.v_tile,
        &mut buffers.s_tile,
        &mut buffers.output,
        &mut buffers.row_max,
        &mut buffers.row_sum,
    ] {
        vector.fill(f32::NAN);
    }
}

fn storage_snapshot(buffers: &TiledAttentionBuffers) -> Vec<(usize, usize, Vec<u32>)> {
    [
        &buffers.q_tile,
        &buffers.k_tile,
        &buffers.v_tile,
        &buffers.s_tile,
        &buffers.output,
        &buffers.row_max,
        &buffers.row_sum,
    ]
    .into_iter()
    .map(|vector| {
        (
            vector.as_ptr() as usize,
            vector.capacity(),
            vector.iter().map(|v| v.to_bits()).collect(),
        )
    })
    .collect()
}

#[test]
fn blocked_attention_matches_independent_f64_and_old_scalar() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(std::panic::catch_unwind(|| assert_finite_close(&[bad], &[0.0])).is_err());
        assert!(std::panic::catch_unwind(|| assert_finite_close(&[0.0], &[bad])).is_err());
    }
    for dim in [1, 3, 7, 16, 31, 32, 63, 64, 127, 128, 129] {
        for seq in [1, 2, 5, 9, 17] {
            let fixture = Fixture::new(seq, dim, 1, 1, false);
            let scale = 1.0 / (dim as f32).sqrt();
            for (br, bc) in [(1, 1), (3, 2), (4, 3), (5, 8)] {
                let config = tile_config(dim, br, bc);
                let mut actual = fixture.output();
                fixture.run(0, scale, &config, &mut buffers(&config), &mut actual);
                let mut old = fixture.output();
                fixture.scalar(0, scale, &config, &mut old);
                let mut independent = fixture.output();
                fixture.f64_reference(0, scale, &mut independent);
                assert!(independent.iter().any(|v| v.abs() > 0.1));
                assert_finite_close(&actual, &old);
                assert_finite_close(&actual, &independent);
            }
        }
    }
    for dim in [32, 128] {
        for seq in [65, 97] {
            let fixture = Fixture::new(seq, dim, 1, 1, false);
            let scale = 1.0 / (dim as f32).sqrt();
            let config = TiledAttentionConfig::new(1, 1, dim);
            let mut actual = fixture.output();
            fixture.run(0, scale, &config, &mut buffers(&config), &mut actual);
            let mut old = fixture.output();
            fixture.scalar(0, scale, &config, &mut old);
            let mut independent = fixture.output();
            fixture.f64_reference(0, scale, &mut independent);
            assert!(independent.iter().any(|v| v.abs() > 0.1));
            assert_finite_close(&actual, &old);
            assert_finite_close(&actual, &independent);
        }
    }
}

#[test]
fn blocked_attention_preserves_strided_gqa_head_offsets() {
    for padded in [false, true] {
        let mut fixture = Fixture::new(9, 7, 4, 2, padded);
        if !padded {
            let old_q = std::mem::take(&mut fixture.q);
            let old_stride = fixture.q_stride;
            fixture.q_stride = 2 * fixture.q_heads * fixture.dim;
            fixture.q = vec![f32::NAN; fixture.seq * fixture.q_stride];
            for row in 0..fixture.seq {
                fixture.q[row * fixture.q_stride..row * fixture.q_stride + old_stride]
                    .copy_from_slice(&old_q[row * old_stride..(row + 1) * old_stride]);
            }
        }
        let config = tile_config(fixture.dim, 4, 3);
        let mut buffers = buffers(&config);
        let mut actual = fixture.output();
        let mut old = fixture.output();
        let mut independent = fixture.output();
        for head in 0..fixture.q_heads {
            let before: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
            fixture.run(head, 0.4, &config, &mut buffers, &mut actual);
            fixture.scalar(head, 0.4, &config, &mut old);
            fixture.f64_reference(head, 0.4, &mut independent);
            assert_finite_close(&actual, &old);
            assert_finite_close(&actual, &independent);
            for row in 0..fixture.seq {
                for lane in 0..fixture.out_stride {
                    if !(head * fixture.dim..(head + 1) * fixture.dim).contains(&lane) {
                        let i = row * fixture.out_stride + lane;
                        assert_eq!(actual[i].to_bits(), before[i], "unselected output lane {i}");
                    }
                }
            }
        }
        assert!(
            actual[0..fixture.dim]
                .iter()
                .zip(&actual[2 * fixture.dim..3 * fixture.dim])
                .any(|(a, b)| (a - b).abs() > 1e-3)
        );
    }
}

#[test]
fn blocked_attention_masks_future_key_and_value_poison() {
    for position in [2, 5] {
        for poison in [f32::NAN, f32::INFINITY] {
            for poison_key in [true, false] {
                let mut fixture = Fixture::new(7, 3, 1, 1, false);
                one_hot_queries(&mut fixture);
                let config = tile_config(3, 4, 3);
                let mut clean = fixture.output();
                fixture.scalar(0, 1.0, &config, &mut clean);
                if poison_key {
                    fixture.k[position * fixture.kv_stride] = poison;
                } else {
                    fixture.v[position * fixture.kv_stride + 1] = poison;
                }
                let mut actual = fixture.output();
                fixture.run(0, 1.0, &config, &mut buffers(&config), &mut actual);
                let mut old = fixture.output();
                fixture.scalar(0, 1.0, &config, &mut old);
                assert_scalar_classification(&actual, &old);
                assert_finite_close(
                    &actual[..position * fixture.out_stride],
                    &clean[..position * fixture.out_stride],
                );
                if poison_key {
                    for row in position..fixture.seq {
                        assert_row_zero(&actual, &fixture, row);
                    }
                }
            }
        }
    }
    let mut fixture = Fixture::new(7, 3, 1, 1, false);
    one_hot_queries(&mut fixture);
    let config = tile_config(3, 4, 3);
    let mut clean = fixture.output();
    fixture.scalar(0, 1.0, &config, &mut clean);
    fixture.v[2 * fixture.kv_stride] += 1_000.0;
    let mut actual = fixture.output();
    fixture.run(0, 1.0, &config, &mut buffers(&config), &mut actual);
    assert_finite_close(
        &actual[..2 * fixture.out_stride],
        &clean[..2 * fixture.out_stride],
    );
    assert!((actual[2 * fixture.out_stride] - clean[2 * fixture.out_stride]).abs() > 1.0);
}

#[test]
fn blocked_attention_unmasked_score_poison_zeros_only_affected_rows() {
    for query_poison in [true, false] {
        for poison in [f32::NAN, f32::INFINITY] {
            let mut fixture = Fixture::new(7, 3, 1, 1, false);
            one_hot_queries(&mut fixture);
            let config = tile_config(3, 4, 2);
            if query_poison {
                fixture.q[fixture.q_stride] = poison;
            } else {
                fixture.k[fixture.kv_stride] = poison;
            }
            let mut actual = fixture.output();
            fixture.run(0, 1.0, &config, &mut buffers(&config), &mut actual);
            let mut old = fixture.output();
            fixture.scalar(0, 1.0, &config, &mut old);
            assert_scalar_classification(&actual, &old);
            assert_row_zero(&actual, &fixture, 1);
            assert!(actual[..fixture.dim].iter().any(|v| v.abs() > 0.1));
            if query_poison {
                assert!(
                    actual[2 * fixture.out_stride..2 * fixture.out_stride + fixture.dim]
                        .iter()
                        .any(|v| v.abs() > 0.1)
                );
            } else {
                for row in 1..fixture.seq {
                    assert_row_zero(&actual, &fixture, row);
                }
            }
        }
    }
    let mut overflow = Fixture::new(3, 1, 1, 1, false);
    overflow.q.fill(1e20);
    overflow.k.fill(1e20);
    let config = tile_config(1, 2, 2);
    let mut actual = overflow.output();
    overflow.run(0, 1.0, &config, &mut buffers(&config), &mut actual);
    for row in 0..overflow.seq {
        assert_row_zero(&actual, &overflow, row);
    }
}

#[test]
fn blocked_attention_rescales_online_numerator_and_denominator() {
    let mut fixture = Fixture::new(9, 1, 1, 1, false);
    fixture
        .k
        .copy_from_slice(&[-80.0, -79.0, -78.0, 0.0, 1.0, 2.0, 80.0, 79.0, 78.0]);
    for row in 0..fixture.seq {
        fixture.q[row] = 0.9 + row as f32 / 20.0;
        fixture.v[row] = (row * row + 3) as f32 / 7.0;
    }
    for scale in [0.0, 0.01, 1.0, 10.0] {
        let config = tile_config(1, 4, 3);
        let mut actual = fixture.output();
        fixture.run(0, scale, &config, &mut buffers(&config), &mut actual);
        let mut old = fixture.output();
        fixture.scalar(0, scale, &config, &mut old);
        let mut independent = fixture.output();
        fixture.f64_reference(0, scale, &mut independent);
        assert_finite_close(&actual, &old);
        assert_finite_close(&actual, &independent);
        assert!((actual[8] - actual[0]).abs() > 1.0);
    }
}

#[test]
fn blocked_attention_preserves_negative_infinity_and_all_invalid_rows() {
    for all_invalid in [false, true] {
        let mut fixture = Fixture::new(7, 1, 1, 1, false);
        fixture.q.fill(1.0);
        fixture.k.fill(f32::NEG_INFINITY);
        if !all_invalid {
            fixture.k[0] = 0.0;
        }
        let config = tile_config(1, 3, 2);
        let mut actual = fixture.output();
        fixture.run(0, 1.0, &config, &mut buffers(&config), &mut actual);
        let mut old = fixture.output();
        fixture.scalar(0, 1.0, &config, &mut old);
        assert_scalar_classification(&actual, &old);
        for row in 0..fixture.seq {
            if all_invalid {
                assert_row_zero(&actual, &fixture, row);
            } else {
                assert_eq!(actual[row].to_bits(), fixture.v[0].to_bits());
            }
        }
    }
    let mut fixture = Fixture::new(5, 1, 1, 1, false);
    fixture.q.fill(1.0);
    fixture
        .k
        .copy_from_slice(&[f32::NEG_INFINITY, -2.0, 0.0, 2.0, 4.0]);
    let config = tile_config(1, 3, 1);
    let mut actual = fixture.output();
    fixture.run(0, 1.0, &config, &mut buffers(&config), &mut actual);
    let mut old = fixture.output();
    fixture.scalar(0, 1.0, &config, &mut old);
    assert_scalar_classification(&actual, &old);
    for row in 0..fixture.seq {
        assert_row_zero(&actual, &fixture, row);
    }
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut fixture = Fixture::new(3, 1, 1, 1, false);
        fixture.q.fill(1.0);
        fixture.k.copy_from_slice(&[0.0, f32::NEG_INFINITY, 0.0]);
        fixture.v[1] = poison;
        let config = tile_config(1, 3, 2);
        let mut actual = fixture.output();
        fixture.run(0, 1.0, &config, &mut buffers(&config), &mut actual);
        let mut old = fixture.output();
        fixture.scalar(0, 1.0, &config, &mut old);
        assert_scalar_classification(&actual, &old);
        assert_eq!(actual[0].to_bits(), fixture.v[0].to_bits());
        assert!(actual[1].is_nan() && actual[2].is_nan());
    }
}

#[test]
fn blocked_attention_overwrites_poisoned_scratch_across_reuse() {
    let config = tile_config(7, 4, 3);
    let mut reused = buffers(&config);
    let snapshot = storage_snapshot(&reused);
    for seq in [17, 3, 9, 1, 5] {
        let fixture = Fixture::new(seq, 7, 4, 2, true);
        for head in [3, 0, 2, 1] {
            poison_scratch(&mut reused);
            let mut actual = fixture.output();
            fixture.run(head, 0.4, &config, &mut reused, &mut actual);
            let mut fresh = fixture.output();
            fixture.run(head, 0.4, &config, &mut buffers(&config), &mut fresh);
            let mut old = fixture.output();
            fixture.scalar(head, 0.4, &config, &mut old);
            let mut independent = fixture.output();
            fixture.f64_reference(head, 0.4, &mut independent);
            assert_finite_close(&actual, &fresh);
            assert_finite_close(&actual, &old);
            assert_finite_close(&actual, &independent);
        }
    }
    let after = storage_snapshot(&reused);
    for (before, after) in snapshot.iter().zip(&after) {
        assert_eq!(
            before.0, after.0,
            "kernel must not reallocate prepared tile storage"
        );
        assert_eq!(before.1, after.1);
        assert_eq!(before.2.len(), after.2.len());
    }
}

#[test]
fn blocked_attention_preserves_zero_length_and_zero_tile_behavior() {
    let fixture = Fixture::new(0, 3, 1, 1, true);
    let config = tile_config(3, 3, 2);
    let mut buffers = buffers(&config);
    poison_scratch(&mut buffers);
    let before = storage_snapshot(&buffers);
    let mut actual = fixture.output();
    let expected = actual.clone();
    fixture.run(0, 1.0, &config, &mut buffers, &mut actual);
    assert_eq!(actual, expected);
    assert_eq!(storage_snapshot(&buffers), before);
    for (br, bc) in [(0, 0), (0, 3), (4, 0)] {
        let fixture = Fixture::new(5, 3, 1, 1, false);
        let config = tile_config(3, br, bc);
        let mut actual = fixture.output();
        fixture.run(0, 0.5, &config, &mut self::buffers(&config), &mut actual);
        let mut old = fixture.output();
        fixture.scalar(0, 0.5, &config, &mut old);
        assert_finite_close(&actual, &old);
    }
    let fixture = Fixture::new(3, 0, 1, 1, false);
    let config = tile_config(0, 2, 2);
    let mut actual = fixture.output();
    let expected = actual.clone();
    fixture.run(0, 1.0, &config, &mut self::buffers(&config), &mut actual);
    assert_eq!(actual, expected);
}

#[derive(Default)]
struct Trace {
    q: Vec<(usize, usize, usize)>,
    kv: Vec<(usize, usize, usize, usize)>,
    scores: Vec<(usize, usize, usize, usize, usize)>,
}

impl WorkObserver for Trace {
    fn packed_q(&mut self, start: usize, rows: usize, dim: usize) {
        self.q.push((start, rows, dim));
    }
    fn packed_kv(&mut self, q_start: usize, kv_start: usize, rows: usize, dim: usize) {
        self.kv.push((q_start, kv_start, rows, dim));
    }
    fn scored(
        &mut self,
        q_start: usize,
        kv_start: usize,
        q_rows: usize,
        kv_rows: usize,
        dim: usize,
    ) {
        self.scores.push((q_start, kv_start, q_rows, kv_rows, dim));
    }
}

#[test]
fn blocked_attention_reuses_each_kv_pack_across_query_rows() {
    let fixture = Fixture::new(9, 7, 1, 1, true);
    let config = tile_config(7, 3, 2);
    let mut buffers = buffers(&config);
    let mut trace = Trace::default();
    let mut actual = fixture.output();
    causal_attention_head_with_observer(
        &fixture.q,
        &fixture.k,
        &fixture.v,
        &mut actual,
        fixture.seq,
        fixture.q_stride,
        fixture.kv_stride,
        fixture.out_stride,
        0,
        0,
        fixture.dim,
        0.4,
        &config,
        &mut buffers,
        &mut trace,
    );
    let mut old = fixture.output();
    fixture.scalar(0, 0.4, &config, &mut old);
    let mut independent = fixture.output();
    fixture.f64_reference(0, 0.4, &mut independent);
    assert_finite_close(&actual, &old);
    assert_finite_close(&actual, &independent);
    assert_eq!(trace.q, vec![(0, 3, 7), (3, 3, 7), (6, 3, 7)]);
    let mut kv = Vec::new();
    let mut scores = Vec::new();
    for q_start in [0, 3, 6] {
        let q_end = q_start + 3;
        for kv_start in (0..q_end).step_by(2) {
            let rows = (q_end - kv_start).min(2);
            kv.push((q_start, kv_start, rows, 7));
            scores.push((q_start, kv_start, 3, rows, 7));
        }
    }
    assert_eq!(kv.len(), 10);
    assert_eq!(trace.kv, kv);
    assert_eq!(trace.scores, scores);
    let old_row_streaming_blocks: usize = (1usize..=fixture.seq).map(|row| row.div_ceil(2)).sum();
    assert_eq!(old_row_streaming_blocks, 25);
    assert!(trace.kv.len() < old_row_streaming_blocks);
    let empty = Fixture::new(0, 7, 1, 1, true);
    let mut empty_trace = Trace::default();
    let mut empty_output = empty.output();
    causal_attention_head_with_observer(
        &empty.q,
        &empty.k,
        &empty.v,
        &mut empty_output,
        empty.seq,
        empty.q_stride,
        empty.kv_stride,
        empty.out_stride,
        0,
        0,
        empty.dim,
        0.4,
        &config,
        &mut buffers,
        &mut empty_trace,
    );
    assert!(empty_trace.q.is_empty() && empty_trace.kv.is_empty() && empty_trace.scores.is_empty());
}

#[test]
fn blocked_attention_prepare_rejects_unaddressable_shapes_before_mutation() {
    let valid = tile_config(7, 3, 2);
    let mut prepared = self::buffers(&valid);
    assert_eq!(
        [
            &prepared.q_tile,
            &prepared.k_tile,
            &prepared.v_tile,
            &prepared.s_tile,
            &prepared.output,
            &prepared.row_max,
            &prepared.row_sum
        ]
        .map(Vec::len),
        [21, 14, 14, 6, 21, 3, 3],
    );
    let high_water = storage_snapshot(&prepared);
    prepare_buffers(&tile_config(1, 1, 1), &mut prepared).unwrap();
    assert_eq!(storage_snapshot(&prepared), high_water);

    let mut buffers = TiledAttentionBuffers::default();
    buffers.q_tile.extend([1.0, f32::NAN, 2.0]);
    buffers.k_tile.push(3.0);
    buffers.v_tile.extend([4.0, 5.0]);
    buffers.s_tile.push(6.0);
    buffers.output.push(7.0);
    buffers.row_max.push(8.0);
    buffers.row_sum.push(9.0);
    let before = storage_snapshot(&buffers);
    let overflowing_score_side = 1usize << (usize::BITS / 2);
    let unaddressable_score_side = overflowing_score_side / 2;
    let unaddressable_floats = isize::MAX as usize / std::mem::size_of::<f32>() + 1;
    for config in [
        tile_config(2, usize::MAX, 1),
        tile_config(2, 1, usize::MAX),
        tile_config(1, overflowing_score_side, overflowing_score_side),
        tile_config(usize::MAX, 1, 1),
        tile_config(unaddressable_floats, 1, 1),
        tile_config(1, 1, unaddressable_floats),
        tile_config(1, unaddressable_score_side, unaddressable_score_side),
    ] {
        assert!(matches!(
            prepare_buffers(&config, &mut buffers),
            Err(InferenceError::InvalidInput(_))
        ));
        assert_eq!(storage_snapshot(&buffers), before);
    }
}

fn old_scalar_attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    q_head_offset: usize,
    kv_head_offset: usize,
    head_dim: usize,
    scale: f32,
    config: &TiledAttentionConfig,
    _buffers: &mut TiledAttentionBuffers,
    acc: &mut [f32],
    tile_scores: &mut [f32],
) {
    let tile_q = config.tile_size_q.max(1);
    let tile_kv = config.tile_size_kv.max(1);

    debug_assert!(acc.len() >= head_dim);
    debug_assert!(tile_scores.len() >= tile_kv);

    for q_start in (0..seq_len).step_by(tile_q) {
        let q_end = (q_start + tile_q).min(seq_len);

        for q_row in q_start..q_end {
            let q_off = q_row * q_stride + q_head_offset;
            let q_vec = &q[q_off..q_off + head_dim];

            acc[..head_dim].fill(0.0);
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            // ADR-080 C1 (#780): tracked independently of the `local_max`/
            // `running_max` folds because `if score > local_max` is `false`
            // for a NaN operand (IEEE 754), silently dropping it from the
            // max the same way `f32::max` does. This is the online-softmax
            // single-row analogue of `flash_causal.rs`'s `block_has_poison`.
            let mut row_poisoned = false;

            let max_kv_pos = q_row + 1;
            for kv_start in (0..max_kv_pos).step_by(tile_kv) {
                let kv_end = (kv_start + tile_kv).min(max_kv_pos);
                let tile_len = kv_end - kv_start;

                let mut local_max = f32::NEG_INFINITY;
                for (local_idx, kv_pos) in (kv_start..kv_end).enumerate() {
                    let k_off = kv_pos * kv_stride + kv_head_offset;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k[k_off + d];
                    }
                    let score = dot * scale;
                    if score.is_nan() || score == f32::INFINITY {
                        row_poisoned = true;
                    }
                    tile_scores[local_idx] = score;
                    if score > local_max {
                        local_max = score;
                    }
                }

                let new_max = running_max.max(local_max);
                let alpha = if running_sum == 0.0 {
                    0.0
                } else {
                    (running_max - new_max).exp()
                };

                for a in acc[..head_dim].iter_mut() {
                    *a *= alpha;
                }
                running_sum *= alpha;

                for (local_idx, &ts) in tile_scores[..tile_len].iter().enumerate() {
                    let weight = (ts - new_max).exp();
                    running_sum += weight;

                    let kv_pos = kv_start + local_idx;
                    let v_off = kv_pos * kv_stride + kv_head_offset;
                    for d in 0..head_dim {
                        acc[d] += weight * v[v_off + d];
                    }
                }

                running_max = new_max;
            }

            let out_off = q_row * out_stride + q_head_offset;
            // ADR-080 C1 (#780): RED before the fix, `running_sum.max(f32::MIN_POSITIVE)`
            // silently replaced a NaN `running_sum` with a tiny-but-positive
            // finite value instead of failing closed, so `1.0 / running_sum`
            // became an enormous (~8.5e37) finite number and the output row
            // exploded into huge finite garbage rather than zero. Structurally
            // distinct from `attention::softmax_row::finalize_row` (this
            // online algorithm writes into a separate `acc` accumulator, not
            // an in-place score row), so this is an inline contract-conformant
            // fix rather than a call to the shared helper: fail closed on a
            // poisoned row OR a non-positive/non-finite denominator.
            if row_poisoned || !running_sum.is_finite() || running_sum <= 0.0 {
                output[out_off..out_off + head_dim].fill(0.0);
            } else {
                let inv_sum = 1.0 / running_sum;
                for d in 0..head_dim {
                    output[out_off + d] = acc[d] * inv_sum;
                }
            }
        }
    }
}
