//! Kernel-level tests for the EmbeddingGemma 2 Metal state. Every kernel is compared with the CPU
//! function it replaces. Whole-model parity lives with the model tests.

use super::*;
use crate::forward::cpu::{elementwise_mul, matmul_bt, rms_norm};
use crate::measurement::gpu_test_lock;
use crate::model::embeddinggemma2::bidirectional_attention;
use crate::model::gemma4_ops::{gemma4_apply_rope, gemma4_gelu_tanh};

fn enforce() -> bool {
    std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some()
}

/// True (after saying so) when this machine has no Metal device and the run is not enforcing one.
fn skip_without_device() -> bool {
    if Device::system_default().is_some() {
        return false;
    }
    assert!(!enforce(), "Metal device required under enforcement");
    eprintln!("SKIP embeddinggemma2 Metal: no Metal device");
    true
}

struct Lcg(u64);

impl Lcg {
    /// Uniform in `[-1, 1)`.
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / (1u64 << 23) as f32 - 1.0
    }
}

fn rand_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| scale * rng.next()).collect()
}

struct Kit {
    ctx: Context,
    pipelines: Pipelines,
}

fn kit(head_dims: &[usize]) -> Kit {
    let ctx = Context::new().expect("Metal context");
    let pipelines = Pipelines::new(&ctx.device, head_dims).expect("pipelines compile");
    Kit { ctx, pipelines }
}

impl Kit {
    fn buffer(&self, values: &[f32], label: &str) -> Buffer {
        upload(&self.ctx.device, values, label).expect("upload")
    }

    fn empty(&self, count: usize, label: &str) -> Buffer {
        allocate(&self.ctx.device, count, label).expect("allocate")
    }
}

fn assert_close(name: &str, got: &[f32], want: &[f32], abs: f32, rel: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        worst = worst.max(diff);
        assert!(
            diff <= abs + rel * w.abs(),
            "{name}: element {i} is {g}, expected {w} (diff {diff})"
        );
    }
    eprintln!("{name}: max abs diff {worst:e}");
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_attention(
    kit: &Kit,
    head_dim: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    kv_heads: usize,
    window: Option<usize>,
) -> Vec<f32> {
    let (qb, kb, vb) = (kit.buffer(q, "q"), kit.buffer(k, "k"), kit.buffer(v, "v"));
    let out = kit.empty(seq * heads * head_dim, "out");
    kit.ctx
        .run(|enc| {
            kit.pipelines.attention(
                enc,
                head_dim,
                &qb,
                &kb,
                &vb,
                &out,
                seq as u32,
                heads as u32,
                kv_heads as u32,
                window.map(|w| w as u32),
            )
        })
        .expect("attention runs");
    read_f32(&out, seq * heads * head_dim)
}

#[test]
fn attention_matches_the_cpu_kernel_for_both_head_widths_and_every_window_shape() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[8, 256, 512]);
    // Sequence lengths are not multiples of the 8-row query block or of either key tile (8 keys
    // at width 256, 4 at width 512). 53 positions exceed two query blocks plus the window of 5.
    let cases: &[(usize, Option<usize>)] = &[
        (1, None),
        (1, Some(3)),
        (37, None),
        (37, Some(0)),
        (37, Some(1)),
        (37, Some(5)),
        (37, Some(36)),
        (37, Some(500)),
        (53, Some(5)),
        (200, Some(40)),
        (200, None),
    ];
    for head_dim in [8usize, 256, 512] {
        for (heads, kv_heads) in [(4usize, 2usize), (4, 1)] {
            for q_scale in [0.15f32, 1.0] {
                for &(seq, window) in cases {
                    let seed = (seq * 31 + head_dim) as u64;
                    let q = rand_vec(seq * heads * head_dim, seed, q_scale);
                    let k = rand_vec(seq * kv_heads * head_dim, seed + 1, 1.0);
                    let v = rand_vec(seq * kv_heads * head_dim, seed + 2, 1.0);
                    let got =
                        run_attention(&kit, head_dim, &q, &k, &v, seq, heads, kv_heads, window);
                    let want =
                        bidirectional_attention(&q, &k, &v, seq, heads, kv_heads, head_dim, window);
                    assert_close(
                        &format!(
                            "attention hd={head_dim} heads={heads}/{kv_heads} q_scale={q_scale} seq={seq} window={window:?}"
                        ),
                        &got,
                        &want,
                        4e-5,
                        0.0,
                    );
                }
            }
        }
    }
}

/// Adds 1.0 to every value of position `pos` in `[seq, kv_heads, head_dim]`.
fn bump(values: &mut [f32], pos: usize, per_position: usize) {
    for v in &mut values[pos * per_position..(pos + 1) * per_position] {
        *v += 1.0;
    }
}

#[test]
fn the_window_contributes_at_distance_w_and_not_at_w_plus_one_in_both_directions() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[256, 512]);
    let (seq, heads, kv_heads, window, row) = (21usize, 4usize, 2usize, 3usize, 10usize);
    for head_dim in [256usize, 512] {
        let q = rand_vec(seq * heads * head_dim, 5, 0.15);
        let k = rand_vec(seq * kv_heads * head_dim, 6, 1.0);
        let v = rand_vec(seq * kv_heads * head_dim, 7, 1.0);
        let per_position = kv_heads * head_dim;
        let base = run_attention(
            &kit,
            head_dim,
            &q,
            &k,
            &v,
            seq,
            heads,
            kv_heads,
            Some(window),
        );
        let width = heads * head_dim;
        let row_of = |out: &[f32]| out[row * width..(row + 1) * width].to_vec();
        for (key, inside) in [
            (row + window, true),
            (row + window + 1, false),
            (row - window, true),
            (row - window - 1, false),
        ] {
            let mut bumped = v.clone();
            bump(&mut bumped, key, per_position);
            let out = run_attention(
                &kit,
                head_dim,
                &q,
                &k,
                &bumped,
                seq,
                heads,
                kv_heads,
                Some(window),
            );
            let (before, after) = (row_of(&base), row_of(&out));
            if inside {
                let change = before
                    .iter()
                    .zip(&after)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    change > 1e-3,
                    "hd {head_dim}: key {key} at distance {} must reach row {row} (change {change})",
                    key.abs_diff(row)
                );
            } else {
                assert!(
                    before
                        .iter()
                        .zip(&after)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "hd {head_dim}: key {key} at distance {} must not reach row {row}",
                    key.abs_diff(row)
                );
            }
        }
    }
}

#[test]
fn attention_is_not_causal_in_a_window_or_over_the_whole_sequence() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[256, 512]);
    let (seq, heads, kv_heads, row) = (21usize, 4usize, 2usize, 6usize);
    for head_dim in [256usize, 512] {
        let q = rand_vec(seq * heads * head_dim, 15, 0.15);
        let k = rand_vec(seq * kv_heads * head_dim, 16, 1.0);
        let v = rand_vec(seq * kv_heads * head_dim, 17, 1.0);
        let per_position = kv_heads * head_dim;
        let width = heads * head_dim;
        for (window, later_key) in [(None, 18usize), (Some(4usize), row + 4), (Some(4), row + 1)] {
            let base = run_attention(&kit, head_dim, &q, &k, &v, seq, heads, kv_heads, window);
            let mut bumped = v.clone();
            bump(&mut bumped, later_key, per_position);
            let out = run_attention(
                &kit, head_dim, &q, &k, &bumped, seq, heads, kv_heads, window,
            );
            let change = base[row * width..(row + 1) * width]
                .iter()
                .zip(&out[row * width..(row + 1) * width])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                change > 1e-3,
                "hd {head_dim} window {window:?}: a later key {later_key} must change row {row}"
            );
        }
    }
}

/// One output row in f64: attention of query `i` over every key it may see, for every head.
#[allow(clippy::too_many_arguments)]
fn oracle_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
    i: usize,
) -> Vec<f64> {
    let mut out = vec![0f64; heads * head_dim];
    for h in 0..heads {
        let kv = h / (heads / kv_heads);
        let keys: Vec<usize> = (0..seq)
            .filter(|&j| window.is_none_or(|w| i.abs_diff(j) <= w))
            .collect();
        let scores: Vec<f64> = keys
            .iter()
            .map(|&j| {
                (0..head_dim)
                    .map(|d| {
                        f64::from(q[(i * heads + h) * head_dim + d])
                            * f64::from(k[(j * kv_heads + kv) * head_dim + d])
                    })
                    .sum()
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        for (&j, w) in keys.iter().zip(&weights) {
            for d in 0..head_dim {
                out[h * head_dim + d] +=
                    w / total * f64::from(v[(j * kv_heads + kv) * head_dim + d]);
            }
        }
    }
    out
}

#[test]
fn attention_at_the_longest_published_sequence_matches_a_row_oracle() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[256, 512]);
    // 8473 is the token count of the longest published input; it is not a multiple of the query
    // block or of any key tile. Rows at both ends, around the middle and in the last block are
    // checked against an f64 oracle, since a full CPU reference at this length is out of reach.
    let (seq, heads, kv_heads) = (8473usize, 4usize, 2usize);
    let rows = [0usize, 1, 7, 8, 4000, 4236, 8464, 8471, 8472];
    for head_dim in [256usize, 512] {
        for (q_scale, window) in [(0.15f32, None), (0.15, Some(512usize)), (1.0, None)] {
            let seed = 900 + head_dim as u64;
            let q = rand_vec(seq * heads * head_dim, seed, q_scale);
            let k = rand_vec(seq * kv_heads * head_dim, seed + 1, 1.0);
            let v = rand_vec(seq * kv_heads * head_dim, seed + 2, 1.0);
            let out = run_attention(&kit, head_dim, &q, &k, &v, seq, heads, kv_heads, window);
            let width = heads * head_dim;
            let mut worst = 0f64;
            for &i in &rows {
                let want = oracle_row(&q, &k, &v, seq, heads, kv_heads, head_dim, window, i);
                for (g, w) in out[i * width..(i + 1) * width].iter().zip(&want) {
                    worst = worst.max((f64::from(*g) - w).abs());
                }
            }
            eprintln!(
                "attention seq={seq} hd={head_dim} q_scale={q_scale} window={window:?}: max abs diff {worst:e}"
            );
            assert!(
                worst < 5e-5,
                "seq {seq} hd {head_dim} q_scale {q_scale} window {window:?}: differs from the oracle by {worst}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Element-wise and row-wise kernels
// ---------------------------------------------------------------------------

#[test]
fn gelu_mul_matches_the_cpu_exact_tanh_gelu() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[]);
    // 4099 values: not a multiple of the 256-thread group, spanning the range where the
    // rational tanh approximation is furthest from the exact tanh.
    let count = 4099;
    let x: Vec<f32> = (0..count)
        .map(|i| -8.0 + 16.0 * i as f32 / (count - 1) as f32)
        .collect();
    for (name, other) in [
        ("plain gelu", vec![1.0f32; count]),
        ("gelu times a signal", rand_vec(count, 21, 1.0)),
    ] {
        let mut want = x.clone();
        gemma4_gelu_tanh(&mut want);
        elementwise_mul(&mut want, &other);
        let gate = kit.buffer(&x, "gate");
        let other_buffer = kit.buffer(&other, "other");
        kit.ctx
            .run(|enc| {
                kit.pipelines
                    .gelu_mul(enc, &gate, &other_buffer, count as u32);
                Ok(())
            })
            .expect("gelu_mul runs");
        assert_close(
            &format!("gelu_mul {name}"),
            &read_f32(&gate, count),
            &want,
            2e-6,
            4e-6,
        );
    }
}

#[test]
fn value_norm_matches_the_cpu_unweighted_rms_norm() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[]);
    let eps = 1e-6f32;
    for width in [4usize, 256, 512] {
        let rows = 37;
        let x = rand_vec(rows * width, 31 + width as u64, 3.0);
        let mut want = x.clone();
        rms_norm(&mut want, &vec![1.0; width], width, eps);
        let buffer = kit.buffer(&x, "x");
        kit.ctx
            .run(|enc| {
                kit.pipelines
                    .norm_noweight(enc, &buffer, rows as u32, width as u32, eps);
                Ok(())
            })
            .expect("norm_noweight runs");
        assert_close(
            &format!("norm_noweight width={width}"),
            &read_f32(&buffer, rows * width),
            &want,
            1e-6,
            5e-6,
        );
    }
}

#[test]
fn scale_matches_the_cpu_product_exactly() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[]);
    let count = 1001;
    let x = rand_vec(count, 41, 2.0);
    let factor = 0.883_883_5f32;
    let buffer = kit.buffer(&x, "x");
    kit.ctx
        .run(|enc| {
            kit.pipelines.scale(enc, &buffer, factor, count as u32);
            Ok(())
        })
        .expect("scale runs");
    let want: Vec<f32> = x.iter().map(|v| v * factor).collect();
    let got = read_f32(&buffer, count);
    assert!(
        got.iter()
            .zip(&want)
            .all(|(g, w)| g.to_bits() == w.to_bits()),
        "scale differs from the CPU product"
    );
}

// ---------------------------------------------------------------------------
// Reused kernels: each one is checked against the CPU function it stands in for, because the
// reuse is only sound where the semantics match exactly.
// ---------------------------------------------------------------------------

#[test]
fn the_reused_matmul_matches_the_cpu_matmul_including_a_weight_offset() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[]);
    // Shapes that are not multiples of the 16x16 tile, and a weight slice that starts partway
    // into a larger buffer, as the per-layer projection slices do.
    for (m, n, k) in [(70usize, 65usize, 37usize), (19, 23, 512), (1, 5, 3)] {
        let a = rand_vec(m * k, 51, 1.0);
        let b = rand_vec(n * k, 52, 1.0);
        let mut want = vec![0f32; m * n];
        matmul_bt(&a, &b, &mut want, m, k, n);
        let pad = 3 * n * k;
        let mut padded = rand_vec(pad, 53, 1.0);
        padded.extend_from_slice(&b);
        let (ab, bb, cb) = (
            kit.buffer(&a, "a"),
            kit.buffer(&padded, "b"),
            kit.empty(m * n, "c"),
        );
        kit.ctx
            .run(|enc| {
                kit.pipelines.matmul(
                    enc,
                    &ab,
                    &bb,
                    (pad * size_of::<f32>()) as u64,
                    &cb,
                    m as u32,
                    n as u32,
                    k as u32,
                );
                Ok(())
            })
            .expect("matmul runs");
        assert_close(
            &format!("matmul {m}x{n}x{k}"),
            &read_f32(&cb, m * n),
            &want,
            1e-5,
            2e-5,
        );
    }
}

#[test]
fn the_reused_weighted_rms_norm_uses_the_weight_as_given() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[]);
    let eps = 1e-6f32;
    for width in [4usize, 256, 512] {
        let rows = 29;
        let x = rand_vec(rows * width, 61, 3.0);
        let weight: Vec<f32> = rand_vec(width, 62, 0.4).iter().map(|w| 1.0 + w).collect();
        let mut want = x.clone();
        rms_norm(&mut want, &weight, width, eps);
        let (xb, wb) = (kit.buffer(&x, "x"), kit.buffer(&weight, "w"));
        kit.ctx
            .run(|enc| {
                kit.pipelines
                    .norm(enc, &xb, &wb, rows as u32, width as u32, eps);
                Ok(())
            })
            .expect("norm runs");
        assert_close(
            &format!("rms_norm width={width}"),
            &read_f32(&xb, rows * width),
            &want,
            1e-6,
            5e-6,
        );
    }
}

#[test]
fn the_reused_rope_matches_the_cpu_stride_half_rotation_at_both_thetas() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[]);
    let (tokens, heads) = (19usize, 4usize);
    for (head_dim, theta) in [(256usize, 1.0e4f64), (512, 1.0e6)] {
        let x = rand_vec(tokens * heads * head_dim, 71, 1.0);
        let inv_freq = gemma4_rope_inv_freq(head_dim, theta, None);
        let positions: Vec<u32> = (0..tokens as u32).collect();
        let (cos, sin) = gemma4_rope_cos_sin(&inv_freq, &positions);
        let mut want = x.clone();
        gemma4_apply_rope(&mut want, &cos, &sin, tokens, heads, head_dim);
        let half = head_dim / 2;
        let compact = |full: &[f32]| -> Vec<f32> {
            full.chunks_exact(head_dim)
                .flat_map(|row| row[..half].iter().copied())
                .collect()
        };
        let (xb, cb, sb) = (
            kit.buffer(&x, "x"),
            kit.buffer(&compact(&cos), "cos"),
            kit.buffer(&compact(&sin), "sin"),
        );
        kit.ctx
            .run(|enc| {
                kit.pipelines.rope(
                    enc,
                    &xb,
                    &cb,
                    &sb,
                    tokens as u32,
                    heads as u32,
                    head_dim as u32,
                );
                Ok(())
            })
            .expect("rope runs");
        assert_close(
            &format!("rope head_dim={head_dim} theta={theta}"),
            &read_f32(&xb, x.len()),
            &want,
            2e-6,
            0.0,
        );
    }
}

#[test]
fn the_reused_copy_and_add_are_exact() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[]);
    let count = 1001;
    let a = rand_vec(count, 81, 1.0);
    let b = rand_vec(count, 82, 1.0);
    let (ab, bb, cb) = (
        kit.buffer(&a, "a"),
        kit.buffer(&b, "b"),
        kit.empty(count, "c"),
    );
    kit.ctx
        .run(|enc| {
            kit.pipelines.copy(enc, &ab, &cb, count as u32);
            kit.pipelines.add(enc, &bb, &cb, count as u32);
            Ok(())
        })
        .expect("copy and add run");
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert!(
        read_f32(&cb, count)
            .iter()
            .zip(&want)
            .all(|(g, w)| g.to_bits() == w.to_bits()),
        "copy then add differs from the CPU sum"
    );
}

// ---------------------------------------------------------------------------
// Validation (no device needed)
// ---------------------------------------------------------------------------

#[test]
fn a_head_width_the_attention_kernel_cannot_map_is_rejected_before_any_device_access() {
    let config = |head_dim: usize| {
        let json = format!(
            r#"{{
              "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
              "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
              "head_dim": {head_dim}, "hidden_size_per_layer_input": 6, "embedding_dim": 10,
              "rms_norm_eps": 1e-6, "sliding_window": 2,
              "layer_types": ["sliding_attention", "full_attention"],
              "hidden_activation": "gelu_pytorch_tanh",
              "rope_parameters": {{
                "full_attention": {{"rope_theta": 1000.0, "rope_type": "default"}},
                "sliding_attention": {{"rope_theta": 100.0, "rope_type": "default"}}
              }}
            }}"#
        );
        EmbeddingGemma2Config::from_config_json_str(&json).expect("config parses")
    };
    validate_config(&config(8)).expect("a width of 8 is supported");
    for head_dim in [2usize, 6] {
        match validate_config(&config(head_dim)) {
            Err(InferenceError::InvalidInput(reason)) => {
                assert!(
                    reason.contains("multiple of 4"),
                    "wrong rejection: {reason}"
                );
            }
            other => panic!("head_dim {head_dim} must be rejected, got {other:?}"),
        }
    }
}
