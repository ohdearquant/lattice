//! Kernel-level tests for the EmbeddingGemma 2 vision tower on Metal.
//!
//! Every kernel and segment is compared with the CPU function it replaces: the axial RoPE kernel
//! against `vision_axial_rope_apply`, the reused attention kernel against `bidirectional_attention`
//! at the vision head width, one whole layer against the CPU tower, and the whole tower on a real
//! checkpoint against the CPU tower. Whole-model parity lives with the model tests.

use metal::Buffer;

use super::super::{Context, Pipelines, allocate, read_f32, upload};
use super::MetalEmbeddingGemma2VisionState;
use crate::error::InferenceError;
use crate::forward::metal_embeddinggemma2::MetalEmbeddingGemma2State;
use crate::measurement::gpu_test_lock;
use crate::model::embeddinggemma2::{EmbeddingGemma2Model, bidirectional_attention};
use crate::model::embeddinggemma2_image::{EmbeddingGemma2Rgb8Image, prepare_image};
use crate::model::embeddinggemma2_vision::{VisionConfig, VisionLayerWeights, VisionWeights};
use crate::model::gemma4_ops::{vision_axial_rope_apply, vision_axial_rope_inv_freq};

fn enforce() -> bool {
    std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some()
}

/// True (after saying so) when this machine has no Metal device and the run is not enforcing one.
fn skip_without_device() -> bool {
    if ::metal::Device::system_default().is_some() {
        return false;
    }
    assert!(!enforce(), "Metal device required under enforcement");
    eprintln!("SKIP embeddinggemma2 vision Metal: no Metal device");
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

fn max_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
}

/// Flattens `(x, y)` patch positions into the interleaved `f32` buffer the kernel reads.
fn position_values(positions: &[[usize; 2]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(positions.len() * 2);
    for &[x, y] in positions {
        out.push(x as f32);
        out.push(y as f32);
    }
    out
}

/// Raster `(x, y)` positions over a `rows x cols` grid.
fn raster(rows: usize, cols: usize) -> Vec<[usize; 2]> {
    (0..rows * cols).map(|i| [i % cols, i / cols]).collect()
}

// ---------------------------------------------------------------------------
// Axial RoPE kernel
// ---------------------------------------------------------------------------

#[test]
fn vision_rope_matches_the_cpu_axial_rope_on_several_grids_and_head_counts() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[]);
    let head_dim = 64usize;
    // A square grid, two non-square grids, a single row and a single patch; one to three heads so
    // a head-count dependent indexing error cannot hide in a one-head case.
    let cases: &[(usize, usize, usize)] = &[
        (3, 3, 1),
        (3, 4, 2),
        (1, 7, 3),
        (5, 5, 2),
        (1, 1, 1),
        (3, 1, 3),
        (7, 2, 2),
    ];
    for &(rows, cols, heads) in cases {
        let positions = raster(rows, cols);
        let tokens = positions.len();
        let inv_freq = vision_axial_rope_inv_freq(head_dim, 100.0);
        let x = rand_vec(
            tokens * heads * head_dim,
            100 + (rows * 31 + cols) as u64,
            2.0,
        );

        let mut want = x.clone();
        vision_axial_rope_apply(&mut want, &positions, &inv_freq, heads, head_dim);

        let xb = kit.buffer(&x, "x");
        let pb = kit.buffer(&position_values(&positions), "positions");
        let fb = kit.buffer(&inv_freq, "inv_freq");
        kit.ctx
            .run(|enc| {
                kit.pipelines.vision_rope(
                    enc,
                    &xb,
                    &pb,
                    &fb,
                    tokens as u32,
                    heads as u32,
                    head_dim as u32,
                );
                Ok(())
            })
            .expect("vision rope runs");
        assert_close(
            &format!("vision_rope {rows}x{cols} heads={heads}"),
            &read_f32(&xb, x.len()).expect("read back"),
            &want,
            1e-5,
            0.0,
        );
    }
}

// ---------------------------------------------------------------------------
// Attention at the vision head width
// ---------------------------------------------------------------------------

#[test]
fn vision_attention_at_head_width_64_matches_the_cpu_kernel() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let kit = kit(&[64]);
    // Odd sequence lengths that are not multiples of the 8-row query block, both ungrouped and
    // grouped queries (two query heads per key/value head), so the head mapping is exercised.
    for (seq, heads, kv_heads) in [(1usize, 1usize, 1usize), (7, 2, 1), (13, 2, 1), (9, 3, 3)] {
        let head_dim = 64usize;
        let q = rand_vec(seq * heads * head_dim, 200 + seq as u64, 1.0);
        let k = rand_vec(seq * kv_heads * head_dim, 300 + seq as u64, 1.0);
        let v = rand_vec(seq * kv_heads * head_dim, 400 + seq as u64, 1.0);
        let (qb, kb, vb) = (
            kit.buffer(&q, "q"),
            kit.buffer(&k, "k"),
            kit.buffer(&v, "v"),
        );
        let out = kit.empty(seq * heads * head_dim, "out");
        kit.ctx
            .run(|enc| {
                kit.pipelines.attention(
                    enc,
                    64,
                    &qb,
                    &kb,
                    &vb,
                    &out,
                    seq as u32,
                    heads as u32,
                    kv_heads as u32,
                    None,
                )
            })
            .expect("attention runs");
        let want = bidirectional_attention(&q, &k, &v, seq, heads, kv_heads, head_dim, None);
        assert_close(
            &format!("vision attention seq={seq} heads={heads} kv={kv_heads}"),
            &read_f32(&out, seq * heads * head_dim).expect("read back"),
            &want,
            2e-5,
            2e-5,
        );
    }
}

// ---------------------------------------------------------------------------
// One vision layer and the whole synthetic tower
// ---------------------------------------------------------------------------

/// A vision configuration with a small hidden width but the released head width of 64, so the
/// axial RoPE pairing and the attention geometry match production while the tensors stay tiny.
fn synthetic_config(layers: usize) -> VisionConfig {
    VisionConfig {
        hidden_size: 64,
        intermediate_size: 32,
        num_hidden_layers: layers,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 64,
        patch_size: 16,
        position_embedding_size: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 100.0,
        text_hidden_size: 16,
    }
}

fn pattern(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let raw = ((seed + 1) * 97 + (i + 1) * 31) % 41;
            (raw as f32 - 20.0) / 40.0
        })
        .collect()
}

fn synthetic_weights(cfg: &VisionConfig) -> VisionWeights {
    let h = cfg.hidden_size;
    let d = cfg.head_dim;
    let q_dim = cfg.num_attention_heads * d;
    let kv_dim = cfg.num_key_value_heads * d;
    let layers = (0..cfg.num_hidden_layers)
        .map(|layer| VisionLayerWeights {
            input_layernorm: vec![1.0; h],
            post_attention_layernorm: vec![1.0; h],
            pre_feedforward_layernorm: vec![1.0; h],
            post_feedforward_layernorm: vec![1.0; h],
            q_norm: vec![1.0; d],
            k_norm: vec![1.0; d],
            q_proj: pattern(q_dim * h, layer * 13 + 1),
            k_proj: pattern(kv_dim * h, layer * 13 + 2),
            v_proj: pattern(kv_dim * h, layer * 13 + 3),
            o_proj: pattern(h * q_dim, layer * 13 + 4),
            gate_proj: pattern(cfg.intermediate_size * h, layer * 13 + 5),
            up_proj: pattern(cfg.intermediate_size * h, layer * 13 + 6),
            down_proj: pattern(h * cfg.intermediate_size, layer * 13 + 7),
        })
        .collect();
    VisionWeights {
        config: cfg.clone(),
        layers,
        input_proj: pattern(h * 3 * cfg.patch_size * cfg.patch_size, 3),
        position_embedding_table: pattern(2 * cfg.position_embedding_size * h, 4),
        embedding_projection: pattern(cfg.text_hidden_size * h, 5),
    }
}

/// The bound on the largest per-element difference for the small synthetic tower: the two paths
/// differ only in summation order and in the libm versus Metal `exp`, `tanh` and `rsqrt`, which
/// give differences near 1e-6 on values of order one. The bound scales with the largest CPU
/// magnitude so a wrong kernel (error of the order of the values) still fails.
fn tower_bound(cpu: &[f32]) -> f32 {
    1e-4 * cpu.iter().fold(1f32, |m, v| m.max(v.abs()))
}

#[test]
fn one_vision_layer_on_metal_matches_the_cpu_layer() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let cfg = synthetic_config(1);
    let weights = synthetic_weights(&cfg);
    let kit = kit(&[cfg.head_dim]);
    let mut state = MetalEmbeddingGemma2VisionState::new(&kit.ctx.device, &weights)
        .expect("vision state builds");

    let positions = raster(3, 3);
    let patch = rand_vec(positions.len() * cfg.hidden_size, 501, 1.5);

    // With one layer the CPU tower is exactly the CPU `run_layer` plus the encoder capture, so a
    // one-layer configuration isolates the layer implementation without widening its visibility.
    let cpu = weights
        .tower_forward(&patch, &positions, &mut || false)
        .expect("cpu tower runs");

    let (stages, _) = state
        .tower_forward(
            &kit.ctx,
            &kit.pipelines,
            &weights,
            &patch,
            &positions,
            &mut || false,
        )
        .expect("metal tower runs");
    assert_eq!(stages.encoder.len(), cpu.encoder.len());
    let limit = tower_bound(&cpu.encoder);
    let diff = max_diff(&stages.encoder, &cpu.encoder);
    eprintln!("one layer: max abs diff {diff:e} (bound {limit:e})");
    assert!(diff <= limit, "one layer differs by {diff}, bound {limit}");
    for (name, got, want) in [
        ("q normed", &stages.layer_0_q_normed, &cpu.layer_0_q_normed),
        ("k normed", &stages.layer_0_k_normed, &cpu.layer_0_k_normed),
        (
            "q rotated",
            &stages.layer_0_q_rotated,
            &cpu.layer_0_q_rotated,
        ),
        (
            "k rotated",
            &stages.layer_0_k_rotated,
            &cpu.layer_0_k_rotated,
        ),
    ] {
        assert!(!want.is_empty(), "{name}: the CPU tower captured nothing");
        assert_eq!(got.len(), want.len(), "{name}: length mismatch");
        let limit = tower_bound(want);
        let diff = max_diff(got, want);
        eprintln!("one layer {name}: max abs diff {diff:e} (bound {limit:e})");
        assert!(diff <= limit, "{name} differs by {diff}, bound {limit}");
    }
}

#[test]
fn the_synthetic_vision_tower_preserves_cancellation_and_finiteness() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let cfg = synthetic_config(2);
    let weights = synthetic_weights(&cfg);
    let kit = kit(&[cfg.head_dim]);
    let mut state = MetalEmbeddingGemma2VisionState::new(&kit.ctx.device, &weights)
        .expect("vision state builds");
    let positions = raster(3, 3);
    let patch = rand_vec(positions.len() * cfg.hidden_size, 601, 1.0);

    // A predicate that reports cancellation stops the tower before any layer runs.
    assert!(matches!(
        state.tower_forward(
            &kit.ctx,
            &kit.pipelines,
            &weights,
            &patch,
            &positions,
            &mut || true,
        ),
        Err(InferenceError::Inference(_))
    ));

    // A patch buffer that does not match the position count is refused.
    assert!(matches!(
        state.tower_forward(
            &kit.ctx,
            &kit.pipelines,
            &weights,
            &patch[..patch.len() - 1],
            &positions,
            &mut || false,
        ),
        Err(InferenceError::Inference(_))
    ));

    let (stages, _) = state
        .tower_forward(
            &kit.ctx,
            &kit.pipelines,
            &weights,
            &patch,
            &positions,
            &mut || false,
        )
        .expect("metal tower runs");
    assert!(stages.encoder.iter().all(|v| v.is_finite()));
    assert_eq!(stages.patch, patch);
}

#[test]
fn a_vision_state_refuses_weights_from_a_different_model() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let weights = synthetic_weights(&synthetic_config(1));
    let other = synthetic_weights(&synthetic_config(2));
    let kit = kit(&[64]);
    let state = MetalEmbeddingGemma2VisionState::new(&kit.ctx.device, &weights)
        .expect("vision state builds");
    state.check_weights(&weights).expect("same weights pass");
    match state.check_weights(&other) {
        Err(InferenceError::InvalidInput(reason)) => {
            assert!(
                reason.contains("different model"),
                "wrong rejection: {reason}"
            );
        }
        result => panic!("weights from another model must be refused, got {result:?}"),
    }
    // A non-finite weight is refused when the state is built, before any device access.
    let mut corrupt = synthetic_weights(&synthetic_config(1));
    corrupt.layers[0].down_proj[3] = f32::NAN;
    assert!(matches!(
        MetalEmbeddingGemma2VisionState::new(&kit.ctx.device, &corrupt),
        Err(InferenceError::InvalidInput(_))
    ));
}

// ---------------------------------------------------------------------------
// Whole tower on the real checkpoint
// ---------------------------------------------------------------------------

/// The released checkpoint directory: `LATTICE_EMBEDDINGGEMMA2_MODEL_DIR`, else
/// `~/.lattice/models/embeddinggemma-2`. `None` (after saying so) when it is not present and the
/// run is not enforcing one.
fn checkpoint_dir() -> Option<std::path::PathBuf> {
    let dir = match std::env::var_os("LATTICE_EMBEDDINGGEMMA2_MODEL_DIR") {
        Some(value) => std::path::PathBuf::from(value),
        None => {
            let home = std::env::var_os("HOME")?;
            std::path::PathBuf::from(home)
                .join(".lattice")
                .join("models")
                .join("embeddinggemma-2")
        }
    };
    if dir.join("model.safetensors").is_file()
        && dir.join("config.json").is_file()
        && dir.join("tokenizer.json").is_file()
    {
        return Some(dir);
    }
    assert!(
        !enforce(),
        "LATTICE_EMBEDDINGGEMMA2_MODEL_DIR={} is not a complete checkpoint while \
         LATTICE_METAL_TEST_ENFORCE=1",
        dir.display()
    );
    eprintln!(
        "SKIP embeddinggemma2 vision Metal: no checkpoint at {}",
        dir.display()
    );
    None
}

/// A square gradient image. `side` must be a multiple of 48 (16-pixel patches in 3x3 pooling
/// cells); 240 is a 15x15 patch grid, 225 valid patches and 25 soft rows.
fn gradient_image(side: usize) -> EmbeddingGemma2Rgb8Image {
    let mut bytes = Vec::with_capacity(side * side * 3);
    for y in 0..side {
        for x in 0..side {
            bytes.push((x * 255 / side.max(1)) as u8);
            bytes.push((y * 255 / side.max(1)) as u8);
            bytes.push(((x + y) * 255 / (2 * side).max(1)) as u8);
        }
    }
    EmbeddingGemma2Rgb8Image::new(side, side, bytes).expect("square image")
}

#[test]
fn the_whole_tower_matches_the_cpu_tower_on_the_released_checkpoint() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let Some(dir) = checkpoint_dir() else {
        return;
    };
    let model = EmbeddingGemma2Model::from_model_dir(&dir).expect("checkpoint loads");
    let mut state = MetalEmbeddingGemma2State::new(&model).expect("text state builds");
    let weights = model.vision_weights().expect("vision weights load");

    let prepared = prepare_image(&gradient_image(240)).expect("image prepares");
    let cpu = weights
        .encode(&prepared, &mut || false)
        .expect("cpu tower runs");
    let (metal, _) = state
        .vision_forward(&weights, &prepared, &mut || false)
        .expect("metal tower runs");

    // The stream after layer 0 is compared first: a pairing or layout error in the first layer
    // shows there before it is mixed through fifteen more layers.
    for (name, got, want) in [
        ("layer 0", &metal.layer_0, &cpu.stages.layer_0),
        ("layer 8", &metal.layer_8, &cpu.stages.layer_8),
        ("layer 15", &metal.layer_15, &cpu.stages.layer_15),
    ] {
        assert_eq!(got.len(), want.len(), "{name}: length mismatch");
        let limit = tower_bound(want);
        let diff = max_diff(got, want);
        eprintln!("tower {name}: max abs diff {diff:e} (bound {limit:e})");
        assert!(
            diff <= limit,
            "tower {name} differs by {diff}, bound {limit}"
        );
    }
    assert_eq!(metal.encoder.len(), cpu.stages.encoder.len());
    let limit = 1e-4 * cpu.stages.encoder.iter().fold(1f32, |m, v| m.max(v.abs()));
    let diff = max_diff(&metal.encoder, &cpu.stages.encoder);
    eprintln!("tower encoder: max abs diff {diff:e} (bound {limit:e})");
    assert!(
        diff <= limit,
        "tower encoder differs by {diff}, bound {limit}"
    );
    assert_eq!(metal.projected.len(), cpu.stages.projected.len());
    assert!(max_diff(&metal.projected, &cpu.stages.projected) <= limit);
}
