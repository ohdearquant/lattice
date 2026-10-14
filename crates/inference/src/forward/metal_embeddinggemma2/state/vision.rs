//! EmbeddingGemma 2 vision tower, f32 forward pass on Metal.
//!
//! The tower runs on the valid patch rows only, exactly as the CPU reference does, and each
//! layer runs as two GPU segments, one command buffer each: the attention block (input RMSNorm,
//! q/k/v projections, the QK norm with a scale-free V norm, the axial RoPE, bidirectional
//! attention at head width 64 with scale 1.0 and no window, the output projection, the
//! post-attention norm and residual add) and the GeGLU MLP block. Both run their matrix products
//! on simdgroup matrices where the device has them (Apple7 family or later), and on the vector
//! kernels otherwise. The residual stream lives in a shared `MTLBuffer` the host writes once and
//! reads at the captured layers.
//!
//! Weights upload once as f32. A device failure returns an error and is never replayed on the
//! host.

use metal::Buffer;

use super::{Context, Pipelines, allocate, as_f32, elements, invalid, runtime, upload, write_f32};
use crate::error::InferenceError;
use crate::model::embeddinggemma2_vision::{
    VisionConfig, VisionLayerWeights, VisionStages, VisionWeights,
};
use crate::model::gemma4_ops::vision_axial_rope_inv_freq;

#[cfg(test)]
mod tests;

fn vision_error(message: impl Into<String>) -> InferenceError {
    InferenceError::Inference(format!("EmbeddingGemma 2 vision Metal: {}", message.into()))
}

/// A sampled hash of a vision configuration and its tensors, used to tell two loaded models
/// apart. It is not a content hash of every tensor.
fn fingerprint(cfg: &VisionConfig, w: &VisionWeights) -> u64 {
    fn mix(hash: &mut u64, value: u64) {
        *hash ^= value;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn sample(hash: &mut u64, values: &[f32]) {
        mix(hash, values.len() as u64);
        let step = (values.len() / 256).max(1);
        for value in values.iter().step_by(step) {
            mix(hash, u64::from(value.to_bits()));
        }
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in [
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.patch_size,
        cfg.position_embedding_size,
        cfg.text_hidden_size,
    ] {
        mix(&mut hash, value as u64);
    }
    sample(&mut hash, &w.input_proj);
    sample(&mut hash, &w.position_embedding_table);
    sample(&mut hash, &w.embedding_projection);
    for layer in &w.layers {
        sample(&mut hash, &layer.q_proj);
        sample(&mut hash, &layer.down_proj);
    }
    hash
}

/// Rejects geometry the reuse cannot serve, before any device access.
fn validate_config(cfg: &VisionConfig) -> Result<(), InferenceError> {
    for (name, value) in [
        ("hidden_size", cfg.hidden_size),
        ("intermediate_size", cfg.intermediate_size),
        ("num_attention_heads", cfg.num_attention_heads),
        ("num_key_value_heads", cfg.num_key_value_heads),
        ("head_dim", cfg.head_dim),
        ("patch_size", cfg.patch_size),
        ("position_embedding_size", cfg.position_embedding_size),
        ("num_hidden_layers", cfg.num_hidden_layers),
    ] {
        if value == 0 {
            return Err(invalid(format!("vision {name} must be positive")));
        }
    }
    if cfg.head_dim != 64 {
        return Err(invalid(format!(
            "vision head_dim {} is not supported; the axial RoPE pairing and the attention \
             kernel are built for 64-wide heads",
            cfg.head_dim
        )));
    }
    if !cfg
        .num_attention_heads
        .is_multiple_of(cfg.num_key_value_heads)
    {
        return Err(invalid(format!(
            "vision num_attention_heads {} must be a multiple of num_key_value_heads {}",
            cfg.num_attention_heads, cfg.num_key_value_heads
        )));
    }
    if !cfg.rms_norm_eps.is_finite() || cfg.rms_norm_eps <= 0.0 {
        return Err(invalid("vision rms_norm_eps must be finite and positive"));
    }
    if !cfg.rope_theta.is_finite() || cfg.rope_theta <= 1.0 {
        return Err(invalid("vision rope_theta must be finite and above 1"));
    }
    // Every uploaded tensor must fit u32 indexing and the host slice capacity.
    let hidden = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let patch = 3 * cfg.patch_size * cfg.patch_size;
    let mut buffers = vec![
        ("input_proj".to_string(), hidden * patch),
        (
            "position_embedding_table".to_string(),
            2 * cfg.position_embedding_size * hidden,
        ),
        (
            "embedding_projection".to_string(),
            cfg.text_hidden_size * hidden,
        ),
    ];
    for l in 0..cfg.num_hidden_layers {
        for (name, count) in [
            ("input_layernorm", hidden),
            ("post_attention_layernorm", hidden),
            ("pre_feedforward_layernorm", hidden),
            ("post_feedforward_layernorm", hidden),
            ("q_norm", cfg.head_dim),
            ("k_norm", cfg.head_dim),
            ("q_proj", q * hidden),
            ("k_proj", kv * hidden),
            ("v_proj", kv * hidden),
            ("o_proj", hidden * q),
            ("gate_proj", cfg.intermediate_size * hidden),
            ("up_proj", cfg.intermediate_size * hidden),
            ("down_proj", hidden * cfg.intermediate_size),
        ] {
            buffers.push((format!("layer {l} {name}"), count));
        }
    }
    for (name, count) in buffers {
        elements(&format!("vision {name}"), count, 1)?;
    }
    Ok(())
}

fn validate_values(name: &str, values: &[f32], expected: usize) -> Result<(), InferenceError> {
    if values.len() != expected {
        return Err(invalid(format!(
            "{name} has {} elements, expected {expected}",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("{name} contains a non-finite weight")));
    }
    Ok(())
}

/// Validates every tensor the vision state uploads, before any device access.
fn validate_weights(cfg: &VisionConfig, w: &VisionWeights) -> Result<(), InferenceError> {
    if w.layers.len() != cfg.num_hidden_layers {
        return Err(invalid(format!(
            "vision weights contain {} layers, expected {}",
            w.layers.len(),
            cfg.num_hidden_layers
        )));
    }
    let hidden = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    validate_values(
        "vision input_proj",
        &w.input_proj,
        hidden * 3 * cfg.patch_size * cfg.patch_size,
    )?;
    validate_values(
        "vision position_embedding_table",
        &w.position_embedding_table,
        2 * cfg.position_embedding_size * hidden,
    )?;
    validate_values(
        "vision embedding_projection",
        &w.embedding_projection,
        cfg.text_hidden_size * hidden,
    )?;
    for (l, layer) in w.layers.iter().enumerate() {
        for (name, values, expected) in [
            ("input_layernorm", &layer.input_layernorm, hidden),
            (
                "post_attention_layernorm",
                &layer.post_attention_layernorm,
                hidden,
            ),
            (
                "pre_feedforward_layernorm",
                &layer.pre_feedforward_layernorm,
                hidden,
            ),
            (
                "post_feedforward_layernorm",
                &layer.post_feedforward_layernorm,
                hidden,
            ),
            ("q_norm", &layer.q_norm, cfg.head_dim),
            ("k_norm", &layer.k_norm, cfg.head_dim),
            ("q_proj", &layer.q_proj, q * hidden),
            ("k_proj", &layer.k_proj, kv * hidden),
            ("v_proj", &layer.v_proj, kv * hidden),
            ("o_proj", &layer.o_proj, hidden * q),
            (
                "gate_proj",
                &layer.gate_proj,
                cfg.intermediate_size * hidden,
            ),
            ("up_proj", &layer.up_proj, cfg.intermediate_size * hidden),
            (
                "down_proj",
                &layer.down_proj,
                hidden * cfg.intermediate_size,
            ),
        ] {
            validate_values(&format!("vision layer {l} {name}"), values, expected)?;
        }
    }
    Ok(())
}

/// The per-layer vision tensors that live on the device.
struct GpuVisionLayer {
    input_norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    o: Buffer,
    post_attention_norm: Buffer,
    pre_ff_norm: Buffer,
    post_ff_norm: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
}

impl GpuVisionLayer {
    fn new(
        device: &metal::Device,
        l: usize,
        w: &VisionLayerWeights,
    ) -> Result<Self, InferenceError> {
        let up = |values: &[f32], name: &str| upload(device, values, &format!("eg2v.l{l}.{name}"));
        Ok(Self {
            input_norm: up(&w.input_layernorm, "input_norm")?,
            q: up(&w.q_proj, "q_proj")?,
            k: up(&w.k_proj, "k_proj")?,
            v: up(&w.v_proj, "v_proj")?,
            q_norm: up(&w.q_norm, "q_norm")?,
            k_norm: up(&w.k_norm, "k_norm")?,
            o: up(&w.o_proj, "o_proj")?,
            post_attention_norm: up(&w.post_attention_layernorm, "post_attention_norm")?,
            pre_ff_norm: up(&w.pre_feedforward_layernorm, "pre_ff_norm")?,
            post_ff_norm: up(&w.post_feedforward_layernorm, "post_ff_norm")?,
            gate: up(&w.gate_proj, "gate_proj")?,
            up: up(&w.up_proj, "up_proj")?,
            down: up(&w.down_proj, "down_proj")?,
        })
    }
}

/// Shared-stream buffers for the valid patch rows, reallocated when the row count grows.
struct VisionScratch {
    rows: usize,
    /// The residual stream.
    hidden: Buffer,
    /// A normed copy of the stream each block's projections consume.
    normed: Buffer,
    /// Two floats per row, the patch's x then y grid coordinate, for the axial RoPE.
    positions: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    /// Layer 0's q and k after the QK norm and before the RoPE, kept for the stage captures.
    q_normed: Buffer,
    k_normed: Buffer,
    /// The attention kernel's output.
    context: Buffer,
    /// The output projection's result.
    o_proj: Buffer,
    /// The MLP block's down-projection result.
    branch: Buffer,
    /// The gated projection, intermediate-wide.
    gate: Buffer,
    /// The up projection, intermediate-wide.
    up: Buffer,
}

/// Per-segment wall time of one tower pass, in microseconds.
///
/// The two GPU fields are the sum of their command buffers' `GPUStartTime`/`GPUEndTime` spans,
/// the two CPU fields are host wall time measured around the host work beside them, and
/// `tower_us` is the whole pass, handoffs included. Not a stable API: `#[doc(hidden)]` is a convention for the
/// `dump_embeddinggemma2_vision` example, which reports the segments of a measured run.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisionTimings {
    /// Sum of the attention-block command buffers' GPU spans.
    pub attention_gpu_us: u64,
    /// Sum of the MLP command buffers' GPU spans.
    pub mlp_gpu_us: u64,
    /// Host time before the attention block: writing the patch rows and their grid positions
    /// into the shared buffers, on the first layer only.
    pub cpu_pre_attention_us: u64,
    /// Host time after the attention block: reading layer 0's q/k stage captures back.
    pub cpu_post_attention_us: u64,
    /// Wall time of the whole tower pass, handoffs included.
    pub tower_us: u64,
}

impl VisionTimings {
    /// Folds one image part's pass into an accumulating total.
    pub(crate) fn add(&mut self, other: &VisionTimings) {
        self.attention_gpu_us += other.attention_gpu_us;
        self.mlp_gpu_us += other.mlp_gpu_us;
        self.cpu_pre_attention_us += other.cpu_pre_attention_us;
        self.cpu_post_attention_us += other.cpu_post_attention_us;
        self.tower_us += other.tower_us;
    }
}

fn micros(seconds: f64) -> u64 {
    (seconds * 1e6).max(0.0).round().min(u64::MAX as f64) as u64
}

/// Layers whose post-layer stream is captured for a parity comparison.
fn stage_is_captured(index: usize) -> bool {
    index == 0 || index == 8 || index == 15
}

/// EmbeddingGemma 2 f32 vision tower on Metal with resident weights.
///
/// Build it once from a loaded model's vision weights with
/// [`MetalEmbeddingGemma2VisionState::new`] and run it through
/// `EmbeddingGemma2Model::encode_parts_at_widths_metal`. The patch embed, pooling and bridge stay
/// on the host in this step; only the tower layers are split. The caller holds the machine GPU
/// lock for the whole GPU operation.
///
/// The reference stage captures (the layer-0 q/k after the norm and after RoPE, and the stream
/// after layers 0, 8 and 15) are filled exactly as the CPU tower fills them, so a dump of the
/// two paths compares stage for stage.
pub(super) struct MetalEmbeddingGemma2VisionState {
    layers: Vec<GpuVisionLayer>,
    cfg: VisionConfig,
    /// Two per-axis blocks of inverse frequencies, shared by every layer.
    inv_freq: Buffer,
    fingerprint: u64,
    scratch: Option<VisionScratch>,
}

impl MetalEmbeddingGemma2VisionState {
    /// Upload one model's vision tower to the GPU as f32.
    ///
    /// # Errors
    /// Returns an error for an unsupported head width, tensor shapes or non-finite weights,
    /// unavailable Metal, or shader and pipeline creation failures. Validation precedes device
    /// access.
    pub(super) fn new(
        device: &metal::Device,
        weights: &VisionWeights,
    ) -> Result<Self, InferenceError> {
        let cfg = weights.config.clone();
        validate_config(&cfg)?;
        validate_weights(&cfg, weights)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(cfg.num_hidden_layers)
            .map_err(|error| runtime(format!("vision layer allocation failed: {error}")))?;
        for (l, layer) in weights.layers.iter().enumerate() {
            layers.push(GpuVisionLayer::new(device, l, layer)?);
        }
        let inv_freq = vision_axial_rope_inv_freq(cfg.head_dim, cfg.rope_theta);
        if inv_freq.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "vision rope_theta produces a non-finite inverse frequency",
            ));
        }
        let inv_freq = upload(device, &inv_freq, "eg2v.inv_freq")?;
        Ok(Self {
            layers,
            fingerprint: fingerprint(&cfg, weights),
            inv_freq,
            cfg,
            scratch: None,
        })
    }

    /// Rejects weights other than the ones this state was built from.
    pub(super) fn check_weights(&self, weights: &VisionWeights) -> Result<(), InferenceError> {
        if weights.config != self.cfg || fingerprint(&weights.config, weights) != self.fingerprint {
            return Err(invalid(
                "this vision state was built from a different model; build one per model",
            ));
        }
        Ok(())
    }

    /// Runs the whole tower on the valid patch rows and returns the reference stage captures with
    /// the per-segment wall times of the pass. `should_cancel` is polled between layers;
    /// in-flight GPU work has drained before each poll.
    pub(super) fn tower_forward(
        &mut self,
        ctx: &Context,
        pipelines: &Pipelines,
        weights: &VisionWeights,
        patch: &[f32],
        positions: &[[usize; 2]],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<(VisionStages, VisionTimings), InferenceError> {
        self.check_weights(weights)?;
        let hidden = self.cfg.hidden_size;
        let rows = positions.len();
        if rows == 0 || patch.len() != rows * hidden {
            return Err(vision_error(format!(
                "patch rows must be {rows}x{hidden}, got {}",
                patch.len()
            )));
        }
        // The RoPE kernel reads coordinates as f32, exact only up to 2^24.
        if positions
            .iter()
            .flatten()
            .any(|&coordinate| coordinate > 1 << 24)
        {
            return Err(vision_error("a patch grid coordinate exceeds 2^24"));
        }
        let pass_start = std::time::Instant::now();
        self.ensure_scratch(ctx, rows)?;
        let mut stages = VisionStages {
            patch: patch.to_vec(),
            layer_0: Vec::new(),
            layer_8: Vec::new(),
            layer_15: Vec::new(),
            encoder: Vec::new(),
            pooler_values: Vec::new(),
            pooler_occupancy: Vec::new(),
            valid: Vec::new(),
            projected: Vec::new(),
            layer_0_q_normed: Vec::new(),
            layer_0_k_normed: Vec::new(),
            layer_0_q_rotated: Vec::new(),
            layer_0_k_rotated: Vec::new(),
            hidden_size: hidden,
            text_hidden: self.cfg.text_hidden_size,
            heads: self.cfg.num_attention_heads,
            head_dim: self.cfg.head_dim,
        };
        let mut timings = VisionTimings::default();
        let write_start = std::time::Instant::now();
        {
            let Some(scratch) = self.scratch.as_ref() else {
                return Err(runtime("vision scratch buffers missing after allocation"));
            };
            write_f32(&scratch.hidden, patch)?;
            let coordinates: Vec<f32> = positions
                .iter()
                .flat_map(|&[x, y]| [x as f32, y as f32])
                .collect();
            write_f32(&scratch.positions, &coordinates)?;
        }
        timings.cpu_pre_attention_us += micros(write_start.elapsed().as_secs_f64());
        for index in 0..self.cfg.num_hidden_layers {
            if should_cancel() {
                return Err(vision_error("cancelled during the vision tower"));
            }
            self.run_layer(
                ctx,
                pipelines,
                index,
                rows,
                index == 0,
                &mut stages,
                &mut timings,
            )?;
            if stage_is_captured(index) {
                let stream = {
                    let Some(scratch) = self.scratch.as_ref() else {
                        return Err(runtime("vision scratch buffers missing after allocation"));
                    };
                    as_f32(&scratch.hidden, rows * hidden)?.to_vec()
                };
                match index {
                    0 => stages.layer_0 = stream,
                    8 => stages.layer_8 = stream,
                    _ => stages.layer_15 = stream,
                }
            }
        }
        let encoder = {
            let Some(scratch) = self.scratch.as_ref() else {
                return Err(runtime("vision scratch buffers missing after allocation"));
            };
            as_f32(&scratch.hidden, rows * hidden)?.to_vec()
        };
        if encoder.iter().any(|value| !value.is_finite()) {
            return Err(runtime("the vision tower produced a non-finite value"));
        }
        stages.encoder = encoder;
        timings.tower_us = micros(pass_start.elapsed().as_secs_f64());
        Ok((stages, timings))
    }

    /// One encoder layer: the attention block, then the MLP block, one command buffer each.
    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &mut self,
        ctx: &Context,
        pipelines: &Pipelines,
        index: usize,
        rows: usize,
        capture: bool,
        stages: &mut VisionStages,
        timings: &mut VisionTimings,
    ) -> Result<(), InferenceError> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let eps = cfg.rms_norm_eps;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let intermediate = cfg.intermediate_size;
        let gpu = self.layers.get(index).ok_or_else(|| {
            vision_error(format!("missing device tensors for vision layer {index}"))
        })?;
        let inv_freq = &self.inv_freq;
        let Some(scratch) = self.scratch.as_ref() else {
            return Err(runtime("vision scratch buffers missing after allocation"));
        };

        // GPU: input norm, q/k/v projections, QK norm with a scale-free V norm, axial RoPE,
        // attention, output projection, post-attention norm, residual add into the stream.
        let span = ctx.run_timed(|enc| {
            let p = pipelines;
            let (r, h) = (rows as u32, hidden as u32);
            p.copy(enc, &scratch.hidden, &scratch.normed, r * h);
            p.norm(enc, &scratch.normed, &gpu.input_norm, r, h, eps);
            p.matmul_matrix(
                enc,
                &scratch.normed,
                &gpu.q,
                0,
                &scratch.q,
                r,
                q_dim as u32,
                h,
            );
            p.matmul_matrix(
                enc,
                &scratch.normed,
                &gpu.k,
                0,
                &scratch.k,
                r,
                kv_dim as u32,
                h,
            );
            p.matmul_matrix(
                enc,
                &scratch.normed,
                &gpu.v,
                0,
                &scratch.v,
                r,
                kv_dim as u32,
                h,
            );
            p.norm(
                enc,
                &scratch.q,
                &gpu.q_norm,
                r * heads as u32,
                head_dim as u32,
                eps,
            );
            p.norm(
                enc,
                &scratch.k,
                &gpu.k_norm,
                r * kv_heads as u32,
                head_dim as u32,
                eps,
            );
            p.norm_noweight(enc, &scratch.v, r * kv_heads as u32, head_dim as u32, eps);
            if capture {
                p.copy(enc, &scratch.q, &scratch.q_normed, r * q_dim as u32);
                p.copy(enc, &scratch.k, &scratch.k_normed, r * kv_dim as u32);
            }
            p.vision_rope(
                enc,
                &scratch.q,
                &scratch.positions,
                inv_freq,
                r,
                heads as u32,
                head_dim as u32,
            );
            p.vision_rope(
                enc,
                &scratch.k,
                &scratch.positions,
                inv_freq,
                r,
                kv_heads as u32,
                head_dim as u32,
            );
            p.attention_matrix(
                enc,
                head_dim,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                &scratch.context,
                r,
                heads as u32,
                kv_heads as u32,
            )?;
            p.matmul_matrix(
                enc,
                &scratch.context,
                &gpu.o,
                0,
                &scratch.o_proj,
                r,
                h,
                q_dim as u32,
            );
            p.norm(enc, &scratch.o_proj, &gpu.post_attention_norm, r, h, eps);
            p.add(enc, &scratch.o_proj, &scratch.hidden, r * h);
            Ok(())
        })?;
        timings.attention_gpu_us += micros(span);

        if capture {
            let read_start = std::time::Instant::now();
            stages.layer_0_q_normed = as_f32(&scratch.q_normed, rows * q_dim)?.to_vec();
            stages.layer_0_k_normed = as_f32(&scratch.k_normed, rows * kv_dim)?.to_vec();
            stages.layer_0_q_rotated = as_f32(&scratch.q, rows * q_dim)?.to_vec();
            stages.layer_0_k_rotated = as_f32(&scratch.k, rows * kv_dim)?.to_vec();
            timings.cpu_post_attention_us += micros(read_start.elapsed().as_secs_f64());
        }

        // GPU: pre-FF norm, gate/up projections, GELU product, down projection, post-FF norm,
        // residual add into the stream.
        let span = ctx.run_timed(|enc| {
            let p = pipelines;
            p.copy(
                enc,
                &scratch.hidden,
                &scratch.normed,
                (rows * hidden) as u32,
            );
            p.norm(
                enc,
                &scratch.normed,
                &gpu.pre_ff_norm,
                rows as u32,
                hidden as u32,
                eps,
            );
            p.matmul_matrix(
                enc,
                &scratch.normed,
                &gpu.gate,
                0,
                &scratch.gate,
                rows as u32,
                intermediate as u32,
                hidden as u32,
            );
            p.matmul_matrix(
                enc,
                &scratch.normed,
                &gpu.up,
                0,
                &scratch.up,
                rows as u32,
                intermediate as u32,
                hidden as u32,
            );
            p.gelu_mul(
                enc,
                &scratch.gate,
                &scratch.up,
                (rows * intermediate) as u32,
            );
            p.matmul_matrix(
                enc,
                &scratch.gate,
                &gpu.down,
                0,
                &scratch.branch,
                rows as u32,
                hidden as u32,
                intermediate as u32,
            );
            p.norm(
                enc,
                &scratch.branch,
                &gpu.post_ff_norm,
                rows as u32,
                hidden as u32,
                eps,
            );
            p.add(
                enc,
                &scratch.branch,
                &scratch.hidden,
                (rows * hidden) as u32,
            );
            Ok(())
        })?;
        timings.mlp_gpu_us += micros(span);
        Ok(())
    }

    /// Grows the scratch buffers when `rows` exceeds their capacity.
    fn ensure_scratch(&mut self, ctx: &Context, rows: usize) -> Result<(), InferenceError> {
        if self.scratch.as_ref().is_some_and(|s| s.rows >= rows) {
            return Ok(());
        }
        self.scratch = None;
        self.scratch = Some(self.allocate_scratch(&ctx.device, rows)?);
        Ok(())
    }

    fn allocate_scratch(
        &self,
        device: &metal::Device,
        rows: usize,
    ) -> Result<VisionScratch, InferenceError> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let buf = |name: &str, width: usize| -> Result<Buffer, InferenceError> {
            let count = elements(&format!("eg2v.{name}"), rows, width)?;
            allocate(device, count, &format!("eg2v.{name}"))
        };
        Ok(VisionScratch {
            rows,
            hidden: buf("hidden", hidden)?,
            normed: buf("normed", hidden)?,
            positions: buf("positions", 2)?,
            q: buf("q", q_dim)?,
            k: buf("k", kv_dim)?,
            v: buf("v", kv_dim)?,
            q_normed: buf("q_normed", q_dim)?,
            k_normed: buf("k_normed", kv_dim)?,
            context: buf("context", q_dim)?,
            o_proj: buf("o_proj", hidden)?,
            branch: buf("branch", hidden)?,
            gate: buf("gate", cfg.intermediate_size)?,
            up: buf("up", cfg.intermediate_size)?,
        })
    }
}
