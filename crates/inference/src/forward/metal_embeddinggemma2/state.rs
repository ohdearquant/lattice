//! Persistent f32 EmbeddingGemma 2 forward on Metal.
//!
//! The command stream mirrors `EmbeddingGemma2Model::run_layer` step for step, so the CPU path is
//! the oracle for every kernel here.

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::error::InferenceError;
use crate::model::embeddinggemma2::{EmbeddingGemma2Model, LayerWeights, Weights};
use crate::model::embeddinggemma2_config::{EmbeddingGemma2Config, EmbeddingGemma2LayerKind};
use crate::model::gemma4_ops::{gemma4_rope_cos_sin, gemma4_rope_inv_freq};

#[cfg(test)]
mod tests;

const COMMON_SHADERS: &str = concat!(
    include_str!("../shaders/rms_reduce.metal"),
    include_str!("../shaders/flash_attention.metal"),
    include_str!("../shaders/ernie45_rope.metal"),
    include_str!("../shaders/embeddinggemma2.metal"),
);
const ATTENTION_SHADER: &str = include_str!("../shaders/embeddinggemma2_attention.metal");
const ELEMENT_THREADS: u64 = 256;
const ATTENTION_ROWS: u64 = 8;
const SIMD_WIDTH: u64 = 32;
/// Attention scaling of this model (queries are already normalized).
const ATTENTION_SCALE: f32 = 1.0;
/// Largest supported attention head width; the key and value tiles of the attention kernel must
/// fit threadgroup memory with at least one key.
const MAX_HEAD_DIM: usize = 2048;

fn invalid(message: impl Into<String>) -> InferenceError {
    InferenceError::InvalidInput(format!("embeddinggemma2 Metal: {}", message.into()))
}

fn runtime(message: impl Into<String>) -> InferenceError {
    InferenceError::Inference(format!("embeddinggemma2 Metal: {}", message.into()))
}

fn elements(name: &str, rows: usize, width: usize) -> Result<usize, InferenceError> {
    let count = rows
        .checked_mul(width)
        .ok_or_else(|| invalid(format!("{name} element count overflow")))?;
    u32::try_from(count).map_err(|_| invalid(format!("{name} exceeds u32 indexing")))?;
    let bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid(format!("{name} byte count overflow")))?;
    if bytes > isize::MAX as usize {
        return Err(invalid(format!("{name} exceeds host slice capacity")));
    }
    Ok(count)
}

fn distinct_head_dims(cfg: &EmbeddingGemma2Config) -> Vec<usize> {
    let mut dims: Vec<usize> = cfg.layer_shapes.iter().map(|s| s.head_dim).collect();
    dims.sort_unstable();
    dims.dedup();
    dims
}

/// Check all shader geometry and host capacities before acquiring a Metal device.
fn validate_config(cfg: &EmbeddingGemma2Config) -> Result<(), InferenceError> {
    for (l, shape) in cfg.layer_shapes.iter().enumerate() {
        if shape.head_dim == 0 || shape.head_dim % 4 != 0 || shape.head_dim > MAX_HEAD_DIM {
            return Err(invalid(format!(
                "layer {l} head_dim {} must be a multiple of 4 in 4..={MAX_HEAD_DIM}",
                shape.head_dim
            )));
        }
        if shape.num_key_value_heads == 0
            || !cfg
                .num_attention_heads
                .is_multiple_of(shape.num_key_value_heads)
        {
            return Err(invalid(format!(
                "layer {l} num_key_value_heads {} must divide {}",
                shape.num_key_value_heads, cfg.num_attention_heads
            )));
        }
    }
    if cfg.layer_shapes.len() != cfg.num_hidden_layers
        || cfg.layer_types.len() != cfg.num_hidden_layers
    {
        return Err(invalid("per-layer tables do not match num_hidden_layers"));
    }
    if !cfg.rms_norm_eps.is_finite() || cfg.rms_norm_eps <= 0.0 {
        return Err(invalid("rms_norm_eps must be finite and positive"));
    }
    for (name, theta) in [
        ("rope_theta_sliding", cfg.rope_theta_sliding),
        ("rope_theta_full", cfg.rope_theta_full),
    ] {
        if !theta.is_finite() || theta <= 1.0 {
            return Err(invalid(format!("{name} must be finite and above 1")));
        }
    }
    u32::try_from(cfg.sliding_window)
        .map_err(|_| invalid("sliding_window exceeds u32 indexing"))?;
    let h = cfg.hidden_size;
    let pl = cfg.hidden_size_per_layer_input;
    for (name, rows, width) in [
        ("per_layer_model_projection", cfg.num_hidden_layers * pl, h),
        ("embedding_projection", cfg.embedding_dim, h),
        ("gate_proj", cfg.intermediate_size, h),
        ("up_proj", cfg.intermediate_size, h),
        ("down_proj", h, cfg.intermediate_size),
        ("per_layer_input_gate", pl, h),
        ("per_layer_projection", h, pl),
    ] {
        elements(name, rows, width)?;
    }
    for (l, shape) in cfg.layer_shapes.iter().enumerate() {
        let q = cfg.num_attention_heads * shape.head_dim;
        let kv = shape.num_key_value_heads * shape.head_dim;
        for (name, rows, width) in [("q_proj", q, h), ("k_proj", kv, h), ("o_proj", h, q)] {
            elements(&format!("layer {l} {name}"), rows, width)?;
        }
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

fn validate_layer(
    cfg: &EmbeddingGemma2Config,
    l: usize,
    w: &LayerWeights,
) -> Result<(), InferenceError> {
    let shape = cfg.layer_shapes[l];
    let h = cfg.hidden_size;
    let pl = cfg.hidden_size_per_layer_input;
    let ff = cfg.intermediate_size;
    let q = cfg.num_attention_heads * shape.head_dim;
    let kv = shape.num_key_value_heads * shape.head_dim;
    for (name, values, expected) in [
        ("input_layernorm", &w.input_layernorm, h),
        ("post_attention_layernorm", &w.post_attention_layernorm, h),
        ("pre_feedforward_layernorm", &w.pre_feedforward_layernorm, h),
        (
            "post_feedforward_layernorm",
            &w.post_feedforward_layernorm,
            h,
        ),
        ("q_proj", &w.q_proj, q * h),
        ("k_proj", &w.k_proj, kv * h),
        ("v_proj", &w.v_proj, kv * h),
        ("o_proj", &w.o_proj, h * q),
        ("q_norm", &w.q_norm, shape.head_dim),
        ("k_norm", &w.k_norm, shape.head_dim),
        ("gate_proj", &w.gate_proj, ff * h),
        ("up_proj", &w.up_proj, ff * h),
        ("down_proj", &w.down_proj, h * ff),
        ("per_layer_input_gate", &w.per_layer_input_gate, pl * h),
        ("per_layer_projection", &w.per_layer_projection, h * pl),
        ("post_per_layer_input_norm", &w.post_per_layer_input_norm, h),
    ] {
        validate_values(&format!("layer {l} {name}"), values, expected)?;
    }
    if !w.layer_scalar.is_finite() {
        return Err(invalid(format!("layer {l} layer_scalar is not finite")));
    }
    Ok(())
}

/// Validates every tensor the state uploads. `embed_tokens` is not uploaded: the token gather
/// runs on the host, so the 262144-row table never occupies device memory.
fn validate_weights(cfg: &EmbeddingGemma2Config, w: &Weights) -> Result<(), InferenceError> {
    if w.layers.len() != cfg.num_hidden_layers {
        return Err(invalid(format!(
            "weights contain {} layers, expected {}",
            w.layers.len(),
            cfg.num_hidden_layers
        )));
    }
    let h = cfg.hidden_size;
    let pl = cfg.hidden_size_per_layer_input;
    validate_values(
        "per_layer_model_projection",
        &w.per_layer_model_projection,
        cfg.num_hidden_layers * pl * h,
    )?;
    validate_values(
        "per_layer_projection_norm",
        &w.per_layer_projection_norm,
        pl,
    )?;
    validate_values("norm", &w.norm, h)?;
    validate_values(
        "embedding_projection",
        &w.embedding_projection,
        cfg.embedding_dim * h,
    )?;
    for (l, layer) in w.layers.iter().enumerate() {
        validate_layer(cfg, l, layer)?;
    }
    Ok(())
}

/// A sampled hash of the configuration and weights. It tells a state apart from a state built
/// from another model; it is not a content hash of every tensor.
fn fingerprint(cfg: &EmbeddingGemma2Config, w: &Weights) -> u64 {
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
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.hidden_size_per_layer_input,
        cfg.embedding_dim,
        cfg.sliding_window,
    ] {
        mix(&mut hash, value as u64);
    }
    sample(&mut hash, &w.embed_tokens);
    sample(&mut hash, &w.per_layer_model_projection);
    sample(&mut hash, &w.norm);
    sample(&mut hash, &w.embedding_projection);
    for layer in &w.layers {
        sample(&mut hash, &layer.q_proj);
        sample(&mut hash, &layer.down_proj);
        sample(&mut hash, &layer.per_layer_projection);
        mix(&mut hash, u64::from(layer.layer_scalar.to_bits()));
    }
    hash
}

// ---------------------------------------------------------------------------
// Device context and buffers
// ---------------------------------------------------------------------------

/// The Metal device, its queue and the Objective-C selectors used to encode one command buffer.
struct Context {
    device: Device,
    queue: CommandQueue,
    command_buffer_selector: objc::runtime::Sel,
    compute_encoder_selector: objc::runtime::Sel,
}

impl Context {
    fn new() -> Result<Self, InferenceError> {
        let device =
            Device::system_default().ok_or_else(|| runtime("no Metal device available"))?;
        let queue_selector = objc::runtime::Sel::register("newCommandQueue");
        // SAFETY: newCommandQueue takes no arguments and returns a nullable owned
        // MTLCommandQueue. Check nil before transferring its +1 ownership to metal-rs.
        let queue = unsafe {
            let raw: *mut metal::MTLCommandQueue =
                objc::Message::send_message(&*device, queue_selector, ())
                    .map_err(|error| runtime(format!("queue creation failed: {error}")))?;
            if raw.is_null() {
                return Err(runtime("queue allocation failed"));
            }
            <CommandQueue as metal::foreign_types::ForeignType>::from_ptr(raw)
        };
        Ok(Self {
            device,
            queue,
            command_buffer_selector: objc::runtime::Sel::register("commandBuffer"),
            compute_encoder_selector: objc::runtime::Sel::register("computeCommandEncoder"),
        })
    }

    /// Encodes with `encode` into one command buffer and one compute encoder, submits it and
    /// waits for completion. The encoder is always ended; a failed encode submits nothing.
    fn run(
        &self,
        encode: impl FnOnce(&ComputeCommandEncoderRef) -> Result<(), InferenceError>,
    ) -> Result<(), InferenceError> {
        // Metal command buffers and encoders are autoreleased even on Rust worker threads.
        objc::rc::autoreleasepool(|| {
            // SAFETY: commandBuffer takes no arguments and returns a nullable,
            // autoreleased MTLCommandBuffer. The checked reference stays inside this
            // pool, which remains alive through command completion.
            let command = unsafe {
                let raw: *mut metal::MTLCommandBuffer =
                    objc::Message::send_message(&*self.queue, self.command_buffer_selector, ())
                        .map_err(|error| runtime(format!("command creation failed: {error}")))?;
                if raw.is_null() {
                    return Err(runtime("command buffer allocation failed"));
                }
                <metal::CommandBufferRef as metal::foreign_types::ForeignTypeRef>::from_ptr(raw)
            };
            // SAFETY: computeCommandEncoder takes no arguments and returns a
            // nullable autoreleased encoder for this command. Nil is rejected before
            // borrowing, and the reference cannot escape the enclosing pool.
            let encoder = unsafe {
                let raw: *mut metal::MTLComputeCommandEncoder =
                    objc::Message::send_message(command, self.compute_encoder_selector, ())
                        .map_err(|error| runtime(format!("encoder creation failed: {error}")))?;
                if raw.is_null() {
                    return Err(runtime("compute encoder allocation failed"));
                }
                <ComputeCommandEncoderRef as metal::foreign_types::ForeignTypeRef>::from_ptr(raw)
            };
            let encoded = encode(encoder);
            encoder.end_encoding();
            encoded?;
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(runtime(format!(
                    "command buffer ended with {:?}",
                    command.status()
                )));
            }
            Ok(())
        })
    }
}

fn allocate(device: &Device, count: usize, label: &str) -> Result<Buffer, InferenceError> {
    let count = elements(label, count, 1)?;
    let bytes = (count * size_of::<f32>()) as u64;
    if bytes > device.max_buffer_length() {
        return Err(runtime(format!(
            "{label} exceeds the device buffer capacity"
        )));
    }
    let selector = objc::runtime::Sel::register("newBufferWithLength:options:");
    // SAFETY: this is the Metal allocation selector with its documented NSUInteger
    // length/options ABI and nullable object return. A raw pointer preserves nil
    // until it can be checked, unlike metal-rs's non-null Buffer return type.
    let raw: *mut metal::MTLBuffer = unsafe {
        objc::Message::send_message(
            &**device,
            selector,
            (bytes.max(4), MTLResourceOptions::StorageModeShared),
        )
    }
    .map_err(|error| runtime(format!("{label} allocation message failed: {error}")))?;
    if raw.is_null() {
        return Err(runtime(format!("{label} allocation failed")));
    }
    // SAFETY: Metal returned a non-null owned (+1) MTLBuffer. Transferring that
    // ownership exactly once to Buffer pairs it with metal-rs's release on drop.
    let buffer = unsafe { <Buffer as metal::foreign_types::ForeignType>::from_ptr(raw) };
    buffer.set_label(label);
    if buffer.contents().is_null() || buffer.length() < bytes {
        return Err(runtime(format!("{label} allocation is not CPU accessible")));
    }
    Ok(buffer)
}

fn upload(device: &Device, values: &[f32], label: &str) -> Result<Buffer, InferenceError> {
    let buffer = allocate(device, values.len(), label)?;
    write_f32(&buffer, values);
    Ok(buffer)
}

/// Copies `values` to the start of a shared buffer. Callers pass a buffer allocated for at least
/// `values.len()` elements, with no GPU work in flight on it.
fn write_f32(buffer: &Buffer, values: &[f32]) {
    debug_assert!(buffer.length() as usize >= size_of_val(values));
    // SAFETY: the shared allocation holds at least values.len() aligned f32 elements and
    // is not submitted to the GPU while this copy runs; it cannot alias the input slice.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().cast::<f32>(),
            values.len(),
        );
    }
}

/// Copies the first `count` elements of a shared buffer after its GPU work has completed.
fn read_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
    debug_assert!(buffer.length() as usize >= count * size_of::<f32>());
    // SAFETY: the buffer holds at least count aligned, initialized f32 elements, and the
    // command that wrote them completed before this CPU read.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), count) }.to_vec()
}

fn bind(enc: &ComputeCommandEncoderRef, index: u64, buffer: &Buffer) {
    enc.set_buffer(index, Some(buffer), 0);
}

fn scalar<T: Copy>(enc: &ComputeCommandEncoderRef, index: u64, value: &T) {
    enc.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}

fn element_dispatch(enc: &ComputeCommandEncoderRef, count: u32) {
    enc.dispatch_thread_groups(
        MTLSize::new(u64::from(count).div_ceil(ELEMENT_THREADS), 1, 1),
        MTLSize::new(ELEMENT_THREADS, 1, 1),
    );
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionParams {
    seq_len: u32,
    heads: u32,
    kv_heads: u32,
    window: u32,
    use_window: u32,
    scale: f32,
    pad: [u32; 2],
}

struct Pipelines {
    matmul: ComputePipelineState,
    norm: ComputePipelineState,
    norm_noweight: ComputePipelineState,
    rope: ComputePipelineState,
    copy: ComputePipelineState,
    add: ComputePipelineState,
    gelu_mul: ComputePipelineState,
    scale: ComputePipelineState,
    attention: Vec<(usize, ComputePipelineState)>,
}

impl Pipelines {
    fn new(device: &Device, head_dims: &[usize]) -> Result<Self, InferenceError> {
        // The shared attention shader file also defines kernels for other models; its
        // geometry placeholders are given a valid dummy geometry, and none of those kernels
        // is dispatched here. Only matmul_bt, rms_norm, copy_buf and add_buf are used from it.
        let mut source = COMMON_SHADERS
            .replace("__FA_HEAD_DIM__", "128")
            .replace("__FA_GQA_GROUPS__", "1")
            .replace("__FUSED_C_HEAD_DIM__", "128")
            .replace("__FUSED_C_HALF_DIM__", "64")
            .replace("__FUSED_C_THREADS__", "64");
        for head_dim in head_dims {
            source.push_str(&ATTENTION_SHADER.replace("__HD__", &head_dim.to_string()));
        }
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|error| runtime(format!("shader compilation failed: {error}")))?;
        let make = |name: &str, threads: u64| -> Result<ComputePipelineState, InferenceError> {
            let function = library
                .get_function(name, None)
                .map_err(|error| runtime(format!("missing {name} kernel: {error}")))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| runtime(format!("{name} pipeline failed: {error}")))?;
            if pipeline.max_total_threads_per_threadgroup() < threads {
                return Err(runtime(format!(
                    "{name} needs {threads} threads per group, pipeline permits {}",
                    pipeline.max_total_threads_per_threadgroup()
                )));
            }
            if pipeline.static_threadgroup_memory_length() > device.max_threadgroup_memory_length()
            {
                return Err(runtime(format!(
                    "{name} exceeds threadgroup memory capacity"
                )));
            }
            Ok(pipeline)
        };
        let attention_threads = ATTENTION_ROWS * SIMD_WIDTH;
        let limits = device.max_threads_per_threadgroup();
        if limits.width < attention_threads.max(ELEMENT_THREADS) || limits.height < 16 {
            return Err(runtime(
                "device threadgroup dimensions cannot support the kernels",
            ));
        }
        let mut attention = Vec::with_capacity(head_dims.len());
        for &head_dim in head_dims {
            let pipeline = make(&format!("eg2_attention_hd{head_dim}"), attention_threads)?;
            if pipeline.thread_execution_width() != SIMD_WIDTH {
                return Err(runtime("attention requires a 32-lane SIMD execution width"));
            }
            attention.push((head_dim, pipeline));
        }
        Ok(Self {
            matmul: make("matmul_bt", ELEMENT_THREADS)?,
            norm: make("rms_norm", ELEMENT_THREADS)?,
            norm_noweight: make("eg2_rms_norm_noweight", ELEMENT_THREADS)?,
            rope: make("ernie45_rope", ELEMENT_THREADS)?,
            copy: make("copy_buf", ELEMENT_THREADS)?,
            add: make("add_buf", ELEMENT_THREADS)?,
            gelu_mul: make("eg2_gelu_mul", ELEMENT_THREADS)?,
            scale: make("eg2_scale", ELEMENT_THREADS)?,
            attention,
        })
    }

    /// `c[m, n] = a[m, k] * b[n, k]^T`, with `b` starting `b_offset` bytes into its buffer.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        b_offset: u64,
        c: &Buffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        enc.set_compute_pipeline_state(&self.matmul);
        bind(enc, 0, a);
        enc.set_buffer(1, Some(b), b_offset);
        bind(enc, 2, c);
        scalar(enc, 3, &m);
        scalar(enc, 4, &n);
        scalar(enc, 5, &k);
        enc.dispatch_thread_groups(
            MTLSize::new(u64::from(n).div_ceil(16), u64::from(m).div_ceil(16), 1),
            MTLSize::new(16, 16, 1),
        );
    }

    /// Row-wise `x * rsqrt(mean(x^2) + eps) * weight` in place.
    fn norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        weight: &Buffer,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.norm);
        bind(enc, 0, x);
        bind(enc, 1, weight);
        scalar(enc, 2, &width);
        scalar(enc, 3, &rows);
        scalar(enc, 4, &eps);
        enc.dispatch_thread_groups(MTLSize::new(u64::from(rows), 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Row-wise `x * rsqrt(mean(x^2) + eps)` in place, with no learned weight.
    fn norm_noweight(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.norm_noweight);
        bind(enc, 0, x);
        scalar(enc, 1, &width);
        scalar(enc, 2, &rows);
        scalar(enc, 3, &eps);
        enc.dispatch_thread_groups(MTLSize::new(u64::from(rows), 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Stride-half RoPE on `[tokens, heads, head_dim]` with compact `[tokens, head_dim / 2]`
    /// cosine and sine tables.
    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        cos: &Buffer,
        sin: &Buffer,
        tokens: u32,
        heads: u32,
        head_dim: u32,
    ) {
        enc.set_compute_pipeline_state(&self.rope);
        bind(enc, 0, x);
        bind(enc, 1, cos);
        bind(enc, 2, sin);
        scalar(enc, 3, &tokens);
        scalar(enc, 4, &heads);
        scalar(enc, 5, &head_dim);
        element_dispatch(enc, tokens * heads * (head_dim / 2));
    }

    fn copy(&self, enc: &ComputeCommandEncoderRef, src: &Buffer, dst: &Buffer, count: u32) {
        enc.set_compute_pipeline_state(&self.copy);
        bind(enc, 0, src);
        bind(enc, 1, dst);
        scalar(enc, 2, &count);
        element_dispatch(enc, count);
    }

    /// `dst += src`.
    fn add(&self, enc: &ComputeCommandEncoderRef, src: &Buffer, dst: &Buffer, count: u32) {
        enc.set_compute_pipeline_state(&self.add);
        bind(enc, 0, src);
        bind(enc, 1, dst);
        scalar(enc, 2, &count);
        element_dispatch(enc, count);
    }

    /// `gate = gelu_tanh(gate) * other`.
    fn gelu_mul(&self, enc: &ComputeCommandEncoderRef, gate: &Buffer, other: &Buffer, count: u32) {
        enc.set_compute_pipeline_state(&self.gelu_mul);
        bind(enc, 0, gate);
        bind(enc, 1, other);
        scalar(enc, 2, &count);
        element_dispatch(enc, count);
    }

    /// `x *= factor`.
    fn scale(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, factor: f32, count: u32) {
        enc.set_compute_pipeline_state(&self.scale);
        bind(enc, 0, x);
        scalar(enc, 1, &factor);
        scalar(enc, 2, &count);
        element_dispatch(enc, count);
    }

    /// Bidirectional attention over `[seq, heads, head_dim]` queries and `[seq, kv_heads,
    /// head_dim]` keys and values. `window = Some(w)` restricts a query at `i` to keys with
    /// `|i - j| <= w`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        head_dim: usize,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        out: &Buffer,
        seq: u32,
        heads: u32,
        kv_heads: u32,
        window: Option<u32>,
    ) -> Result<(), InferenceError> {
        let pipeline = self
            .attention
            .iter()
            .find_map(|(dim, pipeline)| (*dim == head_dim).then_some(pipeline))
            .ok_or_else(|| runtime(format!("no attention pipeline for head_dim {head_dim}")))?;
        let params = AttentionParams {
            seq_len: seq,
            heads,
            kv_heads,
            window: window.unwrap_or(0),
            use_window: u32::from(window.is_some()),
            scale: ATTENTION_SCALE,
            pad: [0; 2],
        };
        enc.set_compute_pipeline_state(pipeline);
        bind(enc, 0, q);
        bind(enc, 1, k);
        bind(enc, 2, v);
        bind(enc, 3, out);
        scalar(enc, 4, &params);
        enc.dispatch_thread_groups(
            MTLSize::new(u64::from(heads), u64::from(seq).div_ceil(ATTENTION_ROWS), 1),
            MTLSize::new(ATTENTION_ROWS * SIMD_WIDTH, 1, 1),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Resident weights and per-call activations
// ---------------------------------------------------------------------------

struct GpuLayer {
    input_norm: Buffer,
    post_attention_norm: Buffer,
    pre_feedforward_norm: Buffer,
    post_feedforward_norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    o: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    ple_gate: Buffer,
    ple_projection: Buffer,
    post_ple_norm: Buffer,
    layer_scalar: f32,
}

impl GpuLayer {
    fn new(device: &Device, l: usize, w: &LayerWeights) -> Result<Self, InferenceError> {
        let up = |values: &[f32], name: &str| upload(device, values, &format!("eg2.l{l}.{name}"));
        Ok(Self {
            input_norm: up(&w.input_layernorm, "input_norm")?,
            post_attention_norm: up(&w.post_attention_layernorm, "post_attention_norm")?,
            pre_feedforward_norm: up(&w.pre_feedforward_layernorm, "pre_feedforward_norm")?,
            post_feedforward_norm: up(&w.post_feedforward_layernorm, "post_feedforward_norm")?,
            q: up(&w.q_proj, "q_proj")?,
            k: up(&w.k_proj, "k_proj")?,
            v: up(&w.v_proj, "v_proj")?,
            o: up(&w.o_proj, "o_proj")?,
            q_norm: up(&w.q_norm, "q_norm")?,
            k_norm: up(&w.k_norm, "k_norm")?,
            gate: up(&w.gate_proj, "gate_proj")?,
            up: up(&w.up_proj, "up_proj")?,
            down: up(&w.down_proj, "down_proj")?,
            ple_gate: up(&w.per_layer_input_gate, "ple_gate")?,
            ple_projection: up(&w.per_layer_projection, "ple_projection")?,
            post_ple_norm: up(&w.post_per_layer_input_norm, "post_ple_norm")?,
            layer_scalar: w.layer_scalar,
        })
    }
}

struct GpuWeights {
    layers: Vec<GpuLayer>,
    per_layer_model_projection: Buffer,
    per_layer_projection_norm: Buffer,
    norm: Buffer,
    embedding_projection: Buffer,
}

/// Activation and RoPE buffers sized for `capacity` tokens. Every buffer is fully overwritten on
/// each forward pass, so a larger capacity than the current call needs is harmless.
struct Scratch {
    capacity: usize,
    hidden: Buffer,
    embeddings: Buffer,
    normed: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    context: Buffer,
    branch: Buffer,
    gate: Buffer,
    up: Buffer,
    ple_input: Buffer,
    ple_gate: Buffer,
    states: Buffer,
    /// Compact `[capacity, head_dim / 2]` cosine and sine tables per layer kind
    /// (sliding, full), built for the kinds that occur.
    rope: [Option<(Buffer, Buffer)>; 2],
}

fn rope_slot(kind: EmbeddingGemma2LayerKind) -> usize {
    usize::from(kind == EmbeddingGemma2LayerKind::Full)
}

impl Scratch {
    fn new(
        device: &Device,
        cfg: &EmbeddingGemma2Config,
        capacity: usize,
    ) -> Result<Self, InferenceError> {
        let h = cfg.hidden_size;
        let max_q = cfg
            .layer_shapes
            .iter()
            .map(|s| cfg.num_attention_heads * s.head_dim)
            .max()
            .unwrap_or(0);
        let max_kv = cfg
            .layer_shapes
            .iter()
            .map(|s| s.num_key_value_heads * s.head_dim)
            .max()
            .unwrap_or(0);
        let buf = |name: &str, width: usize| -> Result<Buffer, InferenceError> {
            let count = elements(&format!("eg2.{name}"), capacity, width)?;
            allocate(device, count, &format!("eg2.{name}"))
        };
        let mut rope: [Option<(Buffer, Buffer)>; 2] = [None, None];
        for (kind, theta) in [
            (EmbeddingGemma2LayerKind::Sliding, cfg.rope_theta_sliding),
            (EmbeddingGemma2LayerKind::Full, cfg.rope_theta_full),
        ] {
            let Some(l) = cfg.layer_types.iter().position(|&k| k == kind) else {
                continue;
            };
            let head_dim = cfg.layer_shapes[l].head_dim;
            let half = head_dim / 2;
            let inv_freq = gemma4_rope_inv_freq(head_dim, theta, None);
            let positions: Vec<u32> = (0..capacity as u32).collect();
            let (cos, sin) = gemma4_rope_cos_sin(&inv_freq, &positions);
            let compact = |full: &[f32]| -> Vec<f32> {
                full.chunks_exact(head_dim)
                    .flat_map(|row| row[..half].iter().copied())
                    .collect()
            };
            let (cos, sin) = (compact(&cos), compact(&sin));
            if cos.iter().chain(&sin).any(|v| !v.is_finite()) {
                return Err(invalid("rope theta produces non-finite RoPE tables"));
            }
            elements("eg2.rope", capacity, half)?;
            rope[rope_slot(kind)] = Some((
                upload(device, &cos, "eg2.rope_cos")?,
                upload(device, &sin, "eg2.rope_sin")?,
            ));
        }
        Ok(Self {
            capacity,
            hidden: buf("hidden", h)?,
            embeddings: buf("embeddings", h)?,
            normed: buf("normed", h)?,
            q: buf("q", max_q)?,
            k: buf("k", max_kv)?,
            v: buf("v", max_kv)?,
            context: buf("context", max_q)?,
            branch: buf("branch", h)?,
            gate: buf("gate", cfg.intermediate_size)?,
            up: buf("up", cfg.intermediate_size)?,
            ple_input: buf("ple_input", cfg.hidden_size_per_layer_input)?,
            ple_gate: buf("ple_gate", cfg.hidden_size_per_layer_input)?,
            states: buf("states", cfg.embedding_dim)?,
            rope,
        })
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// EmbeddingGemma 2 f32 forward pass on Metal with resident weights.
///
/// Create it from a loaded model with [`MetalEmbeddingGemma2State::new`] and use it only with
/// that model, through `EmbeddingGemma2Model::token_states_metal` or
/// `EmbeddingGemma2Model::encode_ids_at_widths_metal`. The scaled token-embedding gather and the
/// mean pooling run on the host; the 24 encoder layers, the final norm and the per-token
/// projection run on the GPU in one command buffer. The caller holds the machine GPU lock for
/// the whole GPU operation.
pub struct MetalEmbeddingGemma2State {
    ctx: Context,
    pipelines: Pipelines,
    weights: GpuWeights,
    cfg: EmbeddingGemma2Config,
    fingerprint: u64,
    scratch: Option<Scratch>,
}

impl MetalEmbeddingGemma2State {
    /// Upload a loaded model's text tower to the GPU as f32.
    ///
    /// # Errors
    /// Returns an error for unsupported geometry (a head width that is not a multiple of 4),
    /// tensor shapes or non-finite weights, unavailable Metal, or shader and pipeline creation
    /// failures. Validation precedes device access.
    pub fn new(model: &EmbeddingGemma2Model) -> Result<Self, InferenceError> {
        let cfg = model.config();
        let host = model.weights();
        validate_config(cfg)?;
        validate_weights(cfg, host)?;
        let ctx = Context::new()?;
        let pipelines = Pipelines::new(&ctx.device, &distinct_head_dims(cfg))?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(cfg.num_hidden_layers)
            .map_err(|error| runtime(format!("layer buffer allocation failed: {error}")))?;
        for (l, layer) in host.layers.iter().enumerate() {
            layers.push(GpuLayer::new(&ctx.device, l, layer)?);
        }
        let weights = GpuWeights {
            layers,
            per_layer_model_projection: upload(
                &ctx.device,
                &host.per_layer_model_projection,
                "eg2.per_layer_model_projection",
            )?,
            per_layer_projection_norm: upload(
                &ctx.device,
                &host.per_layer_projection_norm,
                "eg2.per_layer_projection_norm",
            )?,
            norm: upload(&ctx.device, &host.norm, "eg2.norm")?,
            embedding_projection: upload(
                &ctx.device,
                &host.embedding_projection,
                "eg2.embedding_projection",
            )?,
        };
        Ok(Self {
            ctx,
            pipelines,
            weights,
            cfg: cfg.clone(),
            fingerprint: fingerprint(cfg, host),
            scratch: None,
        })
    }

    /// Rejects a model other than the one this state was built from.
    pub(crate) fn check_model(&self, model: &EmbeddingGemma2Model) -> Result<(), InferenceError> {
        if model.config() != &self.cfg
            || fingerprint(model.config(), model.weights()) != self.fingerprint
        {
            return Err(invalid(
                "this state was built from a different model; build one per model",
            ));
        }
        Ok(())
    }

    /// Per-token states `[seq_len, embedding_dim]` from scaled token embeddings
    /// `[seq_len, hidden_size]`.
    ///
    /// Validation precedes any GPU work, the caller's data is only read, and the output is
    /// returned only after the GPU completed and every value was checked to be finite. There is
    /// no CPU fallback.
    pub(crate) fn forward(
        &mut self,
        embeddings: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let hidden = self.cfg.hidden_size;
        if seq_len == 0 {
            return Err(invalid("cannot embed an empty token sequence"));
        }
        if embeddings.len() != elements("embeddings", seq_len, hidden)? {
            return Err(invalid(
                "embeddings must have the exact [seq_len, hidden] shape",
            ));
        }
        if embeddings.iter().any(|v| !v.is_finite()) {
            return Err(invalid("embeddings contain a non-finite value"));
        }
        let seq = u32::try_from(seq_len).map_err(|_| invalid("seq_len exceeds u32 indexing"))?;
        self.ensure_capacity(seq_len)?;
        let Some(scratch) = self.scratch.as_ref() else {
            return Err(runtime("scratch buffers missing after allocation"));
        };
        write_f32(&scratch.hidden, embeddings);
        write_f32(&scratch.embeddings, embeddings);
        self.ctx.run(|enc| self.encode_forward(enc, scratch, seq))?;
        let count = seq_len * self.cfg.embedding_dim;
        let states = read_f32(&scratch.states, count);
        if states.iter().any(|v| !v.is_finite()) {
            return Err(runtime("forward pass produced a non-finite value"));
        }
        Ok(states)
    }

    /// Grows the scratch buffers when `seq_len` exceeds their capacity. A longer sequence than
    /// any before reallocates; shorter ones reuse the existing buffers.
    fn ensure_capacity(&mut self, seq_len: usize) -> Result<(), InferenceError> {
        if self
            .scratch
            .as_ref()
            .is_some_and(|scratch| scratch.capacity >= seq_len)
        {
            return Ok(());
        }
        self.scratch = None;
        self.scratch = Some(Scratch::new(&self.ctx.device, &self.cfg, seq_len)?);
        Ok(())
    }

    fn encode_forward(
        &self,
        enc: &ComputeCommandEncoderRef,
        sc: &Scratch,
        s: u32,
    ) -> Result<(), InferenceError> {
        let cfg = &self.cfg;
        let p = &self.pipelines;
        let h = cfg.hidden_size as u32;
        for l in 0..cfg.num_hidden_layers {
            self.encode_layer(enc, sc, l, s)?;
        }
        p.norm(enc, &sc.hidden, &self.weights.norm, s, h, cfg.rms_norm_eps);
        p.matmul(
            enc,
            &sc.hidden,
            &self.weights.embedding_projection,
            0,
            &sc.states,
            s,
            cfg.embedding_dim as u32,
            h,
        );
        Ok(())
    }

    /// One encoder layer applied in place to `sc.hidden`, step for step as
    /// `EmbeddingGemma2Model::run_layer`.
    fn encode_layer(
        &self,
        enc: &ComputeCommandEncoderRef,
        sc: &Scratch,
        l: usize,
        s: u32,
    ) -> Result<(), InferenceError> {
        let cfg = &self.cfg;
        let p = &self.pipelines;
        let w = &self.weights.layers[l];
        let eps = cfg.rms_norm_eps;
        let shape = cfg.layer_shapes[l];
        let kind = cfg.layer_types[l];
        let h = cfg.hidden_size as u32;
        let ff = cfg.intermediate_size as u32;
        let pl = cfg.hidden_size_per_layer_input as u32;
        let heads = cfg.num_attention_heads as u32;
        let kv_heads = shape.num_key_value_heads as u32;
        let hd = shape.head_dim as u32;
        let q_dim = heads * hd;
        let kv_dim = kv_heads * hd;
        let Some((cos, sin)) = sc.rope[rope_slot(kind)].as_ref() else {
            return Err(runtime("rope table missing for a layer kind that occurs"));
        };

        // Attention block.
        p.copy(enc, &sc.hidden, &sc.normed, s * h);
        p.norm(enc, &sc.normed, &w.input_norm, s, h, eps);
        p.matmul(enc, &sc.normed, &w.q, 0, &sc.q, s, q_dim, h);
        p.matmul(enc, &sc.normed, &w.k, 0, &sc.k, s, kv_dim, h);
        p.matmul(enc, &sc.normed, &w.v, 0, &sc.v, s, kv_dim, h);
        p.norm(enc, &sc.q, &w.q_norm, s * heads, hd, eps);
        p.norm(enc, &sc.k, &w.k_norm, s * kv_heads, hd, eps);
        p.norm_noweight(enc, &sc.v, s * kv_heads, hd, eps);
        p.rope(enc, &sc.q, cos, sin, s, heads, hd);
        p.rope(enc, &sc.k, cos, sin, s, kv_heads, hd);
        let window = match kind {
            EmbeddingGemma2LayerKind::Sliding => Some(cfg.sliding_window as u32),
            EmbeddingGemma2LayerKind::Full => None,
        };
        p.attention(
            enc,
            shape.head_dim,
            &sc.q,
            &sc.k,
            &sc.v,
            &sc.context,
            s,
            heads,
            kv_heads,
            window,
        )?;
        p.matmul(enc, &sc.context, &w.o, 0, &sc.branch, s, h, q_dim);
        p.norm(enc, &sc.branch, &w.post_attention_norm, s, h, eps);
        p.add(enc, &sc.branch, &sc.hidden, s * h);

        // Feed-forward block.
        p.copy(enc, &sc.hidden, &sc.normed, s * h);
        p.norm(enc, &sc.normed, &w.pre_feedforward_norm, s, h, eps);
        p.matmul(enc, &sc.normed, &w.gate, 0, &sc.gate, s, ff, h);
        p.matmul(enc, &sc.normed, &w.up, 0, &sc.up, s, ff, h);
        p.gelu_mul(enc, &sc.gate, &sc.up, s * ff);
        p.matmul(enc, &sc.gate, &w.down, 0, &sc.branch, s, h, ff);
        p.norm(enc, &sc.branch, &w.post_feedforward_norm, s, h, eps);
        p.add(enc, &sc.branch, &sc.hidden, s * h);

        // Per-layer input block: this layer's slice of the projection-only per-layer signal is
        // computed from the scaled embeddings, as on the CPU path.
        let ple_scale = (f64::from(h)).powf(-0.5) as f32;
        let slice_bytes = u64::from(pl) * u64::from(h) * size_of::<f32>() as u64;
        p.matmul(
            enc,
            &sc.embeddings,
            &self.weights.per_layer_model_projection,
            l as u64 * slice_bytes,
            &sc.ple_input,
            s,
            pl,
            h,
        );
        p.scale(enc, &sc.ple_input, ple_scale, s * pl);
        p.norm(
            enc,
            &sc.ple_input,
            &self.weights.per_layer_projection_norm,
            s,
            pl,
            eps,
        );
        p.matmul(enc, &sc.hidden, &w.ple_gate, 0, &sc.ple_gate, s, pl, h);
        p.gelu_mul(enc, &sc.ple_gate, &sc.ple_input, s * pl);
        p.matmul(
            enc,
            &sc.ple_gate,
            &w.ple_projection,
            0,
            &sc.branch,
            s,
            h,
            pl,
        );
        p.norm(enc, &sc.branch, &w.post_ple_norm, s, h, eps);
        p.add(enc, &sc.branch, &sc.hidden, s * h);

        p.scale(enc, &sc.hidden, w.layer_scalar, s * h);
        Ok(())
    }
}
