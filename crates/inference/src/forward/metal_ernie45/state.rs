//! Persistent f32 ERNIE text prefill using the shared Metal primitives.

#[cfg(all(test, feature = "f16"))]
use std::cell::Cell;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::error::InferenceError;
use crate::model::ernie45::{Ernie45Config, Ernie45LayerWeights, Ernie45Weights, MAX_SEQ_LEN};
use crate::model::gemma4_ops::{gemma4_rope_cos_sin, gemma4_rope_inv_freq};

const SHADERS: &str = concat!(
    include_str!("../shaders/rms_reduce.metal"),
    include_str!("../shaders/flash_attention.metal"),
    include_str!("../shaders/ernie45_rope.metal"),
    include_str!("../shaders/ernie45_embed.metal")
);
const ELEMENT_THREADS: u64 = 256;
const TILE_Q: u32 = 4;
const SIMD_WIDTH: u32 = 32;

fn invalid(message: impl Into<String>) -> InferenceError {
    InferenceError::InvalidInput(format!("ernie45 Metal: {}", message.into()))
}

fn runtime(message: impl Into<String>) -> InferenceError {
    InferenceError::Inference(format!("ernie45 Metal: {}", message.into()))
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

/// Check all shader geometry and host capacities before acquiring a Metal device.
pub(super) fn validate_shape(
    cfg: &Ernie45Config,
    max_seq_len: usize,
) -> Result<(), InferenceError> {
    if cfg.head_dim != 128 {
        return Err(invalid(format!(
            "head_dim must be 128 for fused attention, got {}",
            cfg.head_dim
        )));
    }
    cfg.validate()
        .map_err(|error| invalid(format!("invalid config: {error}")))?;
    if max_seq_len == 0 || max_seq_len > MAX_SEQ_LEN {
        return Err(invalid(format!("max_seq_len must be in 1..={MAX_SEQ_LEN}")));
    }
    if !cfg.rms_norm_eps.is_finite() || cfg.rms_norm_eps <= 0.0 {
        return Err(invalid("rms_norm_eps must be finite and positive"));
    }
    if !cfg.rope_theta.is_finite() || cfg.rope_theta <= 0.0 {
        return Err(invalid("rope_theta must be finite and positive"));
    }
    let groups = cfg.num_attention_heads / cfg.num_key_value_heads;
    if groups > 8 {
        return Err(invalid(
            "GQA group count exceeds the 1024-thread launch limit",
        ));
    }
    let q = elements("query width", cfg.num_attention_heads, cfg.head_dim)?;
    let kv = elements("KV width", cfg.num_key_value_heads, cfg.head_dim)?;
    for (name, width) in [
        ("hidden", cfg.hidden_size),
        ("query", q),
        ("KV", kv),
        ("intermediate", cfg.intermediate_size),
        ("RoPE", cfg.head_dim),
        ("logits", cfg.vocab_size),
    ] {
        elements(name, max_seq_len, width)?;
    }
    for (name, rows, width) in [
        ("q_proj", q, cfg.hidden_size),
        ("k_proj", kv, cfg.hidden_size),
        ("v_proj", kv, cfg.hidden_size),
        ("o_proj", cfg.hidden_size, q),
        ("gate_proj", cfg.intermediate_size, cfg.hidden_size),
        ("up_proj", cfg.intermediate_size, cfg.hidden_size),
        ("down_proj", cfg.hidden_size, cfg.intermediate_size),
        ("embedding and lm_head", cfg.vocab_size, cfg.hidden_size),
    ] {
        elements(name, rows, width)?;
    }
    Ok(())
}

fn validate_layer_weights(
    cfg: &Ernie45Config,
    weights: &Ernie45LayerWeights,
) -> Result<(), InferenceError> {
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    for (name, values, expected) in [
        ("q_proj", &weights.q_proj, q * h),
        ("k_proj", &weights.k_proj, kv * h),
        ("v_proj", &weights.v_proj, kv * h),
        ("o_proj", &weights.o_proj, h * q),
        ("gate_proj", &weights.gate_proj, i * h),
        ("up_proj", &weights.up_proj, i * h),
        ("down_proj", &weights.down_proj, h * i),
        ("input_layernorm", &weights.input_layernorm, h),
        (
            "post_attention_layernorm",
            &weights.post_attention_layernorm,
            h,
        ),
    ] {
        if values.len() != expected {
            return Err(invalid(format!(
                "{name} has {} elements, expected {expected}",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!("{name} contains a non-finite weight")));
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

fn validate_weights(cfg: &Ernie45Config, weights: &Ernie45Weights) -> Result<(), InferenceError> {
    if weights.layers.len() != cfg.num_hidden_layers {
        return Err(invalid(format!(
            "weights contain {} layers, expected {}",
            weights.layers.len(),
            cfg.num_hidden_layers
        )));
    }
    let embedding_count = cfg.vocab_size * cfg.hidden_size;
    validate_values("embed_tokens", &weights.embed_tokens, embedding_count)?;
    validate_values("final_norm", &weights.final_norm, cfg.hidden_size)?;
    validate_values("lm_head", &weights.lm_head, embedding_count)?;
    for (index, layer) in weights.layers.iter().enumerate() {
        validate_layer_weights(cfg, layer)
            .map_err(|error| invalid(format!("layer {index}: {error}")))?;
    }
    Ok(())
}

struct Pipelines {
    matmul: ComputePipelineState,
    norm: ComputePipelineState,
    attention: ComputePipelineState,
    rope: ComputePipelineState,
    silu: ComputePipelineState,
    copy: ComputePipelineState,
    add: ComputePipelineState,
    embedding: ComputePipelineState,
}

impl Pipelines {
    fn new(device: &Device, cfg: &Ernie45Config) -> Result<Self, InferenceError> {
        let groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        let source = SHADERS
            .replace("__FA_HEAD_DIM__", &cfg.head_dim.to_string())
            .replace("__FA_GQA_GROUPS__", &groups.to_string())
            .replace("__FUSED_C_HEAD_DIM__", &cfg.head_dim.to_string())
            .replace("__FUSED_C_HALF_DIM__", &(cfg.head_dim / 2).to_string())
            .replace("__FUSED_C_THREADS__", &(cfg.head_dim / 2).to_string());
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
        let attention_threads = u64::from(TILE_Q * SIMD_WIDTH) * groups as u64;
        let limits = device.max_threads_per_threadgroup();
        if limits.width < attention_threads.max(ELEMENT_THREADS) || limits.height < 16 {
            return Err(runtime(
                "device threadgroup dimensions cannot support the kernels",
            ));
        }
        let attention = make("fused_attention", attention_threads)?;
        if attention.thread_execution_width() != u64::from(SIMD_WIDTH) {
            return Err(runtime(
                "fused attention requires a 32-lane SIMD execution width",
            ));
        }
        Ok(Self {
            matmul: make("matmul_bt", ELEMENT_THREADS)?,
            norm: make("rms_norm", ELEMENT_THREADS)?,
            attention,
            rope: make("ernie45_rope", ELEMENT_THREADS)?,
            silu: make("silu_mul", ELEMENT_THREADS)?,
            copy: make("copy_buf", ELEMENT_THREADS)?,
            add: make("add_buf", ELEMENT_THREADS)?,
            embedding: make("ernie45_embedding", ELEMENT_THREADS)?,
        })
    }
}

struct LayerWeights {
    q: Buffer,
    k: Buffer,
    v: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    input_norm: Buffer,
    post_norm: Buffer,
}

impl LayerWeights {
    fn new(device: &Device, weights: &Ernie45LayerWeights) -> Result<Self, InferenceError> {
        Ok(Self {
            q: upload(device, &weights.q_proj, "ernie45.q_proj")?,
            k: upload(device, &weights.k_proj, "ernie45.k_proj")?,
            v: upload(device, &weights.v_proj, "ernie45.v_proj")?,
            o: upload(device, &weights.o_proj, "ernie45.o_proj")?,
            gate: upload(device, &weights.gate_proj, "ernie45.gate_proj")?,
            up: upload(device, &weights.up_proj, "ernie45.up_proj")?,
            down: upload(device, &weights.down_proj, "ernie45.down_proj")?,
            input_norm: upload(device, &weights.input_layernorm, "ernie45.input_norm")?,
            post_norm: upload(
                device,
                &weights.post_attention_layernorm,
                "ernie45.post_norm",
            )?,
        })
    }
}

struct Weights {
    embedding: Buffer,
    layers: Vec<LayerWeights>,
    final_norm: Buffer,
    lm_head: Buffer,
}

struct Activations {
    hidden: Buffer,
    normed: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention: Buffer,
    attn_proj: Buffer,
    gate: Buffer,
    up: Buffer,
    ffn_out: Buffer,
    logits: Buffer,
    token_ids: Buffer,
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
            (bytes, MTLResourceOptions::StorageModeShared),
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
    // SAFETY: allocate created a suitably aligned shared buffer for values.len() f32s.
    // This new buffer has not been submitted to the GPU and cannot alias the input slice.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().cast::<f32>(),
            values.len(),
        );
    }
    Ok(buffer)
}

/// Metal f32 text prefill with persistent weights and scratch buffers.
///
/// Each call starts at position zero and produces contiguous logits for every
/// input position. This backend requires 128-dimensional heads and does not
/// retain a KV cache between calls.
pub struct MetalErnie45State {
    _device: Device,
    queue: CommandQueue,
    command_buffer_selector: objc::runtime::Sel,
    compute_encoder_selector: objc::runtime::Sel,
    pipelines: Pipelines,
    weights: Weights,
    activations: Activations,
    cos: Buffer,
    sin: Buffer,
    cfg: Ernie45Config,
    max_seq_len: usize,
    #[cfg(all(test, feature = "f16"))]
    pending_counts: Cell<[u32; 7]>,
    #[cfg(all(test, feature = "f16"))]
    last_counts: [u32; 7],
}

impl MetalErnie45State {
    /// Upload a complete decoder and allocate capacity for `max_seq_len` tokens.
    ///
    /// # Errors
    /// Returns an error for unsupported geometry, invalid configuration, tensor
    /// shapes or non-finite weights, excessive capacities, unavailable Metal, or
    /// shader and pipeline creation failures. Validation precedes device access.
    pub fn new(
        cfg: &Ernie45Config,
        weights: &Ernie45Weights,
        max_seq_len: usize,
    ) -> Result<Self, InferenceError> {
        validate_shape(cfg, max_seq_len)?;
        validate_weights(cfg, weights)?;
        let inv_freq = gemma4_rope_inv_freq(cfg.head_dim, cfg.rope_theta, None);
        let positions: Vec<u32> = (0..max_seq_len as u32).collect();
        let (full_cos, full_sin) = gemma4_rope_cos_sin(&inv_freq, &positions);
        let compact = |full: &[f32]| -> Vec<f32> {
            full.chunks_exact(cfg.head_dim)
                .flat_map(|row| row[..cfg.head_dim / 2].iter().copied())
                .collect()
        };
        let cos_values = compact(&full_cos);
        let sin_values = compact(&full_sin);
        if cos_values.iter().chain(&sin_values).any(|v| !v.is_finite()) {
            return Err(invalid("rope_theta produces non-finite RoPE tables"));
        }
        let device =
            Device::system_default().ok_or_else(|| runtime("no Metal device available"))?;
        let pipelines = Pipelines::new(&device, cfg)?;
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
        let command_buffer_selector = objc::runtime::Sel::register("commandBuffer");
        let compute_encoder_selector = objc::runtime::Sel::register("computeCommandEncoder");
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let activations = Activations {
            hidden: allocate(&device, max_seq_len * h, "ernie45.hidden")?,
            normed: allocate(&device, max_seq_len * h, "ernie45.normed")?,
            q: allocate(&device, max_seq_len * q_dim, "ernie45.q")?,
            k: allocate(&device, max_seq_len * kv_dim, "ernie45.k")?,
            v: allocate(&device, max_seq_len * kv_dim, "ernie45.v")?,
            attention: allocate(&device, max_seq_len * q_dim, "ernie45.attention")?,
            attn_proj: allocate(&device, max_seq_len * h, "ernie45.attn_proj")?,
            gate: allocate(&device, max_seq_len * i, "ernie45.gate")?,
            up: allocate(&device, max_seq_len * i, "ernie45.up")?,
            ffn_out: allocate(&device, max_seq_len * h, "ernie45.ffn_out")?,
            logits: allocate(&device, max_seq_len * cfg.vocab_size, "ernie45.logits")?,
            token_ids: allocate(&device, max_seq_len, "ernie45.token_ids")?,
        };
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(cfg.num_hidden_layers)
            .map_err(|error| runtime(format!("layer buffer allocation failed: {error}")))?;
        for layer in &weights.layers {
            layers.push(LayerWeights::new(&device, layer)?);
        }
        let weights = Weights {
            embedding: upload(&device, &weights.embed_tokens, "ernie45.embedding")?,
            layers,
            final_norm: upload(&device, &weights.final_norm, "ernie45.final_norm")?,
            lm_head: upload(&device, &weights.lm_head, "ernie45.lm_head")?,
        };
        let cos = upload(&device, &cos_values, "ernie45.rope_cos")?;
        let sin = upload(&device, &sin_values, "ernie45.rope_sin")?;
        Ok(Self {
            _device: device,
            queue,
            command_buffer_selector,
            compute_encoder_selector,
            pipelines,
            weights,
            activations,
            cos,
            sin,
            cfg: cfg.clone(),
            max_seq_len,
            #[cfg(all(test, feature = "f16"))]
            pending_counts: Cell::new([0; 7]),
            #[cfg(all(test, feature = "f16"))]
            last_counts: [0; 7],
        })
    }

    /// Counts from the last prefill, zero unless it completed successfully.
    #[cfg(all(test, feature = "f16"))]
    pub(super) fn last_dispatch_counts(&self) -> (u32, u32, u32, u32, u32, u32, u32) {
        let [matmul, norm, rope, attention, add, embedding, silu] = self.last_counts;
        (matmul, norm, rope, attention, add, embedding, silu)
    }

    /// Prefill text tokens from position zero into contiguous `[ids.len(), vocab_size]` logits.
    ///
    /// `logits` must contain exactly `ids.len() * vocab_size` elements; every
    /// position is returned in token-major order. Device buffers are reused and
    /// the caller's output is changed only after successful GPU completion and
    /// validation that all logits are finite.
    ///
    /// # Errors
    /// Returns an error for an empty or excessive sequence, out-of-vocabulary
    /// token IDs, an incorrect output length, failed GPU execution, or non-finite
    /// logits. Errors leave `logits` unchanged; there is no CPU fallback.
    pub fn prefill(&mut self, ids: &[u32], logits: &mut [f32]) -> Result<(), InferenceError> {
        #[cfg(all(test, feature = "f16"))]
        {
            self.last_counts = [0; 7];
            self.pending_counts.set([0; 7]);
        }
        let seq_len = ids.len();
        if seq_len == 0 || seq_len > self.max_seq_len {
            return Err(invalid("seq_len is outside the allocated prefill capacity"));
        }
        let count = elements("logits", seq_len, self.cfg.vocab_size)?;
        if logits.len() != count {
            return Err(invalid(
                "logits must have the exact [seq_len,vocab_size] shape",
            ));
        }
        if ids.iter().any(|&id| id as usize >= self.cfg.vocab_size) {
            return Err(invalid("token ID is outside the vocabulary"));
        }
        #[cfg(all(test, feature = "f16"))]
        self.poison_destinations();
        // SAFETY: exclusive access and completion of every previous prefill exclude
        // GPU/CPU readers. The shared allocation holds max_seq_len aligned four-byte
        // elements, so the validated u32 IDs fit and cannot alias this private buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ids.as_ptr(),
                self.activations.token_ids.contents().cast::<u32>(),
                seq_len,
            );
        }
        // Metal command buffers and encoders are autoreleased even on Rust worker threads.
        objc::rc::autoreleasepool(|| {
            let s = seq_len as u32;
            let h = self.cfg.hidden_size as u32;
            let vocab = self.cfg.vocab_size as u32;
            let a = &self.activations;
            // SAFETY: commandBuffer takes no arguments and returns a nullable,
            // autoreleased MTLCommandBuffer. The checked reference stays inside this
            // pool, which remains alive through command completion and readback.
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
            self.embedding(encoder, s, h);
            for weights in &self.weights.layers {
                self.encode_layer(encoder, weights, s);
            }
            self.copy(encoder, &a.hidden, &a.normed, s * h);
            self.norm(encoder, &a.normed, &self.weights.final_norm, s, h);
            self.matmul(
                encoder,
                &a.normed,
                &self.weights.lm_head,
                &a.logits,
                s,
                vocab,
                h,
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(runtime(format!(
                    "command buffer ended with {:?}",
                    command.status()
                )));
            }
            // SAFETY: the command completed before this CPU read. The private shared
            // buffer contains at least count aligned, initialized f32 elements, and
            // exclusive access to the state prevents submission while the slice exists.
            let output =
                unsafe { std::slice::from_raw_parts(a.logits.contents().cast::<f32>(), count) };
            if output.iter().any(|value| !value.is_finite()) {
                return Err(runtime("prefill produced non-finite logits"));
            }
            logits.copy_from_slice(output);
            #[cfg(all(test, feature = "f16"))]
            {
                self.last_counts = self.pending_counts.get();
            }
            Ok(())
        })
    }

    #[cfg(all(test, feature = "f16"))]
    pub(super) fn replace_last_down_projection_for_test(
        &mut self,
        values: &[f32],
    ) -> Result<(), InferenceError> {
        self.last_counts = [0; 7];
        self.pending_counts.set([0; 7]);
        validate_values(
            "last down projection",
            values,
            self.cfg.hidden_size * self.cfg.intermediate_size,
        )?;
        let layer = self
            .weights
            .layers
            .last()
            .ok_or_else(|| invalid("no layers"))?;
        // SAFETY: construction validated this private shared allocation's exact
        // f32 capacity. Every prefill waits before returning, and &mut self excludes
        // concurrent GPU submission or host access; values cannot alias the buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                layer.down.contents().cast::<f32>(),
                values.len(),
            );
        }
        Ok(())
    }

    #[cfg(all(test, feature = "f16"))]
    pub(super) fn last_down_projection_matches_for_test(&self, values: &[f32]) -> bool {
        if values.len() != self.cfg.hidden_size * self.cfg.intermediate_size {
            return false;
        }
        let Some(layer) = self.weights.layers.last() else {
            return false;
        };
        // SAFETY: the private allocation holds values.len() aligned f32 elements.
        // All GPU work completes before a state method returns, and this borrow
        // excludes the mutable borrow needed to submit work or replace the weights.
        let actual = unsafe {
            std::slice::from_raw_parts(layer.down.contents().cast::<f32>(), values.len())
        };
        actual
            .iter()
            .zip(values)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    fn encode_layer(&self, encoder: &ComputeCommandEncoderRef, w: &LayerWeights, s: u32) {
        let h = self.cfg.hidden_size as u32;
        let q = (self.cfg.num_attention_heads * self.cfg.head_dim) as u32;
        let kv = (self.cfg.num_key_value_heads * self.cfg.head_dim) as u32;
        let i = self.cfg.intermediate_size as u32;
        let a = &self.activations;
        self.copy(encoder, &a.hidden, &a.normed, s * h);
        self.norm(encoder, &a.normed, &w.input_norm, s, h);
        self.matmul(encoder, &a.normed, &w.q, &a.q, s, q, h);
        self.matmul(encoder, &a.normed, &w.k, &a.k, s, kv, h);
        self.matmul(encoder, &a.normed, &w.v, &a.v, s, kv, h);
        self.rope(encoder, &a.q, s, self.cfg.num_attention_heads as u32);
        self.rope(encoder, &a.k, s, self.cfg.num_key_value_heads as u32);
        self.attention(encoder, s, q, kv);
        self.matmul(encoder, &a.attention, &w.o, &a.attn_proj, s, h, q);
        self.add(encoder, &a.attn_proj, &a.hidden, s * h);
        self.copy(encoder, &a.hidden, &a.normed, s * h);
        self.norm(encoder, &a.normed, &w.post_norm, s, h);
        self.matmul(encoder, &a.normed, &w.gate, &a.gate, s, i, h);
        self.matmul(encoder, &a.normed, &w.up, &a.up, s, i, h);
        encoder.set_compute_pipeline_state(&self.pipelines.silu);
        bind(encoder, 0, &a.gate);
        bind(encoder, 1, &a.up);
        scalar(encoder, 2, &(s * i));
        element_dispatch(encoder, s * i);
        #[cfg(all(test, feature = "f16"))]
        self.record(6);
        self.matmul(encoder, &a.gate, &w.down, &a.ffn_out, s, h, i);
        self.add(encoder, &a.ffn_out, &a.hidden, s * h);
    }

    #[cfg(all(test, feature = "f16"))]
    fn poison_destinations(&mut self) {
        let a = &self.activations;
        for buffer in [
            &a.hidden,
            &a.normed,
            &a.q,
            &a.k,
            &a.v,
            &a.attention,
            &a.attn_proj,
            &a.gate,
            &a.up,
            &a.ffn_out,
            &a.logits,
        ] {
            let count = buffer.length() as usize / size_of::<f32>();
            // SAFETY: every allocation is a private, aligned, CPU-accessible shared f32
            // buffer. &mut self and the previous prefill's completion exclude concurrent
            // access, and no command for this prefill has been encoded or submitted yet.
            unsafe {
                std::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), count)
                    .fill(f32::NAN);
            }
        }
    }

    #[cfg(all(test, feature = "f16"))]
    fn record(&self, index: usize) {
        let mut counts = self.pending_counts.get();
        counts[index] += 1;
        self.pending_counts.set(counts);
    }

    fn embedding(&self, enc: &ComputeCommandEncoderRef, rows: u32, hidden: u32) {
        enc.set_compute_pipeline_state(&self.pipelines.embedding);
        bind(enc, 0, &self.weights.embedding);
        bind(enc, 1, &self.activations.token_ids);
        bind(enc, 2, &self.activations.hidden);
        scalar(enc, 3, &hidden);
        scalar(enc, 4, &(rows * hidden));
        element_dispatch(enc, rows * hidden);
        #[cfg(all(test, feature = "f16"))]
        self.record(5);
    }

    fn matmul(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        enc.set_compute_pipeline_state(&self.pipelines.matmul);
        bind(enc, 0, a);
        bind(enc, 1, b);
        bind(enc, 2, c);
        scalar(enc, 3, &m);
        scalar(enc, 4, &n);
        scalar(enc, 5, &k);
        enc.dispatch_thread_groups(
            MTLSize::new(u64::from(n).div_ceil(16), u64::from(m).div_ceil(16), 1),
            MTLSize::new(16, 16, 1),
        );
        #[cfg(all(test, feature = "f16"))]
        self.record(0);
    }

    fn norm(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, w: &Buffer, rows: u32, width: u32) {
        enc.set_compute_pipeline_state(&self.pipelines.norm);
        bind(enc, 0, x);
        bind(enc, 1, w);
        scalar(enc, 2, &width);
        scalar(enc, 3, &rows);
        scalar(enc, 4, &self.cfg.rms_norm_eps);
        enc.dispatch_thread_groups(MTLSize::new(u64::from(rows), 1, 1), MTLSize::new(256, 1, 1));
        #[cfg(all(test, feature = "f16"))]
        self.record(1);
    }

    fn rope(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, rows: u32, heads: u32) {
        let head_dim = self.cfg.head_dim as u32;
        enc.set_compute_pipeline_state(&self.pipelines.rope);
        bind(enc, 0, x);
        bind(enc, 1, &self.cos);
        bind(enc, 2, &self.sin);
        scalar(enc, 3, &rows);
        scalar(enc, 4, &heads);
        scalar(enc, 5, &head_dim);
        element_dispatch(enc, rows * heads * (head_dim / 2));
        #[cfg(all(test, feature = "f16"))]
        self.record(2);
    }

    fn attention(&self, enc: &ComputeCommandEncoderRef, rows: u32, q: u32, kv: u32) {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Params {
            seq_len: u32,
            q_dim4: u32,
            kv_dim4: u32,
            num_kv_heads: u32,
            scale: f32,
            pad: [u32; 3],
        }
        let params = Params {
            seq_len: rows,
            q_dim4: q / 4,
            kv_dim4: kv / 4,
            num_kv_heads: self.cfg.num_key_value_heads as u32,
            scale: 1.0 / (self.cfg.head_dim as f32).sqrt(),
            pad: [0; 3],
        };
        let groups = (self.cfg.num_attention_heads / self.cfg.num_key_value_heads) as u32;
        enc.set_compute_pipeline_state(&self.pipelines.attention);
        bind(enc, 0, &self.activations.q);
        bind(enc, 1, &self.activations.k);
        bind(enc, 2, &self.activations.v);
        bind(enc, 3, &self.activations.attention);
        scalar(enc, 4, &params);
        enc.dispatch_thread_groups(
            MTLSize::new(
                u64::from(params.num_kv_heads),
                u64::from(rows.div_ceil(TILE_Q)),
                1,
            ),
            MTLSize::new(u64::from(TILE_Q * groups * SIMD_WIDTH), 1, 1),
        );
        #[cfg(all(test, feature = "f16"))]
        self.record(3);
    }

    fn copy(&self, enc: &ComputeCommandEncoderRef, src: &Buffer, dst: &Buffer, count: u32) {
        enc.set_compute_pipeline_state(&self.pipelines.copy);
        bind(enc, 0, src);
        bind(enc, 1, dst);
        scalar(enc, 2, &count);
        element_dispatch(enc, count);
    }

    fn add(&self, enc: &ComputeCommandEncoderRef, src: &Buffer, dst: &Buffer, count: u32) {
        enc.set_compute_pipeline_state(&self.pipelines.add);
        bind(enc, 0, src);
        bind(enc, 1, dst);
        scalar(enc, 2, &count);
        element_dispatch(enc, count);
        #[cfg(all(test, feature = "f16"))]
        self.record(4);
    }
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
