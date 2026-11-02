//! Shared Metal buffers, command encoding, pipeline construction and shader sources.

#![cfg(target_os = "macos")]
#![warn(missing_docs)]

#[cfg(feature = "metal-gpu")]
use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

#[cfg(feature = "metal-gpu")]
use lattice_inference_core::InferenceError;

/// Repository-internal guards shared by GPU tests and measurement targets.
///
/// This module is hidden from generated documentation and is not a supported
/// production API. It is exported because Cargo builds integration tests, benches,
/// examples, and binaries as separate crates, here and in the crates above this one.
#[cfg(feature = "measurement")]
#[doc(hidden)]
pub mod measurement;
#[cfg(feature = "metal-gpu")]
pub mod msl;

/// Threads per group for elementwise dispatch.
#[cfg(feature = "metal-gpu")]
pub const ELEMENT_THREADS: u64 = 256;

/// Bind the entire buffer at offset zero to a compute argument.
#[cfg(feature = "metal-gpu")]
pub fn bind(enc: &ComputeCommandEncoderRef, index: u64, buffer: &Buffer) {
    enc.set_buffer(index, Some(buffer), 0);
}

/// Copy one initialized value into a compute argument.
///
/// `NoUninit` excludes uninitialized padding from the bytes copied by Metal.
/// A compile-time assertion rejects zero-sized values and values of 4096 bytes
/// or more, so this helper only uploads nonempty values and stays below Metal's
/// documented 4 KB limit.
/// See Apple's [`setBytes:length:atIndex:` documentation](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:)).
#[cfg(feature = "metal-gpu")]
pub fn scalar<T: bytemuck::NoUninit>(enc: &ComputeCommandEncoderRef, index: u64, value: &T) {
    const { assert!(size_of::<T>() > 0 && size_of::<T>() < 4096) };
    enc.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}

/// Dispatch enough elementwise thread groups to cover `count` elements.
#[cfg(feature = "metal-gpu")]
pub fn element_dispatch(enc: &ComputeCommandEncoderRef, count: u32) {
    enc.dispatch_thread_groups(
        MTLSize::new(u64::from(count).div_ceil(ELEMENT_THREADS), 1, 1),
        MTLSize::new(ELEMENT_THREADS, 1, 1),
    );
}

/// Allocation length to request for an empty f32 buffer.
#[derive(Clone, Copy)]
#[cfg(feature = "metal-gpu")]
pub enum ZeroLengthPolicy {
    /// Request exactly zero bytes for an empty buffer.
    Exact,
    /// Request at least one f32 for an empty buffer.
    MinimumF32,
}

#[cfg(feature = "metal-gpu")]
impl ZeroLengthPolicy {
    fn allocation_length(self, bytes: u64) -> u64 {
        match self {
            Self::Exact => bytes,
            Self::MinimumF32 => bytes.max(4),
        }
    }
}

/// Empty-allocation behavior and error constructors for buffer operations.
#[derive(Clone, Copy)]
#[cfg(feature = "metal-gpu")]
pub struct BufferPolicy {
    /// Length policy for an empty allocation.
    pub zero_length: ZeroLengthPolicy,
    /// Constructor for invalid element counts or host lengths.
    pub invalid: fn(String) -> InferenceError,
    /// Constructor for device allocation and accessibility errors.
    pub runtime: fn(String) -> InferenceError,
}

/// Copy strategy for initializing a newly allocated shared buffer.
#[cfg(feature = "metal-gpu")]
pub enum UploadCopy {
    /// Copy f32 values directly into shared storage.
    Direct,
    /// Delegate copying to a caller-supplied checked writer.
    Checked(fn(&Buffer, &[f32]) -> Result<(), InferenceError>),
}

/// Check the element count, u32 indexing range and host f32 slice capacity.
#[cfg(feature = "metal-gpu")]
pub fn elements(
    name: &str,
    rows: usize,
    width: usize,
    invalid: fn(String) -> InferenceError,
) -> Result<usize, InferenceError> {
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

/// Allocate a labeled, CPU-accessible shared buffer for `count` f32 values.
///
/// Checks indexing, host slice and device capacity before allocation. A nil
/// allocation or inaccessible contents is an error. The returned buffer has at
/// least the requested byte length; `policy` controls empty allocation length.
#[cfg(feature = "metal-gpu")]
pub fn allocate(
    device: &Device,
    count: usize,
    label: &str,
    policy: BufferPolicy,
) -> Result<Buffer, InferenceError> {
    let runtime = policy.runtime;
    let count = elements(label, count, 1, policy.invalid)?;
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
            (
                policy.zero_length.allocation_length(bytes),
                MTLResourceOptions::StorageModeShared,
            ),
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

/// Allocate a labeled, CPU-accessible shared buffer and copy `values` into it.
///
/// Uses the nil, capacity and length checks of [`allocate`], then copies directly
/// or calls the supplied checked writer. Empty uploads follow `policy`.
#[cfg(feature = "metal-gpu")]
pub fn upload(
    device: &Device,
    values: &[f32],
    label: &str,
    policy: BufferPolicy,
    copy: UploadCopy,
) -> Result<Buffer, InferenceError> {
    let buffer = allocate(device, values.len(), label, policy)?;
    match copy {
        UploadCopy::Direct => {
            // SAFETY: allocate created a suitably aligned shared buffer for values.len() f32s.
            // This new buffer has not been submitted to the GPU and cannot alias the input slice.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().cast::<f32>(),
                    values.len(),
                );
            }
        }
        UploadCopy::Checked(write) => write(&buffer, values)?,
    }
    Ok(buffer)
}

/// Compile a Metal shader library with fast math disabled.
#[cfg(feature = "metal-gpu")]
pub fn compile_library(
    device: &Device,
    source: &str,
    runtime: fn(String) -> InferenceError,
) -> Result<Library, InferenceError> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    device
        .new_library_with_source(source, &options)
        .map_err(|error| runtime(format!("shader compilation failed: {error}")))
}

/// Create a named compute pipeline and check thread and shared-memory capacity.
#[cfg(feature = "metal-gpu")]
pub fn make_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
    threads: u64,
    runtime: fn(String) -> InferenceError,
) -> Result<ComputePipelineState, InferenceError> {
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
    if pipeline.static_threadgroup_memory_length() > device.max_threadgroup_memory_length() {
        return Err(runtime(format!(
            "{name} exceeds threadgroup memory capacity"
        )));
    }
    Ok(pipeline)
}

#[cfg(all(test, feature = "metal-gpu"))]
mod tests {
    use super::*;

    #[test]
    fn zero_length_policies_set_the_requested_length() {
        assert_eq!(ZeroLengthPolicy::Exact.allocation_length(0), 0);
        assert_eq!(ZeroLengthPolicy::MinimumF32.allocation_length(0), 4);
        for bytes in [4, 12, 4096] {
            assert_eq!(ZeroLengthPolicy::Exact.allocation_length(bytes), bytes);
            assert_eq!(ZeroLengthPolicy::MinimumF32.allocation_length(bytes), bytes);
        }
    }
}
