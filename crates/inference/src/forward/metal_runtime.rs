//! Shared Metal helpers for the model states.

use metal::{Buffer, ComputeCommandEncoderRef, Device, MTLResourceOptions, MTLSize};

use crate::error::InferenceError;

pub(crate) const ELEMENT_THREADS: u64 = 256;

pub(crate) fn bind(enc: &ComputeCommandEncoderRef, index: u64, buffer: &Buffer) {
    enc.set_buffer(index, Some(buffer), 0);
}

pub(crate) fn scalar<T: Copy>(enc: &ComputeCommandEncoderRef, index: u64, value: &T) {
    enc.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}

pub(crate) fn element_dispatch(enc: &ComputeCommandEncoderRef, count: u32) {
    enc.dispatch_thread_groups(
        MTLSize::new(u64::from(count).div_ceil(ELEMENT_THREADS), 1, 1),
        MTLSize::new(ELEMENT_THREADS, 1, 1),
    );
}

#[derive(Clone, Copy)]
pub(crate) enum ZeroLengthPolicy {
    Exact,
    MinimumF32,
}

impl ZeroLengthPolicy {
    fn allocation_length(self, bytes: u64) -> u64 {
        match self {
            Self::Exact => bytes,
            Self::MinimumF32 => bytes.max(4),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BufferPolicy {
    pub(crate) zero_length: ZeroLengthPolicy,
    pub(crate) invalid: fn(String) -> InferenceError,
    pub(crate) runtime: fn(String) -> InferenceError,
}

pub(crate) enum UploadCopy {
    Direct,
    Checked(fn(&Buffer, &[f32]) -> Result<(), InferenceError>),
}

pub(crate) fn elements(
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

pub(crate) fn allocate(
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

pub(crate) fn upload(
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
