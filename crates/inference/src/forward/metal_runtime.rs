//! Shared encoder helpers for the Metal model states.

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

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
