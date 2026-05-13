use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};

use super::api::{GpuForwardError, Result};

pub(super) const TILE: u32 = 16;
/// Workgroup size for the RMS-normalisation compute shader (threads per group).
///
/// Must match the `@workgroup_size(RMS_WG)` annotation in the WGSL source.
/// Changing this value requires a corresponding shader recompile.
pub(super) const RMS_WG: u32 = 256;
/// Workgroup size for the softmax compute shader (threads per group).
///
/// Must match the `@workgroup_size(SOFTMAX_WG)` annotation in the WGSL source.
/// Changing this value requires a corresponding shader recompile.
pub(super) const SOFTMAX_WG: u32 = 128;
pub(super) const ELEM_WG: u32 = 256;
pub(super) const ROPE_WG: u32 = 64;
pub(super) const PARAM_WORDS: usize = 64;
pub(super) const PARAM_SLOT_BYTES: usize = 256;
pub(super) const PARAM_SLOTS_PER_LAYER: usize = 22;
pub(super) const EXTRA_PARAM_SLOTS: usize = 1;

pub(super) const P_M: usize = 0;
pub(super) const P_N: usize = 1;
pub(super) const P_K: usize = 2;
pub(super) const P_ROW_LEN: usize = 3;
pub(super) const P_NUM_ROWS: usize = 4;
pub(super) const P_HEAD_DIM: usize = 5;
pub(super) const P_NUM_HEADS: usize = 6;
pub(super) const P_NUM_KV_HEADS: usize = 7;
pub(super) const P_GROUPS: usize = 8;
pub(super) const P_HALF_DIM: usize = 9;
pub(super) const P_TOTAL_ELEMS: usize = 10;
pub(super) const P_SCALE: usize = 11;
pub(super) const P_EPS: usize = 12;
pub(super) const P_SEQ_LEN: usize = 13;

#[derive(Debug, Default)]
pub(super) struct AllocationStats {
    pub(super) user_buffer_creations: AtomicU64,
    pub(super) queue_submits: AtomicU64,
}

#[repr(C, align(256))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct DispatchParams {
    words: [u32; PARAM_WORDS],
}

impl DispatchParams {
    #[inline]
    pub(super) fn zeroed() -> Self {
        Self {
            words: [0; PARAM_WORDS],
        }
    }

    #[inline]
    pub(super) fn set_u32(&mut self, index: usize, value: u32) {
        self.words[index] = value;
    }

    #[inline]
    pub(super) fn set_f32(&mut self, index: usize, value: f32) {
        self.words[index] = value.to_bits();
    }
}

pub(super) struct ParamPacker {
    slots: Vec<DispatchParams>,
    max_slots: usize,
}

impl ParamPacker {
    pub(super) fn new(max_slots: usize) -> Self {
        Self {
            slots: Vec::with_capacity(max_slots),
            max_slots,
        }
    }

    pub(super) fn push(&mut self, params: DispatchParams) -> Result<u32> {
        if self.slots.len() >= self.max_slots {
            return Err(GpuForwardError::Limit(format!(
                "parameter slot budget exhausted: {}",
                self.max_slots
            )));
        }
        let offset = (self.slots.len() * PARAM_SLOT_BYTES) as u32;
        self.slots.push(params);
        Ok(offset)
    }

    pub(super) fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.slots)
    }
}
