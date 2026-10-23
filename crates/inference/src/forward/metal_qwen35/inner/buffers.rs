use super::*;

// ---------------------------------------------------------------------------
// GPU Buffer Structures
// ---------------------------------------------------------------------------

/// A quantized weight buffer, paired with the byte offset within that buffer where
/// the actual Q4 payload starts.
///
/// For whole-file mmap buffers (H1 optimisation): `buffer` spans the entire
/// mmapped file and `payload_offset` skips the file header.  For regular
/// `new_buffer_with_data` buffers (Q8_0 path and concatenated Q4 tensors):
/// `payload_offset` is 0.  Dispatch helpers always bind via
/// `enc.set_buffer(slot, Some(&qw.buffer), qw.payload_offset)` so both cases
/// are handled uniformly.
pub(crate) struct Q4WeightBuf {
    pub(super) buffer: Buffer,
    /// Byte offset of Q4 payload within `buffer`.
    pub(super) payload_offset: u64,
    // Keeps the mmap alive as long as the Metal buffer needs the memory.
    _mmap: Option<memmap2::Mmap>,
}

impl Q4WeightBuf {
    /// Wrap a normally-allocated Metal buffer (payload at offset 0).
    pub(super) fn from_buffer(buffer: Buffer) -> Self {
        Q4WeightBuf {
            buffer,
            payload_offset: 0,
            _mmap: None,
        }
    }

    /// Wrap a no-copy Metal buffer backed by `mmap` with a non-zero payload offset.
    ///
    /// The GPU is the only thing that will ever read these bytes, so the
    /// caller must present the [`Q4BlocksChecked`] witness that
    /// `open_and_mmap_q4_file` issues for an eagerly checked mapping.
    /// A caller that deferred the block check has no witness to pass and
    /// cannot reach this constructor without visibly unwrapping a `None`.
    fn from_mmap(
        buffer: Buffer,
        payload_offset: u64,
        mmap: memmap2::Mmap,
        _blocks_checked: lattice_weights::q4_weights::Q4BlocksChecked,
    ) -> Self {
        Q4WeightBuf {
            buffer,
            payload_offset,
            _mmap: Some(mmap),
        }
    }
}

/// Open `path`, mmap the whole file, and register it as a Metal no-copy
/// `StorageModeShared` buffer.  Stores the mmap owner inside the returned
/// [`Q4WeightBuf`] so the pages remain valid for the lifetime of the buffer.
///
/// This is the one Q4 load path whose bytes are never decoded on the CPU:
/// the packed nibbles stay resident in the no-copy GPU buffer and the
/// per-block scale/bias is dequantized by the Metal kernel at dispatch
/// time. There is therefore no traversal here to fold a scale/bias check
/// into, and a file with a correct header but a NaN or infinite scale
/// would otherwise load cleanly and propagate the NaN into the logits at
/// the first GEMV. The load requests [`Q4BlockCheck::Now`] so the check
/// happens before the buffer exists.
///
/// That costs one sequential pass over the payload, faulting in pages
/// this no-copy mapping was otherwise designed not to touch — a real cost
/// paid once per weight at load, deliberately, because the alternative is
/// admitting non-finite weights to the GPU.
///
/// # Safety invariant
/// The mmap is read-only (`MAP_PRIVATE`) and the model files must not be
/// modified while the process is running.
#[cfg(feature = "metal-gpu")]
pub(super) fn mmap_q4_weight(
    device: &Device,
    path: &std::path::Path,
    label: &str,
    expected_shape: &[usize],
) -> Result<Q4WeightBuf, String> {
    use lattice_weights::q4_weights::{Q4BlockCheck, open_and_mmap_q4_file};

    let tensor_name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("native Q4 tensor");
    let (header, mmap, blocks_checked) = open_and_mmap_q4_file(
        path,
        Some(expected_shape),
        Q4BlockCheck::Now { tensor_name },
    )?;
    let blocks_checked = blocks_checked.ok_or_else(|| {
        format!(
            "{}: block metadata was not checked before no-copy GPU publication",
            path.display()
        )
    })?;

    // The mmap pointer is page-aligned (guaranteed by the OS).  We create the Metal
    // buffer over the *whole* file so the pointer alignment requirement of
    // new_buffer_with_bytes_no_copy is satisfied.  The Q4 payload is then addressed
    // via enc.set_buffer(slot, ..., header.payload_offset) at dispatch time.
    let buf = device.new_buffer_with_bytes_no_copy(
        mmap.as_ptr().cast(),
        mmap.len() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    buf.set_label(label);

    Ok(Q4WeightBuf::from_mmap(
        buf,
        header.payload_offset,
        mmap,
        blocks_checked,
    ))
}

/// A Q3-quantized weight buffer (ADR-072 P1, #420 Stage 2), paired with the
/// byte offset within `buffer` where the Q3 block payload starts.
///
/// Restricted to MLP gate/up/down projections — see
/// [`crate::weights::q3_weights::validate_q3_mlp_role`]. Mirrors
/// [`Q4WeightBuf`]'s buffer/offset/mmap-lifetime structure exactly; kept as
/// a separate type (not a third `Q4WeightBuf` variant) so the MLP-only
/// role restriction is enforced at construction, not by convention.
pub(crate) struct Q3WeightBuf {
    pub(super) buffer: Buffer,
    /// Byte offset of the Q3 payload within `buffer`.
    pub(super) payload_offset: u64,
    // Keeps the mmap alive as long as the Metal buffer needs the memory.
    _mmap: Option<memmap2::Mmap>,
}

impl Q3WeightBuf {
    /// Wrap a normally-allocated Metal buffer (payload at offset 0).
    ///
    /// Mirrors `Q4WeightBuf::from_buffer` for API-shape parity; unused
    /// until a non-mmap Q3 construction path (e.g. concatenated MoE-style
    /// upload) exists.
    #[allow(dead_code)]
    fn from_buffer(buffer: Buffer) -> Self {
        Q3WeightBuf {
            buffer,
            payload_offset: 0,
            _mmap: None,
        }
    }

    /// Wrap a no-copy Metal buffer backed by `mmap` with a non-zero payload offset.
    ///
    /// Exercised today only by [`mmap_q3_weight`] and its tests; live
    /// checkpoint loading does not yet route MLP tensors through it.
    fn from_mmap(buffer: Buffer, payload_offset: u64, mmap: memmap2::Mmap) -> Self {
        Q3WeightBuf {
            buffer,
            payload_offset,
            _mmap: Some(mmap),
        }
    }
}

/// Open `path`, mmap the whole file, and register it as a Metal no-copy
/// `StorageModeShared` buffer holding a Q3 tensor — mirrors [`mmap_q4_weight`].
///
/// Fails closed on two independent conditions before the mmap ever reaches
/// the GPU: `tensor_name` must name an MLP gate/up/down projection (see
/// [`crate::weights::q3_weights::validate_q3_mlp_role`]), and the file's
/// declared header must not claim a payload larger than the file actually
/// holds (see `validate_q3_header_payload_bounds`) — the same truncation
/// hazard `mmap_q4_weight` guards against, since this path never reads the
/// payload bytes itself.
///
/// # Safety invariant
/// The mmap is read-only (`MAP_PRIVATE`) and the model files must not be
/// modified while the process is running. [`lattice_weights::mmap_trust::open_trusted_mmap_file`]
/// enforces the write-boundary half of that invariant (group/other-writable,
/// foreign ownership, a writable ancestor directory, or a permission-granting
/// extended ACL) the same way [`mmap_q4_weight`] does; same-uid mutation
/// through a second fd remains the documented residual, as for every mmap
/// site in this crate.
///
/// Not yet called from live checkpoint loading (proven end-to-end by this
/// module's `mmap_q3_weight_*` tests instead).
#[cfg(feature = "metal-gpu")]
#[allow(dead_code)]
pub(super) fn mmap_q3_weight(
    device: &Device,
    path: &std::path::Path,
    tensor_name: &str,
    label: &str,
) -> Result<Q3WeightBuf, String> {
    use crate::weights::q3_weights::validate_q3_mlp_role;

    validate_q3_mlp_role(tensor_name).map_err(|e| e.to_string())?;

    let mut handle = lattice_weights::mmap_trust::open_trusted_mmap_file(path)?;
    let header = crate::weights::q3_weights::read_q3_header(&mut handle)
        .map_err(|e| format!("failed to parse Q3 header {}: {e}", path.display()))?;
    let file_len = handle
        .len()
        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
    lattice_weights::q3_weights::validate_q3_header_payload_bounds(&header, file_len, path)
        .map_err(|e| format!("failed to validate Q3 payload {}: {e}", path.display()))?;
    let mmap = lattice_weights::mmap_trust::map_and_verify_trusted(&handle)?;

    let buf = device.new_buffer_with_bytes_no_copy(
        mmap.as_ptr().cast(),
        mmap.len() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    buf.set_label(label);

    Ok(Q3WeightBuf::from_mmap(buf, header.payload_offset, mmap))
}

/// Reusable activation buffers on GPU (pre-allocated for single-token decode).
pub(crate) struct MetalQwen35Activations {
    pub(super) hidden: Buffer,      // [hidden_size]
    pub(super) residual: Buffer,    // [hidden_size]
    pub(super) attn_out: Buffer,    // [max(q_dim, hidden_size)] for attention output
    pub(super) q: Buffer,           // [2 * q_dim] for Q+gate interleaved from q_proj
    pub(super) q_separated: Buffer, // [q_dim] Q after scatter
    pub(super) gate_z: Buffer,      // [q_dim] gate after scatter
    pub(super) k: Buffer,           // [kv_dim]
    pub(super) v: Buffer,           // [kv_dim]
    pub(super) gate: Buffer, // [2 * intermediate_size] for Dense decode gate||up; batch uses first bp rows
    pub(super) up: Buffer,   // [intermediate_size] kept for batch separate-buffer paths
    pub(super) ffn_out: Buffer, // [hidden_size]
    // GDN projection buffers
    pub(super) gdn_qkv: Buffer, // [qkv_dim]  — used by batch/prefill and CPU paths
    pub(super) gdn_z: Buffer,   // [output_dim] — used by batch/prefill and CPU paths
    pub(super) gdn_qkvz: Buffer, // [qkv_dim+output_dim] — decode-only fused projection scratch
    pub(super) gdn_key_scratch: Buffer, // [num_key_heads*(2*key_dim+1) + num_value_heads*2] — H3 split key/value scratch
    pub(super) gdn_raw_out: Buffer,     // [output_dim] — H1 raw recurrence output before norm+SiLU
    pub(super) gdn_chunk: GdnChunkScratch, // chunked-scan scratch (default path; serial if LATTICE_GDN_CHUNKED=0)
    pub(super) logits: Buffer,             // [vocab_size]
    // Top-k compact readback buffers (zero-initialized; sized for MAX_TOP_K=256).
    pub(super) topk_scratch_a: Buffer, // [first_pass_groups * MAX_TOP_K] TopKCandidate pairs
    pub(super) topk_scratch_b: Buffer, // same — ping-pong merge target
    // Flash decode partitioned path (H3): [max_partitions * num_q_heads * (head_dim+2)] f32
    pub(super) attn_partials: Buffer,
    // MTP: pre-final hidden state (before final RMSNorm) for the last processed token.
    // The draft head applies the final norm to it on the CPU on unrotated
    // checkpoints (`MtpHiddenTap`), so this buffer stays pre-norm.
    pub(super) pre_final_hidden: Buffer, // [hidden_size]
    // Pooled-embedding capture: the last processed token's hidden state
    // AFTER the final RMSNorm. Deliberately a separate buffer from
    // `pre_final_hidden` above rather than a reuse: the two hold the same
    // token at different points in the network, they differ by the norm's
    // rescale, and a name that reads plausibly for both is how a caller
    // ends up with numbers from the wrong layer. See
    // `encode_capture_final_hidden`.
    pub(super) final_hidden: Buffer, // [hidden_size]
    // MTP: logits for all K verified tokens (K <= MTP_VERIFY_MAX_TOKENS).
    pub(super) verify_logits: Buffer, // [MTP_VERIFY_MAX_TOKENS * vocab_size]
}

/// KV cache for full-attention layers stored in Metal unified memory.
pub(crate) struct MetalKvCache {
    /// k_cache[full_layer_idx]: flat [max_cache_len * kv_dim]
    pub(super) k_bufs: Vec<Buffer>,
    /// v_cache[full_layer_idx]: flat [max_cache_len * kv_dim]
    pub(super) v_bufs: Vec<Buffer>,
    /// Current number of tokens cached.
    pub(super) seq_len: usize,
    pub(super) kv_dim: usize,
    pub(super) max_cache_len: usize,
}

/// Create a Metal buffer holding f32 data in shared (unified) memory.
pub(super) fn make_buffer(device: &Device, data: &[f32], label: &str) -> Buffer {
    let byte_len = std::mem::size_of_val(data) as u64;
    let buf = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    buf.set_label(label);
    buf
}

/// Create a Metal buffer holding f16 (half-precision) data converted from f32.
///
/// Converts each f32 element to IEEE-754 half precision at load time. The resulting
/// buffer is half the size of the f32 equivalent and is consumed by `matmul_bt_half`.
pub(super) fn make_buffer_f16(device: &Device, data: &[f32], label: &str) -> Buffer {
    let f16_data: Vec<u16> = data.iter().map(|&x| f32_to_f16(x)).collect();
    let byte_len = (f16_data.len() * std::mem::size_of::<u16>()) as u64;
    let buf = device.new_buffer_with_data(
        f16_data.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    buf.set_label(label);
    buf
}

/// Q8_0 block: f16 scale (2 bytes) + 32 int8 values = 34 bytes.
pub(super) const QK8_0: usize = 32;
pub(super) const Q8_0_BLOCK_SIZE: usize = 2 + QK8_0;
pub(super) const QK4_0: usize = 32;
pub(super) const Q4_0_BLOCK_SIZE: usize = 20;

/// Quantize a row of f32 weights to Q8_0 blocks.
///
/// # Errors
///
/// Returns `Err` if any element in `src` is non-finite (NaN or ±inf).
/// IEEE 754: `NaN > x` is always false, so a fold-max over NaN leaves
/// `amax` unchanged and the NaN is silently cast to 0 by the `clamp` below.
pub(super) fn quantize_row_q8_0(src: &[f32]) -> Result<Vec<u8>, String> {
    assert_eq!(src.len() % QK8_0, 0);
    let nblocks = src.len() / QK8_0;
    let mut packed = Vec::with_capacity(nblocks * Q8_0_BLOCK_SIZE);
    for b in 0..nblocks {
        let block = &src[b * QK8_0..(b + 1) * QK8_0];
        for &v in block {
            if !v.is_finite() {
                return Err(format!(
                    "Q8_0 weight block {b} contains a non-finite value ({v}); \
                     source weights must be finite"
                ));
            }
        }
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        packed.extend_from_slice(&f32_to_f16(d).to_ne_bytes());
        for &v in block {
            packed.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    Ok(packed)
}

/// Create a Metal buffer with Q8_0 quantized weights from f32 data.
pub(super) fn make_buffer_q8_0(
    device: &Device,
    data: &[f32],
    label: &str,
) -> Result<Buffer, String> {
    assert_eq!(data.len() % QK8_0, 0);
    let packed = quantize_row_q8_0(data)?;
    let byte_len = packed.len() as u64;
    let buf = device.new_buffer_with_data(
        packed.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    buf.set_label(label);
    Ok(buf)
}

pub(super) fn make_buffer_q4_0(
    device: &Device,
    data: &[f32],
    label: &str,
) -> Result<Buffer, String> {
    assert_eq!(
        data.len() % QK4_0,
        0,
        "make_buffer_q4_0: data length {} not divisible by {QK4_0}",
        data.len()
    );
    let packed = quantize_row_q4_0(data).map_err(|e| e.to_string())?;
    let byte_len = packed.len() as u64;
    let buf = device.new_buffer_with_data(
        packed.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    buf.set_label(label);
    Ok(buf)
}

pub(super) fn make_zero_buffer(device: &Device, num_floats: usize, label: &str) -> Buffer {
    let byte_len = (num_floats * std::mem::size_of::<f32>()) as u64;
    let buf = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    buf.set_label(label);
    buf
}

/// Allocate a zero-initialised f16 buffer holding `num_halves` half-precision elements.
fn make_zero_buffer_f16(device: &Device, num_halves: usize, label: &str) -> Buffer {
    let byte_len = (num_halves * std::mem::size_of::<u16>()) as u64;
    let buf = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    buf.set_label(label);
    buf
}

/// Allocate a zero-initialised byte buffer (not float-sized).
/// Used for top-k candidate scratch buffers whose element size is 8 bytes.
pub(super) fn make_zero_byte_buffer(device: &Device, byte_len: usize, label: &str) -> Buffer {
    let buf = device.new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
    buf.set_label(label);
    buf
}

/// GemmParams layout matching the MSL GemmParams struct (6 × u32 = 24 bytes).
#[repr(C)]
pub(super) struct GemmParams {
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) lda: u32,
    pub(super) ldb: u32,
    pub(super) ldc: u32,
}

pub(super) fn div_ceil(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

/// Convert f32 to IEEE-754 half-precision (f16) stored as u16 bits.
///
/// Thin wrapper over the single always-compiled scalar encoder in
/// [`lattice_weights::half_bits`] (lattice#799) — this module used to
/// carry its own independent copy of the bit-twiddling arithmetic.
#[inline]
pub(super) fn f32_to_f16(x: f32) -> u16 {
    lattice_weights::half_bits::f32_to_f16_bits(x)
}

/// Convert f16 bits (u16) back to f32.
///
/// Used for CPU-side reads of f16 weight buffers when needed (e.g., small
/// GDN projections dispatched on CPU). Thin wrapper over
/// [`lattice_weights::half_bits::f16_bits_to_f32`] (lattice#799).
#[inline]
pub(super) fn f16_to_f32(bits: u16) -> f32 {
    lattice_weights::half_bits::f16_bits_to_f32(bits)
}

/// Convert a contiguous slice of IEEE-754 half-precision values (stored as u16 bits)
/// to f32, writing into a pre-allocated destination buffer.
///
/// Scalar `f16_to_f32` throughout. A 4-lane NEON fast path (`vcvt_f32_f16`) was
/// tried on aarch64 but requires the nightly-only `stdarch_neon_f16` feature
/// (rust-lang/rust#136306) and broke stable-toolchain builds (#568) — this runs
/// off the GPU decode hot path (one row per token embedding lookup / readback),
/// so the scalar cost is not worth the unstable dependency.
///
/// # Safety
/// - `src` must point to at least `n` contiguous, initialized u16 values.
/// - `dst` must point to at least `n` contiguous, writable f32 values.
/// - `n` may be zero (no-op).
pub(super) unsafe fn convert_f16_row(src: *const u16, dst: *mut f32, n: usize) {
    for i in 0..n {
        *dst.add(i) = f16_to_f32(*src.add(i));
    }
}

/// Read f32 slice from a Metal shared-mode buffer.
///
/// SAFETY: Buffer must be StorageModeShared with sufficient length and no
/// concurrent GPU writes (command buffer completed or not yet submitted).
pub(super) unsafe fn read_buffer(buf: &Buffer, len: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    let mut out = vec![0.0f32; len];
    std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
    out
}

/// Device-side layout of one top-k candidate: `struct TopKCandidate { float logit;
/// uint token_id; }` in the shader.
///
/// Module-scoped rather than local to [`MetalQwen35State::read_topk_candidates`] so the
/// readback sites can size their transfers with `size_of::<GpuCandidate>()`. The shader
/// owns this layout, so a literal byte count at those sites would rot silently the first
/// time the shader struct changed and nothing would fail.
#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GpuCandidate {
    pub(super) logit: f32,
    pub(super) token_id: u32,
}

/// Read f32 slice from a Metal shared-mode buffer starting at a given element offset.
///
/// SAFETY: Same requirements as `read_buffer`; `offset_f32 + len` must be within bounds.
#[allow(dead_code)] // debug/calibration helper used for CPU-side readback at non-zero offsets
pub(super) unsafe fn read_buffer_offset(buf: &Buffer, offset_f32: usize, len: usize) -> Vec<f32> {
    let ptr = (buf.contents() as *const f32).add(offset_f32);
    std::slice::from_raw_parts(ptr, len).to_vec()
}

/// Read f16 data from a Metal shared-mode buffer, converting to f32.
///
/// Used for CPU-side reads of f16 weight buffers (GDN small projections).
///
/// SAFETY: Buffer must be StorageModeShared with sufficient length and no
/// concurrent GPU writes (command buffer completed or not yet submitted).
#[allow(dead_code)] // CPU-side f16 readback helper; used in debugging and future calibration paths
pub(super) unsafe fn read_buffer_f16(buf: &Buffer, len: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const u16;
    let mut out = vec![0.0f32; len];
    // SAFETY: ptr has len u16 values; out.as_mut_ptr() has len f32 values.
    convert_f16_row(ptr, out.as_mut_ptr(), len);
    out
}

/// Write f32 slice into a Metal shared-mode buffer.
///
/// SAFETY: Buffer must be StorageModeShared with sufficient capacity and no
/// concurrent GPU reads (command buffer completed or not yet submitted).
pub(super) unsafe fn write_buffer(buf: &Buffer, data: &[f32]) {
    let dst = buf.contents() as *mut f32;
    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
}

impl MetalKvCache {
    pub(super) fn new(
        device: &Device,
        num_full_layers: usize,
        kv_dim: usize,
        max_cache_len: usize,
        use_kv_f16: bool,
    ) -> Self {
        let num_elems = max_cache_len * kv_dim;
        let mut k_bufs = Vec::with_capacity(num_full_layers);
        let mut v_bufs = Vec::with_capacity(num_full_layers);
        for i in 0..num_full_layers {
            if use_kv_f16 {
                k_bufs.push(make_zero_buffer_f16(
                    device,
                    num_elems,
                    &format!("kv_k_{i}"),
                ));
                v_bufs.push(make_zero_buffer_f16(
                    device,
                    num_elems,
                    &format!("kv_v_{i}"),
                ));
            } else {
                k_bufs.push(make_zero_buffer(device, num_elems, &format!("kv_k_{i}")));
                v_bufs.push(make_zero_buffer(device, num_elems, &format!("kv_v_{i}")));
            }
        }
        let cache = Self {
            k_bufs,
            v_bufs,
            seq_len: 0,
            kv_dim,
            max_cache_len,
        };
        // ADR-064: runtime KV layout assertion (debug builds only).
        // Layout is f16 (2 bytes/elem) when use_kv_f16, f32 (4 bytes/elem) otherwise.
        #[cfg(debug_assertions)]
        if num_full_layers > 0 {
            let element_size = if use_kv_f16 { 2 } else { 4 };
            let expected_bytes = max_cache_len * kv_dim * element_size;
            let layout_name = if use_kv_f16 { "f16" } else { "f32" };
            for (i, (k, v)) in cache.k_bufs.iter().zip(cache.v_bufs.iter()).enumerate() {
                debug_assert_eq!(
                    k.length() as usize,
                    expected_bytes,
                    "KV cache K[{i}] size mismatch: expected {layout_name} layout ({expected_bytes} bytes)"
                );
                debug_assert_eq!(
                    v.length() as usize,
                    expected_bytes,
                    "KV cache V[{i}] size mismatch: expected {layout_name} layout ({expected_bytes} bytes)"
                );
            }
        }
        cache
    }

    pub(super) fn reset(&mut self) {
        self.seq_len = 0;
        // No need to zero the buffers — seq_len controls valid range.
    }
}
