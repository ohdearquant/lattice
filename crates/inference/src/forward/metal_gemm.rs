//! Metal GPU-accelerated GEMM for Apple Silicon.
//!
//! Dispatches matrix multiplication to the GPU via Metal compute shaders.
//! Falls back to Accelerate (AMX) when matrices are too small for GPU overhead
//! to be worthwhile.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod gpu {
    use crate::forward::cpu::{validate_gemm_bt, validate_gemm_nn};
    use metal::*;
    use std::sync::OnceLock;

    /// Minimum total work (M*N*K) to justify GPU dispatch.
    /// Below this, CPU AMX is faster due to GPU launch overhead.
    /// Minimum work to dispatch to GPU. Set low to maximize GPU utilization —
    /// even moderate matrices benefit from 30+ GPU cores on Apple Silicon.
    const GPU_DISPATCH_THRESHOLD: u64 = 64 * 64 * 64;

    struct MetalState {
        device: Device,
        queue: CommandQueue,
        pipeline_bt: ComputePipelineState,
        pipeline_nn: ComputePipelineState,
    }

    static METAL: OnceLock<Option<MetalState>> = OnceLock::new();

    const SHADER_SOURCE: &str = include_str!("shaders/gemm.metal");

    fn init_metal() -> Option<MetalState> {
        let device = Device::system_default()?;
        tracing::info!(
            name = device.name(),
            "Metal GPU initialized for GEMM acceleration"
        );

        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_SOURCE, &opts).ok()?;

        let fn_bt = library.get_function("gemm_bt", None).ok()?;
        let fn_nn = library.get_function("gemm_nn", None).ok()?;

        let pipeline_bt = device
            .new_compute_pipeline_state_with_function(&fn_bt)
            .ok()?;
        let pipeline_nn = device
            .new_compute_pipeline_state_with_function(&fn_nn)
            .ok()?;

        Some(MetalState {
            device,
            queue,
            pipeline_bt,
            pipeline_nn,
        })
    }

    fn get_metal() -> Option<&'static MetalState> {
        METAL.get_or_init(init_metal).as_ref()
    }

    fn run_gemm(
        state: &MetalState,
        pipeline: &ComputePipelineState,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
    ) {
        let a_bytes = std::mem::size_of_val(a) as u64;
        let b_bytes = std::mem::size_of_val(b) as u64;
        let c_bytes = (m as u64) * (n as u64) * 4;

        let buf_a = state.device.new_buffer_with_bytes_no_copy(
            a.as_ptr() as *const _,
            a_bytes,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let buf_b = state.device.new_buffer_with_bytes_no_copy(
            b.as_ptr() as *const _,
            b_bytes,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let buf_c = state.device.new_buffer_with_bytes_no_copy(
            c.as_mut_ptr() as *mut _ as *const _,
            c_bytes,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        let cmd = state.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        enc.set_bytes(3, 4, &m as *const u32 as *const _);
        enc.set_bytes(4, 4, &n as *const u32 as *const _);
        enc.set_bytes(5, 4, &k as *const u32 as *const _);

        let tile = 16u64;
        let grid = MTLSize::new(
            (n as u64).div_ceil(tile) * tile,
            (m as u64).div_ceil(tile) * tile,
            1,
        );
        let tg = MTLSize::new(tile, tile, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();
    }

    /// **Unstable**: Metal matmul B^T; Metal pipeline and dispatch threshold may change.
    ///
    /// GPU-accelerated C = A @ B^T. Returns true if GPU was used.
    pub fn metal_matmul_bt(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4,
        // held finding: this standalone Metal wrapper previously had NO argument validation
        // at all). Validated before any GPU buffer is created from these slices below.
        validate_gemm_bt(a.len(), b.len(), c.len(), m, k, n, "metal_matmul_bt");

        let work = (m as u64) * (n as u64) * (k as u64);
        if work < GPU_DISPATCH_THRESHOLD {
            return false;
        }
        let Some(state) = get_metal() else {
            return false;
        };
        run_gemm(
            state,
            &state.pipeline_bt,
            a,
            b,
            c,
            m as u32,
            n as u32,
            k as u32,
        );
        true
    }

    /// Report whether Metal GPU is available and initialized.
    #[allow(dead_code)] // public API for callers to probe Metal availability before dispatch
    pub fn is_available() -> bool {
        get_metal().is_some()
    }

    /// **Unstable**: Metal matmul; Metal pipeline and dispatch threshold may change.
    ///
    /// GPU-accelerated C = A @ B. Returns true if GPU was used.
    pub fn metal_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> bool {
        // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4,
        // held finding — see `metal_matmul_bt`).
        validate_gemm_nn(a.len(), b.len(), c.len(), m, k, n, "metal_matmul");

        let work = (m as u64) * (n as u64) * (k as u64);
        if work < GPU_DISPATCH_THRESHOLD {
            return false;
        }
        let Some(state) = get_metal() else {
            return false;
        };
        run_gemm(
            state,
            &state.pipeline_nn,
            a,
            b,
            c,
            m as u32,
            n as u32,
            k as u32,
        );
        true
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use gpu::{metal_matmul, metal_matmul_bt};

// Stubs for non-macOS or non-metal builds.
/// **Unstable**: Metal matmul B^T stub; returns false when metal-gpu feature is disabled.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub fn metal_matmul_bt(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
    _m: usize,
    _k: usize,
    _n: usize,
) -> bool {
    false
}

/// **Unstable**: Metal matmul stub; returns false when metal-gpu feature is disabled.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub fn metal_matmul(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
    _m: usize,
    _k: usize,
    _n: usize,
) -> bool {
    false
}

// --- release-active argument validation (ADR-080 C4 held finding) ---
// These panic before any Metal device/buffer is touched, so they run without a GPU present:
// `validate_gemm_bt`/`validate_gemm_nn` are called unconditionally at the top of
// `metal_matmul_bt`/`metal_matmul`, ahead of the GPU-availability check.
#[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
mod tests {
    use super::{metal_matmul, metal_matmul_bt};

    #[test]
    #[should_panic(expected = "b too short for n*k")]
    fn metal_matmul_bt_rejects_short_b() {
        let a = [0.0f32; 2];
        let b = [0.0f32; 1]; // needs n*k = 2
        let mut c = [0.0f32; 1];
        metal_matmul_bt(&a, &b, &mut c, 1, 1, 2);
    }

    #[test]
    #[should_panic(expected = "shape overflow: n*k")]
    fn metal_matmul_bt_rejects_overflow() {
        let a = [0.0f32; 2];
        let b = [0.0f32; 2];
        let mut c = [0.0f32; 2];
        // m*k = 2*2 = 4 (no overflow); n*k = usize::MAX*2 overflows.
        metal_matmul_bt(&a, &b, &mut c, 2, 2, usize::MAX);
    }

    #[test]
    #[should_panic(expected = "b too short for k*n")]
    fn metal_matmul_rejects_short_b() {
        let a = [0.0f32; 2];
        let b = [0.0f32; 1]; // needs k*n = 2
        let mut c = [0.0f32; 1];
        metal_matmul(&a, &b, &mut c, 1, 1, 2);
    }
}
