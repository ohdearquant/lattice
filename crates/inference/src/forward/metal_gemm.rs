//! Metal GPU-accelerated GEMM for Apple Silicon.
//!
//! Dispatches matrix multiplication to the GPU via Metal compute shaders.
//! Falls back to Accelerate (AMX) when matrices are too small for GPU overhead
//! to be worthwhile.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod gpu {
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

    const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Tiled GEMM: C[M,N] = A[M,K] @ B[N,K]^T
// B is stored row-major as [N,K], accessed as transposed.
constant uint TILE = 16;

kernel void gemm_bt(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    threadgroup float tA[TILE][TILE];
    threadgroup float tB[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;
    float sum = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint ak = t * TILE + lid.x;
        uint bk = t * TILE + lid.y;

        tA[lid.y][lid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        tB[lid.y][lid.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            sum += tA[lid.y][k] * tB[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Non-transposed: C[M,N] = A[M,K] @ B[K,N]
kernel void gemm_nn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    threadgroup float tA[TILE][TILE];
    threadgroup float tB[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;
    float sum = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint ak = t * TILE + lid.x;
        uint bk = t * TILE + lid.y;

        tA[lid.y][lid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        tB[lid.y][lid.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            sum += tA[lid.y][k] * tB[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"#;

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
    pub fn is_available() -> bool {
        get_metal().is_some()
    }

    /// **Unstable**: Metal matmul; Metal pipeline and dispatch threshold may change.
    ///
    /// GPU-accelerated C = A @ B. Returns true if GPU was used.
    pub fn metal_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> bool {
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
