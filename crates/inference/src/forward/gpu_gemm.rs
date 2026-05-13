//! WebGPU-accelerated GEMM via `wgpu`.
//!
//! Key design: model weights are uploaded to GPU **once** at load time and stay
//! resident. Only small input/output tensors transfer per inference call.
//! Activation buffers are recycled across layers.

#[cfg(feature = "wgpu-gpu")]
mod inner {
    use std::sync::OnceLock;
    use wgpu::util::DeviceExt;

    /// Minimum work (M*N*K) to justify GPU dispatch over CPU Accelerate.
    const GPU_THRESHOLD: u64 = 128 * 128 * 128;

    const GEMM_SHADER: &str = r#"
// Tiled GEMM: C[M,N] = A[M,K] @ B^T[N,K]
// B is stored row-major [N,K], transposed in the multiply.

struct Dims { M: u32, N: u32, K: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn gemm_bt(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let ty = lid.y;
    let tx = lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        let a_col = t * TILE + tx;
        let b_col = t * TILE + ty;

        if (row < dims.M && a_col < dims.K) {
            tileA[ty][tx] = A[row * dims.K + a_col];
        } else {
            tileA[ty][tx] = 0.0;
        }

        // B is [N,K] row-major. For B^T multiply, we read B[col, b_col].
        if (col < dims.N && b_col < dims.K) {
            tileB[ty][tx] = B[col * dims.K + b_col];
        } else {
            tileB[ty][tx] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE; k++) {
            acc += tileA[ty][k] * tileB[k][tx];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}

// Non-transposed: C[M,N] = A[M,K] @ B[K,N]
@compute @workgroup_size(16, 16)
fn gemm_nn(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let ty = lid.y;
    let tx = lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        let a_col = t * TILE + tx;
        let b_row = t * TILE + ty;

        if (row < dims.M && a_col < dims.K) {
            tileA[ty][tx] = A[row * dims.K + a_col];
        } else {
            tileA[ty][tx] = 0.0;
        }

        if (b_row < dims.K && col < dims.N) {
            tileB[ty][tx] = B[b_row * dims.N + col];
        } else {
            tileB[ty][tx] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE; k++) {
            acc += tileA[ty][k] * tileB[k][tx];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}
"#;

    struct GpuState {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline_bt: wgpu::ComputePipeline,
        pipeline_nn: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
    }

    static GPU: OnceLock<Option<GpuState>> = OnceLock::new();

    fn init_gpu() -> Option<GpuState> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let info = adapter.get_info();
        tracing::info!(
            name = info.name,
            backend = ?info.backend,
            "wgpu GPU initialized for GEMM"
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("lattice-inference"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1 << 30, // 1GB
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm"),
            source: wgpu::ShaderSource::Wgsl(GEMM_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm_layout"),
            entries: &[
                bgl_entry(0, true),  // A (read)
                bgl_entry(1, true),  // B (read)
                bgl_entry(2, false), // C (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemm_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_bt = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemm_bt"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gemm_bt"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_nn = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemm_nn"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gemm_nn"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(GpuState {
            device,
            queue,
            pipeline_bt,
            pipeline_nn,
            bind_group_layout,
        })
    }

    fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn get_gpu() -> Option<&'static GpuState> {
        GPU.get_or_init(|| init_gpu()).as_ref()
    }

    fn run_gemm(
        gpu: &GpuState,
        pipeline: &wgpu::ComputePipeline,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
    ) {
        let a_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A"),
                contents: as_bytes(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("B"),
                contents: as_bytes(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_size = (m as u64) * (n as u64) * 4;
        let c_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C"),
            size: c_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims = [m, n, k, 0u32];
        let dims_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: as_bytes(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: c_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm_bg"),
            layout: &gpu.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n + 15) / 16;
            let wg_y = (m + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &readback, 0, c_size);
        gpu.queue.submit(Some(encoder.finish()));

        // Read back result.
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let floats: &[f32] = as_f32_slice(&data);
        c[..floats.len()].copy_from_slice(floats);
    }

    fn as_bytes<T>(data: &[T]) -> &[u8] {
        // SAFETY: A shared slice of `T` may be viewed as bytes for upload; the
        // byte length is exactly element count times element size, and `u8` has
        // alignment 1 so every `T` allocation is valid for this view.
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        }
    }

    fn as_f32_slice(data: &[u8]) -> &[f32] {
        assert!(data.len() % 4 == 0);
        debug_assert_eq!(data.as_ptr().align_offset(std::mem::align_of::<f32>()), 0);
        // SAFETY: wgpu readback buffers for f32 output are sized to a multiple
        // of 4 bytes and mapped storage is expected to be aligned for f32 reads.
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) }
    }

    /// **Unstable**: GPU matmul B^T via wgpu; wgpu pipeline and threshold logic may change.
    ///
    /// GPU C = A @ B^T. Returns true if dispatched to GPU.
    pub fn gpu_matmul_bt(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        if (m as u64) * (n as u64) * (k as u64) < GPU_THRESHOLD {
            return false;
        }
        let Some(gpu) = get_gpu() else { return false };
        run_gemm(gpu, &gpu.pipeline_bt, a, b, c, m as u32, n as u32, k as u32);
        true
    }

    /// **Unstable**: GPU matmul via wgpu; wgpu pipeline and threshold logic may change.
    ///
    /// GPU C = A @ B. Returns true if dispatched to GPU.
    pub fn gpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> bool {
        if (m as u64) * (n as u64) * (k as u64) < GPU_THRESHOLD {
            return false;
        }
        let Some(gpu) = get_gpu() else { return false };
        run_gemm(gpu, &gpu.pipeline_nn, a, b, c, m as u32, n as u32, k as u32);
        true
    }
}

#[cfg(feature = "wgpu-gpu")]
pub use inner::{gpu_matmul, gpu_matmul_bt};

/// **Unstable**: GPU matmul B^T stub; returns false when wgpu-gpu feature is disabled.
#[cfg(not(feature = "wgpu-gpu"))]
pub fn gpu_matmul_bt(_: &[f32], _: &[f32], _: &mut [f32], _: usize, _: usize, _: usize) -> bool {
    false
}
/// **Unstable**: GPU matmul stub; returns false when wgpu-gpu feature is disabled.
#[cfg(not(feature = "wgpu-gpu"))]
pub fn gpu_matmul(_: &[f32], _: &[f32], _: &mut [f32], _: usize, _: usize, _: usize) -> bool {
    false
}
