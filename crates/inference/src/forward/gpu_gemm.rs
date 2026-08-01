//! WebGPU-accelerated GEMM via `wgpu`.
//!
//! Key design: model weights are uploaded to GPU **once** at load time and stay
//! resident. Only small input/output tensors transfer per inference call.
//! Activation buffers are recycled across layers.

#[cfg(feature = "wgpu-gpu")]
mod inner {
    use crate::forward::cpu::{validate_gemm_bt, validate_gemm_nn};
    use std::sync::OnceLock;
    use wgpu::util::DeviceExt;

    /// Minimum work (M*N*K) to justify GPU dispatch over CPU Accelerate.
    const GPU_THRESHOLD: u64 = 128 * 128 * 128;

    const GEMM_SHADER: &str = concat!("\n", include_str!("wgsl/gemm.wgsl"));

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
        // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4,
        // held finding: this standalone wgpu wrapper previously had NO argument validation
        // at all). Validated before any GPU buffer is created from these slices below.
        validate_gemm_bt(a.len(), b.len(), c.len(), m, k, n, "gpu_matmul_bt");

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
        // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4,
        // held finding — see `gpu_matmul_bt`).
        validate_gemm_nn(a.len(), b.len(), c.len(), m, k, n, "gpu_matmul");

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

// --- release-active argument validation (ADR-080 C4 held finding) ---
// These panic before any wgpu device/buffer is touched (validated at the top of
// `gpu_matmul_bt`/`gpu_matmul`, ahead of the GPU-availability check), so they run without a
// GPU/adapter present.
#[cfg(all(test, feature = "wgpu-gpu"))]
mod tests {
    use super::{gpu_matmul, gpu_matmul_bt};

    #[test]
    #[should_panic(expected = "b too short for n*k")]
    fn gpu_matmul_bt_rejects_short_b() {
        let a = [0.0f32; 2];
        let b = [0.0f32; 1]; // needs n*k = 2
        let mut c = [0.0f32; 1];
        gpu_matmul_bt(&a, &b, &mut c, 1, 1, 2);
    }

    #[test]
    #[should_panic(expected = "shape overflow: n*k")]
    fn gpu_matmul_bt_rejects_overflow() {
        let a = [0.0f32; 2];
        let b = [0.0f32; 2];
        let mut c = [0.0f32; 2];
        // m*k = 2*2 = 4 (no overflow); n*k = usize::MAX*2 overflows.
        gpu_matmul_bt(&a, &b, &mut c, 2, 2, usize::MAX);
    }

    #[test]
    #[should_panic(expected = "b too short for k*n")]
    fn gpu_matmul_rejects_short_b() {
        let a = [0.0f32; 2];
        let b = [0.0f32; 1]; // needs k*n = 2
        let mut c = [0.0f32; 1];
        gpu_matmul(&a, &b, &mut c, 1, 1, 2);
    }
}
