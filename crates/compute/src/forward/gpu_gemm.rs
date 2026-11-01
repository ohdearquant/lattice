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
        GPU.get_or_init(init_gpu).as_ref()
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum BufferRole {
        A,
        B,
        Output,
        Parameters,
    }

    #[derive(Debug, PartialEq, Eq)]
    enum DispatchRefusal {
        DimensionM,
        DimensionN,
        DimensionK,
        SizeOverflow(BufferRole),
        BufferSize(BufferRole),
        StorageBindingSize(BufferRole),
        UniformBindingSize,
        WorkgroupCount,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct GemmDimensions {
        m: u32,
        n: u32,
        k: u32,
    }

    impl GemmDimensions {
        fn new([m, n, k]: [usize; 3]) -> Result<Self, DispatchRefusal> {
            Ok(Self {
                m: u32::try_from(m).map_err(|_| DispatchRefusal::DimensionM)?,
                n: u32::try_from(n).map_err(|_| DispatchRefusal::DimensionN)?,
                k: u32::try_from(k).map_err(|_| DispatchRefusal::DimensionK)?,
            })
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct GemmLayout {
        dims: [u32; 4],
        c_size: u64,
        c_len: usize,
    }

    impl GemmLayout {
        fn new(
            dimensions: GemmDimensions,
            a_len: usize,
            b_len: usize,
            limits: &wgpu::Limits,
        ) -> Result<Self, DispatchRefusal> {
            let bytes = |len: usize, role| {
                u64::try_from(len)
                    .ok()
                    .and_then(|len| len.checked_mul(4))
                    .ok_or(DispatchRefusal::SizeOverflow(role))
            };
            let c_size = u64::from(dimensions.m)
                .checked_mul(u64::from(dimensions.n))
                .and_then(|len| len.checked_mul(4))
                .ok_or(DispatchRefusal::SizeOverflow(BufferRole::Output))?;
            let c_len = usize::try_from(c_size / 4)
                .map_err(|_| DispatchRefusal::SizeOverflow(BufferRole::Output))?;
            for (role, size) in [
                (BufferRole::A, bytes(a_len, BufferRole::A)?),
                (BufferRole::B, bytes(b_len, BufferRole::B)?),
                (BufferRole::Output, c_size),
                (BufferRole::Parameters, 16),
            ] {
                if size > limits.max_buffer_size {
                    return Err(DispatchRefusal::BufferSize(role));
                }
                if role == BufferRole::Parameters {
                    if size > u64::from(limits.max_uniform_buffer_binding_size) {
                        return Err(DispatchRefusal::UniformBindingSize);
                    }
                } else if size > u64::from(limits.max_storage_buffer_binding_size) {
                    return Err(DispatchRefusal::StorageBindingSize(role));
                }
            }
            if dimensions.n.div_ceil(16) > limits.max_compute_workgroups_per_dimension
                || dimensions.m.div_ceil(16) > limits.max_compute_workgroups_per_dimension
            {
                return Err(DispatchRefusal::WorkgroupCount);
            }
            Ok(Self {
                dims: [dimensions.m, dimensions.n, dimensions.k, 0],
                c_size,
                c_len,
            })
        }
    }

    trait GemmBackend {
        fn limits(&self) -> wgpu::Limits;
        fn dispatch(&self, a: &[f32], b: &[f32], c: &mut [f32], layout: GemmLayout) -> bool;
    }

    fn run_gemm<B: GemmBackend>(
        backend: impl FnOnce() -> Option<B>,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        dimensions: [usize; 3],
    ) -> bool {
        let Ok(dimensions) = GemmDimensions::new(dimensions) else {
            return false;
        };
        let work = u64::from(dimensions.m)
            .saturating_mul(u64::from(dimensions.n))
            .saturating_mul(u64::from(dimensions.k));
        if work < GPU_THRESHOLD {
            return false;
        }
        let Some(backend) = backend() else {
            return false;
        };
        let Ok(layout) = GemmLayout::new(dimensions, a.len(), b.len(), &backend.limits()) else {
            return false;
        };
        backend.dispatch(a, b, &mut c[..layout.c_len], layout)
    }

    struct GpuBackend<'a> {
        gpu: &'a GpuState,
        pipeline: &'a wgpu::ComputePipeline,
    }

    impl GemmBackend for GpuBackend<'_> {
        fn limits(&self) -> wgpu::Limits {
            self.gpu.device.limits()
        }

        fn dispatch(&self, a: &[f32], b: &[f32], c: &mut [f32], layout: GemmLayout) -> bool {
            let a_buf = self
                .gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("A"),
                    contents: as_bytes(a),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let b_buf = self
                .gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("B"),
                    contents: as_bytes(b),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let c_size = layout.c_size;
            let c_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("C"),
                size: c_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let dims = layout.dims;
            let dims_buf = self
                .gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("dims"),
                    contents: as_bytes(&dims),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let readback = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: c_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let bind_group = self
                .gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gemm_bg"),
                    layout: &self.gpu.bind_group_layout,
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

            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let wg_x = layout.dims[1].div_ceil(16);
                let wg_y = layout.dims[0].div_ceil(16);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            encoder.copy_buffer_to_buffer(&c_buf, 0, &readback, 0, c_size);
            self.gpu.queue.submit(Some(encoder.finish()));

            complete_readback(
                &GpuReadback {
                    device: &self.gpu.device,
                    buffer: &readback,
                },
                c,
            )
        }
    }

    trait GemmReadback {
        fn request_map(&self, sender: std::sync::mpsc::Sender<Result<(), wgpu::BufferAsyncError>>);
        fn poll(&self) -> wgpu::MaintainResult;
        fn with_mapped_bytes(&self, consume: impl FnOnce(&[u8]) -> bool) -> bool;
        fn unmap(&self);
    }

    struct GpuReadback<'a> {
        device: &'a wgpu::Device,
        buffer: &'a wgpu::Buffer,
    }

    impl GemmReadback for GpuReadback<'_> {
        fn request_map(&self, sender: std::sync::mpsc::Sender<Result<(), wgpu::BufferAsyncError>>) {
            self.buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
        }

        fn poll(&self) -> wgpu::MaintainResult {
            self.device.poll(wgpu::Maintain::Wait)
        }

        fn with_mapped_bytes(&self, consume: impl FnOnce(&[u8]) -> bool) -> bool {
            let data = self.buffer.slice(..).get_mapped_range();
            let copied = consume(&data);
            drop(data);
            copied
        }

        fn unmap(&self) {
            self.buffer.unmap();
        }
    }

    fn complete_readback(readback: &impl GemmReadback, c: &mut [f32]) -> bool {
        let (sender, receiver) = std::sync::mpsc::channel();
        readback.request_map(sender);
        // Queue status cannot certify mapping; only a successful callback can.
        let mapped = match readback.poll() {
            wgpu::MaintainResult::SubmissionQueueEmpty | wgpu::MaintainResult::Ok => {
                matches!(receiver.try_recv(), Ok(Ok(())))
            }
        };
        if !mapped {
            readback.unmap();
            return false;
        }
        let copied = readback.with_mapped_bytes(|data| copy_readback(data, c));
        readback.unmap();
        copied
    }

    fn as_bytes<T>(data: &[T]) -> &[u8] {
        // SAFETY: A shared slice of `T` may be viewed as bytes for upload; the
        // byte length is exactly element count times element size, and `u8` has
        // alignment 1 so every `T` allocation is valid for this view.
        unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        }
    }

    fn copy_readback(data: &[u8], output: &mut [f32]) -> bool {
        if data.len() != std::mem::size_of_val(output) {
            return false;
        }
        for (value, bytes) in output.iter_mut().zip(data.chunks_exact(4)) {
            *value = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        true
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

        run_gemm(
            || {
                get_gpu().map(|gpu| GpuBackend {
                    gpu,
                    pipeline: &gpu.pipeline_bt,
                })
            },
            a,
            b,
            c,
            [m, n, k],
        )
    }

    /// **Unstable**: GPU matmul via wgpu; wgpu pipeline and threshold logic may change.
    ///
    /// GPU C = A @ B. Returns true if dispatched to GPU.
    pub fn gpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> bool {
        // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4,
        // held finding — see `gpu_matmul_bt`).
        validate_gemm_nn(a.len(), b.len(), c.len(), m, k, n, "gpu_matmul");

        run_gemm(
            || {
                get_gpu().map(|gpu| GpuBackend {
                    gpu,
                    pipeline: &gpu.pipeline_nn,
                })
            },
            a,
            b,
            c,
            [m, n, k],
        )
    }

    #[cfg(test)]
    mod host_tests {
        use super::*;
        use std::cell::{Cell, RefCell};
        use std::sync::mpsc::Sender;

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Mapping {
            Success,
            Error,
            Pending,
            Disconnected,
        }

        struct TestReadback {
            storage: Vec<u8>,
            offset: usize,
            len: usize,
            mapping: Mapping,
            queue_empty: bool,
            pending: RefCell<Option<Sender<Result<(), wgpu::BufferAsyncError>>>>,
            trace: RefCell<Vec<&'static str>>,
        }

        impl TestReadback {
            fn new(values: &[f32], mapping: Mapping, queue_empty: bool, misaligned: bool) -> Self {
                let len = std::mem::size_of_val(values);
                let alignment = std::mem::align_of::<f32>();
                let mut storage = vec![0; len + alignment];
                let offset = (0..alignment)
                    .find(|offset| {
                        let aligned =
                            (storage.as_ptr() as usize + offset).is_multiple_of(alignment);
                        aligned != misaligned
                    })
                    .unwrap();
                for (bytes, value) in storage[offset..offset + len]
                    .chunks_exact_mut(4)
                    .zip(values)
                {
                    bytes.copy_from_slice(&value.to_ne_bytes());
                }
                Self {
                    storage,
                    offset,
                    len,
                    mapping,
                    queue_empty,
                    pending: RefCell::new(None),
                    trace: RefCell::new(Vec::new()),
                }
            }
        }

        impl GemmReadback for TestReadback {
            fn request_map(&self, sender: Sender<Result<(), wgpu::BufferAsyncError>>) {
                self.trace.borrow_mut().push("map");
                match self.mapping {
                    Mapping::Success => sender.send(Ok(())).unwrap(),
                    Mapping::Error => sender.send(Err(wgpu::BufferAsyncError)).unwrap(),
                    Mapping::Pending => *self.pending.borrow_mut() = Some(sender),
                    Mapping::Disconnected => drop(sender),
                }
            }

            fn poll(&self) -> wgpu::MaintainResult {
                self.trace.borrow_mut().push("poll");
                if self.queue_empty {
                    wgpu::MaintainResult::SubmissionQueueEmpty
                } else {
                    wgpu::MaintainResult::Ok
                }
            }

            fn with_mapped_bytes(&self, consume: impl FnOnce(&[u8]) -> bool) -> bool {
                assert_eq!(self.mapping, Mapping::Success, "unmapped access trap");
                self.trace.borrow_mut().push("view");
                let copied = consume(&self.storage[self.offset..self.offset + self.len]);
                self.trace.borrow_mut().push("drop view");
                copied
            }

            fn unmap(&self) {
                self.trace.borrow_mut().push("unmap");
                self.pending.borrow_mut().take();
            }
        }

        struct TestBackend<'a> {
            limits: wgpu::Limits,
            allocations: &'a Cell<usize>,
            readback: &'a TestReadback,
        }

        impl GemmBackend for TestBackend<'_> {
            fn limits(&self) -> wgpu::Limits {
                self.limits.clone()
            }

            fn dispatch(&self, _: &[f32], _: &[f32], c: &mut [f32], layout: GemmLayout) -> bool {
                assert_eq!(layout.c_size, std::mem::size_of_val(c) as u64);
                self.allocations.set(self.allocations.get() + 5);
                complete_readback(self.readback, c)
            }
        }

        fn generous_limits() -> wgpu::Limits {
            wgpu::Limits {
                max_buffer_size: u64::MAX,
                max_storage_buffer_binding_size: u32::MAX,
                max_uniform_buffer_binding_size: u32::MAX,
                max_compute_workgroups_per_dimension: u32::MAX,
                ..wgpu::Limits::default()
            }
        }

        fn run_finite(
            dimensions: [usize; 3],
            extra: [usize; 2],
            limits: wgpu::Limits,
            readback: &TestReadback,
            output: &mut [f32],
            allocations: &Cell<usize>,
        ) -> bool {
            let [m, n, k] = dimensions;
            assert!(dimensions.into_iter().all(|dimension| dimension <= 512));
            let a = vec![0.0; m * k + extra[0]];
            let b = vec![0.0; n * k + extra[1]];
            run_gemm(
                || {
                    Some(TestBackend {
                        limits,
                        allocations,
                        readback,
                    })
                },
                &a,
                &b,
                output,
                dimensions,
            )
        }

        #[test]
        fn readback_copies_misaligned_and_aligned_native_float_bits() {
            let values = [
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc1_2345),
                f32::from_bits(0x3fa0_0000),
            ];
            for misaligned in [false, true] {
                let readback = TestReadback::new(&values, Mapping::Success, true, misaligned);
                let bytes = &readback.storage[readback.offset..readback.offset + readback.len];
                assert_eq!(
                    !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()),
                    misaligned
                );
                let mut output = [0.0; 3];
                assert!(complete_readback(&readback, &mut output));
                assert_eq!(output.map(f32::to_bits), values.map(f32::to_bits));
                assert_eq!(
                    *readback.trace.borrow(),
                    ["map", "poll", "view", "drop view", "unmap"]
                );
            }
        }

        #[test]
        fn readback_rejects_wrong_length_before_writing_output() {
            let readback = TestReadback::new(&[1.25], Mapping::Success, true, true);
            let mut output = [9.0; 2];
            assert!(!complete_readback(&readback, &mut output));
            assert_eq!(output, [9.0; 2]);
            assert_eq!(
                *readback.trace.borrow(),
                ["map", "poll", "view", "drop view", "unmap"]
            );
        }

        #[test]
        fn runner_requires_map_success_for_both_poll_results() {
            let count = 128 * 128;
            let values = vec![f32::from_bits(0x8000_0000); count];
            for queue_empty in [false, true] {
                for mapping in [
                    Mapping::Success,
                    Mapping::Error,
                    Mapping::Pending,
                    Mapping::Disconnected,
                ] {
                    let readback = TestReadback::new(&values, mapping, queue_empty, true);
                    let allocations = Cell::new(0);
                    let mut output = vec![9.0; count + 1];
                    let dispatched = run_finite(
                        [128, 128, 128],
                        [0, 0],
                        generous_limits(),
                        &readback,
                        &mut output,
                        &allocations,
                    );
                    assert_eq!(dispatched, mapping == Mapping::Success);
                    assert_eq!(allocations.get(), 5);
                    if mapping == Mapping::Success {
                        assert!(
                            output[..count]
                                .iter()
                                .all(|value| value.to_bits() == 0x8000_0000)
                        );
                        assert_eq!(
                            *readback.trace.borrow(),
                            ["map", "poll", "view", "drop view", "unmap"]
                        );
                    } else {
                        assert_eq!(output, vec![9.0; count + 1]);
                        assert_eq!(*readback.trace.borrow(), ["map", "poll", "unmap"]);
                    }
                    assert_eq!(output[count], 9.0);
                    assert!(readback.pending.borrow().is_none());
                }
            }
        }

        #[test]
        fn runner_refuses_all_operand_limits_before_allocation() {
            for (dimensions, extra, role) in [
                ([128, 128, 128], [1, 0], BufferRole::A),
                ([128, 128, 128], [0, 1], BufferRole::B),
                ([256, 256, 128], [0, 0], BufferRole::Output),
            ] {
                let [m, n, k] = dimensions;
                let largest = ((m * k + extra[0]).max(n * k + extra[1]).max(m * n) * 4) as u32;
                let readback = TestReadback::new(&vec![1.25; m * n], Mapping::Success, true, false);
                for storage_limit in [false, true] {
                    let mut limits = generous_limits();
                    limits.max_buffer_size = u64::from(largest);
                    limits.max_storage_buffer_binding_size = largest;
                    if storage_limit {
                        limits.max_storage_buffer_binding_size -= 4;
                    } else {
                        limits.max_buffer_size -= 4;
                    }
                    let expected = if storage_limit {
                        DispatchRefusal::StorageBindingSize(role)
                    } else {
                        DispatchRefusal::BufferSize(role)
                    };
                    assert_eq!(
                        GemmLayout::new(
                            GemmDimensions::new(dimensions).unwrap(),
                            m * k + extra[0],
                            n * k + extra[1],
                            &limits
                        ),
                        Err(expected)
                    );
                    let allocations = Cell::new(0);
                    let mut output = vec![9.0; m * n];
                    assert!(!run_finite(
                        dimensions,
                        extra,
                        limits.clone(),
                        &readback,
                        &mut output,
                        &allocations
                    ));
                    assert_eq!(allocations.get(), 0);
                    assert!(readback.trace.borrow().is_empty());
                    assert_eq!(output, vec![9.0; m * n]);
                    limits.max_buffer_size = u64::from(largest);
                    limits.max_storage_buffer_binding_size = largest;
                    assert!(run_finite(
                        dimensions,
                        extra,
                        limits,
                        &readback,
                        &mut output,
                        &allocations
                    ));
                    assert_eq!(allocations.get(), 5);
                    assert_eq!(output, vec![1.25; m * n]);
                    readback.trace.borrow_mut().clear();
                }
            }
        }

        #[test]
        fn layout_checks_parameters_and_byte_size_overflow() {
            let dimensions = GemmDimensions::new([1, 1, 1]).unwrap();
            let mut limits = generous_limits();
            limits.max_buffer_size = 15;
            assert_eq!(
                GemmLayout::new(dimensions, 1, 1, &limits),
                Err(DispatchRefusal::BufferSize(BufferRole::Parameters))
            );
            limits.max_buffer_size = 16;
            limits.max_uniform_buffer_binding_size = 15;
            assert_eq!(
                GemmLayout::new(dimensions, 1, 1, &limits),
                Err(DispatchRefusal::UniformBindingSize)
            );
            limits.max_uniform_buffer_binding_size = 16;
            assert!(GemmLayout::new(dimensions, 1, 1, &limits).is_ok());
            #[cfg(target_pointer_width = "64")]
            {
                for (a_len, b_len, role) in [
                    (usize::MAX, 1, BufferRole::A),
                    (1, usize::MAX, BufferRole::B),
                ] {
                    assert_eq!(
                        GemmLayout::new(dimensions, a_len, b_len, &generous_limits()),
                        Err(DispatchRefusal::SizeOverflow(role))
                    );
                }
                let dimensions =
                    GemmDimensions::new([u32::MAX as usize, u32::MAX as usize, 1]).unwrap();
                assert_eq!(
                    GemmLayout::new(dimensions, 1, 1, &generous_limits()),
                    Err(DispatchRefusal::SizeOverflow(BufferRole::Output))
                );
            }
        }

        #[test]
        fn runner_checks_grid_limit_before_allocation() {
            let readback = TestReadback::new(&vec![1.25; 128 * 128], Mapping::Success, true, false);
            let allocations = Cell::new(0);
            let mut output = vec![9.0; 128 * 128];
            let mut limits = generous_limits();
            limits.max_compute_workgroups_per_dimension = 7;
            assert_eq!(
                GemmLayout::new(
                    GemmDimensions::new([128, 128, 128]).unwrap(),
                    128 * 128,
                    128 * 128,
                    &limits
                ),
                Err(DispatchRefusal::WorkgroupCount)
            );
            assert!(!run_finite(
                [128, 128, 128],
                [0, 0],
                limits.clone(),
                &readback,
                &mut output,
                &allocations
            ));
            assert_eq!(allocations.get(), 0);
            assert_eq!(output, vec![9.0; 128 * 128]);
            limits.max_compute_workgroups_per_dimension = 8;
            assert!(run_finite(
                [128, 128, 128],
                [0, 0],
                limits,
                &readback,
                &mut output,
                &allocations
            ));
            assert_eq!(allocations.get(), 5);
        }

        #[test]
        #[cfg(target_pointer_width = "64")]
        fn runner_checks_each_dimension_before_backend_access() {
            let oversized = u32::MAX as usize + 1;
            for (axis, expected) in [
                (0, DispatchRefusal::DimensionM),
                (1, DispatchRefusal::DimensionN),
                (2, DispatchRefusal::DimensionK),
            ] {
                let mut dimensions = [128; 3];
                dimensions[axis] = oversized;
                assert_eq!(GemmDimensions::new(dimensions), Err(expected));
                let mut output = [9.0];
                assert!(!run_gemm(
                    || -> Option<TestBackend<'_>> {
                        panic!("dimension guard must precede backend access")
                    },
                    &[],
                    &[],
                    &mut output,
                    dimensions
                ));
                assert_eq!(output, [9.0]);
                dimensions[axis] = u32::MAX as usize;
                assert!(GemmDimensions::new(dimensions).is_ok());
                let reached_backend = Cell::new(false);
                assert!(!run_gemm(
                    || -> Option<TestBackend<'_>> {
                        reached_backend.set(true);
                        None
                    },
                    &[],
                    &[],
                    &mut output,
                    dimensions
                ));
                assert!(reached_backend.get());
            }
            assert_eq!(
                GemmDimensions::new([128; 3]).unwrap(),
                GemmDimensions {
                    m: 128,
                    n: 128,
                    k: 128
                }
            );
        }
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
