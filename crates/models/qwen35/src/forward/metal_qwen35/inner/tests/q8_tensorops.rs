use super::checkpoint_refusals::make_q8_weight_bytes;
use super::gpu_test_lock;
use super::rms_reduce_854_parity::{f32_buf, read_f32};
use super::{MSL_Q8_TILED_SOURCE, cpu_matmul_ref, max_abs_diff};
use metal::{
    CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus, MTLLanguageVersion,
    MTLSize,
};

const GUARD: f32 = 12345.5;
const PREFIX: usize = 16;
const CPU_TOLERANCE: f32 = 0.015;
const TILED_TOLERANCE: f32 = 0.012;

fn pipeline(
    device: &Device,
    source: &str,
    kernel: &str,
    options: &CompileOptions,
) -> ComputePipelineState {
    let library = device
        .new_library_with_source(source, options)
        .unwrap_or_else(|error| panic!("{kernel} compilation failed: {error}"));
    let function = library
        .get_function(kernel, None)
        .expect("probe entry point");
    device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|error| panic!("{kernel} pipeline failed: {error}"))
}

fn guarded_output(m: usize, n: usize, stride: usize) -> Vec<f32> {
    let mut storage = vec![GUARD; PREFIX + m * stride + PREFIX];
    for row in 0..m {
        storage[PREFIX + row * stride..PREFIX + row * stride + n].fill(f32::NAN);
    }
    storage
}

fn logical_output(storage: &[f32], m: usize, n: usize, stride: usize) -> Vec<f32> {
    assert_eq!(storage.len(), PREFIX + m * stride + PREFIX);
    let mut logical = Vec::with_capacity(m * n);
    for (index, &value) in storage.iter().enumerate() {
        let in_matrix = (PREFIX..PREFIX + m * stride).contains(&index);
        if in_matrix && (index - PREFIX) % stride < n {
            logical.push(value);
        }
    }
    assert_eq!(logical.len(), m * n);
    logical
}

fn assert_output_guards(storage: &[f32], m: usize, n: usize, stride: usize) {
    for (index, &value) in storage.iter().enumerate() {
        let in_matrix = (PREFIX..PREFIX + m * stride).contains(&index);
        if !in_matrix || (index - PREFIX) % stride >= n {
            assert_eq!(value.to_bits(), GUARD.to_bits(), "output guard at {index}");
        }
    }
}

fn discrepancies(actual: &[f32], reference: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), reference.len());
    let absolute = max_abs_diff(actual, reference);
    let mut relative = 0.0f32;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let difference = (actual - reference).abs() / reference.abs().max(1e-6);
        if !difference.is_finite() {
            return (absolute, difference);
        }
        relative = relative.max(difference);
    }
    (absolute, relative)
}

#[test]
#[ignore = "requires a Metal 4 TensorOps device and runtime compiler"]
fn gemm_q8_tensorops_model_free_probe() {
    let _gpu_guard = gpu_test_lock();
    let device = Device::system_default().expect("Metal device required for TensorOps probe");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    // The binding's language enum predates Metal 4; the shader checks the selected version.
    let tensorops = pipeline(
        &device,
        include_str!("q8_tensorops.metal"),
        "gemm_q8_tensorops_probe",
        &options,
    );
    let tiled_options = CompileOptions::new();
    tiled_options.set_language_version(MTLLanguageVersion::V3_0);
    tiled_options.set_fast_math_enabled(false);
    let tiled = pipeline(
        &device,
        MSL_Q8_TILED_SOURCE,
        "gemm_q8_tiled",
        &tiled_options,
    );
    for pipeline in [&tensorops, &tiled] {
        assert_eq!(pipeline.thread_execution_width(), 32);
        assert!(pipeline.max_total_threads_per_threadgroup() >= 128);
    }
    eprintln!(
        "Q8 TensorOps device={} fast_math={} tiled_fast_math={}",
        device.name(),
        options.is_fast_math_enabled(),
        tiled_options.is_fast_math_enabled(),
    );
    let queue = device.new_command_queue();
    let shapes = [
        (1usize, 63usize, 32usize),
        (1, 65, 64),
        (25, 63, 32),
        (25, 65, 64),
        (65, 63, 32),
        (65, 65, 64),
        (64, 64, 64),
    ];
    for (m, n, k) in shapes {
        let (packed, weights) = make_q8_weight_bytes(0x268, n, k);
        let mut guarded_packed = vec![0xa5; PREFIX * 4];
        guarded_packed.extend_from_slice(&packed);
        guarded_packed.resize(
            (guarded_packed.len() + PREFIX * 4).next_multiple_of(4),
            0xa5,
        );
        // Reuse the existing upload/readback helpers without interpreting packed bits as numbers.
        let packed_words: Vec<f32> = guarded_packed
            .chunks_exact(4)
            .map(|bytes| f32::from_bits(u32::from_le_bytes(bytes.try_into().unwrap())))
            .collect();
        let qw_buffer = f32_buf(&device, &packed_words);
        for cancellation in [false, true] {
            let mut x: Vec<f32> = (0..m * k)
                .map(|index| ((index * 17 + index / k * 29) % 127) as f32 / 63.0 - 1.0)
                .collect();
            if cancellation {
                for row in 0..m {
                    let scale = 1.0 / (1 + row % 4) as f32;
                    for col in (0..k).step_by(2) {
                        x[row * k + col] = weights[col + 1] * scale;
                        x[row * k + col + 1] = -weights[col] * scale;
                    }
                }
            }
            let reference = cpu_matmul_ref(&x, &weights, m, n, k);
            let mut guarded_x = vec![GUARD; PREFIX];
            guarded_x.extend_from_slice(&x);
            guarded_x.resize(guarded_x.len() + PREFIX, GUARD);
            let x_buffer = f32_buf(&device, &guarded_x);
            let stride = n + 8;
            let output_storage = guarded_output(m, n, stride);
            let tiled_storage = guarded_output(m, n, n);
            let output_buffer = f32_buf(&device, &output_storage);
            let tiled_buffer = f32_buf(&device, &tiled_storage);
            let marker_count = 2 * m.div_ceil(32) * n.div_ceil(32);
            let marker_buffer = f32_buf(&device, &vec![0.0; marker_count]);
            let (m_arg, n_arg, k_arg, stride_arg) = (m as u32, n as u32, k as u32, stride as u32);
            let command = queue.new_command_buffer();
            for (pipeline, output, tensor_path) in [
                (&tensorops, &output_buffer, true),
                (&tiled, &tiled_buffer, false),
            ] {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(&qw_buffer), (PREFIX * 4) as u64);
                encoder.set_buffer(1, Some(&x_buffer), (PREFIX * 4) as u64);
                encoder.set_buffer(2, Some(output), (PREFIX * 4) as u64);
                for (index, value) in [(3, &m_arg), (4, &n_arg), (5, &k_arg)] {
                    encoder.set_bytes(index, 4, value as *const u32 as *const _);
                }
                if tensor_path {
                    encoder.set_bytes(6, 4, &stride_arg as *const u32 as *const _);
                    encoder.set_buffer(7, Some(&marker_buffer), 0);
                }
                let rows_per_group = if tensor_path { 32 } else { 64 };
                encoder.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(rows_per_group) as u64, 1),
                    MTLSize::new(32, 4, 1),
                );
                encoder.end_encoding();
            }
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

            let markers = read_f32(&marker_buffer, marker_count);
            let output_after = read_f32(&output_buffer, output_storage.len());
            let tiled_after = read_f32(&tiled_buffer, tiled_storage.len());
            let output = logical_output(&output_after, m, n, stride);
            let tiled_output = logical_output(&tiled_after, m, n, n);
            let tensor_cpu = discrepancies(&output, &reference);
            let tiled_cpu = discrepancies(&tiled_output, &reference);
            let tensor_tiled = discrepancies(&output, &tiled_output);
            eprintln!(
                "Q8 TensorOps M={m} N={n} K={k} cancellation={cancellation} MSL={} \
                 tensor_cpu_abs={} tensor_cpu_rel={} tiled_cpu_abs={} tiled_cpu_rel={} \
                 tensor_tiled_abs={} tensor_tiled_rel={}",
                markers[0],
                tensor_cpu.0,
                tensor_cpu.1,
                tiled_cpu.0,
                tiled_cpu.1,
                tensor_tiled.0,
                tensor_tiled.1,
            );
            for marker in markers.chunks_exact(2) {
                assert!(
                    marker[0].is_finite() && marker[0] >= 400.0,
                    "MSL version marker"
                );
                assert_eq!(marker[1], 268.0, "TensorOps tile completion marker");
            }
            let packed_after: Vec<u8> = read_f32(&qw_buffer, packed_words.len())
                .iter()
                .flat_map(|word| word.to_bits().to_le_bytes())
                .collect();
            assert_eq!(
                packed_after, guarded_packed,
                "Q8 payload and guards changed"
            );
            let x_after = read_f32(&x_buffer, guarded_x.len());
            for (&actual, &expected) in x_after.iter().zip(&guarded_x) {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "activation buffer changed"
                );
            }
            assert_output_guards(&output_after, m, n, stride);
            assert_output_guards(&tiled_after, m, n, n);
            for (name, (absolute, relative), tolerance) in [
                ("TensorOps/CPU", tensor_cpu, CPU_TOLERANCE),
                ("tiled/CPU", tiled_cpu, CPU_TOLERANCE),
                ("TensorOps/tiled", tensor_tiled, TILED_TOLERANCE),
            ] {
                assert!(
                    absolute.is_finite() && relative.is_finite() && absolute < tolerance,
                    "{name} M={m} N={n} K={k} cancellation={cancellation}: \
                     max_abs={absolute} max_rel={relative} tolerance={tolerance}",
                );
            }
        }
    }
    eprintln!("Q8_TENSOROPS_PROBE_PASS cases=14");
}
