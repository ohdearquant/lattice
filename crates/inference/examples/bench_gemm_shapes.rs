//! GEMM and attention shape microbench: dispatch the production `matmul_bt` and
//! `gemm_q4_tiled` kernels, the simdgroup-matrix `gemm_f32_sg`, and the head-64
//! attention kernels at caller-given shapes, and check their output against a
//! host reference.
//!
//! Both kernels are the existing shader sources, compiled and dispatched the way
//! the library dispatches them:
//!
//! - `matmul_bt` (`Y[M,N] = X[M,K] . W[N,K]^T`, all f32) lives in
//!   `flash_attention.metal`; it is compiled from the same four-file source the
//!   EmbeddingGemma 2 forward pass builds, with the same placeholder
//!   substitutions. Buffers 0/1/2 are X, W, Y; constants 3/4/5 are M, N, K; the
//!   grid is `(ceil(N/16), ceil(M/16), 1)` with 16x16 threads per threadgroup.
//!   Fast math is disabled.
//! - `gemm_q4_tiled` (X f32 `[M,K]`, W in the 20-byte Q4 block layout, Y f32
//!   `[M,N]`) is compiled on its own with the Metal 3.0 language version — the
//!   shader guards its body with `__METAL_VERSION__ >= 300` — and only on a
//!   device that reports the Apple7 GPU family. Buffers 0/1/2 are QW, X, Y;
//!   constants 3/4/5 are M, N, K; the grid is `(ceil(N/32), ceil(M/64), 1)` with
//!   32x4 threads per threadgroup.
//! - `gemm_f32_sg` (`Y[M,N] = X[M,K] . W[N,K]^T`, all f32) runs the same tile as
//!   `gemm_q4_tiled` with f32 operands and accumulators on simdgroup matrices.
//!   It is compiled like `gemm_q4_tiled` (Metal 3.0, Apple7 family); buffers
//!   0/1/2 are X, W, Y, constants 3/4/5 are M, N, K, and the grid and
//!   threadgroup size are `gemm_q4_tiled`'s.
//! - `eg2_attention_hd64` (`attn_vec`) and `eg2_attention_sg_hd64` (`attn_sg`):
//!   bidirectional attention without a window over `[seq, 12, 64]` queries,
//!   keys and values with scale 1, the vision encoder's geometry. Both are
//!   compiled from one library: the four-file source above followed by the
//!   attention shader specialised to head width 64 and the simdgroup-matrix
//!   attention shader, fast math disabled. Buffers 0-3 are Q, K, V, O and buffer
//!   4 holds the attention parameters. `attn_vec` runs one simdgroup per query
//!   row, grid `(heads, ceil(seq/8))` with 256 threads; `attn_sg` runs 32 query
//!   rows per threadgroup on simdgroup matrices, grid `(heads, ceil(seq/32))`
//!   with 128 threads. Each sequence length runs in two input regimes: uniform
//!   (Q, K, V in [-1, 1)) and peaked (Q scaled by 8, so the running maximum
//!   moves across key tiles and the rescale of the partial output carries
//!   weight).
//!
//! Before any timed work the GPU clock is brought up by a fixed, untimed run of
//! `gemm_f32_sg` at 2304x3072x768 (`CLOCK_WARMUP_DISPATCHES` dispatches, each
//! in its own command buffer); its first and last spans are printed as an
//! `@@warmup` line. Without it, a short kernel timed first after an idle pause
//! measures the ramp rather than the kernel.
//!
//! Timing: each repetition encodes one dispatch in its own command buffer,
//! commits it, waits, and reads that command buffer's `GPUStartTime` and
//! `GPUEndTime`. Warmup repetitions are discarded. Per (kernel, shape) the
//! reported figure is the median of the timed GPU spans, with min, max and the
//! implied `2*M*N*K / median_seconds` rate (`4*seq^2*heads*64 / median_seconds`
//! for attention).
//!
//! After the timed repetitions each (kernel, shape) is checked: a deterministic
//! sample of output elements is compared against a host dot product that mirrors
//! the kernel's staging. A sample whose relative error exceeds the kernel's
//! bound exits nonzero, so a timing of a mis-dispatched kernel is never
//! reported. Attention is checked per sampled (query row, head) pair against an
//! f64 softmax over every key: the largest absolute difference over the 64
//! output dimensions, divided by the largest `sum_j p_j |v_j|`. `attn_sg` exits
//! nonzero above its bound and either kernel exits nonzero on a non-finite
//! output; `attn_vec`'s error is reported as a reading.
//!
//! Usage:
//!   cargo run --release --example bench_gemm_shapes -p lattice-inference \
//!     --features f16,metal-gpu -- --shape 2304x768x768 --kernel all --reps 20
//!
//! CLI: `--shape MxNxK` (repeatable; defaults to 2304x768x768, 2304x3072x768,
//! 2304x768x3072, 2304x2304x64),
//! `--kernel matmul_bt|q4_tiled|f32_sg|attn_vec|attn_sg|all` (default all),
//! `--attn-seq N` (repeatable; defaults to 2304 and 2394), `--reps N` (default
//! 20), `--warmup N` (default 3).

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn main() {
    eprintln!("bench_gemm_shapes requires macOS + the metal-gpu feature.");
    std::process::exit(1);
}

/// The GPU lock is taken here and lives until the process exits, so every
/// dispatch below runs under it.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn main() {
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();
    if let Err(error) = run() {
        eprintln!("bench_gemm_shapes: {error}");
        std::process::exit(1);
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), String> {
    use metal::*;

    let options = parse_options(std::env::args().skip(1))?;
    let device = Device::system_default().ok_or("no Metal device available")?;
    let queue = device.new_command_queue();

    let matmul = match options.kernel {
        KernelSelection::MatmulBt | KernelSelection::All => Some(build_matmul_pipeline(&device)?),
        _ => None,
    };
    let q4_tiled = match options.kernel {
        KernelSelection::Q4Tiled | KernelSelection::All => Some(build_q4_tiled_pipeline(&device)?),
        _ => None,
    };
    let f32_sg = match options.kernel {
        KernelSelection::F32Sg | KernelSelection::All => Some(build_f32_sg_pipeline(&device)?),
        _ => None,
    };
    let attention = match options.kernel {
        KernelSelection::AttnVec | KernelSelection::AttnSg | KernelSelection::All => {
            Some(build_attention_pipelines(&device)?)
        }
        _ => None,
    };

    clock_warmup(&device, &queue)?;

    let mut failed = false;
    for shape in &options.shapes {
        if let Some(pipeline) = matmul.as_ref()
            && !run_matmul_bt(&device, &queue, pipeline, *shape, &options)?
        {
            failed = true;
        }
        if let Some(pipeline) = q4_tiled.as_ref()
            && !run_q4_tiled(&device, &queue, pipeline, *shape, &options)?
        {
            failed = true;
        }
        if let Some(pipeline) = f32_sg.as_ref()
            && !run_f32_sg(&device, &queue, pipeline, *shape, &options)?
        {
            failed = true;
        }
    }
    if let Some((vector, matrix)) = attention.as_ref() {
        for &seq in &options.attn_seqs {
            for regime in [Regime::Uniform, Regime::Peaked] {
                let selected: [(&str, &metal::ComputePipelineState, bool); 2] = [
                    (
                        "attn_vec",
                        vector,
                        matches!(
                            options.kernel,
                            KernelSelection::AttnVec | KernelSelection::All
                        ),
                    ),
                    (
                        "attn_sg",
                        matrix,
                        matches!(
                            options.kernel,
                            KernelSelection::AttnSg | KernelSelection::All
                        ),
                    ),
                ];
                for (kernel, pipeline, enabled) in selected {
                    if enabled
                        && !run_attention(&device, &queue, pipeline, kernel, seq, regime, &options)?
                    {
                        failed = true;
                    }
                }
            }
        }
    }
    if failed {
        return Err(
            "a sampled output exceeded its kernel's relative-error bound or was not finite"
                .to_string(),
        );
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Clone, Copy)]
enum KernelSelection {
    MatmulBt,
    Q4Tiled,
    F32Sg,
    AttnVec,
    AttnSg,
    All,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    n: usize,
    k: usize,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
struct Options {
    shapes: Vec<Shape>,
    kernel: KernelSelection,
    attn_seqs: Vec<usize>,
    reps: usize,
    warmup: usize,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const DEFAULT_SHAPES: [Shape; 4] = [
    Shape {
        m: 2304,
        n: 768,
        k: 768,
    },
    Shape {
        m: 2304,
        n: 3072,
        k: 768,
    },
    Shape {
        m: 2304,
        n: 768,
        k: 3072,
    },
    Shape {
        m: 2304,
        n: 2304,
        k: 64,
    },
];

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const DEFAULT_ATTN_SEQS: [usize; 2] = [2304, 2394];
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const DEFAULT_REPS: usize = 20;
/// Untimed dispatches of `gemm_f32_sg` at [`CLOCK_WARMUP_SHAPE`] run before any
/// timed work.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const CLOCK_WARMUP_DISPATCHES: usize = 40;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const CLOCK_WARMUP_SHAPE: Shape = Shape {
    m: 2304,
    n: 3072,
    k: 768,
};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const DEFAULT_WARMUP: usize = 3;
/// Relative-error bound for `matmul_bt`: f32 inputs, f32 accumulation, f32 output.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const MATMUL_REL_BOUND: f64 = 1.0e-3;
/// Relative-error bound for `gemm_q4_tiled`. The kernel decodes each weight in
/// f32 (`nibble * scale + bias`) and stores it, like X, as `half` in its
/// threadgroup tiles. The host reference applies the same two f16 roundings, so
/// the remaining difference is f32 accumulation order, at most about
/// `K * 2^-24` of the summed term magnitudes (1.8e-4 at K=3072). 5e-3 keeps a
/// margin over that while a decode, layout or dispatch mistake lands far above it.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const Q4_REL_BOUND: f64 = 5.0e-3;
/// Relative-error bound for `gemm_f32_sg`: f32 operands and accumulation, so the
/// difference from the f64 reference is accumulation order alone. 1e-6 is f32
/// class (`matmul_bt` measures about 1.6e-7 at K=3072) and sits far below what
/// an operand staged through f16 would produce.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const F32_SG_REL_BOUND: f64 = 1.0e-6;
/// Error bound for `attn_sg`, as a fraction of the largest `sum_j p_j |v_j|` of
/// the checked row: f32 scores, exponentials and accumulation over every key,
/// against an f64 reference. 1e-4 leaves room for the online rescaling across
/// key tiles while a dropped rescale or a misplaced tile lands far above it.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const ATTN_SG_REL_BOUND: f64 = 1.0e-4;
/// Attention geometry of the vision encoder: 12 query and key/value heads of
/// width 64, scale 1.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const ATTN_HEADS: usize = 12;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const ATTN_HEAD_DIM: usize = 64;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const ATTN_SCALE: f32 = 1.0;
/// Number of sampled output elements compared against the host reference.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const SAMPLE_COUNT: usize = 64;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn parse_options(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        shapes: Vec::new(),
        kernel: KernelSelection::All,
        attn_seqs: Vec::new(),
        reps: DEFAULT_REPS,
        warmup: DEFAULT_WARMUP,
    };
    let mut args = args;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--shape" => {
                let value = args.next().ok_or("--shape needs an MxNxK argument")?;
                options.shapes.push(parse_shape(&value)?);
            }
            "--kernel" => {
                let value = args.next().ok_or(
                    "--kernel needs matmul_bt, q4_tiled, f32_sg, attn_vec, attn_sg or all",
                )?;
                options.kernel = match value.as_str() {
                    "matmul_bt" => KernelSelection::MatmulBt,
                    "q4_tiled" => KernelSelection::Q4Tiled,
                    "f32_sg" => KernelSelection::F32Sg,
                    "attn_vec" => KernelSelection::AttnVec,
                    "attn_sg" => KernelSelection::AttnSg,
                    "all" => KernelSelection::All,
                    other => {
                        return Err(format!(
                            "--kernel must be matmul_bt, q4_tiled, f32_sg, attn_vec, attn_sg or all, got {other}"
                        ));
                    }
                };
            }
            "--attn-seq" => {
                let value = args.next().ok_or("--attn-seq needs a sequence length")?;
                let seq = parse_count(&value, "--attn-seq")?;
                if seq == 0 {
                    return Err("--attn-seq must be at least 1".to_string());
                }
                options.attn_seqs.push(seq);
            }
            "--reps" => {
                let value = args.next().ok_or("--reps needs a count")?;
                options.reps = parse_count(&value, "--reps")?;
            }
            "--warmup" => {
                let value = args.next().ok_or("--warmup needs a count")?;
                options.warmup = parse_count(&value, "--warmup")?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if options.shapes.is_empty() {
        options.shapes.extend_from_slice(&DEFAULT_SHAPES);
    }
    if options.attn_seqs.is_empty() {
        options.attn_seqs.extend_from_slice(&DEFAULT_ATTN_SEQS);
    }
    if options.reps == 0 {
        return Err("--reps must be at least 1".to_string());
    }
    Ok(options)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn parse_shape(value: &str) -> Result<Shape, String> {
    let parts: Vec<&str> = value.split('x').collect();
    if parts.len() != 3 {
        return Err(format!("--shape wants MxNxK, got {value}"));
    }
    let mut dims = [0usize; 3];
    for (slot, text) in dims.iter_mut().zip(parts) {
        *slot = text
            .parse::<usize>()
            .map_err(|error| format!("--shape component {text} is not a size: {error}"))?;
        if *slot == 0 {
            return Err(format!("--shape component {text} must be at least 1"));
        }
    }
    Ok(Shape {
        m: dims[0],
        n: dims[1],
        k: dims[2],
    })
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn parse_count(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("{flag} value {value} is not a count: {error}"))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn build_matmul_pipeline(device: &metal::Device) -> Result<metal::ComputePipelineState, String> {
    use metal::CompileOptions;

    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(&matmul_source(), &options)
        .map_err(|error| format!("matmul_bt shader compilation failed: {error}"))?;
    let function = library
        .get_function("matmul_bt", None)
        .map_err(|error| format!("matmul_bt kernel missing: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("matmul_bt pipeline failed: {error}"))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn build_q4_tiled_pipeline(device: &metal::Device) -> Result<metal::ComputePipelineState, String> {
    use metal::{CompileOptions, MTLGPUFamily, MTLLanguageVersion};

    if !device.supports_family(MTLGPUFamily::Apple7) {
        return Err(
            "gemm_q4_tiled needs the Apple7 GPU family (simdgroup matrix operations)".to_string(),
        );
    }
    let options = CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V3_0);
    let library = device
        .new_library_with_source(Q4_TILED_SOURCE, &options)
        .map_err(|error| format!("gemm_q4_tiled shader compilation failed: {error}"))?;
    let function = library
        .get_function("gemm_q4_tiled", None)
        .map_err(|error| format!("gemm_q4_tiled kernel missing: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("gemm_q4_tiled pipeline failed: {error}"))
}

/// The shader source `matmul_bt` is compiled from in the EmbeddingGemma 2
/// forward pass: the same four files, with the same placeholder substitutions
/// for the other kernels they define.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn matmul_source() -> String {
    concat!(
        include_str!("../src/forward/shaders/rms_reduce.metal"),
        include_str!("../src/forward/shaders/flash_attention.metal"),
        include_str!("../src/forward/shaders/ernie45_rope.metal"),
        include_str!("../src/forward/shaders/embeddinggemma2.metal"),
    )
    .replace("__FA_HEAD_DIM__", "128")
    .replace("__FA_GQA_GROUPS__", "1")
    .replace("__FUSED_C_HEAD_DIM__", "128")
    .replace("__FUSED_C_HALF_DIM__", "64")
    .replace("__FUSED_C_THREADS__", "64")
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const Q4_TILED_SOURCE: &str = include_str!("../src/forward/shaders/gemm_q4_tiled.metal");

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const F32_SG_SOURCE: &str = include_str!("../src/forward/shaders/gemm_f32_sg.metal");

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn build_f32_sg_pipeline(device: &metal::Device) -> Result<metal::ComputePipelineState, String> {
    use metal::{CompileOptions, MTLGPUFamily, MTLLanguageVersion};

    if !device.supports_family(MTLGPUFamily::Apple7) {
        return Err(
            "gemm_f32_sg needs the Apple7 GPU family (simdgroup matrix operations)".to_string(),
        );
    }
    let options = CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V3_0);
    options.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(F32_SG_SOURCE, &options)
        .map_err(|error| format!("gemm_f32_sg shader compilation failed: {error}"))?;
    let function = library
        .get_function("gemm_f32_sg", None)
        .map_err(|error| format!("gemm_f32_sg kernel missing: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("gemm_f32_sg pipeline failed: {error}"))
}

/// Deterministic 64-bit counter RNG (SplitMix64), so every buffer and every
/// sample index is reproducible from a fixed seed.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
struct Rng {
    state: u64,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[-1, 1)`, rounded to f32.
    fn signed(&mut self) -> f32 {
        (2.0 * self.unit() - 1.0) as f32
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn activation(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..len).map(|_| rng.signed()).collect()
}

/// Build the Q4 weight payload with the crate's own row quantizer, so the bytes
/// are exactly the block layout `gemm_q4_tiled` decodes: per output row, `K/32`
/// blocks of 20 bytes (two f16 metadata fields followed by 16 packed nibbles).
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn q4_weight_payload(shape: Shape, seed: u64) -> Result<Vec<u8>, String> {
    use lattice_inference::weights::q4_weights::quantize_row_q4_0;

    if !shape.k.is_multiple_of(32) {
        return Err(q4_block_width_error(shape.k));
    }
    let mut rng = Rng::new(seed);
    let mut payload = Vec::with_capacity(shape.n * (shape.k / 32) * 20);
    let mut row = vec![0.0f32; shape.k];
    for _ in 0..shape.n {
        for value in row.iter_mut() {
            *value = rng.signed();
        }
        let encoded = quantize_row_q4_0(&row)
            .map_err(|error| format!("Q4 quantization of a synthetic row failed: {error}"))?;
        payload.extend_from_slice(&encoded);
    }
    Ok(payload)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn q4_block_width_error(k: usize) -> String {
    format!("q4_tiled needs K divisible by 32 (one Q4 block per 32 inputs), got K={k}")
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn shared_buffer(device: &metal::Device, bytes: usize) -> metal::Buffer {
    let length = bytes as u64;
    device.new_buffer(length, metal::MTLResourceOptions::StorageModeShared)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn write_slice(buffer: &metal::Buffer, values: &[f32]) {
    let count = values.len();
    // SAFETY: `buffer` is a shared-storage allocation of at least `values.len()`
    // f32s created for this call; no GPU work is in flight against it; the slice
    // does not outlive the statement.
    unsafe {
        let destination = std::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), count);
        destination.copy_from_slice(values);
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn write_bytes(buffer: &metal::Buffer, bytes: &[u8]) {
    let count = bytes.len();
    // SAFETY: same contract as `write_slice`, with a byte-typed destination.
    unsafe {
        let destination = std::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), count);
        destination.copy_from_slice(bytes);
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn read_slice(buffer: &metal::Buffer, count: usize) -> Vec<f32> {
    // SAFETY: `buffer` is a shared-storage allocation of at least `count` f32s
    // whose producing command buffer has completed, so the bytes are visible to
    // the CPU; the returned `Vec` is a detached copy.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), count).to_vec() }
}

/// Read a completed command buffer's `GPUStartTime`/`GPUEndTime` (`CFTimeInterval`
/// seconds, exposed as no-argument f64 properties) and return the span in
/// microseconds, clamped at zero. The raw message send is needed because the
/// metal crate does not surface these two properties.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn gpu_span_us(command: &metal::CommandBufferRef) -> Result<f64, String> {
    let start_selector = objc::runtime::Sel::register("GPUStartTime");
    let end_selector = objc::runtime::Sel::register("GPUEndTime");
    // SAFETY: both selectors name documented no-argument `CFTimeInterval` (f64)
    // command-buffer properties, and `command` has completed, so the timestamps
    // are final. A send that returns no usable value is reported as an error
    // rather than read as a zero-length span.
    let start: f64 = unsafe {
        objc::Message::send_message(command, start_selector, ())
            .map_err(|error| format!("GPUStartTime read failed: {error}"))?
    };
    let end: f64 = unsafe {
        objc::Message::send_message(command, end_selector, ())
            .map_err(|error| format!("GPUEndTime read failed: {error}"))?
    };
    Ok(((end - start) * 1.0e6).max(0.0))
}

/// Encode one dispatch in its own command buffer, commit, wait for completion,
/// and return the GPU span in microseconds.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn timed_dispatch(
    queue: &metal::CommandQueue,
    encode: impl Fn(&metal::ComputeCommandEncoderRef),
) -> Result<f64, String> {
    let command = queue.new_command_buffer();
    {
        let encoder = command.new_compute_command_encoder();
        encode(encoder);
        encoder.end_encoding();
    }
    command.commit();
    command.wait_until_completed();
    let status = command.status();
    if status != metal::MTLCommandBufferStatus::Completed {
        return Err(format!("command buffer ended with {status:?}"));
    }
    gpu_span_us(command)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn span_stats(spans_us: &[f64]) -> Result<(f64, f64, f64), String> {
    if spans_us.is_empty() {
        return Err("no timed repetitions were collected".to_string());
    }
    let mut sorted = spans_us.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    Ok((
        sorted[sorted.len() / 2],
        sorted[0],
        sorted[sorted.len() - 1],
    ))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn summary(kernel: &str, shape: Shape, spans_us: &[f64]) -> Result<(), String> {
    let (median, min, max) = span_stats(spans_us)?;
    let reps = spans_us.len();
    let (m, n, k) = (shape.m, shape.n, shape.k);
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = if median > 0.0 {
        flops / (median * 1.0e-6) / 1.0e12
    } else {
        0.0
    };
    println!(
        "@@gemm kernel={kernel} m={m} n={n} k={k} reps={reps} median_us={median:.3} min_us={min:.3} max_us={max:.3} tflops={tflops:.4}"
    );
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn report_check(kernel: &str, shape: Shape, max_abs: f64, max_rel: f64) {
    let (m, n, k) = (shape.m, shape.n, shape.k);
    println!(
        "@@check kernel={kernel} m={m} n={n} k={k} max_abs={max_abs:.6e} max_rel={max_rel:.6e}"
    );
}

/// Deterministic sample of output element indices: tile-boundary rows and
/// columns (`BM=64`, `BN=32` in the tiled kernel; 16x16 in `matmul_bt`), the
/// edges, then pseudo-random fill up to `SAMPLE_COUNT`.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn sample_indices(shape: Shape) -> Vec<usize> {
    use std::collections::BTreeSet;

    let mut indices = BTreeSet::new();
    let rows = [
        0,
        15,
        16,
        31,
        32,
        63,
        64,
        65,
        127,
        128,
        shape.m / 2,
        shape.m.saturating_sub(2),
        shape.m - 1,
    ];
    let cols = [
        0,
        15,
        16,
        31,
        32,
        33,
        63,
        64,
        95,
        96,
        shape.n / 2,
        shape.n.saturating_sub(2),
        shape.n - 1,
    ];
    for row in rows {
        for col in cols {
            if row < shape.m && col < shape.n {
                indices.insert(row * shape.n + col);
            }
        }
    }
    let elements = shape.m * shape.n;
    let target = SAMPLE_COUNT.min(elements);
    let mut rng = Rng::new(0x5EED_1234_5678_9ABC);
    while indices.len() < target {
        let index = (rng.next_u64() as usize) % elements;
        indices.insert(index);
    }
    indices.into_iter().collect()
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_matmul_bt(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    shape: Shape,
    options: &Options,
) -> Result<bool, String> {
    let m = u32::try_from(shape.m).map_err(|_| "M exceeds u32 indexing".to_string())?;
    let n = u32::try_from(shape.n).map_err(|_| "N exceeds u32 indexing".to_string())?;
    let k = u32::try_from(shape.k).map_err(|_| "K exceeds u32 indexing".to_string())?;

    let x_values = activation(0x1111_2222_3333_4444, shape.m * shape.k);
    let w_values = activation(0x5555_6666_7777_8888, shape.n * shape.k);
    let x_buffer = shared_buffer(device, shape.m * shape.k * 4);
    let w_buffer = shared_buffer(device, shape.n * shape.k * 4);
    let y_buffer = shared_buffer(device, shape.m * shape.n * 4);
    write_slice(&x_buffer, &x_values);
    write_slice(&w_buffer, &w_values);

    let encode = |encoder: &metal::ComputeCommandEncoderRef| {
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&w_buffer), 0);
        encoder.set_buffer(2, Some(&y_buffer), 0);
        encoder.set_bytes(3, 4, std::ptr::from_ref(&m).cast());
        encoder.set_bytes(4, 4, std::ptr::from_ref(&n).cast());
        encoder.set_bytes(5, 4, std::ptr::from_ref(&k).cast());
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(u64::from(n).div_ceil(16), u64::from(m).div_ceil(16), 1),
            metal::MTLSize::new(16, 16, 1),
        );
    };

    for _ in 0..options.warmup {
        timed_dispatch(queue, encode)?;
    }
    let mut spans = Vec::with_capacity(options.reps);
    for _ in 0..options.reps {
        spans.push(timed_dispatch(queue, encode)?);
    }
    summary("matmul_bt", shape, &spans)?;

    let y = read_slice(&y_buffer, shape.m * shape.n);
    let (max_abs, max_rel) = matmul_reference_error(&x_values, &w_values, &y, shape);
    report_check("matmul_bt", shape, max_abs, max_rel);
    if max_rel > MATMUL_REL_BOUND {
        let bound = MATMUL_REL_BOUND;
        eprintln!("matmul_bt {m}x{n}x{k} relative error {max_rel:.6e} exceeds {bound:.1e}");
        return Ok(false);
    }
    Ok(true)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_q4_tiled(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    shape: Shape,
    options: &Options,
) -> Result<bool, String> {
    if !shape.k.is_multiple_of(32) {
        return Err(q4_block_width_error(shape.k));
    }
    let m = u32::try_from(shape.m).map_err(|_| "M exceeds u32 indexing".to_string())?;
    let n = u32::try_from(shape.n).map_err(|_| "N exceeds u32 indexing".to_string())?;
    let k = u32::try_from(shape.k).map_err(|_| "K exceeds u32 indexing".to_string())?;

    let payload = q4_weight_payload(shape, 0x0A0B_0C0D_0E0F_1011)?;
    let x_values = activation(0x1213_1415_1617_1819, shape.m * shape.k);
    let qw_buffer = shared_buffer(device, payload.len());
    write_bytes(&qw_buffer, &payload);
    let x_buffer = shared_buffer(device, shape.m * shape.k * 4);
    let y_buffer = shared_buffer(device, shape.m * shape.n * 4);
    write_slice(&x_buffer, &x_values);

    let encode = |encoder: &metal::ComputeCommandEncoderRef| {
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&qw_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&y_buffer), 0);
        encoder.set_bytes(3, 4, std::ptr::from_ref(&m).cast());
        encoder.set_bytes(4, 4, std::ptr::from_ref(&n).cast());
        encoder.set_bytes(5, 4, std::ptr::from_ref(&k).cast());
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(u64::from(n).div_ceil(32), u64::from(m).div_ceil(64), 1),
            metal::MTLSize::new(32, 4, 1),
        );
    };

    for _ in 0..options.warmup {
        timed_dispatch(queue, encode)?;
    }
    let mut spans = Vec::with_capacity(options.reps);
    for _ in 0..options.reps {
        spans.push(timed_dispatch(queue, encode)?);
    }
    summary("q4_tiled", shape, &spans)?;

    let y = read_slice(&y_buffer, shape.m * shape.n);
    let (max_abs, max_rel) = q4_reference_error(&x_values, &payload, &y, shape)?;
    report_check("q4_tiled", shape, max_abs, max_rel);
    if max_rel > Q4_REL_BOUND {
        let bound = Q4_REL_BOUND;
        eprintln!("q4_tiled {m}x{n}x{k} relative error {max_rel:.6e} exceeds {bound:.1e}");
        return Ok(false);
    }
    Ok(true)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_f32_sg(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    shape: Shape,
    options: &Options,
) -> Result<bool, String> {
    if !shape.k.is_multiple_of(4) {
        let k = shape.k;
        return Err(format!(
            "f32_sg needs K divisible by 4 (float4 staging), got K={k}"
        ));
    }
    let m = u32::try_from(shape.m).map_err(|_| "M exceeds u32 indexing".to_string())?;
    let n = u32::try_from(shape.n).map_err(|_| "N exceeds u32 indexing".to_string())?;
    let k = u32::try_from(shape.k).map_err(|_| "K exceeds u32 indexing".to_string())?;

    let x_values = activation(0x1111_2222_3333_4444, shape.m * shape.k);
    let w_values = activation(0x5555_6666_7777_8888, shape.n * shape.k);
    let x_buffer = shared_buffer(device, shape.m * shape.k * 4);
    let w_buffer = shared_buffer(device, shape.n * shape.k * 4);
    let y_buffer = shared_buffer(device, shape.m * shape.n * 4);
    write_slice(&x_buffer, &x_values);
    write_slice(&w_buffer, &w_values);

    let encode = |encoder: &metal::ComputeCommandEncoderRef| {
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&w_buffer), 0);
        encoder.set_buffer(2, Some(&y_buffer), 0);
        encoder.set_bytes(3, 4, std::ptr::from_ref(&m).cast());
        encoder.set_bytes(4, 4, std::ptr::from_ref(&n).cast());
        encoder.set_bytes(5, 4, std::ptr::from_ref(&k).cast());
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(u64::from(n).div_ceil(32), u64::from(m).div_ceil(64), 1),
            metal::MTLSize::new(32, 4, 1),
        );
    };

    for _ in 0..options.warmup {
        timed_dispatch(queue, encode)?;
    }
    let mut spans = Vec::with_capacity(options.reps);
    for _ in 0..options.reps {
        spans.push(timed_dispatch(queue, encode)?);
    }
    summary("f32_sg", shape, &spans)?;

    let y = read_slice(&y_buffer, shape.m * shape.n);
    let (max_abs, max_rel) = matmul_reference_error(&x_values, &w_values, &y, shape);
    report_check("f32_sg", shape, max_abs, max_rel);
    if max_rel > F32_SG_REL_BOUND {
        let bound = F32_SG_REL_BOUND;
        eprintln!("f32_sg {m}x{n}x{k} relative error {max_rel:.6e} exceeds {bound:.1e}");
        return Ok(false);
    }
    Ok(true)
}

/// Reference dot product for `matmul_bt`: f32 inputs, the f32 products summed in
/// f64. Returns `(max_abs, max_rel)` over the deterministic sample, where
/// `max_rel` divides the absolute difference by the sum of the term magnitudes —
/// a per-element, magnitude-free normalisation. Dividing by `|reference|` alone
/// is unbounded when a cancellation makes the reference small while its terms
/// are not.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn matmul_reference_error(x: &[f32], w: &[f32], y: &[f32], shape: Shape) -> (f64, f64) {
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for index in sample_indices(shape) {
        let row = index / shape.n;
        let col = index % shape.n;
        let mut reference = 0.0f64;
        let mut magnitude = 0.0f64;
        for step in 0..shape.k {
            let term = f64::from(x[row * shape.k + step]) * f64::from(w[col * shape.k + step]);
            reference += term;
            magnitude += term.abs();
        }
        let difference = (f64::from(y[index]) - reference).abs();
        max_abs = max_abs.max(difference);
        max_rel = max_rel.max(difference / magnitude.max(f64::MIN_POSITIVE));
    }
    (max_abs, max_rel)
}

/// Reference dot product for `gemm_q4_tiled`, mirroring the kernel's staging: X
/// and the decoded weight are rounded to f16 (the threadgroup tiles are `half`),
/// and their f32 product is summed in f64. The kernel accumulates the same
/// products in f32, which the single f16 staging leaves as sub-unit-roundoff
/// summation-order noise inside [`Q4_REL_BOUND`].
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn q4_reference_error(
    x: &[f32],
    payload: &[u8],
    y: &[f32],
    shape: Shape,
) -> Result<(f64, f64), String> {
    use lattice_inference::weights::f16_weights::F16;

    let blocks_per_row = shape.k / 32;
    let weights = dequantized_q4_weights(payload, shape)?;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for index in sample_indices(shape) {
        let row = index / shape.n;
        let col = index % shape.n;
        let mut reference = 0.0f64;
        let mut magnitude = 0.0f64;
        for block in 0..blocks_per_row {
            let block_values = &weights[col * blocks_per_row + block];
            for slot in 0..32 {
                let step = block * 32 + slot;
                let x_half = F16::from_f32(x[row * shape.k + step]).to_f32();
                let w_half = F16::from_f32(block_values[slot]).to_f32();
                let term = f64::from(x_half) * f64::from(w_half);
                reference += term;
                magnitude += term.abs();
            }
        }
        let difference = (f64::from(y[index]) - reference).abs();
        max_abs = max_abs.max(difference);
        max_rel = max_rel.max(difference / magnitude.max(f64::MIN_POSITIVE));
    }
    Ok((max_abs, max_rel))
}

/// Decode the payload with the crate's own Q4 row decoder, the host mirror of the
/// shader's block decode: f16 scale and bias from the first four bytes of a
/// 20-byte block, then `nibble * scale + bias` with the low nibble of each byte
/// first.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn dequantized_q4_weights(payload: &[u8], shape: Shape) -> Result<Vec<[f32; 32]>, String> {
    use lattice_inference::weights::q4_weights::dequantize_row_q4_0;

    let blocks_per_row = shape.k / 32;
    let row_bytes = blocks_per_row * 20;
    let expected = shape.n * row_bytes;
    if payload.len() != expected {
        let actual = payload.len();
        let rows = shape.n;
        let width = shape.k;
        return Err(format!(
            "Q4 payload is {actual} bytes, expected {expected} for N={rows} K={width}"
        ));
    }
    let mut decoded = Vec::with_capacity(shape.n * blocks_per_row);
    for row in 0..shape.n {
        let channel = &payload[row * row_bytes..(row + 1) * row_bytes];
        let values = dequantize_row_q4_0(channel, shape.k);
        for block in 0..blocks_per_row {
            let mut slots = [0.0f32; 32];
            slots.copy_from_slice(&values[block * 32..block * 32 + 32]);
            decoded.push(slots);
        }
    }
    Ok(decoded)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const ATTENTION_SOURCE: &str =
    include_str!("../src/forward/shaders/embeddinggemma2_attention.metal");

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const ATTENTION_SG_SOURCE: &str = include_str!("../src/forward/shaders/attention_sg_hd64.metal");

/// Layout of `Eg2AttentionParams` in `embeddinggemma2.metal`.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[repr(C)]
struct AttentionParams {
    seq_len: u32,
    heads: u32,
    kv_heads: u32,
    window: u32,
    use_window: u32,
    scale: f32,
    pad: [u32; 2],
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Clone, Copy)]
enum Regime {
    Uniform,
    Peaked,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
impl Regime {
    fn name(self) -> &'static str {
        match self {
            Regime::Uniform => "uniform",
            Regime::Peaked => "peaked",
        }
    }

    fn query_factor(self) -> f32 {
        match self {
            Regime::Uniform => 1.0,
            Regime::Peaked => 8.0,
        }
    }
}

/// Build `eg2_attention_hd64` and `eg2_attention_sg_hd64` from one library: the
/// four-file source, the attention shader at head width 64, then the
/// simdgroup-matrix attention shader, with fast math disabled.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn build_attention_pipelines(
    device: &metal::Device,
) -> Result<(metal::ComputePipelineState, metal::ComputePipelineState), String> {
    use metal::{CompileOptions, MTLGPUFamily};

    if !device.supports_family(MTLGPUFamily::Apple7) {
        return Err(
            "eg2_attention_sg_hd64 needs the Apple7 GPU family (simdgroup matrix operations)"
                .to_string(),
        );
    }
    let mut source = matmul_source();
    source.push_str(&ATTENTION_SOURCE.replace("__HD__", "64"));
    source.push_str(ATTENTION_SG_SOURCE);
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(&source, &options)
        .map_err(|error| format!("attention shader compilation failed: {error}"))?;
    let make = |name: &str, threads: u64| -> Result<metal::ComputePipelineState, String> {
        let function = library
            .get_function(name, None)
            .map_err(|error| format!("{name} kernel missing: {error}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("{name} pipeline failed: {error}"))?;
        if pipeline.max_total_threads_per_threadgroup() < threads {
            let permitted = pipeline.max_total_threads_per_threadgroup();
            return Err(format!(
                "{name} needs {threads} threads per group, pipeline permits {permitted}"
            ));
        }
        if pipeline.thread_execution_width() != 32 {
            return Err(format!("{name} requires a 32-lane SIMD execution width"));
        }
        Ok(pipeline)
    };
    Ok((
        make("eg2_attention_hd64", 256)?,
        make("eg2_attention_sg_hd64", 128)?,
    ))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_attention(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    kernel: &str,
    seq: usize,
    regime: Regime,
    options: &Options,
) -> Result<bool, String> {
    let seq_len = u32::try_from(seq).map_err(|_| "sequence length exceeds u32".to_string())?;
    let width = ATTN_HEADS * ATTN_HEAD_DIM;
    let factor = regime.query_factor();
    let q_values: Vec<f32> = activation(0x2122_2324_2526_2728, seq * width)
        .into_iter()
        .map(|value| value * factor)
        .collect();
    let k_values = activation(0x3132_3334_3536_3738, seq * width);
    let v_values = activation(0x4142_4344_4546_4748, seq * width);
    let q_buffer = shared_buffer(device, seq * width * 4);
    let k_buffer = shared_buffer(device, seq * width * 4);
    let v_buffer = shared_buffer(device, seq * width * 4);
    let o_buffer = shared_buffer(device, seq * width * 4);
    write_slice(&q_buffer, &q_values);
    write_slice(&k_buffer, &k_values);
    write_slice(&v_buffer, &v_values);
    let heads = ATTN_HEADS as u32;
    let params = AttentionParams {
        seq_len,
        heads,
        kv_heads: heads,
        window: 0,
        use_window: 0,
        scale: ATTN_SCALE,
        pad: [0; 2],
    };
    let (rows_per_group, threads) = if kernel == "attn_sg" {
        (32u64, 128u64)
    } else {
        (8u64, 256u64)
    };

    let encode = |encoder: &metal::ComputeCommandEncoderRef| {
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&q_buffer), 0);
        encoder.set_buffer(1, Some(&k_buffer), 0);
        encoder.set_buffer(2, Some(&v_buffer), 0);
        encoder.set_buffer(3, Some(&o_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<AttentionParams>() as u64,
            std::ptr::from_ref(&params).cast(),
        );
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(
                u64::from(heads),
                u64::from(seq_len).div_ceil(rows_per_group),
                1,
            ),
            metal::MTLSize::new(threads, 1, 1),
        );
    };

    for _ in 0..options.warmup {
        timed_dispatch(queue, encode)?;
    }
    let mut spans = Vec::with_capacity(options.reps);
    for _ in 0..options.reps {
        spans.push(timed_dispatch(queue, encode)?);
    }
    let (median, min, max) = span_stats(&spans)?;
    let flops = 4.0 * (seq as f64) * (seq as f64) * width as f64;
    let tflops = if median > 0.0 {
        flops / (median * 1.0e-6) / 1.0e12
    } else {
        0.0
    };
    let reps = spans.len();
    let regime_name = regime.name();
    println!(
        "@@attn kernel={kernel} s={seq} heads={heads} hd={ATTN_HEAD_DIM} regime={regime_name} reps={reps} median_us={median:.3} min_us={min:.3} max_us={max:.3} tflops={tflops:.4}"
    );

    let o = read_slice(&o_buffer, seq * width);
    let nonfinite = o.iter().filter(|value| !value.is_finite()).count();
    let (max_abs, max_rel) = attention_reference_error(&q_values, &k_values, &v_values, &o, seq);
    println!(
        "@@check kernel={kernel} s={seq} regime={regime_name} max_abs={max_abs:.6e} max_rel={max_rel:.6e} nonfinite={nonfinite}"
    );
    if nonfinite > 0 {
        eprintln!("{kernel} s={seq} {regime_name}: {nonfinite} output values are not finite");
        return Ok(false);
    }
    if kernel == "attn_sg" && max_rel > ATTN_SG_REL_BOUND {
        let bound = ATTN_SG_REL_BOUND;
        eprintln!(
            "{kernel} s={seq} {regime_name} relative error {max_rel:.6e} exceeds {bound:.1e}"
        );
        return Ok(false);
    }
    Ok(true)
}

/// Deterministic sample of (query row, head) pairs: tile-boundary rows (8-row
/// and 32-row blocks, 32-key tiles), the last rows, both edge heads, then
/// pseudo-random fill up to `SAMPLE_COUNT`.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn attention_samples(seq: usize) -> Vec<(usize, usize)> {
    use std::collections::BTreeSet;

    let mut pairs = BTreeSet::new();
    let rows = [
        0,
        7,
        8,
        31,
        32,
        33,
        63,
        64,
        seq / 2,
        seq.saturating_sub(33),
        seq.saturating_sub(32),
        seq.saturating_sub(2),
        seq - 1,
    ];
    for row in rows {
        for head in [0, ATTN_HEADS - 1] {
            if row < seq {
                pairs.insert((row, head));
            }
        }
    }
    let target = SAMPLE_COUNT.min(seq * ATTN_HEADS);
    let mut rng = Rng::new(0x5EED_A77E_0000_0064);
    while pairs.len() < target {
        let row = (rng.next_u64() as usize) % seq;
        let head = (rng.next_u64() as usize) % ATTN_HEADS;
        pairs.insert((row, head));
    }
    pairs.into_iter().collect()
}

/// f64 softmax attention for each sampled (row, head) pair over every key.
/// Returns `(max_abs, max_rel)`, where a pair's relative error is its largest
/// absolute difference over the head's outputs divided by its largest
/// `sum_j p_j |v_j|`, the magnitude the output is a weighted average of.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn attention_reference_error(q: &[f32], k: &[f32], v: &[f32], o: &[f32], seq: usize) -> (f64, f64) {
    let width = ATTN_HEADS * ATTN_HEAD_DIM;
    let scale = f64::from(ATTN_SCALE);
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut scores = vec![0.0f64; seq];
    for (row, head) in attention_samples(seq) {
        let base = head * ATTN_HEAD_DIM;
        let query = &q[row * width + base..row * width + base + ATTN_HEAD_DIM];
        let mut peak = f64::NEG_INFINITY;
        for (key, score) in scores.iter_mut().enumerate() {
            let keys = &k[key * width + base..key * width + base + ATTN_HEAD_DIM];
            let dot: f64 = query
                .iter()
                .zip(keys)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum();
            *score = dot * scale;
            peak = peak.max(*score);
        }
        let mut denominator = 0.0f64;
        for score in scores.iter_mut() {
            *score = (*score - peak).exp();
            denominator += *score;
        }
        let mut worst_difference = 0.0f64;
        let mut largest_magnitude = 0.0f64;
        for dim in 0..ATTN_HEAD_DIM {
            let mut reference = 0.0f64;
            let mut magnitude = 0.0f64;
            for (key, weight) in scores.iter().enumerate() {
                let value = f64::from(v[key * width + base + dim]);
                reference += weight * value;
                magnitude += weight * value.abs();
            }
            reference /= denominator;
            magnitude /= denominator;
            let difference = (f64::from(o[row * width + base + dim]) - reference).abs();
            worst_difference = worst_difference.max(difference);
            largest_magnitude = largest_magnitude.max(magnitude);
        }
        max_abs = max_abs.max(worst_difference);
        max_rel = max_rel.max(worst_difference / largest_magnitude.max(f64::MIN_POSITIVE));
    }
    (max_abs, max_rel)
}

/// Bring the GPU clock up before anything is timed: a fixed number of untimed
/// `gemm_f32_sg` dispatches at [`CLOCK_WARMUP_SHAPE`], each in its own command
/// buffer. The first and last spans are printed so a run records the ramp it
/// discarded.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn clock_warmup(device: &metal::Device, queue: &metal::CommandQueue) -> Result<(), String> {
    let pipeline = build_f32_sg_pipeline(device)?;
    let shape = CLOCK_WARMUP_SHAPE;
    let m = u32::try_from(shape.m).map_err(|_| "M exceeds u32 indexing".to_string())?;
    let n = u32::try_from(shape.n).map_err(|_| "N exceeds u32 indexing".to_string())?;
    let k = u32::try_from(shape.k).map_err(|_| "K exceeds u32 indexing".to_string())?;
    let x_buffer = shared_buffer(device, shape.m * shape.k * 4);
    let w_buffer = shared_buffer(device, shape.n * shape.k * 4);
    let y_buffer = shared_buffer(device, shape.m * shape.n * 4);
    write_slice(
        &x_buffer,
        &activation(0x6162_6364_6566_6768, shape.m * shape.k),
    );
    write_slice(
        &w_buffer,
        &activation(0x7172_7374_7576_7778, shape.n * shape.k),
    );
    let encode = |encoder: &metal::ComputeCommandEncoderRef| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&w_buffer), 0);
        encoder.set_buffer(2, Some(&y_buffer), 0);
        encoder.set_bytes(3, 4, std::ptr::from_ref(&m).cast());
        encoder.set_bytes(4, 4, std::ptr::from_ref(&n).cast());
        encoder.set_bytes(5, 4, std::ptr::from_ref(&k).cast());
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(u64::from(n).div_ceil(32), u64::from(m).div_ceil(64), 1),
            metal::MTLSize::new(32, 4, 1),
        );
    };
    let mut first = 0.0f64;
    let mut last = 0.0f64;
    for index in 0..CLOCK_WARMUP_DISPATCHES {
        let span = timed_dispatch(queue, encode)?;
        if index == 0 {
            first = span;
        }
        last = span;
    }
    let dispatches = CLOCK_WARMUP_DISPATCHES;
    println!(
        "@@warmup kernel=f32_sg m={m} n={n} k={k} dispatches={dispatches} first_us={first:.3} last_us={last:.3}"
    );
    Ok(())
}
