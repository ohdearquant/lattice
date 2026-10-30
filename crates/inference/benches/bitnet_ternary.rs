//! Synthetic BitNet-shaped scalar and explicit NEON ternary matvec baselines.
//!
//! Inputs are prepacked and prequantized outside the timed loop. This measures
//! the kernels, not a model load, activation quantization, or model throughput.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
#[cfg(target_arch = "aarch64")]
use lattice_compute::forward::bitnet_kernel::matvec_ternary_neon;
use lattice_compute::forward::bitnet_kernel::{matvec_ternary_scalar, packed_row_bytes};

struct Fixture {
    n: usize,
    k: usize,
    input: Vec<i8>,
    packed: Vec<u8>,
    alphas: Vec<f32>,
}

fn fixture(n: usize, k: usize) -> Fixture {
    let input = (0..k).map(|i| [-128, -127, 0, 127][i % 4]).collect();
    let packed = (0..n * packed_row_bytes(k))
        .map(|i| ((i * 73 + 0xe4) & 0xff) as u8)
        .collect();
    Fixture {
        n,
        k,
        input,
        packed,
        alphas: vec![1.0; n],
    }
}

fn scalar(f: &Fixture, output: &mut [f32]) {
    matvec_ternary_scalar(&f.input, 0.25, &f.packed, &f.alphas, f.n, f.k, output);
}

#[cfg(target_arch = "aarch64")]
fn neon(f: &Fixture, output: &mut [f32]) {
    // SAFETY: the selected benchmark checks NEON support, and the fixture
    // constructs exact n/k-sized buffers before this function is called.
    unsafe {
        matvec_ternary_neon(&f.input, 0.25, &f.packed, &f.alphas, f.n, f.k, output);
    }
}

fn bench_bitnet_ternary(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitnet_ternary");
    for (n, k) in [(2560usize, 2560usize), (6912, 2560)] {
        group.throughput(Throughput::Elements((n * k) as u64));
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("n{n}_k{k}")),
            &(n, k),
            |b, &(n, k)| {
                let f = fixture(n, k);
                let mut output = vec![f32::NAN; n];
                scalar(&f, &mut output);
                assert!(output.iter().all(|value| value.is_finite()));
                eprintln!("[BITNET_SCALAR_WITNESS] n={n} k={k} rows={}", output.len());
                b.iter(|| {
                    scalar(black_box(&f), black_box(&mut output));
                    black_box(&output);
                });
            },
        );
        // The NEON arm exists only on aarch64, so a full run of this target on
        // another architecture measures the scalar arm and nothing else.
        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(
            BenchmarkId::new("neon", format!("n{n}_k{k}")),
            &(n, k),
            |b, &(n, k)| {
                assert!(
                    std::arch::is_aarch64_feature_detected!("neon"),
                    "selected NEON benchmark requires NEON"
                );
                let f = fixture(n, k);
                let mut reference = vec![f32::NAN; n];
                let mut output = vec![f32::NAN; n];
                scalar(&f, &mut reference);
                neon(&f, &mut output);
                assert_eq!(output, reference, "NEON must match scalar before timing");
                eprintln!(
                    "[BITNET_NEON_BENCH_WITNESS] n={n} k={k} rows={}",
                    output.len()
                );
                b.iter(|| {
                    neon(black_box(&f), black_box(&mut output));
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_bitnet_ternary);
criterion_main!(benches);
