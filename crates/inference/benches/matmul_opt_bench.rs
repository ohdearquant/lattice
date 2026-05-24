use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::forward::cpu::matmul_bt;

fn bench_matmul_bt(c: &mut Criterion) {
    let sizes: &[(usize, usize, usize, &str)] = &[
        (1, 4096, 4096, "m1_k4096_n4096"),
        (32, 4096, 4096, "m32_k4096_n4096"),
        (128, 4096, 11008, "m128_k4096_n11008"),
    ];

    let mut group = c.benchmark_group("matmul_bt");

    for &(m, k, n, label) in sizes {
        let elements = (2 * m * k * n) as u64;
        group.throughput(Throughput::Elements(elements));

        group.bench_with_input(
            BenchmarkId::new("dispatch", label),
            &(m, k, n),
            |bench, &(m, k, n)| {
                let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
                let b: Vec<f32> = (0..n * k).map(|i| (i % 89) as f32 * 0.01).collect();
                let mut c_out = vec![0.0f32; m * n];
                bench.iter(|| {
                    matmul_bt(black_box(&a), black_box(&b), black_box(&mut c_out), m, k, n);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_matmul_bt);
criterion_main!(benches);
