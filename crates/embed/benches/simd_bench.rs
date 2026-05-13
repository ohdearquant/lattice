//! SIMD benchmark suite for lattice-embed.
//!
//! Validates SOTA performance from lion.cog migration:
//! - NEON float32: ~75ns for 384-dim cosine_similarity
//! - NEON int8: ~29ns for 384-dim (22.6x vs scalar)
//! - Throughput: >10 Gelem/s with int8

use std::hint::black_box;
use std::time::{Duration, Instant};

use lattice_embed::simd::{
    QuantizedVector, cosine_similarity, dot_product, euclidean_distance, normalize, simd_config,
};

const DIM: usize = 384;
const ITERATIONS: usize = 10_000;
const WARMUP: usize = 1_000;

fn generate_random_vector(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..dim)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench_cosine_similarity_f32() -> (Duration, f64) {
    let a = generate_random_vector(DIM, 42);
    let b = generate_random_vector(DIM, 123);

    // Warmup
    for _ in 0..WARMUP {
        black_box(cosine_similarity(black_box(&a), black_box(&b)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(cosine_similarity(black_box(&a), black_box(&b)));
    }
    let elapsed = start.elapsed();

    let per_op = elapsed / ITERATIONS as u32;
    let throughput = (DIM as f64 * 3.0 * ITERATIONS as f64) / elapsed.as_secs_f64() / 1e9;
    (per_op, throughput)
}

fn bench_dot_product_f32() -> (Duration, f64) {
    let a = generate_random_vector(DIM, 42);
    let b = generate_random_vector(DIM, 123);

    // Warmup
    for _ in 0..WARMUP {
        black_box(dot_product(black_box(&a), black_box(&b)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(dot_product(black_box(&a), black_box(&b)));
    }
    let elapsed = start.elapsed();

    let per_op = elapsed / ITERATIONS as u32;
    let throughput = (DIM as f64 * 2.0 * ITERATIONS as f64) / elapsed.as_secs_f64() / 1e9;
    (per_op, throughput)
}

fn bench_normalize_f32() -> Duration {
    let original = generate_random_vector(DIM, 42);
    // Pre-allocate working buffer to avoid heap allocation in hot loop
    let mut v = original.clone();

    // Warmup
    for _ in 0..WARMUP {
        v.copy_from_slice(&original);
        normalize(black_box(&mut v));
        black_box(&v);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v.copy_from_slice(&original);
        normalize(black_box(&mut v));
        black_box(&v);
    }
    let elapsed = start.elapsed();

    elapsed / ITERATIONS as u32
}

fn bench_euclidean_distance_f32() -> Duration {
    let a = generate_random_vector(DIM, 42);
    let b = generate_random_vector(DIM, 123);

    // Warmup
    for _ in 0..WARMUP {
        black_box(euclidean_distance(black_box(&a), black_box(&b)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(euclidean_distance(black_box(&a), black_box(&b)));
    }
    let elapsed = start.elapsed();

    elapsed / ITERATIONS as u32
}

fn bench_dot_product_i8() -> (Duration, f64) {
    let a = generate_random_vector(DIM, 42);
    let b = generate_random_vector(DIM, 123);
    let a_q = QuantizedVector::from_f32(&a);
    let b_q = QuantizedVector::from_f32(&b);

    // Warmup
    for _ in 0..WARMUP {
        black_box(a_q.dot_product(black_box(&b_q)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(a_q.dot_product(black_box(&b_q)));
    }
    let elapsed = start.elapsed();

    let per_op = elapsed / ITERATIONS as u32;
    let throughput = (DIM as f64 * 2.0 * ITERATIONS as f64) / elapsed.as_secs_f64() / 1e9;
    (per_op, throughput)
}

fn bench_cosine_similarity_i8() -> (Duration, f64) {
    let a = generate_random_vector(DIM, 42);
    let b = generate_random_vector(DIM, 123);
    let a_q = QuantizedVector::from_f32(&a);
    let b_q = QuantizedVector::from_f32(&b);

    // Warmup
    for _ in 0..WARMUP {
        black_box(a_q.cosine_similarity(black_box(&b_q)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(a_q.cosine_similarity(black_box(&b_q)));
    }
    let elapsed = start.elapsed();

    let per_op = elapsed / ITERATIONS as u32;
    let throughput = (DIM as f64 * 2.0 * ITERATIONS as f64) / elapsed.as_secs_f64() / 1e9;
    (per_op, throughput)
}

fn bench_scalar_cosine() -> Duration {
    let a = generate_random_vector(DIM, 42);
    let b = generate_random_vector(DIM, 123);

    // Warmup
    for _ in 0..WARMUP {
        let dot: f32 = black_box(&a)
            .iter()
            .zip(black_box(&b).iter())
            .map(|(x, y)| x * y)
            .sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        black_box(dot / (norm_a * norm_b));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let dot: f32 = black_box(&a)
            .iter()
            .zip(black_box(&b).iter())
            .map(|(x, y)| x * y)
            .sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        black_box(dot / (norm_a * norm_b));
    }
    let elapsed = start.elapsed();

    elapsed / ITERATIONS as u32
}

/// Calculate GFLOPS for an operation.
/// - dot_product: 2 FLOPs/element (multiply + add)
/// - cosine_similarity: 6 FLOPs/element (3 FMAs for dot, norm_a, norm_b)
/// - normalize: 3 FLOPs/element (square + add + multiply by inv_norm)
/// - euclidean: 3 FLOPs/element (sub + square + add)
fn gflops(dim: usize, flops_per_elem: usize, time_ns: u64) -> f64 {
    let total_flops = dim * flops_per_elem;
    total_flops as f64 / time_ns as f64
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║     lattice-embed SIMD Performance Benchmark                          ║");
    println!("║     Target: lion.cog SOTA (75ns NEON f32, 29ns i8)                       ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Detect SIMD capabilities
    let config = simd_config();
    println!("SIMD Configuration:");
    println!("  AVX2:        {}", config.avx2_enabled);
    println!("  FMA:         {}", config.fma_enabled);
    println!("  AVX512-VNNI: {}", config.avx512vnni_enabled);
    println!("  NEON:        {}", config.neon_enabled);
    println!();

    println!("Benchmark Parameters:");
    println!("  Vector dimension: {DIM}");
    println!("  Iterations:       {ITERATIONS}");
    println!("  Warmup:           {WARMUP}");
    println!();

    // Note about cosine similarity
    println!("NOTE: cosine_similarity computes full cosine (dot + both norms) in a single pass.");
    println!("      For pre-normalized vectors, use dot_product directly (~30% faster).");
    println!();

    // Run benchmarks
    println!("┌─────────────────────────────┬───────────┬─────────────┬──────────┬────────────┐");
    println!("│ Operation                   │ Time/op   │ Throughput  │ GFLOPS   │ Target     │");
    println!("├─────────────────────────────┼───────────┼─────────────┼──────────┼────────────┤");

    // Scalar baseline
    let scalar_time = bench_scalar_cosine();
    let scalar_gflops = gflops(DIM, 6, scalar_time.as_nanos() as u64);
    println!(
        "│ cosine (scalar, baseline)   │ {:>7}  │      -      │ {:>5.1}    │ ~650ns     │",
        format!("{:?}", scalar_time),
        scalar_gflops
    );

    // Float32 SIMD - full cosine (single-pass dot + norms)
    let (cos_time, cos_tp) = bench_cosine_similarity_f32();
    let cos_gflops = gflops(DIM, 6, cos_time.as_nanos() as u64);
    let speedup_vs_scalar = scalar_time.as_nanos() as f64 / cos_time.as_nanos() as f64;
    println!(
        "│ cosine (SIMD, full)         │ {:>7}  │ {:>5.1} Gelem/s│ {:>5.1}    │ <90ns      │",
        format!("{:?}", cos_time),
        cos_tp,
        cos_gflops
    );

    // Dot product - for pre-normalized vectors
    let (dot_time, dot_tp) = bench_dot_product_f32();
    let dot_gflops = gflops(DIM, 2, dot_time.as_nanos() as u64);
    println!(
        "│ dot_product (pre-normalized)│ {:>7}  │ {:>5.1} Gelem/s│ {:>5.1}    │ <35ns      │",
        format!("{:?}", dot_time),
        dot_tp,
        dot_gflops
    );

    let norm_time = bench_normalize_f32();
    let norm_gflops = gflops(DIM, 3, norm_time.as_nanos() as u64);
    println!(
        "│ normalize (SIMD)            │ {:>7}  │      -      │ {:>5.1}    │ <60ns      │",
        format!("{:?}", norm_time),
        norm_gflops
    );

    let eucl_time = bench_euclidean_distance_f32();
    let eucl_gflops = gflops(DIM, 3, eucl_time.as_nanos() as u64);
    println!(
        "│ euclidean_dist (SIMD)       │ {:>7}  │      -      │ {:>5.1}    │ ~90ns      │",
        format!("{:?}", eucl_time),
        eucl_gflops
    );

    println!("├─────────────────────────────┼───────────┼─────────────┼──────────┼────────────┤");

    // Int8 quantized
    let (i8_dot_time, i8_dot_tp) = bench_dot_product_i8();
    let i8_dot_gflops = gflops(DIM, 2, i8_dot_time.as_nanos() as u64);
    println!(
        "│ dot_product (int8)          │ {:>7}  │ {:>5.1} Gelem/s│ {:>5.1}    │ <30ns      │",
        format!("{:?}", i8_dot_time),
        i8_dot_tp,
        i8_dot_gflops
    );

    let (i8_cos_time, i8_cos_tp) = bench_cosine_similarity_i8();
    let i8_cos_gflops = gflops(DIM, 6, i8_cos_time.as_nanos() as u64);
    println!(
        "│ cosine (int8, pre-quant)    │ {:>7}  │ {:>5.1} Gelem/s│ {:>5.1}    │ <35ns      │",
        format!("{:?}", i8_cos_time),
        i8_cos_tp,
        i8_cos_gflops
    );

    println!("└─────────────────────────────┴───────────┴─────────────┴──────────┴────────────┘");

    // Summary
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║ SPEEDUP SUMMARY                                                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Scalar → SIMD float32 (full cosine):  {speedup_vs_scalar:>5.1}x                              ║"
    );

    let i8_vs_scalar = scalar_time.as_nanos() as f64 / i8_cos_time.as_nanos() as f64;
    println!(
        "║ Scalar → SIMD int8:                   {i8_vs_scalar:>5.1}x                              ║"
    );

    let i8_vs_f32 = cos_time.as_nanos() as f64 / i8_cos_time.as_nanos() as f64;
    println!(
        "║ Float32 → Int8:                       {i8_vs_f32:>5.1}x                              ║"
    );
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Validation against lion.cog targets
    println!();
    println!("VALIDATION (lion.cog SOTA targets):");

    let cos_pass = cos_time.as_nanos() <= 90;
    let i8_pass = i8_cos_time.as_nanos() <= 35;
    let speedup_pass = i8_vs_scalar >= 15.0;

    println!(
        "  ✓ cosine_similarity SIMD <= 90ns:  {} ({:?})",
        if cos_pass { "PASS" } else { "FAIL" },
        cos_time
    );
    println!(
        "  ✓ cosine_similarity int8 <= 35ns:  {} ({:?})",
        if i8_pass { "PASS" } else { "FAIL" },
        i8_cos_time
    );
    println!(
        "  ✓ Total speedup >= 15x:            {} ({:.1}x)",
        if speedup_pass { "PASS" } else { "FAIL" },
        i8_vs_scalar
    );

    println!();
    if cos_pass && i8_pass && speedup_pass {
        println!("🚀 ALL SOTA PERFORMANCE TARGETS MET!");
    } else {
        println!("⚠️  Some targets not met - check individual results above");
    }
}
