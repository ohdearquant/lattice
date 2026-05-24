//! Performance regression tests for lattice-inference.
//!
//! These are pass/fail time-bounded assertions, not benchmarks.
//! Bounds are deliberately generous (50-100x over expected) to remain
//! stable across debug builds and CI variance.

use std::time::Instant;

#[test]
fn matmul_bt_not_catastrophically_slow() {
    // matmul_bt at (m=4, k=512, n=512): 100 iterations must finish in < 5s (debug).
    // Expected scalar path ~500µs/iter -> 100 iters ~50ms; 5s gives 100x slack.
    let m = 4usize;
    let k = 512usize;
    let n = 512usize;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; n * k]; // matmul_bt: B is (n, k) row-major
    let mut c = vec![0.0f32; m * n];

    let start = Instant::now();
    for _ in 0..100 {
        lattice_inference::forward::matmul_bt(&a, &b, &mut c, m, k, n);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 5,
        "matmul_bt regression: 100 iters at (m=4,k=512,n=512) took {elapsed:?}, expected < 5s"
    );
}

#[test]
fn matmul_not_catastrophically_slow() {
    // matmul (A * B) at (m=4, k=128, n=128): 100 iterations must finish in < 5s.
    let m = 4usize;
    let k = 128usize;
    let n = 128usize;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let start = Instant::now();
    for _ in 0..100 {
        let _c = lattice_inference::forward::matmul(&a, &b, m, k, n);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 5,
        "matmul regression: 100 iters at (m=4,k=128,n=128) took {elapsed:?}, expected < 5s"
    );
}

#[test]
fn rms_norm_not_catastrophically_slow() {
    // rms_norm at hidden=512, 1000 iterations must finish in < 2s.
    // Expected: ~5µs/iter -> 1000 iters ~5ms; 2s gives 400x slack.
    let hidden = 512usize;
    let num_tokens = 4usize;
    let mut x = vec![0.5f32; num_tokens * hidden];
    let gamma = vec![1.0f32; hidden];

    let start = Instant::now();
    for _ in 0..1000 {
        lattice_inference::forward::rms_norm(&mut x, &gamma, hidden, 1e-6);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "rms_norm regression: 1000 iters at hidden=512 took {elapsed:?}, expected < 2s"
    );
}

#[test]
fn silu_inplace_not_catastrophically_slow() {
    // silu_inplace at len=2048, 2000 iterations must finish in < 2s.
    let mut x = vec![0.5f32; 2048];

    let start = Instant::now();
    for _ in 0..2000 {
        lattice_inference::forward::silu_inplace(&mut x);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "silu_inplace regression: 2000 iters at len=2048 took {elapsed:?}, expected < 2s"
    );
}

#[test]
fn elementwise_mul_not_catastrophically_slow() {
    // elementwise_mul at len=2048, 2000 iterations must finish in < 2s.
    let mut a = vec![2.0f32; 2048];
    let b = vec![0.5f32; 2048];

    let start = Instant::now();
    for _ in 0..2000 {
        lattice_inference::forward::elementwise_mul(&mut a, &b);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "elementwise_mul regression: 2000 iters at len=2048 took {elapsed:?}, expected < 2s"
    );
}
