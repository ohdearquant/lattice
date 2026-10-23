use super::*;

#[test]
fn rms_norm_forward_roundtrip() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let w = vec![1.0f32; 4];
    let (normed, inv_rms) = rms_norm_forward(&x, &w, 1e-6);
    let mean_sq: f32 = x.iter().map(|xi| xi * xi).sum::<f32>() / 4.0;
    let expected_inv = 1.0 / (mean_sq + 1e-6f32).sqrt();
    assert!((inv_rms - expected_inv).abs() < 1e-6, "inv_rms mismatch");
    let expected_norm: Vec<f32> = x.iter().map(|xi| xi * expected_inv).collect();
    for (a, b) in normed.iter().zip(expected_norm.iter()) {
        assert!((a - b).abs() < 1e-5, "normed mismatch {a} vs {b}");
    }
}

#[test]
fn swiglu_forward_smoke() {
    let hidden = 2;
    let inter = 3;
    let x = vec![1.0f32, -0.5];
    let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
    let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
    let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let (out, gate_pre, up_pre) = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
    assert_eq!(out.len(), hidden);
    assert_eq!(gate_pre.len(), inter);
    assert_eq!(up_pre.len(), inter);
}

// ADR-080 C4 held finding: `swiglu_forward` previously truncated a too-short `x` via
// `.zip(x.iter())` silently instead of rejecting the shape like `matmul_bt` does.
#[test]
#[should_panic(expected = "a too short for m*k")]
fn swiglu_forward_rejects_short_activation() {
    let hidden = 2;
    let inter = 3;
    let x = vec![1.0f32]; // too short: needs hidden = 2
    let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
    let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
    let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let _ = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
}

#[test]
#[should_panic(expected = "b too short for n*k")]
fn swiglu_forward_rejects_short_w_gate() {
    let hidden = 2;
    let inter = 3;
    let x = vec![1.0f32, -0.5];
    let w_gate = vec![1.0f32, 0.0]; // too short: needs inter*hidden = 6
    let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
    let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let _ = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
}

#[test]
#[should_panic(expected = "b too short for n*k")]
fn swiglu_forward_rejects_short_w_down() {
    let hidden = 2;
    let inter = 3;
    let x = vec![1.0f32, -0.5];
    let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
    let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
    let w_down = vec![1.0f32, 0.0]; // too short: needs hidden*inter = 6
    let _ = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
}

#[test]
fn swiglu_forward_accepts_oversized_activation() {
    let hidden = 2;
    let inter = 3;
    let x = vec![1.0f32, -0.5, 99.0]; // oversized by 1
    let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
    let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
    let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let (out, gate_pre, up_pre) = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
    assert_eq!(out.len(), hidden);
    assert_eq!(gate_pre.len(), inter);
    assert_eq!(up_pre.len(), inter);
}

// Sequence-level forward vs the single-position `swiglu_forward` reference. `matmul_bt`
// reassociates the dot products, so parity is a relative tolerance, not bit-exact; the
// tolerance and error formula match the parity tests in `ops.rs`.
const PARITY_TOL: f64 = 1e-4;

fn xorshift_fill(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

fn rel_err(reference: &[f32], got: &[f32]) -> f64 {
    assert_eq!(reference.len(), got.len(), "length mismatch");
    let diff_sq: f64 = reference
        .iter()
        .zip(got.iter())
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    let norm_sq: f64 = reference.iter().map(|&a| (a as f64).powi(2)).sum();
    (diff_sq / norm_sq.max(1e-30)).sqrt()
}

/// Runs `swiglu_forward_seq` on rows `start..end` of a larger `[seq, hidden]` input and
/// compares every returned row with `swiglu_forward` on the same position.
fn assert_seq_matches_per_position(start: usize, end: usize) {
    // Odd, unequal dims: a transposed or swapped operand cannot pass by symmetry.
    let (seq, hidden, inter) = (11usize, 37usize, 53usize);
    // `xorshift_fill` forces the seed odd, so the seeds here are distinct odd numbers:
    // adjacent even/odd seeds would give gate and up the identical matrix, and a swap
    // of the two would then pass unnoticed.
    let x = xorshift_fill(101, seq * hidden, 1.0);
    let w_gate = xorshift_fill(103, inter * hidden, 0.3);
    let w_up = xorshift_fill(105, inter * hidden, 0.3);
    let w_down = xorshift_fill(107, hidden * inter, 0.3);

    let rows = end - start;
    let (out, gate_pre, up_pre) = swiglu_forward_seq(
        &x[start * hidden..end * hidden],
        &w_gate,
        &w_up,
        &w_down,
        rows,
        hidden,
        inter,
    );
    assert_eq!(out.len(), rows * hidden);
    assert_eq!(gate_pre.len(), rows * inter);
    assert_eq!(up_pre.len(), rows * inter);

    let mut ref_out = Vec::new();
    let mut ref_gate = Vec::new();
    let mut ref_up = Vec::new();
    for t in start..end {
        let (o, g, u) = swiglu_forward(
            &x[t * hidden..(t + 1) * hidden],
            &w_gate,
            &w_up,
            &w_down,
            hidden,
            inter,
        );
        ref_out.extend_from_slice(&o);
        ref_gate.extend_from_slice(&g);
        ref_up.extend_from_slice(&u);
    }

    assert!(
        rel_err(&ref_gate, &ref_up) > 0.1,
        "reference gate_pre and up_pre must differ, or a gate/up swap is undetectable"
    );
    for (name, reference, got) in [
        ("out", &ref_out, &out),
        ("gate_pre", &ref_gate, &gate_pre),
        ("up_pre", &ref_up, &up_pre),
    ] {
        assert!(
            reference.iter().any(|v| v.abs() > 1e-3),
            "{name}: reference is all ~0, comparison would be vacuous"
        );
        let err = rel_err(reference, got);
        eprintln!("swiglu_forward_seq {start}..{end} {name} rel_err={err:.2e}");
        assert!(
            err < PARITY_TOL,
            "swiglu_forward_seq {start}..{end} {name} vs per-position rel_err {err:.2e} >= {PARITY_TOL:.2e}"
        );
    }
}

#[test]
fn swiglu_forward_seq_parity_full_range() {
    assert_seq_matches_per_position(0, 11);
}

// The terminal-layer shape: a block that does not start at position 0 and stops short of
// the end of the sequence.
#[test]
fn swiglu_forward_seq_parity_range_not_starting_at_zero() {
    assert_seq_matches_per_position(4, 10);
}

#[test]
fn swiglu_forward_seq_parity_single_row_at_last_position() {
    assert_seq_matches_per_position(10, 11);
}

#[test]
fn swiglu_forward_seq_empty_block_returns_empty() {
    let (out, gate_pre, up_pre) = swiglu_forward_seq(&[], &[], &[], &[], 0, 4, 6);
    assert!(out.is_empty() && gate_pre.is_empty() && up_pre.is_empty());
}

#[test]
#[should_panic(expected = "a too short for m*k")]
fn swiglu_forward_seq_rejects_short_activation() {
    let (hidden, inter) = (2usize, 3usize);
    let x = vec![1.0f32, -0.5, 0.25]; // 2 rows need 4 values
    let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
    let w_up = vec![0.5f32; 6];
    let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let _ = swiglu_forward_seq(&x, &w_gate, &w_up, &w_down, 2, hidden, inter);
}

// Zero-width intermediate or hidden dimension: both reach `matmul_bt` with an empty
// dimension, which must produce correctly sized (empty or zero) outputs, not abort.
#[test]
fn swiglu_forward_seq_zero_intermediate_returns_zero_output() {
    let (rows, hidden) = (3usize, 4usize);
    let x = vec![0.5f32; rows * hidden];
    let (out, gate_pre, up_pre) = swiglu_forward_seq(&x, &[], &[], &[], rows, hidden, 0);
    assert_eq!(out, vec![0.0f32; rows * hidden]);
    assert!(gate_pre.is_empty() && up_pre.is_empty());
}

#[test]
fn swiglu_forward_seq_zero_hidden_returns_empty_output() {
    let (rows, inter) = (3usize, 5usize);
    let (out, gate_pre, up_pre) = swiglu_forward_seq(&[], &[], &[], &[], rows, 0, inter);
    assert!(out.is_empty());
    assert_eq!(gate_pre, vec![0.0f32; rows * inter]);
    assert_eq!(up_pre, vec![0.0f32; rows * inter]);
}
