//! Host-side f32 reference oracle for the chunkwise WY/DPLR Gated-DeltaNet
//! (GDN) prefill recurrence.
//!
//! This is a line-by-line transcription of a validated NumPy reference
//! (`sequential_gdn` / `chunkwise_gdn`, see
//! `tests/fixtures/gdn_chunk/generate.py`) that already asserts chunkwise ==
//! sequential to <= 1e-5. It exists purely as a parity oracle: the B=64 and
//! B=128 Metal chunked-prefill kernels are validated against it (and,
//! transitively, against the committed NumPy-generated fixtures), never a
//! production path itself.
//!
//! Matrices are flat row-major `Vec<f32>` / `&[f32]` with explicit
//! `(rows, cols)` passed alongside — no `ndarray`/`nalgebra` dependency.
//!
//! Lattice's GDN state `H` is value-major, `[d_v, d_k]` (Lattice's
//! `H = S^T` relative to the more common key-major convention). Every
//! function below keeps that layout end to end; nothing here transposes to
//! key-major.
//!
//! Two DISTINCT `tril` semantics appear in `chunkwise_gdn` and must not be
//! conflated: `a_plain` / `a_gamma` use strictly-lower (`k=-1`, diagonal
//! excluded — the unit diagonal comes from the `I +` in the NumPy
//! reference), while `lqk` uses lower-INCLUDING-diagonal (`k=0`).

#[inline]
fn idx(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

/// Row L2 normalization: `x / sqrt(sum(x^2, axis=1) + eps)`.
///
/// `x` is `rows x cols` row-major. Returns a new `rows x cols` buffer.
///
/// ADR-080 C1 fail-closed (#850): a row with a non-finite lane makes `sum_sq`
/// non-finite (mirrors `attention::gdn::l2_normalize_vec`); such a row is left at
/// its zero-initialized default rather than divided through, matching the
/// direct-assignment (never multiply/divide-through-a-poisoned-value) contract.
pub fn l2_normalize_rows(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        if !sum_sq.is_finite() {
            // out row already zero-initialized; leave it as the fail-closed zero row.
            continue;
        }
        let denom = (sum_sq + eps).sqrt();
        for c in 0..cols {
            out[idx(r, c, cols)] = row[c] / denom;
        }
    }
    out
}

/// Forward substitution for a unit-lower-triangular system `A X = B`.
///
/// `a` is `c x c` row-major; only its strictly-lower entries (`j < i`) are
/// read — the diagonal is implicitly 1 and is never referenced. `b` is
/// `c x d` row-major. Returns `x`, `c x d` row-major.
pub fn solve_unit_lower(a: &[f32], b: &[f32], c: usize, d: usize) -> Vec<f32> {
    let mut x = b.to_vec();
    for i in 1..c {
        for col in 0..d {
            let mut acc = 0.0f32;
            for (j, xj) in x.chunks_exact(d).enumerate().take(i) {
                acc += a[idx(i, j, c)] * xj[col];
            }
            x[idx(i, col, d)] -= acc;
        }
    }
    x
}

/// Sequential gated-delta recurrence in Lattice's value-major `H = S^T`
/// layout.
///
/// Shapes: `q`, `k` are `[t, dk]`; `v` is `[t, dv]`; `beta`, `alpha` are
/// `[t]`; `h0` is `[dv, dk]` (value-major state).
///
/// Recurrence, per token `i`:
/// ```text
/// h_decayed = alpha[i] * h                 // [dv, dk]
/// kv        = h_decayed @ k[i]             // [dv]
/// r         = beta[i] * (v[i] - kv)        // [dv]
/// h         = h_decayed + outer(r, k[i])   // [dv, dk]
/// out[i]    = (h @ q[i]) * scale           // [dv]
/// ```
///
/// Returns `(out [t, dv], h [dv, dk])`.
#[allow(clippy::too_many_arguments)]
pub fn sequential_gdn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    alpha: &[f32],
    h0: &[f32],
    t: usize,
    dk: usize,
    dv: usize,
    apply_scale: bool,
) -> (Vec<f32>, Vec<f32>) {
    let scale = if apply_scale {
        1.0 / (dk as f32).sqrt()
    } else {
        1.0
    };
    let mut h = h0.to_vec(); // [dv, dk]
    let mut out = vec![0.0f32; t * dv];

    for i in 0..t {
        let qi = &q[i * dk..(i + 1) * dk];
        let ki = &k[i * dk..(i + 1) * dk];
        let vi = &v[i * dv..(i + 1) * dv];
        let a = alpha[i];
        let b = beta[i];

        // h_decayed = alpha[i] * h
        let mut h_decayed = vec![0.0f32; dv * dk];
        for (hd, hv) in h_decayed.iter_mut().zip(h.iter()) {
            *hd = a * hv;
        }

        // kv = h_decayed @ k[i]  -> [dv]
        let mut kv = vec![0.0f32; dv];
        for row in 0..dv {
            let hrow = &h_decayed[row * dk..(row + 1) * dk];
            kv[row] = hrow.iter().zip(ki).map(|(hv, kv_)| hv * kv_).sum();
        }

        // r = beta[i] * (v[i] - kv)
        let mut r = vec![0.0f32; dv];
        for row in 0..dv {
            r[row] = b * (vi[row] - kv[row]);
        }

        // h = h_decayed + outer(r, k[i])
        for row in 0..dv {
            for col in 0..dk {
                h_decayed[idx(row, col, dk)] += r[row] * ki[col];
            }
        }
        h = h_decayed;

        // out[i] = (h @ q[i]) * scale
        let out_row = &mut out[i * dv..(i + 1) * dv];
        for (row, out_val) in out_row.iter_mut().enumerate() {
            let hrow = &h[row * dk..(row + 1) * dk];
            let dot: f32 = hrow.iter().zip(qi).map(|(hv, qv)| hv * qv).sum();
            *out_val = dot * scale;
        }
    }

    (out, h)
}

/// Exact chunkwise WY/DPLR formulation of the same recurrence, processing
/// `chunk_size` tokens at a time. See module docs for the value-major `H`
/// convention and the two distinct `tril` semantics.
///
/// For one chunk of `c` tokens (`gamma_j = prod_{r<=j} alpha_r`), every decay
/// factor below is algebraically a ratio `gamma_i / gamma_j` (i>=j) or a bare
/// `gamma_i`, all bounded in `[0, 1]` since `alpha` is bounded in `(0, 1]`.
/// Rather than materializing `gamma` and `gamma_inv = 1/gamma` in linear
/// space (which underflows to `0.0` and `inf` respectively within a single
/// chunk on strongly-decaying real heads, poisoning the chunk with `0*inf =
/// NaN`), every factor is computed as `exp` of a difference of the
/// log-cumulative decay `gamma_log_i = sum_{r<=i} ln(alpha_r)` — the same
/// log-space formulation the Metal kernel already uses. This is a pure
/// numerical-stability reformulation: the values computed are identical (up
/// to f32 rounding) to the linear-space ratios; nothing about the
/// recurrence's math changes.
/// ```text
/// g       = k_c @ k_c^T                                    // [c, c]
/// a_plain = I + tril(diag(beta_c) * g, k=-1)                // strictly lower
/// w       = solve_unit_lower(a_plain, diag(beta_c) @ k_c)   // [c, dk]
/// a_gamma = I + tril(diag(beta_c)*g * exp(gamma_log[:,None] - gamma_log[None,:]), k=-1)
/// u       = solve_unit_lower(a_gamma, diag(beta_c) @ v_c)   // [c, dv]
/// r       = u - exp(gamma_log)[:,None] * (w @ h^T)          // [c, dv]
/// lqk     = tril((q_c @ k_c^T) * exp(gamma_log[:,None] - gamma_log[None,:]), k=0)  // includes diag
/// out_c   = (exp(gamma_log)[:,None] * (q_c @ h^T) + lqk @ r) * scale
/// h       = exp(gamma_log_end) * h + r^T @ (exp(gamma_log_end - gamma_log)[:,None] * k_c)
/// ```
///
/// Returns `(out [t_total, dv], h [dv, dk])`.
#[allow(clippy::too_many_arguments)]
pub fn chunkwise_gdn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    alpha: &[f32],
    h0: &[f32],
    t_total: usize,
    dk: usize,
    dv: usize,
    chunk_size: usize,
    apply_scale: bool,
) -> (Vec<f32>, Vec<f32>) {
    assert!(chunk_size > 0, "chunk_size must be positive");
    let scale = if apply_scale {
        1.0 / (dk as f32).sqrt()
    } else {
        1.0
    };
    let mut h = h0.to_vec(); // [dv, dk]
    let mut out = vec![0.0f32; t_total * dv];

    let mut start = 0usize;
    while start < t_total {
        let end = (start + chunk_size).min(t_total);
        let c = end - start;

        let q_c = &q[start * dk..end * dk];
        let k_c = &k[start * dk..end * dk];
        let v_c = &v[start * dv..end * dv];
        let b_c = &beta[start..end];
        let a_c = &alpha[start..end];

        // gamma_log[i] = sum_{r<=i} ln(alpha_c[r]) (log-space cumulative decay).
        // `alpha_c[i]` is always in `(0, 1]` by construction (a decay gate),
        // so `ln` is finite; the `max(f32::MIN_POSITIVE, ..)` guard only
        // fires on a degenerate exact-zero input and keeps `ln` finite (near
        // the kernel's own -88 log-alpha floor) instead of producing -inf.
        let mut gamma_log = vec![0.0f32; c];
        let mut running_log = 0.0f32;
        for (i, gl) in gamma_log.iter_mut().enumerate() {
            running_log += a_c[i].max(f32::MIN_POSITIVE).ln();
            *gl = running_log;
        }
        let gamma_log_end = gamma_log[c - 1];

        // g = k_c @ k_c^T  -> [c, c]
        let mut g = vec![0.0f32; c * c];
        for i in 0..c {
            let ki = &k_c[i * dk..(i + 1) * dk];
            for j in 0..c {
                let kj = &k_c[j * dk..(j + 1) * dk];
                g[idx(i, j, c)] = ki.iter().zip(kj).map(|(a, b)| a * b).sum();
            }
        }

        // a_plain = I + tril(diag(beta_c) * g, k=-1)  (STRICTLY lower; diag
        // is implicit-1 from the `I +`, never written explicitly here since
        // solve_unit_lower never reads the diagonal — but we still set it
        // to 1.0 to keep `a_plain` a faithful, inspectable transcription).
        let mut a_plain = vec![0.0f32; c * c];
        for i in 0..c {
            a_plain[idx(i, i, c)] = 1.0;
            for j in 0..i {
                a_plain[idx(i, j, c)] = b_c[i] * g[idx(i, j, c)];
            }
        }

        // rhs_k = beta_c[:,None] * k_c ; rhs_v = beta_c[:,None] * v_c
        let mut rhs_k = vec![0.0f32; c * dk];
        for i in 0..c {
            for col in 0..dk {
                rhs_k[idx(i, col, dk)] = b_c[i] * k_c[idx(i, col, dk)];
            }
        }
        let mut rhs_v = vec![0.0f32; c * dv];
        for i in 0..c {
            for col in 0..dv {
                rhs_v[idx(i, col, dv)] = b_c[i] * v_c[idx(i, col, dv)];
            }
        }

        let w = solve_unit_lower(&a_plain, &rhs_k, c, dk); // [c, dk]

        // a_gamma = I + tril((diag(beta_c)*g) * exp(gamma_log[i]-gamma_log[j]), k=-1)
        // (STRICTLY lower, same as a_plain — NOT the k=0 semantics used by lqk below.)
        // exp(gamma_log[i]-gamma_log[j]) replaces the linear-space
        // gamma[i]*gamma_inv[j]; for i>j this is the same ratio gamma_i/gamma_j
        // bounded in (0,1], computed without ever forming 1/gamma.
        let mut a_gamma = vec![0.0f32; c * c];
        for i in 0..c {
            a_gamma[idx(i, i, c)] = 1.0;
            for j in 0..i {
                a_gamma[idx(i, j, c)] =
                    b_c[i] * g[idx(i, j, c)] * (gamma_log[i] - gamma_log[j]).exp();
            }
        }
        let u = solve_unit_lower(&a_gamma, &rhs_v, c, dv); // [c, dv]

        // w_h0 = w @ h^T  -> [c, dv]  (h is [dv, dk])
        let mut w_h0 = vec![0.0f32; c * dv];
        for i in 0..c {
            let wi = &w[i * dk..(i + 1) * dk];
            for row in 0..dv {
                let hrow = &h[row * dk..(row + 1) * dk];
                w_h0[idx(i, row, dv)] = wi.iter().zip(hrow).map(|(a, b)| a * b).sum();
            }
        }

        // r = u - exp(gamma_log)[:,None] * w_h0
        let mut r = vec![0.0f32; c * dv];
        for i in 0..c {
            let gamma_i = gamma_log[i].exp();
            for col in 0..dv {
                r[idx(i, col, dv)] = u[idx(i, col, dv)] - gamma_i * w_h0[idx(i, col, dv)];
            }
        }

        // q_h0 = q_c @ h^T  -> [c, dv]
        let mut q_h0 = vec![0.0f32; c * dv];
        for i in 0..c {
            let qi = &q_c[i * dk..(i + 1) * dk];
            for row in 0..dv {
                let hrow = &h[row * dk..(row + 1) * dk];
                q_h0[idx(i, row, dv)] = qi.iter().zip(hrow).map(|(a, b)| a * b).sum();
            }
        }

        // qk = q_c @ k_c^T  -> [c, c]
        let mut qk = vec![0.0f32; c * c];
        for i in 0..c {
            let qi = &q_c[i * dk..(i + 1) * dk];
            for j in 0..c {
                let kj = &k_c[j * dk..(j + 1) * dk];
                qk[idx(i, j, c)] = qi.iter().zip(kj).map(|(a, b)| a * b).sum();
            }
        }

        // lqk = tril(qk * exp(gamma_log[i]-gamma_log[j]), k=0)
        // NOTE: k=0 — this INCLUDES the diagonal, unlike a_plain/a_gamma above
        // (i==j => exp(0) == 1, matching the linear-space gamma[i]/gamma[i]).
        let mut lqk = vec![0.0f32; c * c];
        for i in 0..c {
            for j in 0..=i {
                lqk[idx(i, j, c)] = qk[idx(i, j, c)] * (gamma_log[i] - gamma_log[j]).exp();
            }
        }

        // out[start:end] = (exp(gamma_log)[:,None]*q_h0 + lqk @ r) * scale
        for i in 0..c {
            let mut lqk_r_row = vec![0.0f32; dv];
            for j in 0..=i {
                let lij = lqk[idx(i, j, c)];
                for col in 0..dv {
                    lqk_r_row[col] += lij * r[idx(j, col, dv)];
                }
            }
            let gamma_i = gamma_log[i].exp();
            for col in 0..dv {
                let val = gamma_i * q_h0[idx(i, col, dv)] + lqk_r_row[col];
                out[(start + i) * dv + col] = val * scale;
            }
        }

        // k_right = exp(gamma_log_end - gamma_log)[:,None] * k_c
        let mut k_right = vec![0.0f32; c * dk];
        for i in 0..c {
            let scale_i = (gamma_log_end - gamma_log[i]).exp();
            for col in 0..dk {
                k_right[idx(i, col, dk)] = scale_i * k_c[idx(i, col, dk)];
            }
        }

        // h = exp(gamma_log_end) * h + r^T @ k_right  -> [dv, dk]
        let gamma_end = gamma_log_end.exp();
        let mut h_next = vec![0.0f32; dv * dk];
        for (hn, hv) in h_next.iter_mut().zip(h.iter()) {
            *hn = gamma_end * hv;
        }
        for i in 0..c {
            for row in 0..dv {
                let rv = r[idx(i, row, dv)];
                for col in 0..dk {
                    h_next[idx(row, col, dk)] += rv * k_right[idx(i, col, dk)];
                }
            }
        }
        h = h_next;

        start = end;
    }

    (out, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::path::PathBuf;

    /// Fixture format written by `tests/fixtures/gdn_chunk/generate.py`.
    #[derive(Deserialize)]
    struct Fixture {
        length: usize,
        dk: usize,
        dv: usize,
        chunk_size: usize,
        q: Vec<Vec<f32>>,
        k: Vec<Vec<f32>>,
        v: Vec<Vec<f32>>,
        beta: Vec<f32>,
        alpha: Vec<f32>,
        h0: Vec<Vec<f32>>,
        out_seq: Vec<Vec<f32>>,
        h_seq: Vec<Vec<f32>>,
        out_chk: Vec<Vec<f32>>,
        h_chk: Vec<Vec<f32>>,
    }

    fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("gdn_chunk")
    }

    fn load_fixture(name: &str) -> Fixture {
        let path = fixtures_dir().join(name);
        let data = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        serde_json::from_str(&data)
            .unwrap_or_else(|e| panic!("bad JSON in {}: {e}", path.display()))
    }

    /// NaN/Inf-honest `max|a-b|`. A plain `.fold(0.0, f32::max)` silently drops
    /// a NaN operand (IEEE `maxNum` returns the non-NaN side), so a
    /// catastrophically non-finite output can read as a clean `0.0` and slip
    /// through a `<= TOL` tolerance gate. This surfaces any non-finite
    /// difference instead, so the gate fails closed on it. (This is exactly the
    /// failure the strong-decay fixture exists to catch: the pre-fix linear
    /// oracle produced NaN, which a max-fold gate reported as `0.0`.)
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
        let mut max = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if !d.is_finite() {
                return d;
            }
            if d > max {
                max = d;
            }
        }
        max
    }

    fn count_nonfinite(v: &[f32]) -> usize {
        v.iter().filter(|x| !x.is_finite()).count()
    }

    const TOL: f32 = 1.0e-5;

    fn run_case(fixture_name: &str) {
        let fx = load_fixture(fixture_name);
        let q = flatten(&fx.q);
        let k = flatten(&fx.k);
        let v = flatten(&fx.v);
        let h0 = flatten(&fx.h0);
        let expected_out_seq = flatten(&fx.out_seq);
        let expected_h_seq = flatten(&fx.h_seq);
        let expected_out_chk = flatten(&fx.out_chk);
        let expected_h_chk = flatten(&fx.h_chk);

        let (rust_out_seq, rust_h_seq) = sequential_gdn(
            &q, &k, &v, &fx.beta, &fx.alpha, &h0, fx.length, fx.dk, fx.dv, true,
        );
        let (rust_out_chk, rust_h_chk) = chunkwise_gdn(
            &q,
            &k,
            &v,
            &fx.beta,
            &fx.alpha,
            &h0,
            fx.length,
            fx.dk,
            fx.dv,
            fx.chunk_size,
            true,
        );

        for (label, buf) in [
            ("rust_out_seq", &rust_out_seq),
            ("rust_h_seq", &rust_h_seq),
            ("rust_out_chk", &rust_out_chk),
            ("rust_h_chk", &rust_h_chk),
        ] {
            let n = count_nonfinite(buf);
            assert_eq!(
                n, 0,
                "{fixture_name}: {label} contains {n} non-finite (NaN/Inf) values — \
                 a decay-underflow regression the max-fold gate would silently pass"
            );
        }

        let d_seq_out = max_abs_diff(&rust_out_seq, &expected_out_seq);
        let d_seq_h = max_abs_diff(&rust_h_seq, &expected_h_seq);
        let d_chk_out = max_abs_diff(&rust_out_chk, &expected_out_chk);
        let d_chk_h = max_abs_diff(&rust_h_chk, &expected_h_chk);
        let d_internal_out = max_abs_diff(&rust_out_chk, &rust_out_seq);
        let d_internal_h = max_abs_diff(&rust_h_chk, &rust_h_seq);

        println!(
            "gdn_chunk_ref fixture={fixture_name} \
             rust_seq_vs_numpy_seq(out={d_seq_out:.3e}, h={d_seq_h:.3e}) \
             rust_chk_vs_numpy_chk(out={d_chk_out:.3e}, h={d_chk_h:.3e}) \
             rust_chk_vs_rust_seq(out={d_internal_out:.3e}, h={d_internal_h:.3e})"
        );

        assert!(
            d_seq_out <= TOL,
            "{fixture_name}: rust sequential_gdn out diverged from NumPy: max_abs={d_seq_out}"
        );
        assert!(
            d_seq_h <= TOL,
            "{fixture_name}: rust sequential_gdn h diverged from NumPy: max_abs={d_seq_h}"
        );
        assert!(
            d_chk_out <= TOL,
            "{fixture_name}: rust chunkwise_gdn out diverged from NumPy: max_abs={d_chk_out}"
        );
        assert!(
            d_chk_h <= TOL,
            "{fixture_name}: rust chunkwise_gdn h diverged from NumPy: max_abs={d_chk_h}"
        );
        assert!(
            d_internal_out <= TOL,
            "{fixture_name}: rust chunkwise_gdn out diverged from rust sequential_gdn (the \
             equivalence gate the Metal kernels rely on): max_abs={d_internal_out}"
        );
        assert!(
            d_internal_h <= TOL,
            "{fixture_name}: rust chunkwise_gdn h diverged from rust sequential_gdn (the \
             equivalence gate the Metal kernels rely on): max_abs={d_internal_h}"
        );
    }

    #[test]
    fn gdn_chunk_ref_parity_seed7_chunk64() {
        run_case("case_seed7_len191_chunk64.json");
    }

    #[test]
    fn gdn_chunk_ref_parity_seed11_chunk128() {
        run_case("case_seed11_len384_chunk128.json");
    }

    #[test]
    fn gdn_chunk_ref_parity_seed23_uneven_tail() {
        run_case("case_seed23_len130_chunk64.json");
    }

    /// gdn175 S1 fix regression: a strongly-decaying head (alpha as low as
    /// 1e-4) whose linear-space `cumprod(alpha)` underflows to exactly
    /// `0.0f32` within one C=64 chunk on the pre-fix oracle (`gamma_inv =
    /// 1/0.0 = inf`, `gamma[i]*gamma_inv[j] = 0*inf = NaN`). The log-space
    /// reformulation must still reproduce `sequential_gdn` to <= 1e-5 here.
    #[test]
    fn gdn_chunk_ref_parity_seed31_strong_decay() {
        run_case("case_seed31_len160_chunk64_strongdecay.json");
    }

    /// Gate-integrity proof: `max_abs_diff` must never let a non-finite
    /// difference read as a small in-tolerance value. A `.fold(0.0, f32::max)`
    /// silently drops the NaN/Inf operand and returns `0.0` here, which would
    /// pass a `<= TOL` assert on a catastrophically wrong output.
    #[test]
    fn max_abs_diff_is_nan_and_inf_honest() {
        let clean = [1.0f32, 2.0, 3.0];

        let nan_side = [1.0f32, f32::NAN, 3.0];
        let d = max_abs_diff(&clean, &nan_side);
        assert!(
            !d.is_finite(),
            "max_abs_diff silently dropped a NaN operand (got {d}); the gate is blind"
        );

        let inf_side = [1.0f32, f32::INFINITY, 3.0];
        let di = max_abs_diff(&clean, &inf_side);
        assert!(
            !di.is_finite(),
            "max_abs_diff silently dropped an Inf operand (got {di}); the gate is blind"
        );
    }

    /// Mutation-sensitivity proof for the strong-decay fixture: on the pre-fix
    /// linear-space oracle, `cumprod(alpha)` underflows to exactly `0.0f32`
    /// within the first C=64 chunk, so `1.0/gamma` is `+inf` and the decay
    /// factor `gamma[i] * gamma_inv[j]` becomes `0*inf = NaN`. The log-space
    /// reformulation shipped in this PR never forms `1.0/gamma`. If this assert
    /// ever fails, the fixture no longer exercises the regression and the
    /// `gdn_chunk_ref_parity_seed31_strong_decay` gate is decoration.
    #[test]
    fn seed31_fixture_triggers_linear_gamma_underflow() {
        let fx = load_fixture("case_seed31_len160_chunk64_strongdecay.json");
        let c = fx.chunk_size;
        let mut gamma = 1.0f32;
        let mut underflowed = false;
        for i in 0..c {
            gamma *= fx.alpha[i];
            if gamma == 0.0 {
                underflowed = true;
                break;
            }
        }
        assert!(
            underflowed,
            "seed31 strong-decay fixture no longer underflows the linear cumprod within a \
             chunk; it no longer guards the gamma-underflow regression this PR fixes"
        );
        let gamma_inv = 1.0f32 / gamma;
        assert!(
            !gamma_inv.is_finite(),
            "expected 1/0 = inf, got {gamma_inv}"
        );
        assert!(
            (0.0f32 * gamma_inv).is_nan(),
            "expected the pre-fix decay factor 0*inf to be NaN"
        );
    }

    /// ADR-080 C1 fail-closed (#850) table test for the chunkwise oracle's row
    /// normalizer: a row with a non-finite lane must be left at its zero-
    /// initialized default (whole-row zero by non-assignment, equivalent to
    /// direct zero-assignment), never divided through a non-finite `denom`
    /// (`NaN / x == NaN`, `x / inf == 0` only masks the corrupted row's OTHER
    /// lanes while the corrupted lane itself stays non-finite). Finite rows
    /// (all-zero, ordinary) must be numerically unchanged.
    ///
    /// Mutation-sensitive: reverting the `if !sum_sq.is_finite() { continue; }`
    /// guard back to unconditional division makes the `nan_row`/`inf_row` cases
    /// below fail (the output row keeps a non-finite lane).
    #[test]
    fn l2_normalize_rows_fail_closed_table() {
        let cols = 3usize;
        let cases: &[(&str, &[f32], bool)] = &[
            ("nan_lane", &[f32::NAN, 1.0, 2.0], true),
            ("pos_inf_lane", &[f32::INFINITY, 1.0, 2.0], true),
            ("neg_inf_lane", &[f32::NEG_INFINITY, 1.0, 2.0], true),
            ("all_zero", &[0.0, 0.0, 0.0], true),
        ];
        for (label, row, expect_all_zero) in cases {
            let out = l2_normalize_rows(row, 1, cols, 1e-6);
            assert!(
                out.iter().all(|x| x.is_finite()),
                "case {label}: l2_normalize_rows output must be fully finite, got {out:?}"
            );
            if *expect_all_zero {
                assert!(
                    out.iter().all(|x| *x == 0.0),
                    "case {label}: expected the whole row zeroed, got {out:?}"
                );
            }
        }

        // ordinary finite row: numerically unchanged from the pre-#850 formula.
        let ordinary = l2_normalize_rows(&[3.0f32, 4.0], 1, 2, 0.0);
        assert!((ordinary[0] - 0.6).abs() < 1e-6);
        assert!((ordinary[1] - 0.8).abs() < 1e-6);

        // multi-row: a non-finite row must not contaminate an adjacent finite row.
        let multi = l2_normalize_rows(&[f32::NAN, 0.0, 0.0, 3.0, 4.0, 0.0], 2, 3, 0.0);
        assert!(
            multi[0..3].iter().all(|x| *x == 0.0),
            "row 0 (poisoned) must be all-zero, got {:?}",
            &multi[0..3]
        );
        assert!(
            (multi[3] - 0.6).abs() < 1e-6 && (multi[4] - 0.8).abs() < 1e-6,
            "row 1 (clean) must be unaffected by row 0's poisoning, got {:?}",
            &multi[3..6]
        );
    }
}
