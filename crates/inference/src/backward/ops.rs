// Backward pass primitives for CPU f32 training.
// Each function computes a VJP (vector-Jacobian product) given the upstream
// gradient and cached activations from the forward pass.

/// Cross-entropy backward: dL/dlogits at completion positions only.
///
/// p = softmax(logits); at position i (completion), dL/dlogits[i] = (p[i] - onehot(target[i])) / N_comp.
/// At prompt/masked positions, dL/dlogits = 0 (output is zeros there).
///
/// `logits` shape: [seq_len, vocab_size]
/// `targets` shape: [seq_len] — u32, only positions in completion_range matter
/// `completion_start` — first index in the completion (0-indexed in sequence)
/// returns `dlogits` shape: [seq_len, vocab_size]
pub fn cross_entropy_backward(
    logits: &[f32],
    targets: &[u32],
    seq_len: usize,
    vocab_size: usize,
    completion_start: usize,
) -> Vec<f32> {
    assert_eq!(logits.len(), seq_len * vocab_size);
    assert_eq!(targets.len(), seq_len);

    let n_comp = (seq_len - completion_start) as f32;
    let mut dlogits = vec![0.0f32; seq_len * vocab_size];

    for t in completion_start..seq_len {
        let logit_row = &logits[t * vocab_size..(t + 1) * vocab_size];
        let target = targets[t] as usize;

        let max = logit_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f64;
        for &l in logit_row {
            sum_exp += ((l - max) as f64).exp();
        }
        let inv_sum = (1.0 / sum_exp) as f32;

        let drow = &mut dlogits[t * vocab_size..(t + 1) * vocab_size];
        for (v, d) in drow.iter_mut().enumerate() {
            let p = ((logit_row[v] - max).exp()) * inv_sum;
            let indicator = if v == target { 1.0f32 } else { 0.0 };
            *d = (p - indicator) / n_comp;
        }
    }
    dlogits
}

/// Linear VJP (base weights frozen): given upstream grad g=dL/dy (output-side),
/// computes dL/dx = W^T g.
///
/// `w` is row-major [d_out, d_in].  `g` is [d_out].  Returns [d_in].
pub fn linear_vjp(w: &[f32], g: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    assert_eq!(w.len(), d_out * d_in);
    assert_eq!(g.len(), d_out);

    let mut dx = vec![0.0f32; d_in];
    for i in 0..d_out {
        let gi = g[i];
        if gi == 0.0 {
            continue;
        }
        let row = &w[i * d_in..(i + 1) * d_in];
        for (j, &wij) in row.iter().enumerate() {
            dx[j] += wij * gi;
        }
    }
    dx
}

/// LoRA VJP: given upstream g=dL/dy and forward-cached h=A·x,
/// computes grad_A, grad_B, and dL/dx contribution.
///
/// dL/dB[i,r] = scale * g[i] * h[r]    → shape [d_out, rank]
/// dL/dA[r,j] = scale * (B^T g)[r] * x[j] → shape [rank, d_in]
/// dL/dx += scale * A^T (B^T g)          → shape [d_in]
///
/// Returns (grad_b, grad_a, dx_delta).
pub fn lora_vjp(
    g: &[f32],
    x: &[f32],
    h: &[f32],
    a: &[f32],
    b: &[f32],
    rank: usize,
    d_in: usize,
    d_out: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(a.len(), rank * d_in);
    assert_eq!(b.len(), d_out * rank);
    assert_eq!(g.len(), d_out);
    assert_eq!(x.len(), d_in);
    assert_eq!(h.len(), rank);

    // grad_B[i, r] = scale * g[i] * h[r]
    let mut grad_b = vec![0.0f32; d_out * rank];
    for i in 0..d_out {
        for r in 0..rank {
            grad_b[i * rank + r] = scale * g[i] * h[r];
        }
    }

    // bt_g[r] = B^T g = Σ_i B[i,r] * g[i]
    let mut bt_g = vec![0.0f32; rank];
    for r in 0..rank {
        let mut acc = 0.0f32;
        for i in 0..d_out {
            acc += b[i * rank + r] * g[i];
        }
        bt_g[r] = acc;
    }

    // grad_A[r, j] = scale * bt_g[r] * x[j]
    let mut grad_a = vec![0.0f32; rank * d_in];
    for r in 0..rank {
        for j in 0..d_in {
            grad_a[r * d_in + j] = scale * bt_g[r] * x[j];
        }
    }

    // dx = A^T (bt_g): dx[j] = scale * Σ_r A[r,j] * bt_g[r]
    let mut dx = vec![0.0f32; d_in];
    for r in 0..rank {
        let scaled_bt_g = scale * bt_g[r];
        let row = &a[r * d_in..(r + 1) * d_in];
        for (j, &aij) in row.iter().enumerate() {
            dx[j] += aij * scaled_bt_g;
        }
    }

    (grad_b, grad_a, dx)
}

/// RMSNorm backward (weight frozen).
///
/// Forward: y_i = x_i * (1 + w_i) * inv_rms   where inv_rms = 1/sqrt(mean(x^2) + eps)
/// Shifted-gamma convention — matches `qwen35_rms_norm` in norm.rs.
///
/// Backward: given g = dL/dy,
///   dL/dx_j = (1 + w_j) * g_j * inv_rms  -  x_j * inv_rms^3/D * Σ_i g_i*(1+w_i)*x_i
///
/// Derivation: y_i = (1+w_i)*x_i*r, r = (mean(x^2)+eps)^(-1/2).
///   ∂r/∂x_j = -x_j * r^3 / D
///   dL/dx_j = (1+w_j)*r*g_j + Σ_i g_i*(1+w_i)*x_i * (-x_j*r^3/D)
///
/// `x` is the pre-norm input, `w` is the RMSNorm weight (both length D).
/// `inv_rms` = 1 / sqrt(mean(x^2) + eps) from the forward.
/// `g` is dL/dy (length D). Returns dL/dx (length D).
pub fn rmsnorm_backward(x: &[f32], w: &[f32], inv_rms: f32, g: &[f32]) -> Vec<f32> {
    let d = x.len();
    assert_eq!(w.len(), d);
    assert_eq!(g.len(), d);

    // sum_xwg = Σ_j x_j * (1 + w_j) * g_j
    let sum_xwg: f32 = (0..d).map(|j| x[j] * (1.0 + w[j]) * g[j]).sum();

    let inv_rms3_over_d = inv_rms * inv_rms * inv_rms / d as f32;

    let mut dx = vec![0.0f32; d];
    for i in 0..d {
        dx[i] = (1.0 + w[i]) * g[i] * inv_rms - x[i] * inv_rms3_over_d * sum_xwg;
    }
    dx
}

/// RoPE backward (stride-half pairing, first rope_dim dimensions rotated).
///
/// The rotation R(θ) is orthogonal, so R(θ)^T = R(-θ) = inverse rotation.
/// For pair (i, half+i): forward is [x0*c - x1*s,  x0*s + x1*c]
///                        backward: [g0*c + g1*s, -g0*s + g1*c]
///
/// `g` is dL/dy (length head_dim). `rope_dim` <= head_dim.
/// `cos_vals`, `sin_vals` length rope_dim/2.
/// Returns dL/dx (length head_dim).
pub fn rope_backward(g: &[f32], cos_vals: &[f32], sin_vals: &[f32], rope_dim: usize) -> Vec<f32> {
    let head_dim = g.len();
    assert!(rope_dim <= head_dim);
    assert_eq!(rope_dim % 2, 0);
    let half = rope_dim / 2;
    assert_eq!(cos_vals.len(), half);
    assert_eq!(sin_vals.len(), half);

    let mut dx = g.to_vec();
    for i in 0..half {
        let c = cos_vals[i];
        let s = sin_vals[i];
        let g0 = g[i];
        let g1 = g[half + i];
        dx[i] = g0 * c + g1 * s;
        dx[half + i] = -g0 * s + g1 * c;
    }
    dx
}

/// SwiGLU backward.
///
/// Forward: gate = W_gate x, up = W_up x
///          s = silu(gate) = gate * sigmoid(gate)
///          m = s * up
///          y = W_down m
///
/// Backward: given dL/dm = W_down^T dL/dy,
///   dL/dup = dL/dm * s
///   dL/ds  = dL/dm * up
///   dL/dgate = dL/ds * silu'(gate)  where silu'(a) = sigma(a) + a*sigma(a)*(1-sigma(a))
///   dL/dx = W_gate^T dL/dgate + W_up^T dL/dup
///
/// Returns (dx [hidden], dm [inter]) where dm = W_down^T dy (the down-proj pre-activation grad).
pub fn swiglu_backward(
    dy: &[f32],
    gate_pre: &[f32],
    up_pre: &[f32],
    w_down: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    hidden: usize,
    inter: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(dy.len(), hidden);
    assert_eq!(gate_pre.len(), inter);
    assert_eq!(up_pre.len(), inter);
    assert_eq!(w_down.len(), hidden * inter);
    assert_eq!(w_gate.len(), inter * hidden);
    assert_eq!(w_up.len(), inter * hidden);

    // dL/dm = W_down^T dy   [inter]
    let mut dm = vec![0.0f32; inter];
    for j in 0..inter {
        let mut acc = 0.0f32;
        for i in 0..hidden {
            acc += w_down[i * inter + j] * dy[i];
        }
        dm[j] = acc;
    }

    let mut dx = vec![0.0f32; hidden];
    let mut d_gate_pre = vec![0.0f32; inter];

    for j in 0..inter {
        let a = gate_pre[j];
        let sigma_a = 1.0 / (1.0 + (-a).exp());
        let silu_a = a * sigma_a;
        let up_j = up_pre[j];
        let dm_j = dm[j];

        let d_up = dm_j * silu_a;
        let d_s = dm_j * up_j;
        let silu_prime = sigma_a + a * sigma_a * (1.0 - sigma_a);
        d_gate_pre[j] = d_s * silu_prime;

        let up_row = &w_up[j * hidden..(j + 1) * hidden];
        for (i, &wu) in up_row.iter().enumerate() {
            dx[i] += wu * d_up;
        }
    }

    for j in 0..inter {
        let wg_row = &w_gate[j * hidden..(j + 1) * hidden];
        let dg = d_gate_pre[j];
        for (i, &wg) in wg_row.iter().enumerate() {
            dx[i] += wg * dg;
        }
    }

    (dx, dm)
}

#[cfg(test)]
fn rel_err(analytic: &[f32], fd: &[f32]) -> f64 {
    let diff_sq: f64 = analytic
        .iter()
        .zip(fd.iter())
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    let norm_sq: f64 = analytic.iter().map(|&a| (a as f64).powi(2)).sum();
    (diff_sq / norm_sq.max(1e-30)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;
    const TOL: f64 = 1e-3;

    #[test]
    fn cross_entropy_backward_gradcheck() {
        let seq_len = 4;
        let vocab = 8;
        let completion_start = 2;

        let logits: Vec<f32> = (0..seq_len * vocab)
            .map(|i| (i as f32) * 0.1 - 2.0)
            .collect();
        let targets: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab as u32).collect();

        let loss_fn = |logits: &[f32]| -> f32 {
            let mut total = 0.0f64;
            let n_comp = (seq_len - completion_start) as f64;
            for t in completion_start..seq_len {
                let row = &logits[t * vocab..(t + 1) * vocab];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f64 = row.iter().map(|&x| ((x - max) as f64).exp()).sum();
                let lse = (max as f64) + sum_exp.ln();
                total += lse - logits[t * vocab + targets[t] as usize] as f64;
            }
            (total / n_comp) as f32
        };

        let analytic = cross_entropy_backward(&logits, &targets, seq_len, vocab, completion_start);

        let mut fd = vec![0.0f32; seq_len * vocab];
        for k in 0..seq_len * vocab {
            let mut lp = logits.clone();
            let mut lm = logits.clone();
            lp[k] += EPS;
            lm[k] -= EPS;
            fd[k] = (loss_fn(&lp) - loss_fn(&lm)) / (2.0 * EPS);
        }

        let err = rel_err(&analytic, &fd);
        eprintln!("cross_entropy_backward rel_err={err:.2e}");
        assert!(err < TOL, "cross_entropy rel_err {err:.2e} >= {TOL:.2e}");
    }

    #[test]
    fn linear_vjp_gradcheck() {
        let d_in = 6;
        let d_out = 4;
        let w: Vec<f32> = (0..d_in * d_out).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let x: Vec<f32> = (0..d_in).map(|i| i as f32 * 0.3 - 0.5).collect();
        let g: Vec<f32> = (0..d_out).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

        let loss_fn = |x: &[f32]| -> f32 {
            (0..d_out)
                .map(|i| {
                    let row = &w[i * d_in..(i + 1) * d_in];
                    let wx_i: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                    g[i] * wx_i
                })
                .sum()
        };

        let analytic = linear_vjp(&w, &g, d_in, d_out);

        let mut fd = vec![0.0f32; d_in];
        for j in 0..d_in {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += EPS;
            xm[j] -= EPS;
            fd[j] = (loss_fn(&xp) - loss_fn(&xm)) / (2.0 * EPS);
        }

        let err = rel_err(&analytic, &fd);
        eprintln!("linear_vjp rel_err={err:.2e}");
        assert!(err < TOL, "linear_vjp rel_err {err:.2e} >= {TOL:.2e}");
    }

    #[test]
    fn lora_vjp_gradcheck() {
        let rank = 2;
        let d_in = 3;
        let d_out = 4;
        let scale = 0.5f32;

        let a: Vec<f32> = (0..rank * d_in)
            .map(|i| (i as f32 + 1.0) * 0.2 - 0.5)
            .collect();
        let b: Vec<f32> = (0..d_out * rank)
            .map(|i| (i as f32 + 1.0) * 0.15 - 0.4)
            .collect();
        let x: Vec<f32> = (0..d_in).map(|i| (i as f32 + 1.0) * 0.4).collect();
        let g: Vec<f32> = (0..d_out).map(|i| (i as f32 + 1.0) * 0.3 - 0.5).collect();

        let forward = |a: &[f32], b: &[f32], x: &[f32]| -> f32 {
            let mut h = vec![0.0f32; rank];
            for r in 0..rank {
                h[r] = a[r * d_in..(r + 1) * d_in]
                    .iter()
                    .zip(x.iter())
                    .map(|(ai, xi)| ai * xi)
                    .sum();
            }
            let mut y = vec![0.0f32; d_out];
            for i in 0..d_out {
                y[i] = scale
                    * b[i * rank..(i + 1) * rank]
                        .iter()
                        .zip(h.iter())
                        .map(|(bi, hi)| bi * hi)
                        .sum::<f32>();
            }
            g.iter().zip(y.iter()).map(|(gi, yi)| gi * yi).sum()
        };

        let h: Vec<f32> = (0..rank)
            .map(|r| {
                a[r * d_in..(r + 1) * d_in]
                    .iter()
                    .zip(x.iter())
                    .map(|(ai, xi)| ai * xi)
                    .sum()
            })
            .collect();

        let (grad_b, grad_a, dx) = lora_vjp(&g, &x, &h, &a, &b, rank, d_in, d_out, scale);

        let mut fd_a = vec![0.0f32; rank * d_in];
        for k in 0..rank * d_in {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[k] += EPS;
            am[k] -= EPS;
            fd_a[k] = (forward(&ap, &b, &x) - forward(&am, &b, &x)) / (2.0 * EPS);
        }

        let mut fd_b = vec![0.0f32; d_out * rank];
        for k in 0..d_out * rank {
            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[k] += EPS;
            bm[k] -= EPS;
            fd_b[k] = (forward(&a, &bp, &x) - forward(&a, &bm, &x)) / (2.0 * EPS);
        }

        let mut fd_x = vec![0.0f32; d_in];
        for k in 0..d_in {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += EPS;
            xm[k] -= EPS;
            fd_x[k] = (forward(&a, &b, &xp) - forward(&a, &b, &xm)) / (2.0 * EPS);
        }

        let err_a = rel_err(&grad_a, &fd_a);
        let err_b = rel_err(&grad_b, &fd_b);
        let err_x = rel_err(&dx, &fd_x);
        eprintln!("lora_vjp: grad_A rel_err={err_a:.2e} grad_B={err_b:.2e} dx={err_x:.2e}");
        assert!(err_a < TOL, "lora grad_A rel_err {err_a:.2e} >= {TOL:.2e}");
        assert!(err_b < TOL, "lora grad_B rel_err {err_b:.2e} >= {TOL:.2e}");
        assert!(err_x < TOL, "lora dx rel_err {err_x:.2e} >= {TOL:.2e}");
    }

    #[test]
    fn rmsnorm_backward_gradcheck() {
        let d = 8;
        let eps = 1e-6f32;

        let x: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.3 - 1.0).collect();
        let w: Vec<f32> = (0..d).map(|i| 1.0 + i as f32 * 0.05).collect();
        let g: Vec<f32> = (0..d)
            .map(|i| if i % 3 == 0 { 1.0 } else { -0.5 })
            .collect();

        let rms = |x: &[f32]| -> f32 {
            let mean_sq: f32 = x.iter().map(|xi| xi * xi).sum::<f32>() / d as f32;
            (mean_sq + eps).sqrt()
        };

        let loss_fn = |x: &[f32]| -> f32 {
            let r = rms(x);
            x.iter()
                .zip(w.iter())
                .zip(g.iter())
                .map(|((&xi, &wi), &gi)| gi * xi * (1.0 + wi) / r)
                .sum()
        };

        let inv_rms_val = 1.0 / rms(&x);
        let analytic = rmsnorm_backward(&x, &w, inv_rms_val, &g);

        let mut fd = vec![0.0f32; d];
        for k in 0..d {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += EPS;
            xm[k] -= EPS;
            fd[k] = (loss_fn(&xp) - loss_fn(&xm)) / (2.0 * EPS);
        }

        let err = rel_err(&analytic, &fd);
        eprintln!("rmsnorm_backward rel_err={err:.2e}");
        assert!(err < TOL, "rmsnorm rel_err {err:.2e} >= {TOL:.2e}");
    }

    #[test]
    fn rope_backward_gradcheck() {
        let head_dim = 8;
        let rope_dim = 4;
        let half = rope_dim / 2;

        let cos_vals: Vec<f32> = (0..half).map(|i| (i as f32 * 0.3).cos()).collect();
        let sin_vals: Vec<f32> = (0..half).map(|i| (i as f32 * 0.3).sin()).collect();

        let x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.2).collect();
        let g: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 + 1.0) * 0.15 - 1.0)
            .collect();

        let rope_fwd = |x: &[f32]| -> Vec<f32> {
            let mut y = x.to_vec();
            for i in 0..half {
                let c = cos_vals[i];
                let s = sin_vals[i];
                let x0 = x[i];
                let x1 = x[half + i];
                y[i] = x0 * c - x1 * s;
                y[half + i] = x0 * s + x1 * c;
            }
            y
        };

        let loss_fn = |x: &[f32]| -> f32 {
            let y = rope_fwd(x);
            y.iter().zip(g.iter()).map(|(yi, gi)| yi * gi).sum()
        };

        let analytic = rope_backward(&g, &cos_vals, &sin_vals, rope_dim);

        let mut fd = vec![0.0f32; head_dim];
        for k in 0..head_dim {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += EPS;
            xm[k] -= EPS;
            fd[k] = (loss_fn(&xp) - loss_fn(&xm)) / (2.0 * EPS);
        }

        let err = rel_err(&analytic, &fd);
        eprintln!("rope_backward rel_err={err:.2e}");
        assert!(err < TOL, "rope rel_err {err:.2e} >= {TOL:.2e}");
    }

    #[test]
    fn swiglu_backward_gradcheck() {
        let hidden = 4;
        let inter = 6;

        let w_gate: Vec<f32> = (0..inter * hidden)
            .map(|i| (i as f32 + 1.0) * 0.1 - 1.5)
            .collect();
        let w_up: Vec<f32> = (0..inter * hidden)
            .map(|i| (i as f32 + 1.0) * 0.12 - 1.2)
            .collect();
        let w_down: Vec<f32> = (0..hidden * inter)
            .map(|i| (i as f32 + 1.0) * 0.08 - 0.9)
            .collect();

        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let dy: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.2 - 1.0).collect();

        let swiglu_fwd = |x: &[f32]| -> f32 {
            let mut gate = vec![0.0f32; inter];
            let mut up = vec![0.0f32; inter];
            for j in 0..inter {
                gate[j] = w_gate[j * hidden..(j + 1) * hidden]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                up[j] = w_up[j * hidden..(j + 1) * hidden]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| a * b)
                    .sum();
            }
            let mut m = vec![0.0f32; inter];
            for j in 0..inter {
                let s = gate[j] / (1.0 + (-gate[j]).exp());
                m[j] = s * up[j];
            }
            let mut y = vec![0.0f32; hidden];
            for i in 0..hidden {
                y[i] = w_down[i * inter..(i + 1) * inter]
                    .iter()
                    .zip(m.iter())
                    .map(|(a, b)| a * b)
                    .sum();
            }
            dy.iter().zip(y.iter()).map(|(d, yi)| d * yi).sum()
        };

        let gate_pre: Vec<f32> = (0..inter)
            .map(|j| {
                w_gate[j * hidden..(j + 1) * hidden]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        let up_pre: Vec<f32> = (0..inter)
            .map(|j| {
                w_up[j * hidden..(j + 1) * hidden]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();

        let (analytic_dx, _dm) = swiglu_backward(
            &dy, &gate_pre, &up_pre, &w_down, &w_gate, &w_up, hidden, inter,
        );

        let mut fd = vec![0.0f32; hidden];
        for k in 0..hidden {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += EPS;
            xm[k] -= EPS;
            fd[k] = (swiglu_fwd(&xp) - swiglu_fwd(&xm)) / (2.0 * EPS);
        }

        let err = rel_err(&analytic_dx, &fd);
        eprintln!("swiglu_backward rel_err={err:.2e}");
        assert!(err < TOL, "swiglu rel_err {err:.2e} >= {TOL:.2e}");
    }
}
