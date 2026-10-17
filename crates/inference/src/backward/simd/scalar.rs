use super::{Completion, ForcedExecutionError, LoraGradients, Operation};
use crate::backward::ops::BackwardMath;

type Result<T> = std::result::Result<T, ForcedExecutionError>;

fn require(condition: bool, operation: Operation, problem: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(ForcedExecutionError::Shape { operation, problem })
    }
}

fn product(a: usize, b: usize, operation: Operation) -> Result<usize> {
    let size = a.checked_mul(b).ok_or(ForcedExecutionError::Shape {
        operation,
        problem: "shape product overflow",
    })?;
    extent(size, operation)
}

fn extent(size: usize, operation: Operation) -> Result<usize> {
    require(
        size <= isize::MAX as usize / std::mem::size_of::<f32>(),
        operation,
        "shape byte size exceeds allocation limit",
    )?;
    Ok(size)
}

fn ce_shape(
    logits: &[f32],
    targets: &[u32],
    seq_len: usize,
    vocab_size: usize,
    completion_start: usize,
) -> Result<usize> {
    let op = Operation::CrossEntropyBackward;
    let size = product(seq_len, vocab_size, op)?;
    require(logits.len() == size, op, "logits length")?;
    require(targets.len() == seq_len, op, "targets length")?;
    require(completion_start <= seq_len, op, "completion start")?;
    Ok(size)
}

pub(super) fn cross_entropy_backward(
    completion: Completion<'_>,
    logits: &[f32],
    targets: &[u32],
    seq_len: usize,
    vocab_size: usize,
    completion_start: usize,
) -> Result<Vec<f32>> {
    let size = ce_shape(logits, targets, seq_len, vocab_size, completion_start)?;
    let mut output = vec![0.0; size];
    cross_entropy_backward_into(
        completion,
        logits,
        targets,
        seq_len,
        vocab_size,
        completion_start,
        &mut output,
    )?;
    Ok(output)
}

pub(super) fn cross_entropy_backward_into(
    completion: Completion<'_>,
    logits: &[f32],
    targets: &[u32],
    seq_len: usize,
    vocab_size: usize,
    completion_start: usize,
    dlogits: &mut [f32],
) -> Result<()> {
    let op = Operation::CrossEntropyBackward;
    let size = ce_shape(logits, targets, seq_len, vocab_size, completion_start)?;
    require(dlogits.len() == size, op, "dlogits length")?;
    dlogits.fill(0.0);
    let n_comp = (seq_len - completion_start) as f32;
    for t in completion_start..seq_len {
        let row = &logits[t * vocab_size..(t + 1) * vocab_size];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f64;
        for &value in row {
            sum_exp += ((value - max) as f64).exp();
        }
        let inv_sum = (1.0 / sum_exp) as f32;
        for (v, output) in dlogits[t * vocab_size..(t + 1) * vocab_size]
            .iter_mut()
            .enumerate()
        {
            let p = (row[v] - max).exp() * inv_sum;
            let indicator = if v == targets[t] as usize { 1.0 } else { 0.0 };
            *output = (p - indicator) / n_comp;
        }
    }
    completion.finish(op)
}

fn linear_shape(w: &[f32], g: &[f32], d_in: usize, d_out: usize) -> Result<()> {
    let op = Operation::LinearVjp;
    let size = product(d_out, d_in, op)?;
    extent(d_in, op)?;
    require(w.len() == size, op, "weights length")?;
    require(g.len() == d_out, op, "gradient length")
}

fn linear_core(w: &[f32], g: &[f32], d_in: usize, d_out: usize, dx: &mut [f32]) {
    dx.fill(0.0);
    for i in 0..d_out {
        let gi = g[i];
        let row = &w[i * d_in..(i + 1) * d_in];
        for (j, &wij) in row.iter().enumerate() {
            dx[j] += wij * gi;
        }
    }
}

pub(super) fn linear_vjp(
    completion: Completion<'_>,
    w: &[f32],
    g: &[f32],
    d_in: usize,
    d_out: usize,
) -> Result<Vec<f32>> {
    linear_shape(w, g, d_in, d_out)?;
    let mut dx = vec![0.0; d_in];
    linear_vjp_into(completion, w, g, d_in, d_out, &mut dx)?;
    Ok(dx)
}

pub(super) fn linear_vjp_into(
    completion: Completion<'_>,
    w: &[f32],
    g: &[f32],
    d_in: usize,
    d_out: usize,
    dx: &mut [f32],
) -> Result<()> {
    linear_shape(w, g, d_in, d_out)?;
    require(dx.len() == d_in, Operation::LinearVjp, "dx length")?;
    linear_core(w, g, d_in, d_out, dx);
    completion.finish(Operation::LinearVjp)
}

fn lora_shape(
    g: &[f32],
    x: &[f32],
    h: &[f32],
    a: &[f32],
    b: &[f32],
    rank: usize,
    d_in: usize,
    d_out: usize,
) -> Result<()> {
    let op = Operation::LoraVjp;
    let a_size = product(rank, d_in, op)?;
    let b_size = product(d_out, rank, op)?;
    extent(d_in, op)?;
    extent(rank, op)?;
    require(a.len() == a_size, op, "A length")?;
    require(b.len() == b_size, op, "B length")?;
    require(g.len() == d_out, op, "gradient length")?;
    require(x.len() == d_in, op, "x length")?;
    require(h.len() == rank, op, "cached h length")
}

fn lora_core(
    g: &[f32],
    x: &[f32],
    h: &[f32],
    a: &[f32],
    b: &[f32],
    rank: usize,
    d_in: usize,
    d_out: usize,
    scale: f32,
    grad_b: &mut [f32],
    grad_a: &mut [f32],
    dx: &mut [f32],
    bt_g: &mut [f32],
) {
    for i in 0..d_out {
        for r in 0..rank {
            grad_b[i * rank + r] = scale * g[i] * h[r];
        }
    }
    for r in 0..rank {
        let mut sum = 0.0f32;
        for i in 0..d_out {
            sum += b[i * rank + r] * g[i];
        }
        bt_g[r] = sum;
    }
    for r in 0..rank {
        for j in 0..d_in {
            grad_a[r * d_in + j] = scale * bt_g[r] * x[j];
        }
    }
    dx.fill(0.0);
    for r in 0..rank {
        let scaled = scale * bt_g[r];
        for j in 0..d_in {
            dx[j] += a[r * d_in + j] * scaled;
        }
    }
}

pub(super) fn lora_vjp(
    completion: Completion<'_>,
    g: &[f32],
    x: &[f32],
    h: &[f32],
    a: &[f32],
    b: &[f32],
    rank: usize,
    d_in: usize,
    d_out: usize,
    scale: f32,
) -> Result<LoraGradients> {
    lora_shape(g, x, h, a, b, rank, d_in, d_out)?;
    let mut grad_b = vec![0.0; b.len()];
    let mut grad_a = vec![0.0; a.len()];
    let mut dx = vec![0.0; d_in];
    let mut bt_g = vec![0.0; rank];
    lora_vjp_into(
        completion,
        g,
        x,
        h,
        a,
        b,
        rank,
        d_in,
        d_out,
        scale,
        &mut grad_b,
        &mut grad_a,
        &mut dx,
        &mut bt_g,
    )?;
    Ok((grad_b, grad_a, dx))
}

pub(super) fn lora_vjp_into(
    completion: Completion<'_>,
    g: &[f32],
    x: &[f32],
    h: &[f32],
    a: &[f32],
    b: &[f32],
    rank: usize,
    d_in: usize,
    d_out: usize,
    scale: f32,
    grad_b: &mut [f32],
    grad_a: &mut [f32],
    dx: &mut [f32],
    bt_g: &mut [f32],
) -> Result<()> {
    let op = Operation::LoraVjp;
    lora_shape(g, x, h, a, b, rank, d_in, d_out)?;
    require(grad_b.len() == b.len(), op, "grad B length")?;
    require(grad_a.len() == a.len(), op, "grad A length")?;
    require(dx.len() == d_in, op, "dx length")?;
    require(bt_g.len() == rank, op, "bt_g scratch length")?;
    lora_core(
        g, x, h, a, b, rank, d_in, d_out, scale, grad_b, grad_a, dx, bt_g,
    );
    completion.finish(op)
}

fn rms_shape(x: &[f32], w: &[f32], g: &[f32]) -> Result<()> {
    let op = Operation::RmsnormBackward;
    extent(x.len(), op)?;
    require(w.len() == x.len(), op, "weights length")?;
    require(g.len() == x.len(), op, "gradient length")
}

pub(super) fn rmsnorm_backward(
    completion: Completion<'_>,
    x: &[f32],
    w: &[f32],
    inv_rms: f32,
    g: &[f32],
) -> Result<Vec<f32>> {
    rms_shape(x, w, g)?;
    let mut dx = vec![0.0; x.len()];
    rmsnorm_backward_into(completion, x, w, inv_rms, g, &mut dx)?;
    Ok(dx)
}

pub(super) fn rmsnorm_backward_into(
    completion: Completion<'_>,
    x: &[f32],
    w: &[f32],
    inv_rms: f32,
    g: &[f32],
    dx: &mut [f32],
) -> Result<()> {
    let op = Operation::RmsnormBackward;
    rms_shape(x, w, g)?;
    let d = x.len();
    require(dx.len() == d, op, "dx length")?;
    let sum_xwg: f32 = (0..d).map(|j| x[j] * w[j] * g[j]).sum();
    let inv_rms3_over_d = inv_rms * inv_rms * inv_rms / d as f32;
    for i in 0..d {
        dx[i] = w[i] * g[i] * inv_rms - x[i] * inv_rms3_over_d * sum_xwg;
    }
    completion.finish(op)
}

fn rope_shape(g: &[f32], cos: &[f32], sin: &[f32], rope_dim: usize) -> Result<()> {
    let op = Operation::RopeBackward;
    extent(g.len(), op)?;
    require(rope_dim <= g.len(), op, "rotated width exceeds head")?;
    require(rope_dim.is_multiple_of(2), op, "rotated width is odd")?;
    require(cos.len() == rope_dim / 2, op, "cosine length")?;
    require(sin.len() == rope_dim / 2, op, "sine length")
}

pub(super) fn rope_backward(
    completion: Completion<'_>,
    g: &[f32],
    cos: &[f32],
    sin: &[f32],
    rope_dim: usize,
) -> Result<Vec<f32>> {
    rope_shape(g, cos, sin, rope_dim)?;
    let mut dx = vec![0.0; g.len()];
    rope_backward_into(completion, g, cos, sin, rope_dim, &mut dx)?;
    Ok(dx)
}

pub(super) fn rope_backward_into(
    completion: Completion<'_>,
    g: &[f32],
    cos: &[f32],
    sin: &[f32],
    rope_dim: usize,
    dx: &mut [f32],
) -> Result<()> {
    let op = Operation::RopeBackward;
    rope_shape(g, cos, sin, rope_dim)?;
    require(dx.len() == g.len(), op, "dx length")?;
    dx.copy_from_slice(g);
    let half = rope_dim / 2;
    for i in 0..half {
        let g0 = g[i];
        let g1 = g[half + i];
        dx[i] = g0 * cos[i] + g1 * sin[i];
        dx[half + i] = -g0 * sin[i] + g1 * cos[i];
    }
    completion.finish(op)
}

fn swiglu_shape(
    dy: &[f32],
    gate_pre: &[f32],
    up_pre: &[f32],
    w_down: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    hidden: usize,
    inter: usize,
) -> Result<()> {
    let op = Operation::SwigluBackward;
    let size = product(hidden, inter, op)?;
    extent(hidden, op)?;
    extent(inter, op)?;
    require(dy.len() == hidden, op, "gradient length")?;
    require(gate_pre.len() == inter, op, "gate length")?;
    require(up_pre.len() == inter, op, "up length")?;
    require(w_down.len() == size, op, "down weights length")?;
    require(w_gate.len() == size, op, "gate weights length")?;
    require(w_up.len() == size, op, "up weights length")
}

pub(super) fn swiglu_backward(
    completion: Completion<'_>,
    dy: &[f32],
    gate_pre: &[f32],
    up_pre: &[f32],
    w_down: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    hidden: usize,
    inter: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    swiglu_shape(dy, gate_pre, up_pre, w_down, w_gate, w_up, hidden, inter)?;
    let mut dx = vec![0.0; hidden];
    let mut dm = vec![0.0; inter];
    let mut d_gate = vec![0.0; inter];
    swiglu_backward_into(
        completion,
        dy,
        gate_pre,
        up_pre,
        w_down,
        w_gate,
        w_up,
        hidden,
        inter,
        &mut dx,
        &mut dm,
        &mut d_gate,
    )?;
    Ok((dx, dm))
}

pub(super) fn swiglu_backward_into(
    completion: Completion<'_>,
    dy: &[f32],
    gate_pre: &[f32],
    up_pre: &[f32],
    w_down: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    hidden: usize,
    inter: usize,
    dx: &mut [f32],
    dm: &mut [f32],
    d_gate: &mut [f32],
) -> Result<()> {
    let op = Operation::SwigluBackward;
    swiglu_shape(dy, gate_pre, up_pre, w_down, w_gate, w_up, hidden, inter)?;
    require(dx.len() == hidden, op, "dx length")?;
    require(dm.len() == inter, op, "dm length")?;
    require(d_gate.len() == inter, op, "gate scratch length")?;
    for j in 0..inter {
        let mut sum = 0.0f32;
        for i in 0..hidden {
            sum += w_down[i * inter + j] * dy[i];
        }
        dm[j] = sum;
    }
    dx.fill(0.0);
    for j in 0..inter {
        let a = gate_pre[j];
        let sigma = 1.0 / (1.0 + (-a).exp());
        let silu = a * sigma;
        let d_up = dm[j] * silu;
        let d_s = dm[j] * up_pre[j];
        let silu_prime = sigma + a * sigma * (1.0 - sigma);
        d_gate[j] = d_s * silu_prime;
        for i in 0..hidden {
            dx[i] += w_up[j * hidden + i] * d_up;
        }
    }
    for i in 0..hidden {
        let mut gate_sum = 0.0f32;
        for j in 0..inter {
            gate_sum += w_gate[j * hidden + i] * d_gate[j];
        }
        dx[i] += gate_sum;
    }
    completion.finish(op)
}

pub(super) struct ScalarMath;

impl BackwardMath for ScalarMath {
    fn linear_vjp(w: &[f32], g: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
        let mut dx = vec![0.0; d_in];
        linear_core(w, g, d_in, d_out, &mut dx);
        dx
    }

    fn lora_vjp(
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
        let mut grad_b = vec![0.0; b.len()];
        let mut grad_a = vec![0.0; a.len()];
        let mut dx = vec![0.0; d_in];
        let mut bt_g = vec![0.0; rank];
        lora_core(
            g,
            x,
            h,
            a,
            b,
            rank,
            d_in,
            d_out,
            scale,
            &mut grad_b,
            &mut grad_a,
            &mut dx,
            &mut bt_g,
        );
        (grad_b, grad_a, dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backward::ops;
    use crate::backward::simd::{Backend, ForcedDispatch, force_scalar};
    use crate::forward::cpu::observe_matmul_calls;

    fn values(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + phase) * 0.73).sin() * 0.4)
            .collect()
    }

    fn close(actual: &[f32], expected: &[f32], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().chain(expected).all(|x| x.is_finite()));
        let difference: f64 = actual
            .iter()
            .zip(expected)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        let norm: f64 = expected.iter().map(|&x| f64::from(x).powi(2)).sum();
        let relative = (difference / norm.max(1e-30)).sqrt();
        assert!(relative < tolerance, "relative error {relative}");
    }

    fn derivative(input: &[f32], loss: impl Fn(&[f32]) -> f64) -> Vec<f32> {
        (0..input.len())
            .map(|index| {
                let mut plus = input.to_vec();
                let mut minus = input.to_vec();
                plus[index] += 1e-3;
                minus[index] -= 1e-3;
                ((loss(&plus) - loss(&minus)) / (f64::from(plus[index]) - f64::from(minus[index])))
                    as f32
            })
            .collect()
    }

    fn completed(probe: &ForcedDispatch, operation: Operation) {
        assert_eq!(probe.verify_for(operation), Ok(Backend::Scalar));
    }

    fn rejected<T>(result: Result<T>, probe: &ForcedDispatch, operation: Operation) {
        assert!(
            matches!(result, Err(ForcedExecutionError::Shape { operation: op, .. }) if op == operation)
        );
        assert!(probe.verify().is_err());
    }

    fn capacity_rejected<T>(result: Result<T>, probe: &ForcedDispatch, operation: Operation) {
        assert!(matches!(
            &result,
            Err(ForcedExecutionError::Shape {
                problem: "shape byte size exceeds allocation limit",
                ..
            })
        ));
        rejected(result, probe, operation);
    }

    #[test]
    fn cross_entropy_matches_independent_loss_and_masked_rows() {
        let logits = values(21, 0.3);
        let targets = [6, 2, 4];
        let probe = force_scalar();
        let actual = probe
            .cross_entropy_backward(&logits, &targets, 3, 7, 1)
            .unwrap();
        completed(&probe, Operation::CrossEntropyBackward);
        let expected = derivative(&logits, |input| {
            (1..3)
                .map(|t| {
                    let row = &input[t * 7..(t + 1) * 7];
                    let sum: f64 = row.iter().map(|&x| f64::from(x).exp()).sum();
                    (sum.ln() - f64::from(row[targets[t] as usize])) / 2.0
                })
                .sum()
        });
        close(&actual, &expected, 1e-3);
        assert!(actual[..7].iter().all(|x| x.to_bits() == 0));
        close(
            &actual,
            &ops::cross_entropy_backward(&logits, &targets, 3, 7, 1),
            1e-4,
        );
    }

    #[test]
    fn linear_has_independent_gradient_and_ascending_accumulation() {
        let w = values(15, 0.2);
        let g = values(3, 1.1);
        let x = values(5, 2.2);
        let probe = force_scalar();
        let actual = probe.linear_vjp(&w, &g, 5, 3).unwrap();
        completed(&probe, Operation::LinearVjp);
        let expected = derivative(&x, |input| {
            (0..3)
                .map(|i| {
                    f64::from(g[i])
                        * (0..5)
                            .map(|j| f64::from(w[i * 5 + j]) * f64::from(input[j]))
                            .sum::<f64>()
                })
                .sum()
        });
        close(&actual, &expected, 1e-3);
        let ordered = force_scalar()
            .linear_vjp(&[1e20, -1e20, 1.0], &[1.0; 3], 1, 3)
            .unwrap();
        assert_eq!(ordered, [1.0]);
    }

    fn lora_loss(a: &[f32], b: &[f32], x: &[f32], g: &[f32], scale: f32) -> f64 {
        (0..g.len())
            .map(|i| {
                f64::from(g[i])
                    * f64::from(scale)
                    * (0..2)
                        .map(|r| {
                            f64::from(b[i * 2 + r])
                                * (0..x.len())
                                    .map(|j| f64::from(a[r * x.len() + j]) * f64::from(x[j]))
                                    .sum::<f64>()
                        })
                        .sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn lora_all_three_outputs_match_independent_gradients() {
        let a = values(6, 0.4);
        let b = values(10, 1.2);
        let x = values(3, 2.3);
        let g = values(5, 3.7);
        let h: Vec<f32> = (0..2)
            .map(|r| (0..3).map(|j| a[r * 3 + j] * x[j]).sum())
            .collect();
        let scale = 0.37;
        let probe = force_scalar();
        let (gb, ga, dx) = probe.lora_vjp(&g, &x, &h, &a, &b, 2, 3, 5, scale).unwrap();
        completed(&probe, Operation::LoraVjp);
        close(
            &ga,
            &derivative(&a, |v| lora_loss(v, &b, &x, &g, scale)),
            1e-3,
        );
        close(
            &gb,
            &derivative(&b, |v| lora_loss(&a, v, &x, &g, scale)),
            1e-3,
        );
        close(
            &dx,
            &derivative(&x, |v| lora_loss(&a, &b, v, &g, scale)),
            1e-3,
        );
        let changed_h = [0.9, -0.7];
        let changed = force_scalar()
            .lora_vjp(&g, &x, &changed_h, &a, &b, 2, 3, 5, scale)
            .unwrap();
        assert_ne!(changed.0, gb);
        assert_eq!(changed.1, ga);
        assert_eq!(changed.2, dx);
    }

    #[test]
    fn rmsnorm_matches_independent_normalized_forward() {
        let x = values(7, 0.1);
        let w = values(7, 1.6);
        let g = values(7, 2.4);
        let mean: f32 = x.iter().map(|x| x * x).sum::<f32>() / 7.0;
        let inv = (mean + 1e-4).sqrt().recip();
        let probe = force_scalar();
        let actual = probe.rmsnorm_backward(&x, &w, inv, &g).unwrap();
        completed(&probe, Operation::RmsnormBackward);
        let expected = derivative(&x, |input| {
            let mean: f64 = input.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / 7.0;
            let rms = (mean + 1e-4).sqrt();
            (0..7)
                .map(|i| f64::from(g[i]) * f64::from(w[i]) * f64::from(input[i]) / rms)
                .sum()
        });
        close(&actual, &expected, 1e-3);
        close(&actual, &ops::rmsnorm_backward(&x, &w, inv, &g), 1e-4);
    }

    #[test]
    fn rope_uses_stride_half_inverse_and_preserves_tail_bits() {
        let g = [1.0, 2.0, 3.0, 4.0, -0.0, f32::from_bits(0x7fc0_0023)];
        let probe = force_scalar();
        let actual = probe
            .rope_backward(&g, &[0.0, 0.6], &[1.0, 0.8], 4)
            .unwrap();
        completed(&probe, Operation::RopeBackward);
        assert_eq!(&actual[..4], &[3.0, 4.4, -1.0, 0.8000001]);
        assert_eq!(actual[4].to_bits(), g[4].to_bits());
        assert_eq!(actual[5].to_bits(), g[5].to_bits());
        let identity = force_scalar().rope_backward(&g, &[], &[], 0).unwrap();
        assert_eq!(
            identity.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            g.map(f32::to_bits)
        );
    }

    #[test]
    fn rope_matches_independent_forward_derivative() {
        let g = values(7, 0.2);
        let x = values(7, 2.1);
        let cos = [0.6, 0.8, 0.0];
        let sin = [0.8, -0.6, 1.0];
        let actual = force_scalar().rope_backward(&g, &cos, &sin, 6).unwrap();
        let expected = derivative(&x, |v| {
            let rotated: f64 = (0..3)
                .map(|i| {
                    let c = f64::from(cos[i]);
                    let s = f64::from(sin[i]);
                    f64::from(g[i]) * (f64::from(v[i]) * c - f64::from(v[3 + i]) * s)
                        + f64::from(g[3 + i]) * (f64::from(v[i]) * s + f64::from(v[3 + i]) * c)
                })
                .sum();
            rotated + f64::from(g[6]) * f64::from(v[6])
        });
        close(&actual, &expected, 1e-3);
    }

    #[test]
    fn swiglu_both_outputs_match_independent_forward_derivatives() {
        let x = values(3, 0.1);
        let dy = values(3, 1.3);
        let down = values(15, 2.2);
        let gate = values(15, 3.4);
        let up = values(15, 4.7);
        let project = |w: &[f32]| -> Vec<f32> {
            (0..5)
                .map(|j| (0..3).map(|i| w[j * 3 + i] * x[i]).sum())
                .collect()
        };
        let gate_pre = project(&gate);
        let up_pre = project(&up);
        let probe = force_scalar();
        let (dx, dm) = probe
            .swiglu_backward(&dy, &gate_pre, &up_pre, &down, &gate, &up, 3, 5)
            .unwrap();
        completed(&probe, Operation::SwigluBackward);
        let loss = |m: &[f32]| -> f64 {
            (0..3)
                .map(|i| {
                    f64::from(dy[i])
                        * (0..5)
                            .map(|j| f64::from(down[i * 5 + j]) * f64::from(m[j]))
                            .sum::<f64>()
                })
                .sum()
        };
        let m: Vec<f32> = (0..5)
            .map(|j| gate_pre[j] / (1.0 + (-gate_pre[j]).exp()) * up_pre[j])
            .collect();
        close(&dm, &derivative(&m, loss), 1e-3);
        let expected_dx = derivative(&x, |v| {
            let m: Vec<f64> = (0..5)
                .map(|j| {
                    let a: f64 = (0..3)
                        .map(|i| f64::from(gate[j * 3 + i]) * f64::from(v[i]))
                        .sum();
                    let b: f64 = (0..3)
                        .map(|i| f64::from(up[j * 3 + i]) * f64::from(v[i]))
                        .sum();
                    a / (1.0 + (-a).exp()) * b
                })
                .collect();
            (0..3)
                .map(|i| {
                    f64::from(dy[i])
                        * (0..5)
                            .map(|j| f64::from(down[i * 5 + j]) * m[j])
                            .sum::<f64>()
                })
                .sum()
        });
        close(&dx, &expected_dx, 1e-3);
    }

    #[test]
    fn swiglu_reduces_gate_separately_before_adding_the_up_branch() {
        let up_weight = 1e20_f32 / 80.0;
        let up_contribution = up_weight * 80.0;
        let probe = force_scalar();
        let (dx, dm) = probe
            .swiglu_backward(
                &[1.0],
                &[80.0, 0.0, 0.0],
                &[0.0, 2.0, 2.0],
                &[1.0; 3],
                &[0.0, -up_contribution, 1.0],
                &[up_weight, 0.0, 0.0],
                1,
                3,
            )
            .unwrap();
        completed(&probe, Operation::SwigluBackward);
        assert_eq!(dm, [1.0; 3]);
        assert_eq!(dx, [0.0]);
        let mut interleaved = up_contribution;
        for contribution in [0.0, -up_contribution, 1.0] {
            interleaved += contribution;
        }
        assert_eq!(interleaved, 1.0);
    }

    #[test]
    fn forced_reductions_bypass_shared_gemm_with_real_positive_controls() {
        for (d_in, d_out, rank) in [(1, 1, 1), (3, 5, 2), (8, 8, 3), (17, 11, 5)] {
            for mode in 0..3 {
                let w = values(d_in * d_out, 0.3);
                let mut g = values(d_out, 1.4);
                if mode == 1 {
                    for value in g.iter_mut().skip(1) {
                        *value = 0.0;
                    }
                } else if mode == 2 {
                    g.fill(0.0);
                }
                let (ordinary, calls) =
                    observe_matmul_calls(|| ops::linear_vjp(&w, &g, d_in, d_out));
                assert!(calls > 0);
                let (actual, calls) = observe_matmul_calls(|| {
                    force_scalar().linear_vjp(&w, &g, d_in, d_out).unwrap()
                });
                assert_eq!(calls, 0);
                close(&actual, &ordinary, 1e-4);
                let x = values(d_in, 2.7);
                let h = values(rank, 3.9);
                let a = values(rank * d_in, 4.1);
                let b = values(d_out * rank, 5.8);
                let (ordinary, calls) = observe_matmul_calls(|| {
                    ops::lora_vjp(&g, &x, &h, &a, &b, rank, d_in, d_out, 0.37)
                });
                assert!(calls > 0);
                let (actual, calls) = observe_matmul_calls(|| {
                    force_scalar()
                        .lora_vjp(&g, &x, &h, &a, &b, rank, d_in, d_out, 0.37)
                        .unwrap()
                });
                assert_eq!(calls, 0);
                close(&actual.0, &ordinary.0, 1e-4);
                close(&actual.1, &ordinary.1, 1e-4);
                close(&actual.2, &ordinary.2, 1e-4);
            }
        }
        for gate_pre in [vec![-80.0, -20.0, 0.0, 20.0, 80.0], values(5, 0.9)] {
            let dy = values(3, 1.2);
            let up_pre = values(5, 2.3);
            let down = values(15, 3.5);
            let gate = values(15, 4.9);
            let up = values(15, 5.4);
            let (ordinary, calls) = observe_matmul_calls(|| {
                ops::swiglu_backward(&dy, &gate_pre, &up_pre, &down, &gate, &up, 3, 5)
            });
            assert!(calls > 0);
            let (actual, calls) = observe_matmul_calls(|| {
                force_scalar()
                    .swiglu_backward(&dy, &gate_pre, &up_pre, &down, &gate, &up, 3, 5)
                    .unwrap()
            });
            assert_eq!(calls, 0);
            close(&actual.0, &ordinary.0, 1e-4);
            close(&actual.1, &ordinary.1, 1e-4);
        }
    }

    #[test]
    fn empty_dimensions_have_explicit_portable_results() {
        assert!(
            force_scalar()
                .cross_entropy_backward(&[], &[], 0, 7, 0)
                .unwrap()
                .is_empty()
        );
        assert!(
            force_scalar()
                .cross_entropy_backward(&[], &[0, 0], 2, 0, 0)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            force_scalar()
                .cross_entropy_backward(&[f32::NAN; 3], &[0], 1, 3, 1)
                .unwrap(),
            [0.0; 3]
        );
        assert_eq!(force_scalar().linear_vjp(&[], &[], 3, 0).unwrap(), [0.0; 3]);
        assert!(
            force_scalar()
                .linear_vjp(&[], &[f32::NAN], 0, 1)
                .unwrap()
                .is_empty()
        );
        let (gb, ga, dx) = force_scalar()
            .lora_vjp(&[1.0; 2], &[2.0; 3], &[], &[], &[], 0, 3, 2, f32::NAN)
            .unwrap();
        assert!(gb.is_empty() && ga.is_empty());
        assert_eq!(dx, [0.0; 3]);
        assert!(
            force_scalar()
                .rmsnorm_backward(&[], &[], f32::NAN, &[])
                .unwrap()
                .is_empty()
        );
        assert!(
            force_scalar()
                .rope_backward(&[], &[], &[], 0)
                .unwrap()
                .is_empty()
        );
        let (dx, dm) = force_scalar()
            .swiglu_backward(&[1.0; 3], &[], &[], &[], &[], &[], 3, 0)
            .unwrap();
        assert_eq!(dx, [0.0; 3]);
        assert!(dm.is_empty());
        let (dx, dm) = force_scalar()
            .swiglu_backward(&[], &[f32::NAN; 2], &[1.0; 2], &[], &[], &[], 0, 2)
            .unwrap();
        assert!(dx.is_empty());
        assert_eq!(dm, [0.0; 2]);
    }

    #[test]
    fn nonfinite_and_zero_factors_are_not_sanitized() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let ce = force_scalar()
                .cross_entropy_backward(&[bad, 0.0], &[0], 1, 2, 0)
                .unwrap();
            if bad == f32::NEG_INFINITY {
                assert_eq!(ce, [-1.0, 1.0]);
            } else {
                assert!(ce.iter().all(|x| x.is_nan()));
            }
            assert!(force_scalar().linear_vjp(&[bad], &[0.0], 1, 1).unwrap()[0].is_nan());
            let rotated = force_scalar()
                .rope_backward(&[bad, 1.0], &[0.0], &[1.0], 2)
                .unwrap();
            assert!(rotated[0].is_nan());
            if bad.is_nan() {
                assert!(rotated[1].is_nan());
            } else {
                assert_eq!(rotated[1], -bad);
            }
            let (gb, ga, dx) = force_scalar()
                .lora_vjp(&[bad], &[1.0], &[1.0], &[1.0], &[1.0], 1, 1, 1, 0.0)
                .unwrap();
            assert!(gb[0].is_nan() && ga[0].is_nan() && dx[0].is_nan());
            assert!(
                force_scalar()
                    .rmsnorm_backward(&[1.0], &[0.0], 1.0, &[bad])
                    .unwrap()[0]
                    .is_nan()
            );
            let (dx, dm) = force_scalar()
                .swiglu_backward(&[0.0], &[bad], &[1.0], &[1.0], &[1.0], &[1.0], 1, 1)
                .unwrap();
            assert!(dx[0].is_nan());
            assert_eq!(dm, [0.0]);
        }
        let ce = force_scalar()
            .cross_entropy_backward(&[0.0, 0.0], &[9], 1, 2, 0)
            .unwrap();
        assert_eq!(ce, [0.5, 0.5]);
        let ce = force_scalar()
            .cross_entropy_backward(&[f32::NEG_INFINITY; 2], &[0], 1, 2, 0)
            .unwrap();
        assert!(ce.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn into_replaces_every_output_and_scratch_element() {
        let mut ce = [f32::NAN; 4];
        force_scalar()
            .cross_entropy_backward_into(&[0.0; 4], &[0, 1], 2, 2, 1, &mut ce)
            .unwrap();
        assert_eq!(ce, [0.0, 0.0, 0.5, -0.5]);
        let mut dx = [f32::NAN; 2];
        force_scalar()
            .linear_vjp_into(&[1.0, 2.0], &[3.0], 2, 1, &mut dx)
            .unwrap();
        assert_eq!(dx, [3.0, 6.0]);
        let (mut gb, mut ga, mut bt_g) = ([f32::NAN], [f32::NAN; 2], [f32::NAN]);
        force_scalar()
            .lora_vjp_into(
                &[2.0],
                &[3.0, 4.0],
                &[5.0],
                &[6.0, 7.0],
                &[8.0],
                1,
                2,
                1,
                0.5,
                &mut gb,
                &mut ga,
                &mut dx,
                &mut bt_g,
            )
            .unwrap();
        assert_eq!(gb, [5.0]);
        assert_eq!(ga, [24.0, 32.0]);
        assert_eq!(dx, [48.0, 56.0]);
        assert_eq!(bt_g, [16.0]);
        force_scalar()
            .rmsnorm_backward_into(&[0.0; 2], &[2.0; 2], 0.5, &[3.0; 2], &mut dx)
            .unwrap();
        assert_eq!(dx, [3.0; 2]);
        force_scalar()
            .rope_backward_into(&[3.0, 4.0], &[0.0], &[1.0], 2, &mut dx)
            .unwrap();
        assert_eq!(dx, [4.0, -3.0]);
        let (mut dm, mut dg) = ([f32::NAN], [f32::NAN]);
        force_scalar()
            .swiglu_backward_into(
                &[2.0, 3.0],
                &[0.0],
                &[4.0],
                &[5.0, 6.0],
                &[7.0, 8.0],
                &[9.0, 10.0],
                2,
                1,
                &mut dx,
                &mut dm,
                &mut dg,
            )
            .unwrap();
        assert_eq!(dm, [28.0]);
        assert_eq!(dg, [56.0]);
        assert_eq!(dx, [392.0, 448.0]);
    }

    #[test]
    fn cross_entropy_rejects_all_shape_errors_before_writes() {
        for (logits, targets, seq, vocab, start, output) in [
            (1, 1, 1, 2, 0, 2),
            (2, 0, 1, 2, 0, 2),
            (2, 1, 1, 2, 2, 2),
            (2, 1, 1, 2, 0, 3),
            (0, 0, usize::MAX, 2, 0, 2),
        ] {
            let probe = force_scalar();
            let mut output = vec![13.0; output];
            rejected(
                probe.cross_entropy_backward_into(
                    &vec![0.0; logits],
                    &vec![0; targets],
                    seq,
                    vocab,
                    start,
                    &mut output,
                ),
                &probe,
                Operation::CrossEntropyBackward,
            );
            assert!(output.iter().all(|&x| x == 13.0));
        }
    }

    #[test]
    fn linear_rejects_all_shape_errors_before_writes() {
        for (weights, gradient, d_in, d_out, output) in [
            (1, 1, 2, 1, 2),
            (2, 0, 2, 1, 2),
            (2, 1, 2, 1, 3),
            (0, 0, usize::MAX, 2, 2),
        ] {
            let probe = force_scalar();
            let mut output = vec![13.0; output];
            rejected(
                probe.linear_vjp_into(
                    &vec![0.0; weights],
                    &vec![0.0; gradient],
                    d_in,
                    d_out,
                    &mut output,
                ),
                &probe,
                Operation::LinearVjp,
            );
            assert!(output.iter().all(|&x| x == 13.0));
        }
    }

    #[test]
    fn lora_rejects_each_input_output_and_scratch_shape_before_writes() {
        for bad in 0..11 {
            let mut lengths = [2, 3, 1, 3, 2, 2, 3, 3, 1];
            if bad < lengths.len() {
                lengths[bad] += 1;
            }
            let rank = if bad == 9 { usize::MAX } else { 1 };
            let d_out = if bad == 10 { usize::MAX } else { 2 };
            let d_in = if bad == 10 { 0 } else { 3 };
            let rank = if bad == 10 { 2 } else { rank };
            let (mut gb, mut ga, mut dx, mut scratch) = (
                vec![13.0; lengths[5]],
                vec![13.0; lengths[6]],
                vec![13.0; lengths[7]],
                vec![13.0; lengths[8]],
            );
            let probe = force_scalar();
            rejected(
                probe.lora_vjp_into(
                    &vec![0.0; lengths[0]],
                    &vec![0.0; lengths[1]],
                    &vec![0.0; lengths[2]],
                    &vec![0.0; lengths[3]],
                    &vec![0.0; lengths[4]],
                    rank,
                    d_in,
                    d_out,
                    0.5,
                    &mut gb,
                    &mut ga,
                    &mut dx,
                    &mut scratch,
                ),
                &probe,
                Operation::LoraVjp,
            );
            assert!(
                gb.iter()
                    .chain(&ga)
                    .chain(&dx)
                    .chain(&scratch)
                    .all(|&x| x == 13.0)
            );
        }
    }

    #[test]
    fn rmsnorm_and_rope_reject_shapes_before_writes() {
        for bad in 0..3 {
            let mut lengths = [2, 2, 2];
            lengths[bad] += 1;
            let mut dx = vec![13.0; lengths[2]];
            let probe = force_scalar();
            rejected(
                probe.rmsnorm_backward_into(
                    &[1.0; 2],
                    &vec![1.0; lengths[0]],
                    0.5,
                    &vec![1.0; lengths[1]],
                    &mut dx,
                ),
                &probe,
                Operation::RmsnormBackward,
            );
            assert!(dx.iter().all(|&x| x == 13.0));
        }
        for (rope_dim, cos, sin, output) in [
            (3, 1, 1, 4),
            (6, 3, 3, 4),
            (4, 1, 2, 4),
            (4, 2, 1, 4),
            (4, 2, 2, 5),
        ] {
            let mut dx = vec![13.0; output];
            let probe = force_scalar();
            rejected(
                probe.rope_backward_into(
                    &[1.0; 4],
                    &vec![1.0; cos],
                    &vec![1.0; sin],
                    rope_dim,
                    &mut dx,
                ),
                &probe,
                Operation::RopeBackward,
            );
            assert!(dx.iter().all(|&x| x == 13.0));
        }
    }

    #[test]
    fn swiglu_rejects_each_input_output_and_scratch_shape_before_writes() {
        for bad in 0..10 {
            let mut lengths = [2, 3, 3, 6, 6, 6, 2, 3, 3];
            if bad < lengths.len() {
                lengths[bad] += 1;
            }
            let hidden = if bad == 9 { usize::MAX } else { 2 };
            let (mut dx, mut dm, mut scratch) = (
                vec![13.0; lengths[6]],
                vec![13.0; lengths[7]],
                vec![13.0; lengths[8]],
            );
            let probe = force_scalar();
            rejected(
                probe.swiglu_backward_into(
                    &vec![0.0; lengths[0]],
                    &vec![0.0; lengths[1]],
                    &vec![0.0; lengths[2]],
                    &vec![0.0; lengths[3]],
                    &vec![0.0; lengths[4]],
                    &vec![0.0; lengths[5]],
                    hidden,
                    3,
                    &mut dx,
                    &mut dm,
                    &mut scratch,
                ),
                &probe,
                Operation::SwigluBackward,
            );
            assert!(dx.iter().chain(&dm).chain(&scratch).all(|&x| x == 13.0));
        }
    }

    #[test]
    fn zero_factor_shapes_still_reject_unallocatable_outputs_and_scratch() {
        let huge = isize::MAX as usize / std::mem::size_of::<f32>() + 1;
        let probe = force_scalar();
        capacity_rejected(
            probe.linear_vjp(&[], &[], huge, 0),
            &probe,
            Operation::LinearVjp,
        );
        for (rank, d_in, d_out) in [(0, huge, 0), (huge, 0, 0)] {
            let probe = force_scalar();
            capacity_rejected(
                probe.lora_vjp(&[], &[], &[], &[], &[], rank, d_in, d_out, 1.0),
                &probe,
                Operation::LoraVjp,
            );
        }
        for (hidden, inter) in [(huge, 0), (0, huge)] {
            let probe = force_scalar();
            capacity_rejected(
                probe.swiglu_backward(&[], &[], &[], &[], &[], &[], hidden, inter),
                &probe,
                Operation::SwigluBackward,
            );
        }
        let probe = force_scalar();
        capacity_rejected(
            probe.cross_entropy_backward(&[], &[], 1, huge, 0),
            &probe,
            Operation::CrossEntropyBackward,
        );
        let mut output = [13.0];
        let probe = force_scalar();
        capacity_rejected(
            probe.linear_vjp_into(&[], &[], huge, 0, &mut output),
            &probe,
            Operation::LinearVjp,
        );
        assert_eq!(output, [13.0]);
    }
}
