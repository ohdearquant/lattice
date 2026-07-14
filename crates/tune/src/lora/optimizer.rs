//! Adam/AdamW state and exact single-layer LoRA gradient computation.
//!
//! Adam state and bias correction are tracked per parameter key, never with a
//! global timestep. [`compute_lora_gradients`] uses the same MSE arithmetic as
//! [`adapt_step`](super::online::adapt_step) without changing adapter weights.
//!
//! See `docs/lora-core.md` for the optimizer and local-refit equations.

use std::collections::HashMap;

use super::LoraAdapter;
use crate::error::TuneError;

/// Stateful Adam/AdamW moments and per-tensor bias-correction counts.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#adamstatestep) for the timestep invariant.
pub struct AdamState {
    /// First moment estimates (exponential moving average of gradients).
    m: HashMap<String, Vec<f32>>,
    /// Second moment estimates (exponential moving average of squared gradients).
    v: HashMap<String, Vec<f32>>,
    /// Per-key step counter (1-based once a key has been stepped); each key's
    /// own count drives its bias-correction denominators.
    pub t: HashMap<String, usize>,
}

impl AdamState {
    /// Create a fresh optimizer state with zero moments and step counter.
    pub fn new() -> Self {
        Self {
            m: HashMap::new(),
            v: HashMap::new(),
            t: HashMap::new(),
        }
    }

    /// Perform an Adam or AdamW update on `params` using `grads`.
    /// `key` identifies a stable, fixed-length tensor; `decoupled` selects AdamW.
    ///
    /// # Panics
    ///
    /// Panics when `params` and `grads` have different lengths.
    /// See [`docs/lora-core.md`](../../docs/lora-core.md#adamstatestep) for the update and per-key timestep policy.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        key: &str,
        params: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        decoupled: bool,
    ) {
        assert_eq!(
            params.len(),
            grads.len(),
            "params and grads must have the same length"
        );
        let n = params.len();

        let m = self
            .m
            .entry(key.to_string())
            .or_insert_with(|| vec![0.0f32; n]);
        let v = self
            .v
            .entry(key.to_string())
            .or_insert_with(|| vec![0.0f32; n]);

        // Bias correction uses a timestep per tensor, never a shared counter.
        let t = {
            let c = self.t.entry(key.to_string()).or_insert(0);
            *c += 1;
            *c
        };

        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        for i in 0..n {
            let g = grads[i];

            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;

            // AdamW decouples weight decay from the gradient update.
            if decoupled && weight_decay != 0.0 {
                params[i] -= lr * weight_decay * params[i];
            }

            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

impl Default for AdamState {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradients of the LoRA MSE loss with respect to a single layer's A and B matrices.
#[derive(Debug)]
pub struct LoraGradients {
    /// Gradient with respect to B, row-major `(d_out, rank)`.
    pub grad_b: Vec<f32>,
    /// Gradient with respect to A, row-major `(rank, d_in)`.
    pub grad_a: Vec<f32>,
    /// MSE loss `||scale·B·A·input - target||²` (sum, not mean).
    pub loss: f32,
}

/// Compute local MSE gradients without applying an update.
///
/// Uses the same forward and backward arithmetic as
/// [`adapt_step`](super::online::adapt_step). Returns [`TuneError::Training`]
/// for a missing layer and [`TuneError::DimensionMismatch`] for bad input sizes.
pub fn compute_lora_gradients(
    adapter: &LoraAdapter,
    layer_idx: usize,
    module: &str,
    input: &[f32],
    target_delta: &[f32],
) -> Result<LoraGradients, TuneError> {
    let scale = adapter.config().scale();

    let lora = adapter
        .layers()
        .get(&(layer_idx, module.to_string()))
        .ok_or_else(|| TuneError::Training(format!("no LoRA layer for ({layer_idx}, {module})")))?;

    let rank = lora.rank;
    let d_in = lora.d_in;
    let d_out = lora.d_out;

    if input.len() != d_in {
        return Err(TuneError::DimensionMismatch {
            expected: d_in,
            actual: input.len(),
        });
    }
    if target_delta.len() != d_out {
        return Err(TuneError::DimensionMismatch {
            expected: d_out,
            actual: target_delta.len(),
        });
    }

    let mut intermediate = vec![0.0f32; rank];
    for r in 0..rank {
        let row = &lora.a[r * d_in..(r + 1) * d_in];
        intermediate[r] = row.iter().zip(input.iter()).map(|(a, x)| a * x).sum();
    }

    let mut delta = vec![0.0f32; d_out];
    for i in 0..d_out {
        let row = &lora.b[i * rank..(i + 1) * rank];
        let acc: f32 = row
            .iter()
            .zip(intermediate.iter())
            .map(|(b, h)| b * h)
            .sum();
        delta[i] = scale * acc;
    }

    let error: Vec<f32> = delta
        .iter()
        .zip(target_delta.iter())
        .map(|(d, t)| d - t)
        .collect();
    let loss: f32 = error.iter().map(|e| e * e).sum();

    let mut grad_b = vec![0.0f32; d_out * rank];
    for i in 0..d_out {
        for r in 0..rank {
            grad_b[i * rank + r] = 2.0 * scale * error[i] * intermediate[r];
        }
    }

    let mut bt_error = vec![0.0f32; rank];
    for r in 0..rank {
        bt_error[r] = (0..d_out).map(|i| lora.b[i * rank + r] * error[i]).sum();
    }

    let mut grad_a = vec![0.0f32; rank * d_in];
    for r in 0..rank {
        for j in 0..d_in {
            grad_a[r * d_in + j] = 2.0 * scale * bt_error[r] * input[j];
        }
    }

    Ok(LoraGradients {
        grad_b,
        grad_a,
        loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{LoraAdapter, LoraConfig, LoraLayer};
    use std::collections::HashMap;

    /// rank=2, d_in=3, d_out=2, scale=1.0 (alpha=2.0, rank=2)
    fn make_small_adapter() -> LoraAdapter {
        let config = LoraConfig {
            rank: 2,
            alpha: 2.0,
            target_modules: vec!["q_proj".into()],
        };
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".into()),
            LoraLayer {
                a: vec![
                    1.0, 0.5, 0.0, // row 0
                    0.0, 0.5, 1.0, // row 1
                ],
                b: vec![
                    1.0, 0.0, // row 0
                    0.0, 1.0, // row 1
                ],
                d_in: 3,
                d_out: 2,
                rank: 2,
            },
        );
        LoraAdapter::new(config, layers).expect("valid adapter config")
    }

    #[test]
    fn test_gradient_computation() {
        let adapter = make_small_adapter();
        let input = vec![1.0f32, 2.0, 3.0];
        let target_delta = vec![5.0f32, 5.0];

        let grads = compute_lora_gradients(&adapter, 0, "q_proj", &input, &target_delta)
            .expect("compute_lora_gradients must succeed");

        assert_eq!(
            grads.grad_b.len(),
            4,
            "grad_b must have d_out*rank elements"
        );
        assert_eq!(grads.grad_a.len(), 6, "grad_a must have rank*d_in elements");
        assert!(grads.loss > 0.0, "loss must be positive");
        assert!(
            grads.grad_b.iter().any(|&g| g.abs() > 1e-9)
                || grads.grad_a.iter().any(|&g| g.abs() > 1e-9),
            "gradients must be non-zero"
        );
    }

    #[test]
    fn test_gradient_computation_missing_layer() {
        let adapter = make_small_adapter();
        let err = compute_lora_gradients(&adapter, 99, "missing", &[1.0, 2.0], &[1.0])
            .expect_err("missing layer must return Err");
        assert!(
            matches!(err, TuneError::Training(_)),
            "expected Training variant, got {err:?}"
        );
    }

    #[test]
    fn test_gradient_computation_dimension_mismatch() {
        let adapter = make_small_adapter();
        let err = compute_lora_gradients(&adapter, 0, "q_proj", &[1.0, 2.0], &[1.0, 1.0])
            .expect_err("wrong input length must return Err");
        assert!(
            matches!(err, TuneError::DimensionMismatch { .. }),
            "expected DimensionMismatch, got {err:?}"
        );
    }

    #[test]
    fn test_adam_step_updates_params() {
        let mut state = AdamState::new();
        let mut params = vec![1.0f32, 2.0, 3.0];
        let grads = vec![0.1f32, 0.2, 0.3];
        let original = params.clone();

        state.step(
            "test",
            &mut params,
            &grads,
            0.01,
            0.9,
            0.999,
            1e-8,
            0.0,
            false,
        );

        for (p, o) in params.iter().zip(original.iter()) {
            assert_ne!(p, o, "params must change after Adam step");
        }
        assert_eq!(
            state.t["test"], 1,
            "key 'test' step counter must be 1 after first step"
        );
    }

    /// Verify that each key advances its own independent timestep counter.
    ///
    /// The prior shared-counter bug caused the 2nd..Nth key stepped in a single
    /// optimizer round to receive an inflated `t`, defeating Adam's warmup. This
    /// test exercises the exact failure mode: step keys a/b/c/d once each, then
    /// step only "a" again. Every key must reflect only its own update count.
    #[test]
    fn test_adam_per_key_timestep_no_cross_advance() {
        let mut state = AdamState::new();
        let grads = vec![0.1f32, 0.2];
        let mut pa = vec![1.0f32, 1.0];
        let mut pb = vec![2.0f32, 2.0];
        let mut pc = vec![3.0f32, 3.0];
        let mut pd = vec![4.0f32, 4.0];

        for (key, params) in [
            ("a", &mut pa),
            ("b", &mut pb),
            ("c", &mut pc),
            ("d", &mut pd),
        ] {
            state.step(key, params, &grads, 0.01, 0.9, 0.999, 1e-8, 0.0, false);
        }

        assert_eq!(state.t["a"], 1, "key 'a' must be at t=1 after one step");
        assert_eq!(state.t["b"], 1, "key 'b' must be at t=1 after one step");
        assert_eq!(state.t["c"], 1, "key 'c' must be at t=1 after one step");
        assert_eq!(state.t["d"], 1, "key 'd' must be at t=1 after one step");

        state.step("a", &mut pa, &grads, 0.01, 0.9, 0.999, 1e-8, 0.0, false);

        assert_eq!(
            state.t["a"], 2,
            "key 'a' must advance to t=2 on its second step"
        );
        assert_eq!(
            state.t["b"], 1,
            "key 'b' must not advance when only 'a' is stepped"
        );
        assert_eq!(
            state.t["c"], 1,
            "key 'c' must not advance when only 'a' is stepped"
        );
        assert_eq!(
            state.t["d"], 1,
            "key 'd' must not advance when only 'a' is stepped"
        );
    }
}
