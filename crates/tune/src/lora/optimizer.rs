//! Adam and AdamW optimizers for LoRA weight updates.
//!
//! Provides [`AdamState`] for stateful Adam/AdamW updates and
//! [`compute_lora_gradients`] which mirrors the forward/backward pass of
//! [`adapt_step`](super::online::adapt_step) without applying the update.

use std::collections::HashMap;

use super::LoraAdapter;
use crate::error::TuneError;

/// Stateful Adam / AdamW optimizer.
///
/// Tracks per-parameter first and second moment estimates keyed by an
/// arbitrary string (typically `"{layer_idx}_{module}_{param}"` where
/// `param` is `"a"` or `"b"`).  Each key advances its OWN Adam timestep, so
/// bias correction reflects how many updates that tensor has received — not
/// how many `step` calls occurred across all keys.  (A single shared counter
/// over-advances `t` whenever one optimiser step updates several tensors, as
/// LoRA training does — every tensor then bias-corrects with a `t` far larger
/// than its true update count, inflating m̂/√v̂ and over-stepping early updates,
/// which defeats Adam's warmup.  Per-key `t` matches MLX/PyTorch, which key the
/// timestep per parameter.)
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

    /// Perform one Adam (or AdamW) update step on `params` in place.
    ///
    /// # Arguments
    ///
    /// * `key` - Unique string key identifying this parameter tensor.
    /// * `params` - Parameter slice to update in place.
    /// * `grads` - Gradient slice (same length as `params`).
    /// * `lr` - Learning rate for this step (may be scheduled externally).
    /// * `beta1` - First moment decay (typical: 0.9).
    /// * `beta2` - Second moment decay (typical: 0.999).
    /// * `eps` - Numerical stability constant (typical: 1e-8).
    /// * `weight_decay` - L2 / decoupled weight decay coefficient.
    /// * `decoupled` - If `true`, apply AdamW-style decoupled weight decay
    ///   (`θ -= lr * λ * θ`) before the Adam gradient step.  If `false`,
    ///   weight_decay is unused (standard Adam).
    ///
    /// # Panics
    ///
    /// Panics if `params.len() != grads.len()`.
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

        // Lazily initialise moment vectors to zero.
        let m = self
            .m
            .entry(key.to_string())
            .or_insert_with(|| vec![0.0f32; n]);
        let v = self
            .v
            .entry(key.to_string())
            .or_insert_with(|| vec![0.0f32; n]);

        // Advance THIS key's own timestep. step() is called once per parameter
        // tensor, so a single shared counter would treat the 2nd..Nth tensor of
        // one optimiser step as if many steps had elapsed — bias-correcting with
        // an inflated `t` (m̂/√v̂ runs too large, over-stepping early updates and
        // defeating Adam's warmup). Per-key `t` matches MLX/PyTorch.
        let t = {
            let c = self.t.entry(key.to_string()).or_insert(0);
            *c += 1;
            *c
        };

        // Bias-correction denominators (this key's own t).
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        for i in 0..n {
            let g = grads[i];

            // Update biased first moment: m = β₁·m + (1-β₁)·g
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            // Update biased second moment: v = β₂·v + (1-β₂)·g²
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            // Bias-corrected estimates.
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;

            // AdamW: decoupled weight decay applied before the gradient step.
            if decoupled && weight_decay != 0.0 {
                params[i] -= lr * weight_decay * params[i];
            }

            // θ -= lr · m̂ / (√v̂ + ε)
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

impl Default for AdamState {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Gradient computation ────────────────────────────────────────────────────

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

/// Compute LoRA gradients without applying an update.
///
/// Mirrors the forward / backward arithmetic in [`adapt_step`](super::online::adapt_step)
/// exactly, returning raw gradients instead of performing an SGD step.
///
/// # Forward pass
///
/// ```text
/// intermediate = A @ input          (rank,)
/// delta        = scale * B @ intermediate  (d_out,)
/// error        = delta - target_delta
/// loss         = ||error||²
/// ```
///
/// # Backward pass
///
/// ```text
/// dL/dB[i,r] = 2 · scale · error[i] · intermediate[r]
/// dL/dA[r,j] = 2 · scale · (Bᵀ·error)[r] · input[j]
/// ```
///
/// # Errors
///
/// * [`TuneError::Training`]          – layer `(layer_idx, module)` not found.
/// * [`TuneError::DimensionMismatch`] – `input` or `target_delta` wrong length.
pub fn compute_lora_gradients(
    adapter: &LoraAdapter,
    layer_idx: usize,
    module: &str,
    input: &[f32],
    target_delta: &[f32],
) -> Result<LoraGradients, TuneError> {
    let scale = adapter.config.scale();

    let lora = adapter
        .layers
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

    // Forward: intermediate = A @ input  (rank,)
    let mut intermediate = vec![0.0f32; rank];
    for r in 0..rank {
        let row = &lora.a[r * d_in..(r + 1) * d_in];
        intermediate[r] = row.iter().zip(input.iter()).map(|(a, x)| a * x).sum();
    }

    // Forward: delta = scale * B @ intermediate  (d_out,)
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

    // Residual and loss (sum of squared errors, matching adapt_step).
    let error: Vec<f32> = delta
        .iter()
        .zip(target_delta.iter())
        .map(|(d, t)| d - t)
        .collect();
    let loss: f32 = error.iter().map(|e| e * e).sum();

    // dL/dB[i*rank + r] = 2 * scale * error[i] * intermediate[r]
    let mut grad_b = vec![0.0f32; d_out * rank];
    for i in 0..d_out {
        for r in 0..rank {
            grad_b[i * rank + r] = 2.0 * scale * error[i] * intermediate[r];
        }
    }

    // bt_error[r] = Σ_i B[i*rank + r] * error[i]
    let mut bt_error = vec![0.0f32; rank];
    for r in 0..rank {
        bt_error[r] = (0..d_out).map(|i| lora.b[i * rank + r] * error[i]).sum();
    }

    // dL/dA[r*d_in + j] = 2 * scale * bt_error[r] * input[j]
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
        LoraAdapter::new(config, layers)
    }

    // ─── gradient computation tests ────────────────────────────────────────

    #[test]
    fn test_gradient_computation() {
        let adapter = make_small_adapter();
        let input = vec![1.0f32, 2.0, 3.0];
        let target_delta = vec![5.0f32, 5.0];

        let grads = compute_lora_gradients(&adapter, 0, "q_proj", &input, &target_delta)
            .expect("compute_lora_gradients must succeed");

        // grad_b shape: d_out * rank = 2 * 2 = 4
        assert_eq!(
            grads.grad_b.len(),
            4,
            "grad_b must have d_out*rank elements"
        );
        // grad_a shape: rank * d_in = 2 * 3 = 6
        assert_eq!(grads.grad_a.len(), 6, "grad_a must have rank*d_in elements");
        // Loss must be positive because delta != target.
        assert!(grads.loss > 0.0, "loss must be positive");
        // At least one gradient element must be non-zero.
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
        // input too short (d_in=3, passing 2)
        let err = compute_lora_gradients(&adapter, 0, "q_proj", &[1.0, 2.0], &[1.0, 1.0])
            .expect_err("wrong input length must return Err");
        assert!(
            matches!(err, TuneError::DimensionMismatch { .. }),
            "expected DimensionMismatch, got {err:?}"
        );
    }

    // ─── Adam state tests ───────────────────────────────────────────────────

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

        // Step each key once.
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

        // Step only "a" a second time.
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

    // Integration tests for AdamW weight decay and Adam-vs-SGD convergence
    // require the train_lora loop (issue #88). They will land alongside that.
}
