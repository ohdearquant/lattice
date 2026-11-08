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

        // Lazily initialise moment vectors to zero.
        let m = self
            .m
            .entry(key.to_string())
            .or_insert_with(|| vec![0.0f32; n]);
        let v = self
            .v
            .entry(key.to_string())
            .or_insert_with(|| vec![0.0f32; n]);

        // Advance only this tensor's timestep for correct bias correction.
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

impl AdamState {
    /// Encode every key's moments and step count, keys in byte order, so a run
    /// stopped between steps can continue with the same optimizer state.
    ///
    /// Layout, little-endian: `u32` key count, then per key a `u32` name length,
    /// the UTF-8 name, `u64` step count, `u64` element count `n`, `n` first-moment
    /// `f32` values and `n` second-moment `f32` values. [`AdamState::from_bytes`]
    /// is the inverse and restores every value bit for bit.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut keys: Vec<&String> = self.m.keys().collect();
        keys.sort();
        let capacity = keys.iter().try_fold(4usize, |total, &key| {
            let moments = self.m[key]
                .len()
                .checked_add(self.v.get(key).map_or(0, Vec::len))?;
            total
                .checked_add(4 + 8 + 8)?
                .checked_add(key.len())?
                .checked_add(moments.checked_mul(size_of::<f32>())?)
        });
        let mut out = Vec::with_capacity(capacity.unwrap_or(0));
        out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
        for key in keys {
            let m = &self.m[key];
            let empty = Vec::new();
            let v = self.v.get(key).unwrap_or(&empty);
            let t = self.t.get(key).copied().unwrap_or(0);
            out.extend_from_slice(&(key.len() as u32).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            out.extend_from_slice(&(t as u64).to_le_bytes());
            out.extend_from_slice(&(m.len() as u64).to_le_bytes());
            for x in m {
                out.extend_from_slice(&x.to_le_bytes());
            }
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// Decode a state written by [`AdamState::to_bytes`].
    ///
    /// Refuses truncated or trailing bytes, a non-UTF-8 or repeated key, a zero
    /// step count, and any length that the remaining input cannot hold, so a
    /// damaged file is an error rather than a partial or oversized state.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TuneError> {
        fn take<'a>(input: &mut &'a [u8], n: usize, what: &str) -> Result<&'a [u8], TuneError> {
            if input.len() < n {
                return Err(TuneError::Training(format!(
                    "optimizer state truncated reading {what}: need {n} bytes, {} left",
                    input.len()
                )));
            }
            let (head, rest) = input.split_at(n);
            *input = rest;
            Ok(head)
        }
        fn u32_at(input: &mut &[u8], what: &str) -> Result<u32, TuneError> {
            let b = take(input, 4, what)?;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
        fn u64_at(input: &mut &[u8], what: &str) -> Result<u64, TuneError> {
            let b = take(input, 8, what)?;
            let mut a = [0u8; 8];
            a.copy_from_slice(b);
            Ok(u64::from_le_bytes(a))
        }
        fn floats(input: &mut &[u8], n: usize, what: &str) -> Result<Vec<f32>, TuneError> {
            let len = n.checked_mul(4).ok_or_else(|| {
                TuneError::Training(format!("optimizer state {what} length overflows"))
            })?;
            Ok(take(input, len, what)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }

        let mut input = bytes;
        let count = u32_at(&mut input, "key count")? as usize;
        let mut state = Self::new();
        for index in 0..count {
            let name_len = u32_at(&mut input, "key length")? as usize;
            let name = std::str::from_utf8(take(&mut input, name_len, "key name")?)
                .map_err(|_| {
                    TuneError::Training(format!("optimizer state key {index} is not UTF-8"))
                })?
                .to_string();
            let t = u64_at(&mut input, "step count")?;
            if t == 0 {
                return Err(TuneError::Training(format!(
                    "optimizer state key {name} has step count 0"
                )));
            }
            let t = usize::try_from(t).map_err(|_| {
                TuneError::Training(format!("optimizer state key {name} step count overflows"))
            })?;
            let n = usize::try_from(u64_at(&mut input, "element count")?).map_err(|_| {
                TuneError::Training(format!("optimizer state key {name} length overflows"))
            })?;
            if n.checked_mul(8).is_none_or(|need| need > input.len()) {
                return Err(TuneError::Training(format!(
                    "optimizer state key {name} claims {n} elements; {} bytes remain",
                    input.len()
                )));
            }
            let m = floats(&mut input, n, "first moment")?;
            let v = floats(&mut input, n, "second moment")?;
            if state.m.contains_key(&name) {
                return Err(TuneError::Training(format!(
                    "optimizer state repeats key {name}"
                )));
            }
            state.m.insert(name.clone(), m);
            state.v.insert(name.clone(), v);
            state.t.insert(name, t);
        }
        if !input.is_empty() {
            return Err(TuneError::Training(format!(
                "optimizer state has {} trailing bytes",
                input.len()
            )));
        }
        Ok(state)
    }

    /// Element count of each key's moments, for checking a decoded state
    /// against the tensors it will update.
    pub fn key_lengths(&self) -> HashMap<String, usize> {
        self.m.iter().map(|(k, m)| (k.clone(), m.len())).collect()
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
mod tests;
