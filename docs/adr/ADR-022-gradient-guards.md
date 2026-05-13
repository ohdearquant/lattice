# ADR-022: Gradient Guard Strategy

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-fann

## Context

Neural network training is susceptible to numeric instability. Gradients can become NaN
or Inf due to:

- **Exploding gradients**: Large weight updates causing overflow in deep networks
- **Vanishing gradients**: Near-zero values producing NaN after division operations
- **Unstable activations**: Softmax overflow, log(0), or division by zero in loss
  computation
- **Learning rate issues**: Excessively high learning rates amplifying numeric errors

Silent NaN propagation is particularly dangerous: once a single NaN enters the gradient
computation, it infects all downstream values. A network can train for epochs with
corrupted weights before the issue surfaces in evaluation metrics.

The standard approach of failing immediately on NaN detection (Section 10.1 forward pass
guards) is appropriate for inference but too strict for training. Training naturally
explores unstable regions of the loss landscape, especially in early epochs with random
initialization.

## Decision

We implement a **configurable gradient guard strategy** via the `GradientGuardStrategy`
enum with three variants:

| Strategy    | Behavior                                           | Default For            |
| ----------- | -------------------------------------------------- | ---------------------- |
| `Error`     | Return `FannError::NumericInstability` immediately | Development, debugging |
| `Sanitize`  | Replace NaN/Inf values with 0.0, continue training | Production training    |
| `SkipBatch` | Discard the problematic batch, continue with next  | Noisy data, robustness |

```rust
pub enum GradientGuardStrategy {
    /// Fail immediately on NaN/Inf (strict mode)
    Error,
    /// Replace NaN/Inf with 0.0 and continue
    Sanitize,
    /// Skip the current batch entirely
    SkipBatch,
}

impl Default for GradientGuardStrategy {
    fn default() -> Self {
        Self::Error  // Safe default: fail loudly
    }
}
```

The strategy is configured via `TrainingConfig::gradient_guard` and checked after each
gradient computation:

```rust
fn apply_gradient_guard(
    gradients: &mut [f32],
    strategy: GradientGuardStrategy,
) -> Result<GradientGuardAction, FannError> {
    let has_invalid = gradients.iter().any(|g| !g.is_finite());

    if !has_invalid {
        return Ok(GradientGuardAction::Continue);
    }

    match strategy {
        GradientGuardStrategy::Error => {
            Err(FannError::NumericInstability("NaN/Inf in gradients".into()))
        }
        GradientGuardStrategy::Sanitize => {
            for g in gradients.iter_mut() {
                if !g.is_finite() {
                    *g = 0.0;
                }
            }
            Ok(GradientGuardAction::Sanitized)
        }
        GradientGuardStrategy::SkipBatch => {
            Ok(GradientGuardAction::SkipBatch)
        }
    }
}
```

## Consequences

### Positive

- **Training robustness**: Networks can recover from transient numeric issues rather
  than failing immediately
- **Debuggability**: `Error` mode (default) ensures issues surface during development
- **Flexibility**: Different strategies suit different deployment contexts (dev vs prod)
- **Observability**: Sanitization and skip counts can be tracked for diagnostics

### Negative

- **Root cause masking**: `Sanitize` mode can hide fundamental issues (bad learning
  rate, architecture problems)
- **Silent degradation**: Excessive sanitization may produce a trained but
  poorly-performing network
- **API complexity**: Users must understand the trade-offs between strategies
- **Testing burden**: Each strategy path requires validation

### Neutral

- **Default is strict**: Choosing `Error` as default means new users see issues
  immediately; they must opt-in to lenient modes

## Alternatives Considered

### 1. No Guards (Silent Propagation)

Allow NaN/Inf to flow through training without detection.

- **Pros**: Zero overhead, simplest implementation
- **Cons**: NaN infection spreads silently, corrupted weights discovered only at
  evaluation, debugging nightmare
- **Rejected**: Violates design goal of "no silent NaN/Inf propagation" (TDS Section
  1.2)

### 2. Always Fail (Error Only)

Single strategy: immediately return error on any NaN/Inf.

- **Pros**: Simple, deterministic, forces users to fix root causes
- **Cons**: Training on noisy real-world data often encounters transient instabilities;
  strict mode makes such training infeasible
- **Rejected**: Too strict for production training workloads; would require users to
  implement their own retry/sanitization logic

### 3. Always Sanitize (Lenient Only)

Single strategy: silently replace all NaN/Inf with 0.0.

- **Pros**: Training never fails due to numeric issues
- **Cons**: Hides fundamental problems, may produce poorly-trained networks, no signal
  to users that something is wrong
- **Rejected**: Violates debugging experience; users should know when issues occur, even
  if they choose to continue

### 4. Gradient Clipping Only

Clip gradient magnitudes to a maximum value instead of detecting NaN/Inf.

- **Pros**: Prevents explosion before it happens
- **Cons**: Does not address NaN (already occurred), gradient clipping is orthogonal to
  NaN handling, can mask learning rate issues
- **Rejected**: Gradient clipping is a complementary technique (could be added), not a
  replacement for NaN/Inf detection

## References

- TDS-network-architecture.md Section 7.3: GradientGuardStrategy
- TDS-network-architecture.md Section 10: Numeric Stability
- TDS-network-architecture.md Section 1.2: Design Goals ("No silent NaN/Inf
  propagation")
