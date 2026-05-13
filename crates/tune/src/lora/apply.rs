//! Runtime LoRA application: out += scale * (B @ (A @ x)).
//!
//! Pure f32 CPU math for single-token decode. Metal GPU LoRA is a future extension.
//!
//! The LoRA decomposition replaces a weight update dW with two low-rank matrices:
//! dW = B @ A, where A is (rank, d_in) and B is (d_out, rank).
//! During inference: output += (alpha / rank) * B @ (A @ x).

use super::LoraLayer;

/// Apply a single LoRA layer to a base projection output.
///
/// Computes: `output[0..d_out] += scale * B @ (A @ x)`
///
/// where:
/// - `x` has length `lora.d_in`
/// - `output` has length `lora.d_out`
/// - `A` is row-major `(rank, d_in)`
/// - `B` is row-major `(d_out, rank)`
///
/// This is the decode hot path (single token, M=1), so we use two
/// sequential matvecs rather than a fused kernel.
pub fn apply_lora(lora: &LoraLayer, scale: f32, x: &[f32], output: &mut [f32]) {
    debug_assert_eq!(x.len(), lora.d_in, "x length must equal d_in");
    debug_assert!(output.len() >= lora.d_out, "output length must be >= d_out");

    // Step 1: intermediate = A @ x  -> shape (rank,)
    // A is row-major (rank, d_in), so row r dot x gives intermediate[r].
    let rank = lora.rank;
    let d_in = lora.d_in;
    let d_out = lora.d_out;

    let mut intermediate = vec![0.0f32; rank];
    for (r, inter) in intermediate.iter_mut().enumerate() {
        let row = &lora.a[r * d_in..(r + 1) * d_in];
        let acc: f32 = row.iter().zip(x.iter()).map(|(a, x)| a * x).sum();
        *inter = acc;
    }

    // Step 2: output += scale * B @ intermediate  -> accumulate into (d_out,)
    // B is row-major (d_out, rank), so row r dot intermediate gives the contribution.
    for (r, out) in output[..d_out].iter_mut().enumerate() {
        let row = &lora.b[r * rank..(r + 1) * rank];
        let acc: f32 = row
            .iter()
            .zip(intermediate.iter())
            .map(|(b, i)| b * i)
            .sum();
        *out += scale * acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_lora_identity_like() {
        // rank=1, d_in=3, d_out=2
        // A = [[1, 0, 0]]  (1x3)
        // B = [[1], [0]]    (2x1)
        // x = [5, 3, 1]
        // A @ x = [5]
        // B @ [5] = [5, 0]
        // scale = 2.0 => output += [10, 0]
        let lora = LoraLayer {
            a: vec![1.0, 0.0, 0.0],
            b: vec![1.0, 0.0],
            d_in: 3,
            d_out: 2,
            rank: 1,
        };

        let x = [5.0, 3.0, 1.0];
        let mut output = [100.0, 200.0]; // pre-existing base output
        apply_lora(&lora, 2.0, &x, &mut output);

        assert!((output[0] - 110.0).abs() < 1e-6);
        assert!((output[1] - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_lora_rank2() {
        // rank=2, d_in=2, d_out=2
        // A = [[1, 0], [0, 1]]  (2x2, identity)
        // B = [[1, 2], [3, 4]]  (2x2)
        // x = [1, 1]
        // A @ x = [1, 1]
        // B @ [1, 1] = [3, 7]
        // scale = 0.5 => output += [1.5, 3.5]
        let lora = LoraLayer {
            a: vec![1.0, 0.0, 0.0, 1.0],
            b: vec![1.0, 2.0, 3.0, 4.0],
            d_in: 2,
            d_out: 2,
            rank: 2,
        };

        let x = [1.0, 1.0];
        let mut output = [0.0, 0.0];
        apply_lora(&lora, 0.5, &x, &mut output);

        assert!((output[0] - 1.5).abs() < 1e-6);
        assert!((output[1] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_lora_zero_scale() {
        let lora = LoraLayer {
            a: vec![1.0, 2.0],
            b: vec![3.0, 4.0],
            d_in: 2,
            d_out: 2,
            rank: 1,
        };

        let x = [1.0, 1.0];
        let mut output = [10.0, 20.0];
        apply_lora(&lora, 0.0, &x, &mut output);

        // Output should be unchanged
        assert!((output[0] - 10.0).abs() < 1e-6);
        assert!((output[1] - 20.0).abs() < 1e-6);
    }
}
