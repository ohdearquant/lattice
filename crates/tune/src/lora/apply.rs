//! Runtime LoRA application: out += scale * (B @ (A @ x)).
//!
//! Pure f32 CPU math for single-token decode. Metal GPU LoRA is a future extension.
//!
//! The LoRA decomposition replaces a weight update dW with two low-rank matrices:
//! dW = B @ A, where A is (rank, d_in) and B is (d_out, rank).
//! During inference: output += (alpha / rank) * B @ (A @ x).

use super::LoraLayer;

/// Ranks at or below this bound use a stack buffer for the `A @ x`
/// intermediate instead of a heap `Vec`. Covers every documented typical
/// rank (4-64, see [`super::LoraConfig::rank`]) with headroom; larger ranks
/// fall back to a heap allocation and stay correct, just not allocation-free.
const STACK_RANK_CAPACITY: usize = 128;

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
/// sequential matvecs rather than a fused kernel. Called once per hooked
/// row, so the `A @ x` intermediate is scratch space, not output — see
/// `STACK_RANK_CAPACITY` for why it avoids a heap allocation.
///
/// Fails closed: if `x`, `output`, or the layer's own `a`/`b` buffers don't
/// match the layer's declared `rank`/`d_in`/`d_out`, this is a no-op rather
/// than an out-of-bounds slice.
pub fn apply_lora(lora: &LoraLayer, scale: f32, x: &[f32], output: &mut [f32]) {
    // Adapter data is untrusted input: a `LoraLayer` reaching this function
    // may not satisfy its own declared geometry (e.g. an empty or
    // short buffer that slipped past an upstream validation guard, since
    // this function is reachable from more than one caller). Rather than
    // trust the caller and slice unchecked, verify the buffers actually
    // match the declared `rank`/`d_in`/`d_out` and no-op otherwise.
    let rank = lora.rank;
    let d_in = lora.d_in;
    let d_out = lora.d_out;
    let shapes_match = x.len() == d_in
        && output.len() >= d_out
        && rank.checked_mul(d_in) == Some(lora.a.len())
        && d_out.checked_mul(rank) == Some(lora.b.len());
    if !shapes_match {
        return;
    }

    // Step 1: intermediate = A @ x  -> shape (rank,)
    // A is row-major (rank, d_in), so row r dot x gives intermediate[r].
    let mut stack_buf = [0.0f32; STACK_RANK_CAPACITY];
    let mut heap_buf = Vec::new();
    let intermediate: &mut [f32] = if rank <= STACK_RANK_CAPACITY {
        &mut stack_buf[..rank]
    } else {
        heap_buf.resize(rank, 0.0);
        &mut heap_buf
    };
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

    /// Ranks above `STACK_RANK_CAPACITY` take the heap-fallback branch for
    /// the `A @ x` intermediate. `A` is the `rank x rank` identity, so
    /// `intermediate == x`; `B` is a single all-ones row, so the result is
    /// `scale * sum(x)` — checkable without hand-writing 200 products.
    #[test]
    fn test_apply_lora_rank_above_stack_capacity_uses_heap_fallback_correctly() {
        let rank = STACK_RANK_CAPACITY + 72;
        let mut a = vec![0.0f32; rank * rank];
        for i in 0..rank {
            a[i * rank + i] = 1.0;
        }
        let b = vec![1.0f32; rank];
        let lora = LoraLayer {
            a,
            b,
            d_in: rank,
            d_out: 1,
            rank,
        };

        let x: Vec<f32> = (0..rank).map(|i| i as f32 * 0.01).collect();
        let expected: f32 = x.iter().sum();

        let mut output = [0.0f32];
        apply_lora(&lora, 2.0, &x, &mut output);

        assert!(
            (output[0] - 2.0 * expected).abs() < 1e-3,
            "got {}, want {}",
            output[0],
            2.0 * expected
        );
    }

    /// `apply_lora` must never index into a layer's `a`/`b` buffers past
    /// their actual length: a layer whose declared `rank`/`d_in` implies a
    /// buffer larger than what `a` actually holds (malformed/untrusted
    /// adapter data that reached this function through some path other than
    /// `LoraAdapter::new`) must be a no-op, not an out-of-bounds slice.
    #[test]
    fn test_apply_lora_mismatched_a_buffer_is_noop_not_panic() {
        let lora = LoraLayer {
            a: vec![], // declares rank=1, d_in=3 but holds zero elements
            b: vec![1.0, 0.0],
            d_in: 3,
            d_out: 2,
            rank: 1,
        };

        let x = [5.0, 3.0, 1.0];
        let mut output = [100.0, 200.0];
        apply_lora(&lora, 2.0, &x, &mut output);

        assert_eq!(
            output,
            [100.0, 200.0],
            "mismatched buffer must leave output untouched"
        );
    }

    /// Simulates `apply_lora_rows` calling `apply_lora` once per token row:
    /// each call's stack scratch buffer must be reinitialized fresh, not
    /// carry state from the previous row.
    #[test]
    fn test_apply_lora_repeated_calls_are_independent_per_row() {
        let lora = LoraLayer {
            a: vec![1.0, 0.0, 0.0, 1.0],
            b: vec![1.0, 2.0, 3.0, 4.0],
            d_in: 2,
            d_out: 2,
            rank: 2,
        };

        let rows = [[1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [5.0, -1.0]];
        for row in rows {
            let mut output = [0.0, 0.0];
            apply_lora(&lora, 1.0, &row, &mut output);

            // A = identity => intermediate = row.
            let expected0 = row[0] + 2.0 * row[1];
            let expected1 = 3.0 * row[0] + 4.0 * row[1];
            assert!((output[0] - expected0).abs() < 1e-6);
            assert!((output[1] - expected1).abs() < 1e-6);
        }
    }
}
