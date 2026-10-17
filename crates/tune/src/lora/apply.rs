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
/// than an out-of-bounds slice or a partial write.
///
/// `output` is checked for EXACT width, not `>= d_out`. The inequality it
/// replaced rejected an over-declared `d_out` but accepted an under-declared
/// one, and an under-declared `d_out` is the worse direction: the check passes,
/// the accumulate loop writes `output[..d_out]`, and the projection row comes
/// back with an update computed for a geometry the model does not have across
/// its prefix and base weights across the rest. No panic, no error, a finite
/// in-range result. Every consumer in this workspace declares the exact
/// projection width — both `validate_against` and `validate_against_bert`
/// reject any `d_in`/`d_out` disagreement outright, `LoraAdapter::apply`
/// documents that slices must match the layer's shape, and the engine hands
/// this function `chunks_exact_mut` rows at the projection's own width — so
/// there is no prefix-adaptation use for the inequality to serve.
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
        && output.len() == d_out
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
mod tests;
