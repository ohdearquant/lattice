//! Weighted blending of multiple LoRA adapters into a single rank-Σr adapter.
//!
//! # Why blend instead of loading multiple adapters
//!
//! The Metal inference path holds exactly one `Option<MetalLoraAdapter>` slot.
//! Rather than adding a Metal kernel that dispatches across N adapters,
//! we pre-blend on the CPU once per request: the result is a standard single
//! adapter loaded through the existing single-slot path with no kernel changes.
//!
//! # Blend math (exact, not approximate)
//!
//! For adapter `e` with effective scale `s_e = alpha_e / rank_e` and mixture
//! weight `w_e`:
//!
//! ```text
//! Δ_e x = s_e · B_e @ (A_e @ x)
//! blend  = Σ_e  w_e · Δ_e x
//!        = B_blend @ (A_blend @ x)
//! ```
//!
//! where
//! - `A_blend = vconcat[A_1; …; A_N]`          shape `(Σr_e) × d_in`
//! - `B_blend = hconcat[(w_1·s_1)·B_1 | …]`    shape `d_out × (Σr_e)`
//!
//! The `w_e · s_e` coefficient is folded into the B column block so the
//! blended adapter's effective scale is exactly **1.0** (`alpha = rank_total`).

use crate::error::{Result, TuneError};
use crate::lora::{LoraAdapter, LoraConfig, LoraLayer};
use std::collections::HashMap;

/// Maximum total rank allowed across all adapters in a single blend.
///
/// The GEMV kernels that consume the blended adapter assume a modest rank budget
/// (≤ ~64 per adapter in a typical mixture).  This cap allows a generous mixture
/// while bounding allocations and rejecting adversarially large adapter pools.
pub(crate) const MAX_BLEND_RANK_TOTAL: usize = 4096;

/// Aggregate cap on a blended adapter's total element count, summed across
/// every (layer_idx, module) projection: Σ rank_total·(d_in + d_out). At f32
/// this bounds the blended-adapter allocation to ~4 GiB. MAX_BLEND_RANK_TOTAL
/// bounds one projection; this bounds the whole blend, so a full-model adapter
/// that keeps every projection near the per-group cap cannot drive a multi-GiB
/// aggregate allocation. A realistic micro-LoRA mixture is far below this; a
/// large-model mixture at modest summed rank stays within budget, while a
/// rank-4096-everywhere adapter (tens of GiB) is rejected before any allocation.
pub(crate) const MAX_BLEND_TOTAL_ELEMENTS: usize = 1 << 30; // 1,073,741,824 elements ≈ 4 GiB f32

/// Blend a set of `(adapter, mixture_weight)` pairs into one rank-Σr adapter.
///
/// Each adapter in `adapters` contributes to the blended result with its
/// corresponding weight `w_e`.  The per-adapter `alpha/rank` scale is folded
/// into the B column block, so the returned adapter's effective scale is 1.0
/// (i.e., `LoraConfig { alpha: rank_total, rank: rank_total }`).
///
/// # Target-module semantics
///
/// For each `(layer_idx, module)` pair that appears in **any** of the input
/// adapters, a blended layer is produced.  An adapter that does not cover a
/// given pair contributes zero (it is simply absent from the vertical/horizontal
/// concatenation for that pair).
///
/// # Errors
///
/// Returns an error if:
/// - `adapters` is empty
/// - Any weight is not finite
/// - Two adapters share a `(layer_idx, module)` pair but disagree on `d_in` or `d_out`
/// - The summed rank for a single projection exceeds `MAX_BLEND_RANK_TOTAL`
/// - The aggregate blend size across all projections exceeds `MAX_BLEND_TOTAL_ELEMENTS`
/// - A layer's A or B buffer length does not match its declared `rank`, `d_in`, or `d_out`
/// - Rank accumulation or a size product overflows `usize`
pub fn blend_lora_adapters(adapters: &[(&LoraAdapter, f32)]) -> Result<LoraAdapter> {
    if adapters.is_empty() {
        return Err(TuneError::Validation(
            "blend_lora_adapters: adapters slice must not be empty".into(),
        ));
    }

    for (idx, (_, w)) in adapters.iter().enumerate() {
        if !w.is_finite() {
            return Err(TuneError::Validation(format!(
                "blend_lora_adapters: weight at index {idx} is not finite ({w})"
            )));
        }
    }

    // Collect the union of all (layer_idx, module) keys, grouped by key.
    // Value: list of (LoraLayer ref, effective_weight = w_e * s_e).
    let mut grouped: HashMap<(usize, String), Vec<(&LoraLayer, f32)>> = HashMap::new();
    for (adapter, weight) in adapters {
        let s_e = adapter.config.scale();
        let eff = weight * s_e;
        for ((layer_idx, module), layer) in &adapter.layers {
            grouped
                .entry((*layer_idx, module.clone()))
                .or_default()
                .push((layer, eff));
        }
    }

    // Bound the TOTAL planned allocation across every (layer_idx, module) group.
    // The per-group MAX_BLEND_RANK_TOTAL cap bounds each projection, but a
    // full-model adapter can keep every group near the cap and still drive a
    // multi-GiB aggregate blend. Sum rank_total*(d_in+d_out) over all groups with
    // checked arithmetic and fail closed before any allocation.
    let mut planned_elems: usize = 0;
    for ((layer_idx, module), entries) in &grouped {
        let (first, _) = entries[0]; // each key was inserted with >=1 layer
        let dims = first.d_in.checked_add(first.d_out).ok_or_else(|| {
            TuneError::Validation(format!(
                "blend_lora_adapters: layer {layer_idx} module '{module}' d_in+d_out overflowed usize"
            ))
        })?;
        let mut group_rank: usize = 0;
        for (layer, _) in entries {
            group_rank = group_rank.checked_add(layer.rank).ok_or_else(|| {
                TuneError::Validation("blend_lora_adapters: rank_total overflowed usize".into())
            })?;
        }
        let group_elems = group_rank.checked_mul(dims).ok_or_else(|| {
            TuneError::Validation(
                "blend_lora_adapters: rank_total*(d_in+d_out) overflowed usize".into(),
            )
        })?;
        planned_elems = planned_elems.checked_add(group_elems).ok_or_else(|| {
            TuneError::Validation(
                "blend_lora_adapters: aggregate blend element count overflowed usize".into(),
            )
        })?;
    }
    if planned_elems > MAX_BLEND_TOTAL_ELEMENTS {
        return Err(TuneError::Validation(format!(
            "blend_lora_adapters: aggregate blend size {planned_elems} elements exceeds \
             MAX_BLEND_TOTAL_ELEMENTS={MAX_BLEND_TOTAL_ELEMENTS} (~{} GiB f32); reduce the \
             number of adapters, their rank, or the number of target projections",
            (MAX_BLEND_TOTAL_ELEMENTS * 4) / (1024 * 1024 * 1024)
        )));
    }

    // Blend each (layer_idx, module) group independently.
    let mut blended_layers: HashMap<(usize, String), LoraLayer> = HashMap::new();
    for ((layer_idx, module), entries) in &grouped {
        let blended = blend_layer_entries(entries, &(layer_idx, module))?;
        blended_layers.insert((*layer_idx, module.clone()), blended);
    }

    // Collect total rank and target modules for the blended config.
    // We set alpha = rank so scale() == 1.0 — weights+scale already folded into B.
    let total_rank: usize = blended_layers.values().map(|l| l.rank).max().unwrap_or(0);
    let target_modules: Vec<String> = {
        let mut mods: Vec<String> = grouped
            .keys()
            .map(|(_, m)| m.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        mods.sort();
        mods
    };

    let config = LoraConfig {
        rank: total_rank,
        alpha: total_rank as f32,
        target_modules,
    };

    Ok(LoraAdapter::new(config, blended_layers))
}

/// Blend multiple `(LoraLayer, effective_weight)` entries for a single
/// `(layer_idx, module)` projection into one higher-rank layer.
///
/// `effective_weight = mixture_weight * (alpha / rank)` has already been
/// computed by the caller so this function is pure linear algebra.
fn blend_layer_entries(
    entries: &[(&LoraLayer, f32)],
    key: &(&usize, &String),
) -> Result<LoraLayer> {
    debug_assert!(!entries.is_empty());

    let (first_layer, _) = entries[0];
    let d_in = first_layer.d_in;
    let d_out = first_layer.d_out;

    // Validate that all entries share the same projection shape.
    for (idx, (layer, _)) in entries.iter().enumerate() {
        if layer.d_in != d_in || layer.d_out != d_out {
            return Err(TuneError::Validation(format!(
                "blend_lora_adapters: layer {} module '{}' has mismatched dimensions \
                 (entry 0: d_in={d_in}, d_out={d_out}; entry {idx}: d_in={}, d_out={})",
                key.0, key.1, layer.d_in, layer.d_out
            )));
        }
    }

    // Accumulate rank_total with overflow protection and a hard cap.
    let mut rank_total: usize = 0;
    for (layer, _) in entries {
        rank_total = rank_total.checked_add(layer.rank).ok_or_else(|| {
            TuneError::Validation("blend_layer_entries: rank_total overflowed usize".into())
        })?;
    }
    if rank_total > MAX_BLEND_RANK_TOTAL {
        return Err(TuneError::Validation(format!(
            "blend_layer_entries: summed rank {rank_total} exceeds \
             MAX_BLEND_RANK_TOTAL={MAX_BLEND_RANK_TOTAL}"
        )));
    }

    // Validate source slice lengths before any allocation: a malformed adapter
    // whose A or B buffer is the wrong size would cause out-of-bounds copies.
    for (idx, (layer, _)) in entries.iter().enumerate() {
        let expected_a = layer.rank.checked_mul(d_in).ok_or_else(|| {
            TuneError::Validation("blend_layer_entries: rank*d_in overflowed usize".into())
        })?;
        let expected_b = d_out.checked_mul(layer.rank).ok_or_else(|| {
            TuneError::Validation("blend_layer_entries: d_out*rank overflowed usize".into())
        })?;
        if layer.a.len() != expected_a {
            return Err(TuneError::Validation(format!(
                "blend_layer_entries: entry {idx} A slice length {} \
                 does not match rank*d_in={}*{}={expected_a}",
                layer.a.len(),
                layer.rank,
                d_in,
            )));
        }
        if layer.b.len() != expected_b {
            return Err(TuneError::Validation(format!(
                "blend_layer_entries: entry {idx} B slice length {} \
                 does not match d_out*rank={}*{}={expected_b}",
                layer.b.len(),
                d_out,
                layer.rank,
            )));
        }
    }

    let a_buf_len = rank_total.checked_mul(d_in).ok_or_else(|| {
        TuneError::Validation("blend_layer_entries: rank_total * d_in overflowed usize".into())
    })?;
    let b_buf_len = d_out.checked_mul(rank_total).ok_or_else(|| {
        TuneError::Validation("blend_layer_entries: d_out * rank_total overflowed usize".into())
    })?;

    // A_blend: vertical stack of A blocks.
    // Layout: rows [0..r_1] from adapter 1, then [r_1..r_1+r_2] from adapter 2, …
    // Each row has `d_in` elements.
    let mut a_blend = Vec::with_capacity(a_buf_len);
    for (layer, _) in entries {
        a_blend.extend_from_slice(&layer.a);
    }

    // B_blend: horizontal concatenation of scaled B column blocks.
    // B is stored row-major (d_out × rank), so row r is `b[r*rank..(r+1)*rank]`.
    // After blending, row r of B_blend is:
    //   [eff_1 * B_1[r, :] | eff_2 * B_2[r, :] | …]
    let mut b_blend = vec![0.0f32; b_buf_len];
    let mut col_offset = 0usize;
    for (layer, eff_weight) in entries {
        let r_e = layer.rank;
        for row in 0..d_out {
            let dst_start = row * rank_total + col_offset;
            let src_start = row * r_e;
            for c in 0..r_e {
                b_blend[dst_start + c] = eff_weight * layer.b[src_start + c];
            }
        }
        col_offset += r_e;
    }

    Ok(LoraLayer {
        a: a_blend,
        b: b_blend,
        d_in,
        d_out,
        rank: rank_total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Build a minimal single-layer adapter with a random-ish A and B.
    fn make_adapter(rank: usize, d_in: usize, d_out: usize, seed: f32) -> LoraAdapter {
        let a: Vec<f32> = (0..rank * d_in).map(|i| (i as f32 + seed) * 0.1).collect();
        let b: Vec<f32> = (0..d_out * rank)
            .map(|i| (i as f32 + seed) * 0.05)
            .collect();
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".to_string()),
            LoraLayer {
                a,
                b,
                d_in,
                d_out,
                rank,
            },
        );
        LoraAdapter::new(
            LoraConfig {
                rank,
                alpha: rank as f32, // scale = 1.0
                target_modules: vec!["q_proj".into()],
            },
            layers,
        )
    }

    // Apply a LoraAdapter to x and return the delta (excludes base output).
    fn lora_delta(adapter: &LoraAdapter, x: &[f32]) -> Vec<f32> {
        let (_, layer) = adapter.layers.iter().next().unwrap();
        let scale = adapter.config.scale();
        let rank = layer.rank;
        let d_in = layer.d_in;
        let d_out = layer.d_out;

        // intermediate = A @ x
        let mut inter = vec![0.0f32; rank];
        for r in 0..rank {
            inter[r] = (0..d_in).map(|c| layer.a[r * d_in + c] * x[c]).sum();
        }

        // delta = scale * B @ inter
        let mut delta = vec![0.0f32; d_out];
        for row in 0..d_out {
            delta[row] = scale
                * (0..rank)
                    .map(|c| layer.b[row * rank + c] * inter[c])
                    .sum::<f32>();
        }
        delta
    }

    // Apply a LoraLayer (with explicit scale) to x and return the delta.
    fn layer_delta(layer: &LoraLayer, scale: f32, x: &[f32]) -> Vec<f32> {
        let rank = layer.rank;
        let d_in = layer.d_in;
        let d_out = layer.d_out;

        let mut inter = vec![0.0f32; rank];
        for r in 0..rank {
            inter[r] = (0..d_in).map(|c| layer.a[r * d_in + c] * x[c]).sum();
        }

        let mut delta = vec![0.0f32; d_out];
        for row in 0..d_out {
            delta[row] = scale
                * (0..rank)
                    .map(|c| layer.b[row * rank + c] * inter[c])
                    .sum::<f32>();
        }
        delta
    }

    // -----------------------------------------------------------------------
    // Required assertion 1: single adapter at weight 1.0 is numerically
    // identical to that adapter (max-diff 0.0 or < 1e-6).
    // -----------------------------------------------------------------------
    #[test]
    fn blend_single_adapter_identity() {
        let rank = 2;
        let d_in = 4;
        let d_out = 4;
        let adapter = make_adapter(rank, d_in, d_out, 1.0);

        let blended = blend_lora_adapters(&[(&adapter, 1.0)]).unwrap();

        // Generate a test input vector.
        let x: Vec<f32> = (0..d_in).map(|i| (i + 1) as f32).collect();

        let delta_orig = lora_delta(&adapter, &x);
        // Blended adapter has scale=1.0 (alpha=rank_total).
        let blended_layer = blended.layers.get(&(0, "q_proj".to_string())).unwrap();
        let delta_blend = layer_delta(blended_layer, blended.config.scale(), &x);

        let max_diff = delta_orig
            .iter()
            .zip(&delta_blend)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "single-adapter blend should be identical to original; max-diff={max_diff}"
        );
    }

    // -----------------------------------------------------------------------
    // Required assertion 2: blend of two adapters equals explicit sum.
    // w1·Δ1(x) + w2·Δ2(x) matches the blended adapter's output (max-diff < 1e-5).
    // -----------------------------------------------------------------------
    #[test]
    fn blend_two_adapters_equals_sum() {
        let rank = 3;
        let d_in = 5;
        let d_out = 6;
        let w1 = 0.4f32;
        let w2 = 0.6f32;

        let adapter1 = make_adapter(rank, d_in, d_out, 0.0);
        let adapter2 = make_adapter(rank, d_in, d_out, 10.0);

        let blended = blend_lora_adapters(&[(&adapter1, w1), (&adapter2, w2)]).unwrap();

        let x: Vec<f32> = (0..d_in).map(|i| (i + 1) as f32 * 0.5).collect();

        let d1 = lora_delta(&adapter1, &x);
        let d2 = lora_delta(&adapter2, &x);
        // Expected: w1*Δ1 + w2*Δ2
        let expected: Vec<f32> = d1.iter().zip(&d2).map(|(a, b)| w1 * a + w2 * b).collect();

        let blended_layer = blended.layers.get(&(0, "q_proj".to_string())).unwrap();
        let actual = layer_delta(blended_layer, blended.config.scale(), &x);

        let max_diff = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "blend of two adapters must equal w1·Δ1+w2·Δ2; max-diff={max_diff}"
        );
    }

    // -----------------------------------------------------------------------
    // Required assertion 3: rank of blended adapter equals Σr_e.
    // -----------------------------------------------------------------------
    #[test]
    fn blend_rank_equals_sum_of_ranks() {
        let adapter1 = make_adapter(1, 4, 4, 0.0);
        let adapter2 = make_adapter(2, 4, 4, 5.0);

        let blended = blend_lora_adapters(&[(&adapter1, 0.5), (&adapter2, 0.5)]).unwrap();

        let blended_layer = blended.layers.get(&(0, "q_proj".to_string())).unwrap();
        assert_eq!(
            blended_layer.rank,
            1 + 2,
            "blended rank must equal sum of individual ranks"
        );
    }

    // -----------------------------------------------------------------------
    // Edge-case: empty adapters slice returns an error.
    // -----------------------------------------------------------------------
    #[test]
    fn blend_empty_returns_error() {
        let result: Result<LoraAdapter> = blend_lora_adapters(&[]);
        assert!(result.is_err(), "empty adapters should return an error");
    }

    // -----------------------------------------------------------------------
    // Edge-case: non-finite weight returns an error.
    // -----------------------------------------------------------------------
    #[test]
    fn blend_non_finite_weight_returns_error() {
        let adapter = make_adapter(1, 4, 4, 0.0);
        let result = blend_lora_adapters(&[(&adapter, f32::NAN)]);
        assert!(result.is_err(), "NaN weight should return an error");
    }

    // -----------------------------------------------------------------------
    // Summed rank exceeding MAX_BLEND_RANK_TOTAL returns an error.
    // -----------------------------------------------------------------------
    #[test]
    fn blend_rank_exceeds_cap_returns_error() {
        use super::MAX_BLEND_RANK_TOTAL;
        // rank = cap + 1; d_in=1, d_out=1 → A and B are tiny (~32 KB total).
        let rank = MAX_BLEND_RANK_TOTAL + 1;
        let d_in = 1;
        let d_out = 1;
        let a: Vec<f32> = vec![0.1f32; rank * d_in];
        let b: Vec<f32> = vec![0.1f32; d_out * rank];
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".to_string()),
            LoraLayer {
                a,
                b,
                d_in,
                d_out,
                rank,
            },
        );
        let adapter = LoraAdapter::new(
            LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec!["q_proj".into()],
            },
            layers,
        );
        let result = blend_lora_adapters(&[(&adapter, 1.0)]);
        assert!(
            result.is_err(),
            "summed rank exceeding cap must return an error"
        );
    }

    // -----------------------------------------------------------------------
    // Mismatched A slice length returns an error.
    // -----------------------------------------------------------------------
    #[test]
    fn blend_mismatched_a_length_returns_error() {
        let rank = 2;
        let d_in = 4;
        let d_out = 4;
        // A is one element short of the required rank * d_in.
        let a: Vec<f32> = vec![0.0f32; rank * d_in - 1];
        let b: Vec<f32> = vec![0.0f32; d_out * rank];
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".to_string()),
            LoraLayer {
                a,
                b,
                d_in,
                d_out,
                rank,
            },
        );
        let adapter = LoraAdapter::new(
            LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec!["q_proj".into()],
            },
            layers,
        );
        let result = blend_lora_adapters(&[(&adapter, 1.0)]);
        assert!(
            result.is_err(),
            "mismatched A slice length must return an error"
        );
    }

    // -----------------------------------------------------------------------
    // Mismatched B slice length returns an error.
    // -----------------------------------------------------------------------
    #[test]
    fn blend_mismatched_b_length_returns_error() {
        let rank = 2;
        let d_in = 4;
        let d_out = 4;
        let a: Vec<f32> = vec![0.0f32; rank * d_in];
        // B is one element short of the required d_out * rank.
        let b: Vec<f32> = vec![0.0f32; d_out * rank - 1];
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".to_string()),
            LoraLayer {
                a,
                b,
                d_in,
                d_out,
                rank,
            },
        );
        let adapter = LoraAdapter::new(
            LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec!["q_proj".into()],
            },
            layers,
        );
        let result = blend_lora_adapters(&[(&adapter, 1.0)]);
        assert!(
            result.is_err(),
            "mismatched B slice length must return an error"
        );
    }

    // -----------------------------------------------------------------------
    // alpha != rank causes scale() to be folded exactly once into B.
    //
    // With alpha=4, rank=2 → scale=2.0.  Blending a single adapter at
    // mixture weight w should produce: w * scale * B @ (A @ x).
    // -----------------------------------------------------------------------
    #[test]
    fn blend_folds_alpha_rank_scale_once() {
        let rank = 2usize;
        let alpha = 4.0f32; // scale = 4.0 / 2 = 2.0
        let d_in = 3usize;
        let d_out = 3usize;
        let w = 0.5f32;

        let a: Vec<f32> = (0..rank * d_in).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..d_out * rank).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".to_string()),
            LoraLayer {
                a: a.clone(),
                b: b.clone(),
                d_in,
                d_out,
                rank,
            },
        );
        let adapter = LoraAdapter::new(
            LoraConfig {
                rank,
                alpha,
                target_modules: vec!["q_proj".into()],
            },
            layers,
        );

        let blended = blend_lora_adapters(&[(&adapter, w)]).unwrap();
        let blended_layer = blended.layers.get(&(0, "q_proj".to_string())).unwrap();

        let x: Vec<f32> = (0..d_in).map(|i| (i + 1) as f32).collect();
        let scale = alpha / rank as f32; // = 2.0

        // Reference: w * scale * B @ (A @ x) — scale folded exactly once.
        let mut inter = vec![0.0f32; rank];
        for r in 0..rank {
            inter[r] = (0..d_in).map(|c| a[r * d_in + c] * x[c]).sum();
        }
        let expected: Vec<f32> = (0..d_out)
            .map(|row| w * scale * (0..rank).map(|c| b[row * rank + c] * inter[c]).sum::<f32>())
            .collect();

        // Blended adapter has alpha=rank_total so scale()=1.0; use layer_delta directly.
        let actual = layer_delta(blended_layer, blended.config.scale(), &x);

        let max_diff = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "alpha/rank scale must be folded exactly once; max-diff={max_diff}"
        );
    }

    // -----------------------------------------------------------------------
    // Edge-case: adapters with different ranks blend correctly.
    // -----------------------------------------------------------------------
    #[test]
    fn blend_asymmetric_ranks() {
        let r1 = 1;
        let r2 = 4;
        let d_in = 3;
        let d_out = 3;
        let w1 = 0.3f32;
        let w2 = 0.7f32;

        let adapter1 = make_adapter(r1, d_in, d_out, 1.0);
        let adapter2 = make_adapter(r2, d_in, d_out, 2.0);

        let blended = blend_lora_adapters(&[(&adapter1, w1), (&adapter2, w2)]).unwrap();
        let blended_layer = blended.layers.get(&(0, "q_proj".to_string())).unwrap();

        assert_eq!(blended_layer.rank, r1 + r2);

        let x: Vec<f32> = (0..d_in).map(|i| (i as f32) + 1.0).collect();
        let expected_d1 = lora_delta(&adapter1, &x);
        let expected_d2 = lora_delta(&adapter2, &x);
        let expected: Vec<f32> = expected_d1
            .iter()
            .zip(&expected_d2)
            .map(|(a, b)| w1 * a + w2 * b)
            .collect();

        let actual = layer_delta(blended_layer, blended.config.scale(), &x);

        let max_diff = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 1e-5, "asymmetric-rank blend max-diff={max_diff}");
    }

    // -----------------------------------------------------------------------
    // Contract: a malformed huge dimension returns Err, never panics.
    //
    // rank=2, d_in=usize::MAX/2+1, d_out=1 with empty a and b=[0.0;2].
    // Through the public entry the aggregate pre-pass catches it first
    // (rank_total*(d_in+d_out) overflows usize via checked_mul → Err); the
    // per-entry checked_mul is defense-in-depth for the same overflow class,
    // isolated directly by blend_layer_entries_per_entry_product_overflow.
    // -----------------------------------------------------------------------
    #[test]
    fn blend_malformed_huge_dim_returns_err_not_panic() {
        let rank = 2usize;
        let d_in = usize::MAX / 2 + 1;
        let d_out = 1usize;
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".to_string()),
            LoraLayer {
                a: vec![],          // empty; pre-pass fails before slice-length check
                b: vec![0.0f32; 2], // rank * d_out = 2 * 1 = 2
                d_in,
                d_out,
                rank,
            },
        );
        let adapter = LoraAdapter::new(
            LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec!["q_proj".into()],
            },
            layers,
        );
        let result = blend_lora_adapters(&[(&adapter, 1.0)]);
        // The pre-pass catches the overflow in rank_total*(d_in+d_out); the
        // per-entry checked_mul is defense-in-depth for the same class.
        assert!(
            result.is_err(),
            "malformed huge dim must return Err, not panic"
        );
    }

    // -----------------------------------------------------------------------
    // Isolates the per-entry rank*d_in checked_mul directly. Through the public
    // blend_lora_adapters the aggregate pre-pass catches this overflow class
    // first; calling the per-group helper with a single malformed entry
    // (rank_total below the per-projection cap, no aggregate stage) reaches the
    // per-entry guard as the first and only overflow check. Pinning the error
    // message makes it mutation-sensitive: reverting the checked_mul makes the
    // overflow either panic (debug) or surface via a different guard (release),
    // both of which fail this assertion.
    // -----------------------------------------------------------------------
    #[test]
    fn blend_layer_entries_per_entry_product_overflow_returns_err() {
        let rank = 2usize;
        let d_in = usize::MAX / 2 + 1; // rank*d_in = 2 * 2^63 overflows usize
        let d_out = 1usize;
        let layer = LoraLayer {
            a: vec![],                     // length checked only after the product
            b: vec![0.0f32; rank * d_out], // well-formed B (= 2)
            d_in,
            d_out,
            rank,
        };
        let layer_idx = 0usize;
        let module = "q_proj".to_string();
        let entries = [(&layer, 1.0f32)];
        let key = (&layer_idx, &module);
        let err = super::blend_layer_entries(&entries, &key)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("rank*d_in overflowed"),
            "expected the per-entry rank*d_in guard to fire; got: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // Contract: an aggregate element budget overrun returns Err before alloc.
    //
    // 65 layers each with rank=4096, d_in=2048, d_out=2048, empty a/b so the
    // test allocates nothing. Per-group: rank_total=4096 == cap (passes the
    // per-projection check). Aggregate: 4096*(2048+2048)*65 = 1,090,519,040
    // > MAX_BLEND_TOTAL_ELEMENTS (1<<30 = 1,073,741,824).
    // -----------------------------------------------------------------------
    #[test]
    fn blend_aggregate_budget_exceeded_returns_err() {
        use super::MAX_BLEND_TOTAL_ELEMENTS;
        let rank = MAX_BLEND_RANK_TOTAL; // 4096 — exactly at per-group cap, passes it
        let d_in = 2048usize;
        let d_out = 2048usize;
        let mut layers = HashMap::new();
        for idx in 0..65usize {
            layers.insert(
                (idx, "q_proj".to_string()),
                LoraLayer {
                    a: vec![], // empty; pre-pass reads only declared fields
                    b: vec![],
                    d_in,
                    d_out,
                    rank,
                },
            );
        }
        let adapter = LoraAdapter::new(
            LoraConfig {
                rank,
                alpha: rank as f32,
                target_modules: vec!["q_proj".into()],
            },
            layers,
        );
        let result = blend_lora_adapters(&[(&adapter, 1.0)]);
        assert!(result.is_err(), "aggregate budget exceeded must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("aggregate") || msg.contains("MAX_BLEND_TOTAL_ELEMENTS"),
            "error message must mention the aggregate budget; got: {msg}"
        );
    }
}
