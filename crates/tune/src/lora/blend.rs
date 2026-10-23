//! Exact CPU blending of adapters into one higher-rank adapter.
//!
//! The blend vertically concatenates A blocks and horizontally concatenates B
//! blocks after folding in each mixture weight and `alpha / rank` scale. Its
//! returned configuration sets `alpha == rank_total`, so its scale is exactly 1.
//!
//! See `docs/lora-core.md` for the derivation, limits, and failure behavior.

use crate::error::{Result, TuneError};
use crate::lora::{LoraAdapter, LoraConfig, LoraLayer};
use lattice_fann::lora as shared;
use std::collections::HashMap;

/// Maximum summed rank for one blended projection.
///
/// Re-exported from [`lattice_fann::lora::MAX_BLEND_RANK_TOTAL`], the shared
/// cap this crate and `lattice-inference`'s Metal LoRA blend both enforce.
/// Only referenced by tests below — non-test code calls `shared::*` directly
/// so the cap itself lives in exactly one place.
#[cfg(test)]
pub(crate) const MAX_BLEND_RANK_TOTAL: usize = shared::MAX_BLEND_RANK_TOTAL;

/// Blend a set of `(adapter, mixture_weight)` pairs into one rank-Σr adapter.
///
/// Folds each source's mixture weight and scale into B; absent projections are
/// omitted. Returns an error for invalid inputs, shapes, or resource limits.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#blend_lora_adapters) for the derivation and allocation limits.
pub fn blend_lora_adapters(adapters: &[(&LoraAdapter, f32)]) -> Result<LoraAdapter> {
    if adapters.is_empty() {
        return Err(TuneError::Validation(
            "blend_lora_adapters: adapters slice must not be empty".into(),
        ));
    }

    for (idx, (_, w)) in adapters.iter().enumerate() {
        shared::check_finite_weight("blend_lora_adapters", idx, *w)
            .map_err(TuneError::Validation)?;
    }

    // An adapter with no layers passes the outer `adapters.is_empty()` guard
    // above (the outer slice itself is non-empty) but contributes nothing to
    // `grouped` below, silently producing an empty-but-`Ok` blended adapter
    // instead of an admission error — the same failure shape
    // `blend_lora_layer_data` in `lattice-inference` rejects for its own
    // empty inner layer slices.
    for (idx, (adapter, _)) in adapters.iter().enumerate() {
        if adapter.layers().is_empty() {
            return Err(TuneError::Validation(format!(
                "blend_lora_adapters: adapters[{idx}] has no layers"
            )));
        }
    }

    // Group the union of projection keys with their folded source scales.
    let mut grouped: HashMap<(usize, String), Vec<(&LoraLayer, f32)>> = HashMap::new();
    for (adapter, weight) in adapters {
        let s_e = adapter.config().scale();
        let eff = weight * s_e;
        for ((layer_idx, module), layer) in adapter.layers() {
            grouped
                .entry((*layer_idx, module.clone()))
                .or_default()
                .push((layer, eff));
        }
    }

    // Reject aggregate allocation overflow or budget excess before allocating.
    let mut planned_elems: usize = 0;
    for ((layer_idx, module), entries) in &grouped {
        let (first, _) = entries[0]; // each key was inserted with >=1 layer
        let mut group_rank: usize = 0;
        for (layer, _) in entries {
            group_rank = shared::accumulate_rank(group_rank, layer.rank, "blend_lora_adapters")
                .map_err(TuneError::Validation)?;
        }
        let group_elems = shared::checked_group_elements(
            "blend_lora_adapters",
            *layer_idx,
            module,
            group_rank,
            first.d_in,
            first.d_out,
        )
        .map_err(TuneError::Validation)?;
        planned_elems =
            shared::accumulate_planned_elements(planned_elems, group_elems, "blend_lora_adapters")
                .map_err(TuneError::Validation)?;
    }
    shared::check_aggregate_elements_cap(planned_elems, "blend_lora_adapters")
        .map_err(TuneError::Validation)?;

    // Blend each (layer_idx, module) group independently.
    let mut blended_layers: HashMap<(usize, String), LoraLayer> = HashMap::new();
    for ((layer_idx, module), entries) in &grouped {
        let blended = blend_layer_entries(entries, &(layer_idx, module))?;
        blended_layers.insert((*layer_idx, module.clone()), blended);
    }

    // Set alpha = rank so the returned adapter's scale is one.
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
        dtype: "f32".into(),
    };

    LoraAdapter::new(config, blended_layers)
}

/// Blend one projection's layers after their effective weights were folded.
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
        shared::check_dims_match(
            "blend_lora_adapters",
            *key.0,
            key.1,
            d_in,
            d_out,
            idx,
            layer.d_in,
            layer.d_out,
        )
        .map_err(TuneError::Validation)?;
    }

    // Accumulate rank_total with overflow protection and a hard cap.
    let mut rank_total: usize = 0;
    for (layer, _) in entries {
        rank_total = shared::accumulate_rank(rank_total, layer.rank, "blend_layer_entries")
            .map_err(TuneError::Validation)?;
    }
    shared::check_rank_total_cap(rank_total, "blend_layer_entries")
        .map_err(TuneError::Validation)?;

    // Validate source buffers before copying their declared row-major shapes.
    for (idx, (layer, _)) in entries.iter().enumerate() {
        shared::check_buffer_lengths(
            "blend_layer_entries",
            idx,
            layer.rank,
            d_in,
            d_out,
            layer.a.len(),
            layer.b.len(),
        )
        .map_err(TuneError::Validation)?;
    }

    let a_buf_len = rank_total.checked_mul(d_in).ok_or_else(|| {
        TuneError::Validation("blend_layer_entries: rank_total * d_in overflowed usize".into())
    })?;
    let b_buf_len = d_out.checked_mul(rank_total).ok_or_else(|| {
        TuneError::Validation("blend_layer_entries: d_out * rank_total overflowed usize".into())
    })?;

    // Stack A blocks vertically in source order.
    let mut a_blend = Vec::with_capacity(a_buf_len);
    for (layer, _) in entries {
        a_blend.extend_from_slice(&layer.a);
    }

    // Concatenate scaled B blocks horizontally in each row.
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
mod tests;
