use super::{GenerationPreparation, MetalQwen35State};
use metal::{Buffer, MTLResourceOptions};

// ---------------------------------------------------------------------------
// LoRA adapter state (ADR-045 step 3)
// ---------------------------------------------------------------------------

/// Metal buffers for a single LoRA-adapted projection.
pub(super) struct MetalLoraProjection {
    pub(super) a_buf: Buffer,
    pub(super) b_buf: Buffer,
    pub(super) rank: u32,
    pub(super) d_in: u32,
    pub(super) d_out: u32,
}

/// Loaded LoRA adapter state on Metal GPU.
pub(crate) struct MetalLoraAdapter {
    /// Nested lookup: layer_idx → module_name → projection. No allocation on lookup.
    pub(super) projections: Vec<std::collections::HashMap<String, MetalLoraProjection>>,
    pub(super) scale: f32,
    /// Shared rank scratch: `max_rows * max_rank` floats. A decode step uses the first
    /// row; a batched prefill chunk uses one `rank`-wide row per token.
    pub(super) intermediate: Buffer,
    // max rank used to size `intermediate` buffer at load time; stored for adapter introspection
    pub(super) max_rank: u32,
    /// Token rows `intermediate` can hold: the state's maximum prefill chunk.
    pub(super) max_rows: usize,
}

impl MetalLoraAdapter {
    pub(super) fn get_projection(
        &self,
        layer_idx: usize,
        module: &str,
    ) -> Option<&MetalLoraProjection> {
        self.projections.get(layer_idx)?.get(module)
    }
}

/// Input data for a single LoRA layer to be loaded onto GPU.
pub struct LoraLayerData {
    /// Transformer layer index (0-based).
    pub layer_idx: usize,
    /// Projection module name (e.g., `"q_proj"`, `"o_proj"`).
    pub module: String,
    /// A matrix, row-major `(rank × d_in)`.
    pub a: Vec<f32>,
    /// B matrix, row-major `(d_out × rank)`.
    pub b: Vec<f32>,
    /// LoRA rank.
    pub rank: usize,
    /// Input dimension.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
}

/// Maximum total rank allowed across all adapters in a single blend.
///
/// The Metal GEMV kernels assume a modest rank budget (≤ ~64 per adapter in a
/// typical mixture).  This cap bounds allocations and rejects adversarially large
/// adapter pools before `Vec::with_capacity`.
///
/// Re-exported from [`lattice_fann::lora::MAX_BLEND_RANK_TOTAL`], the shared
/// cap this crate and `lattice-tune`'s CPU LoRA blend both enforce. Only
/// referenced by tests below — non-test code calls `lattice_fann::lora::*`
/// directly so the cap itself lives in exactly one place.
#[cfg(test)]
pub(crate) const MAX_BLEND_RANK_TOTAL: usize = lattice_fann::lora::MAX_BLEND_RANK_TOTAL;

/// Reject any element of `inputs` whose inner layer slice is empty.
///
/// An empty inner slice passes an outer `inputs.is_empty()` check (the
/// outer slice itself is non-empty) but contributes nothing to a
/// group-by-`(layer_idx, module)` pass, silently producing an empty
/// result instead of an admission error. Split out from
/// [`blend_lora_layer_data`] so callers that mutate state before
/// blending (e.g. [`MetalQwen35State::generate_with_lora_mixture`], which
/// unloads the currently loaded adapter first) can run this check first
/// and reject the request before that mutation, not after.
fn reject_empty_inner_layers(
    inputs: &[(&[LoraLayerData], f32)],
) -> Result<(), lattice_inference_core::InferenceError> {
    for (idx, (layers, _)) in inputs.iter().enumerate() {
        if layers.is_empty() {
            return Err(lattice_inference_core::InferenceError::InvalidInput(
                format!("blend_lora_layer_data: inputs[{idx}] has an empty layer slice"),
            ));
        }
    }
    Ok(())
}

/// Blend multiple sets of [`LoraLayerData`] into one rank-Σr set for use with
/// the single-slot [`MetalQwen35State::load_lora_adapter`] API.
///
/// # Blend math
///
/// For adapter `e` with layer data `L_e` and mixture weight `w_e`, the
/// effective delta is `w_e · B_e @ (A_e @ x)` (the `alpha/rank` scale is
/// NOT present in `LoraLayerData` — it is passed separately as `scale` to
/// `load_lora_adapter`; here `w_e` IS that effective scale).
///
/// The blended matrices are:
/// - `A_blend = vconcat[A_1; …; A_N]`       shape `(Σrank_e) × d_in`
/// - `B_blend = hconcat[w_1·B_1 | …]`       shape `d_out × (Σrank_e)`
///
/// The resulting layer set should be loaded with `scale = 1.0` because
/// the per-adapter weights are already folded into the B blocks.
///
/// # Errors
///
/// Returns an error if:
/// - `inputs` is empty
/// - Any element of `inputs` has an empty inner layer slice
/// - Any weight is not finite
/// - Two adapters have conflicting `d_in` / `d_out` for the same `(layer_idx, module)`
/// - The summed rank for a single projection exceeds `MAX_BLEND_RANK_TOTAL`
/// - The aggregate blend size across all projections exceeds `MAX_BLEND_TOTAL_ELEMENTS`
/// - A layer's A or B buffer length does not match its declared `rank`, `d_in`, or `d_out`
/// - Rank accumulation or a size product overflows `usize`
pub fn blend_lora_layer_data(
    inputs: &[(&[LoraLayerData], f32)],
) -> Result<Vec<LoraLayerData>, lattice_inference_core::InferenceError> {
    use lattice_inference_core::InferenceError;
    use std::collections::HashMap;

    if inputs.is_empty() {
        return Err(InferenceError::InvalidInput(
            "blend_lora_layer_data: inputs must not be empty".into(),
        ));
    }

    // An empty inner layer slice passes the outer `inputs.is_empty()`
    // guard above (the outer slice itself is non-empty) but contributes
    // nothing to `grouped` below, silently producing `Ok(Vec::new())`
    // instead of an admission error. Callers that unload the currently
    // loaded adapter before blending (e.g. `generate_with_lora_mixture`)
    // must reject this before that mutation, not after — see
    // `reject_empty_inner_layers` and its call site there.
    reject_empty_inner_layers(inputs)?;

    for (idx, (_, w)) in inputs.iter().enumerate() {
        lattice_fann::lora::check_finite_weight("blend_lora_layer_data", idx, *w)
            .map_err(InferenceError::InvalidInput)?;
    }

    // Group by (layer_idx, module): collect refs to layer data and effective weights.
    // Keyed by `&str` borrowed from each layer's own `module: String` (which
    // outlives this function body via `inputs`), not a per-layer clone.
    let mut grouped: HashMap<(usize, &str), Vec<(&LoraLayerData, f32)>> = HashMap::new();
    for (layers, weight) in inputs {
        for layer in *layers {
            grouped
                .entry((layer.layer_idx, layer.module.as_str()))
                .or_default()
                .push((layer, *weight));
        }
    }

    // Pre-allocation planning -- the per-group MAX_BLEND_RANK_TOTAL cap,
    // the aggregate MAX_BLEND_TOTAL_ELEMENTS cap, and cross-adapter
    // dimension agreement -- is the ONE shared check the adapter
    // residency registry's state publication also runs (issue #1735), so
    // a `GET /v1/lora` report of whether the resident set can be blended
    // and this function's own refusal can never disagree. It sees only
    // each entry's shape, never an A/B buffer.
    //
    // `grouped` above is already the grouping `plan_blend` would build
    // internally from a flat iterator, so this calls `plan_grouped`
    // directly over `grouped`'s own entries instead of `plan_blend`,
    // skipping a second `HashMap` and per-group `Vec` that would
    // otherwise re-derive the same grouping from scratch on every blend.
    let groups = grouped.iter().map(|(&(layer_idx, module), entries)| {
        (
            layer_idx,
            module,
            entries
                .iter()
                .map(|(layer, _)| lattice_fann::lora::BlendProjection {
                    layer_idx: layer.layer_idx,
                    module: layer.module.as_str(),
                    rank: layer.rank,
                    d_in: layer.d_in,
                    d_out: layer.d_out,
                }),
        )
    });
    let planned = lattice_fann::lora::plan_grouped("blend_lora_layer_data", groups)
        .map_err(InferenceError::InvalidInput)?;

    let mut result: Vec<LoraLayerData> = Vec::with_capacity(planned.len());
    for plan in planned {
        let entries = grouped.get(&(plan.layer_idx, plan.module)).ok_or_else(|| {
            InferenceError::Inference(format!(
                "blend_lora_layer_data: planned group (layer {}, module {}) is absent \
                         from the grouped adapter set",
                plan.layer_idx, plan.module
            ))
        })?;
        let d_in = plan.d_in;
        let d_out = plan.d_out;
        let rank_total = plan.rank_total;

        // Validate source slice lengths before any allocation: a malformed adapter
        // whose A or B buffer is the wrong size would cause out-of-bounds copies.
        // `plan_grouped` never sees these buffers, so this check stays here.
        for (idx, (entry, _)) in entries.iter().enumerate() {
            lattice_fann::lora::check_buffer_lengths(
                "blend_lora_layer_data",
                idx,
                entry.rank,
                d_in,
                d_out,
                entry.a.len(),
                entry.b.len(),
            )
            .map_err(InferenceError::InvalidInput)?;
        }

        let a_buf_len = rank_total.checked_mul(d_in).ok_or_else(|| {
            InferenceError::InvalidInput(
                "blend_lora_layer_data: rank_total * d_in overflowed usize".into(),
            )
        })?;
        let b_buf_len = d_out.checked_mul(rank_total).ok_or_else(|| {
            InferenceError::InvalidInput(
                "blend_lora_layer_data: d_out * rank_total overflowed usize".into(),
            )
        })?;

        // A_blend: vertical stack of A matrices.
        let mut a_blend: Vec<f32> = Vec::with_capacity(a_buf_len);
        for (layer, _) in entries.iter() {
            a_blend.extend_from_slice(&layer.a);
        }

        // B_blend: horizontal concat of weight-scaled B column blocks.
        // B is row-major (d_out × rank); row r: b[r*rank..(r+1)*rank].
        let mut b_blend = vec![0.0f32; b_buf_len];
        let mut col_offset = 0usize;
        for (layer, eff_weight) in entries.iter() {
            let r_e = layer.rank;
            for row in 0..d_out {
                let dst = row * rank_total + col_offset;
                let src = row * r_e;
                for c in 0..r_e {
                    b_blend[dst + c] = eff_weight * layer.b[src + c];
                }
            }
            col_offset += r_e;
        }

        result.push(LoraLayerData {
            layer_idx: plan.layer_idx,
            module: plan.module.to_string(),
            a: a_blend,
            b: b_blend,
            rank: rank_total,
            d_in,
            d_out,
        });
    }

    Ok(result)
}

impl MetalQwen35State {
    /// Returns the expected `(d_in, d_out)` for a LoRA projection given the layer type.
    ///
    /// Returns an error if `module` is not valid for the layer type at `layer_idx`,
    /// or if the module name is not recognised.
    pub(super) fn expected_lora_shape(
        cfg: &lattice_arch::qwen35_config::Qwen35Config,
        layer_idx: usize,
        module: &str,
    ) -> Result<(usize, usize), lattice_inference_core::InferenceError> {
        use lattice_inference_core::InferenceError;

        // The refusal set is DERIVED from the servable set rather than spelled
        // again here. Written out, this list was a second copy of a fact the
        // trainers also hold, and two independently maintained copies of
        // "which GDN modules can be served" drift silently: the adapter
        // trains, the loss is minimised for a delta this kernel will never
        // apply, and nothing says so until load. `crates/inference/src/
        // lora_hook.rs` is the one copy; both ends read it.
        if lattice_compute::lora_hook::GDN_LORA_MODULES.contains(&module)
            && !lattice_compute::lora_hook::gdn_lora_module_is_servable(module)
        {
            if cfg.is_full_attention(layer_idx) {
                return Err(InferenceError::Inference(format!(
                    "unknown LoRA module '{module}'"
                )));
            }
            return Err(InferenceError::Inference(format!(
                "module '{module}' is not yet supported in Metal forward (consumed inside \
                     fused GDN recurrence kernel); target other GDN modules instead"
            )));
        }

        let shape = crate::lora_hook::qwen35_projection_shape(cfg, layer_idx, module)
            .map_err(InferenceError::Inference)?;
        Ok((shape.d_in, shape.d_out))
    }

    /// Load a LoRA adapter onto the Metal GPU for inference.
    ///
    /// Resolves the QuaRot rotation seed to use before uploading anything: when
    /// `quarot_seed` is `None`, the base model's own rotation seed is used (so an
    /// adapter loaded onto a QuaRot-rotated base is rotated by default, rather than
    /// silently skipped); when `quarot_seed` is `Some`, it must agree with the
    /// base's seed if the base has one, and the base must actually be rotated if
    /// it does not. See `# Errors` below. Once resolved, rotation corrections are
    /// applied for a `Some` seed, then all A/B matrices are uploaded to Metal
    /// shared-memory buffers.
    ///
    /// Each `LoraLayerData` must name a valid Qwen3.5 projection module for its
    /// layer type, and its `(d_in, d_out)` must match the model's actual projection
    /// shape. Full-attention modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
    /// GDN modules: `in_proj_qkv`, `in_proj_z`, `out_proj`.
    /// (`in_proj_a`, `in_proj_b` not yet supported — consumed inside fused kernels).
    /// MLP modules (both layer types): `gate_proj`, `up_proj`, `down_proj`.
    ///
    /// This is the deliberate low-level entry point: `scale` is applied
    /// exactly as given, with no cross-check against any adapter
    /// identity (no descriptor, no per-layer rank comparison, nothing
    /// derived from `layers` itself). The caller owns the invariant that
    /// `scale` is the correct `alpha / rank` for the adapter these
    /// `layers` actually represent — this function has no way to verify
    /// that on its own. Prefer [`Self::load_lora_adapter_with_descriptor`]
    /// when `scale` should instead be derived from, and checked against,
    /// a [`lattice_fann::lora::LoraDescriptor`]'s declared rank and
    /// target modules.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `layers` is empty
    /// - `scale` is not finite (NaN/inf)
    /// - An adapter is already loaded (call `unload_lora_adapter` first)
    /// - A module name is unknown or invalid for the layer type
    /// - `(d_in, d_out)` does not match the model projection shape
    /// - A/B vector lengths are inconsistent with rank × dimensions
    /// - `layer_idx` is out of range
    /// - Duplicate `(layer_idx, module)` entries exist
    /// - Any module is not in the rotation plan (when a seed resolves, from
    ///   `quarot_seed` or the base model)
    /// - Rank or dimensions overflow `u32`
    /// - `quarot_seed` is `Some` and disagrees with the base model's own QuaRot
    ///   rotation seed
    /// - `quarot_seed` is `Some` but the base model is not QuaRot-rotated
    pub fn load_lora_adapter(
        &mut self,
        mut layers: Vec<LoraLayerData>,
        scale: f32,
        quarot_seed: Option<u64>,
    ) -> Result<(), lattice_inference_core::InferenceError> {
        use lattice_inference_core::InferenceError;
        use lattice_quant::quarot::lora::{LoraLayerMut, rotate_adapter_for_quarot};

        if self.lora.is_some() {
            return Err(InferenceError::Inference(
                "LoRA adapter already loaded; call unload_lora_adapter first".into(),
            ));
        }
        if layers.is_empty() {
            return Err(InferenceError::Inference(
                "load_lora_adapter: layers must not be empty".into(),
            ));
        }
        if !scale.is_finite() {
            return Err(InferenceError::Inference(format!(
                "load_lora_adapter: scale must be finite, got {scale}"
            )));
        }

        let hidden_dim = self.engine.config.hidden_size;

        // Resolve the seed to rotate with, rather than trusting `quarot_seed`
        // alone: an adapter that names no seed of its own is rotated with the
        // base's own seed when the base is QuaRot-rotated, instead of silently
        // loading unrotated onto a rotated base (see this function's doc comment).
        let resolved_seed = match (quarot_seed, self.engine.base_quarot_seed) {
            (None, base_seed) => base_seed,
            (Some(adapter_seed), None) => {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: adapter specifies QuaRot seed {adapter_seed}, but \
                         the base model has no QuaRot rotation seed (base is not rotated)"
                )));
            }
            (Some(adapter_seed), Some(base_seed)) if adapter_seed != base_seed => {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: adapter QuaRot seed {adapter_seed} does not match \
                         base model QuaRot rotation seed {base_seed}"
                )));
            }
            (Some(adapter_seed), Some(_)) => Some(adapter_seed),
        };

        if let Some(seed) = resolved_seed {
            let plan = lattice_quant::quarot::plan::qwen35_residual_stream_linear_layers();
            let lora_muts: Vec<LoraLayerMut<'_>> = layers
                .iter_mut()
                .map(|l| LoraLayerMut {
                    layer_idx: l.layer_idx,
                    module: &l.module,
                    a: &mut l.a,
                    b: &mut l.b,
                    rank: l.rank,
                    d_in: l.d_in,
                    d_out: l.d_out,
                })
                .collect();
            rotate_adapter_for_quarot(lora_muts, seed, hidden_dim, &plan)?;
        }

        let device = &self.engine.device;
        let num_layers = self.engine.config.num_hidden_layers;
        let mut max_rank = 0u32;
        let mut projections: Vec<std::collections::HashMap<String, MetalLoraProjection>> = (0
            ..num_layers)
            .map(|_| std::collections::HashMap::new())
            .collect();

        for layer in &layers {
            if layer.rank == 0 {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: layer {} module '{}' has rank=0",
                    layer.layer_idx, layer.module
                )));
            }
            if layer.d_in == 0 || layer.d_out == 0 {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: layer {} module '{}' has zero dimension \
                         (d_in={}, d_out={})",
                    layer.layer_idx, layer.module, layer.d_in, layer.d_out
                )));
            }
            if layer.layer_idx >= num_layers {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: layer_idx {} >= num_hidden_layers {}",
                    layer.layer_idx, num_layers
                )));
            }
            let (exp_d_in, exp_d_out) =
                Self::expected_lora_shape(&self.engine.config, layer.layer_idx, &layer.module)?;
            if layer.d_in != exp_d_in || layer.d_out != exp_d_out {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: layer {} module '{}' has (d_in={}, d_out={}) \
                         but model expects (d_in={}, d_out={})",
                    layer.layer_idx, layer.module, layer.d_in, layer.d_out, exp_d_in, exp_d_out
                )));
            }
            let expected_a = layer.rank.checked_mul(layer.d_in).ok_or_else(|| {
                InferenceError::Inference(format!(
                    "load_lora_adapter: layer {} module '{}' rank*d_in overflow",
                    layer.layer_idx, layer.module
                ))
            })?;
            let expected_b = layer.d_out.checked_mul(layer.rank).ok_or_else(|| {
                InferenceError::Inference(format!(
                    "load_lora_adapter: layer {} module '{}' d_out*rank overflow",
                    layer.layer_idx, layer.module
                ))
            })?;
            if layer.a.len() != expected_a {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: layer {} module '{}' A length {} != rank*d_in {}",
                    layer.layer_idx,
                    layer.module,
                    layer.a.len(),
                    expected_a
                )));
            }
            if layer.b.len() != expected_b {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: layer {} module '{}' B length {} != d_out*rank {}",
                    layer.layer_idx,
                    layer.module,
                    layer.b.len(),
                    expected_b
                )));
            }
            let rank = u32::try_from(layer.rank).map_err(|_| {
                InferenceError::Inference(format!(
                    "load_lora_adapter: rank {} exceeds u32",
                    layer.rank
                ))
            })?;
            let d_in = u32::try_from(layer.d_in).map_err(|_| {
                InferenceError::Inference(format!(
                    "load_lora_adapter: d_in {} exceeds u32",
                    layer.d_in
                ))
            })?;
            let d_out = u32::try_from(layer.d_out).map_err(|_| {
                InferenceError::Inference(format!(
                    "load_lora_adapter: d_out {} exceeds u32",
                    layer.d_out
                ))
            })?;
            if rank > max_rank {
                max_rank = rank;
            }
            let a_buf = device.new_buffer_with_data(
                layer.a.as_ptr() as *const _,
                (layer.a.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let b_buf = device.new_buffer_with_data(
                layer.b.as_ptr() as *const _,
                (layer.b.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            if projections[layer.layer_idx].contains_key(&layer.module) {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: duplicate projection for layer {} module '{}'",
                    layer.layer_idx, layer.module
                )));
            }
            projections[layer.layer_idx].insert(
                layer.module.clone(),
                MetalLoraProjection {
                    a_buf,
                    b_buf,
                    rank,
                    d_in,
                    d_out,
                },
            );
        }

        let max_rows = self.session.max_prefill.max(1);
        let intermediate_bytes = (max_rows as u64)
            .checked_mul(max_rank as u64)
            .and_then(|floats| floats.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| {
                InferenceError::Inference(format!(
                    "load_lora_adapter: rank scratch size overflow \
                         (max_prefill={max_rows}, max_rank={max_rank})"
                ))
            })?;
        let intermediate =
            device.new_buffer(intermediate_bytes, MTLResourceOptions::StorageModeShared);

        self.lora = Some(MetalLoraAdapter {
            projections,
            scale,
            intermediate,
            max_rank,
            max_rows,
        });
        // The cross-turn cache's adapter
        // identity (`cross_turn_metadata`'s `adapter_id`) is a shape-based
        // hash (max_rank/scale/projection count), not a content hash, so
        // two different adapters with the same shape can collide on the
        // same `AdapterId`. The metadata hash is
        // defense-in-depth; the unconditional clear here is the actual
        // correctness mechanism.
        self.cross_turn_prefix_cache.clear();

        Ok(())
    }

    /// Load a LoRA adapter using the shared cross-crate
    /// [`lattice_fann::lora::LoraDescriptor`] as the source of the adapter's
    /// scale and declared target modules, instead of a bare `scale: f32`
    /// the caller must have derived correctly on its own.
    ///
    /// Validates `descriptor.alpha`/effective scale (finite),
    /// `descriptor.target_modules` (recognized names), that every
    /// nonempty `layers` entry's rank matches `descriptor.rank`, and that
    /// the set of modules present in `layers` matches
    /// `descriptor.target_modules` exactly — before delegating to
    /// [`Self::load_lora_adapter`] with `descriptor.scale()`. Every
    /// per-layer, per-architecture shape check `load_lora_adapter` already
    /// performs still applies on top of this.
    ///
    /// `descriptor.rank` and `descriptor.target_modules` are the inputs
    /// `descriptor.scale()` is computed from; without this check a caller
    /// could pass `layers` built at one rank alongside a `descriptor` built
    /// at another (or covering a different module set) and silently get
    /// the wrong scale applied to correctly-shaped buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor's alpha/scale is not finite, if
    /// `target_modules` contains an unrecognized name, if any layer's
    /// rank disagrees with `descriptor.rank`, if the loaded module set
    /// disagrees with `descriptor.target_modules`, or for any reason
    /// [`Self::load_lora_adapter`] itself would reject the call.
    pub fn load_lora_adapter_with_descriptor(
        &mut self,
        layers: Vec<LoraLayerData>,
        descriptor: &lattice_fann::lora::LoraDescriptor,
        quarot_seed: Option<u64>,
    ) -> Result<(), lattice_inference_core::InferenceError> {
        use lattice_inference_core::InferenceError;

        descriptor
            .validate()
            .map_err(InferenceError::InvalidInput)?;
        lattice_fann::lora::validate_target_modules(
            &descriptor.target_modules,
            lattice_fann::lora::KNOWN_LORA_TARGET_MODULES,
        )
        .map_err(InferenceError::InvalidInput)?;

        for layer in &layers {
            if layer.rank != descriptor.rank {
                return Err(InferenceError::InvalidInput(format!(
                    "load_lora_adapter_with_descriptor: layer {} module '{}' has \
                         rank={} but descriptor declares rank={}",
                    layer.layer_idx, layer.module, layer.rank, descriptor.rank
                )));
            }
        }

        let mut layer_modules: Vec<&str> = layers.iter().map(|l| l.module.as_str()).collect();
        layer_modules.sort_unstable();
        layer_modules.dedup();
        let mut declared_modules: Vec<&str> = descriptor
            .target_modules
            .iter()
            .map(String::as_str)
            .collect();
        declared_modules.sort_unstable();
        declared_modules.dedup();
        if layer_modules != declared_modules {
            return Err(InferenceError::InvalidInput(format!(
                "load_lora_adapter_with_descriptor: loaded layer modules {layer_modules:?} \
                     do not match descriptor.target_modules {declared_modules:?}"
            )));
        }

        self.load_lora_adapter(layers, descriptor.scale(), quarot_seed)
    }

    /// Unload the currently loaded LoRA adapter, freeing GPU buffers.
    pub fn unload_lora_adapter(&mut self) {
        self.lora = None;
        // See the matching comment in
        // `load_lora_adapter` — any adapter identity change must
        // invalidate the retained cross-turn entry.
        self.cross_turn_prefix_cache.clear();
    }

    /// Returns true if a LoRA adapter is currently loaded.
    pub fn has_lora_adapter(&self) -> bool {
        self.lora.is_some()
    }

    /// Blend multiple LoRA adapter layer sets into one rank-Σr layer set, then
    /// generate text through the existing single-adapter path.
    ///
    /// # Why blend instead of dispatching N adapters
    ///
    /// The Metal forward path holds exactly one `Option<MetalLoraAdapter>` slot.
    /// Rather than changing that kernel contract, we pre-blend on the CPU once
    /// at request start: the result is a valid single adapter loaded through the
    /// unchanged `load_lora_adapter` path.
    ///
    /// # Arguments
    ///
    /// * `adapter_weights` — slices of `(Vec<LoraLayerData>, mixture_weight)`.
    ///   Each element is one adapter's layer data paired with its effective weight
    ///   `w_e`.  `mixture_weight` IS the fully pre-scaled effective weight: the
    ///   caller must fold any `alpha / rank` scaling (and the mixture fraction)
    ///   into it before calling this function.  `blend_lora_layer_data` folds
    ///   `w_e` directly into the B column blocks and loads the result with
    ///   `scale = 1.0`; passing a raw mixture fraction without the `alpha/rank`
    ///   factor will produce under-scaled output.
    /// * `prompt` — prompt text, tokenised internally.
    /// * `tokenizer` — BPE tokeniser.
    /// * `gen_cfg` — generation configuration.
    ///
    /// The adapter is unloaded after generation so the slot is free for the next
    /// request.
    ///
    /// # Decode cost model
    ///
    /// Per-token LoRA GEMV cost grows linearly with `rank_total = Σ rank_e`
    /// across the mixture.  The Metal kernel assumes modest rank (≤ ~64);
    /// a large `k × r` budget will linearly increase decode latency.
    ///
    /// # Errors
    ///
    /// Returns an error if blending fails (e.g. dimension mismatch across
    /// adapters for the same projection), if adapter loading fails, or if the
    /// request fails the direct generator's preflight. Admission errors are
    /// returned before the existing adapter or prefix cache is changed.
    pub fn generate_with_lora_mixture(
        &mut self,
        adapter_weights: &[(&[LoraLayerData], f32)],
        prompt: &str,
        tokenizer: &lattice_tokenizer::BpeTokenizer,
        gen_cfg: &crate::generation::GenerateConfig,
    ) -> Result<crate::generation::GenerateOutput, lattice_inference_core::InferenceError> {
        if adapter_weights.is_empty() {
            return Err(lattice_inference_core::InferenceError::InvalidInput(
                "generate_with_lora_mixture: adapter_weights must not be empty".into(),
            ));
        }

        match self.prepare_direct_generation(prompt, tokenizer, gen_cfg)? {
            GenerationPreparation::Complete(output) => return Ok(output),
            GenerationPreparation::Ready(_) => {}
        }

        // Reject an empty inner layer slice here, before the adapter unload
        // below — `blend_lora_layer_data` also runs this check, but only
        // after the unload has already destroyed the previously loaded
        // adapter. Admission errors must be returned before any existing
        // adapter or prefix cache is changed (see this method's doc comment).
        reject_empty_inner_layers(adapter_weights)?;

        // Unload any previously loaded adapter so the slot is free.
        self.unload_lora_adapter();

        // Blend on the CPU. The per-adapter scale (alpha/rank) and mixture weight
        // are folded into the B column blocks; the resulting blended adapter uses
        // an effective scale of 1.0.
        let blended = blend_lora_layer_data(adapter_weights)?;

        // The blended adapter has weights-already-scaled; pass scale=1.0.
        self.load_lora_adapter(blended, 1.0, None)?;

        let output = self.generate(prompt, tokenizer, gen_cfg);

        // Always unload after the request so the slot does not leak into the
        // next call on this state object — regardless of whether generate()
        // succeeded or failed closed (e.g. an exhausted grammar mask, #611).
        self.unload_lora_adapter();

        output
    }
}
