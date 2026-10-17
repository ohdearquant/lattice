//! Worker-local adapter weights and the currently materialized mixture.

use super::ApiError;
use super::lora::{
    AdapterControlError, AdapterIndex, AdapterMetadata, LoraSelection, ResidencyLimits,
    unknown_adapter, validate_scales, validate_unique_ids,
};
use crate::forward::metal_qwen35::{LoraLayerData, MetalQwen35State, blend_lora_layer_data};
use lattice_fann::lora::LoraDescriptor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub(crate) trait AdapterSlot {
    fn load(&mut self, layers: Vec<LoraLayerData>) -> Result<(), String>;
    fn unload(&mut self);
}

impl AdapterSlot for MetalQwen35State {
    fn load(&mut self, layers: Vec<LoraLayerData>) -> Result<(), String> {
        self.load_lora_adapter(layers, 1.0, None)
            .map_err(|err| err.to_string())
    }

    fn unload(&mut self) {
        self.unload_lora_adapter();
    }
}

struct ResidentAdapter {
    metadata: AdapterMetadata,
    layers: Vec<LoraLayerData>,
    descriptor: LoraDescriptor,
    payload_bytes: usize,
}

pub(crate) struct ResidencyRegistry {
    residents: HashMap<u32, ResidentAdapter>,
    identities: HashMap<(String, String), u32>,
    resident_bytes: usize,
    limits: ResidencyLimits,
    next_id: Option<u32>,
    // Order determines concatenated rank order and hence floating-point reduction.
    applied: Vec<LoraSelection>,
    index: Arc<RwLock<AdapterIndex>>,
    // Cached feasibility of a blend over the FULL resident set (issue #1735).
    // Valid exactly as long as `residents` is unchanged: recomputed in `load`
    // and `unload`, the only two places that add or remove a resident, and
    // copied verbatim by `publish` -- never re-derived there. `apply` moves
    // `applied`, not `residents`, so it never invalidates this cache even
    // though it calls `publish` once or twice per call.
    blend_refusal: Option<String>,
    #[cfg(test)]
    blends: usize,
    // Counts calls to `recompute_blend_refusal`, i.e. how many times the
    // full-resident-set plan actually re-runs -- distinct from `blends`,
    // which counts real GPU blend uploads inside `apply`.
    #[cfg(test)]
    blend_plans: usize,
}

impl ResidencyRegistry {
    pub(crate) fn new(index: Arc<RwLock<AdapterIndex>>, limits: ResidencyLimits) -> Self {
        Self {
            residents: HashMap::new(),
            identities: HashMap::new(),
            resident_bytes: 0,
            limits,
            next_id: Some(0),
            applied: Vec::new(),
            index,
            // An empty resident set is trivially feasible (`plan_blend` of no
            // projections returns `Ok`, never a refusal), so the starting
            // cache is exactly what `recompute_blend_refusal` would compute
            // here -- no need to run it over an empty map. This registry has
            // no constructor that starts with residents already loaded; a
            // future one would need to call `recompute_blend_refusal` after
            // building `residents` instead of assuming this.
            blend_refusal: None,
            #[cfg(test)]
            blends: 0,
            #[cfg(test)]
            blend_plans: 0,
        }
    }

    /// Recompute the FULL resident set's blend feasibility (issue #1735) and
    /// cache it on `self.blend_refusal`.
    ///
    /// Called only where residency changes -- `load` after inserting, `unload`
    /// after removing -- because the resident set is the plan's entire input
    /// and nothing else this registry does can move it. `publish` reads the
    /// cache instead of re-running this on every call, including the two
    /// calls `apply` can make in one invocation when it unloads a previous
    /// blend before loading the next.
    ///
    /// A routed request applies EVERY resident adapter (ADR-094 decision 2),
    /// so that is exactly the set a blend feasibility report has to cover --
    /// not just the adapters an operator most recently applied. Computed via
    /// the SAME shared plan `blend_lora_layer_data` itself runs, so this can
    /// never disagree with the blend a routed request actually meets.
    fn recompute_blend_refusal(&mut self) {
        let projections = self.residents.values().flat_map(|adapter| {
            adapter
                .layers
                .iter()
                .map(|layer| lattice_fann::lora::BlendProjection {
                    layer_idx: layer.layer_idx,
                    module: layer.module.as_str(),
                    rank: layer.rank,
                    d_in: layer.d_in,
                    d_out: layer.d_out,
                })
        });
        self.blend_refusal = lattice_fann::lora::plan_blend("lora_residency", projections).err();
        #[cfg(test)]
        {
            self.blend_plans += 1;
        }
    }

    pub(crate) fn load(
        &mut self,
        name: String,
        path: String,
        layers: Vec<LoraLayerData>,
        descriptor: LoraDescriptor,
    ) -> Result<u32, AdapterControlError> {
        // Identity is lexical: aliases and changed file contents are not detected.
        let identity = (name.clone(), path.clone());
        if let Some(&id) = self.identities.get(&identity) {
            return Ok(id);
        }
        descriptor.validate()?;
        lattice_fann::lora::validate_target_modules(
            &descriptor.target_modules,
            lattice_fann::lora::KNOWN_LORA_TARGET_MODULES,
        )?;
        if layers.is_empty() || descriptor.rank == 0 {
            return Err("resident LoRA adapter must have layers and a nonzero rank".into());
        }
        let mut modules: Vec<_> = layers.iter().map(|layer| layer.module.as_str()).collect();
        modules.sort_unstable();
        modules.dedup();
        let mut declared: Vec<_> = descriptor
            .target_modules
            .iter()
            .map(String::as_str)
            .collect();
        declared.sort_unstable();
        declared.dedup();
        if modules != declared || layers.iter().any(|layer| layer.rank != descriptor.rank) {
            return Err(
                "resident LoRA layers disagree with descriptor rank or target modules".into(),
            );
        }
        if self.residents.len() >= self.limits.max_adapters {
            return Err(AdapterControlError::CountLimit {
                limit: self.limits.max_adapters,
            });
        }
        let byte_limit = || AdapterControlError::ByteLimit {
            limit: self.limits.max_bytes,
        };
        let payload_bytes = layers
            .iter()
            .try_fold(0usize, |sum, layer| {
                let bytes = layer
                    .a
                    .len()
                    .checked_add(layer.b.len())?
                    .checked_mul(size_of::<f32>())?;
                sum.checked_add(bytes)
            })
            .ok_or_else(byte_limit)?;
        let total_bytes = self
            .resident_bytes
            .checked_add(payload_bytes)
            .filter(|&total| total <= self.limits.max_bytes)
            .ok_or_else(byte_limit)?;
        let id = self.next_id.ok_or(AdapterControlError::IdExhausted)?;
        // A removed identifier must never redirect a queued request to new weights.
        self.next_id = id.checked_add(1);
        let metadata = AdapterMetadata {
            id,
            name,
            path,
            rank: descriptor.rank,
            layers: layers.len(),
        };
        self.residents.insert(
            id,
            ResidentAdapter {
                metadata,
                layers,
                descriptor,
                payload_bytes,
            },
        );
        self.identities.insert(identity, id);
        self.resident_bytes = total_bytes;
        // Residency just changed: the cached plan is stale before `publish`
        // copies it into `AdapterIndex`.
        self.recompute_blend_refusal();
        self.publish();
        Ok(id)
    }

    pub(crate) fn unload(
        &mut self,
        id: u32,
        slot: &mut impl AdapterSlot,
    ) -> Result<u32, AdapterControlError> {
        let adapter = self
            .residents
            .remove(&id)
            .ok_or(AdapterControlError::NotFound(id))?;
        if self.applied.iter().any(|entry| entry.id == id) {
            slot.unload();
            self.applied.clear();
        }
        self.resident_bytes -= adapter.payload_bytes;
        self.identities
            .remove(&(adapter.metadata.name, adapter.metadata.path));
        // Residency just changed: the cached plan is stale before `publish`
        // copies it into `AdapterIndex`.
        self.recompute_blend_refusal();
        self.publish();
        Ok(id)
    }

    pub(crate) fn metadata(&self, id: u32) -> Result<AdapterMetadata, AdapterControlError> {
        self.residents
            .get(&id)
            .map(|adapter| adapter.metadata.clone())
            .ok_or(AdapterControlError::NotFound(id))
    }

    pub(crate) fn apply(
        &mut self,
        selection: &[LoraSelection],
        slot: &mut impl AdapterSlot,
    ) -> Result<(), ApiError> {
        validate_scales(selection)?;
        validate_unique_ids(selection)?;
        // Resolve again after dequeue: an unload may have passed client validation.
        let inputs = selection
            .iter()
            .map(|entry| {
                let adapter = self
                    .residents
                    .get(&entry.id)
                    .ok_or_else(|| unknown_adapter(entry.id))?;
                Ok((
                    adapter.layers.as_slice(),
                    entry.scale * adapter.descriptor.scale(),
                ))
            })
            .collect::<Result<Vec<_>, ApiError>>()?;
        // Preserve order: concatenating ranks in another order changes FP reduction.
        if self.applied == selection {
            return Ok(());
        }
        let blended = if selection.is_empty() {
            None
        } else {
            #[cfg(test)]
            {
                self.blends += 1;
            }
            Some(
                blend_lora_layer_data(&inputs).map_err(|err| ApiError::BadRequest {
                    message: err.to_string(),
                    code: "lora_apply_failed",
                })?,
            )
        };
        // Validate the blend before destroying the previous slot. A GPU load failure
        // after unload leaves base active; publish that state, never a stale cache hit.
        if !self.applied.is_empty() {
            slot.unload();
            self.applied.clear();
            self.publish();
        }
        if let Some(layers) = blended {
            slot.load(layers).map_err(|message| ApiError::BadRequest {
                message,
                code: "lora_apply_failed",
            })?;
        }
        self.applied = selection.to_vec();
        self.publish();
        Ok(())
    }

    fn publish(&self) {
        let mut adapters: Vec<_> = self
            .residents
            .values()
            .map(|adapter| adapter.metadata.clone())
            .collect();
        adapters.sort_by_key(|adapter| adapter.id);

        // The cache, not a fresh plan: residency is unchanged since the last
        // `recompute_blend_refusal` (`load`/`unload`), and `apply` -- which
        // can call `publish` twice in one invocation -- never touches
        // `residents`, so re-planning here would recompute the identical
        // answer on every applied selection instead of only when the
        // resident set itself moves.
        let blend_refusal = self.blend_refusal.clone();

        // Only whole snapshots are assigned; recovering a poisoned lock cannot expose
        // a partially mutated metadata record.
        *self
            .index
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = AdapterIndex {
            adapters,
            applied: self.applied.clone(),
            blend_refusal,
        };
    }
}

#[cfg(test)]
mod tests;
