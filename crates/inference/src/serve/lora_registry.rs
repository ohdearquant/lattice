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
mod tests {
    use super::*;

    #[derive(Default)]
    struct Slot {
        layers: Vec<LoraLayerData>,
        uploads: usize,
        unloads: usize,
        fail: bool,
    }
    impl AdapterSlot for Slot {
        fn load(&mut self, layers: Vec<LoraLayerData>) -> Result<(), String> {
            if self.fail {
                return Err("upload refused".into());
            }
            self.layers = layers;
            self.uploads += 1;
            Ok(())
        }
        fn unload(&mut self) {
            self.layers.clear();
            self.unloads += 1;
        }
    }
    impl Slot {
        fn output(&self) -> f32 {
            10.0 + self
                .layers
                .iter()
                .map(|layer| {
                    layer
                        .a
                        .iter()
                        .zip(&layer.b)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                })
                .sum::<f32>()
        }
    }
    fn registry() -> ResidencyRegistry {
        ResidencyRegistry::new(
            Arc::new(RwLock::new(AdapterIndex::default())),
            ResidencyLimits::default(),
        )
    }
    fn load(registry: &mut ResidencyRegistry, name: &str) -> u32 {
        registry
            .load(
                name.into(),
                format!("{name}.safetensors"),
                vec![LoraLayerData {
                    layer_idx: 0,
                    module: "q_proj".into(),
                    a: vec![2.0],
                    b: vec![3.0],
                    rank: 1,
                    d_in: 1,
                    d_out: 1,
                }],
                LoraDescriptor {
                    rank: 1,
                    alpha: 2.0,
                    target_modules: vec!["q_proj".into()],
                    dtype: "f32".into(),
                },
            )
            .unwrap()
    }

    fn limited(max_adapters: usize, max_bytes: usize) -> ResidencyRegistry {
        ResidencyRegistry::new(
            Arc::new(RwLock::new(AdapterIndex::default())),
            ResidencyLimits {
                max_adapters,
                max_bytes,
            },
        )
    }

    fn try_load(
        registry: &mut ResidencyRegistry,
        name: &str,
        path: &str,
    ) -> Result<u32, AdapterControlError> {
        registry.load(
            name.into(),
            path.into(),
            vec![LoraLayerData {
                layer_idx: 0,
                module: "q_proj".into(),
                a: vec![2.0],
                b: vec![3.0],
                rank: 1,
                d_in: 1,
                d_out: 1,
            }],
            LoraDescriptor {
                rank: 1,
                alpha: 2.0,
                target_modules: vec!["q_proj".into()],
                dtype: "f32".into(),
            },
        )
    }

    fn assert_rejected_without_state_change(
        registry: &mut ResidencyRegistry,
    ) -> AdapterControlError {
        let id = try_load(registry, "a", "a.safetensors").unwrap();
        let selection = [LoraSelection { id, scale: 1.0 }];
        let mut slot = Slot::default();
        registry.apply(&selection, &mut slot).unwrap();
        let before = serde_json::to_value(registry.index.read().unwrap().clone()).unwrap();
        let next_id = registry.next_id;
        let bytes = registry.resident_bytes;
        let output = slot.output();
        let error = try_load(registry, "b", "b.safetensors").unwrap_err();
        assert_eq!(
            serde_json::to_value(registry.index.read().unwrap().clone()).unwrap(),
            before
        );
        assert_eq!(registry.next_id, next_id);
        assert_eq!(registry.resident_bytes, bytes);
        assert_eq!(registry.residents.len(), 1);
        assert_eq!(registry.identities.len(), 1);
        registry.apply(&selection, &mut slot).unwrap();
        assert_eq!(slot.output(), output);
        assert_eq!(slot.uploads, 1);
        assert_eq!(slot.unloads, 0);
        error
    }

    #[test]
    fn count_limit_preserves_residents_and_applied_selection() {
        let mut registry = limited(1, 1024);
        let error = assert_rejected_without_state_change(&mut registry);
        assert!(matches!(
            error,
            AdapterControlError::CountLimit { limit: 1 }
        ));
    }

    #[test]
    fn byte_limit_preserves_residents_and_applied_selection() {
        let mut registry = limited(10, 8);
        let error = assert_rejected_without_state_change(&mut registry);
        assert!(matches!(error, AdapterControlError::ByteLimit { limit: 8 }));
    }

    #[test]
    fn dedup_at_capacity_does_not_spend_another_budget_slot() {
        let mut registry = limited(1, 8);
        let id = try_load(&mut registry, "a", "a.safetensors").unwrap();
        for _ in 0..10 {
            assert_eq!(try_load(&mut registry, "a", "a.safetensors").unwrap(), id);
        }
        assert_eq!(registry.residents.len(), 1);
        assert_eq!(registry.resident_bytes, 8);
        assert_eq!(registry.next_id, Some(1));
        let mut slot = Slot::default();
        registry.unload(id, &mut slot).unwrap();
        assert!(registry.residents.is_empty());
        assert!(registry.identities.is_empty());
        assert_eq!(registry.resident_bytes, 0);
        assert!(registry.unload(id, &mut slot).is_err());
        assert_eq!(
            try_load(&mut registry, "a", "a.safetensors").unwrap(),
            id + 1
        );
    }

    #[test]
    fn reused_identity_preserves_original_weights_and_metadata() {
        let mut registry = limited(1, 8);
        let id = try_load(&mut registry, "a", "a.safetensors").unwrap();
        let metadata = registry.metadata(id).unwrap();
        let reused = registry
            .load(
                "a".into(),
                "a.safetensors".into(),
                vec![LoraLayerData {
                    layer_idx: 0,
                    module: "q_proj".into(),
                    a: vec![100.0; 2],
                    b: vec![100.0; 2],
                    rank: 2,
                    d_in: 1,
                    d_out: 1,
                }],
                LoraDescriptor {
                    rank: 2,
                    alpha: 2.0,
                    target_modules: vec!["q_proj".into()],
                    dtype: "f32".into(),
                },
            )
            .unwrap();
        assert_eq!(reused, id);
        assert_eq!(registry.metadata(reused).unwrap(), metadata);
        assert_eq!(registry.resident_bytes, 8);
        let mut slot = Slot::default();
        registry
            .apply(&[LoraSelection { id, scale: 1.0 }], &mut slot)
            .unwrap();
        assert_eq!(slot.output(), 22.0);
    }

    #[test]
    fn oversized_first_payload_leaves_empty_registry() {
        let mut registry = limited(10, 7);
        assert!(matches!(
            try_load(&mut registry, "a", "a.safetensors"),
            Err(AdapterControlError::ByteLimit { limit: 7 })
        ));
        assert!(registry.residents.is_empty());
        assert!(registry.identities.is_empty());
        assert_eq!(registry.resident_bytes, 0);
        assert_eq!(registry.next_id, Some(0));
        assert!(registry.index.read().unwrap().adapters.is_empty());
    }

    #[test]
    fn identity_uses_both_exact_name_and_path() {
        let mut registry = limited(3, 24);
        let a = try_load(&mut registry, "a", "file.safetensors").unwrap();
        let b = try_load(&mut registry, "b", "file.safetensors").unwrap();
        let c = try_load(&mut registry, "a", "./file.safetensors").unwrap();
        assert_eq!((a, b, c), (0, 1, 2));
        assert_eq!(registry.resident_bytes, 24);
    }

    #[test]
    fn lowered_byte_limit_rejects_without_underflow() {
        let mut registry = limited(10, 16);
        try_load(&mut registry, "a", "a.safetensors").unwrap();
        registry.limits.max_bytes = 1;
        assert!(matches!(
            try_load(&mut registry, "b", "b.safetensors"),
            Err(AdapterControlError::ByteLimit { limit: 1 })
        ));
        assert_eq!(registry.resident_bytes, 8);
        assert_eq!(registry.residents.len(), 1);
    }

    #[test]
    fn total_byte_overflow_is_rejected() {
        let mut registry = limited(10, usize::MAX);
        registry.resident_bytes = usize::MAX - 4;
        assert!(matches!(
            try_load(&mut registry, "a", "a.safetensors"),
            Err(AdapterControlError::ByteLimit { .. })
        ));
        assert!(registry.residents.is_empty());
        assert_eq!(registry.next_id, Some(0));
    }

    #[tokio::test]
    async fn capacity_failure_emits_distinct_http_code() {
        use axum::response::IntoResponse;
        for mut registry in [limited(1, 1024), limited(10, 8)] {
            try_load(&mut registry, "a", "a.safetensors").unwrap();
            let error = try_load(&mut registry, "b", "b.safetensors").unwrap_err();
            let response = ApiError::from(error).into_response();
            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["error"]["code"], "lora_residency_limit_exceeded");
            assert_ne!(body["error"]["code"], "lora_load_failed");
        }
    }

    #[test]
    fn repeated_identity_reuses_one_resident() {
        let mut registry = registry();
        let first = load(&mut registry, "same");
        for _ in 0..8 {
            assert_eq!(load(&mut registry, "same"), first);
        }
        assert_eq!(registry.residents.len(), 1);
        assert_eq!(registry.next_id, Some(1));
    }

    #[test]
    fn residents_are_listed_without_applying_and_ids_are_never_reused() {
        let mut registry = registry();
        let mut slot = Slot::default();
        let first = load(&mut registry, "first");
        let second = load(&mut registry, "second");
        let snapshot = registry.index.read().unwrap().clone();
        assert_eq!(
            snapshot
                .adapters
                .iter()
                .map(|adapter| adapter.id)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(snapshot.applied.is_empty());
        assert_eq!(snapshot.adapters[1].name, "second");
        registry.unload(first, &mut slot).unwrap();
        let third = load(&mut registry, "third");
        assert_eq!((first, second, third), (0, 1, 2));
        assert!(
            registry
                .apply(
                    &[LoraSelection {
                        id: first,
                        scale: 1.0
                    }],
                    &mut slot
                )
                .unwrap_err()
                .message()
                .contains('0')
        );
        registry.next_id = Some(u32::MAX);
        assert_eq!(load(&mut registry, "last"), u32::MAX);
        assert!(registry.next_id.is_none());
    }

    #[test]
    fn duplicate_ids_are_rejected_before_blending_or_replacing_slot() {
        let mut registry = registry();
        let mut slot = Slot::default();
        let a = load(&mut registry, "a");
        let b = load(&mut registry, "b");
        let original = [LoraSelection { id: a, scale: 1.0 }];
        registry.apply(&original, &mut slot).unwrap();
        let duplicate = [
            LoraSelection { id: a, scale: 0.5 },
            LoraSelection { id: b, scale: 0.25 },
            LoraSelection {
                id: a,
                scale: -0.25,
            },
        ];
        let error = registry.apply(&duplicate, &mut slot).unwrap_err();
        assert!(matches!(error, ApiError::BadRequest { .. }));
        assert_eq!(error.code(), "lora_duplicate_adapter_id");
        assert_eq!(error.message(), format!("duplicate LoRA adapter id {a}"));
        assert_eq!((registry.blends, slot.uploads, slot.unloads), (1, 1, 0));
        assert_eq!(slot.output(), 22.0);
        assert_eq!(registry.applied, original);
        assert_eq!(registry.index.read().unwrap().applied, original);
    }

    #[test]
    fn same_mixture_blends_once_changed_order_or_scale_reblends() {
        let mut registry = registry();
        let mut slot = Slot::default();
        let a = load(&mut registry, "a");
        let b = load(&mut registry, "b");
        let mut selection = vec![
            LoraSelection { id: a, scale: 0.5 },
            LoraSelection { id: b, scale: 0.25 },
        ];
        registry.apply(&selection, &mut slot).unwrap();
        assert_eq!(slot.output(), 19.0);
        registry.apply(&selection, &mut slot).unwrap();
        assert_eq!((registry.blends, slot.uploads, slot.unloads), (1, 1, 0));
        selection.reverse();
        registry.apply(&selection, &mut slot).unwrap();
        assert_eq!((registry.blends, slot.uploads, slot.unloads), (2, 2, 1));
        selection[0].scale = 1.0;
        registry.apply(&selection, &mut slot).unwrap();
        assert_eq!(registry.blends, 3);
    }

    /// The blend-feasibility plan (issue #1735) is cached on the registry and
    /// recomputed only where residency itself moves -- `load` and `unload` --
    /// never inside `apply`, however many times a selection changes or `apply`
    /// calls `publish`. Reverting `publish` to recompute the plan itself
    /// (re-adding a call to `recompute_blend_refusal` -- or the old inline
    /// `plan_blend` over `self.residents` -- at the top of `publish`) makes
    /// `blend_plans` climb past 2 on the first `apply` call below and this
    /// test fails.
    #[test]
    fn apply_never_replans_feasibility_but_load_and_unload_do() {
        let mut registry = registry();
        let mut slot = Slot::default();
        let a = load(&mut registry, "a");
        let b = load(&mut registry, "b");
        assert_eq!(registry.blend_plans, 2, "one recompute per load");

        let mut selection = vec![
            LoraSelection { id: a, scale: 0.5 },
            LoraSelection { id: b, scale: 0.25 },
        ];
        registry.apply(&selection, &mut slot).unwrap();
        registry.apply(&selection, &mut slot).unwrap();
        selection.reverse();
        registry.apply(&selection, &mut slot).unwrap();
        selection[0].scale = 1.0;
        registry.apply(&selection, &mut slot).unwrap();
        assert_eq!(
            registry.blends, 3,
            "three distinct selections were actually blended"
        );
        assert_eq!(
            registry.blend_plans, 2,
            "apply() must read the cached plan, never recompute it"
        );

        registry.unload(a, &mut slot).unwrap();
        assert_eq!(
            registry.blend_plans, 3,
            "unload changes residency and must recompute"
        );
    }

    #[test]
    fn absent_selection_restores_base_output_with_residents_present() {
        let mut registry = registry();
        let mut slot = Slot::default();
        let id = load(&mut registry, "a");
        registry.apply(&[], &mut slot).unwrap();
        assert_eq!(slot.output(), 10.0);
        registry
            .apply(&[LoraSelection { id, scale: 1.0 }], &mut slot)
            .unwrap();
        assert_eq!(slot.output(), 22.0);
        registry.apply(&[], &mut slot).unwrap();
        assert_eq!(slot.output(), 10.0);
        assert_eq!(registry.residents.len(), 1);
        assert!(registry.index.read().unwrap().applied.is_empty());
    }

    #[test]
    fn invalid_descriptors_are_rejected_but_finite_zero_alpha_is_resident() {
        let mut registry = registry();
        let mut slot = Slot::default();
        for (rank, alpha) in [(0, 1.0), (1, f32::NAN), (1, f32::INFINITY)] {
            let result = registry.load(
                "bad".into(),
                "bad.safetensors".into(),
                vec![LoraLayerData {
                    layer_idx: 0,
                    module: "q_proj".into(),
                    a: vec![2.0],
                    b: vec![3.0],
                    rank,
                    d_in: 1,
                    d_out: 1,
                }],
                LoraDescriptor {
                    rank,
                    alpha,
                    target_modules: vec!["q_proj".into()],
                    dtype: "f32".into(),
                },
            );
            assert!(result.is_err());
            assert_eq!(registry.next_id, Some(0));
            assert!(registry.index.read().unwrap().adapters.is_empty());
        }
        let id = registry
            .load(
                "zero".into(),
                "zero.safetensors".into(),
                vec![LoraLayerData {
                    layer_idx: 0,
                    module: "q_proj".into(),
                    a: vec![2.0],
                    b: vec![3.0],
                    rank: 1,
                    d_in: 1,
                    d_out: 1,
                }],
                LoraDescriptor {
                    rank: 1,
                    alpha: 0.0,
                    target_modules: vec!["q_proj".into()],
                    dtype: "f32".into(),
                },
            )
            .unwrap();
        registry
            .apply(&[LoraSelection { id, scale: 1.0 }], &mut slot)
            .unwrap();
        assert_eq!(slot.output(), 10.0);
        assert_eq!((slot.uploads, registry.residents.len()), (1, 1));
    }

    #[test]
    fn unload_applied_adapter_clears_cache_and_failed_upload_never_hits() {
        let mut registry = registry();
        let mut slot = Slot::default();
        let a = load(&mut registry, "a");
        let b = load(&mut registry, "b");
        registry
            .apply(&[LoraSelection { id: a, scale: 1.0 }], &mut slot)
            .unwrap();
        slot.fail = true;
        let next = [LoraSelection { id: b, scale: 1.0 }];
        assert!(registry.apply(&next, &mut slot).is_err());
        assert!(registry.index.read().unwrap().applied.is_empty());
        slot.fail = false;
        registry.apply(&next, &mut slot).unwrap();
        registry.unload(b, &mut slot).unwrap();
        assert_eq!(slot.output(), 10.0);
        assert!(registry.apply(&next, &mut slot).is_err());
        assert!(
            registry
                .unload(90, &mut slot)
                .unwrap_err()
                .to_string()
                .contains("90")
        );
    }

    /// Loads one adapter with the given rank and projection shape, all at
    /// `(layer_idx=0, module="q_proj")` -- shaped so a caller loading two of
    /// these can sum their ranks at the SAME projection.
    fn load_shaped(
        registry: &mut ResidencyRegistry,
        name: &str,
        rank: usize,
        d_in: usize,
        d_out: usize,
    ) -> u32 {
        registry
            .load(
                name.into(),
                format!("{name}.safetensors"),
                vec![LoraLayerData {
                    layer_idx: 0,
                    module: "q_proj".into(),
                    a: vec![0.0f32; rank * d_in],
                    b: vec![0.0f32; d_out * rank],
                    rank,
                    d_in,
                    d_out,
                }],
                LoraDescriptor {
                    rank,
                    alpha: 1.0,
                    target_modules: vec!["q_proj".into()],
                    dtype: "f32".into(),
                },
            )
            .unwrap()
    }

    fn matching_artifact(names: &[&str]) -> crate::router_state::RouterArtifact {
        crate::router_state::RouterArtifact {
            version: 1,
            adapter_names: names.iter().map(|n| (*n).to_string()).collect(),
            representation: crate::router_state::TrainedRepresentation {
                embedding_model: "gme-qwen35".into(),
                pooling: "mean_visual".into(),
                prompt_source: "last_user_message".into(),
                loader_format: "test".into(),
                input_width: 4,
            },
            gate_bytes: vec![0],
        }
    }

    /// Issue #1735: `GET /v1/lora` reports a blend refusal for a resident
    /// set whose names match the gate exactly but whose summed rank exceeds
    /// the shared budget, and the identical set reduced to fit reports
    /// routable.
    #[test]
    fn a_resident_set_over_the_rank_budget_is_unroutable_and_reduced_is_routable() {
        // Two adapters at the SAME (layer_idx, module), so their ranks sum:
        // over budget (2049 + 2049 = 4098 > MAX_BLEND_RANK_TOTAL = 4096).
        let mut over = registry();
        load_shaped(&mut over, "a", 2049, 1, 1);
        load_shaped(&mut over, "b", 2049, 1, 1);
        let over_state = crate::serve::routing::routability(
            &matching_artifact(&["a", "b"]),
            &over.index.read().unwrap(),
        );
        assert!(!over_state.routable(), "the summed rank exceeds the budget");
        assert!(
            over_state
                .blend_refusal
                .as_deref()
                .is_some_and(|m| m.contains("MAX_BLEND_RANK_TOTAL")),
            "got: {:?}",
            over_state.blend_refusal
        );

        // The identical shape, reduced to fit (2000 + 2000 = 4000 <= 4096).
        let mut fits = registry();
        load_shaped(&mut fits, "a", 2000, 1, 1);
        load_shaped(&mut fits, "b", 2000, 1, 1);
        let fits_state = crate::serve::routing::routability(
            &matching_artifact(&["a", "b"]),
            &fits.index.read().unwrap(),
        );
        assert!(fits_state.routable(), "reduced to fit, the set must route");
        assert_eq!(fits_state.blend_refusal, None);
    }

    /// Consistency (issue #1735): for the same resident set, the published
    /// plan's verdict must equal what `apply` -- the real blend -- does.
    /// This is the property the whole refactor exists for: one shared
    /// function computes both, so they cannot disagree.
    #[test]
    fn the_published_blend_refusal_agrees_with_what_apply_actually_does() {
        let mut slot = Slot::default();

        let mut over = registry();
        let a = load_shaped(&mut over, "a", 2049, 1, 1);
        let b = load_shaped(&mut over, "b", 2049, 1, 1);
        assert!(
            over.index.read().unwrap().blend_refusal.is_some(),
            "published state must already flag the over-budget set"
        );
        let selection = [
            LoraSelection { id: a, scale: 1.0 },
            LoraSelection { id: b, scale: 1.0 },
        ];
        assert!(
            over.apply(&selection, &mut slot).is_err(),
            "the real blend must refuse exactly when the published plan says it will"
        );

        let mut fits = registry();
        let a = load_shaped(&mut fits, "a", 2000, 1, 1);
        let b = load_shaped(&mut fits, "b", 2000, 1, 1);
        assert!(
            fits.index.read().unwrap().blend_refusal.is_none(),
            "published state must not flag a set within budget"
        );
        let selection = [
            LoraSelection { id: a, scale: 1.0 },
            LoraSelection { id: b, scale: 1.0 },
        ];
        assert!(
            fits.apply(&selection, &mut slot).is_ok(),
            "the real blend must succeed exactly when the published plan says it will"
        );
    }
}
