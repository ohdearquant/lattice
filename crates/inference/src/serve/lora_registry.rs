//! Worker-local adapter weights and the currently materialized mixture.

use super::ApiError;
use super::lora::{AdapterIndex, AdapterMetadata, LoraSelection, unknown_adapter, validate_scales};
use crate::forward::metal_qwen35::{LoraLayerData, MetalQwen35State, blend_lora_layer_data};
use lattice_fann::lora::LoraDescriptor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub(super) trait AdapterSlot {
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
}

pub(super) struct ResidencyRegistry {
    residents: HashMap<u32, ResidentAdapter>,
    next_id: Option<u32>,
    // Order determines concatenated rank order and hence floating-point reduction.
    applied: Vec<LoraSelection>,
    index: Arc<RwLock<AdapterIndex>>,
    #[cfg(test)]
    blends: usize,
}

impl ResidencyRegistry {
    pub(super) fn new(index: Arc<RwLock<AdapterIndex>>) -> Self {
        Self {
            residents: HashMap::new(),
            next_id: Some(0),
            applied: Vec::new(),
            index,
            #[cfg(test)]
            blends: 0,
        }
    }

    pub(super) fn load(
        &mut self,
        name: String,
        path: String,
        layers: Vec<LoraLayerData>,
        descriptor: LoraDescriptor,
    ) -> Result<u32, String> {
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
        let id = self.next_id.ok_or("LoRA adapter id space exhausted")?;
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
            },
        );
        self.publish();
        Ok(id)
    }

    pub(super) fn unload(&mut self, id: u32, slot: &mut impl AdapterSlot) -> Result<u32, String> {
        if !self.residents.contains_key(&id) {
            return Err(format!("unknown LoRA adapter id {id}"));
        }
        if self.applied.iter().any(|entry| entry.id == id) {
            slot.unload();
            self.applied.clear();
        }
        self.residents.remove(&id);
        self.publish();
        Ok(id)
    }

    pub(super) fn apply(
        &mut self,
        selection: &[LoraSelection],
        slot: &mut impl AdapterSlot,
    ) -> Result<(), ApiError> {
        validate_scales(selection)?;
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
        // Only whole snapshots are assigned; recovering a poisoned lock cannot expose
        // a partially mutated metadata record.
        *self
            .index
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = AdapterIndex {
            adapters,
            applied: self.applied.clone(),
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
        ResidencyRegistry::new(Arc::new(RwLock::new(AdapterIndex::default())))
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
        assert!(registry.unload(90, &mut slot).unwrap_err().contains("90"));
    }
}
