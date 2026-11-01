use crate::serve::lora_registry::AdapterSlot;
use lattice_inference::forward::metal_qwen35::MetalQwen35State;
use lattice_lora::LoraLayerData;

impl AdapterSlot for MetalQwen35State {
    fn load(&mut self, layers: Vec<LoraLayerData>) -> Result<(), String> {
        self.load_lora_adapter(layers, 1.0, None)
            .map_err(|err| err.to_string())
    }

    fn unload(&mut self) {
        self.unload_lora_adapter();
    }
}
