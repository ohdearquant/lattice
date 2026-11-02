#[cfg(feature = "qwen35")]
#[test]
fn router_model_layer_data_is_the_shared_layer_data() {
    use lattice_model_qwen35::forward::metal_qwen35::LoraLayerData;
    let _: fn(LoraLayerData) -> lattice_lora::LoraLayerData = |layer| layer;
    let _: fn(lattice_lora::LoraLayerData) -> LoraLayerData = |layer| layer;
}
