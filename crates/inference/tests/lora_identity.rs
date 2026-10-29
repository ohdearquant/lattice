#[cfg(all(feature = "qwen35", feature = "mixture"))]
#[test]
fn inference_facade_layer_data_is_the_shared_layer_data() {
    use lattice_fann::lora::LoraLayerData as SharedLayerData;
    use lattice_inference::forward::metal_qwen35::LoraLayerData;
    let _: fn(LoraLayerData) -> SharedLayerData = |layer| layer;
    let _: fn(SharedLayerData) -> LoraLayerData = |layer| layer;
}
