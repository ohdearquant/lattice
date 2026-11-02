#[test]
fn serve_facade_layer_data_is_the_shared_layer_data() {
    use lattice_inference::forward::metal_qwen35::LoraLayerData;
    let _: fn(LoraLayerData) -> lattice_lora::LoraLayerData = |layer| layer;
    let _: fn(lattice_lora::LoraLayerData) -> LoraLayerData = |layer| layer;
}
