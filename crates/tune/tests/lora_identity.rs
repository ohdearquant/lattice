#[test]
fn tune_layer_is_the_shared_layer() {
    let _: fn(lattice_tune::LoraLayer) -> lattice_lora::LoraLayer = |layer| layer;
    let _: fn(lattice_lora::LoraLayer) -> lattice_tune::lora::LoraLayer = |layer| layer;
    let _: fn(&lattice_tune::LoraLayer, f32, &[f32], &mut [f32]) = lattice_lora::apply_lora;
    let _: fn(&lattice_lora::LoraLayer, f32, &[f32], &mut [f32]) = lattice_tune::lora::apply_lora;
}
