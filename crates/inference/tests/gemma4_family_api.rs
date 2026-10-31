use std::marker::PhantomData;

#[test]
fn config_and_scalar_paths_remain_independent_of_the_family() {
    let _: PhantomData<lattice_arch::gemma4_config::Gemma4Config> =
        PhantomData::<lattice_inference::model::gemma4_config::Gemma4Config>;
    let facade: fn(&[u32], &[f32], usize, &mut [f32]) =
        lattice_inference::model::gemma4_ops::gemma4_scaled_embedding;
    let lower: fn(&[u32], &[f32], usize, &mut [f32]) =
        lattice_compute::gemma_ops::gemma4_scaled_embedding;
    assert!(std::ptr::fn_addr_eq(facade, lower));
    let facade: fn(&mut [f32], f32) = lattice_inference::model::gemma4_ops::gemma4_logit_softcap;
    let lower: fn(&mut [f32], f32) = lattice_compute::gemma_ops::gemma4_logit_softcap;
    assert!(std::ptr::fn_addr_eq(facade, lower));
}

#[cfg(all(feature = "gemma4", feature = "test-utils"))]
#[test]
fn test_utils_factory_keeps_its_old_public_path() {
    let _: fn(usize) -> lattice_model_gemma4::Gemma4Model =
        lattice_inference::model::gemma4_model::tiny_zero_model_with_vocab;
}

#[cfg(all(feature = "gemma4", feature = "serving-support"))]
#[test]
fn tracing_bridge_keeps_the_arch_observation_types() {
    type Traced = Result<
        (
            lattice_arch::generation::GenerateOutput,
            lattice_arch::decoder::driver::DriverTrace,
        ),
        lattice_inference_core::InferenceError,
    >;
    let _: fn(
        &lattice_model_gemma4::Gemma4Model,
        &[u32],
        &lattice_arch::generation::GenerateConfig,
    ) -> Traced = lattice_model_gemma4::__internal::generate_traced;
}
