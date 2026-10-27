#![cfg(feature = "qwen3")]

use lattice_inference::error::InferenceError;
use lattice_model_qwen3_embedding as family;
use lattice_weights::{QwenWeights, SafetensorsFile, ShardedSafetensors};
use std::marker::PhantomData;

#[test]
fn model_and_profiling_paths_name_the_family_types() {
    let _: PhantomData<family::QwenModel> = PhantomData::<lattice_inference::QwenModel>;
    let _: PhantomData<family::QwenModel> = PhantomData::<lattice_inference::model::QwenModel>;
    let _: PhantomData<family::QwenModel> =
        PhantomData::<lattice_inference::model::qwen::QwenModel>;
    let _: PhantomData<family::ModelInferenceConfig> =
        PhantomData::<lattice_inference::model::qwen::ModelInferenceConfig>;
    let _: PhantomData<family::ProfileTimings> = PhantomData::<lattice_inference::ProfileTimings>;
    let _: PhantomData<family::ProfileTimings> =
        PhantomData::<lattice_inference::model::ProfileTimings>;
    let _: PhantomData<family::ProfileTimings> =
        PhantomData::<lattice_inference::model::qwen::ProfileTimings>;
    let _: PhantomData<family::LayerTimings> = PhantomData::<lattice_inference::LayerTimings>;
    let _: PhantomData<family::LayerTimings> =
        PhantomData::<lattice_inference::model::LayerTimings>;
    let _: PhantomData<family::LayerTimings> =
        PhantomData::<lattice_inference::model::qwen::LayerTimings>;
    let _: PhantomData<family::QwenConfig> = PhantomData::<lattice_inference::QwenConfig>;
    let _: PhantomData<family::QwenConfig> = PhantomData::<lattice_inference::model::QwenConfig>;
    let _: PhantomData<family::QwenConfig> =
        PhantomData::<lattice_inference::model::qwen::QwenConfig>;
}

#[test]
fn loader_paths_keep_the_storage_and_owner_borrows() {
    use lattice_inference::model::qwen::weight_loading as facade;
    use lattice_model_qwen3_embedding::weight_loading as family_loading;
    type Borrowed = for<'a> fn(
        &'a SafetensorsFile,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) -> Result<QwenWeights<'a>, InferenceError>;
    type Owned = fn(
        &mut ShardedSafetensors,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) -> Result<family_loading::OwnedQwenWeights, InferenceError>;
    let facade_borrowed: Borrowed = facade::load_weights;
    let family_borrowed: Borrowed = family_loading::load_weights;
    assert!(std::ptr::fn_addr_eq(facade_borrowed, family_borrowed));
    let facade_owned: Owned = facade::load_owned_weights;
    let family_owned: Owned = family_loading::load_owned_weights;
    assert!(std::ptr::fn_addr_eq(facade_owned, family_owned));
    let _: PhantomData<family_loading::OwnedQwenWeights> = PhantomData::<facade::OwnedQwenWeights>;
    let _: for<'a> fn(&'a facade::OwnedQwenWeights) -> QwenWeights<'a> =
        family_loading::OwnedQwenWeights::assemble_borrowed_weights;
}

#[test]
fn encoder_and_platform_backend_paths_name_the_family_types() {
    let _: PhantomData<family::QwenEncoder> =
        PhantomData::<lattice_inference::encoder::QwenEncoder>;
    let _: PhantomData<family::metal::MetalForwardPass> =
        PhantomData::<lattice_inference::forward::metal::MetalForwardPass>;
    fn require_encoder<T: lattice_arch::encoder::Encoder>() {}
    require_encoder::<lattice_inference::encoder::QwenEncoder>();
}
