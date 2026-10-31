#![cfg(feature = "paddleocr-vl")]

use std::marker::PhantomData;

#[test]
fn paddleocr_public_paths_name_the_family_types() {
    use lattice_inference::model::paddleocr_vl as facade;
    use lattice_inference::vision::{
        paddleocr_preprocess as facade_pre, paddleocr_vit as facade_vit,
    };
    use lattice_model_ernie45::model::paddleocr_vl as family;
    use lattice_model_ernie45::vision::{
        paddleocr_preprocess as family_pre, paddleocr_vit as family_vit,
    };

    let _: PhantomData<family::PaddleOcrVlModel> = PhantomData::<facade::PaddleOcrVlModel>;
    let _: PhantomData<family::PrefillTrace> = PhantomData::<facade::PrefillTrace>;
    let _: PhantomData<family_pre::PaddleOcrImageProcessorConfig> =
        PhantomData::<facade_pre::PaddleOcrImageProcessorConfig>;
    let _: PhantomData<family_pre::PreprocessedImage> =
        PhantomData::<facade_pre::PreprocessedImage>;
    let _: PhantomData<family_vit::PaddleOcrVisionConfig> =
        PhantomData::<facade_vit::PaddleOcrVisionConfig>;
    let _: PhantomData<family_vit::PaddleOcrVitLayerWeights> =
        PhantomData::<facade_vit::PaddleOcrVitLayerWeights>;
    let _: PhantomData<family_vit::PaddleOcrVisionWeights> =
        PhantomData::<facade_vit::PaddleOcrVisionWeights>;
    let _: PhantomData<family_vit::PaddleOcrVisionTrace> =
        PhantomData::<facade_vit::PaddleOcrVisionTrace>;
}

#[test]
fn paddleocr_argmax_is_the_family_function() {
    let facade: fn(&[f32]) -> u32 = lattice_inference::model::paddleocr_vl::argmax_u32;
    let family: fn(&[f32]) -> u32 = lattice_model_ernie45::model::paddleocr_vl::argmax_u32;
    assert!(std::ptr::fn_addr_eq(facade, family));
}
