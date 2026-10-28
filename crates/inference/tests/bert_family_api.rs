#![cfg(feature = "bert")]

use lattice_inference::{BertConfig, BertModel, CrossEncoderModel};

#[test]
fn bert_public_paths_name_the_family_types() {
    let _: fn(lattice_model_bert::BertEncoder) -> lattice_inference::encoder::BertEncoder =
        |value| value;
    let _: fn(lattice_model_bert::BertConfig) -> BertConfig = |value| value;
    let _: fn(lattice_model_bert::BertModel) -> BertModel = |value| value;
    let _: fn(lattice_model_bert::CrossEncoderModel) -> CrossEncoderModel = |value| value;
    let _: fn(lattice_model_bert::BertModel) -> lattice_inference::model::bert::BertModel =
        |value| value;
    let _: fn(
        lattice_model_bert::CrossEncoderModel,
    ) -> lattice_inference::model::cross_encoder::CrossEncoderModel = |value| value;
    let _: fn(
        lattice_model_bert::lora::BertLoraGeometry,
    ) -> lattice_inference::lora_hook::BertLoraGeometry = |value| value;
    let _ = lattice_inference::model::weight_loading::load_weights;
    let _ = lattice_inference::model::bert::weight_loading::load_classifier_weights;
}

#[test]
fn bert_descriptor_reexport_is_the_family_static() {
    assert!(std::ptr::eq(
        &lattice_inference::family::BERT,
        &lattice_model_bert::family::BERT
    ));
}
