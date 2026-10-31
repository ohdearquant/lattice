//! A consumer outside `lattice-inference` building its family table, resolving a checkpoint's
//! `model_type`, loading through the family descriptor, and driving a loaded decoder through
//! `Box<dyn DecoderModel>` with the shared driver.
#![cfg(feature = "native")]

use lattice_arch::decoder::driver;
use lattice_inference::bert_pooling_fixture::{DIMENSIONS, INPUTS, write_checkpoint};
use lattice_inference::family::{
    BERT, Backend, DecoderModel, FamilyDescriptor, LoadRequest, ModelKind, QWEN35,
    Qwen35CpuDecoder, Registry, RegistryError, ResolveKey,
};
use lattice_inference::model::qwen35::test_support::tiny_zero_model;
use lattice_inference::{BertModel, GenerateConfig, InferenceError, StopReason};
use std::path::Path;

fn model_type(dir: &Path) -> String {
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json reads");
    let config: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    config["model_type"]
        .as_str()
        .expect("config.json names a model_type")
        .to_string()
}

#[test]
fn a_consumer_resolves_and_loads_the_bert_family_from_its_checkpoint() {
    let parent = tempfile::tempdir().expect("a temporary directory");
    let dir = parent.path().join("checkpoint");
    write_checkpoint(&dir).expect("the synthetic checkpoint writes");
    let registry = Registry::new(&[&QWEN35, &BERT]).expect("the consumer's table is valid");

    let kind = model_type(&dir);
    let family = registry
        .resolve(ResolveKey::ModelType(&kind), ModelKind::Encoder)
        .expect("the checkpoint's model_type resolves");
    assert_eq!(family.name(), "bert");
    assert_eq!(
        registry
            .resolve(ResolveKey::ModelType(&kind), ModelKind::Decoder)
            .map(FamilyDescriptor::name),
        Err(RegistryError::WrongKind {
            requested: ModelKind::Decoder,
            families: vec!["bert"],
        })
    );

    // The descriptor and the load request cross the thread boundary; the loaded encoder is
    // built on the worker that uses it.
    let request = LoadRequest::new(&dir);
    let encoder = std::thread::spawn(move || family.load_encoder(&request))
        .join()
        .expect("the worker joins")
        .expect("the encoder loads");
    let expected = BertModel::from_directory(&dir)
        .expect("the checkpoint loads directly")
        .encode_batch(&INPUTS)
        .expect("the model encodes");
    let outputs = encoder.encode_texts(&INPUTS).expect("the encoder encodes");
    assert_eq!(encoder.output_dimension().get(), DIMENSIONS);
    for (output, vector) in outputs.iter().zip(&expected) {
        assert_eq!(output.vector(), vector.as_slice());
    }

    let metal = LoadRequest::new(&dir).with_backend(Backend::Metal);
    assert!(matches!(
        family.load_encoder(&metal).map(|_| ()),
        Err(InferenceError::UnsupportedModel(_))
    ));
    assert!(
        family
            .load_encoder(&LoadRequest::new(parent.path().join("missing")))
            .is_err()
    );
}

fn greedy(max_new_tokens: usize) -> GenerateConfig {
    let mut config = GenerateConfig::default();
    config.max_new_tokens = max_new_tokens;
    config.temperature = 0.0;
    config.stop_token_ids = vec![];
    config.enable_thinking = false;
    config
}

#[test]
fn a_consumer_drives_a_loaded_decoder_through_the_trait_object() {
    let expected = tiny_zero_model()
        .generate("abc", &greedy(4))
        .expect("the facade generates");
    let mut decoder: Box<dyn DecoderModel> = Box::new(Qwen35CpuDecoder::new(tiny_zero_model()));
    for _ in 0..2 {
        let request = decoder
            .admit("abc", greedy(4))
            .expect("the prompt is admitted");
        let text = decoder.text();
        let mut stream = text.stream();
        let mut session = decoder.session(&request).expect("the session opens");
        let mut full = String::new();
        let mut offsets = Vec::new();
        let never_cancel = || false;
        let result = driver::run_into(
            session.as_mut(),
            request.config(),
            request.think_close_id(),
            request.reasoning_newline_boundary().cloned(),
            request.prompt_ids(),
            request.eos_token_id(),
            false,
            &never_cancel,
            |_| {},
            |id, out| stream.push_into(id, out),
            &mut full,
            &mut offsets,
            |_, _| true,
            || {},
            String::new,
        )
        .expect("the driver runs the session");
        drop(session);
        full.push_str(&stream.finish());
        assert_eq!(result.generated_ids, expected.token_ids);
        assert_eq!(full, expected.text);
        assert_eq!(Some(result.stop_reason), expected.stop_reason);
        assert_eq!(result.stop_reason, StopReason::Length);
    }
}
