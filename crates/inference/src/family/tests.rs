use super::*;

#[test]
fn the_registry_resolves_each_family_by_kind() {
    let registry = registry().expect("the crate's table is valid");
    let name = |key, kind| registry.resolve(key, kind).map(FamilyDescriptor::name);
    #[cfg(feature = "qwen35")]
    assert_eq!(
        name(ResolveKey::ModelType("qwen3_5_moe"), ModelKind::Decoder),
        Ok("qwen35")
    );
    #[cfg(feature = "bert")]
    assert_eq!(
        name(ResolveKey::ModelType("bert"), ModelKind::Encoder),
        Ok("bert")
    );
    #[cfg(feature = "embeddinggemma2")]
    assert_eq!(
        name(ResolveKey::Family("embeddinggemma2"), ModelKind::Encoder),
        Ok("embeddinggemma2")
    );
    #[cfg(feature = "bert")]
    assert_eq!(
        name(ResolveKey::ModelType("bert"), ModelKind::Decoder),
        Err(RegistryError::WrongKind {
            requested: ModelKind::Decoder,
            families: vec!["bert"],
        })
    );
    assert_eq!(
        name(ResolveKey::ModelType("qwen3"), ModelKind::Encoder),
        Err(RegistryError::UnregisteredModelType {
            model_type: "qwen3".into()
        })
    );
    #[cfg(feature = "qwen35")]
    {
        let metal = LoadRequest::new("/nonexistent").with_backend(Backend::Metal);
        assert!(matches!(
            QWEN35.load_decoder(&metal).map(|_| ()),
            Err(InferenceError::UnsupportedModel(message)) if message == "family qwen35 does not offer the metal backend"
        ));
        assert!(
            QWEN35
                .load_decoder(&LoadRequest::new("/nonexistent"))
                .is_err()
        );
    }
    #[cfg(feature = "bert")]
    assert!(
        BERT.load_encoder(&LoadRequest::new("/nonexistent"))
            .is_err()
    );
}

#[cfg(not(feature = "embeddinggemma2"))]
#[test]
fn registry_omits_disabled_embeddinggemma2() {
    assert!(
        registry()
            .unwrap()
            .resolve(ResolveKey::Family("embeddinggemma2"), ModelKind::Encoder)
            .is_err()
    );
}

#[cfg(not(feature = "qwen35"))]
#[test]
fn registry_omits_disabled_qwen35() {
    assert!(
        registry()
            .unwrap()
            .resolve(ResolveKey::Family("qwen35"), ModelKind::Decoder)
            .is_err()
    );
}

#[test]
#[cfg(not(feature = "bert"))]
fn registry_does_not_register_a_disabled_bert_family() {
    let registry = registry().expect("the enabled family table is valid");
    assert!(matches!(
        registry.resolve(ResolveKey::Family("bert"), ModelKind::Encoder),
        Err(RegistryError::UnavailableFamily { .. })
    ));
}
