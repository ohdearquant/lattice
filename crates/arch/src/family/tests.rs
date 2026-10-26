use super::*;

fn decoder_loader(request: &LoadRequest) -> Result<Box<dyn DecoderModel>, InferenceError> {
    Err(InferenceError::ModelNotFound(format!(
        "decoder loader ran for {} on {}",
        request.model_dir().display(),
        request.backend()
    )))
}

fn encoder_loader(request: &LoadRequest) -> Result<Arc<dyn Encoder>, InferenceError> {
    Err(InferenceError::ModelNotFound(format!(
        "encoder loader ran for {} on {}",
        request.model_dir().display(),
        request.backend()
    )))
}

static TEXT: FamilyDescriptor =
    FamilyDescriptor::decoder("text", &["text_a", "shared"], decoder_loader);
static EMBED: FamilyDescriptor =
    FamilyDescriptor::encoder("embed", &["embed_a", "shared"], encoder_loader);
static BOTH: FamilyDescriptor = FamilyDescriptor::decoder("both", &["both_a"], decoder_loader)
    .with_encoder(encoder_loader)
    .with_backends(&[Backend::Cpu, Backend::Metal]);
static TEXT_AGAIN: FamilyDescriptor = FamilyDescriptor::decoder("text", &["other"], decoder_loader);
static TEXT_RIVAL: FamilyDescriptor =
    FamilyDescriptor::decoder("rival", &["text_a"], decoder_loader);
static EMBED_RIVAL: FamilyDescriptor =
    FamilyDescriptor::encoder("aardvark", &["text_a"], encoder_loader);
static SELF_CLAIM: FamilyDescriptor =
    FamilyDescriptor::encoder("twice", &["x", "x"], encoder_loader);

fn registry() -> Registry {
    Registry::new(&[&TEXT, &EMBED, &BOTH]).expect("a valid table")
}

fn loader_ran(result: Result<(), InferenceError>) -> bool {
    matches!(result, Err(InferenceError::ModelNotFound(_)))
}

#[test]
fn a_valid_table_keeps_its_order_and_lets_kinds_share_a_model_type() {
    let registry = registry();
    let names: Vec<&str> = registry.entries().iter().map(|d| d.name()).collect();
    assert_eq!(names, ["text", "embed", "both"]);
    assert_eq!(
        registry
            .resolve(ResolveKey::ModelType("shared"), ModelKind::Decoder)
            .map(FamilyDescriptor::name),
        Ok("text")
    );
    assert_eq!(
        registry
            .resolve(ResolveKey::ModelType("shared"), ModelKind::Encoder)
            .map(FamilyDescriptor::name),
        Ok("embed")
    );
    assert!(
        Registry::new(&[])
            .expect("an empty table")
            .entries()
            .is_empty()
    );
}

#[test]
fn duplicate_names_and_claims_are_refused_when_the_table_is_built() {
    assert_eq!(
        Registry::new(&[&TEXT, &EMBED, &TEXT_AGAIN]).map(|_| ()),
        Err(RegistryError::DuplicateFamily { name: "text" })
    );
    assert_eq!(
        Registry::new(&[&EMBED, &EMBED]).map(|_| ()),
        Err(RegistryError::DuplicateFamily { name: "embed" })
    );
    assert_eq!(
        Registry::new(&[&TEXT_RIVAL, &EMBED, &TEXT]).map(|_| ()),
        Err(RegistryError::DuplicateClaim {
            kind: ModelKind::Decoder,
            model_type: "text_a",
            families: vec!["rival", "text"],
        })
    );
    assert_eq!(
        Registry::new(&[&SELF_CLAIM]).map(|_| ()),
        Err(RegistryError::DuplicateClaim {
            kind: ModelKind::Encoder,
            model_type: "x",
            families: vec!["twice", "twice"],
        })
    );
    assert!(
        Registry::new(&[&TEXT, &EMBED_RIVAL]).is_ok(),
        "different kinds may share text_a"
    );
}

#[test]
fn resolution_is_exact_and_reports_misses_and_wrong_kinds_apart() {
    let registry = registry();
    assert_eq!(
        registry
            .resolve(ResolveKey::Family("both"), ModelKind::Encoder)
            .map(FamilyDescriptor::name),
        Ok("both")
    );
    assert_eq!(
        registry
            .resolve(ResolveKey::Family("Text"), ModelKind::Decoder)
            .map(FamilyDescriptor::name),
        Err(RegistryError::UnavailableFamily {
            name: "Text".into()
        })
    );
    assert_eq!(
        registry
            .resolve(ResolveKey::ModelType("text_"), ModelKind::Decoder)
            .map(FamilyDescriptor::name),
        Err(RegistryError::UnregisteredModelType {
            model_type: "text_".into()
        })
    );
    assert_eq!(
        registry
            .resolve(ResolveKey::Family("text"), ModelKind::Encoder)
            .map(FamilyDescriptor::name),
        Err(RegistryError::WrongKind {
            requested: ModelKind::Encoder,
            families: vec!["text"],
        })
    );
    assert_eq!(
        registry
            .resolve(ResolveKey::ModelType("embed_a"), ModelKind::Decoder)
            .map(FamilyDescriptor::name),
        Err(RegistryError::WrongKind {
            requested: ModelKind::Decoder,
            families: vec!["embed"],
        })
    );
    let rivals = Registry::new(&[&TEXT, &EMBED_RIVAL]).expect("a valid table");
    assert_eq!(
        rivals
            .resolve(ResolveKey::ModelType("text_a"), ModelKind::Encoder)
            .map(FamilyDescriptor::name),
        Ok("aardvark")
    );
}

#[test]
fn loading_refuses_a_missing_kind_or_backend_before_the_loader_runs() {
    let cpu = LoadRequest::new("/checkpoint");
    let metal = LoadRequest::new("/checkpoint").with_backend(Backend::Metal);
    assert_eq!(cpu.backend(), Backend::Cpu);
    assert_eq!(metal.model_dir(), Path::new("/checkpoint"));

    assert!(loader_ran(TEXT.load_decoder(&cpu).map(|_| ())));
    assert!(matches!(
        TEXT.load_decoder(&metal).map(|_| ()),
        Err(InferenceError::UnsupportedModel(message)) if message == "family text does not offer the metal backend"
    ));
    assert!(matches!(
        TEXT.load_encoder(&cpu).map(|_| ()),
        Err(InferenceError::UnsupportedModel(message)) if message == "family text offers no encoder"
    ));
    assert!(matches!(
        EMBED.load_decoder(&cpu).map(|_| ()),
        Err(InferenceError::UnsupportedModel(message)) if message == "family embed offers no decoder"
    ));
    assert!(loader_ran(BOTH.load_decoder(&metal).map(|_| ())));
    assert!(loader_ran(BOTH.load_encoder(&metal).map(|_| ())));
}

#[test]
fn descriptors_report_what_they_were_built_with() {
    assert_eq!(TEXT.backends(), &[Backend::Cpu]);
    assert_eq!(BOTH.backends(), &[Backend::Cpu, Backend::Metal]);
    assert_eq!(EMBED.model_types(), &["embed_a", "shared"]);
    assert!(BOTH.offers(ModelKind::Decoder) && BOTH.offers(ModelKind::Encoder));
    assert!(!EMBED.offers(ModelKind::Decoder));
    let debug = format!("{BOTH:?}");
    assert!(
        debug.contains("decoder: true") && debug.contains("encoder: true"),
        "{debug}"
    );
}

#[test]
fn registry_errors_convert_by_cause() {
    assert!(matches!(
        InferenceError::from(RegistryError::DuplicateFamily { name: "a" }),
        InferenceError::InvalidInput(_)
    ));
    let miss = InferenceError::from(RegistryError::UnavailableFamily { name: "a".into() });
    assert!(
        matches!(&miss, InferenceError::UnsupportedModel(m) if m == "family a is not available in this registry"),
        "{miss}"
    );
    assert_eq!(
        RegistryError::WrongKind {
            requested: ModelKind::Decoder,
            families: vec!["a", "b"],
        }
        .to_string(),
        "a, b offers no decoder"
    );
}
