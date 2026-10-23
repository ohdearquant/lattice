use super::*;
use crate::model_format::ModelFamily;
use crate::serve::route::{select_route, select_standalone_route};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const FORMATS: [FormatFixture; 5] = [
    FormatFixture::SafetensorsFile,
    FormatFixture::SafetensorsIndex,
    FormatFixture::BothSafetensorsSentinels,
    FormatFixture::Q4Only,
    FormatFixture::Empty,
];

/// One F32 tensor `w` of shape `[1]`: the smallest file format
/// detection reads as an unquantized safetensors checkpoint.
fn write_minimal_safetensors(path: &Path) {
    let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0u8; 4]);
    fs::write(path, bytes).expect("write minimal safetensors");
}

#[derive(Debug, Clone, Copy)]
enum FormatFixture {
    SafetensorsFile,
    SafetensorsIndex,
    BothSafetensorsSentinels,
    Q4Only,
    Empty,
}

impl FormatFixture {
    fn expected(self) -> ModelFormat {
        match self {
            Self::SafetensorsFile | Self::SafetensorsIndex | Self::BothSafetensorsSentinels => {
                ModelFormat::Safetensors
            }
            Self::Q4Only => ModelFormat::Q4,
            Self::Empty => ModelFormat::Unknown,
        }
    }

    fn write(self, directory: &Path) {
        match self {
            Self::SafetensorsFile => {
                write_minimal_safetensors(&directory.join("model.safetensors"));
            }
            Self::SafetensorsIndex => {
                write_minimal_safetensors(&directory.join("model-00001-of-00001.safetensors"));
                fs::write(
                    directory.join("model.safetensors.index.json"),
                    br#"{"weight_map":{"w":"model-00001-of-00001.safetensors"}}"#,
                )
                .expect("write safetensors index");
            }
            Self::BothSafetensorsSentinels => {
                write_minimal_safetensors(&directory.join("model.safetensors"));
                fs::write(
                    directory.join("model.safetensors.index.json"),
                    br#"{"weight_map":{"w":"model.safetensors"}}"#,
                )
                .expect("write safetensors index");
                fs::write(directory.join("weights.q4"), b"q4").expect("write q4 sentinel");
            }
            Self::Q4Only => {
                fs::write(directory.join("weights.q4"), b"q4").expect("write q4 sentinel");
            }
            Self::Empty => {}
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MetadataFixture {
    Qwen,
    QwenMoe,
    GemmaUnsupportedProfile,
    Missing,
    Unreadable,
    Malformed,
    Oversized,
    WrongType,
    NestedOnly,
}

impl MetadataFixture {
    fn expected_model_type(self) -> Option<&'static str> {
        match self {
            Self::Qwen => Some("qwen3_5"),
            Self::QwenMoe => Some("qwen3_5_moe"),
            Self::GemmaUnsupportedProfile => Some("gemma4"),
            Self::WrongType => Some("bert"),
            Self::Missing
            | Self::Unreadable
            | Self::Malformed
            | Self::Oversized
            | Self::NestedOnly => None,
        }
    }

    fn expected_provider(self) -> &'static str {
        match self {
            Self::GemmaUnsupportedProfile => "gemma",
            Self::Qwen | Self::QwenMoe => "qwen",
            Self::Missing
            | Self::Unreadable
            | Self::Malformed
            | Self::Oversized
            | Self::WrongType
            | Self::NestedOnly => "qwen",
        }
    }

    fn write(self, directory: &Path) {
        let path = directory.join("config.json");
        match self {
            Self::Qwen => {
                fs::write(&path, br#"{"model_type":"qwen3_5"}"#).expect("write qwen config")
            }
            Self::QwenMoe => {
                fs::write(&path, br#"{"model_type":"qwen3_5_moe"}"#).expect("write qwen moe config")
            }
            Self::GemmaUnsupportedProfile => fs::write(
                &path,
                br#"{"model_type":"gemma4","architectures":["unsupported"]}"#,
            )
            .expect("write unsupported Gemma profile"),
            Self::Missing => {}
            Self::Unreadable => fs::create_dir(&path).expect("create non-file config path"),
            Self::Malformed => fs::write(&path, b"{").expect("write malformed config"),
            Self::Oversized => {
                let bytes =
                    vec![b' '; crate::model::config_file::MAX_CONFIG_JSON_BYTES as usize + 1];
                fs::write(&path, bytes).expect("write oversized config");
            }
            Self::WrongType => {
                fs::write(&path, br#"{"model_type":"bert"}"#).expect("write other model type")
            }
            Self::NestedOnly => fs::write(&path, br#"{"text_config":{"model_type":"gemma4"}}"#)
                .expect("write nested-only model type"),
        }
    }
}

#[test]
fn format_and_top_level_model_type_matrix_selects_expected_provider() {
    let metadata = [
        MetadataFixture::Qwen,
        MetadataFixture::QwenMoe,
        MetadataFixture::GemmaUnsupportedProfile,
        MetadataFixture::Missing,
        MetadataFixture::Unreadable,
        MetadataFixture::Malformed,
        MetadataFixture::Oversized,
        MetadataFixture::WrongType,
        MetadataFixture::NestedOnly,
    ];

    for format_fixture in FORMATS {
        for metadata_fixture in metadata {
            let temp = tempfile::tempdir().expect("create checkpoint tempdir");
            let directory = temp.path().join("checkpoint");
            fs::create_dir(&directory).expect("create checkpoint directory");
            format_fixture.write(&directory);
            metadata_fixture.write(&directory);

            let evidence = inspect_checkpoint(&directory);
            assert_eq!(evidence.format(), format_fixture.expected());
            assert_eq!(
                evidence.model_type.as_deref(),
                metadata_fixture.expected_model_type(),
                "metadata fixture {metadata_fixture:?}"
            );
            match select(evidence, ServingEntry::Standalone) {
                Ok(selected) => assert_eq!(
                    selected.provider.name(),
                    metadata_fixture.expected_provider(),
                    "format fixture {format_fixture:?}, metadata fixture {metadata_fixture:?}"
                ),
                Err(SelectionError::Route {
                    refusal: RouteRefusal::GemmaMetalUnsupported,
                    ..
                }) => assert!(matches!(
                    (format_fixture, metadata_fixture),
                    (
                        FormatFixture::Q4Only,
                        MetadataFixture::GemmaUnsupportedProfile
                    )
                )),
                Err(error) => panic!(
                    "unexpected refusal for format fixture {format_fixture:?}, metadata fixture {metadata_fixture:?}: {error:?}"
                ),
            }
        }
    }

    let temp = tempfile::tempdir().expect("create missing checkpoint parent");
    let missing = temp.path().join("missing");
    let evidence = inspect_checkpoint(&missing);
    assert_eq!(evidence.format(), ModelFormat::Unknown);
    assert_eq!(evidence.model_type, None);
    let selected = select(evidence, ServingEntry::Standalone)
        .expect("missing directory preserves standalone no-route path");
    assert_eq!(selected.provider.name(), "qwen");
    assert_eq!(selected.legacy_route(), None);
}

#[test]
fn gemma_identity_is_selected_even_for_an_unsupported_or_malformed_profile() {
    for config in [
        br#"{"model_type":"gemma4","architectures":["unsupported"]}"#.as_slice(),
        br#"{"model_type":"gemma4","text_config":{}}"#.as_slice(),
    ] {
        let temp = tempfile::tempdir().expect("create checkpoint tempdir");
        write_minimal_safetensors(&temp.path().join("model.safetensors"));
        fs::write(temp.path().join("config.json"), config).expect("write Gemma config");
        let selected = select(inspect_checkpoint(temp.path()), ServingEntry::Lattice)
            .expect("identity claim does not validate the Gemma profile");
        assert_eq!(selected.provider.name(), "gemma");
        assert_eq!(selected.legacy_route(), Some(ServedRoute::GEMMA4_CPU));
    }
}

#[test]
fn provider_route_projections_match_the_existing_route_functions() {
    let family_providers: [(&dyn RegisteredProvider, ModelFamily, &str); 2] = [
        (&providers::QWEN_PROVIDER, ModelFamily::Qwen35, "qwen3_5"),
        (&providers::GEMMA_PROVIDER, ModelFamily::Gemma4, "gemma4"),
    ];

    for (provider, family, model_type) in family_providers {
        for format in [
            ModelFormat::Safetensors,
            ModelFormat::Q4,
            ModelFormat::Unknown,
        ] {
            for entry in [ServingEntry::Lattice, ServingEntry::Standalone] {
                let evidence = CheckpointEvidence {
                    directory: PathBuf::new(),
                    format,
                    model_type: Some(model_type.to_owned()),
                };
                let provider_list = [provider];
                let selected = select_with_registry(
                    evidence,
                    entry,
                    &ProviderRegistry {
                        providers: &provider_list,
                        legacy_generation: &providers::QWEN_PROVIDER,
                    },
                );
                let actual = selected
                    .map(|selection| selection.legacy_route())
                    .map_err(|error| match error {
                        SelectionError::Route { refusal, .. } => refusal,
                        SelectionError::Ambiguous { .. } => {
                            panic!("one provider cannot produce an ambiguity")
                        }
                    });
                let expected = match entry {
                    ServingEntry::Lattice => select_route(format, family).map(Some),
                    ServingEntry::Standalone => match select_standalone_route(format, family) {
                        Ok(route) => Ok(Some(route)),
                        Err(RouteRefusal::UnrecognizedFormat) => Ok(None),
                        Err(refusal) => Err(refusal),
                    },
                };
                assert_eq!(actual, expected, "{family:?}, {format:?}, {entry:?}");
            }
        }
    }
}

#[test]
fn lattice_plan_admission_preserves_route_markers_refusals_and_implicit_actions() {
    let metadata = [
        MetadataFixture::Qwen,
        MetadataFixture::QwenMoe,
        MetadataFixture::GemmaUnsupportedProfile,
        MetadataFixture::Missing,
        MetadataFixture::Unreadable,
        MetadataFixture::Malformed,
        MetadataFixture::Oversized,
        MetadataFixture::WrongType,
        MetadataFixture::NestedOnly,
    ];

    for format_fixture in FORMATS {
        for metadata_fixture in metadata {
            let temp = tempfile::tempdir().expect("create checkpoint tempdir");
            let directory = temp.path().join("checkpoint");
            fs::create_dir(&directory).expect("create checkpoint directory");
            format_fixture.write(&directory);
            metadata_fixture.write(&directory);

            let evidence = inspect_checkpoint(&directory);
            let family = if metadata_fixture.expected_provider() == "gemma" {
                ModelFamily::Gemma4
            } else {
                ModelFamily::Qwen35
            };
            let expected_route = select_route(evidence.format, family);
            let actual = select(evidence.clone(), ServingEntry::Lattice);

            match expected_route {
                Ok(route) => {
                    let selected = actual.expect("supported route is admitted without loading");
                    let expected_marker = route.selection_marker(evidence.format);
                    assert_eq!(
                        selected.lattice_selection_marker().as_deref(),
                        Some(expected_marker.as_str()),
                        "{format_fixture:?}, {metadata_fixture:?}"
                    );
                    assert_eq!(selected.legacy_route(), Some(route));
                    assert_eq!(
                        selected.implicit_embeddings("fixture model"),
                        selected.implicit_pooled_embeddings("fixture model"),
                        "{format_fixture:?}, {metadata_fixture:?}"
                    );
                }
                Err(refusal) => {
                    let error = actual.err().expect("unsupported route keeps its refusal");
                    assert_eq!(
                        error.message(),
                        refusal.message(&directory),
                        "{format_fixture:?}, {metadata_fixture:?}"
                    );
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn provider_standalone_descriptions_keep_the_existing_labels_and_markers() {
    let formats = [
        ModelFormat::Safetensors,
        ModelFormat::Q4,
        ModelFormat::Unknown,
        ModelFormat::MlxQuantized,
        ModelFormat::Gguf,
        ModelFormat::Refused,
    ];
    let providers: [(&dyn ServingProvider, ModelFamily, &str); 2] = [
        (&providers::QWEN_PROVIDER, ModelFamily::Qwen35, "qwen3_5"),
        (&providers::GEMMA_PROVIDER, ModelFamily::Gemma4, "gemma4"),
    ];

    for (provider, family, model_type) in providers {
        for format in formats {
            let evidence = CheckpointEvidence {
                directory: PathBuf::from("checkpoint"),
                format,
                model_type: Some(model_type.to_owned()),
            };
            let route = provider
                .legacy_route(&evidence, ServingEntry::Standalone)
                .ok()
                .flatten();
            let description = provider.standalone_startup_description(format, route);
            let expected_route = select_standalone_route(format, family).ok();
            let expected_marker = expected_route.map(|route| route.selection_marker(format));
            let expected_label = match (family, format, expected_route) {
                (ModelFamily::Gemma4, ModelFormat::Safetensors, Some(_)) => "safetensors",
                (_, ModelFormat::Q4, _) => "q4",
                (_, ModelFormat::Safetensors, _) => "bf16",
                _ => "unknown",
            };
            assert_eq!(description.route_marker, expected_marker);
            assert_eq!(description.loading_label, expected_label);
        }
    }
}

#[test]
fn ambiguity_is_sorted_duplicate_sensitive_and_order_independent() {
    let alpha = TestProvider::new("alpha", true, Ok(Some(ServedRoute::QWEN35_CPU)));
    let zeta = TestProvider::new("zeta", true, Ok(Some(ServedRoute::GEMMA4_CPU)));
    let forward = [&zeta as &dyn RegisteredProvider, &alpha];
    let reverse = [&alpha as &dyn RegisteredProvider, &zeta];
    let legacy = TestProvider::new("legacy", false, Ok(None));

    let forward_error = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &forward,
            legacy_generation: &legacy,
        },
    )
    .err()
    .expect("two providers must be ambiguous");
    let reverse_error = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &reverse,
            legacy_generation: &legacy,
        },
    )
    .err()
    .expect("reversing registration must remain ambiguous");
    assert_eq!(forward_error, reverse_error);
    assert_eq!(
        forward_error,
        SelectionError::Ambiguous {
            directory: PathBuf::new(),
            claimants: vec!["alpha", "zeta"]
        }
    );
    assert_eq!(
        forward_error.message(),
        "ambiguous serving providers for '': alpha, zeta"
    );
    assert_eq!(alpha.route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(zeta.route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(alpha.admission_calls.load(Ordering::SeqCst), 0);
    assert_eq!(zeta.admission_calls.load(Ordering::SeqCst), 0);

    let duplicated = [&alpha as &dyn RegisteredProvider, &alpha];
    let duplicate_error = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &duplicated,
            legacy_generation: &legacy,
        },
    )
    .err()
    .expect("the same provider listed twice is two claims");
    assert_eq!(
        duplicate_error,
        SelectionError::Ambiguous {
            directory: PathBuf::new(),
            claimants: vec!["alpha", "alpha"]
        }
    );
    assert_eq!(alpha.route_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn positive_claim_refusal_does_not_reach_the_legacy_provider() {
    let claimant = TestProvider::new("claimant", true, Err(RouteRefusal::GemmaMetalUnsupported));
    let legacy = TestProvider::new("legacy", false, Ok(None));
    let providers = [&claimant as &dyn RegisteredProvider, &legacy];
    let error = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &providers,
            legacy_generation: &legacy,
        },
    )
    .err()
    .expect("a claimed provider's route refusal is final");

    assert!(matches!(
        error,
        SelectionError::Route {
            refusal: RouteRefusal::GemmaMetalUnsupported,
            ..
        }
    ));
    assert_eq!(claimant.admission_calls.load(Ordering::SeqCst), 1);
    assert_eq!(claimant.route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(legacy.route_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn legacy_provider_route_is_reached_only_when_no_provider_claims() {
    let nonclaimant = TestProvider::new("nonclaimant", false, Ok(None));
    let legacy = TestProvider::new("legacy", false, Ok(Some(ServedRoute::QWEN35_CPU)));
    let providers = [&nonclaimant as &dyn RegisteredProvider, &legacy];
    let selected = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &providers,
            legacy_generation: &legacy,
        },
    )
    .expect("zero claims binds the registered legacy provider");

    assert_eq!(selected.provider.name(), "legacy");
    assert_eq!(nonclaimant.admission_calls.load(Ordering::SeqCst), 0);
    assert_eq!(legacy.admission_calls.load(Ordering::SeqCst), 1);
    assert_eq!(nonclaimant.route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(legacy.route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(selected.legacy_route(), Some(ServedRoute::QWEN35_CPU));
    assert_eq!(legacy.route_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn unknown_format_matches_the_legacy_route_behavior_on_each_entry() {
    for config in [None, Some(br#"{"model_type":"gemma4"}"#.as_slice())] {
        let temp = tempfile::tempdir().expect("create checkpoint tempdir");
        if let Some(config) = config {
            fs::write(temp.path().join("config.json"), config).expect("write model config");
        }
        let evidence = inspect_checkpoint(temp.path());
        let lattice = select(evidence, ServingEntry::Lattice)
            .err()
            .expect("lattice still refuses unknown format");
        assert!(matches!(
            lattice,
            SelectionError::Route {
                refusal: RouteRefusal::UnrecognizedFormat,
                ..
            }
        ));

        let standalone = select(inspect_checkpoint(temp.path()), ServingEntry::Standalone)
            .expect("standalone keeps its old loader error path");
        assert_eq!(standalone.legacy_route(), None);
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn gemma_standalone_refuses_present_options_in_legacy_order() {
    let cases = [
        (
            StandaloneOptionPresence {
                preload_vision: true,
                tokenizer_dir: false,
                resident_count: false,
                resident_bytes: false,
                positive_reasoning_budget: false,
            },
            "--preload-vision",
        ),
        (
            StandaloneOptionPresence {
                preload_vision: false,
                tokenizer_dir: true,
                resident_count: false,
                resident_bytes: false,
                positive_reasoning_budget: false,
            },
            "--tokenizer-dir",
        ),
        (
            StandaloneOptionPresence {
                preload_vision: false,
                tokenizer_dir: false,
                resident_count: true,
                resident_bytes: false,
                positive_reasoning_budget: false,
            },
            "--max-resident-adapters",
        ),
        (
            StandaloneOptionPresence {
                preload_vision: false,
                tokenizer_dir: false,
                resident_count: false,
                resident_bytes: true,
                positive_reasoning_budget: false,
            },
            "--max-resident-adapter-bytes",
        ),
        (
            StandaloneOptionPresence {
                preload_vision: false,
                tokenizer_dir: false,
                resident_count: false,
                resident_bytes: false,
                positive_reasoning_budget: true,
            },
            "--reasoning-budget",
        ),
    ];

    for (presence, flag) in cases {
        let selected = select(
            CheckpointEvidence {
                directory: PathBuf::new(),
                format: ModelFormat::Safetensors,
                model_type: Some("gemma4".to_owned()),
            },
            ServingEntry::Standalone,
        )
        .expect("Gemma safetensors selects its standalone provider");
        assert_eq!(
            selected.validate_standalone_options(ServingEntry::Standalone, &presence),
            Err(format!(
                "unsupported_feature: {flag} is not supported for Gemma 4 checkpoints"
            )),
            "option {flag}"
        );
    }

    let selected = select(
        CheckpointEvidence {
            directory: PathBuf::new(),
            format: ModelFormat::Safetensors,
            model_type: Some("gemma4".to_owned()),
        },
        ServingEntry::Standalone,
    )
    .expect("Gemma safetensors selects its standalone provider");
    assert_eq!(
        selected.validate_standalone_options(
            ServingEntry::Standalone,
            &StandaloneOptionPresence {
                preload_vision: true,
                tokenizer_dir: true,
                resident_count: true,
                resident_bytes: true,
                positive_reasoning_budget: true,
            }
        ),
        Err(
            "unsupported_feature: --preload-vision is not supported for Gemma 4 checkpoints"
                .to_owned()
        )
    );
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn qwen_standalone_accepts_all_present_options() {
    let selected = select(
        CheckpointEvidence {
            directory: PathBuf::new(),
            format: ModelFormat::Safetensors,
            model_type: Some("qwen3_5".to_owned()),
        },
        ServingEntry::Standalone,
    )
    .expect("Qwen safetensors selects its standalone provider");
    assert_eq!(
        selected.validate_standalone_options(
            ServingEntry::Standalone,
            &StandaloneOptionPresence {
                preload_vision: true,
                tokenizer_dir: true,
                resident_count: true,
                resident_bytes: true,
                positive_reasoning_budget: true,
            }
        ),
        Ok(())
    );
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn unknown_gemma_standalone_skips_option_refusals() {
    let selected = select(
        CheckpointEvidence {
            directory: PathBuf::new(),
            format: ModelFormat::Unknown,
            model_type: Some("gemma4".to_owned()),
        },
        ServingEntry::Standalone,
    )
    .expect("Unknown format defers standalone failure to its legacy loader path");
    assert_eq!(selected.legacy_route(), None);
    assert_eq!(
        selected.validate_standalone_options(
            ServingEntry::Standalone,
            &StandaloneOptionPresence {
                preload_vision: true,
                tokenizer_dir: true,
                resident_count: true,
                resident_bytes: true,
                positive_reasoning_budget: true,
            }
        ),
        Ok(())
    );
}

fn empty_evidence() -> CheckpointEvidence {
    CheckpointEvidence {
        directory: PathBuf::new(),
        format: ModelFormat::Safetensors,
        model_type: None,
    }
}

#[test]
fn pooled_selection_uses_one_claim_and_the_legacy_provider_for_zero_claims() {
    let evidence = empty_evidence();
    let claimant = PooledTestProvider::new("alpha", true);
    let nonclaimant = PooledTestProvider::new("other", false);
    let fallback = PooledTestProvider::new("legacy", false);
    let one_provider: [&dyn RegisteredProvider; 1] = [&claimant];
    let no_claim_providers: [&dyn RegisteredProvider; 1] = [&nonclaimant];

    let selected = select_pooled_with_registry(&evidence, &one_provider, &fallback)
        .expect("one pooled claim selects its provider");
    assert_eq!(selected.name(), "alpha");

    let selected = select_pooled_with_registry(&evidence, &no_claim_providers, &fallback)
        .expect("zero pooled claims use the legacy provider");
    assert_eq!(selected.name(), "legacy");
}

#[test]
fn pooled_selection_refuses_two_claims_in_sorted_order() {
    let evidence = empty_evidence();
    let zeta = PooledTestProvider::new("zeta", true);
    let alpha = PooledTestProvider::new("alpha", true);
    let fallback = PooledTestProvider::new("legacy", false);
    let providers: [&dyn RegisteredProvider; 2] = [&zeta, &alpha];

    let error = select_pooled_with_registry(&evidence, &providers, &fallback)
        .err()
        .expect("two pooled claims are ambiguous");
    assert_eq!(
        error,
        EmbeddingSelectionError::Ambiguous {
            providers: vec!["alpha", "zeta"]
        }
    );
}

#[test]
fn text_selection_uses_one_claim_and_refuses_two_in_sorted_order() {
    let evidence = empty_evidence();
    let alpha = TextTestProvider::new("alpha", true);
    let zeta = TextTestProvider::new("zeta", true);
    let other = TextTestProvider::new("other", false);
    let fallback = TextTestProvider::new("legacy", false);

    let one_claim: [&dyn RegisteredProvider; 2] = [&other, &alpha];
    let selected = select_text_with_registry(&evidence, &one_claim, &fallback)
        .expect("one text claim selects its provider");
    assert_eq!(selected.name(), "alpha");

    let two_claims: [&dyn RegisteredProvider; 2] = [&zeta, &alpha];
    let error = select_text_with_registry(&evidence, &two_claims, &fallback)
        .err()
        .expect("two text claims are ambiguous");
    assert_eq!(
        error,
        EmbeddingSelectionError::Ambiguous {
            providers: vec!["alpha", "zeta"]
        }
    );
}

#[cfg(feature = "test-utils")]
#[test]
fn text_selection_uses_the_bert_loader_for_unknown_and_missing_model_types() {
    use crate::bert_pooling_fixture::write_checkpoint;
    use serde_json::Value;
    use std::fs;

    let temp = tempfile::tempdir().expect("temporary directory must be created");
    for (name, model_type) in [("unknown-type", Some("other")), ("missing-type", None)] {
        let path = temp.path().join(name);
        write_checkpoint(&path).expect("deterministic checkpoint must be written");
        let config_path = path.join("config.json");
        let mut config: Value = serde_json::from_slice(
            &fs::read(&config_path).expect("fixture config must be readable"),
        )
        .expect("fixture config must parse");
        match model_type {
            Some(model_type) => config["model_type"] = Value::String(model_type.to_owned()),
            None => {
                config
                    .as_object_mut()
                    .expect("fixture config must be an object")
                    .remove("model_type");
            }
        }
        fs::write(
            &config_path,
            serde_json::to_vec(&config).expect("fixture config must serialize"),
        )
        .expect("fixture config must be updated");

        let service = plan_text_embeddings(&path, None)
            .expect("zero claims must use the legacy BERT loader")
            .load()
            .expect("legacy BERT loader must accept this directory");
        assert_eq!(service.representation().loader_format, "bert-safetensors");
    }
}

#[test]
fn text_registration_does_not_change_qwen_or_gemma_generation_selection() {
    assert!(providers::BERT_PROVIDER.generation().is_none());
    assert!(providers::BERT_PROVIDER.pooled_embeddings().is_none());

    for (model_type, expected) in [("qwen3_5", "qwen"), ("gemma4", "gemma")] {
        let selected = select(
            CheckpointEvidence {
                directory: PathBuf::from("checkpoint"),
                format: ModelFormat::Safetensors,
                model_type: Some(model_type.to_owned()),
            },
            ServingEntry::Lattice,
        )
        .expect("existing generation provider must remain selected");
        assert_eq!(selected.provider.name(), expected);
    }
}

#[test]
fn implicit_pooled_plan_preserves_qwen_directory_and_gemma_message() {
    let qwen_directory = PathBuf::from("/models/qwen");
    let qwen = SelectedProvider {
        provider: &providers::QWEN_PROVIDER,
        evidence: CheckpointEvidence {
            directory: qwen_directory.clone(),
            format: ModelFormat::Safetensors,
            model_type: Some("qwen3_5".to_owned()),
        },
        entry: ServingEntry::Lattice,
        lattice_plan: None,
    };
    assert_eq!(
        qwen.implicit_pooled_embeddings("qwen display"),
        ImplicitPooledEmbeddingPlan {
            directory_to_try: Some(qwen_directory),
            disabled_line: None,
        }
    );

    let gemma = SelectedProvider {
        provider: &providers::GEMMA_PROVIDER,
        evidence: CheckpointEvidence {
            directory: PathBuf::from("/models/gemma"),
            format: ModelFormat::Safetensors,
            model_type: Some("gemma4".to_owned()),
        },
        entry: ServingEntry::Lattice,
        lattice_plan: None,
    };
    assert_eq!(
        gemma.implicit_pooled_embeddings("gemma display"),
        ImplicitPooledEmbeddingPlan {
            directory_to_try: None,
            disabled_line: Some(
                "Embeddings disabled (gemma display): not available for Gemma 4 checkpoints."
                    .to_owned()
            ),
        }
    );
}

#[test]
fn lattice_plan_opens_without_a_legacy_route_projection() {
    let opened = Arc::new(AtomicBool::new(false));
    let provider = PlanOnlyProvider {
        opened: Arc::clone(&opened),
    };
    let registered: [&dyn RegisteredProvider; 1] = [&provider];
    let selected = select_with_registry(
        CheckpointEvidence {
            directory: PathBuf::from("checkpoint"),
            format: ModelFormat::Safetensors,
            model_type: Some("fixture".to_owned()),
        },
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &registered,
            legacy_generation: &providers::QWEN_PROVIDER,
        },
    )
    .expect("the plan-only provider admits its checkpoint");

    assert_eq!(selected.legacy_route(), None);
    assert_eq!(
        selected.lattice_selection_marker().as_deref(),
        Some("[route] selected family=fixture backend=cpu format=safetensors")
    );
    let opened_route = selected.open_lattice(
        LatticeOpenOptions {
            tokenizer_dir: None,
            max_pending: 1,
            max_resident_adapters: 0,
            max_resident_adapter_bytes: 0,
            preload_vision: false,
            context_window: 0,
        },
        Arc::new(Semaphore::new(1)),
    );
    assert!(opened_route.is_ok());
    assert!(opened.load(Ordering::SeqCst));
}

struct PlanOnlyProvider {
    opened: Arc<AtomicBool>,
}

struct PlanOnlyLatticeRoutePlan {
    opened: Arc<AtomicBool>,
    directory: PathBuf,
}

impl LatticeRoutePlan for PlanOnlyLatticeRoutePlan {
    fn selection_marker(&self) -> String {
        "[route] selected family=fixture backend=cpu format=safetensors".to_owned()
    }

    fn implicit_embeddings(&self, _model: &str) -> ImplicitPooledEmbeddingPlan {
        ImplicitPooledEmbeddingPlan {
            directory_to_try: Some(self.directory.clone()),
            disabled_line: None,
        }
    }

    fn open(
        self: Box<Self>,
        options: LatticeOpenOptions,
        generation_admission: Arc<Semaphore>,
    ) -> Result<crate::serving_host::ServingHandle, String> {
        let _ = options;
        self.opened.store(true, Ordering::SeqCst);
        let model = crate::model::qwen35::test_support::tiny_zero_model_with_context(64);
        let cpu = providers::qwen::cpu::from_model(model);
        Ok(crate::serving_host::ServingHandle::shared_cpu(
            cpu,
            generation_admission,
        ))
    }
}

impl ServingProvider for PlanOnlyProvider {
    fn name(&self) -> &'static str {
        "fixture"
    }

    fn claims(&self, evidence: &CheckpointEvidence) -> bool {
        evidence.model_type.as_deref() == Some("fixture")
    }

    fn admit_lattice(
        &self,
        evidence: &CheckpointEvidence,
    ) -> Result<Box<dyn LatticeRoutePlan>, RouteRefusal> {
        Ok(Box::new(PlanOnlyLatticeRoutePlan {
            opened: Arc::clone(&self.opened),
            directory: evidence.directory.clone(),
        }))
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn validate_standalone_options(
        &self,
        _entry: ServingEntry,
        _route: Option<ServedRoute>,
        _presence: &StandaloneOptionPresence,
    ) -> Result<(), String> {
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_startup_description(
        &self,
        _format: ModelFormat,
        _route: Option<ServedRoute>,
    ) -> StandaloneStartupDescription {
        StandaloneStartupDescription {
            route_marker: None,
            loading_label: "unknown",
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn prepare_standalone_load(
        &self,
        _evidence: CheckpointEvidence,
        _route: Option<ServedRoute>,
        _options: StandaloneLoadOptions,
    ) -> Result<StandaloneLoadPlan, String> {
        Err("fixture provider does not load standalone models".to_owned())
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_factory(
        &self,
        _evidence: CheckpointEvidence,
        _options: StandaloneLoadOptions,
        _qwen_loader: StandaloneQwenLoader,
    ) -> Result<crate::serving_factory::ServingFactory, String> {
        Err("fixture provider does not build standalone factories".to_owned())
    }
}

impl RegisteredProvider for PlanOnlyProvider {
    fn generation(&self) -> Option<&dyn ServingProvider> {
        Some(self)
    }

    fn pooled_embeddings(&self) -> Option<&dyn PooledEmbeddingProvider> {
        None
    }

    fn text_embeddings(&self) -> Option<&dyn TextEmbeddingProvider> {
        None
    }
}

struct PooledTestProvider {
    name: &'static str,
    claim: bool,
}

impl PooledTestProvider {
    fn new(name: &'static str, claim: bool) -> Self {
        Self { name, claim }
    }
}

impl RegisteredProvider for PooledTestProvider {
    fn generation(&self) -> Option<&dyn ServingProvider> {
        None
    }

    fn pooled_embeddings(&self) -> Option<&dyn PooledEmbeddingProvider> {
        Some(self)
    }

    fn text_embeddings(&self) -> Option<&dyn TextEmbeddingProvider> {
        None
    }
}

impl PooledEmbeddingProvider for PooledTestProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn claims(&self, _evidence: &CheckpointEvidence) -> bool {
        self.claim
    }

    fn plan(&self, _directory: PathBuf) -> Box<dyn PooledEmbeddingLoadPlan> {
        Box::new(TestPooledLoadPlan)
    }
}

struct TestPooledLoadPlan;

impl PooledEmbeddingLoadPlan for TestPooledLoadPlan {
    fn load(
        self: Box<Self>,
    ) -> Result<crate::serving_embeddings::PooledEmbeddingHandle, crate::error::InferenceError>
    {
        Err(crate::error::InferenceError::Inference(
            "test pooled service is not loadable".to_owned(),
        ))
    }
}

struct TestProvider {
    name: &'static str,
    claim: bool,
    route: Result<Option<ServedRoute>, RouteRefusal>,
    route_calls: AtomicUsize,
    admission_calls: AtomicUsize,
}

impl TestProvider {
    fn new(
        name: &'static str,
        claim: bool,
        route: Result<Option<ServedRoute>, RouteRefusal>,
    ) -> Self {
        Self {
            name,
            claim,
            route,
            route_calls: AtomicUsize::new(0),
            admission_calls: AtomicUsize::new(0),
        }
    }
}

struct TestLatticeRoutePlan {
    route: Option<ServedRoute>,
}

impl LatticeRoutePlan for TestLatticeRoutePlan {
    fn selection_marker(&self) -> String {
        self.route
            .map(|route| route.selection_marker(ModelFormat::Safetensors))
            .unwrap_or_default()
    }

    fn implicit_embeddings(&self, _model: &str) -> ImplicitPooledEmbeddingPlan {
        ImplicitPooledEmbeddingPlan {
            directory_to_try: None,
            disabled_line: None,
        }
    }

    fn open(
        self: Box<Self>,
        options: LatticeOpenOptions,
        generation_admission: Arc<Semaphore>,
    ) -> Result<crate::serving_host::ServingHandle, String> {
        let _ = (self, options, generation_admission);
        Err("test provider does not open a lattice route".to_owned())
    }
}

impl ServingProvider for TestProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn claims(&self, _evidence: &CheckpointEvidence) -> bool {
        self.claim
    }

    fn admit_lattice(
        &self,
        _evidence: &CheckpointEvidence,
    ) -> Result<Box<dyn LatticeRoutePlan>, RouteRefusal> {
        self.admission_calls.fetch_add(1, Ordering::SeqCst);
        self.route
            .map(|route| Box::new(TestLatticeRoutePlan { route }) as Box<dyn LatticeRoutePlan>)
    }

    fn legacy_route(
        &self,
        _evidence: &CheckpointEvidence,
        _entry: ServingEntry,
    ) -> Result<Option<ServedRoute>, RouteRefusal> {
        self.route_calls.fetch_add(1, Ordering::SeqCst);
        self.route
    }

    fn implicit_pooled_embeddings(
        &self,
        evidence: &CheckpointEvidence,
        _model: &str,
    ) -> ImplicitPooledEmbeddingPlan {
        ImplicitPooledEmbeddingPlan {
            directory_to_try: Some(evidence.directory.clone()),
            disabled_line: None,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn validate_standalone_options(
        &self,
        _entry: ServingEntry,
        _route: Option<ServedRoute>,
        _presence: &StandaloneOptionPresence,
    ) -> Result<(), String> {
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_startup_description(
        &self,
        format: ModelFormat,
        route: Option<ServedRoute>,
    ) -> StandaloneStartupDescription {
        StandaloneStartupDescription {
            route_marker: route.map(|route| route.selection_marker(format)),
            loading_label: match format {
                ModelFormat::Q4 => "q4",
                ModelFormat::Safetensors => "bf16",
                _ => "unknown",
            },
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn prepare_standalone_load(
        &self,
        _evidence: CheckpointEvidence,
        _route: Option<ServedRoute>,
        _options: StandaloneLoadOptions,
    ) -> Result<StandaloneLoadPlan, String> {
        Err("test provider does not build a standalone load plan".to_owned())
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_factory(
        &self,
        _evidence: CheckpointEvidence,
        _options: StandaloneLoadOptions,
        _qwen_loader: StandaloneQwenLoader,
    ) -> Result<ServingFactory, String> {
        Err("test provider does not build a factory".to_owned())
    }
}

impl RegisteredProvider for TestProvider {
    fn generation(&self) -> Option<&dyn ServingProvider> {
        Some(self)
    }

    fn pooled_embeddings(&self) -> Option<&dyn PooledEmbeddingProvider> {
        Some(self)
    }

    fn text_embeddings(&self) -> Option<&dyn TextEmbeddingProvider> {
        None
    }
}

impl PooledEmbeddingProvider for TestProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn claims(&self, _evidence: &CheckpointEvidence) -> bool {
        self.claim
    }

    fn plan(&self, _directory: PathBuf) -> Box<dyn PooledEmbeddingLoadPlan> {
        Box::new(TestPooledLoadPlan)
    }
}

struct TextTestProvider {
    name: &'static str,
    claim: bool,
}

impl TextTestProvider {
    fn new(name: &'static str, claim: bool) -> Self {
        Self { name, claim }
    }
}

impl RegisteredProvider for TextTestProvider {
    fn generation(&self) -> Option<&dyn ServingProvider> {
        None
    }

    fn pooled_embeddings(&self) -> Option<&dyn PooledEmbeddingProvider> {
        None
    }

    fn text_embeddings(&self) -> Option<&dyn TextEmbeddingProvider> {
        Some(self)
    }
}

impl TextEmbeddingProvider for TextTestProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn claims(&self, _evidence: &CheckpointEvidence) -> bool {
        self.claim
    }

    fn plan(
        &self,
        _directory: PathBuf,
        _pooling_override: Option<String>,
    ) -> Box<dyn TextEmbeddingLoadPlan> {
        Box::new(TestTextLoadPlan)
    }
}

struct TestTextLoadPlan;

impl TextEmbeddingLoadPlan for TestTextLoadPlan {
    fn load(self: Box<Self>) -> Result<crate::serving_embeddings::TextEmbeddingHandle, String> {
        Err("test text service is not loadable".to_owned())
    }
}

mod lattice_route_markers;
