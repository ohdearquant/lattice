use super::*;
use crate::model_format::ModelFamily;
use crate::serve::route::{select_route, select_standalone_route};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    let family_providers: [(&dyn ServingProvider, ModelFamily, &str); 2] = [
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
                        legacy: &providers::QWEN_PROVIDER,
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
fn ambiguity_is_sorted_duplicate_sensitive_and_order_independent() {
    let alpha = TestProvider::new("alpha", true, Ok(Some(ServedRoute::QWEN35_CPU)));
    let zeta = TestProvider::new("zeta", true, Ok(Some(ServedRoute::GEMMA4_CPU)));
    let forward = [&zeta as &dyn ServingProvider, &alpha];
    let reverse = [&alpha as &dyn ServingProvider, &zeta];
    let legacy = TestProvider::new("legacy", false, Ok(None));

    let forward_error = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &forward,
            legacy: &legacy,
        },
    )
    .err()
    .expect("two providers must be ambiguous");
    let reverse_error = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &reverse,
            legacy: &legacy,
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

    let duplicated = [&alpha as &dyn ServingProvider, &alpha];
    let duplicate_error = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &duplicated,
            legacy: &legacy,
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
    let providers = [&claimant as &dyn ServingProvider, &legacy];
    let error = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &providers,
            legacy: &legacy,
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
    assert_eq!(claimant.route_calls.load(Ordering::SeqCst), 1);
    assert_eq!(legacy.route_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn legacy_provider_route_is_reached_only_when_no_provider_claims() {
    let nonclaimant = TestProvider::new("nonclaimant", false, Ok(None));
    let legacy = TestProvider::new("legacy", false, Ok(Some(ServedRoute::QWEN35_CPU)));
    let providers = [&nonclaimant as &dyn ServingProvider, &legacy];
    let selected = select_with_registry(
        empty_evidence(),
        ServingEntry::Lattice,
        &ProviderRegistry {
            providers: &providers,
            legacy: &legacy,
        },
    )
    .expect("zero claims binds the registered legacy provider");

    assert_eq!(selected.provider.name(), "legacy");
    assert_eq!(nonclaimant.route_calls.load(Ordering::SeqCst), 0);
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

struct TestProvider {
    name: &'static str,
    claim: bool,
    route: Result<Option<ServedRoute>, RouteRefusal>,
    route_calls: AtomicUsize,
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
        }
    }
}

impl ServingProvider for TestProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn claims(&self, _evidence: &CheckpointEvidence) -> bool {
        self.claim
    }

    fn legacy_route(
        &self,
        _evidence: &CheckpointEvidence,
        _entry: ServingEntry,
    ) -> Result<Option<ServedRoute>, RouteRefusal> {
        self.route_calls.fetch_add(1, Ordering::SeqCst);
        self.route
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
        super::standalone_startup_description(format, route)
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
