//! Directory-aware provider selection for serving entry points.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::forward::metal_qwen35::MetalQwen35State;
use crate::model_format::ModelFormat;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::metal_worker::WorkerMetadata;
use crate::serve::route::{RouteRefusal, ServedRoute};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_factory::ServingFactory;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::tokenizer::bpe::BpeTokenizer;
use std::path::{Path, PathBuf};

pub mod providers;

/// A binary's serving route policy.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServingEntry {
    /// The `lattice serve` command.
    Lattice,
    /// The standalone `lattice_serve` binary.
    Standalone,
}

/// Checkpoint facts collected at the serving startup inspection stage.
#[doc(hidden)]
#[derive(Clone)]
pub struct CheckpointEvidence {
    directory: PathBuf,
    format: ModelFormat,
    model_type: Option<String>,
}

impl CheckpointEvidence {
    /// The checkpoint format observed during inspection.
    pub fn format(&self) -> ModelFormat {
        self.format
    }
}

/// Inspect a checkpoint directory without turning probe failures into errors.
#[doc(hidden)]
pub fn inspect_checkpoint(directory: &Path) -> CheckpointEvidence {
    let format = crate::model_format::detect_format(directory);
    let model_type = crate::model::config_file::read_config_json_bounded(
        &directory.join("config.json"),
        "config.json",
    )
    .ok()
    .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
    .and_then(|root| {
        root.get("model_type")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    });

    CheckpointEvidence {
        directory: directory.to_path_buf(),
        format,
        model_type,
    }
}

/// Why serving provider selection could not produce a route.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    /// More than one registered provider positively claimed the checkpoint.
    Ambiguous {
        /// The inspected checkpoint directory.
        directory: PathBuf,
        /// The provider names, in sorted order.
        claimants: Vec<&'static str>,
    },
    /// The selected provider's legacy route projection refused this entry.
    Route {
        /// The inspected checkpoint directory.
        directory: PathBuf,
        /// The existing route refusal.
        refusal: RouteRefusal,
    },
}

impl SelectionError {
    /// Render the established route refusal or the sorted claimant list.
    pub fn message(&self) -> String {
        match self {
            Self::Ambiguous {
                directory,
                claimants,
            } => format!(
                "ambiguous serving providers for '{}': {}",
                directory.display(),
                claimants.join(", ")
            ),
            Self::Route { directory, refusal } => refusal.message(directory),
        }
    }
}

/// Presence of standalone flags that a provider may reject.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub struct StandaloneOptionPresence {
    /// Whether `--preload-vision` appeared as a separate argument.
    pub preload_vision: bool,
    /// Whether `--tokenizer-dir` appeared as a separate argument.
    pub tokenizer_dir: bool,
    /// Whether `--max-resident-adapters` appeared as a separate argument.
    pub resident_count: bool,
    /// Whether `--max-resident-adapter-bytes` appeared as a separate argument.
    pub resident_bytes: bool,
    /// Whether the parsed reasoning budget is positive.
    pub positive_reasoning_budget: bool,
}

/// Options needed to construct the standalone serving factory.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub struct StandaloneLoadOptions {
    /// Tokenizer file used by the existing Qwen worker loader.
    pub tokenizer_path: PathBuf,
    /// Whether Qwen vision weights should be loaded before worker startup.
    pub preload_vision: bool,
}

/// Binary-owned worker loader builder retained at the standalone call site.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub type StandaloneQwenLoader = Box<
    dyn FnOnce(
            PathBuf,
            PathBuf,
            ModelFormat,
        ) -> Box<
            dyn FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
                + Send
                + 'static,
        > + Send
        + 'static,
>;

/// The provider selected for one checkpoint and its existing route projection.
#[doc(hidden)]
pub struct SelectedProvider<'a> {
    #[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
    provider: &'a dyn ServingProvider,
    legacy_route: Option<ServedRoute>,
}

impl SelectedProvider<'_> {
    /// The route the entry has historically exposed, if the entry defers it.
    pub fn legacy_route(&self) -> Option<ServedRoute> {
        self.legacy_route
    }

    /// Validate standalone flag presence through the selected provider.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn validate_standalone_options(
        &self,
        entry: ServingEntry,
        presence: &StandaloneOptionPresence,
    ) -> Result<(), String> {
        self.provider
            .validate_standalone_options(entry, self.legacy_route, presence)
    }

    /// Build the standalone worker factory through the selected provider.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn into_standalone_factory(
        self,
        entry: ServingEntry,
        evidence: CheckpointEvidence,
        options: StandaloneLoadOptions,
        qwen_loader: StandaloneQwenLoader,
    ) -> Result<ServingFactory, String> {
        if entry != ServingEntry::Standalone {
            return Err("standalone factory requested for a non-standalone entry".to_owned());
        }
        self.provider
            .standalone_factory(evidence, options, qwen_loader)
    }
}

pub(crate) trait ServingProvider: Sync {
    fn name(&self) -> &'static str;
    fn claims(&self, evidence: &CheckpointEvidence) -> bool;
    fn legacy_route(
        &self,
        evidence: &CheckpointEvidence,
        entry: ServingEntry,
    ) -> Result<Option<ServedRoute>, RouteRefusal>;

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn validate_standalone_options(
        &self,
        entry: ServingEntry,
        route: Option<ServedRoute>,
        presence: &StandaloneOptionPresence,
    ) -> Result<(), String>;

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn standalone_factory(
        &self,
        evidence: CheckpointEvidence,
        options: StandaloneLoadOptions,
        qwen_loader: StandaloneQwenLoader,
    ) -> Result<ServingFactory, String>;
}

pub(crate) struct ProviderRegistry<'a> {
    providers: &'a [&'a dyn ServingProvider],
    legacy: &'a dyn ServingProvider,
}

static LEGACY_PROVIDER: &dyn ServingProvider = &providers::QWEN_PROVIDER;

/// Select a provider and project the route for one serving entry.
#[doc(hidden)]
pub fn select(
    evidence: CheckpointEvidence,
    entry: ServingEntry,
) -> Result<SelectedProvider<'static>, SelectionError> {
    let registry = ProviderRegistry {
        providers: &providers::PROVIDERS,
        legacy: LEGACY_PROVIDER,
    };
    select_with_registry(evidence, entry, &registry)
}

pub(crate) fn select_with_registry<'a>(
    evidence: CheckpointEvidence,
    entry: ServingEntry,
    registry: &ProviderRegistry<'a>,
) -> Result<SelectedProvider<'a>, SelectionError> {
    let claimants = registry
        .providers
        .iter()
        .copied()
        .filter(|provider| provider.claims(&evidence))
        .collect::<Vec<_>>();

    if claimants.len() > 1 {
        let mut names = claimants
            .iter()
            .map(|provider| provider.name())
            .collect::<Vec<_>>();
        names.sort_unstable();
        return Err(SelectionError::Ambiguous {
            directory: evidence.directory.clone(),
            claimants: names,
        });
    }

    let provider = if claimants.len() == 1 {
        claimants[0]
    } else {
        registry.legacy
    };
    let legacy_route =
        provider
            .legacy_route(&evidence, entry)
            .map_err(|refusal| SelectionError::Route {
                directory: evidence.directory.clone(),
                refusal,
            })?;

    Ok(SelectedProvider {
        provider,
        legacy_route,
    })
}

#[cfg(test)]
mod tests {
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
                    fs::write(directory.join("model.safetensors"), b"sentinel")
                        .expect("write safetensors sentinel");
                }
                Self::SafetensorsIndex => {
                    fs::write(directory.join("model.safetensors.index.json"), b"{}")
                        .expect("write safetensors index sentinel");
                }
                Self::BothSafetensorsSentinels => {
                    fs::write(directory.join("model.safetensors"), b"sentinel")
                        .expect("write safetensors sentinel");
                    fs::write(directory.join("model.safetensors.index.json"), b"{}")
                        .expect("write safetensors index sentinel");
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
                Self::QwenMoe => fs::write(&path, br#"{"model_type":"qwen3_5_moe"}"#)
                    .expect("write qwen moe config"),
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
            fs::write(temp.path().join("model.safetensors"), b"sentinel")
                .expect("write safetensors sentinel");
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
                    let actual =
                        selected
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
        let claimant =
            TestProvider::new("claimant", true, Err(RouteRefusal::GemmaMetalUnsupported));
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
        fn standalone_factory(
            &self,
            _evidence: CheckpointEvidence,
            _options: StandaloneLoadOptions,
            _qwen_loader: StandaloneQwenLoader,
        ) -> Result<ServingFactory, String> {
            Err("test provider does not build a factory".to_owned())
        }
    }
}
