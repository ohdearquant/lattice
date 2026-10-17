//! Provisioning a router state directory (ADR-096 decision 2).
//!
//! `lattice router init` mints the first artifact a directory holds. Its
//! output is a **deterministic untrained bootstrap**: a gate whose logits are
//! zero, carrying the named `Uniform` policy with nothing learned. That is
//! legitimate to serve, because it is the baseline an operator asked for, and
//! it is not evidence of learning. Serving it and serving a learned gate are
//! different claims, so the artifact records which one it is and
//! `GET /v1/lora` reports it.
//!
//! The width is not a flag. ADR-096 decision 2 requires it to be measured
//! through the same loader contract the serving path uses, so the caller
//! passes the dimension count its own embedding model produced rather than a
//! number someone typed: a mistyped width produces an artifact that refuses at
//! every startup, and the refusal names the gate rather than the typo.
//!
//! The three refusals here exist because each permissive form fails toward
//! looking successful. An occupied directory is not initialised over. A
//! duplicate or empty adapter name is refused rather than written, because
//! the serving path matches columns to adapters BY NAME and a duplicate makes
//! one column unreachable while every check still passes. And initialising
//! does not enable learning: that is a separate explicit action, and nothing
//! in this module performs it.

use std::path::PathBuf;

use crate::router_state::{
    DEFAULT_ENTROPY_FLOOR, DEFAULT_MAX_CONSECUTIVE_COLLAPSED, DEFAULT_TAU_FLOOR,
    MIN_EXPLORATION_SCALE, RouterArtifactError, RouterPolicy, TrainedRepresentation,
};

// The two functions below are the only part of this module that needs a gate
// network, and `lattice-fann` arrives with either of these features. The
// types and the specification validation are compiled unconditionally, so a
// build without a gate backend still refuses a bad specification rather than
// having no opinion about one.
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
use crate::router_state::{RouterArtifact, RouterLineage, RouterProvenance, publish_bootstrap};
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
use std::path::Path;

/// The version counter a fresh directory's first artifact carries.
pub const BOOTSTRAP_VERSION: u64 = 1;

/// What goes wrong provisioning a state directory.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RouterInitError {
    /// The directory already holds artifacts. Initialising over them would
    /// discard a served history, and a mistyped destination is exactly the
    /// case where the permissive form looks like it worked.
    #[error(
        "router state directory {dir} already holds version(s) {present:?}; \
         `router init` does not overwrite, because the directory it would \
         overwrite is the one a running server is pinned into"
    )]
    Occupied {
        /// The directory asked for.
        dir: PathBuf,
        /// What is already there.
        present: Vec<u64>,
    },

    /// The ordered adapter specification is not usable as a column order.
    #[error("adapter specification is not usable: {problem}")]
    InvalidSpec {
        /// What is wrong with it.
        problem: String,
    },

    /// The zero gate could not be built or serialized.
    #[error("building the bootstrap gate failed: {message}")]
    Gate {
        /// The builder's own message.
        message: String,
    },

    /// Writing the artifact or committing the head failed.
    #[error(transparent)]
    Artifact(#[from] RouterArtifactError),
}

/// The policy a bootstrap carries: nothing learned, and either shape.
///
/// With `tau` absent this is the original uniform bootstrap, field for field.
/// Every guard field is present and zero rather than absent, because a uniform
/// policy has no collapse history to bound and no temperature to floor, and
/// recording that as an absence would make the artifact's policy shape depend
/// on how it was produced -- which is the one thing a hashed contract must not
/// do, because then two policies that serve identically hash differently.
///
/// With `tau` present the kind is `softmax` and the three guard fields carry
/// the router's own defaults rather than zero. That difference is the whole
/// point: on a softmax policy those fields ARE read, and zero in them is
/// ADR-091 decision 3's guard recorded as disabled. The uniform bootstrap's
/// zeros are correct because nothing reads them; copying them here would be
/// the same bytes meaning the opposite thing.
///
/// `training_state` stays `bootstrap` either way. A softmax bootstrap over a
/// zero gate has learned exactly as much as a uniform one -- nothing -- and
/// serves exactly the same mixture, because a softmax over equal logits IS
/// `1/k`. What changes is that the mixture is now a function of the scores, so
/// a perturbation of the scores can reach it. That is what makes the artifact
/// explorable, and it is not evidence of learning.
pub fn bootstrap_policy(tau: Option<f64>, exploration_scale: Option<f64>) -> RouterPolicy {
    let learned_shape = tau.is_some();
    RouterPolicy {
        kind: if learned_shape { "softmax" } else { "uniform" }.to_string(),
        // Refused by `validate()` on a uniform policy, and that is the point:
        // nothing reads a tau there, so recording one would be a number an
        // operator could change with no effect.
        tau,
        epsilon: 0.0,
        tau_floor: if learned_shape {
            f64::from(DEFAULT_TAU_FLOOR)
        } else {
            0.0
        },
        entropy_floor: if learned_shape {
            f64::from(DEFAULT_ENTROPY_FLOOR)
        } else {
            0.0
        },
        max_consecutive_collapsed: if learned_shape {
            DEFAULT_MAX_CONSECUTIVE_COLLAPSED as u32
        } else {
            0
        },
        training_state: "bootstrap".to_string(),
        // No objective was optimised. An empty string here would read as a
        // recorded objective with an unprintable name.
        objective_version: None,
        exploration_scale,
    }
}

/// The ordered specification a bootstrap is minted from.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapSpec {
    /// The adapter names in gate column order. This IS the schema: ADR-096
    /// decision 1 makes it immutable within a lineage.
    pub adapter_names: Vec<String>,
    /// The representation, with `input_width` measured through the serving
    /// path's own loader rather than declared.
    pub representation: TrainedRepresentation,
    /// The schema identity this lineage is named by.
    pub schema_id: String,
    /// The base model the adapters attach to.
    pub base_model: String,
    /// The softmax temperature, when the bootstrap is to be minted in the shape
    /// a learning run can use. Absent mints the original uniform bootstrap.
    pub tau: Option<f64>,
    /// The exploration scale the artifact records, when one is to be recorded.
    /// Requires `tau`: see [`BootstrapSpec::validate`].
    pub exploration_scale: Option<f64>,
}

impl BootstrapSpec {
    /// Refuse a specification that cannot be a column order.
    ///
    /// A duplicate name is the dangerous one: the serving path matches columns
    /// to adapters by name, so a duplicate leaves one column permanently
    /// unreachable while residency, width and hash checks all pass.
    pub fn validate(&self) -> Result<(), RouterInitError> {
        let bad = |problem: String| RouterInitError::InvalidSpec { problem };
        if self.adapter_names.is_empty() {
            return Err(bad("no adapter names were given".to_string()));
        }
        if let Some(blank) = self.adapter_names.iter().find(|n| n.trim().is_empty()) {
            return Err(bad(format!(
                "adapter name {blank:?} is empty, and a column matched by an \
                 empty name can never be routed to"
            )));
        }
        let mut seen = std::collections::BTreeSet::new();
        for name in &self.adapter_names {
            if !seen.insert(name.as_str()) {
                return Err(bad(format!(
                    "adapter name {name:?} appears twice; columns are matched \
                     by name, so one of them could never be selected"
                )));
            }
        }
        if self.representation.input_width == 0 {
            return Err(bad(
                "the measured input width is 0, which is what an embedding \
                 model that did not load produces"
                    .to_string(),
            ));
        }
        if self.schema_id.trim().is_empty() {
            return Err(bad("the schema id is empty".to_string()));
        }
        if self.base_model.trim().is_empty() {
            return Err(bad("the base model is empty".to_string()));
        }
        // The three checks below duplicate refusals `RouterPolicy::validate`
        // already makes, deliberately. That one runs at write time and names
        // the policy; this one runs here and names the flags the operator
        // typed, which is this module's stated rule for every other field.
        // Without it, `bootstrap_artifact` -- which does not write -- returns
        // an artifact whose policy is invalid, and the callers that never write
        // it (tests, and any future dry run) get no refusal at all.
        if let Some(tau) = self.tau {
            if !tau.is_finite() || tau <= 0.0 {
                return Err(bad(format!(
                    "tau {tau} must be finite and greater than 0.0"
                )));
            }
            if tau < f64::from(DEFAULT_TAU_FLOOR) {
                return Err(bad(format!(
                    "tau {tau} is below the floor {DEFAULT_TAU_FLOOR} this bootstrap records, \
                     so the artifact would refuse at every startup"
                )));
            }
        }
        if let Some(scale) = self.exploration_scale {
            if self.tau.is_none() {
                return Err(bad(format!(
                    "--exploration-scale requires --tau: a scale {scale} was given without a \
                     tau, so the policy would be uniform, the serving path would select every \
                     trained column, and a draw could not change a decision that is 1/k \
                     whatever the scores are"
                )));
            }
            if !scale.is_finite() || scale < MIN_EXPLORATION_SCALE {
                return Err(bad(format!(
                    "exploration scale {scale} must be finite and at least {MIN_EXPLORATION_SCALE}"
                )));
            }
        }
        Ok(())
    }
}

/// Build the untrained bootstrap artifact for `spec`, without writing it.
///
/// Deterministic: the same specification produces the same bytes and therefore
/// the same content hash, on any machine. A randomly initialised gate would
/// also be untrained, and would additionally be unreproducible -- two
/// operators provisioning the same lineage would get two artifacts that differ
/// in a way neither could explain.
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
pub fn bootstrap_artifact(spec: &BootstrapSpec) -> Result<RouterArtifact, RouterInitError> {
    use lattice_fann::{Activation, Layer, Network};

    spec.validate()?;

    // One linear layer of zeros: every column scores exactly 0 for every
    // input, so selection is whatever the uniform policy does with a tie. That
    // is the baseline being asked for, stated in the weights rather than in a
    // flag the serving path would have to honour.
    let layer = Layer::zeros(
        spec.representation.input_width as usize,
        spec.adapter_names.len(),
        Activation::Linear,
    )
    .map_err(|e| RouterInitError::Gate {
        message: e.to_string(),
    })?;
    let network = Network::new(vec![layer]).map_err(|e| RouterInitError::Gate {
        message: e.to_string(),
    })?;

    Ok(RouterArtifact {
        version: BOOTSTRAP_VERSION,
        adapter_names: spec.adapter_names.clone(),
        representation: spec.representation.clone(),
        gate_bytes: network.to_bytes(),
        provenance: RouterProvenance::recorded(
            bootstrap_policy(spec.tau, spec.exploration_scale),
            RouterLineage {
                schema_id: spec.schema_id.clone(),
                base_model: spec.base_model.clone(),
                // A bootstrap has no parent, and no admission record: it was
                // not admitted on evidence, it was asked for.
                parent: None,
                admission_record: None,
            },
        ),
    })
}

/// Provision `dir` with the bootstrap for `spec`, refusing an occupied one.
///
/// Publication is serialized and create-only. The bootstrap is constructed
/// internally, staged outside the selectable namespace, then its synced files
/// and committed HEAD are published in order. A crash before HEAD leaves no
/// selectable format-5 bootstrap; retry initialization in a fresh directory.
/// An error after HEAD replacement reports an uncertain commit and requires
/// recovery through normal startup selection before any activation.
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
pub fn init_directory(dir: &Path, spec: &BootstrapSpec) -> Result<PathBuf, RouterInitError> {
    publish_bootstrap(dir, spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn representation(input_width: u64) -> TrainedRepresentation {
        TrainedRepresentation {
            embedding_model: "qwen3.5-0.8b".into(),
            pooling: "mean_visual".into(),
            prompt_source: "last_user_message".into(),
            loader_format: "qwen35-f16-decoder".into(),
            input_width,
        }
    }

    fn spec() -> BootstrapSpec {
        BootstrapSpec {
            adapter_names: vec!["technical".into(), "casual".into()],
            representation: representation(1024),
            schema_id: "schema-tech-casual-v1".into(),
            base_model: "qwen3.5-0.8b".into(),
            tau: None,
            exploration_scale: None,
        }
    }

    /// The same lineage in the shape a learning run can use: the zero gate
    /// under a softmax policy, recorded with a scale to explore at.
    fn explorable_spec() -> BootstrapSpec {
        BootstrapSpec {
            tau: Some(1.0),
            exploration_scale: Some(1.0),
            ..spec()
        }
    }

    /// The bootstrap policy is what `validate()` accepts, and it is uniform
    /// with nothing learned. Written as a test rather than trusted because the
    /// constructor and the validator are two statements about the same shape.
    #[test]
    fn the_bootstrap_policy_is_a_valid_uniform_policy_with_nothing_learned() {
        let policy = bootstrap_policy(None, None);
        policy
            .validate()
            .expect("the bootstrap policy must validate");
        assert_eq!(policy.kind, "uniform");
        assert_eq!(policy.training_state, "bootstrap");
        assert_eq!(policy.tau, None);
        assert_eq!(policy.objective_version, None);
    }

    /// The softmax bootstrap is the same `bootstrap` training state, and its
    /// guard fields carry the router's own floors rather than the uniform
    /// bootstrap's zeros -- because on a softmax policy those fields ARE read,
    /// and a zero there is ADR-091 decision 3's guard recorded as disabled.
    #[test]
    fn the_softmax_bootstrap_records_the_router_floors_and_stays_untrained() {
        let policy = bootstrap_policy(Some(1.0), Some(0.5));
        policy
            .validate()
            .expect("the softmax bootstrap policy must validate");
        assert_eq!(policy.kind, "softmax");
        assert_eq!(policy.tau, Some(1.0));
        assert_eq!(policy.training_state, "bootstrap");
        assert_eq!(policy.tau_floor, f64::from(DEFAULT_TAU_FLOOR));
        assert_eq!(policy.entropy_floor, f64::from(DEFAULT_ENTROPY_FLOOR));
        assert_eq!(
            policy.max_consecutive_collapsed,
            DEFAULT_MAX_CONSECUTIVE_COLLAPSED as u32
        );
        assert_eq!(policy.exploration_scale, Some(0.5));

        // The uniform bootstrap is untouched: same shape, zeros in the guard
        // fields, no scale. A softmax bootstrap that changed this one too would
        // be a default, and ADR-091 decision 5 gates the default.
        let uniform = bootstrap_policy(None, None);
        assert_eq!(uniform.kind, "uniform");
        assert_eq!(uniform.tau_floor, 0.0);
        assert_eq!(uniform.entropy_floor, 0.0);
        assert_eq!(uniform.max_consecutive_collapsed, 0);
        assert_eq!(uniform.exploration_scale, None);
    }

    /// The refusals that make `--tau` / `--exploration-scale` a pair, plus the
    /// boundary cases each flag carries. Refused here, at the specification,
    /// rather than at the first startup that loads the directory.
    #[test]
    fn the_learning_shapes_are_refused_at_the_flags_that_produced_them() {
        let cases: Vec<(&str, BootstrapSpec)> = vec![
            (
                "a scale with no tau",
                BootstrapSpec {
                    exploration_scale: Some(1.0),
                    ..spec()
                },
            ),
            (
                "a tau below the floor the artifact itself records",
                BootstrapSpec {
                    tau: Some(f64::from(DEFAULT_TAU_FLOOR) / 2.0),
                    ..spec()
                },
            ),
            (
                "a non-positive tau",
                BootstrapSpec {
                    tau: Some(0.0),
                    ..spec()
                },
            ),
            (
                "a NaN tau",
                BootstrapSpec {
                    tau: Some(f64::NAN),
                    ..spec()
                },
            ),
            (
                "a scale below the router's own minimum",
                BootstrapSpec {
                    tau: Some(1.0),
                    exploration_scale: Some(MIN_EXPLORATION_SCALE / 2.0),
                    ..spec()
                },
            ),
            (
                "a NaN scale",
                BootstrapSpec {
                    tau: Some(1.0),
                    exploration_scale: Some(f64::NAN),
                    ..spec()
                },
            ),
        ];
        for (name, bad) in cases {
            assert!(
                matches!(bad.validate(), Err(RouterInitError::InvalidSpec { .. })),
                "{name} should be refused"
            );
        }
        // The control, in the same test: the shape these differ from is
        // accepted, so the refusals above are about the field and not about the
        // learning shape being rejected wholesale.
        explorable_spec()
            .validate()
            .expect("the explorable specification must be accepted");
    }

    /// Each way a specification can be unusable, with a must-match control in
    /// the same test: a validator that refused everything would otherwise read
    /// as thorough.
    #[test]
    fn spec_validation_covers_each_unusable_column_order() {
        spec()
            .validate()
            .expect("control: the ordinary specification must pass");

        let cases: Vec<(&str, BootstrapSpec)> = vec![
            (
                "no names",
                BootstrapSpec {
                    adapter_names: vec![],
                    ..spec()
                },
            ),
            (
                "an empty name",
                BootstrapSpec {
                    adapter_names: vec!["technical".into(), "  ".into()],
                    ..spec()
                },
            ),
            (
                "a duplicate name",
                BootstrapSpec {
                    adapter_names: vec!["technical".into(), "technical".into()],
                    ..spec()
                },
            ),
            (
                "a zero width",
                BootstrapSpec {
                    representation: representation(0),
                    ..spec()
                },
            ),
            (
                "an empty schema id",
                BootstrapSpec {
                    schema_id: String::new(),
                    ..spec()
                },
            ),
            (
                "an empty base model",
                BootstrapSpec {
                    base_model: " ".into(),
                    ..spec()
                },
            ),
        ];
        for (name, bad) in cases {
            assert!(
                matches!(bad.validate(), Err(RouterInitError::InvalidSpec { .. })),
                "{name} should be refused"
            );
        }
    }

    #[cfg(any(feature = "mixture", feature = "metal-gpu"))]
    mod with_a_gate_backend {
        use super::*;

        fn scratch(name: &str) -> PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "lattice-router-init-{}-{}-{}",
                name,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            dir
        }

        /// Deterministic: the same specification produces the same bytes on
        /// any machine. A randomly initialised gate would also be untrained
        /// and would additionally be unreproducible.
        #[test]
        fn the_same_specification_produces_the_same_artifact() {
            let a = bootstrap_artifact(&spec()).expect("build");
            let b = bootstrap_artifact(&spec()).expect("build again");
            assert_eq!(a.gate_bytes, b.gate_bytes);
            assert_eq!(a.content_hash(), b.content_hash());

            // The control: a DIFFERENT specification produces a different
            // artifact, so the equality above is not a constant.
            let other = bootstrap_artifact(&BootstrapSpec {
                adapter_names: vec!["technical".into(), "legal".into()],
                ..spec()
            })
            .expect("build the other");
            assert_ne!(a.content_hash(), other.content_hash());
        }

        /// The gate's logits are zero for every column, which is the baseline
        /// stated in the weights rather than in a flag the serving path would
        /// have to honour. Read back through the same deserializer the server
        /// uses, not from the builder's own value.
        #[test]
        fn the_bootstrap_gate_scores_every_column_zero() {
            let artifact = bootstrap_artifact(&spec()).expect("build");
            let mut network =
                lattice_fann::Network::from_bytes(&artifact.gate_bytes).expect("the gate parses");
            assert_eq!(network.num_inputs(), 1024);
            assert_eq!(network.num_outputs(), 2);

            for input in [vec![0.0f32; 1024], vec![1.0f32; 1024], vec![-3.5f32; 1024]] {
                let out = network.forward(&input).expect("forward");
                assert_eq!(
                    out,
                    &[0.0, 0.0],
                    "an untrained gate must score every column the same"
                );
            }
        }

        /// A bootstrap is labelled as one, and that is what keeps serving it
        /// from reading as evidence of learning.
        #[test]
        fn the_bootstrap_is_labelled_untrained_and_has_no_parent() {
            let artifact = bootstrap_artifact(&spec()).expect("build");
            let policy = artifact.provenance.policy().expect("a recorded policy");
            assert_eq!(policy.training_state, "bootstrap");
            assert_eq!(policy.kind, "uniform");
            let lineage = artifact.provenance.lineage().expect("a recorded lineage");
            assert_eq!(lineage.schema_id, "schema-tech-casual-v1");
            assert_eq!(lineage.parent, None);
            assert_eq!(
                lineage.admission_record, None,
                "a bootstrap was asked for, not admitted on evidence"
            );
        }

        /// The head is committed, so the directory selects under the new rule
        /// from its first artifact rather than falling back to the counter.
        #[cfg(unix)]
        #[test]
        fn init_writes_the_artifact_and_commits_the_head() {
            let dir = scratch("fresh");
            let path = init_directory(&dir, &spec()).expect("init");
            assert_eq!(path, crate::router_state::manifest_path(&dir, 1));

            let head = crate::router_state::read_head(&dir)
                .expect("read head")
                .expect("a head was committed");
            assert_eq!(head.version, BOOTSTRAP_VERSION);
            assert_eq!(
                head.content_sha256,
                bootstrap_artifact(&spec()).expect("build").content_hash()
            );

            let served = crate::router_state::load_selected(&dir)
                .expect("select")
                .expect("an artifact");
            assert_eq!(served.adapter_names, spec().adapter_names);

            let _ = std::fs::remove_dir_all(&dir);
        }

        /// Initialising over an occupied directory would discard a served
        /// history, and a mistyped destination is exactly the case where the
        /// permissive form looks like it worked.
        #[cfg(unix)]
        #[test]
        fn init_refuses_an_occupied_directory_and_changes_nothing() {
            let dir = scratch("occupied");
            init_directory(&dir, &spec()).expect("the first init");
            let before = std::fs::read(crate::router_state::manifest_path(&dir, 1))
                .expect("read the first manifest");

            let other = BootstrapSpec {
                adapter_names: vec!["legal".into(), "medical".into()],
                ..spec()
            };
            match init_directory(&dir, &other) {
                Err(RouterInitError::Occupied { present, .. }) => assert_eq!(present, vec![1]),
                other => panic!("expected a refusal, got {other:?}"),
            }
            let after = std::fs::read(crate::router_state::manifest_path(&dir, 1))
                .expect("read the manifest again");
            assert_eq!(before, after, "the refused init rewrote the directory");

            // The control: the same second specification succeeds in a
            // directory that is actually empty, so the refusal is about
            // occupancy and not about the specification.
            let empty = scratch("occupied-control");
            init_directory(&empty, &other).expect("a fresh directory takes it");

            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_dir_all(&empty);
        }

        /// A refused specification must not leave a half-provisioned
        /// directory: the validation runs before anything is created.
        #[cfg(unix)]
        #[test]
        fn a_refused_specification_creates_nothing() {
            let dir = scratch("refused");
            let bad = BootstrapSpec {
                adapter_names: vec!["technical".into(), "technical".into()],
                ..spec()
            };
            assert!(matches!(
                init_directory(&dir, &bad),
                Err(RouterInitError::InvalidSpec { .. })
            ));
            assert!(!dir.exists(), "a refused init created the directory");

            // The control: the good specification does create it.
            init_directory(&dir, &spec()).expect("init");
            assert!(dir.exists());

            let _ = std::fs::remove_dir_all(&dir);
        }

        /// The two bootstraps differ in their recorded policy and nowhere
        /// else: same adapters, same gate bytes, different content hash. If the
        /// hash did not cover the policy, an operator could not tell a served
        /// uniform baseline from a served explorable one by version label,
        /// which is the only thing `GET /v1/lora` reports.
        #[test]
        fn the_explorable_bootstrap_differs_from_the_uniform_one_only_in_its_policy() {
            let uniform = bootstrap_artifact(&spec()).expect("build uniform");
            let learned = bootstrap_artifact(&explorable_spec()).expect("build explorable");

            assert_eq!(uniform.gate_bytes, learned.gate_bytes);
            assert_eq!(uniform.adapter_names, learned.adapter_names);
            assert_ne!(uniform.content_hash(), learned.content_hash());

            let policy = learned
                .provenance
                .policy()
                .expect("a bootstrap records its policy");
            assert_eq!(policy.kind, "softmax");
            assert_eq!(policy.exploration_scale, Some(1.0));
        }

        /// A learnable directory can actually be minted and read back through
        /// the same path a server uses. The refusal being guarded against is an
        /// artifact that validates in memory and is rejected on load.
        #[cfg(unix)]
        #[test]
        fn an_explorable_directory_round_trips_through_the_serving_reader() {
            let dir = scratch("explorable-roundtrip");
            init_directory(&dir, &explorable_spec()).expect("init");
            let disposition = crate::router_state::resolve_startup(Some(&dir), None)
                .expect("the serving path must resolve the directory it just minted");
            let crate::router_state::StartupDisposition::Loaded(resolved) = disposition else {
                panic!("a minted directory must resolve to a loaded gate");
            };
            let policy = resolved
                .artifact
                .provenance
                .policy()
                .expect("a bootstrap records its policy");
            assert_eq!(policy.kind, "softmax");
            assert_eq!(policy.tau, Some(1.0));
            assert_eq!(policy.exploration_scale, Some(1.0));
            assert_eq!(policy.training_state, "bootstrap");
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[cfg(unix)]
        #[test]
        fn concurrent_initializers_serialize_and_only_one_commits() {
            use std::sync::{Arc, Barrier, mpsc};
            let temporary = tempfile::tempdir().unwrap();
            let dir = temporary.path().join("router");
            let barrier = Arc::new(Barrier::new(3));
            let (completed_tx, completed_rx) = mpsc::channel();
            let mut workers = Vec::new();
            for _ in 0..2 {
                let barrier = Arc::clone(&barrier);
                let dir = dir.clone();
                let completed_tx = completed_tx.clone();
                workers.push(std::thread::spawn(move || {
                    barrier.wait();
                    let outcome = init_directory(&dir, &spec());
                    completed_tx.send(outcome).unwrap();
                }));
            }
            drop(completed_tx);
            barrier.wait();
            let outcomes: Vec<_> = (0..2)
                .map(|_| {
                    completed_rx
                        .recv_timeout(std::time::Duration::from_secs(30))
                        .expect("concurrent initialization must finish before joining workers")
                })
                .collect();
            for worker in workers {
                worker.join().unwrap();
            }
            assert_eq!(outcomes.iter().filter(|result| result.is_ok()).count(), 1);
            assert_eq!(
                outcomes
                    .iter()
                    .filter(|result| matches!(result, Err(RouterInitError::Occupied { .. })))
                    .count(),
                1
            );
            let selected = crate::router_state::load_selected(&dir).unwrap().unwrap();
            assert_eq!(selected, bootstrap_artifact(&spec()).unwrap());
            assert!(dir.join(".router.lock").is_file());
            assert!(!dir.join(".committed").exists());
        }

        #[cfg(unix)]
        #[test]
        fn initialization_refuses_head_receipt_staging_and_orphan_remnants() {
            for name in [
                "HEAD.json",
                ".committed",
                ".staging",
                "router-1.gate",
                "unrelated-data",
            ] {
                let temporary = tempfile::tempdir().unwrap();
                let dir = temporary.path();
                let path = dir.join(name);
                std::fs::write(&path, b"retained evidence").unwrap();
                assert!(matches!(
                    init_directory(dir, &spec()),
                    Err(RouterInitError::Artifact(
                        RouterArtifactError::AlreadyExists { .. }
                    ))
                ));
                assert_eq!(std::fs::read(&path).unwrap(), b"retained evidence");
                assert!(!crate::router_state::manifest_path(dir, 1).exists());
            }
        }

        #[cfg(unix)]
        #[test]
        #[allow(deprecated)]
        fn a_modified_bootstrap_value_cannot_bypass_the_constructor_authority() {
            let temporary = tempfile::tempdir().unwrap();
            let dir = temporary.path().join("router");
            let mut changed = bootstrap_artifact(&spec()).unwrap();
            changed.gate_bytes[0] ^= 0xff;
            assert!(matches!(
                crate::router_state::write_artifact(&dir, &changed),
                Err(RouterArtifactError::PublicationAuthorityRequired { .. })
            ));
            assert!(matches!(
                crate::router_state::commit_head(&dir, &changed),
                Err(RouterArtifactError::PublicationAuthorityRequired { .. })
            ));
            assert!(!dir.exists());
            init_directory(&dir, &spec()).unwrap();
            assert_eq!(
                crate::router_state::load_selected(&dir).unwrap().unwrap(),
                bootstrap_artifact(&spec()).unwrap()
            );
        }
    }
}
