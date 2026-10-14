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

use crate::router_state::{RouterArtifactError, RouterPolicy, TrainedRepresentation};

// The two functions below are the only part of this module that needs a gate
// network, and `lattice-fann` arrives with either of these features. The
// types and the specification validation are compiled unconditionally, so a
// build without a gate backend still refuses a bad specification rather than
// having no opinion about one.
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
use crate::router_state::{
    RouterArtifact, RouterLineage, RouterProvenance, commit_head, versions, write_artifact,
};
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

/// The policy a bootstrap carries: uniform, with nothing learned.
///
/// Every guard field is present and zero rather than absent. A bootstrap has
/// no collapse history to bound and no temperature to floor, and recording
/// that as an absence would make the artifact's policy shape depend on how it
/// was produced -- which is the one thing a hashed contract must not do,
/// because then two policies that serve identically hash differently.
pub fn bootstrap_policy() -> RouterPolicy {
    RouterPolicy {
        kind: "uniform".to_string(),
        // Refused by `validate()` on a uniform policy, and that is the point:
        // nothing reads a tau here, so recording one would be a number an
        // operator could change with no effect.
        tau: None,
        epsilon: 0.0,
        tau_floor: 0.0,
        entropy_floor: 0.0,
        max_consecutive_collapsed: 0,
        training_state: "bootstrap".to_string(),
        // No objective was optimised. An empty string here would read as a
        // recorded objective with an unprintable name.
        objective_version: None,
        // A bootstrap is deterministic: it serves the same mixture for every
        // request, and recording a scale would claim it explores. Absence, not
        // zero -- zero is a scale, and `validate()` refuses it.
        exploration_scale: None,
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
            bootstrap_policy(),
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
/// The head is committed after the artifact is on disk, in that order, so a
/// crash between the two leaves a directory with an artifact and no head --
/// which selects by counter and finds the same artifact. The reverse order
/// would leave a head naming a version that is not there, and every unpinned
/// startup would refuse.
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
pub fn init_directory(dir: &Path, spec: &BootstrapSpec) -> Result<PathBuf, RouterInitError> {
    let artifact = bootstrap_artifact(spec)?;

    // Checked after the spec and before anything is written. `versions()`
    // answers an empty list for a directory that does not exist, which is the
    // ordinary case here and not an error.
    let present = versions(dir)?;
    if !present.is_empty() {
        return Err(RouterInitError::Occupied {
            dir: dir.to_path_buf(),
            present,
        });
    }

    let path = write_artifact(dir, &artifact)?;
    commit_head(dir, &artifact)?;
    Ok(path)
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
        }
    }

    /// The bootstrap policy is what `validate()` accepts, and it is uniform
    /// with nothing learned. Written as a test rather than trusted because the
    /// constructor and the validator are two statements about the same shape.
    #[test]
    fn the_bootstrap_policy_is_a_valid_uniform_policy_with_nothing_learned() {
        let policy = bootstrap_policy();
        policy
            .validate()
            .expect("the bootstrap policy must validate");
        assert_eq!(policy.kind, "uniform");
        assert_eq!(policy.training_state, "bootstrap");
        assert_eq!(policy.tau, None);
        assert_eq!(policy.objective_version, None);
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
    }
}
