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
