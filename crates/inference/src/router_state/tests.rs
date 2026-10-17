use super::*;

fn raw_fixture_write(
    dir: &Path,
    artifact: &RouterArtifact,
) -> Result<PathBuf, RouterArtifactError> {
    let path = manifest_path(dir, artifact.version);
    let json = manifest_bytes(artifact, &path)?;
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(gate_path(dir, artifact.version), &artifact.gate_bytes).unwrap();
    std::fs::write(&path, json).unwrap();
    Ok(path)
}

#[cfg(unix)]
fn commit_fixture(dir: &Path, artifact: &RouterArtifact) -> Result<PathBuf, RouterArtifactError> {
    storage::commit_fixture(dir, artifact)
}

fn representation() -> TrainedRepresentation {
    TrainedRepresentation {
        embedding_model: "gme-qwen35".into(),
        pooling: "mean_visual".into(),
        prompt_source: "last_user_message".into(),
        loader_format: "qwen35-f16-decoder".into(),
        input_width: 8,
    }
}

fn artifact() -> RouterArtifact {
    RouterArtifact {
        version: 7,
        adapter_names: vec!["legal".into(), "medical".into(), "code".into()],
        representation: representation(),
        gate_bytes: vec![1, 2, 3, 4, 5],
        provenance: RouterProvenance::Legacy,
    }
}

fn policy() -> RouterPolicy {
    RouterPolicy {
        kind: "softmax".into(),
        tau: Some(0.7),
        epsilon: 0.05,
        tau_floor: 0.1,
        entropy_floor: 0.35,
        max_consecutive_collapsed: 3,
        training_state: "learned".into(),
        objective_version: Some("whole-mixture-v1".into()),
        exploration_scale: None,
    }
}

fn lineage() -> RouterLineage {
    RouterLineage {
        schema_id: "schema-abc123".into(),
        base_model: "qwen3.5-0.8b".into(),
        parent: Some("6:0123456789abcdef".into()),
        admission_record: Some("admission-42".into()),
    }
}

fn recorded() -> RouterArtifact {
    RouterArtifact {
        provenance: RouterProvenance::recorded(policy(), lineage()),
        ..artifact()
    }
}

/// The hash of a legacy artifact is the hash it had before recorded
/// provenance existed, byte for byte.
///
/// The constant is not a snapshot of what this code currently produces.
/// It was derived independently, by reimplementing the PRE-CHANGE hash
/// layout from the old source text in a different language and running it
/// over this fixture. If it were a snapshot, this test would pass no matter
/// what the legacy path did, and the property it exists to hold -- that
/// every artifact already on disk still verifies -- would be untested.
#[test]
fn a_legacy_artifact_hashes_exactly_as_it_did_before_provenance_existed() {
    assert_eq!(
        artifact().content_hash(),
        "efbcc19e4dec67850a74df365728be02349fa5fef60c1fb2793c6da587cb6a53"
    );
    assert_eq!(
        artifact().provenance.format(),
        ROUTER_ARTIFACT_FORMAT_LEGACY
    );
}

#[test]
fn a_recorded_artifact_round_trips_and_is_written_as_the_new_revision() {
    let dir = scratch("recorded");
    let original = recorded();
    raw_fixture_write(&dir, &original).expect("write");

    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
    assert!(
        raw.contains(&format!("\"format\": {ROUTER_ARTIFACT_FORMAT}")),
        "a recorded artifact must declare the new revision: {raw}"
    );

    assert_eq!(read_artifact(&dir, 7).expect("read"), original);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_legacy_manifest_carries_no_policy_keys_at_all() {
    // Not merely "no policy is read". An older build parses this file, and
    // two `null`s it has never seen are a different document from the one
    // it was written against.
    let dir = scratch("legacyshape");
    raw_fixture_write(&dir, &artifact()).expect("write");
    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
    assert!(
        !raw.contains("policy"),
        "legacy manifest mentions policy: {raw}"
    );
    assert!(
        !raw.contains("lineage"),
        "legacy manifest mentions lineage: {raw}"
    );
    assert!(raw.contains(&format!("\"format\": {ROUTER_ARTIFACT_FORMAT_LEGACY}")));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_declared_revision_and_the_recorded_policy_must_agree_in_both_directions() {
    // Direction 1: the new revision without a policy. Defaulting one in is
    // exactly how a legacy gate would quietly become learner-eligible.
    let dir = scratch("missingpolicy");
    raw_fixture_write(&dir, &recorded()).expect("write");
    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read");
    let stripped: serde_json::Value = {
        let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        v.as_object_mut().expect("object").remove("policy");
        v
    };
    std::fs::write(
        manifest_path(&dir, 7),
        serde_json::to_vec_pretty(&stripped).expect("serialize"),
    )
    .expect("rewrite");
    match read_artifact(&dir, 7) {
        Err(RouterArtifactError::ProvenanceMismatch { found, problem, .. }) => {
            assert_eq!(found, ROUTER_ARTIFACT_FORMAT);
            assert!(problem.contains("policy"), "{problem}");
        }
        other => panic!("expected a provenance mismatch, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);

    // Direction 2: the legacy revision carrying a policy. Dropping it would
    // serve the file while ignoring what it says about itself.
    let dir = scratch("extrapolicy");
    raw_fixture_write(&dir, &artifact()).expect("write");
    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read");
    let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    v.as_object_mut().expect("object").insert(
        "policy".into(),
        serde_json::to_value(policy()).expect("policy json"),
    );
    std::fs::write(
        manifest_path(&dir, 7),
        serde_json::to_vec_pretty(&v).expect("serialize"),
    )
    .expect("rewrite");
    match read_artifact(&dir, 7) {
        Err(RouterArtifactError::ProvenanceMismatch { found, problem, .. }) => {
            assert_eq!(found, ROUTER_ARTIFACT_FORMAT_LEGACY);
            assert!(problem.contains("policy"), "{problem}");
        }
        other => panic!("expected a provenance mismatch, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `validate()` runs on the READ path too, not only at write. An artifact
/// outlives the process that wrote it and may be hand-edited, so a policy
/// that was never servable must be refused when it is read back rather
/// than trusted because the writer that produced it is gone.
#[test]
fn an_invalid_policy_is_refused_on_read_and_not_only_at_write() {
    let dir = scratch("readbadpolicy");
    raw_fixture_write(&dir, &recorded()).expect("write");
    assert_eq!(
        read_artifact(&dir, 7).expect("control: the fixture reads"),
        recorded()
    );

    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
    let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    v["policy"]["kind"] = serde_json::Value::String("argmax".into());
    std::fs::write(
        manifest_path(&dir, 7),
        serde_json::to_vec_pretty(&v).expect("serialize"),
    )
    .expect("rewrite");

    match read_artifact(&dir, 7) {
        Err(RouterArtifactError::InvalidPolicy { problem }) => {
            assert!(problem.contains("argmax"), "{problem}");
        }
        other => panic!("expected an invalid-policy refusal on read, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every recorded field is inside the hash, checked one field at a time.
///
/// A single "the recorded artifact hashes differently from the legacy one"
/// assertion would pass with all but one field left out, which is the
/// defect this revision exists to avoid repeating: the last format put two
/// declared fields outside the hash and nothing noticed for a day.
#[test]
fn every_recorded_field_changes_the_hash() {
    let base = recorded();
    let base_hash = base.content_hash();
    assert_ne!(base_hash, artifact().content_hash());

    type Mutation = Box<dyn Fn(&mut RouterPolicy, &mut RouterLineage)>;
    let mutate: Vec<(&str, Mutation)> = vec![
        (
            "kind",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| {
                p.kind = "uniform".into();
                p.tau = None;
            }),
        ),
        (
            "tau",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.tau = Some(0.9)),
        ),
        (
            "epsilon",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.epsilon = 0.06),
        ),
        (
            "tau_floor",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.tau_floor = 0.11),
        ),
        (
            "entropy_floor",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.entropy_floor = 0.36),
        ),
        (
            "max_consecutive_collapsed",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.max_consecutive_collapsed = 4),
        ),
        (
            "training_state",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| {
                p.training_state = "bootstrap".into()
            }),
        ),
        (
            "objective_version",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| {
                p.objective_version = Some("whole-mixture-v2".into())
            }),
        ),
        (
            "objective_version absent",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.objective_version = None),
        ),
        (
            "schema_id",
            Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| {
                l.schema_id = "schema-def456".into()
            }),
        ),
        (
            "base_model",
            Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| {
                l.base_model = "qwen3.5-4b".into()
            }),
        ),
        (
            "parent",
            Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| {
                l.parent = Some("5:fedcba9876543210".into())
            }),
        ),
        (
            "parent absent",
            Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| l.parent = None),
        ),
        (
            "admission_record",
            Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| {
                l.admission_record = Some("admission-43".into())
            }),
        ),
        (
            "admission_record absent",
            Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| l.admission_record = None),
        ),
        (
            "exploration_scale",
            Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.exploration_scale = Some(0.5)),
        ),
    ];

    for (name, apply) in mutate {
        let mut p = policy();
        let mut l = lineage();
        apply(&mut p, &mut l);
        let changed = RouterArtifact {
            provenance: RouterProvenance::recorded(p, l),
            ..artifact()
        };
        assert_ne!(
            changed.content_hash(),
            base_hash,
            "changing {name} left the content hash unchanged, so that field is outside it"
        );
    }
}

/// A bare counter pin, which is ADR-095's spelling and the one these
/// tests were written against.
fn pin(version: u64) -> RouterPin {
    RouterPin::parse(&format!("{version}")).expect("a bare counter parses")
}

/// An absent optional string and an empty one are different claims, and the
/// length prefixes alone cannot tell them apart.
#[test]
fn an_absent_optional_does_not_hash_as_an_empty_one() {
    let absent = RouterArtifact {
        provenance: RouterProvenance::recorded(
            policy(),
            RouterLineage {
                parent: None,
                ..lineage()
            },
        ),
        ..artifact()
    };
    let empty = RouterArtifact {
        provenance: RouterProvenance::recorded(
            policy(),
            RouterLineage {
                parent: Some(String::new()),
                ..lineage()
            },
        ),
        ..artifact()
    };
    assert_ne!(absent.content_hash(), empty.content_hash());
}

#[test]
fn an_inconsistent_policy_is_refused_at_write_and_leaves_no_payload() {
    // The ordering matters as much as the refusal: `versions()` counts a
    // version present when its MANIFEST exists, so a payload written before
    // a refusal is invisible to every reader and silently overwritten by the
    // next write of the same counter.
    let dir = scratch("badpolicy");
    let bad = RouterArtifact {
        provenance: RouterProvenance::recorded(
            RouterPolicy {
                tau: None,
                ..policy()
            },
            lineage(),
        ),
        ..artifact()
    };
    match raw_fixture_write(&dir, &bad) {
        Err(RouterArtifactError::InvalidPolicy { problem }) => {
            assert!(problem.contains("tau"), "{problem}");
        }
        other => panic!("expected an invalid-policy refusal, got {other:?}"),
    }
    assert!(
        !gate_path(&dir, 7).exists(),
        "a refused write left a gate payload behind"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The scale a policy explores at is part of the version, not a setting
/// beside it. Decision 5's whole claim is that one label names one
/// behaviour, and a deterministic gate and a sampled one are two.
///
/// Would fail if `content_hash` stopped hashing `exploration_scale`, either
/// the presence bit or the value: the absent-versus-present arm pins the
/// presence bit and the two-values arm pins the value, because a hash that
/// covered only one of them would still pass a test that changed both at
/// once.
#[test]
fn the_exploration_scale_is_inside_the_version_hash() {
    let deterministic = RouterArtifact {
        provenance: RouterProvenance::recorded(policy(), lineage()),
        ..artifact()
    };
    let sampled = RouterArtifact {
        provenance: RouterProvenance::recorded(
            RouterPolicy {
                exploration_scale: Some(0.5),
                ..policy()
            },
            lineage(),
        ),
        ..artifact()
    };
    let sampled_wider = RouterArtifact {
        provenance: RouterProvenance::recorded(
            RouterPolicy {
                exploration_scale: Some(1.5),
                ..policy()
            },
            lineage(),
        ),
        ..artifact()
    };

    assert_ne!(
        deterministic.content_hash(),
        sampled.content_hash(),
        "absent and present must not hash alike, or the presence byte is dead"
    );
    assert_ne!(
        sampled.content_hash(),
        sampled_wider.content_hash(),
        "two scales must not hash alike, or the value is outside the label"
    );
    // Control: the hash is stable for an unchanged policy, so the two
    // inequalities above are about the field and not about the fixture.
    assert_eq!(deterministic.content_hash(), deterministic.content_hash());
}

/// A bootstrap explores at nothing. Recorded as an absence rather than as
/// a zero: a zero would be a scale, and a refused one.
///
/// Would fail if the bootstrap recorded `Some(0.0)` -- which `validate`
/// also refuses, so the two arms agree -- or any scale at all.
#[test]
fn a_bootstrap_policy_records_no_exploration_scale() {
    let bootstrap = crate::router_init::bootstrap_policy(None, None);
    assert_eq!(bootstrap.exploration_scale, None);
    bootstrap
        .validate()
        .expect("the bootstrap policy must validate");
}

/// A recorded artifact that does not carry the scale field is refused at
/// LOAD, not read as deterministic. This is the "no serde default"
/// property: a writer that dropped the field would produce exactly the
/// value a default would supply, so a default would make the omission
/// silent.
///
/// Would fail if `RouterPolicy::exploration_scale` lost its
/// `deserialize_with` or gained a `#[serde(default)]`: the hand-edited
/// manifest below would parse.
#[test]
fn an_artifact_that_does_not_say_its_scale_is_refused_at_load() {
    let dir = scratch("missing-scale");
    raw_fixture_write(&dir, &recorded()).expect("write");
    // The control: the fixture reads before the field is removed.
    read_artifact(&dir, 7).expect("the fixture must read");

    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
    let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    v["policy"]
        .as_object_mut()
        .expect("a policy object")
        .remove("exploration_scale");
    std::fs::write(
        manifest_path(&dir, 7),
        serde_json::to_vec_pretty(&v).expect("serialize"),
    )
    .expect("rewrite");

    match read_artifact(&dir, 7) {
        Err(RouterArtifactError::Malformed { message, .. }) => {
            assert!(
                message.contains("exploration_scale"),
                "the refusal must name the missing field: {message}"
            );
        }
        other => panic!("expected a malformed-manifest refusal, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two artifacts that differ only in the scale -- or in whether it is
/// present -- are two versions on disk, not one.
///
/// Would fail if the scale were written beside the artifact rather than
/// hashed into it, which is the pin-buys-one-behaviour property decision 5
/// exists to hold.
#[test]
fn two_artifacts_that_differ_only_in_the_scale_hash_differently() {
    let dir = scratch("scale-version");
    // Two counters, so both survive on disk and the round-trip is real
    // rather than the second write overwriting the first.
    let deterministic = RouterArtifact {
        version: 7,
        provenance: RouterProvenance::recorded(policy(), lineage()),
        ..artifact()
    };
    let sampled = RouterArtifact {
        version: 8,
        provenance: RouterProvenance::recorded(
            RouterPolicy {
                exploration_scale: Some(0.25),
                ..policy()
            },
            lineage(),
        ),
        ..artifact()
    };
    raw_fixture_write(&dir, &deterministic).expect("write deterministic");
    raw_fixture_write(&dir, &sampled).expect("write sampled");
    let a = read_artifact(&dir, 7).expect("read deterministic");
    let b = read_artifact(&dir, 8).expect("read sampled");
    assert_ne!(a.content_hash(), b.content_hash());
    assert_eq!(a.content_hash(), deterministic.content_hash());
    assert_eq!(b.content_hash(), sampled.content_hash());
    // The operator-facing identity differs in its hash half, which is the
    // property a pin depends on.
    assert_ne!(
        a.version_label().split_once(':').expect("a label").1,
        b.version_label().split_once(':').expect("a label").1
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn policy_validation_covers_each_way_a_policy_can_be_unservable() {
    let cases: Vec<(&str, RouterPolicy)> = vec![
        (
            "kind",
            RouterPolicy {
                kind: "argmax".into(),
                ..policy()
            },
        ),
        (
            "training_state",
            RouterPolicy {
                training_state: "trained".into(),
                ..policy()
            },
        ),
        (
            "softmax with no tau",
            RouterPolicy {
                tau: None,
                ..policy()
            },
        ),
        (
            "uniform with a tau",
            RouterPolicy {
                kind: "uniform".into(),
                tau: Some(0.7),
                ..policy()
            },
        ),
        (
            "non-finite tau",
            RouterPolicy {
                tau: Some(f64::NAN),
                ..policy()
            },
        ),
        (
            "non-finite epsilon",
            RouterPolicy {
                epsilon: f64::INFINITY,
                ..policy()
            },
        ),
        (
            "tau below its floor",
            RouterPolicy {
                tau: Some(0.05),
                ..policy()
            },
        ),
        (
            "negative epsilon",
            RouterPolicy {
                epsilon: -0.01,
                ..policy()
            },
        ),
        (
            "uniform with an exploration scale",
            RouterPolicy {
                kind: "uniform".into(),
                tau: None,
                exploration_scale: Some(1.0),
                ..policy()
            },
        ),
        (
            "exploration scale below the minimum",
            RouterPolicy {
                exploration_scale: Some(MIN_EXPLORATION_SCALE / 2.0),
                ..policy()
            },
        ),
        (
            "non-finite exploration scale",
            RouterPolicy {
                exploration_scale: Some(f64::NAN),
                ..policy()
            },
        ),
        (
            "negative exploration scale",
            RouterPolicy {
                exploration_scale: Some(-1.0),
                ..policy()
            },
        ),
    ];
    for (name, p) in cases {
        assert!(p.validate().is_err(), "{name} was accepted by validate()");
    }
    // The must-match control. Without it every arm above would pass against
    // a `validate` that rejected everything, including the policy this
    // module writes.
    policy()
        .validate()
        .expect("the fixture policy must validate");
}

/// A version of `recorded()` under a chosen counter, so a test can put two
/// admitted artifacts in one directory.
fn recorded_at(version: u64) -> RouterArtifact {
    RouterArtifact {
        version,
        ..recorded()
    }
}

fn head_for(artifact: &RouterArtifact) -> HeadRecord {
    HeadRecord {
        format: artifact.provenance.format(),
        version: artifact.version,
        content_sha256: artifact.content_hash(),
    }
}

fn write_head_fixture(dir: &Path, head: &HeadRecord) {
    std::fs::create_dir_all(dir).expect("create fixture directory");
    std::fs::write(
        head_path(dir),
        serde_json::to_vec(head).expect("serialize fixture head"),
    )
    .expect("write fixture head");
}

#[test]
fn only_an_absent_head_is_reported_as_none() {
    let dir = scratch("head-absence");
    assert_eq!(read_head(&dir).expect("absent directory"), None);
    std::fs::create_dir_all(&dir).expect("create directory");
    assert_eq!(read_head(&dir).expect("absent head"), None);

    std::fs::create_dir(head_path(&dir)).expect("put a directory at the head path");
    assert!(matches!(
        read_head(&dir),
        Err(RouterArtifactError::HeadUnreadable { .. })
    ));
    std::fs::remove_dir(head_path(&dir)).expect("remove the invalid head");

    let file = dir.join("not-a-directory");
    std::fs::write(&file, b"file").expect("create a non-directory parent");
    assert!(matches!(
        read_head(&file),
        Err(RouterArtifactError::Lock { .. })
    ));

    let expected = head_for(&recorded());
    write_head_fixture(&dir, &expected);
    assert_eq!(read_head(&dir).expect("valid head"), Some(expected));
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_dangling_head_symlink_does_not_restore_counter_selection() {
    let dir = scratch("head-dangling");
    raw_fixture_write(&dir, &artifact()).expect("write a legacy version");
    std::os::unix::fs::symlink(dir.join("missing-head"), head_path(&dir))
        .expect("create a dangling head symlink");
    assert!(matches!(
        read_head(&dir),
        Err(RouterArtifactError::HeadUnreadable { .. })
    ));
    assert!(matches!(
        load_selected(&dir),
        Err(RouterArtifactError::HeadUnreadable { .. })
    ));

    std::fs::remove_file(head_path(&dir)).expect("remove the symlink");
    assert_eq!(read_head(&dir).expect("absent head"), None);
    assert_eq!(
        load_selected(&dir)
            .expect("legacy selection")
            .expect("artifact")
            .version,
        7
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn head_revision_and_full_hash_are_validated_before_selection() {
    let dir = scratch("head-fields");
    let original = recorded();
    raw_fixture_write(&dir, &original).expect("write the recorded artifact");
    let expected = head_for(&original);
    for format in [0, ROUTER_ARTIFACT_FORMAT_LEGACY, ROUTER_ARTIFACT_FORMAT + 1] {
        write_head_fixture(
            &dir,
            &HeadRecord {
                format,
                ..expected.clone()
            },
        );
        assert!(matches!(
            read_head(&dir),
            Err(RouterArtifactError::InvalidHead { .. })
        ));
        assert!(matches!(
            load_selected(&dir),
            Err(RouterArtifactError::InvalidHead { .. })
        ));
    }
    for content_sha256 in [
        String::new(),
        expected.content_sha256[..16].to_string(),
        "a".repeat(63),
        "a".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
    ] {
        write_head_fixture(
            &dir,
            &HeadRecord {
                content_sha256,
                ..expected.clone()
            },
        );
        assert!(matches!(
            read_head(&dir),
            Err(RouterArtifactError::InvalidHead { .. })
        ));
        assert!(matches!(
            load_selected(&dir),
            Err(RouterArtifactError::InvalidHead { .. })
        ));
    }
    write_head_fixture(&dir, &expected);
    assert_eq!(read_head(&dir).expect("valid head"), Some(expected));
    assert_eq!(
        load_selected(&dir)
            .expect("valid selection")
            .expect("artifact"),
        original
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_head_must_match_the_stored_artifact_revision() {
    let dir = scratch("head-revision-mismatch");
    let legacy = artifact();
    raw_fixture_write(&dir, &legacy).expect("write legacy");
    write_head_fixture(
        &dir,
        &HeadRecord {
            format: ROUTER_ARTIFACT_FORMAT,
            ..head_for(&legacy)
        },
    );
    assert!(matches!(
        load_selected(&dir),
        Err(RouterArtifactError::HeadFormatMismatch {
            head_format: ROUTER_ARTIFACT_FORMAT,
            artifact_format: ROUTER_ARTIFACT_FORMAT_LEGACY,
            ..
        })
    ));
    let before = std::fs::read(head_path(&dir)).expect("read head before refusal");
    assert!(matches!(
        commit_fixture(&dir, &recorded()),
        Err(RouterArtifactError::HeadFormatMismatch { .. })
    ));
    assert_eq!(
        std::fs::read(head_path(&dir)).expect("read unchanged head"),
        before
    );

    let recorded = recorded();
    raw_fixture_write(&dir, &recorded).expect("replace the fixture with recorded provenance");
    std::fs::remove_file(head_path(&dir)).expect("remove the inconsistent fixture head");
    commit_fixture(&dir, &recorded).expect("commit a matching revision");
    assert_eq!(
        load_selected(&dir).expect("select").expect("artifact"),
        recorded
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_head_commit_rechecks_the_supplied_identity_against_disk() {
    let dir = scratch("head-commit-identity");
    let previous = recorded_at(1);
    let stored = recorded_at(2);
    raw_fixture_write(&dir, &previous).expect("write previous");
    commit_fixture(&dir, &previous).expect("commit previous");
    raw_fixture_write(&dir, &stored).expect("write next");
    let before = std::fs::read(head_path(&dir)).expect("read previous head");
    let mut supplied = stored.clone();
    supplied.gate_bytes[0] ^= 0xff;
    match commit_fixture(&dir, &supplied) {
        Err(RouterArtifactError::HeadDisagreesWithArtifact {
            version,
            head_hash,
            artifact_hash,
            ..
        }) => {
            assert_eq!(version, 2);
            assert_eq!(head_hash, supplied.content_hash());
            assert_eq!(artifact_hash, stored.content_hash());
        }
        other => panic!("expected the proposed head to disagree, got {other:?}"),
    }
    assert_eq!(
        std::fs::read(head_path(&dir)).expect("read unchanged head"),
        before
    );
    assert!(!dir.join("HEAD.json.staged-2").exists());
    assert_eq!(read_artifact(&dir, 2).expect("stored artifact"), stored);

    commit_fixture(&dir, &stored).expect("commit the actual stored identity");
    assert_eq!(read_head(&dir).expect("new head"), Some(head_for(&stored)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_head_commit_refuses_a_corrupt_payload_without_changing_the_head() {
    let dir = scratch("head-commit-corrupt");
    let previous = recorded_at(1);
    let next = recorded_at(2);
    raw_fixture_write(&dir, &previous).expect("write previous");
    commit_fixture(&dir, &previous).expect("commit previous");
    raw_fixture_write(&dir, &next).expect("write next");
    let before = std::fs::read(head_path(&dir)).expect("read previous head");
    let mut corrupt = next.gate_bytes.clone();
    corrupt[0] ^= 0xff;
    std::fs::write(gate_path(&dir, 2), corrupt).expect("corrupt the stored payload");
    assert!(matches!(
        commit_fixture(&dir, &next),
        Err(RouterArtifactError::HashMismatch { version: 2, .. })
    ));
    assert_eq!(
        std::fs::read(head_path(&dir)).expect("read unchanged head"),
        before
    );
    assert!(!dir.join("HEAD.json.staged-2").exists());

    std::fs::write(gate_path(&dir, 2), &next.gate_bytes).expect("restore fixture payload");
    commit_fixture(&dir, &next).expect("commit the intact artifact");
    assert_eq!(read_head(&dir).expect("new head"), Some(head_for(&next)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_format_4_artifact_cannot_create_or_replace_a_head() {
    let dir = scratch("head-commit-legacy");
    let legacy = artifact();
    raw_fixture_write(&dir, &legacy).expect("write legacy");
    assert!(matches!(
        commit_fixture(&dir, &legacy),
        Err(RouterArtifactError::InvalidHead { .. })
    ));
    assert!(!head_path(&dir).exists());
    assert!(!dir.join("HEAD.json.staged-7").exists());

    let recorded = recorded_at(1);
    raw_fixture_write(&dir, &recorded).expect("write recorded");
    commit_fixture(&dir, &recorded).expect("commit recorded");
    let before = std::fs::read(head_path(&dir)).expect("read recorded head");
    assert!(matches!(
        commit_fixture(&dir, &legacy),
        Err(RouterArtifactError::InvalidHead { .. })
    ));
    assert_eq!(
        std::fs::read(head_path(&dir)).expect("read unchanged head"),
        before
    );
    assert_eq!(
        read_artifact(&dir, legacy.version).expect("legacy unchanged"),
        legacy
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_headless_format_5_directory_refuses_counter_selection() {
    let dir = scratch("headless-recorded");
    let original = recorded();
    raw_fixture_write(&dir, &original).expect("write recorded");
    assert!(matches!(
        load_selected(&dir),
        Err(RouterArtifactError::MissingHead { version: 7, .. })
    ));
    assert!(matches!(
        resolve_startup(Some(&dir), None),
        Err(RouterArtifactError::MissingHead { version: 7, .. })
    ));
    assert_eq!(
        read_artifact(&dir, 7).expect("the artifact is intact"),
        original
    );

    commit_fixture(&dir, &original).expect("commit the head");
    assert_eq!(
        load_selected(&dir)
            .expect("committed selection")
            .expect("artifact"),
        original
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn headless_mixed_revisions_refuse_regardless_of_counter_order() {
    for recorded_version in [3, 9] {
        let dir = scratch(&format!("headless-mixed-{recorded_version}"));
        let recorded = recorded_at(recorded_version);
        let legacy = RouterArtifact {
            version: if recorded_version == 3 { 9 } else { 3 },
            ..artifact()
        };
        raw_fixture_write(&dir, &recorded).expect("write recorded");
        raw_fixture_write(&dir, &legacy).expect("write legacy");
        match load_selected(&dir) {
            Err(RouterArtifactError::MissingHead { version, .. }) => {
                assert_eq!(version, recorded_version);
            }
            other => panic!("expected a missing head for format 5, got {other:?}"),
        }
        commit_fixture(&dir, &recorded).expect("commit the recorded head");
        assert_eq!(
            load_selected(&dir)
                .expect("committed selection")
                .expect("artifact"),
            recorded
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn headless_selection_refuses_a_malformed_lower_manifest() {
    let dir = scratch("headless-malformed-lower");
    let lower = RouterArtifact {
        version: 3,
        ..artifact()
    };
    raw_fixture_write(&dir, &lower).expect("write lower legacy version");
    raw_fixture_write(&dir, &artifact()).expect("write higher legacy version");
    let path = manifest_path(&dir, 3);
    let intact = std::fs::read(&path).expect("read lower manifest");
    std::fs::write(&path, b"{").expect("truncate lower manifest");
    assert!(matches!(
        load_selected(&dir),
        Err(RouterArtifactError::Malformed { path: failed, .. }) if failed == path
    ));
    std::fs::write(&path, intact).expect("restore lower manifest");
    assert_eq!(
        load_selected(&dir)
            .expect("legacy selection")
            .expect("artifact")
            .version,
        7
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn headless_selection_refuses_an_unknown_lower_revision() {
    let dir = scratch("headless-unknown-lower");
    raw_fixture_write(&dir, &artifact()).expect("write higher legacy version");
    std::fs::write(manifest_path(&dir, 3), br#"{"format":6}"#)
        .expect("write an unknown lower revision");
    assert!(matches!(
        load_selected(&dir),
        Err(RouterArtifactError::UnknownFormat { found: 6, .. })
    ));
    std::fs::remove_file(manifest_path(&dir, 3)).expect("remove unknown revision");
    assert_eq!(
        load_selected(&dir)
            .expect("legacy selection")
            .expect("artifact")
            .version,
        7
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn headless_revision_inspection_does_not_load_lower_gates() {
    let dir = scratch("headless-lower-gate");
    raw_fixture_write(
        &dir,
        &RouterArtifact {
            version: 3,
            ..artifact()
        },
    )
    .expect("write lower legacy version");
    raw_fixture_write(&dir, &artifact()).expect("write higher legacy version");
    std::fs::remove_file(gate_path(&dir, 3)).expect("remove the unused lower gate");
    assert!(read_artifact(&dir, 3).is_err());
    assert_eq!(
        load_selected(&dir)
            .expect("legacy selection")
            .expect("artifact")
            .version,
        7
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn headless_revision_inspection_obeys_the_manifest_read_cap() {
    let dir = scratch("headless-manifest-cap");
    raw_fixture_write(&dir, &artifact()).expect("write higher legacy version");
    let path = manifest_path(&dir, 3);
    let mut oversized = br#"{"format":4}"#.to_vec();
    oversized.resize(MAX_ROUTER_MANIFEST_LEN as usize + 1, b' ');
    std::fs::write(&path, oversized).expect("write oversized lower manifest");
    assert!(matches!(
        load_selected(&dir),
        Err(RouterArtifactError::Read { path: failed, .. }) if failed == path
    ));
    std::fs::remove_file(path).expect("remove oversized manifest");
    assert_eq!(
        load_selected(&dir)
            .expect("legacy selection")
            .expect("artifact")
            .version,
        7
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn format_5_pins_require_the_canonical_label_or_full_hash() {
    let dir = scratch("recorded-pin-shape");
    let original = recorded();
    raw_fixture_write(&dir, &original).expect("write recorded");
    commit_fixture(&dir, &original).expect("commit the fixture identity");
    let hash = original.content_hash();
    for hash_prefix in [
        None,
        Some(String::new()),
        Some(hash[..1].to_string()),
        Some(hash[..15].to_string()),
        Some(hash[..17].to_string()),
        Some(hash[..63].to_string()),
        Some(format!("{hash}a")),
        Some("A".repeat(16)),
        Some("g".repeat(64)),
    ] {
        let requested = RouterPin {
            version: original.version,
            hash_prefix,
        };
        assert!(
            matches!(
                resolve_startup(Some(&dir), Some(&requested)),
                Err(RouterArtifactError::PinMalformed { .. })
            ),
            "invalid recorded identity was accepted: {requested:?}"
        );
    }
    let parsed_short = RouterPin::parse(&format!("7:{}", &hash[..1])).expect("hex syntax");
    assert!(matches!(
        resolve_startup(Some(&dir), Some(&parsed_short)),
        Err(RouterArtifactError::PinMalformed { .. })
    ));
    for label in [original.version_label(), format!("7:{hash}")] {
        let requested = RouterPin::parse(&label).expect("identity pin");
        match resolve_startup(Some(&dir), Some(&requested)).expect("matching identity") {
            StartupDisposition::Loaded(resolved) => {
                assert_eq!(resolved.artifact, original);
                assert!(resolved.pinned);
            }
            other => panic!("expected the pinned artifact, got {other:?}"),
        }
    }
    let wrong = format!(
        "7:{}{}",
        if hash.starts_with('0') { "1" } else { "0" },
        &hash[1..16]
    );
    let requested = RouterPin::parse(&wrong).expect("valid identity syntax");
    assert!(matches!(
        resolve_startup(Some(&dir), Some(&requested)),
        Err(RouterArtifactError::PinIdentityMismatch { .. })
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

/// With no head record the directory is a format-4 one and ADR-095's rule
/// applies unchanged. The control is the same call on the same directory
/// after a head is committed: if `load_selected` ignored the head, both
/// halves would answer 9 and the test would pass while proving nothing.
#[cfg(unix)]
#[test]
fn with_no_head_the_highest_counter_is_served_and_a_head_changes_that() {
    let dir = scratch("head-vs-counter");
    for version in [3, 9] {
        raw_fixture_write(
            &dir,
            &RouterArtifact {
                version,
                ..artifact()
            },
        )
        .expect("write a legacy version");
    }

    let by_counter = load_selected(&dir).expect("select").expect("an artifact");
    assert_eq!(by_counter.version, 9, "no head means the counter rule");

    raw_fixture_write(&dir, &recorded_at(2)).expect("write recorded version 2");
    commit_fixture(&dir, &recorded_at(2)).expect("commit head at 2");
    let by_head = load_selected(&dir).expect("select").expect("an artifact");
    assert_eq!(
        by_head.version, 2,
        "a head governs unpinned selection even against a higher counter"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A staged or rejected candidate is exactly this shape on disk: a higher
/// counter present, not named by the head. It must never be promoted by
/// surviving a restart, which is what the write-then-scan rule would do.
#[cfg(unix)]
#[test]
fn a_higher_counter_not_named_by_the_head_is_never_promoted() {
    let dir = scratch("rejected-candidate");
    raw_fixture_write(&dir, &recorded_at(1)).expect("write 1");
    commit_fixture(&dir, &recorded_at(1)).expect("commit head at 1");
    raw_fixture_write(&dir, &recorded_at(2)).expect("write the candidate");

    for round in 0..2 {
        let served = load_selected(&dir).expect("select").expect("an artifact");
        assert_eq!(
            served.version, 1,
            "round {round}: the rejected candidate was promoted by a restart"
        );
    }

    // The control: the same directory once 2 is actually admitted.
    commit_fixture(&dir, &recorded_at(2)).expect("admit 2");
    assert_eq!(
        load_selected(&dir)
            .expect("select")
            .expect("an artifact")
            .version,
        2
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A head naming a version that is not there refuses. Falling back to the
/// next-highest is the promotion the head exists to prevent, and it would
/// happen at the moment someone deleted the admitted artifact by hand.
#[cfg(unix)]
#[test]
fn a_head_naming_an_absent_version_refuses_rather_than_falling_back() {
    let dir = scratch("head-absent");
    raw_fixture_write(&dir, &recorded_at(1)).expect("write 1");
    raw_fixture_write(&dir, &recorded_at(2)).expect("write 2");
    commit_fixture(&dir, &recorded_at(2)).expect("commit head at 2");

    // The control, before the removal: this directory loads.
    assert_eq!(
        load_selected(&dir)
            .expect("select")
            .expect("an artifact")
            .version,
        2
    );

    std::fs::remove_file(manifest_path(&dir, 2)).expect("remove the admitted manifest");
    match load_selected(&dir) {
        Err(RouterArtifactError::HeadNamesAbsentVersion { version, .. }) => {
            assert_eq!(version, 2);
        }
        other => panic!("expected a refusal naming version 2, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// The counter can be written twice and the pair cannot. A later write of
/// the same number reads on disk exactly like an ordinary refit, and it is
/// not one: the head is now pointing at an artifact that was never
/// admitted.
#[cfg(unix)]
#[test]
fn a_head_whose_hash_no_longer_matches_its_version_refuses() {
    let dir = scratch("head-hash");
    raw_fixture_write(&dir, &recorded_at(4)).expect("write 4");
    commit_fixture(&dir, &recorded_at(4)).expect("commit head at 4");
    assert_eq!(
        load_selected(&dir)
            .expect("select")
            .expect("an artifact")
            .version,
        4,
        "control: the committed pair loads"
    );

    let replaced = RouterArtifact {
        gate_bytes: vec![9, 9, 9],
        ..recorded_at(4)
    };
    raw_fixture_write(&dir, &replaced).expect("rewrite 4 with other content");
    match load_selected(&dir) {
        Err(RouterArtifactError::HeadDisagreesWithArtifact {
            version,
            head_hash,
            artifact_hash,
            ..
        }) => {
            assert_eq!(version, 4);
            assert_ne!(head_hash, artifact_hash);
            assert_eq!(artifact_hash, replaced.content_hash());
        }
        other => panic!("expected a hash disagreement, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// An unreadable head is an error, never an absence. Reporting it absent
/// would silently reinstate the counter rule on the one directory whose
/// admission was interrupted, which is the directory where the counter
/// rule is most likely to serve a rejected candidate.
#[cfg(unix)]
#[test]
fn a_present_but_unparseable_head_is_an_error_not_an_absence() {
    let dir = scratch("head-garbage");
    raw_fixture_write(&dir, &recorded_at(1)).expect("write 1");
    commit_fixture(&dir, &recorded_at(1)).expect("commit head at 1");
    raw_fixture_write(&dir, &recorded_at(2)).expect("write 2");

    std::fs::write(head_path(&dir), b"{\"format\": 5, \"vers").expect("truncate the head");
    match read_head(&dir) {
        Err(RouterArtifactError::HeadUnreadable { path, .. }) => {
            assert_eq!(path, head_path(&dir));
        }
        other => panic!("expected an unreadable head, got {other:?}"),
    }
    assert!(
        matches!(
            load_selected(&dir),
            Err(RouterArtifactError::HeadUnreadable { .. })
        ),
        "a truncated head must not fall through to the counter rule and serve 2"
    );

    // Absence is distinguishable from corruption, but neither authorizes
    // counter selection over these format-5 artifacts.
    std::fs::remove_file(head_path(&dir)).expect("remove the head");
    assert_eq!(read_head(&dir).expect("absent head"), None);
    assert!(matches!(
        load_selected(&dir),
        Err(RouterArtifactError::MissingHead { version: 1, .. })
    ));

    let _ = std::fs::remove_dir_all(&dir);
}

/// Committing a head for a version that was never written produces a
/// directory that refuses every unpinned startup. Refusing at the commit
/// is the only place that state can still be avoided.
#[cfg(unix)]
#[test]
fn a_head_cannot_be_committed_for_a_version_that_is_not_on_disk() {
    let dir = scratch("head-commit-absent");
    raw_fixture_write(&dir, &recorded_at(1)).expect("write 1");

    match commit_fixture(&dir, &recorded_at(2)) {
        Err(RouterArtifactError::HeadNamesAbsentVersion { version, .. }) => {
            assert_eq!(version, 2);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(
        !head_path(&dir).exists(),
        "a refused commit must leave no head behind"
    );

    // The control: the version that IS on disk commits.
    commit_fixture(&dir, &recorded_at(1)).expect("commit head at 1");
    assert!(head_path(&dir).exists());

    let _ = std::fs::remove_dir_all(&dir);
}

/// The head is renamed into place, so a reader sees one record or the
/// other and the staging file does not survive the commit.
#[cfg(unix)]
#[test]
fn committing_a_head_replaces_the_previous_one_and_leaves_nothing_staged() {
    let dir = scratch("head-replace");
    raw_fixture_write(&dir, &recorded_at(1)).expect("write 1");
    raw_fixture_write(&dir, &recorded_at(2)).expect("write 2");
    commit_fixture(&dir, &recorded_at(1)).expect("commit 1");
    commit_fixture(&dir, &recorded_at(2)).expect("commit 2");

    let head = read_head(&dir).expect("read").expect("a head");
    assert_eq!(head.version, 2);
    assert_eq!(head.format, ROUTER_ARTIFACT_FORMAT);
    assert_eq!(head.content_sha256, recorded_at(2).content_hash());

    let staged: Vec<_> = std::fs::read_dir(&dir)
        .expect("read dir")
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".staged-"))
        .collect();
    assert!(staged.is_empty(), "staging files survived: {staged:?}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// A pin does not consult the head, so a head advancing while a pin is in
/// force changes the served gate for nobody until the pin is lifted. The
/// control is the unpinned call on the same directory, which does move.
#[cfg(unix)]
#[test]
fn a_pin_disables_promotion_and_a_head_advance_does_not_reach_it() {
    let dir = scratch("pin-vs-head");
    raw_fixture_write(&dir, &recorded_at(1)).expect("write 1");
    raw_fixture_write(&dir, &recorded_at(2)).expect("write 2");
    commit_fixture(&dir, &recorded_at(1)).expect("commit 1");

    let requested = RouterPin::parse(&recorded_at(1).version_label()).expect("identity pin");
    let pinned = |dir: &Path| match resolve_startup(Some(dir), Some(&requested)) {
        Ok(StartupDisposition::Loaded(r)) => {
            assert!(r.pinned);
            r.artifact.version
        }
        other => panic!("expected a pinned load, got {other:?}"),
    };
    assert_eq!(pinned(&dir), 1);

    commit_fixture(&dir, &recorded_at(2)).expect("advance the head to 2");
    assert_eq!(pinned(&dir), 1, "the head advance reached the pin");
    assert_eq!(
        load_selected(&dir)
            .expect("select")
            .expect("an artifact")
            .version,
        2,
        "control: the unpinned path did move, so the advance really happened"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A pin naming a full identity is refused when that counter now holds a
/// different artifact. A bare counter cannot bypass that refusal.
#[cfg(unix)]
#[test]
fn a_pin_naming_an_identity_refuses_when_the_counter_holds_another_artifact() {
    let dir = scratch("pin-identity");
    let original = recorded_at(5);
    raw_fixture_write(&dir, &original).expect("write 5");
    commit_fixture(&dir, &original).expect("commit the original identity");
    let label = original.version_label();
    let requested = RouterPin::parse(&label).expect("a label parses as a pin");

    match resolve_startup(Some(&dir), Some(&requested)) {
        Ok(StartupDisposition::Loaded(r)) => assert_eq!(r.artifact.version_label(), label),
        other => panic!("control: the matching identity should load, got {other:?}"),
    }

    let replaced = RouterArtifact {
        gate_bytes: vec![7, 7],
        ..recorded_at(5)
    };
    raw_fixture_write(&dir, &replaced).expect("rewrite 5");
    match resolve_startup(Some(&dir), Some(&requested)) {
        Err(RouterArtifactError::PinIdentityMismatch {
            version,
            requested: asked,
            found,
        }) => {
            assert_eq!(version, 5);
            assert_eq!(asked, label);
            assert_eq!(found, replaced.version_label());
        }
        other => panic!("expected an identity mismatch, got {other:?}"),
    }
    assert!(matches!(
        resolve_startup(Some(&dir), Some(&pin(5))),
        Err(RouterArtifactError::PinMalformed { .. })
    ));
    let replacement_pin =
        RouterPin::parse(&replaced.version_label()).expect("the replacement identity");
    assert!(matches!(
        resolve_startup(Some(&dir), Some(&replacement_pin)),
        Err(RouterArtifactError::MissingCommitProof { .. })
    ));

    let _ = std::fs::remove_dir_all(&dir);
}

/// A format-4 artifact in a directory that has a head stays reachable by
/// pin. That is what keeps a rollback target available across the revision
/// boundary, and it is the one path by which a legacy artifact can still
/// be served from such a directory.
#[cfg(unix)]
#[test]
fn a_format_4_version_stays_reachable_by_pin_beneath_a_head() {
    let dir = scratch("legacy-under-head");
    let legacy = RouterArtifact {
        version: 8,
        ..artifact()
    };
    raw_fixture_write(&dir, &legacy).expect("write the legacy artifact");
    raw_fixture_write(&dir, &recorded_at(2)).expect("write the admitted one");
    commit_fixture(&dir, &recorded_at(2)).expect("commit the head at 2");

    assert_eq!(
        load_selected(&dir)
            .expect("select")
            .expect("an artifact")
            .version,
        2,
        "the head governs unpinned selection against the higher legacy counter"
    );
    match resolve_startup(Some(&dir), Some(&pin(8))) {
        Ok(StartupDisposition::Loaded(r)) => {
            assert_eq!(r.artifact.version, 8);
            assert_eq!(r.artifact.provenance, RouterProvenance::Legacy);
            assert!(r.pinned);
        }
        other => panic!("expected the legacy artifact by pin, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Both spellings and each way a pin can be unusable, with a must-match
/// control in the same test so a parser that rejected everything would not
/// read as thorough.
#[test]
fn pin_parsing_covers_each_spelling_and_each_refusal() {
    let bare = RouterPin::parse("7").expect("a bare counter");
    assert_eq!(bare.version, 7);
    assert_eq!(bare.hash_prefix, None);
    assert_eq!(bare.label(), "7");

    let full = RouterPin::parse("7:8363b3f04c702da3").expect("a full identity");
    assert_eq!(full.version, 7);
    assert_eq!(full.hash_prefix.as_deref(), Some("8363b3f04c702da3"));
    assert_eq!(full.label(), "7:8363b3f04c702da3");

    for (text, why) in [
        ("", "an empty pin names nothing"),
        ("seven", "a word is not a counter"),
        ("7:", "a trailing colon names no identity"),
        ("7:ZZZZ", "the hash half must be hex"),
        (
            "7:8363B3F0",
            "an uppercase hash is not the spelling we print",
        ),
        ("-1", "a negative counter is not a version"),
    ] {
        assert!(
            matches!(
                RouterPin::parse(text),
                Err(RouterArtifactError::PinMalformed { .. })
            ),
            "{text:?} should be refused: {why}"
        );
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lattice-router-artifact-{}-{}-{}",
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

#[test]
fn a_written_artifact_reads_back_byte_identical() {
    let dir = scratch("roundtrip");
    let original = artifact();
    raw_fixture_write(&dir, &original).expect("write");
    let read = read_artifact(&dir, 7).expect("read");
    assert_eq!(read, original);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_modified_payload_is_refused_rather_than_loaded() {
    let dir = scratch("payload");
    raw_fixture_write(&dir, &artifact()).expect("write");

    // One byte, the smallest change a corrupt or edited payload can be.
    let mut bytes = std::fs::read(gate_path(&dir, 7)).expect("read gate");
    bytes[0] ^= 0xff;
    std::fs::write(gate_path(&dir, 7), &bytes).expect("rewrite gate");

    match read_artifact(&dir, 7) {
        Err(RouterArtifactError::HashMismatch { version, .. }) => assert_eq!(version, 7),
        other => panic!("expected a hash mismatch, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn editing_an_adapter_name_changes_the_version_and_is_refused() {
    // The property the serving path's name match depends on: a name list
    // that could be edited without changing the version would reintroduce
    // the unverifiable pairing one level up, in the manifest instead of in
    // the caller's slice.
    let dir = scratch("names");
    raw_fixture_write(&dir, &artifact()).expect("write");

    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
    assert!(
        raw.contains("medical"),
        "fixture must contain the name it edits"
    );
    let edited = raw.replace("medical", "finance");
    std::fs::write(manifest_path(&dir, 7), edited).expect("rewrite manifest");

    match read_artifact(&dir, 7) {
        Err(RouterArtifactError::HashMismatch {
            recorded, computed, ..
        }) => {
            assert_ne!(recorded, computed);
        }
        other => panic!("expected a hash mismatch, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_hash_covers_the_names_so_two_name_lists_never_collide() {
    // Length-prefixing is what makes this true. Without it these two lists
    // hash the same bytes, and they describe different column-to-adapter
    // correspondences -- the exact confusion the hash exists to catch.
    let a = RouterArtifact {
        version: 1,
        adapter_names: vec!["ab".into(), "c".into()],
        representation: representation(),
        gate_bytes: vec![9],
        provenance: RouterProvenance::Legacy,
    };
    let b = RouterArtifact {
        version: 1,
        adapter_names: vec!["a".into(), "bc".into()],
        representation: representation(),
        gate_bytes: vec![9],
        provenance: RouterProvenance::Legacy,
    };
    assert_ne!(a.content_hash(), b.content_hash());
}

#[test]
fn reordering_the_names_changes_the_hash() {
    // Order is how the artifact records which column was which, so two
    // orders are two different claims and must not share a version.
    let mut reordered = artifact();
    reordered.adapter_names.swap(0, 2);
    assert_ne!(artifact().content_hash(), reordered.content_hash());
}

#[test]
fn writing_a_new_version_leaves_the_previous_one_readable() {
    // Rollback needs the previous gate to still be on disk, without anyone
    // having remembered to take a backup.
    let dir = scratch("versions");
    raw_fixture_write(&dir, &artifact()).expect("write v7");
    let next = RouterArtifact {
        version: 8,
        adapter_names: vec!["legal".into(), "medical".into(), "code".into()],
        representation: representation(),
        gate_bytes: vec![6, 7, 8],
        provenance: RouterProvenance::Legacy,
    };
    raw_fixture_write(&dir, &next).expect("write v8");

    assert_eq!(
        read_artifact(&dir, 7).expect("v7 still readable"),
        artifact()
    );
    assert_eq!(read_artifact(&dir, 8).expect("v8 readable"), next);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unknown_format_revision_is_refused_not_parsed_as_this_one() {
    let dir = scratch("format");
    raw_fixture_write(&dir, &artifact()).expect("write");
    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
    // Bump from the revision the fixture actually wrote, which is the
    // legacy one because `artifact()` records no provenance. Writing
    // `ROUTER_ARTIFACT_FORMAT` here would match nothing once the two
    // revisions diverged, and the replacement would silently be a no-op.
    let written = artifact().provenance.format();
    let bumped = raw.replace(
        &format!("\"format\": {written}"),
        &format!("\"format\": {}", ROUTER_ARTIFACT_FORMAT + 1),
    );
    assert_ne!(bumped, raw, "the format field must have been rewritten");
    std::fs::write(manifest_path(&dir, 7), bumped).expect("rewrite");

    match read_artifact(&dir, 7) {
        Err(RouterArtifactError::UnknownFormat { found, known, .. }) => {
            assert_eq!(found, ROUTER_ARTIFACT_FORMAT + 1);
            assert_eq!(
                known,
                format!("{ROUTER_ARTIFACT_FORMAT_LEGACY} and {ROUTER_ARTIFACT_FORMAT}")
            );
        }
        other => panic!("expected an unknown-format refusal, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_manifest_whose_version_disagrees_with_its_filename_is_refused() {
    // The counter is not inside the content hash, so this edit leaves the
    // hash valid: without an explicit comparison the file loads clean and
    // reports a version the operator never asked for. The control below is
    // the same directory reading fine before the edit, so a refusal here
    // cannot come from a broken fixture.
    let dir = scratch("versionskew");
    raw_fixture_write(&dir, &artifact()).expect("write");
    assert_eq!(read_artifact(&dir, 7).expect("control reads"), artifact());

    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
    let skewed = raw.replace("\"version\": 7", "\"version\": 3");
    assert_ne!(skewed, raw, "the version field must have been rewritten");
    std::fs::write(manifest_path(&dir, 7), skewed).expect("rewrite");

    match read_artifact(&dir, 7) {
        Err(RouterArtifactError::VersionMismatch {
            declared,
            from_filename,
            ..
        }) => {
            assert_eq!(declared, 3);
            assert_eq!(from_filename, 7);
        }
        other => panic!("expected a version mismatch, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_declared_payload_length_that_does_not_match_the_payload_is_refused() {
    // `gate_len` is written by every writer and, until it was compared,
    // read by nobody. It is outside the content hash too, so a manifest
    // that belongs to a different payload passes the hash check whenever
    // the payload it describes is the one on disk.
    let dir = scratch("gatelen");
    raw_fixture_write(&dir, &artifact()).expect("write");
    assert_eq!(read_artifact(&dir, 7).expect("control reads"), artifact());

    let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
    let skewed = raw.replace("\"gate_len\": 5", "\"gate_len\": 4");
    assert_ne!(skewed, raw, "the gate_len field must have been rewritten");
    std::fs::write(manifest_path(&dir, 7), skewed).expect("rewrite");

    match read_artifact(&dir, 7) {
        Err(RouterArtifactError::GateLengthMismatch {
            declared, actual, ..
        }) => {
            assert_eq!(declared, 4);
            assert_eq!(actual, 5);
        }
        other => panic!("expected a gate-length mismatch, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_absent_version_is_not_found_rather_than_a_read_failure() {
    // Absent and corrupt must not be one answer: a server that has never
    // been given a gate and one whose gate was truncated need different
    // operator responses.
    let dir = scratch("absent");
    raw_fixture_write(&dir, &artifact()).expect("write");
    match read_artifact(&dir, 99) {
        Err(RouterArtifactError::NotFound { version, .. }) => assert_eq!(version, 99),
        other => panic!("expected NotFound, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn versions_lists_what_is_there_ascending_and_load_latest_takes_the_highest() {
    let dir = scratch("select");
    for v in [3u64, 11, 7] {
        raw_fixture_write(
            &dir,
            &RouterArtifact {
                version: v,
                adapter_names: vec![format!("a{v}")],
                representation: representation(),
                gate_bytes: vec![v as u8],
                provenance: RouterProvenance::Legacy,
            },
        )
        .expect("write");
    }
    assert_eq!(versions(&dir).expect("versions"), vec![3, 7, 11]);
    let latest = load_selected(&dir).expect("load").expect("some");
    assert_eq!(
        latest.version, 11,
        "11 must win over 7, not sort as a string"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_or_absent_directory_is_none_not_an_error() {
    // A server that has never been given a gate is a state, not a failure.
    let dir = scratch("empty");
    assert_eq!(versions(&dir).expect("absent dir"), Vec::<u64>::new());
    assert!(load_selected(&dir).expect("absent dir").is_none());
    std::fs::create_dir_all(&dir).expect("mkdir");
    assert!(load_selected(&dir).expect("empty dir").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_directory_holding_an_unloadable_artifact_errors_rather_than_reading_as_no_router() {
    // The distinction step 2 rests on: "no gate configured" and "the gate
    // you configured is broken" must not both start a server quietly.
    let dir = scratch("broken");
    raw_fixture_write(&dir, &artifact()).expect("write");
    let mut bytes = std::fs::read(gate_path(&dir, 7)).expect("read");
    bytes[0] ^= 0xff;
    std::fs::write(gate_path(&dir, 7), &bytes).expect("corrupt");

    match load_selected(&dir) {
        Err(RouterArtifactError::HashMismatch { .. }) => {}
        Ok(None) => panic!("a corrupt artifact read as no-router, which is the bug"),
        other => panic!("expected a hash mismatch, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_payload_without_its_manifest_is_invisible_to_both_readers() {
    // The crash-between-writes state. `versions` and `read_artifact` must
    // agree, or a version gets listed and then refuses to load.
    let dir = scratch("halfwritten");
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(gate_path(&dir, 5), [1, 2, 3]).expect("orphan payload");
    assert_eq!(versions(&dir).expect("versions"), Vec::<u64>::new());
    assert!(load_selected(&dir).expect("load").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn only_the_exact_spelling_this_module_writes_counts_as_a_version() {
    // `router-007.json` would otherwise claim version 7 and shadow the
    // real one, with the shadowing decided by directory order.
    let dir = scratch("spelling");
    raw_fixture_write(&dir, &artifact()).expect("write v7");
    std::fs::write(dir.join("router-007.json"), "{}").expect("decoy");
    std::fs::write(dir.join("router-x.json"), "{}").expect("decoy");
    std::fs::write(dir.join("router-7.json.bak"), "{}").expect("decoy");
    assert_eq!(versions(&dir).expect("versions"), vec![7]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_flag_is_the_only_input_that_yields_no_router() {
    // The whole point of the resolver's shape: NoRouter is unreachable
    // from any configured directory, so "broken router" can never arrive
    // at a server as "no router".
    assert!(matches!(
        resolve_startup(None, None).expect("no flag"),
        StartupDisposition::NoRouter
    ));
}

#[test]
fn a_configured_but_empty_directory_refuses_rather_than_starting_without_a_router() {
    let dir = scratch("startup-empty");
    std::fs::create_dir_all(&dir).expect("mkdir");
    match resolve_startup(Some(&dir), None) {
        Err(RouterArtifactError::EmptyDirectory { .. }) => {}
        Ok(StartupDisposition::NoRouter) => {
            panic!("a configured directory read as no-router, which is the state this refuses")
        }
        other => panic!("expected EmptyDirectory, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_configured_directory_that_does_not_exist_refuses() {
    let dir = scratch("startup-absent");
    match resolve_startup(Some(&dir), None) {
        Err(RouterArtifactError::EmptyDirectory { .. }) => {}
        other => panic!("expected a refusal for an absent configured dir, got {other:?}"),
    }
}

#[test]
fn a_configured_directory_with_a_corrupt_artifact_refuses_with_the_artifacts_own_error() {
    // Not a generic "could not start": the operator needs to know the hash
    // failed, because the remedy differs from an empty directory's.
    let dir = scratch("startup-corrupt");
    raw_fixture_write(&dir, &artifact()).expect("write");
    let mut bytes = std::fs::read(gate_path(&dir, 7)).expect("read");
    bytes[0] ^= 0xff;
    std::fs::write(gate_path(&dir, 7), &bytes).expect("corrupt");

    match resolve_startup(Some(&dir), None) {
        Err(RouterArtifactError::HashMismatch { version, .. }) => assert_eq!(version, 7),
        other => panic!("expected the artifact's own hash error, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_pin_with_no_directory_refuses_and_is_not_ignored() {
    // The dangerous reading is that a pin with nothing to resolve against
    // is harmless. It is the opposite: the operator believes a named
    // version is serving and the server has no router at all, with both
    // facts silent. Distinct from NotFound because the remedies differ --
    // a missing flag, not a missing file.
    match resolve_startup(None, Some(&pin(7))) {
        Err(RouterArtifactError::PinWithoutState { version }) => assert_eq!(version, 7),
        Ok(StartupDisposition::NoRouter) => {
            panic!("a pin was ignored, which is the state this refuses")
        }
        other => panic!("expected PinWithoutState, got {other:?}"),
    }
}

#[test]
fn a_pin_selects_its_version_and_not_the_highest() {
    // The arm that separates a pin from a no-op. Two versions present and
    // the pin names the LOWER one, so a resolver that ignored the pin
    // would still return successfully with a valid artifact -- it would
    // just be the wrong one. Pinning to the highest version would pass
    // against both the correct and the broken implementation.
    let dir = scratch("startup-pin-selects");
    raw_fixture_write(&dir, &artifact()).expect("write v7");
    raw_fixture_write(
        &dir,
        &RouterArtifact {
            version: 9,
            adapter_names: vec!["legal".into()],
            representation: representation(),
            gate_bytes: vec![4, 2],
            provenance: RouterProvenance::Legacy,
        },
    )
    .expect("write v9");
    match resolve_startup(Some(&dir), Some(&pin(7))).expect("load") {
        StartupDisposition::Loaded(r) => {
            assert_eq!(r.artifact.version, 7, "the pin did not select its version");
            assert!(r.pinned, "a pinned load must report itself pinned");
        }
        other => panic!("expected Loaded, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_absent_pinned_version_refuses_rather_than_falling_back_to_the_latest() {
    // A fallback here would serve precisely the artifact the operator was
    // pinning away from, under a flag that says otherwise. The directory
    // deliberately HOLDS a loadable artifact, so a resolver that fell back
    // would succeed and look healthy.
    let dir = scratch("startup-pin-absent");
    raw_fixture_write(&dir, &artifact()).expect("write v7");
    match resolve_startup(Some(&dir), Some(&pin(99))) {
        Err(RouterArtifactError::NotFound { version, .. }) => assert_eq!(version, 99),
        Ok(StartupDisposition::Loaded(r)) => panic!(
            "fell back to version {} instead of refusing the absent pin",
            r.artifact.version
        ),
        other => panic!("expected NotFound, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_pinned_corrupt_artifact_refuses_with_the_hash_error_not_the_pin_error() {
    // A pin does not bypass verification, and the refusal must name the
    // real cause: the remedy for a corrupt artifact is not the remedy for
    // a mistyped version.
    let dir = scratch("startup-pin-corrupt");
    raw_fixture_write(&dir, &artifact()).expect("write");
    let mut bytes = std::fs::read(gate_path(&dir, 7)).expect("read");
    bytes[0] ^= 0xff;
    std::fs::write(gate_path(&dir, 7), &bytes).expect("corrupt");
    match resolve_startup(Some(&dir), Some(&pin(7))) {
        Err(RouterArtifactError::HashMismatch { version, .. }) => assert_eq!(version, 7),
        other => panic!("expected HashMismatch, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_good_directory_loads_its_highest_version() {
    let dir = scratch("startup-good");
    raw_fixture_write(&dir, &artifact()).expect("write v7");
    raw_fixture_write(
        &dir,
        &RouterArtifact {
            version: 9,
            adapter_names: vec!["legal".into()],
            representation: representation(),
            gate_bytes: vec![4, 2],
            provenance: RouterProvenance::Legacy,
        },
    )
    .expect("write v9");
    match resolve_startup(Some(&dir), None).expect("load") {
        StartupDisposition::Loaded(r) => {
            assert_eq!(r.artifact.version, 9);
            assert!(!r.pinned, "an unpinned load must not report itself pinned");
        }
        other => panic!("expected Loaded, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_representation_is_inside_the_content_hash() {
    // Leo's ruling, and the reason is the pin. A field stored BESIDE the
    // artifact could be edited without producing a new version, so a
    // pinned version would mean two different things at two different
    // times. Each field is varied alone, because a hash that covered only
    // one of them would still pass a test that changed all of them at once.
    //
    // The fields are ENUMERATED from the serialized struct rather than
    // listed here. The first version of this test listed three closures,
    // one per field, and when `prompt_source` was added the list stayed at
    // three: the new field rode outside the hash and this test passed. A
    // hand-written list of what a struct contains is a claim nobody
    // re-derives when they add to the struct, so it decays in exactly the
    // direction that reads as coverage.
    let base = artifact();
    let value = serde_json::to_value(&base.representation).expect("the representation serializes");
    let fields = value
        .as_object()
        .expect("a struct serializes to a JSON object");
    assert!(
        !fields.is_empty(),
        "no fields were enumerated, so the loop below asserts nothing"
    );

    for (name, original) in fields {
        let changed = match original {
            serde_json::Value::String(text) => serde_json::Value::String(format!("{text}-x")),
            serde_json::Value::Number(number) => {
                let n = number
                    .as_u64()
                    .expect("a numeric field is an unsigned count");
                serde_json::json!(n + 1)
            }
            other => panic!("field {name} has type {other:?}, which this arm cannot vary"),
        };
        let mut mutated = fields.clone();
        mutated.insert(name.clone(), changed);
        let representation: TrainedRepresentation =
            serde_json::from_value(serde_json::Value::Object(mutated))
                .expect("the varied representation deserializes");

        let mut other = base.clone();
        other.representation = representation;
        assert_ne!(
            base.content_hash(),
            other.content_hash(),
            "representation field {name} changed without changing the hash"
        );
    }

    // The must-match control: an untouched copy hashes identically, so the
    // inequalities above are about the edits and not about instability.
    assert_eq!(base.content_hash(), base.clone().content_hash());
}

#[test]
fn the_version_label_carries_both_halves() {
    let a = artifact();
    let label = a.version_label();
    assert!(label.starts_with("7:"), "counter half missing from {label}");
    assert_eq!(
        label.len(),
        "7:".len() + 16,
        "hash half wrong width in {label}"
    );
    let mut changed = a.clone();
    changed.gate_bytes.push(0);
    assert_ne!(a.version_label(), changed.version_label());
}
