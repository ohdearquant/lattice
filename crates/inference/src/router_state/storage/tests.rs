use super::*;

fn artifact(version: u64, recorded: bool) -> RouterArtifact {
    let provenance = if recorded {
        RouterProvenance::recorded(
            crate::router_init::bootstrap_policy(None, None),
            RouterLineage {
                schema_id: "storage-test-schema".into(),
                base_model: "storage-test-base".into(),
                parent: None,
                admission_record: None,
            },
        )
    } else {
        RouterProvenance::Legacy
    };
    RouterArtifact {
        version,
        adapter_names: vec!["one".into(), "two".into()],
        representation: TrainedRepresentation {
            embedding_model: "fixture".into(),
            pooling: "mean_visual".into(),
            prompt_source: "last_user_message".into(),
            loader_format: "fixture".into(),
            input_width: 2,
        },
        gate_bytes: vec![1, 2, 3],
        provenance,
    }
}

// Test authority exercises the production transaction without providing a
// callable learned-admission producer in a production build.
fn publish(
    dir: &Path,
    artifact: &RouterArtifact,
    expected: Option<&HeadRecord>,
    failure: Option<Phase>,
) -> Result<PathBuf, RouterArtifactError> {
    let _lock = writer_lock(dir)?;
    publish_recorded_locked(dir, artifact, expected, &mut |phase| {
        if failure == Some(phase) {
            Err(write_error(
                dir,
                format!("injected failure after {phase:?}"),
            ))
        } else {
            Ok(())
        }
    })
}

fn raw_fixture_write(dir: &Path, artifact: &RouterArtifact) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(gate_path(dir, artifact.version), &artifact.gate_bytes).unwrap();
    std::fs::write(
        manifest_path(dir, artifact.version),
        manifest_bytes(artifact, &manifest_path(dir, artifact.version)).unwrap(),
    )
    .unwrap();
}

fn pin(artifact: &RouterArtifact) -> RouterPin {
    RouterPin::parse(&artifact.version_label()).unwrap()
}

fn selected(dir: &Path) -> RouterArtifact {
    load_selected(dir).unwrap().unwrap()
}

#[test]
#[allow(deprecated)]
fn public_compatibility_writers_cannot_publish_recorded_bytes() {
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path().join("refused");
    let recorded = artifact(1, true);
    for result in [
        write_artifact(&dir, &recorded),
        write_legacy_artifact(&dir, &recorded),
        commit_head(&dir, &recorded),
    ] {
        match result {
            Err(RouterArtifactError::PublicationAuthorityRequired { identity: refused }) => {
                assert_eq!(refused, identity(&recorded))
            }
            other => panic!("expected authority refusal, got {other:?}"),
        }
    }
    assert!(!dir.exists());
    let legacy = artifact(1, false);
    write_artifact(&dir, &legacy).unwrap();
    assert_eq!(selected(&dir), legacy);
    assert!(matches!(
        commit_head(&dir, &legacy),
        Err(RouterArtifactError::InvalidHead { .. })
    ));
    assert!(!head_path(&dir).exists());
}

#[test]
#[allow(deprecated)]
fn compatibility_latest_uses_committed_selection() {
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    write_legacy_artifact(dir, &artifact(8, false)).unwrap();
    assert_eq!(load_latest(dir).unwrap().unwrap().version, 8);
    let first = artifact(1, true);
    publish(dir, &first, None, None).unwrap();
    raw_fixture_write(dir, &artifact(9, true));
    assert_eq!(load_latest(dir).unwrap(), Some(first));
    std::fs::remove_file(head_path(dir)).unwrap();
    assert!(matches!(
        load_latest(dir),
        Err(RouterArtifactError::MissingHead { .. })
    ));
}

#[test]
fn legacy_publication_is_create_only_across_both_formats() {
    for recorded in [false, true] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let original = artifact(1, recorded);
        if recorded {
            publish(dir, &original, None, None).unwrap();
        } else {
            write_legacy_artifact(dir, &original).unwrap();
        }
        let gate_before = std::fs::read(gate_path(dir, 1)).unwrap();
        let manifest_before = std::fs::read(manifest_path(dir, 1)).unwrap();
        let head_before = read_head(dir).unwrap();
        let replacement = RouterArtifact {
            gate_bytes: vec![9, 8],
            ..artifact(1, false)
        };
        assert!(matches!(
            write_legacy_artifact(dir, &replacement),
            Err(RouterArtifactError::AlreadyExists { .. })
        ));
        assert!(matches!(
            write_legacy_artifact(dir, &original),
            Err(RouterArtifactError::AlreadyExists { .. })
                | Err(RouterArtifactError::PublicationAuthorityRequired { .. })
        ));
        assert_eq!(std::fs::read(gate_path(dir, 1)).unwrap(), gate_before);
        assert_eq!(
            std::fs::read(manifest_path(dir, 1)).unwrap(),
            manifest_before
        );
        assert_eq!(read_head(dir).unwrap(), head_before);
        write_legacy_artifact(dir, &artifact(2, false)).unwrap();
        assert_eq!(read_head(dir).unwrap(), head_before);
    }
}

#[test]
fn orphan_payloads_and_manifests_block_publication() {
    for gate in [true, false] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let path = if gate {
            gate_path(dir, 1)
        } else {
            manifest_path(dir, 1)
        };
        std::fs::write(&path, b"preserve this orphan").unwrap();
        assert!(
            matches!(write_legacy_artifact(dir, &artifact(1, false)), Err(RouterArtifactError::AlreadyExists { path: refused }) if refused == path)
        );
        assert_eq!(std::fs::read(path).unwrap(), b"preserve this orphan");
    }
}

#[test]
fn recorded_publication_cannot_reuse_a_legacy_or_recorded_counter() {
    for recorded in [false, true] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let original = artifact(1, recorded);
        if recorded {
            publish(dir, &original, None, None).unwrap();
        } else {
            write_legacy_artifact(dir, &original).unwrap();
        }
        let expected = read_head(dir).unwrap();
        assert!(matches!(
            publish(dir, &artifact(1, true), expected.as_ref(), None),
            Err(RouterArtifactError::AlreadyExists { .. })
        ));
        assert_eq!(read_artifact(dir, 1).unwrap(), original);
        assert_eq!(read_head(dir).unwrap(), expected);
    }
}

#[test]
fn a_stale_expected_head_refuses_before_candidate_materialization() {
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let first = artifact(1, true);
    publish(dir, &first, None, None).unwrap();
    for expected in [None, Some(record_for(&artifact(9, true)))] {
        assert!(matches!(
            publish(dir, &artifact(2, true), expected.as_ref(), None),
            Err(RouterArtifactError::StaleHead { .. })
        ));
        assert!(!gate_path(dir, 2).exists());
        assert!(!manifest_path(dir, 2).exists());
        assert!(!receipt_path(dir, 1).exists());
        assert_eq!(selected(dir), first);
    }
}

#[test]
fn the_previous_head_receipt_precedes_head_replacement() {
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let first = artifact(1, true);
    let second = artifact(2, true);
    publish(dir, &first, None, None).unwrap();
    assert!(!receipt_path(dir, 1).exists());
    let previous = record_for(&first);
    {
        let _lock = writer_lock(dir).unwrap();
        let mut observed = false;
        publish_recorded_locked(dir, &second, Some(&previous), &mut |phase| {
            if phase == Phase::ReceiptPublished {
                observed = true;
                assert_eq!(read_head_unlocked(dir).unwrap(), Some(previous.clone()));
                assert_eq!(read_receipt(dir, 1).unwrap(), Some(previous.clone()));
                assert!(!receipt_path(dir, 2).exists());
            }
            Ok(())
        })
        .unwrap();
        assert!(observed);
    }
    assert_eq!(selected(dir), second);
    assert!(resolve_startup(Some(dir), Some(&pin(&first))).is_ok());
}

#[test]
fn historical_receipts_allow_exact_rollback_with_missing_or_corrupt_head() {
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let first = artifact(1, true);
    publish(dir, &first, None, None).unwrap();
    publish(dir, &artifact(2, true), Some(&record_for(&first)), None).unwrap();
    for corrupt in [true, false] {
        if corrupt {
            std::fs::write(head_path(dir), b"{").unwrap();
        } else {
            std::fs::remove_file(head_path(dir)).unwrap();
        }
        assert!(load_selected(dir).is_err());
        let StartupDisposition::Loaded(resolved) =
            resolve_startup(Some(dir), Some(&pin(&first))).unwrap()
        else {
            panic!("missing pinned artifact")
        };
        assert_eq!(resolved.artifact, first);
        assert!(resolved.pinned);
        assert!(matches!(
            resolve_startup(Some(dir), Some(&pin(&artifact(2, true)))),
            Err(RouterArtifactError::MissingCommitProof { .. })
        ));
    }
}

#[test]
fn unproved_historical_identity_names_its_migration() {
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let unproved = artifact(1, true);
    raw_fixture_write(dir, &unproved);
    publish(dir, &artifact(2, true), None, None).unwrap();
    let error = resolve_startup(Some(dir), Some(&pin(&unproved))).unwrap_err();
    assert!(
        matches!(&error, RouterArtifactError::MissingCommitProof { identity: refused, .. } if refused == &identity(&unproved))
    );
    assert!(
        error
            .to_string()
            .contains("rerun bootstrap into a fresh directory")
    );
    assert!(error.to_string().contains("later admission"));
    assert_eq!(read_artifact(dir, 1).unwrap(), unproved);
}

#[test]
fn conflicting_or_corrupt_receipts_refuse_an_advance() {
    for bytes in [
        b"{".to_vec(),
        serde_json::to_vec(&record_for(&artifact(8, true))).unwrap(),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let first = artifact(1, true);
        publish(dir, &first, None, None).unwrap();
        std::fs::create_dir(dir.join(RECEIPT_DIR)).unwrap();
        std::fs::write(receipt_path(dir, 1), &bytes).unwrap();
        assert!(publish(dir, &artifact(2, true), Some(&record_for(&first)), None).is_err());
        assert_eq!(selected(dir), first);
        assert_eq!(std::fs::read(receipt_path(dir, 1)).unwrap(), bytes);
        assert!(matches!(
            resolve_startup(Some(dir), Some(&pin(&artifact(2, true)))),
            Err(RouterArtifactError::MissingCommitProof { .. })
        ));
    }
}

#[test]
fn a_reused_receipt_is_not_replaced_after_an_interrupted_advance() {
    use std::os::unix::fs::MetadataExt;
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let first = artifact(1, true);
    publish(dir, &first, None, None).unwrap();
    assert!(
        publish(
            dir,
            &artifact(2, true),
            Some(&record_for(&first)),
            Some(Phase::BeforeHeadRename)
        )
        .is_err()
    );
    let receipt = receipt_path(dir, 1);
    let inode = std::fs::metadata(&receipt).unwrap().ino();
    let bytes = std::fs::read(&receipt).unwrap();
    publish(dir, &artifact(3, true), Some(&record_for(&first)), None).unwrap();
    assert_eq!(std::fs::metadata(&receipt).unwrap().ino(), inode);
    assert_eq!(std::fs::read(&receipt).unwrap(), bytes);
    assert_eq!(selected(dir).version, 3);
    assert!(resolve_startup(Some(dir), Some(&pin(&first))).is_ok());
    assert!(matches!(
        resolve_startup(Some(dir), Some(&pin(&artifact(2, true)))),
        Err(RouterArtifactError::MissingCommitProof { .. })
    ));
}

#[test]
fn failures_before_head_replacement_preserve_the_previous_selection() {
    for phase in [
        Phase::Staged,
        Phase::GatePublished,
        Phase::ManifestPublished,
        Phase::ReceiptPublished,
        Phase::BeforeHeadRename,
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let first = artifact(1, true);
        let second = artifact(2, true);
        publish(dir, &first, None, None).unwrap();
        let before = std::fs::read(head_path(dir)).unwrap();
        let error = publish(dir, &second, Some(&record_for(&first)), Some(phase)).unwrap_err();
        assert!(matches!(error, RouterArtifactError::Write { .. }));
        assert_eq!(std::fs::read(head_path(dir)).unwrap(), before);
        assert_eq!(selected(dir), first);
        assert!(resolve_startup(Some(dir), Some(&pin(&second))).is_err());
        assert!(!receipt_path(dir, 2).exists());
        assert_eq!(
            receipt_path(dir, 1).exists(),
            matches!(phase, Phase::ReceiptPublished | Phase::BeforeHeadRename)
        );
    }
}

#[test]
fn interrupted_first_publication_never_selects_an_uncommitted_artifact() {
    for phase in [
        Phase::Staged,
        Phase::GatePublished,
        Phase::ManifestPublished,
        Phase::BeforeHeadRename,
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let first = artifact(1, true);
        assert!(publish(dir, &first, None, Some(phase)).is_err());
        assert_eq!(read_head(dir).unwrap(), None);
        assert!(!receipt_path(dir, 1).exists());
        match phase {
            Phase::Staged | Phase::GatePublished => {
                assert_eq!(load_selected(dir).unwrap(), None)
            }
            _ => assert!(matches!(
                load_selected(dir),
                Err(RouterArtifactError::MissingHead { version: 1, .. })
            )),
        }
        assert!(resolve_startup(Some(dir), Some(&pin(&first))).is_err());
    }
}

#[test]
fn failure_after_head_replacement_reports_an_uncertain_commit() {
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let first = artifact(1, true);
    let second = artifact(2, true);
    publish(dir, &first, None, None).unwrap();
    let error = publish(
        dir,
        &second,
        Some(&record_for(&first)),
        Some(Phase::HeadRenamed),
    )
    .unwrap_err();
    assert!(
        matches!(error, RouterArtifactError::CommitUncertain { identity: named, .. } if named == identity(&second))
    );
    assert_eq!(selected(dir), second);
    assert_eq!(read_receipt(dir, 1).unwrap(), Some(record_for(&first)));
    assert!(!receipt_path(dir, 2).exists());
}

#[test]
fn legacy_post_link_sync_failure_reports_the_visible_uncertain_version() {
    use sync_probe::Kind;
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let first = artifact(1, false);
    let second = artifact(2, false);
    publish_legacy(dir, &first).unwrap();
    let failure_dir = dir.to_path_buf();
    let (result, events) = sync_probe::run(
        move |event| {
            event.kind == Kind::Directory
                && event.path == failure_dir
                && manifest_path(&failure_dir, 2).exists()
        },
        || publish_legacy(dir, &second),
    );
    let error = result.expect_err("the post-link directory sync must be reached");
    assert!(
        error
            .to_string()
            .contains("injected synchronization failure")
    );
    assert!(
        matches!(error, RouterArtifactError::CommitUncertain { identity: named, .. } if named == identity(&second))
    );
    assert_eq!(events.last().unwrap().path, dir);
    assert_eq!(selected(dir), second);
    assert_eq!(read_artifact(dir, 1).unwrap(), first);
    assert_eq!(read_head(dir).unwrap(), None);
    assert!(!receipt_path(dir, 2).exists());
    assert!(matches!(
        publish_legacy(dir, &second),
        Err(RouterArtifactError::AlreadyExists { .. })
    ));
}

#[test]
fn read_only_legacy_and_copied_modern_directories_need_no_new_lock() {
    use std::os::unix::fs::PermissionsExt;
    for recorded in [false, true] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let original = artifact(1, recorded);
        raw_fixture_write(dir, &original);
        if recorded {
            std::fs::write(
                head_path(dir),
                serde_json::to_vec(&record_for(&original)).unwrap(),
            )
            .unwrap();
        }
        let permissions = std::fs::metadata(dir).unwrap().permissions();
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        assert_eq!(selected(dir), original);
        assert!(resolve_startup(Some(dir), Some(&pin(&original))).is_ok());
        assert!(!dir.join(LOCK_FILE).exists());
        std::fs::set_permissions(dir, permissions).unwrap();
    }
}

#[test]
fn a_lock_created_during_compatibility_read_retries_even_an_error() {
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let mut reads = 0;
    let result = with_read_lock(dir, || {
        reads += 1;
        if reads == 1 {
            let _lock = writer_lock(dir)?;
            return Err(write_error(dir, "discarded unlocked read"));
        }
        Ok(42)
    })
    .unwrap();
    assert_eq!(result, 42);
    assert_eq!(reads, 2);
    assert!(dir.join(LOCK_FILE).exists());
}

#[test]
fn directory_sync_stops_at_the_observed_creation_boundary() {
    for already_present in [false, true] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path().join("first/second/router");
        if already_present {
            std::fs::create_dir_all(&dir).unwrap();
        }
        let mut observed = Vec::new();
        ensure_dir_with_sync(&dir, |path| {
            observed.push(path.to_path_buf());
            sync_dir(path)
        })
        .unwrap();
        let canonical = std::fs::canonicalize(&dir).unwrap();
        let count = if already_present { 2 } else { 4 };
        let expected: Vec<_> = canonical
            .ancestors()
            .take(count)
            .map(Path::to_path_buf)
            .collect();
        assert_eq!(observed, expected);
        if !already_present {
            assert_eq!(
                observed.last().unwrap(),
                &std::fs::canonicalize(temporary.path()).unwrap()
            );
        }
    }
}

#[test]
fn publication_below_a_0711_ancestor_syncs_only_the_writable_boundary() {
    use std::os::unix::fs::PermissionsExt;

    struct RestorePermissions {
        path: PathBuf,
        permissions: std::fs::Permissions,
    }
    impl Drop for RestorePermissions {
        fn drop(&mut self) {
            let _ = std::fs::set_permissions(&self.path, self.permissions.clone());
        }
    }

    for target_exists in [false, true] {
        let temporary = tempfile::tempdir().unwrap();
        let ancestor = temporary.path().join("ancestor");
        let writable = ancestor.join("writable");
        let dir = writable.join("router");
        std::fs::create_dir_all(&writable).unwrap();
        if target_exists {
            std::fs::create_dir(&dir).unwrap();
        }
        let canonical_ancestor = std::fs::canonicalize(&ancestor).unwrap();
        let restore = RestorePermissions {
            path: ancestor.clone(),
            permissions: std::fs::metadata(&ancestor).unwrap().permissions(),
        };
        std::fs::set_permissions(&ancestor, std::fs::Permissions::from_mode(0o711)).unwrap();
        assert_eq!(
            std::fs::metadata(&ancestor).unwrap().permissions().mode() & 0o777,
            0o711
        );
        let original = artifact(1, true);
        let (result, events) = sync_probe::run(|_| false, || publish(&dir, &original, None, None));
        drop(restore);
        result.unwrap();
        assert_eq!(selected(&dir), original);
        let canonical_dir = std::fs::canonicalize(&dir).unwrap();
        let canonical_writable = std::fs::canonicalize(&writable).unwrap();
        for expected in [&canonical_dir, &canonical_writable] {
            assert!(events.iter().any(|event| {
                event.kind == sync_probe::Kind::Directory && &event.path == expected
            }));
        }
        // The current user owns this fixture, so 0711 still permits reads.
        // Observed paths make the boundary check independent of credentials.
        for forbidden in canonical_ancestor.ancestors() {
            assert!(!events.iter().any(|event| {
                event.kind == sync_probe::Kind::Directory && event.path == forbidden
            }));
        }
    }
}

#[test]
fn concurrent_existing_target_retries_only_its_failed_parent_sync() {
    use std::sync::mpsc;
    use std::time::Duration;

    let temporary = tempfile::tempdir().unwrap();
    let parent = std::fs::canonicalize(temporary.path()).unwrap();
    let dir = parent.join("router");
    let (ready_tx, ready_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let first_dir = dir.clone();
    let first_parent = parent.clone();
    let worker = std::thread::spawn(move || {
        let mut attempted = Vec::new();
        let result = ensure_dir_with_sync(&first_dir, |path| {
            attempted.push(path.to_path_buf());
            if path == first_parent {
                ready_tx.send(()).unwrap();
                resume_rx.recv_timeout(Duration::from_secs(30)).unwrap();
                return Err(write_error(path, "injected parent synchronization failure"));
            }
            sync_dir(path)
        });
        finished_tx.send((result, attempted)).unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    assert!(dir.is_dir());
    let mut recovered = Vec::new();
    ensure_dir_with_sync(&dir, |path| {
        recovered.push(path.to_path_buf());
        sync_dir(path)
    })
    .unwrap();
    let expected = vec![dir, parent.clone()];
    assert_eq!(recovered, expected);
    resume_tx.send(()).unwrap();
    let (result, attempted) = finished_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    worker.join().unwrap();
    assert!(matches!(result, Err(RouterArtifactError::Write { path, .. }) if path == parent));
    assert_eq!(attempted, expected);
}

#[test]
fn directory_sync_terminates_for_relative_dot_and_root_paths() {
    let cwd = std::env::current_dir().unwrap();
    let temporary = tempfile::tempdir_in(&cwd).unwrap();
    let relative = temporary.path().strip_prefix(&cwd).unwrap().join("router");
    assert!(!relative.is_absolute());
    for path in [relative.as_path(), Path::new("."), Path::new("/")] {
        let mut observed = Vec::new();
        ensure_dir_with_sync(path, |ancestor| {
            observed.push(ancestor.to_path_buf());
            Ok(())
        })
        .unwrap();
        let canonical = std::fs::canonicalize(path).unwrap();
        let expected: Vec<_> = canonical
            .ancestors()
            .take(2)
            .map(Path::to_path_buf)
            .collect();
        assert_eq!(observed, expected);
        if path == Path::new("/") {
            assert_eq!(observed.len(), 1);
        }
    }
}

#[test]
fn directory_creation_accepts_symlink_ancestors_but_refuses_a_selected_symlink() {
    use std::os::unix::fs::symlink;
    let temporary = tempfile::tempdir().unwrap();
    let actual = temporary.path().join("actual");
    std::fs::create_dir(&actual).unwrap();
    let alias = temporary.path().join("alias");
    symlink(&actual, &alias).unwrap();
    let dir = alias.join("new-parent/router");
    let original = artifact(1, false);
    write_legacy_artifact(&dir, &original).unwrap();
    assert_eq!(selected(&dir), original);
    assert_eq!(
        std::fs::canonicalize(&dir).unwrap(),
        std::fs::canonicalize(actual.join("new-parent/router")).unwrap()
    );
    assert!(matches!(
        ensure_dir(&alias),
        Err(RouterArtifactError::Write { .. })
    ));
    assert!(
        std::fs::symlink_metadata(alias)
            .unwrap()
            .file_type()
            .is_symlink()
    );
}

#[test]
fn existing_locks_are_readable_without_directory_write_permission() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let original = artifact(1, false);
    write_legacy_artifact(dir, &original).unwrap();
    let path = dir.join(LOCK_FILE);
    let inode = std::fs::metadata(&path).unwrap().ino();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
    let permissions = std::fs::metadata(dir).unwrap().permissions();
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    assert_eq!(selected(dir), original);
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
    std::fs::set_permissions(dir, permissions).unwrap();
}

#[test]
fn invalid_lock_and_staging_paths_refuse_without_following_symlinks() {
    use std::os::unix::fs::symlink;
    for name in [LOCK_FILE, STAGING_DIR] {
        let temporary = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        symlink(outside.path(), dir.join(name)).unwrap();
        assert!(write_legacy_artifact(dir, &artifact(1, false)).is_err());
        assert!(!manifest_path(dir, 1).exists());
        assert_eq!(std::fs::read_dir(outside.path()).unwrap().count(), 0);
        if name == LOCK_FILE {
            assert!(load_selected(dir).is_err());
        }
    }
    let temporary = tempfile::tempdir().unwrap();
    std::fs::create_dir(temporary.path().join(LOCK_FILE)).unwrap();
    assert!(matches!(
        load_selected(temporary.path()),
        Err(RouterArtifactError::Lock { .. })
    ));
}

#[test]
fn canonical_and_receipt_symlinks_cannot_be_replaced_or_authorize_selection() {
    use std::os::unix::fs::symlink;
    for gate in [true, false] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let path = if gate {
            gate_path(dir, 1)
        } else {
            manifest_path(dir, 1)
        };
        symlink(dir.join("absent-target"), &path).unwrap();
        assert!(matches!(
            write_legacy_artifact(dir, &artifact(1, false)),
            Err(RouterArtifactError::AlreadyExists { .. })
        ));
        assert!(
            std::fs::symlink_metadata(path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let first = artifact(1, true);
    raw_fixture_write(dir, &first);
    std::fs::create_dir(dir.join(RECEIPT_DIR)).unwrap();
    let external = dir.join("forged.json");
    std::fs::write(&external, serde_json::to_vec(&record_for(&first)).unwrap()).unwrap();
    symlink(external, receipt_path(dir, 1)).unwrap();
    assert!(matches!(
        resolve_startup(Some(dir), Some(&pin(&first))),
        Err(RouterArtifactError::MissingCommitProof { .. })
    ));
}

#[test]
fn a_head_or_historical_receipt_reserves_its_counter_after_payload_loss() {
    for historical in [false, true] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let first = artifact(1, true);
        publish(dir, &first, None, None).unwrap();
        if historical {
            publish(dir, &artifact(2, true), Some(&record_for(&first)), None).unwrap();
        }
        let head = std::fs::read(head_path(dir)).unwrap();
        std::fs::remove_file(gate_path(dir, 1)).unwrap();
        std::fs::remove_file(manifest_path(dir, 1)).unwrap();
        assert!(matches!(
            write_legacy_artifact(dir, &artifact(1, false)),
            Err(RouterArtifactError::AlreadyExists { .. })
        ));
        assert!(!gate_path(dir, 1).exists());
        assert!(!manifest_path(dir, 1).exists());
        assert_eq!(std::fs::read(head_path(dir)).unwrap(), head);
    }
}

#[test]
fn independent_handles_observe_locks_inside_real_critical_sections() {
    use std::fs::TryLockError;
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    write_legacy_artifact(dir, &artifact(8, false)).unwrap();
    let lock_path = dir.join(LOCK_FILE);
    let independent_writer = || {
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap()
    };
    {
        let control = independent_writer();
        control.try_lock().unwrap();
    }
    with_read_lock(dir, || {
        let contender = independent_writer();
        assert!(matches!(
            contender.try_lock(),
            Err(TryLockError::WouldBlock)
        ));
        Ok(())
    })
    .unwrap();
    {
        let control = independent_writer();
        control.try_lock().unwrap();
    }
    let mut observed = false;
    {
        let _lock = writer_lock(dir).unwrap();
        publish_recorded_locked(dir, &artifact(1, true), None, &mut |phase| {
            if phase == Phase::Staged {
                observed = true;
                let reader = File::open(&lock_path).unwrap();
                assert!(matches!(
                    reader.try_lock_shared(),
                    Err(TryLockError::WouldBlock)
                ));
            }
            Ok(())
        })
        .unwrap();
    }
    assert!(observed);
    let control = File::open(&lock_path).unwrap();
    control.try_lock_shared().unwrap();
}

#[test]
fn actual_sync_operations_precede_each_publication_boundary() {
    use sync_probe::{Event, Kind};
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let first = artifact(1, true);
    let second = artifact(2, true);
    publish(dir, &first, None, None).unwrap();
    let mut stages = Vec::new();
    let mut before_final_sync = 0;
    let (result, events) = sync_probe::run(
        |_| false,
        || {
            let _lock = writer_lock(dir).unwrap();
            publish_recorded_locked(dir, &second, Some(&record_for(&first)), &mut |phase| {
                let events = sync_probe::events();
                stages.push(phase);
                let file_position = |name: &str| {
                    events
                        .iter()
                        .position(|event| {
                            event.kind == Kind::File
                                && event.path.file_name().is_some_and(|found| found == name)
                        })
                        .unwrap()
                };
                let gate = file_position("router-2.gate");
                let manifest = file_position("router-2.json");
                let stage = events[gate].path.parent().unwrap().to_path_buf();
                assert!(file_position(LOCK_FILE) < gate);
                assert!(gate < manifest);
                match phase {
                    Phase::Staged | Phase::GatePublished => {
                        assert_eq!(
                            events.last().unwrap(),
                            &Event {
                                kind: Kind::Directory,
                                path: stage
                            }
                        );
                        assert_eq!(events.len(), manifest + 2);
                    }
                    Phase::ManifestPublished => {
                        assert_eq!(
                            events.last().unwrap(),
                            &Event {
                                kind: Kind::Directory,
                                path: dir.to_path_buf()
                            }
                        );
                        assert_eq!(events.len(), manifest + 3);
                    }
                    Phase::ReceiptPublished => {
                        let receipt = file_position("previous-head.json");
                        assert_eq!(
                            &events[receipt + 1..],
                            &[
                                Event {
                                    kind: Kind::Directory,
                                    path: stage
                                },
                                Event {
                                    kind: Kind::Directory,
                                    path: dir.join(RECEIPT_DIR)
                                },
                            ]
                        );
                    }
                    Phase::BeforeHeadRename | Phase::HeadRenamed => {
                        let head = file_position(ROUTER_HEAD_FILE);
                        assert_eq!(
                            &events[head + 1..],
                            &[Event {
                                kind: Kind::Directory,
                                path: stage
                            }]
                        );
                        before_final_sync = events.len();
                    }
                }
                Ok(())
            })
        },
    );
    result.unwrap();
    assert_eq!(
        stages,
        [
            Phase::Staged,
            Phase::GatePublished,
            Phase::ManifestPublished,
            Phase::ReceiptPublished,
            Phase::BeforeHeadRename,
            Phase::HeadRenamed
        ]
    );
    assert_eq!(events.len(), before_final_sync + 1);
    assert_eq!(
        events.last().unwrap(),
        &Event {
            kind: Kind::Directory,
            path: dir.to_path_buf()
        }
    );
    assert_eq!(selected(dir), second);
}

#[test]
fn actual_sync_failures_preserve_selection_or_report_uncertainty() {
    use sync_probe::Kind;
    #[derive(Clone, Copy, Debug)]
    enum Failure {
        Lock,
        Gate,
        Manifest,
        Receipt,
        Head,
        MaterializedDirectory,
        ReceiptDirectory,
        FinalDirectory,
    }
    for failure in [
        Failure::Lock,
        Failure::Gate,
        Failure::Manifest,
        Failure::Receipt,
        Failure::Head,
        Failure::MaterializedDirectory,
        Failure::ReceiptDirectory,
        Failure::FinalDirectory,
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let first = artifact(1, true);
        let second = artifact(2, true);
        publish(dir, &first, None, None).unwrap();
        let injected_dir = dir.to_path_buf();
        let (result, events) = sync_probe::run(
            move |event| {
                let file_named = |name: &str| {
                    event.kind == Kind::File
                        && event.path.file_name().is_some_and(|found| found == name)
                };
                match failure {
                    Failure::Lock => file_named(LOCK_FILE),
                    Failure::Gate => file_named("router-2.gate"),
                    Failure::Manifest => file_named("router-2.json"),
                    Failure::Receipt => file_named("previous-head.json"),
                    Failure::Head => file_named(ROUTER_HEAD_FILE),
                    Failure::MaterializedDirectory => {
                        event.kind == Kind::Directory
                            && event.path == injected_dir
                            && manifest_path(&injected_dir, 2).exists()
                    }
                    Failure::ReceiptDirectory => {
                        event.kind == Kind::Directory
                            && event.path == injected_dir.join(RECEIPT_DIR)
                            && receipt_path(&injected_dir, 1).exists()
                    }
                    Failure::FinalDirectory => {
                        event.kind == Kind::Directory
                            && event.path == injected_dir
                            && read_head_unlocked(&injected_dir)
                                .unwrap()
                                .is_some_and(|head| head.version == 2)
                    }
                }
            },
            || publish(dir, &second, Some(&record_for(&first)), None),
        );
        let error = result.expect_err("the selected real sync operation must be reached");
        assert!(
            error
                .to_string()
                .contains("injected synchronization failure"),
            "{failure:?}: {error}"
        );
        assert!(!events.is_empty());
        if matches!(failure, Failure::FinalDirectory) {
            assert!(
                matches!(error, RouterArtifactError::CommitUncertain { identity: named, .. } if named == identity(&second))
            );
            assert_eq!(selected(dir), second);
        } else {
            assert!(matches!(error, RouterArtifactError::Write { .. }));
            assert_eq!(selected(dir), first);
            assert!(resolve_startup(Some(dir), Some(&pin(&second))).is_err());
        }
    }
}

#[test]
fn reused_receipts_require_file_and_directory_syncs_before_advancing() {
    use sync_probe::{Event, Kind};
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let first = artifact(1, true);
    publish(dir, &first, None, None).unwrap();
    assert!(
        publish(
            dir,
            &artifact(2, true),
            Some(&record_for(&first)),
            Some(Phase::BeforeHeadRename)
        )
        .is_err()
    );
    let receipt = receipt_path(dir, 1);
    let failure_path = receipt.clone();
    let (result, _) = sync_probe::run(
        move |event| event.kind == Kind::File && event.path == failure_path,
        || publish(dir, &artifact(3, true), Some(&record_for(&first)), None),
    );
    assert!(matches!(result, Err(RouterArtifactError::Write { .. })));
    assert_eq!(selected(dir), first);
    let mut observed = false;
    let (result, _) = sync_probe::run(
        |_| false,
        || {
            let _lock = writer_lock(dir).unwrap();
            publish_recorded_locked(
                dir,
                &artifact(4, true),
                Some(&record_for(&first)),
                &mut |phase| {
                    if phase == Phase::ReceiptPublished {
                        observed = true;
                        let events = sync_probe::events();
                        assert_eq!(
                            &events[events.len() - 3..],
                            &[
                                Event {
                                    kind: Kind::File,
                                    path: receipt.clone()
                                },
                                Event {
                                    kind: Kind::Directory,
                                    path: dir.join(RECEIPT_DIR)
                                },
                                Event {
                                    kind: Kind::Directory,
                                    path: dir.to_path_buf()
                                },
                            ]
                        );
                    }
                    Ok(())
                },
            )
        },
    );
    result.unwrap();
    assert!(observed);
    assert_eq!(selected(dir).version, 4);
}

#[test]
fn a_shared_reader_waits_for_the_publisher_and_observes_the_committed_state() {
    use std::sync::mpsc;
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path().to_path_buf();
    let lock = writer_lock(&dir).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let reader_dir = dir.clone();
    let reader = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        done_tx.send(load_selected(&reader_dir)).unwrap();
    });
    started_rx.recv().unwrap();
    assert!(matches!(
        done_rx.recv_timeout(std::time::Duration::from_millis(25)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    let first = artifact(1, true);
    publish_recorded_locked(&dir, &first, None, &mut |_| Ok(())).unwrap();
    drop(lock);
    assert_eq!(
        done_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .unwrap(),
        Some(first)
    );
    reader.join().unwrap();
}
