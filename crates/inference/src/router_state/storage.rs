//! Cooperating readers and publishers share a persistent lock inode. Manual
//! mutation or replacement of that inode is outside this storage protocol.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

const LOCK_FILE: &str = ".router.lock";
const STAGING_DIR: &str = ".staging";
const RECEIPT_DIR: &str = ".committed";
static STAGE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn write_error(path: &Path, error: impl std::fmt::Display) -> RouterArtifactError {
    RouterArtifactError::Write {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

fn lock_error(path: &Path, error: impl std::fmt::Display) -> RouterArtifactError {
    RouterArtifactError::Lock {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

fn present(path: &Path) -> Result<bool, std::io::Error> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn open_lock(path: &Path, writer: bool) -> Result<File, RouterArtifactError> {
    if present(path).map_err(|error| lock_error(path, error))? {
        let metadata = std::fs::symlink_metadata(path).map_err(|error| lock_error(path, error))?;
        if !metadata.is_file() {
            return Err(lock_error(
                path,
                "the persistent lock must be a regular file",
            ));
        }
    }
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(writer)
        .create(writer)
        .truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|error| lock_error(path, error))?;
    if !file
        .metadata()
        .map_err(|error| lock_error(path, error))?
        .is_file()
    {
        return Err(lock_error(
            path,
            "the persistent lock must be a regular file",
        ));
    }
    if writer {
        file.lock().map_err(|error| lock_error(path, error))?;
    } else {
        file.lock_shared()
            .map_err(|error| lock_error(path, error))?;
    }
    Ok(file)
}

pub(super) fn with_read_lock<T>(
    dir: &Path,
    mut read: impl FnMut() -> Result<T, RouterArtifactError>,
) -> Result<T, RouterArtifactError> {
    let path = dir.join(LOCK_FILE);
    if present(&path).map_err(|error| lock_error(&path, error))? {
        let _lock = open_lock(&path, false)?;
        return read();
    }
    let result = read();
    // Publishers create the lock before all state mutations and never unlink
    // it. Two absent observations let old read-only directories linearize here.
    if !present(&path).map_err(|error| lock_error(&path, error))? {
        return result;
    }
    let _lock = open_lock(&path, false)?;
    read()
}

#[cfg(all(test, unix))]
mod sync_probe {
    use std::cell::RefCell;

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum Kind {
        File,
        Directory,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(super) struct Event {
        pub(super) kind: Kind,
        pub(super) path: PathBuf,
    }

    struct Trace {
        events: Vec<Event>,
        fail: Box<dyn FnMut(&Event) -> bool>,
    }

    thread_local! {
        static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
    }

    pub(super) fn observe(kind: Kind, path: &Path) -> Result<(), RouterArtifactError> {
        TRACE.with(|trace| {
            let mut trace = trace.borrow_mut();
            let Some(trace) = trace.as_mut() else {
                return Ok(());
            };
            let event = Event {
                kind,
                path: path.to_path_buf(),
            };
            trace.events.push(event.clone());
            if (trace.fail)(&event) {
                return Err(write_error(path, "injected synchronization failure"));
            }
            Ok(())
        })
    }

    pub(super) fn events() -> Vec<Event> {
        TRACE.with(|trace| trace.borrow().as_ref().unwrap().events.clone())
    }

    pub(super) fn run<T>(
        fail: impl FnMut(&Event) -> bool + 'static,
        operation: impl FnOnce() -> T,
    ) -> (T, Vec<Event>) {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                TRACE.with(|trace| {
                    trace.borrow_mut().take();
                });
            }
        }
        TRACE.with(|trace| {
            assert!(
                trace.borrow().is_none(),
                "nested sync probes are unsupported"
            );
            *trace.borrow_mut() = Some(Trace {
                events: Vec::new(),
                fail: Box::new(fail),
            });
        });
        let _reset = Reset;
        let result = operation();
        (result, events())
    }
}

fn sync_file(file: &File, path: &Path) -> Result<(), RouterArtifactError> {
    #[cfg(all(test, unix))]
    sync_probe::observe(sync_probe::Kind::File, path)?;
    file.sync_all().map_err(|error| write_error(path, error))
}

fn sync_dir(path: &Path) -> Result<(), RouterArtifactError> {
    #[cfg(unix)]
    {
        #[cfg(test)]
        sync_probe::observe(sync_probe::Kind::Directory, path)?;
        File::open(path)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| write_error(path, error))
    }
    #[cfg(not(unix))]
    {
        Err(write_error(
            path,
            "durable router publication requires supported directory synchronization",
        ))
    }
}

fn ensure_dir(path: &Path) -> Result<(), RouterArtifactError> {
    ensure_dir_with_sync(path, sync_dir)
}

fn ensure_dir_with_sync(
    path: &Path,
    mut sync: impl FnMut(&Path) -> Result<(), RouterArtifactError>,
) -> Result<(), RouterArtifactError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return Err(write_error(path, "not a directory")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(path).map_err(|error| write_error(path, error))?;
            if !std::fs::symlink_metadata(path)
                .map_err(|error| write_error(path, error))?
                .is_dir()
            {
                return Err(write_error(path, "not a directory"));
            }
        }
        Err(error) => return Err(write_error(path, error)),
    }
    // An existing name may have been created by a concurrent initializer or
    // survived a failed sync. Each successful writer establishes durability
    // independently. Canonical ancestry permits ordinary /tmp and /var aliases
    // while the selected directory itself must remain a real directory.
    let canonical = std::fs::canonicalize(path).map_err(|error| write_error(path, error))?;
    for ancestor in canonical.ancestors() {
        sync(ancestor)?;
    }
    Ok(())
}

fn writer_lock(dir: &Path) -> Result<File, RouterArtifactError> {
    #[cfg(not(unix))]
    {
        Err(write_error(
            dir,
            "durable router publication requires supported directory synchronization",
        ))
    }
    #[cfg(unix)]
    {
        ensure_dir(dir)?;
        let path = dir.join(LOCK_FILE);
        let lock = open_lock(&path, true)?;
        sync_file(&lock, &path)?;
        sync_dir(dir)?;
        Ok(lock)
    }
}

fn vacant(path: &Path) -> Result<(), RouterArtifactError> {
    if present(path).map_err(|error| write_error(path, error))? {
        return Err(RouterArtifactError::AlreadyExists {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn stage_dir(dir: &Path) -> Result<PathBuf, RouterArtifactError> {
    let parent = dir.join(STAGING_DIR);
    ensure_dir(&parent)?;
    for _ in 0..32 {
        let sequence = STAGE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| write_error(&parent, error))?
            .as_nanos();
        let stage = parent.join(format!("{}-{nonce}-{sequence}", std::process::id()));
        match std::fs::create_dir(&stage) {
            Ok(()) => {
                sync_dir(&parent)?;
                return Ok(stage);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(write_error(&stage, error)),
        }
    }
    Err(write_error(
        &parent,
        "could not allocate a unique staging directory",
    ))
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), RouterArtifactError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| write_error(path, error))?;
    file.write_all(bytes)
        .map_err(|error| write_error(path, error))?;
    sync_file(&file, path)
}

fn publish_link(source: &Path, destination: &Path) -> Result<(), RouterArtifactError> {
    std::fs::hard_link(source, destination).map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            RouterArtifactError::AlreadyExists {
                path: destination.to_path_buf(),
            }
        } else {
            write_error(destination, error)
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Staged,
    GatePublished,
    ManifestPublished,
    #[cfg(any(all(test, unix), feature = "mixture", feature = "metal-gpu"))]
    ReceiptPublished,
    #[cfg(any(all(test, unix), feature = "mixture", feature = "metal-gpu"))]
    BeforeHeadRename,
    #[cfg(any(all(test, unix), feature = "mixture", feature = "metal-gpu"))]
    HeadRenamed,
}

fn stage_artifact(dir: &Path, artifact: &RouterArtifact) -> Result<PathBuf, RouterArtifactError> {
    vacant(&gate_path(dir, artifact.version))?;
    vacant(&manifest_path(dir, artifact.version))?;
    vacant(&receipt_path(dir, artifact.version))?;
    let json = manifest_bytes(artifact, &manifest_path(dir, artifact.version))?;
    let stage = stage_dir(dir)?;
    write_synced(&gate_path(&stage, artifact.version), &artifact.gate_bytes)?;
    write_synced(&manifest_path(&stage, artifact.version), &json)?;
    sync_dir(&stage)?;
    let stored = read_artifact(&stage, artifact.version)?;
    if stored != *artifact {
        return Err(write_error(
            &stage,
            "staged artifact does not match its source",
        ));
    }
    Ok(stage)
}

fn materialize(
    dir: &Path,
    stage: &Path,
    artifact: &RouterArtifact,
    fault: &mut impl FnMut(Phase) -> Result<(), RouterArtifactError>,
) -> Result<(), RouterArtifactError> {
    fault(Phase::Staged)?;
    publish_link(
        &gate_path(stage, artifact.version),
        &gate_path(dir, artifact.version),
    )?;
    fault(Phase::GatePublished)?;
    publish_link(
        &manifest_path(stage, artifact.version),
        &manifest_path(dir, artifact.version),
    )?;
    sync_dir(dir)?;
    fault(Phase::ManifestPublished)
}

pub(super) fn publish_legacy(
    dir: &Path,
    artifact: &RouterArtifact,
) -> Result<PathBuf, RouterArtifactError> {
    if artifact.provenance.format() != ROUTER_ARTIFACT_FORMAT_LEGACY {
        return Err(RouterArtifactError::PublicationAuthorityRequired {
            identity: identity(artifact),
        });
    }
    let _lock = writer_lock(dir)?;
    if read_head_unlocked(dir)?.is_some_and(|head| head.version == artifact.version) {
        return Err(RouterArtifactError::AlreadyExists {
            path: head_path(dir),
        });
    }
    let stage = stage_artifact(dir, artifact)?;
    materialize(dir, &stage, artifact, &mut |_| Ok(()))?;
    let _ = std::fs::remove_dir_all(stage);
    Ok(manifest_path(dir, artifact.version))
}

fn identity(artifact: &RouterArtifact) -> String {
    format!("{}:{}", artifact.version, artifact.content_hash())
}

fn receipt_path(dir: &Path, version: u64) -> PathBuf {
    dir.join(RECEIPT_DIR).join(format!("router-{version}.json"))
}

fn read_receipt(dir: &Path, version: u64) -> Result<Option<HeadRecord>, RouterArtifactError> {
    let container = dir.join(RECEIPT_DIR);
    if present(&container).map_err(|error| write_error(&container, error))?
        && !std::fs::symlink_metadata(&container)
            .map_err(|error| write_error(&container, error))?
            .is_dir()
    {
        return Err(write_error(
            &container,
            "the receipt namespace must be a directory, not a symlink",
        ));
    }
    let path = receipt_path(dir, version);
    if !present(&path).map_err(|error| write_error(&path, error))? {
        return Ok(None);
    }
    let metadata = std::fs::symlink_metadata(&path).map_err(|error| write_error(&path, error))?;
    if !metadata.is_file() {
        return Err(write_error(
            &path,
            "a committed receipt must be a regular file",
        ));
    }
    let bytes = read_bytes_bounded(&path, MAX_ROUTER_HEAD_LEN).map_err(|error| {
        RouterArtifactError::Read {
            path: path.clone(),
            source: error.into(),
        }
    })?;
    let receipt: HeadRecord =
        serde_json::from_slice(&bytes).map_err(|error| RouterArtifactError::Malformed {
            path: path.clone(),
            message: error.to_string(),
        })?;
    receipt.validate(&path)?;
    if receipt.version != version {
        return Err(RouterArtifactError::VersionMismatch {
            path,
            declared: receipt.version,
            from_filename: version,
        });
    }
    Ok(Some(receipt))
}

pub(super) fn require_commit_proof(
    dir: &Path,
    artifact: &RouterArtifact,
) -> Result<(), RouterArtifactError> {
    let expected = HeadRecord {
        format: ROUTER_ARTIFACT_FORMAT,
        version: artifact.version,
        content_sha256: artifact.content_hash(),
    };
    let head = read_head_unlocked(dir);
    if matches!(&head, Ok(Some(current)) if current == &expected) {
        return Ok(());
    }
    match read_receipt(dir, artifact.version) {
        Ok(Some(receipt)) if receipt == expected => Ok(()),
        result => {
            let problem = match result {
                Ok(Some(_)) => "the historical receipt names another identity".to_string(),
                Ok(None) => match head {
                    Err(error) => {
                        format!("HEAD is unusable ({error}) and the historical receipt is absent")
                    }
                    _ => "HEAD does not name this identity and the historical receipt is absent"
                        .to_string(),
                },
                Err(error) => format!("the historical receipt is unusable ({error})"),
            };
            Err(RouterArtifactError::MissingCommitProof {
                identity: identity(artifact),
                problem,
            })
        }
    }
}

#[cfg(any(all(test, unix), feature = "mixture", feature = "metal-gpu"))]
fn record_for(artifact: &RouterArtifact) -> HeadRecord {
    HeadRecord {
        format: artifact.provenance.format(),
        version: artifact.version,
        content_sha256: artifact.content_hash(),
    }
}

#[cfg(any(all(test, unix), feature = "mixture", feature = "metal-gpu"))]
fn checked_head(
    dir: &Path,
    expected: Option<&HeadRecord>,
) -> Result<Option<HeadRecord>, RouterArtifactError> {
    let actual = read_head_unlocked(dir)?;
    if actual.as_ref() != expected {
        return Err(RouterArtifactError::StaleHead {
            expected: expected.cloned(),
            actual,
        });
    }
    if let Some(head) = &actual {
        let artifact = read_artifact(dir, head.version)?;
        head.check_artifact(&head_path(dir), &artifact)?;
    }
    Ok(actual)
}

#[cfg(any(all(test, unix), feature = "mixture", feature = "metal-gpu"))]
fn preserve_receipt(
    dir: &Path,
    stage: &Path,
    previous: &HeadRecord,
) -> Result<(), RouterArtifactError> {
    let path = receipt_path(dir, previous.version);
    if let Some(receipt) = read_receipt(dir, previous.version)? {
        if receipt != *previous {
            return Err(write_error(
                &path,
                "existing receipt disagrees with the previous committed HEAD",
            ));
        }
        let file = File::open(&path).map_err(|error| write_error(&path, error))?;
        sync_file(&file, &path)?;
        sync_dir(&dir.join(RECEIPT_DIR))?;
        return sync_dir(dir);
    }
    ensure_dir(&dir.join(RECEIPT_DIR))?;
    let staged = stage.join("previous-head.json");
    let bytes = serde_json::to_vec_pretty(previous).map_err(|error| write_error(&staged, error))?;
    write_synced(&staged, &bytes)?;
    sync_dir(stage)?;
    publish_link(&staged, &path)?;
    sync_dir(&dir.join(RECEIPT_DIR))
}

#[cfg(any(all(test, unix), feature = "mixture", feature = "metal-gpu"))]
fn advance_head(
    dir: &Path,
    stage: &Path,
    artifact: &RouterArtifact,
    previous: Option<&HeadRecord>,
    fault: &mut impl FnMut(Phase) -> Result<(), RouterArtifactError>,
) -> Result<PathBuf, RouterArtifactError> {
    let head = record_for(artifact);
    head.validate(&head_path(dir))?;
    let stored = match read_artifact(dir, artifact.version) {
        Err(RouterArtifactError::NotFound { version, .. }) => {
            return Err(RouterArtifactError::HeadNamesAbsentVersion {
                path: head_path(dir),
                version,
            });
        }
        other => other?,
    };
    head.check_artifact(&head_path(dir), &stored)?;
    if let Some(previous) = previous {
        preserve_receipt(dir, stage, previous)?;
        fault(Phase::ReceiptPublished)?;
    }
    let staged = stage.join(ROUTER_HEAD_FILE);
    let bytes = serde_json::to_vec_pretty(&head).map_err(|error| write_error(&staged, error))?;
    write_synced(&staged, &bytes)?;
    sync_dir(stage)?;
    fault(Phase::BeforeHeadRename)?;
    std::fs::rename(&staged, head_path(dir))
        .map_err(|error| write_error(&head_path(dir), error))?;
    let durable = fault(Phase::HeadRenamed).and_then(|()| sync_dir(dir));
    if let Err(error) = durable {
        return Err(RouterArtifactError::CommitUncertain {
            identity: identity(artifact),
            message: error.to_string(),
        });
    }
    Ok(manifest_path(dir, artifact.version))
}

#[cfg(any(all(test, unix), feature = "mixture", feature = "metal-gpu"))]
fn publish_recorded_locked(
    dir: &Path,
    artifact: &RouterArtifact,
    expected: Option<&HeadRecord>,
    fault: &mut impl FnMut(Phase) -> Result<(), RouterArtifactError>,
) -> Result<PathBuf, RouterArtifactError> {
    record_for(artifact).validate(&head_path(dir))?;
    let previous = checked_head(dir, expected)?;
    let stage = stage_artifact(dir, artifact)?;
    materialize(dir, &stage, artifact, fault)?;
    let path = advance_head(dir, &stage, artifact, previous.as_ref(), fault)?;
    let _ = std::fs::remove_dir_all(stage);
    Ok(path)
}

#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
pub(crate) fn publish_bootstrap(
    dir: &Path,
    spec: &crate::router_init::BootstrapSpec,
) -> Result<PathBuf, crate::router_init::RouterInitError> {
    use crate::router_init::{RouterInitError, bootstrap_artifact};
    let artifact = bootstrap_artifact(spec)?;
    let _lock = writer_lock(dir)?;
    let present = versions(dir)?;
    if !present.is_empty() {
        return Err(RouterInitError::Occupied {
            dir: dir.to_path_buf(),
            present,
        });
    }
    // Even an incomplete old publication is occupied. Retrying into a fresh
    // directory preserves its evidence instead of guessing a crash phase.
    for entry in std::fs::read_dir(dir).map_err(|error| write_error(dir, error))? {
        let entry = entry.map_err(|error| write_error(dir, error))?;
        if entry.file_name() != LOCK_FILE {
            return Err(RouterInitError::Artifact(
                RouterArtifactError::AlreadyExists { path: entry.path() },
            ));
        }
    }
    Ok(publish_recorded_locked(dir, &artifact, None, &mut |_| {
        Ok(())
    })?)
}

#[cfg(all(test, unix))]
pub(super) fn commit_fixture(
    dir: &Path,
    artifact: &RouterArtifact,
) -> Result<PathBuf, RouterArtifactError> {
    record_for(artifact).validate(&head_path(dir))?;
    let _lock = writer_lock(dir)?;
    let previous = read_head_unlocked(dir)?;
    let previous = checked_head(dir, previous.as_ref())?;
    let stage = stage_dir(dir)?;
    let path = advance_head(dir, &stage, artifact, previous.as_ref(), &mut |_| Ok(()))?;
    let _ = std::fs::remove_dir_all(stage);
    Ok(path)
}

#[cfg(all(test, unix))]
mod tests {
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
    fn both_existing_and_new_directories_sync_every_canonical_ancestor() {
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
            let expected: Vec<_> = canonical.ancestors().map(Path::to_path_buf).collect();
            assert_eq!(observed, expected);
            assert_eq!(observed.last().unwrap(), Path::new("/"));

            let parent = canonical.parent().unwrap();
            let mut attempted = Vec::new();
            let error = ensure_dir_with_sync(&dir, |path| {
                attempted.push(path.to_path_buf());
                if path == parent {
                    return Err(write_error(path, "injected parent synchronization failure"));
                }
                sync_dir(path)
            })
            .unwrap_err();
            assert!(matches!(error, RouterArtifactError::Write { path, .. } if path == parent));
            assert_eq!(attempted, expected[..2]);
            let mut retry = Vec::new();
            ensure_dir_with_sync(&dir, |path| {
                retry.push(path.to_path_buf());
                sync_dir(path)
            })
            .unwrap();
            assert_eq!(retry, expected);
        }
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
            let expected: Vec<_> = canonical.ancestors().map(Path::to_path_buf).collect();
            assert_eq!(observed, expected);
            assert_eq!(observed.last().unwrap(), Path::new("/"));
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
}

#[cfg(all(test, not(unix)))]
mod unsupported_tests {
    use super::*;

    #[test]
    fn publication_refuses_unsupported_directory_durability() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path().join("uncreated");
        assert!(matches!(
            writer_lock(&dir),
            Err(RouterArtifactError::Write { .. })
        ));
        assert!(!dir.exists());
    }
}
