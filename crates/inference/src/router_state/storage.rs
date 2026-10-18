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
mod sync_probe;

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
    let missing = match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => false,
        Ok(_) => return Err(write_error(path, "not a directory")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
        Err(error) => return Err(write_error(path, error)),
    };
    let creation_boundary = if missing {
        let mut ancestor = path.parent().unwrap_or_else(|| Path::new("."));
        let boundary = loop {
            if ancestor.as_os_str().is_empty() {
                ancestor = Path::new(".");
            }
            match std::fs::metadata(ancestor) {
                Ok(metadata) if metadata.is_dir() => {
                    break std::fs::canonicalize(ancestor)
                        .map_err(|error| write_error(ancestor, error))?;
                }
                Ok(_) => return Err(write_error(ancestor, "not a directory")),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    if ancestor == Path::new(".") || ancestor.parent().is_none() {
                        return Err(write_error(ancestor, error));
                    }
                    ancestor = ancestor.parent().unwrap_or_else(|| Path::new("."));
                }
                Err(error) => return Err(write_error(ancestor, error)),
            }
        };
        std::fs::create_dir_all(path).map_err(|error| write_error(path, error))?;
        Some(boundary)
    } else {
        None
    };
    if !std::fs::symlink_metadata(path)
        .map_err(|error| write_error(path, error))?
        .is_dir()
    {
        return Err(write_error(path, "not a directory"));
    }
    let canonical = std::fs::canonicalize(path).map_err(|error| write_error(path, error))?;
    // An existing target may have survived a failed parent sync. Its parent
    // repairs that name; older pre-existing ancestry is outside this boundary.
    let boundary =
        creation_boundary.unwrap_or_else(|| canonical.parent().unwrap_or(&canonical).to_path_buf());
    if !canonical.starts_with(&boundary) {
        return Err(write_error(path, "directory escaped its creation boundary"));
    }
    for ancestor in canonical.ancestors() {
        sync(ancestor)?;
        if ancestor == boundary {
            break;
        }
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
    let durable = sync_dir(dir).and_then(|()| fault(Phase::ManifestPublished));
    durable.map_err(|error| {
        if artifact.provenance.format() == ROUTER_ARTIFACT_FORMAT_LEGACY {
            // Counter selection can see both legacy names before this sync.
            RouterArtifactError::CommitUncertain {
                identity: identity(artifact),
                message: error.to_string(),
            }
        } else {
            error
        }
    })
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
mod tests;

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
