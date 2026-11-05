use std::path::{Path, PathBuf};

const TEMP_ATTEMPTS: usize = 64;

pub(super) struct StagingFile {
    pub(super) path: PathBuf,
    _lock: TargetLock,
}

pub(super) struct TargetLock {
    target: PathBuf,
    _file: std::fs::File,
}

impl StagingFile {
    #[cfg(all(feature = "download", not(target_arch = "wasm32")))]
    pub(super) fn replace(self, target: &Path) -> std::io::Result<()> {
        std::fs::rename(&self.path, target)
    }

    pub(super) fn publish_noclobber(self, target: &Path) -> std::io::Result<()> {
        std::fs::hard_link(&self.path, target)
    }
}

impl Drop for StagingFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl TargetLock {
    pub(super) fn create_stage(self) -> std::io::Result<(StagingFile, std::fs::File)> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

        let path = &self.target;
        let filename = path.file_name().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "staging target has no filename",
            )
        })?;
        sweep_stages(path)?;
        for _ in 0..TEMP_ATTEMPTS {
            let sequence = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
            let mut name = std::ffi::OsString::from(".");
            name.push(filename);
            name.push(format!(".part-{}-{sequence}", std::process::id()));
            let stage = path.with_file_name(name);
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o644);
            }
            match options.open(&stage) {
                Ok(file) => {
                    return Ok((
                        StagingFile {
                            path: stage,
                            _lock: self,
                        },
                        file,
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique staging file",
        ))
    }
}

#[cfg(all(test, feature = "download", not(target_arch = "wasm32")))]
pub(super) fn create_target(path: &Path) -> std::io::Result<(StagingFile, std::fs::File)> {
    lock_target(path)?.create_stage()
}

pub(super) fn lock_path(path: &Path) -> std::io::Result<PathBuf> {
    let filename = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "staging target has no filename",
        )
    })?;
    let mut name = std::ffi::OsString::from(".");
    name.push(filename);
    name.push(".lock");
    Ok(path.with_file_name(name))
}

pub(super) fn lock_target(path: &Path) -> std::io::Result<TargetLock> {
    let lock_path = lock_path(path)?;
    let nonregular = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "staging lock is not a regular file: {}",
                lock_path.display()
            ),
        )
    };
    match std::fs::symlink_metadata(&lock_path) {
        Ok(metadata) if !metadata.is_file() => return Err(nonregular()),
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o644);
    }
    let lock = options.open(&lock_path)?;
    if !lock.metadata()?.is_file() {
        return Err(nonregular());
    }
    #[cfg(all(test, not(target_arch = "wasm32")))]
    observe_lock_attempt(&lock)?;
    lock.lock()?;
    // Keep the inode: unlinking it would let new waiters lock a different file.
    Ok(TargetLock {
        target: path.to_owned(),
        _file: lock,
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    static NEXT_LOCK_OBSERVER: std::cell::RefCell<Option<Box<dyn FnOnce(bool)>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn observe_next_lock(observer: impl FnOnce(bool) + 'static) {
    NEXT_LOCK_OBSERVER.with(|slot| *slot.borrow_mut() = Some(Box::new(observer)));
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn observe_lock_attempt(file: &std::fs::File) -> std::io::Result<()> {
    let observer = NEXT_LOCK_OBSERVER.with(|slot| slot.borrow_mut().take());
    if let Some(observer) = observer {
        let contended = match file.try_lock() {
            Ok(()) => {
                file.unlock()?;
                false
            }
            Err(std::fs::TryLockError::WouldBlock) => true,
            Err(std::fs::TryLockError::Error(error)) => return Err(error),
        };
        observer(contended);
    }
    Ok(())
}

fn sweep_stages(path: &Path) -> std::io::Result<()> {
    let filename = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "staging target has no filename",
        )
    })?;
    let mut prefix = std::ffi::OsString::from(".");
    prefix.push(filename);
    prefix.push(".part-");
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let legacy = path.with_extension("part");
    for entry in std::fs::read_dir(parent)? {
        let entry = entry?;
        let name = entry.file_name();
        let unique_stage = name
            .as_encoded_bytes()
            .strip_prefix(prefix.as_encoded_bytes())
            .is_some_and(|suffix| {
                let mut numbers = suffix.split(|byte| *byte == b'-');
                let numeric = |part: Option<&[u8]>| {
                    part.is_some_and(|part| !part.is_empty() && part.iter().all(u8::is_ascii_digit))
                };
                numeric(numbers.next()) && numeric(numbers.next()) && numbers.next().is_none()
            });
        let is_legacy =
            name.as_os_str() == legacy.file_name().unwrap_or_default() && legacy != path;
        if unique_stage || is_legacy {
            // remove_file unlinks a symlink itself and never traverses its target.
            match std::fs::remove_file(entry.path()) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}
