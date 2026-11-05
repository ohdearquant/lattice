use std::path::{Path, PathBuf};

const TEMP_ATTEMPTS: usize = 64;

pub(super) struct StagingFile {
    pub(super) path: PathBuf,
    _lock: std::fs::File,
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

pub(super) fn create_target(path: &Path) -> std::io::Result<(StagingFile, std::fs::File)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

    let filename = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "staging target has no filename",
        )
    })?;
    let lock = lock_target(path)?;
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
                        _lock: lock,
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

fn lock_target(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o644);
    }
    let lock = options.open(lock_path(path)?)?;
    lock.lock()?;
    // Keep the inode: unlinking it would let new waiters lock a different file.
    Ok(lock)
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
