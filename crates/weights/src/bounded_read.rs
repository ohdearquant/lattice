use std::fs::OpenOptions;
use std::io::{self, Read};
use std::path::Path;

/// Failure classes produced by regular-file opening and bounded reads.
#[derive(Debug)]
pub enum BoundedReadError {
    /// The opened path does not identify a regular file.
    NotRegularFile,
    /// The file exceeded the requested byte cap.
    TooLarge { len: u64, cap: u64 },
    /// An operating-system or decoding error occurred.
    Io(io::Error),
}

/// Open a file for reading and verify that the opened handle identifies a regular file.
///
/// On Unix the open uses `O_NONBLOCK` before inspecting the handle. On other
/// platforms opening a special path may block before its type can be checked.
///
/// # Errors
/// Returns [`BoundedReadError::NotRegularFile`] for an opened non-regular file,
/// or [`BoundedReadError::Io`] when opening or inspecting the handle fails.
pub fn open_regular_file(path: &Path) -> Result<std::fs::File, BoundedReadError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(BoundedReadError::Io)?;
    let metadata = file.metadata().map_err(BoundedReadError::Io)?;
    if !metadata.is_file() {
        return Err(BoundedReadError::NotRegularFile);
    }
    Ok(file)
}

/// Read a regular file, bounded to `cap` bytes.
///
/// Uses [`open_regular_file`] and its platform-specific opening guarantees.
pub fn read_bytes_bounded(path: &Path, cap: u64) -> Result<Vec<u8>, BoundedReadError> {
    let file = open_regular_file(path)?;
    let file_len = file.metadata().map_err(BoundedReadError::Io)?.len();
    if file_len > cap {
        return Err(BoundedReadError::TooLarge { len: file_len, cap });
    }

    let mut buf = Vec::new();
    file.take(cap.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(BoundedReadError::Io)?;
    if buf.len() as u64 > cap {
        return Err(BoundedReadError::TooLarge {
            len: buf.len() as u64,
            cap,
        });
    }
    Ok(buf)
}

/// Read bounded file bytes and decode them as UTF-8.
pub fn read_text_bounded(path: &Path, cap: u64) -> Result<String, BoundedReadError> {
    let bytes = read_bytes_bounded(path, cap)?;
    String::from_utf8(bytes)
        .map_err(|error| BoundedReadError::Io(io::Error::new(io::ErrorKind::InvalidData, error)))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::FileTypeExt;

    #[test]
    fn open_regular_file_accepts_regular_and_refuses_fifo() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "lattice-open-regular-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).unwrap();
        let regular = dir.join("regular");
        std::fs::write(&regular, b"ordinary file").unwrap();
        let mut file = open_regular_file(&regular).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"ordinary file");
        drop(file);
        assert!(matches!(
            open_regular_file(&dir),
            Err(BoundedReadError::NotRegularFile)
        ));

        let fifo = dir.join("pipe");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo is required for the Unix opener fixture");
        assert!(status.success(), "mkfifo failed: {status}");
        assert!(
            std::fs::symlink_metadata(&fifo)
                .unwrap()
                .file_type()
                .is_fifo()
        );
        assert!(matches!(
            open_regular_file(&fifo),
            Err(BoundedReadError::NotRegularFile)
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
