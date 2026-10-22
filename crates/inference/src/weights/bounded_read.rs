use std::fs::OpenOptions;
use std::io::{self, Read};
use std::path::Path;

/// Failure classes produced by the bounded file reader.
#[derive(Debug)]
pub(crate) enum BoundedReadError {
    /// The opened path does not identify a regular file.
    NotRegularFile,
    /// The file exceeded the requested byte cap.
    TooLarge { len: u64, cap: u64 },
    /// An operating-system or decoding error occurred.
    Io(io::Error),
}

/// Read a regular file, bounded to `cap` bytes.
///
/// On Unix the handle is opened with `O_NONBLOCK`, so a FIFO or device at
/// `path` fails instead of hanging the caller. On other platforms only the
/// regular-file check after `open` guards against special paths; opening one
/// there may block, and the non-blocking guarantee is not claimed.
pub(crate) fn read_bytes_bounded(path: &Path, cap: u64) -> Result<Vec<u8>, BoundedReadError> {
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
    let file_len = metadata.len();
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
pub(crate) fn read_text_bounded(path: &Path, cap: u64) -> Result<String, BoundedReadError> {
    let bytes = read_bytes_bounded(path, cap)?;
    String::from_utf8(bytes)
        .map_err(|error| BoundedReadError::Io(io::Error::new(io::ErrorKind::InvalidData, error)))
}
