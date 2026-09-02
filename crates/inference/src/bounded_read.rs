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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;

    #[test]
    fn read_bytes_bounded_rejects_a_directory() {
        let directory = tempfile::tempdir().unwrap();
        let err = read_bytes_bounded(directory.path(), 1024)
            .expect_err("a directory must be rejected as a regular file");
        assert!(matches!(err, BoundedReadError::NotRegularFile));
    }

    #[cfg(unix)]
    #[test]
    fn read_bytes_bounded_rejects_a_fifo_without_blocking() {
        use std::os::unix::ffi::OsStrExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input");
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: `c_path` is a valid, NUL-terminated path for the duration of the call.
        let result = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(result, 0, "mkfifo failed: {}", io::Error::last_os_error());

        let err = read_bytes_bounded(&path, 1024)
            .expect_err("a FIFO must be rejected without opening a writer");
        assert!(matches!(err, BoundedReadError::NotRegularFile));
    }

    #[test]
    fn read_bytes_bounded_rejects_a_sparse_file_over_the_cap() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input");
        let file = File::create(&path).unwrap();
        file.set_len(1025).unwrap();

        let err = read_bytes_bounded(&path, 1024).expect_err("cap + 1 must be rejected");
        assert!(matches!(
            err,
            BoundedReadError::TooLarge {
                len: 1025,
                cap: 1024
            }
        ));
    }

    #[test]
    fn read_bytes_bounded_accepts_a_file_at_the_cap() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input");
        let mut file = File::create(&path).unwrap();
        file.write_all(&[b'x'; 1024]).unwrap();

        let bytes = read_bytes_bounded(&path, 1024).expect("a file at the cap must be accepted");
        assert_eq!(bytes, vec![b'x'; 1024]);
    }

    #[test]
    fn read_text_bounded_rejects_invalid_utf8() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input");
        fs::write(&path, [0xff]).unwrap();

        let err = read_text_bounded(&path, 1024).expect_err("invalid UTF-8 must be rejected");
        match err {
            BoundedReadError::Io(error) => assert_eq!(error.kind(), io::ErrorKind::InvalidData),
            other => panic!("wrong error variant: {other:?}"),
        }
    }
}
