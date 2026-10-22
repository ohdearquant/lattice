use crate::weights::bounded_read::{BoundedReadError, read_bytes_bounded, read_text_bounded};
use std::io;

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
