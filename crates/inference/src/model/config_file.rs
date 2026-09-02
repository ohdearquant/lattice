//! Bounded, identity-checked reading of checkpoint config files such as `config.json` and `preprocessor_config.json`.

use crate::error::InferenceError;
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Upper bound, in bytes, on a checkpoint configuration file accepted from a checkpoint directory (FIX 5).
///
/// `from_config_json` / `from_config_json_validated` (both independent `read_to_string`
/// call sites) previously materialized the entire file into a `String` with no size limit,
/// and `serde_json`'s `#[serde(default)]`/passthrough deserialization accepts unknown
/// fields rather than rejecting them -- so an arbitrarily large ignored top-level field
/// (e.g. a multi-gigabyte junk string under an unused key) exhausts memory before any
/// bounded field-level validation in `validate()` ever runs. Real Qwen3.5/3.6 `config.json`
/// files, including ones with a nested `vision_config`, are well under 100 KiB; 8 MiB
/// (8,388,608) leaves nearly two orders of magnitude of headroom while rejecting the
/// unbounded case.
pub(crate) const MAX_CONFIG_JSON_BYTES: u64 = 8_388_608;

/// Read a named config file into a `String`, rejecting an oversized file before it is
/// materialized. The identity and size checks are made on the open handle, and the read
/// is bounded by that same handle, so a path swapped between check and read cannot bypass
/// the cap. See [`MAX_CONFIG_JSON_BYTES`] docs. The size-cap fires before `read_to_string`
/// allocates a same-sized buffer -- admission-order applies to file parsing, not just
/// tensor bytes (mirrors the safetensors index cap in `weights/f32_weights.rs`).
pub(crate) fn read_config_json_bounded(path: &Path, what: &str) -> Result<String, InferenceError> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NONBLOCK);
    let file = options.open(path).map_err(InferenceError::Io)?;
    let metadata = file.metadata().map_err(InferenceError::Io)?;
    if !metadata.is_file() {
        return Err(InferenceError::Inference(format!(
            "{what} at {} is not a regular file",
            path.display()
        )));
    }
    let file_len = metadata.len();
    if file_len > MAX_CONFIG_JSON_BYTES {
        return Err(InferenceError::Inference(format!(
            "{what} at {} is {file_len} bytes, exceeding MAX_CONFIG_JSON_BYTES \
             ({MAX_CONFIG_JSON_BYTES})",
            path.display()
        )));
    }
    let mut raw = String::new();
    file.take(MAX_CONFIG_JSON_BYTES + 1)
        .read_to_string(&mut raw)
        .map_err(InferenceError::Io)?;
    if raw.len() > MAX_CONFIG_JSON_BYTES as usize {
        return Err(InferenceError::Inference(format!(
            "{what} at {} is exceeding MAX_CONFIG_JSON_BYTES ({MAX_CONFIG_JSON_BYTES})",
            path.display()
        )));
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_json_reader_rejects_a_directory() {
        let directory = tempfile::tempdir().unwrap();
        let err = read_config_json_bounded(directory.path(), "config.json")
            .expect_err("a directory must be rejected as a config file");
        assert!(
            err.to_string().contains("is not a regular file"),
            "wrong error: {err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn config_json_reader_rejects_a_fifo_without_blocking() {
        use std::os::unix::ffi::OsStrExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.json");
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: `c_path` is a valid, NUL-terminated path for the duration of the call.
        let result = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(
            result,
            0,
            "mkfifo failed: {}",
            std::io::Error::last_os_error()
        );

        let err = read_config_json_bounded(&path, "config.json")
            .expect_err("a FIFO must be rejected as a config file");
        assert!(
            err.to_string().contains("is not a regular file"),
            "wrong error: {err}"
        );
    }
}
