//! Bounded, identity-checked reading of checkpoint config files such as `config.json` and `preprocessor_config.json`.

use crate::bounded_read::{BoundedReadError, read_text_bounded};
use crate::error::InferenceError;
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
/// the cap. See [`MAX_CONFIG_JSON_BYTES`] docs.
pub(crate) fn read_config_json_bounded(path: &Path, what: &str) -> Result<String, InferenceError> {
    read_text_bounded(path, MAX_CONFIG_JSON_BYTES).map_err(|error| match error {
        BoundedReadError::NotRegularFile => InferenceError::Inference(format!(
            "{what} at {} is not a regular file",
            path.display()
        )),
        BoundedReadError::TooLarge { len, cap } => InferenceError::Inference(format!(
            "{what} at {} is {len} bytes, exceeding MAX_CONFIG_JSON_BYTES ({cap})",
            path.display()
        )),
        BoundedReadError::Io(error) => InferenceError::Io(error),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_json_reader_maps_a_directory_error() {
        let directory = tempfile::tempdir().unwrap();
        let err = read_config_json_bounded(directory.path(), "config.json")
            .expect_err("a directory must be rejected as a config file");
        assert!(
            err.to_string().contains("is not a regular file"),
            "wrong error: {err}"
        );
    }
}
