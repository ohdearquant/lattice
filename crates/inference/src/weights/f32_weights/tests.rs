use super::*;
use std::fs;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// RAII guard for a scratch file under the OS temp directory: removes the
/// file when dropped, including when the drop runs during an unwinding
/// panic. Derefs to `Path` so call sites read like they hold a `PathBuf`.
///
/// Call [`TempFileGuard::into_path`] when a test genuinely needs the file
/// to outlive the guard (e.g. for manual post-mortem inspection).
struct TempFileGuard(std::path::PathBuf);

impl TempFileGuard {
    fn into_path(self) -> std::path::PathBuf {
        let path = self.0.clone();
        std::mem::forget(self);
        path
    }
}

impl std::ops::Deref for TempFileGuard {
    type Target = std::path::Path;

    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

impl AsRef<std::path::Path> for TempFileGuard {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// RAII guard for a scratch directory under the OS temp directory: removes
/// the directory tree when dropped, including when the drop runs during an
/// unwinding panic. Derefs to `Path` so call sites read like they hold a
/// `PathBuf`.
///
/// Call [`TempDirGuard::into_path`] when a test genuinely needs the
/// directory to outlive the guard (e.g. for manual post-mortem inspection).
struct TempDirGuard(std::path::PathBuf);

impl TempDirGuard {
    fn into_path(self) -> std::path::PathBuf {
        let path = self.0.clone();
        std::mem::forget(self);
        path
    }
}

impl std::ops::Deref for TempDirGuard {
    type Target = std::path::Path;

    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

impl AsRef<std::path::Path> for TempDirGuard {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn temp_path(name: &str) -> TempFileGuard {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("invariant: system time is after UNIX_EPOCH")
        .as_nanos();
    TempFileGuard(std::env::temp_dir().join(format!(
        "{}_{}_{}.safetensors",
        name,
        std::process::id(),
        nanos
    )))
}

#[test]
fn temp_dir_guard_removes_directory_on_normal_drop() {
    let dir = temp_dir("lattice_guard_drop_test");
    let path = dir.to_path_buf();
    assert!(path.is_dir(), "temp_dir must create the directory eagerly");

    drop(dir);

    assert!(
        !path.exists(),
        "TempDirGuard must remove its directory when dropped normally"
    );
}

#[test]
fn temp_dir_guard_removes_directory_on_unwind() {
    let dir = temp_dir("lattice_guard_unwind_test");
    let path = dir.to_path_buf();
    assert!(path.is_dir(), "temp_dir must create the directory eagerly");

    // Rust runs destructors while a panic unwinds the stack, so moving the
    // guard into a closure that panics -- then catching that panic --
    // exercises exactly the cleanup path a failing/panicking test relies
    // on. This assumes the default `panic = "unwind"` strategy; under
    // `panic = "abort"` this test would abort the whole binary instead of
    // failing quietly, which is the intended, visible failure mode for
    // that profile misconfiguration.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _dir = dir;
        panic!("intentional panic to exercise TempDirGuard unwind cleanup");
    }));

    assert!(result.is_err(), "the inner closure must have panicked");
    assert!(
        !path.exists(),
        "TempDirGuard must remove its directory when its owning frame unwinds"
    );
}

#[test]
fn temp_dir_guard_into_path_opts_out_of_cleanup() {
    let dir = temp_dir("lattice_guard_leak_opt_out_test");
    let path = dir.into_path();

    assert!(
        path.is_dir(),
        "into_path must hand back the still-existing directory without removing it"
    );

    fs::remove_dir_all(&path).ok();
}

#[test]
fn temp_file_guard_removes_file_on_normal_drop() {
    let guard = temp_path("lattice_guard_file_drop_test");
    write_raw_safetensors(&guard, r#"{"__metadata__":{"format":"pt"}}"#, &[]);
    let path = guard.to_path_buf();
    assert!(path.is_file(), "test setup must have created the file");

    drop(guard);

    assert!(
        !path.exists(),
        "TempFileGuard must remove its file when dropped normally"
    );
}

#[test]
fn temp_file_guard_into_path_opts_out_of_cleanup() {
    let guard = temp_path("lattice_guard_file_leak_opt_out_test");
    write_raw_safetensors(&guard, r#"{"__metadata__":{"format":"pt"}}"#, &[]);
    let path = guard.into_path();

    assert!(
        path.is_file(),
        "into_path must hand back the still-existing file without removing it"
    );

    fs::remove_file(&path).ok();
}

#[test]
fn test_parse_small_safetensors_file() {
    let path = temp_path("lattice_weights_test");
    let header = r#"{
            "__metadata__": {"format": "pt"},
            "vec": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "mat": {"dtype": "F32", "shape": [2, 2], "data_offsets": [8, 24]}
        }"#
    .replace(['\n', ' '], "");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    let mut file = File::create(&path).expect("test setup: create safetensors file");
    file.write_all(&bytes)
        .expect("test setup: write safetensors bytes");
    drop(file);

    let st = SafetensorsFile::open(&path).expect("test setup: parse safetensors file");
    let (vec_data, vec_shape) = st
        .get_f32_tensor("vec")
        .expect("test setup: vec tensor exists");
    let (mat_data, mat_shape) = st
        .get_f32_tensor("mat")
        .expect("test setup: mat tensor exists");

    assert_eq!(vec_shape, &[2]);
    assert_eq!(mat_shape, &[2, 2]);
    assert_eq!(vec_data, &[1.0, 2.0]);
    assert_eq!(mat_data, &[3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn duplicate_tensor_name_in_safetensors_header_is_rejected() {
    let header = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    let err = SafetensorsFile::from_bytes(bytes)
        .expect_err("duplicate tensor names must not collapse into the header map");
    assert!(
        err.to_string().contains("duplicate tensor name"),
        "error must identify the duplicate header member: {err}"
    );
}

#[test]
fn duplicate_tensor_object_members_are_rejected() {
    let duplicate_members = [
        r#""dtype":"F32","dtype":"F32","shape":[1],"data_offsets":[0,4]"#,
        r#""dtype":"F32","shape":[1],"shape":[1],"data_offsets":[0,4]"#,
        r#""dtype":"F32","shape":[1],"data_offsets":[0,4],"data_offsets":[0,4]"#,
        r#""dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1,"extra":2"#,
        r#""dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":{"nested":1,"nested":2}"#,
    ];

    for members in duplicate_members {
        let header = format!(r#"{{"tensor":{{{members}}}}}"#);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());

        let err = SafetensorsFile::from_bytes(bytes)
            .expect_err("duplicate keys anywhere in a tensor object must be rejected");
        assert!(
            err.to_string().contains("duplicate"),
            "error must identify the duplicate object member: {err}"
        );
    }
}

#[test]
fn metadata_must_be_a_unique_string_map() {
    let invalid_headers = [
        r#"{"__metadata__":{"source":"a","source":"b"},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"__metadata__":{"source":1},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"__metadata__":[],"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"__metadata__":{},"__metadata__":{},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
    ];

    for header in invalid_headers {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        SafetensorsFile::from_bytes(bytes)
            .expect_err("safetensors metadata must be one unique string-to-string map");
    }
}

#[test]
fn utf8_strings_decode_canonically_before_duplicate_checks() {
    let header = r#"{"__metadata__":{"作者":"café 🚀"},"t\u00e9nsor\ud83d\ude80":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    let parsed = SafetensorsFile::from_bytes(bytes)
        .expect("valid UTF-8 and a valid escaped surrogate pair must parse");
    assert!(
        parsed.has_tensor("ténsor🚀"),
        "raw UTF-8 and escaped surrogate pairs must decode to canonical scalar values"
    );

    let duplicate = r#"{"__metadata__":{"é":"raw","\u00e9":"escaped"},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(duplicate.len() as u64).to_le_bytes());
    bytes.extend_from_slice(duplicate.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    SafetensorsFile::from_bytes(bytes)
        .expect_err("raw and escaped forms of the same metadata key are duplicates");
}

#[test]
fn malformed_json_numbers_are_rejected() {
    let invalid_members = [
        r#""extra":-"#,
        r#""extra":1+2"#,
        r#""extra":01"#,
        r#""extra":-01"#,
        r#""extra":1."#,
        r#""extra":1e"#,
        r#""extra":1e+"#,
        r#""shape":[01]"#,
        r#""data_offsets":[00,4]"#,
    ];

    for invalid in invalid_members {
        let members = match invalid.split_once(':').map(|(key, _)| key) {
            Some(r#""shape""#) => {
                format!(r#""dtype":"F32",{invalid},"data_offsets":[0,4]"#)
            }
            Some(r#""data_offsets""#) => {
                format!(r#""dtype":"F32","shape":[1],{invalid}"#)
            }
            _ => format!(r#""dtype":"F32","shape":[1],"data_offsets":[0,4],{invalid}"#),
        };
        let header = format!(r#"{{"tensor":{{{members}}}}}"#);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());

        assert!(
            SafetensorsFile::from_bytes(bytes).is_err(),
            "invalid JSON number must fail closed: {header}"
        );
    }
}

#[test]
fn valid_json_numbers_in_unknown_values_are_accepted() {
    let header = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":[0,-1,1.5,1e2,-0.25E+2]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    SafetensorsFile::from_bytes(bytes)
        .expect("RFC 8259 numbers in an unknown extension value must remain valid");
}

#[test]
fn malformed_unicode_surrogates_are_rejected() {
    for escaped in [r#"\ud83d"#, r#"\ude80"#, r#"\ud83dX"#] {
        let header = format!(
            r#"{{"__metadata__":{{"value":"{escaped}"}},"tensor":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        SafetensorsFile::from_bytes(bytes)
            .expect_err("a malformed JSON Unicode surrogate must fail closed");
    }
}

#[test]
fn trailing_comma_in_header_object_is_rejected() {
    let header = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    SafetensorsFile::from_bytes(bytes)
        .expect_err("the safetensors header must be valid JSON without a trailing comma");
}

#[test]
fn non_space_bytes_after_top_level_header_are_rejected() {
    let tensor = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let header = format!("{tensor}THIS_IS_NOT_JSON");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    SafetensorsFile::from_bytes(bytes)
        .expect_err("the declared header may contain only space padding after its JSON object");
}

#[test]
fn space_padding_after_top_level_header_is_accepted() {
    let tensor = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let header = format!("{tensor}   ");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    SafetensorsFile::from_bytes(bytes).expect("ASCII space is valid safetensors header padding");
}

// #1368: `SafetensorsFile::open` is a raw-open entry point (used directly
// by the runtime model loaders, not just via `open_manifest_entry_once`)
// that must route through the mmap trust boundary the same as every
// other checkpoint mmap site in this crate. This proves the wiring, not
// just the underlying `reject_if_open_mmap_file_untrusted` predicate
// (already covered directly in `mmap_trust`'s own tests).
#[cfg(unix)]
#[test]
fn open_rejects_a_group_or_other_writable_checkpoint_file() {
    use std::os::unix::fs::PermissionsExt;

    let path = temp_path("lattice_weights_writable_checkpoint");
    let header = r#"{"vec":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    let mut file = File::create(&path).expect("test setup: create safetensors file");
    file.write_all(&bytes)
        .expect("test setup: write safetensors bytes");
    drop(file);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o666)).expect("chmod 0o666");

    let err = SafetensorsFile::open(&path)
        .expect_err("a group/other-writable checkpoint file must be refused");
    assert!(
        matches!(&err, InferenceError::InvalidSafetensors(msg) if msg.contains("refusing to load")),
        "expected a trust-boundary refusal, got: {err:?}"
    );

    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).expect("chmod 0o600");
    SafetensorsFile::open(&path).expect("an owner-only checkpoint file must still be accepted");
}

// #1380: `SafetensorsFile::open` is a symlink-following checkpoint entry
// point (it must follow a HuggingFace hub-cache checkpoint's
// final-component symlink, so it cannot use `open_trusted_mmap_file`'s
// `O_NOFOLLOW`) that used to call plain `File::open` before any
// file-type check ran. A FIFO planted at the checkpoint path -- an
// attacker or a misconfigured deployment -- blocked this call
// indefinitely. This proves the fix by asserting the call returns (any
// return, not just an error) inside a generous deadline, then separately
// asserts it specifically rejected the FIFO as non-regular. `mkfifo`
// (POSIX, not macOS-only) keeps this test portable to the Linux CI legs
// that run this suite.
#[test]
fn open_rejects_a_fifo_without_blocking() {
    let guard = temp_path("lattice_weights_fifo_checkpoint");
    let path = guard.to_path_buf();
    let status = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo must succeed to set up this test");

    let (tx, rx) = std::sync::mpsc::channel();
    let probe_path = path.clone();
    // A pre-fix blocking `File::open` on this FIFO (no writer ever
    // connects) would hang this thread forever; running the call on a
    // detached thread and racing it against a deadline on
    // `rx.recv_timeout` turns "hangs forever" into an observable test
    // failure instead of an actually-hung test process.
    std::thread::spawn(move || {
        let result = SafetensorsFile::open(&probe_path);
        let _ = tx.send(result);
    });
    let result = rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
        "SafetensorsFile::open did not return within 5s -- it blocked on the \
             planted FIFO, meaning the open-time regular-file guard regressed",
    );
    let err = result.expect_err("a FIFO must be rejected, not accepted, for a checkpoint load");
    assert!(
        matches!(&err, InferenceError::InvalidSafetensors(msg) if msg.contains("not a regular file")),
        "expected a not-a-regular-file refusal, got: {err:?}"
    );
}

#[test]
fn test_rejects_shape_byte_length_mismatch() {
    let path = temp_path("lattice_weights_bad_shape");
    let header = r#"{
            "bad": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}
        }"#
    .replace(['\n', ' '], "");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in [1.0f32, 2.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    let mut file = File::create(&path).expect("test setup: create safetensors file");
    file.write_all(&bytes)
        .expect("test setup: write safetensors bytes");
    drop(file);

    let err = SafetensorsFile::open(&path)
        .expect_err("shape byte length mismatch should be rejected at open");
    assert!(
        err.to_string().contains("byte length mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rejects_one_byte_extra_payload() {
    // shape [2] f32 wants exactly 8 bytes; data_offsets declares 9.
    let path = temp_path("lattice_weights_extra_byte");
    let header = r#"{
            "extra": {"dtype": "F32", "shape": [2], "data_offsets": [0, 9]}
        }"#
    .replace(['\n', ' '], "");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in [1.0f32, 2.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.push(0); // one extra byte so the declared 9-byte span exists in the file

    let mut file = File::create(&path).expect("test setup: create safetensors file");
    file.write_all(&bytes)
        .expect("test setup: write safetensors bytes");
    drop(file);

    let err = SafetensorsFile::open(&path)
        .expect_err("one-byte-extra payload should be rejected at open");
    assert!(
        err.to_string().contains("byte length mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rejects_aliased_tensor_ranges() {
    let path = temp_path("lattice_weights_aliased_ranges");
    let header = r#"{
            "key": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "value": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        }"#
    .replace(['\n', ' '], "");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    let mut file = File::create(&path).expect("test setup: create safetensors file");
    file.write_all(&bytes)
        .expect("test setup: write safetensors bytes");
    drop(file);

    let err = SafetensorsFile::open(&path)
        .expect_err("aliased tensor byte ranges must be rejected at open");
    assert!(
        err.to_string().contains("non-contiguous"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rejects_leading_hole_in_data_section() {
    let path = temp_path("lattice_weights_leading_hole");
    let header = r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    write_raw_safetensors(&path, header, &[0; 8]);

    let err = SafetensorsFile::open(&path)
        .expect_err("the first tensor must start at data-section offset zero");
    assert!(
        err.to_string().contains("non-contiguous"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rejects_internal_hole_in_data_section() {
    let path = temp_path("lattice_weights_internal_hole");
    let header = r#"{
            "a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
            "b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}
        }"#
    .replace(['\n', ' '], "");
    write_raw_safetensors(&path, &header, &[0; 12]);

    let err = SafetensorsFile::open(&path)
        .expect_err("tensor ranges must cover the data section contiguously");
    assert!(
        err.to_string().contains("non-contiguous"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rejects_trailing_payload_outside_tensor_ranges() {
    let path = temp_path("lattice_weights_trailing_payload");
    let header = r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    write_raw_safetensors(&path, header, &[0; 8]);

    let err = SafetensorsFile::open(&path)
        .expect_err("tensor ranges must exhaust the complete data section");
    assert!(
        err.to_string().contains("trailing or missing payload"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rejects_reversed_tensor_range() {
    let path = temp_path("lattice_weights_reversed_range");
    let header = r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#;
    write_raw_safetensors(&path, header, &[0; 4]);

    let err =
        SafetensorsFile::open(&path).expect_err("a tensor range must not end before it starts");
    assert!(
        err.to_string().contains("invalid data_offsets"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rejects_header_exceeding_depth_limit() {
    // Nested well past MAX_SAFETENSORS_HEADER_DEPTH, but the whole header is a
    // couple hundred bytes -- far under MAX_SAFETENSORS_HEADER_BYTES, so only
    // the depth bound can be what rejects this fixture.
    let depth = MAX_SAFETENSORS_HEADER_DEPTH + 8;
    let mut header =
        String::from(r#"{"tensor":{"dtype":"F32","shape":[0],"data_offsets":[0,0],"extra":"#);
    header.push_str(&"[".repeat(depth));
    header.push_str(&"]".repeat(depth));
    header.push_str("}}");
    assert!(header.len() < MAX_SAFETENSORS_HEADER_BYTES);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());

    let err = SafetensorsFile::from_bytes(bytes)
        .expect_err("header nesting past the depth limit must be rejected");
    assert!(
        err.to_string().contains("depth limit"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rejects_header_exceeding_size_limit() {
    // A single valid JSON string value padded past MAX_SAFETENSORS_HEADER_BYTES,
    // with the full declared header actually present in the buffer (so the
    // separate "header extends past end of file" check can't be what rejects
    // this instead). A bare string value never recurses through skip_value, so
    // this fixture stays at depth 0 -- only the size bound can reject it.
    let prefix = r#"{"__metadata__":""#;
    let suffix = r#""}"#;
    let pad_len = MAX_SAFETENSORS_HEADER_BYTES + 1024 - prefix.len() - suffix.len();
    let header_len = prefix.len() + pad_len + suffix.len();
    assert!(header_len > MAX_SAFETENSORS_HEADER_BYTES);

    let mut bytes = Vec::with_capacity(8 + header_len);
    bytes.extend_from_slice(&(header_len as u64).to_le_bytes());
    bytes.extend_from_slice(prefix.as_bytes());
    bytes.extend_from_slice(&vec![b'a'; pad_len]);
    bytes.extend_from_slice(suffix.as_bytes());

    let err = SafetensorsFile::from_bytes(bytes)
        .expect_err("a header past the size limit must be rejected");
    assert!(
        err.to_string().contains("exceeds limit"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_parses_header_within_depth_and_size_bounds() {
    // A legitimate small header: under both the depth limit and the size
    // limit. Must still parse, so neither bound above can be read as
    // "reject everything" and still pass the suite.
    let header = r#"{
            "__metadata__": {"format": "pt"},
            "t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
        }"#
    .replace(['\n', ' '], "");
    assert!(header.len() < MAX_SAFETENSORS_HEADER_BYTES);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in [1.0f32, 2.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    let sf =
        SafetensorsFile::from_bytes(bytes).expect("a header within both bounds must still parse");
    let (data, shape) = sf
        .get_f32_tensor("t")
        .expect("tensor within a bounded header must still load");
    assert_eq!(shape, &[2]);
    assert_eq!(data, &[1.0, 2.0]);
}

fn write_raw_safetensors(path: &std::path::Path, header: &str, raw: &[u8]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(raw);
    let mut file = File::create(path).expect("test setup: create safetensors file");
    file.write_all(&bytes)
        .expect("test setup: write safetensors bytes");
}

/// Write a single raw tensor entry (arbitrary dtype/bytes) to a fresh
/// safetensors file, for fixtures that need control over the exact
/// on-disk bit pattern (NaN/Inf bit patterns, non-float dtypes, ...).
fn write_raw_tensor(path: &std::path::Path, name: &str, dtype: &str, shape: &[usize], raw: &[u8]) {
    let shape_str = format!(
        "[{}]",
        shape
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let header = format!(
        r#"{{"{name}":{{"dtype":"{dtype}","shape":{shape_str},"data_offsets":[0,{}]}}}}"#,
        raw.len()
    );
    write_raw_safetensors(path, &header, raw);
}

#[test]
fn test_open_mapped_f32_rejects_nan() {
    let path = temp_path("lattice_weights_f32_nan_mapped");
    let mut raw = Vec::new();
    raw.extend_from_slice(&1.0f32.to_le_bytes());
    raw.extend_from_slice(&f32::NAN.to_le_bytes());
    write_raw_tensor(&path, "t", "F32", &[2], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("NaN in mapped F32 data must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_open_mapped_f32_rejects_positive_infinity() {
    let path = temp_path("lattice_weights_f32_posinf_mapped");
    let mut raw = Vec::new();
    raw.extend_from_slice(&f32::INFINITY.to_le_bytes());
    raw.extend_from_slice(&2.0f32.to_le_bytes());
    write_raw_tensor(&path, "t", "F32", &[2], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("+inf in mapped F32 data must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_open_mapped_f32_rejects_negative_infinity() {
    let path = temp_path("lattice_weights_f32_neginf_mapped");
    let mut raw = Vec::new();
    raw.extend_from_slice(&1.0f32.to_le_bytes());
    raw.extend_from_slice(&f32::NEG_INFINITY.to_le_bytes());
    write_raw_tensor(&path, "t", "F32", &[2], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("-inf in mapped F32 data must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_from_bytes_owned_f32_rejects_non_finite() {
    let header = r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&f32::NAN.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    let sf = SafetensorsFile::from_bytes(bytes).expect("from_bytes: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("NaN via from_bytes (owned backing) must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_get_f32_tensor_caches_validation_after_first_call() {
    let path = temp_path("lattice_weights_validation_cache");
    write_single_f32_tensor(&path, "t", &[1.0, 2.0, 3.0]);

    let sf = SafetensorsFile::open(&path).expect("open: valid file");
    let meta = sf.tensors.get("t").expect("tensor tracked");
    assert!(
        meta.validated.get().is_none(),
        "must not be marked validated before first access"
    );

    sf.get_f32_tensor("t").expect("first access validates");
    assert!(
        meta.validated.get().is_some(),
        "must be marked validated after first access"
    );

    // Second access must not panic or error — it should skip re-scanning.
    sf.get_f32_tensor("t")
        .expect("second access reuses cached validation");
}

fn half_bits_fixture(len: usize, seed: u32, exponent_mask: u16) -> Vec<u16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mut bits = (state >> 16) as u16;
            if bits & exponent_mask == exponent_mask {
                bits ^= exponent_mask & exponent_mask.wrapping_neg();
            }
            bits
        })
        .collect()
}

fn f32_fixture(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mut bits = state;
            if bits & 0x7F80_0000 == 0x7F80_0000 {
                bits ^= 0x0080_0000;
            }
            f32::from_bits(bits)
        })
        .collect()
}

fn le_bytes_of_u16(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn bit_patterns(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Writes a one-tensor file whose payload starts `payload_phase` bytes past a
/// 4-byte boundary of the file start. The space-padded header is what moves the
/// payload, so the same fixture can build both the aligned and the misaligned
/// F32 arm on a page-aligned mapping.
fn write_tensor_with_payload_phase(
    path: &std::path::Path,
    dtype: &str,
    len: usize,
    raw: &[u8],
    payload_phase: usize,
) {
    let base = format!(
        r#"{{"t":{{"dtype":"{dtype}","shape":[{len}],"data_offsets":[0,{}]}}}}"#,
        raw.len()
    );
    let pad = (payload_phase + 4 - (8 + base.len()) % 4) % 4;
    let header = format!("{base}{}", " ".repeat(pad));
    write_raw_safetensors(path, &header, raw);
}

fn converted_is_cached(file: &SafetensorsFile, name: &str) -> bool {
    file.tensors
        .get(name)
        .expect("tensor tracked")
        .converted_f32
        .get()
        .is_some()
}

/// `(label, dtype, element count, payload bytes, payload phase)`.
fn owned_read_cases() -> Vec<(&'static str, &'static str, usize, Vec<u8>, usize)> {
    let bf16 = half_bits_fixture(96, 11, 0x7F80);
    let f16 = half_bits_fixture(96, 23, 0x7C00);
    let f32_values = f32_fixture(96, 37);
    let f32_bytes: Vec<u8> = f32_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    vec![
        ("bf16", "BF16", bf16.len(), le_bytes_of_u16(&bf16), 0),
        ("f16", "F16", f16.len(), le_bytes_of_u16(&f16), 0),
        ("f32_aligned", "F32", f32_values.len(), f32_bytes.clone(), 0),
        ("f32_misaligned", "F32", f32_values.len(), f32_bytes, 2),
    ]
}

#[test]
fn owned_read_hands_over_the_cached_conversion_instead_of_cloning_it() {
    let path = temp_path("lattice_weights_owned_take_bf16");
    let bits = half_bits_fixture(64, 7, 0x7F80);
    write_tensor_with_payload_phase(&path, "BF16", bits.len(), &le_bytes_of_u16(&bits), 0);
    let expected: Vec<f32> = bits
        .iter()
        .map(|&b| crate::weights::half_bits::bf16_bits_to_f32(b))
        .collect();

    let mut owned_only = SafetensorsFile::open(&path).expect("open: valid file");
    let (values, shape) = owned_only
        .get_f32_tensor_owned("t")
        .expect("owned read of a finite BF16 tensor");
    assert_eq!(shape, vec![bits.len()]);
    assert_eq!(bit_patterns(&values), bit_patterns(&expected));
    assert!(
        !converted_is_cached(&owned_only, "t"),
        "an owned read must not leave a second f32 copy alive in the source"
    );

    let mut after_borrow = SafetensorsFile::open(&path).expect("open: valid file");
    let cached_ptr = after_borrow
        .get_f32_tensor("t")
        .expect("borrowed read populates the cache")
        .0
        .as_ptr();
    assert!(converted_is_cached(&after_borrow, "t"));
    let (values, _) = after_borrow
        .get_f32_tensor_owned("t")
        .expect("owned read after a borrowed read");
    assert_eq!(
        values.as_ptr(),
        cached_ptr,
        "the owned vector must be the cached allocation, not a clone of it"
    );
    assert!(!converted_is_cached(&after_borrow, "t"));
}

#[test]
fn owned_read_matches_borrowed_read_bit_for_bit() {
    for (label, dtype, len, raw, phase) in owned_read_cases() {
        let path = temp_path(&format!("lattice_weights_owned_vs_borrowed_{label}"));
        write_tensor_with_payload_phase(&path, dtype, len, &raw, phase);

        let borrowed_file = SafetensorsFile::open(&path).expect("open: valid file");
        let (borrowed, borrowed_shape) = borrowed_file
            .get_f32_tensor("t")
            .expect("borrowed read of a finite tensor");

        let mut owned_file = SafetensorsFile::open(&path).expect("open: valid file");
        let payload = &owned_file.data.as_slice()[owned_file.data_offset..];
        let payload_is_aligned = payload.as_ptr().align_offset(std::mem::align_of::<f32>()) == 0;
        assert_eq!(
            payload_is_aligned,
            phase == 0,
            "{label}: the fixture must produce the payload alignment it names"
        );
        let zero_copy_view = cfg!(target_endian = "little") && dtype == "F32" && payload_is_aligned;
        assert_eq!(
            converted_is_cached(&borrowed_file, "t"),
            !zero_copy_view,
            "{label}: only the aligned F32 view is served without a cached conversion"
        );

        let (owned, owned_shape) = owned_file
            .get_f32_tensor_owned("t")
            .expect("owned read of a finite tensor");
        assert_eq!(owned.len(), len, "{label}");
        assert_eq!(owned_shape.as_slice(), borrowed_shape, "{label}");
        assert_eq!(
            bit_patterns(&owned),
            bit_patterns(borrowed),
            "{label}: owned and borrowed values must agree bit for bit"
        );
        assert!(
            !converted_is_cached(&owned_file, "t"),
            "{label}: no cached conversion may outlive an owned read"
        );
    }
}

#[test]
fn owned_read_of_non_finite_bf16_is_refused_like_the_borrowed_read() {
    let cases = [("nan", 0x7FC0u16), ("pos_inf", 0x7F80), ("neg_inf", 0xFF80)];
    for (label, bad_bits) in cases {
        let path = temp_path(&format!("lattice_weights_owned_non_finite_{label}"));
        let raw = le_bytes_of_u16(&[0x3F80, bad_bits]);
        write_raw_tensor(&path, "t", "BF16", &[2], &raw);

        let borrowed_file = SafetensorsFile::open(&path).expect("open: valid file");
        let borrowed_err = borrowed_file
            .get_f32_tensor("t")
            .expect_err("non-finite BF16 must be refused on the borrowed path");

        let mut owned_file = SafetensorsFile::open(&path).expect("open: valid file");
        let owned_err = owned_file
            .get_f32_tensor_owned("t")
            .expect_err("non-finite BF16 must be refused on the owned path");
        assert!(
            matches!(owned_err, InferenceError::InvalidSafetensors(_)),
            "{label}: unexpected error class: {owned_err:?}"
        );
        assert!(matches!(
            borrowed_err,
            InferenceError::InvalidSafetensors(_)
        ));
        assert_eq!(owned_err.to_string(), borrowed_err.to_string(), "{label}");
        assert!(
            owned_err.to_string().contains("non-finite")
                && owned_err.to_string().contains("element index 1"),
            "{label}: unexpected error: {owned_err}"
        );

        let retry_err = owned_file
            .get_f32_tensor_owned("t")
            .expect_err("a refused tensor stays refused on a second owned read");
        assert_eq!(retry_err.to_string(), borrowed_err.to_string(), "{label}");
        let borrowed_retry = owned_file
            .get_f32_tensor("t")
            .expect_err("a refused tensor stays refused on a later borrowed read");
        assert_eq!(
            borrowed_retry.to_string(),
            borrowed_err.to_string(),
            "{label}"
        );
    }
}

#[test]
fn borrowed_read_after_owned_read_returns_the_same_values_and_keeps_validation() {
    for (label, dtype, len, raw, phase) in owned_read_cases() {
        let path = temp_path(&format!("lattice_weights_borrowed_after_owned_{label}"));
        write_tensor_with_payload_phase(&path, dtype, len, &raw, phase);

        let mut file = SafetensorsFile::open(&path).expect("open: valid file");
        let (owned, owned_shape) = file
            .get_f32_tensor_owned("t")
            .expect("owned read of a finite tensor");
        assert!(
            file.tensors
                .get("t")
                .expect("tensor tracked")
                .validated
                .get()
                .is_some_and(Result::is_ok),
            "{label}: the owned read must leave a passing validation result behind"
        );

        let (borrowed, borrowed_shape) = file
            .get_f32_tensor("t")
            .expect("borrowed read after an owned read");
        assert_eq!(borrowed_shape, owned_shape.as_slice(), "{label}");
        assert_eq!(
            bit_patterns(borrowed),
            bit_patterns(&owned),
            "{label}: a borrowed read after an owned read must see the same values"
        );
    }
}

#[test]
fn sharded_owned_read_hands_over_the_shard_cache_and_keeps_the_refusal() {
    let dir = temp_dir("lattice_sharded_owned_take_test");
    let good_bits = half_bits_fixture(32, 5, 0x7F80);
    write_raw_tensor(
        &dir.join("good.safetensors"),
        "tensor.good",
        "BF16",
        &[good_bits.len()],
        &le_bytes_of_u16(&good_bits),
    );
    write_raw_tensor(
        &dir.join("bad.safetensors"),
        "tensor.bad",
        "BF16",
        &[2],
        &le_bytes_of_u16(&[0x3F80, 0x7FC0]),
    );
    let index_path = dir.join("model.safetensors.index.json");
    fs::write(
        &index_path,
        r#"{"weight_map": {"tensor.good": "good.safetensors", "tensor.bad": "bad.safetensors"}}"#,
    )
    .expect("test setup: write index");

    let mut st = ShardedSafetensors::open_index(&index_path).expect("index parses");
    let (values, shape) = st
        .get_f32_tensor_owned("tensor.good")
        .expect("owned read through the sharded source");
    assert_eq!(shape, vec![good_bits.len()]);
    let expected: Vec<f32> = good_bits
        .iter()
        .map(|&b| crate::weights::half_bits::bf16_bits_to_f32(b))
        .collect();
    assert_eq!(bit_patterns(&values), bit_patterns(&expected));
    let shard = st.shards.get("good.safetensors").expect("shard is open");
    assert!(
        !converted_is_cached(shard, "tensor.good"),
        "the shard must not keep a second f32 copy after an owned read"
    );

    let err = st
        .get_f32_tensor_owned("tensor.bad")
        .expect_err("non-finite BF16 must be refused through the sharded source");
    assert!(
        matches!(err, InferenceError::InvalidSafetensors(_))
            && err.to_string().contains("non-finite"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn bf16_payload_reads_bytes_in_place_without_widening() {
    let path = temp_path("lattice_weights_bf16_payload_bytes");
    let bits = half_bits_fixture(33, 3, 0x7F80);
    let raw = le_bytes_of_u16(&bits);
    write_tensor_with_payload_phase(&path, "BF16", bits.len(), &raw, 2);

    let sf = SafetensorsFile::open(&path).expect("open: valid file");
    assert_eq!(
        sf.bf16_payload("t").expect("finite BF16 payload"),
        raw.as_slice()
    );
    assert_eq!(
        sf.cached_f32_tensor_count(),
        0,
        "reading the payload in place must not materialize an f32 copy"
    );
    assert!(matches!(
        sf.bf16_payload("absent"),
        Err(InferenceError::MissingTensor(_))
    ));
}

#[test]
fn bf16_payload_refuses_non_finite_values_like_the_widening_path() {
    let cases = [
        ("nan_first", 0usize, 0x7FC0u16),
        ("pos_inf_tail", 32, 0x7F80),
        ("neg_inf_middle", 17, 0xFF80),
    ];
    for (label, index, bad_bits) in cases {
        let path = temp_path(&format!("lattice_weights_bf16_payload_{label}"));
        let mut bits = half_bits_fixture(33, 9, 0x7F80);
        bits[index] = bad_bits;
        write_raw_tensor(&path, "t", "BF16", &[bits.len()], &le_bytes_of_u16(&bits));

        let payload_err = SafetensorsFile::open(&path)
            .expect("open: valid file")
            .bf16_payload("t")
            .expect_err("a non-finite BF16 payload must be refused");
        let widening_err = SafetensorsFile::open(&path)
            .expect("open: valid file")
            .get_f32_tensor("t")
            .expect_err("the widening path refuses the same tensor");
        assert!(
            matches!(payload_err, InferenceError::InvalidSafetensors(_)),
            "{label}: unexpected error class: {payload_err:?}"
        );
        assert!(
            payload_err
                .to_string()
                .contains(&format!("element index {index}")),
            "{label}: unexpected error: {payload_err}"
        );
        assert_eq!(payload_err.to_string(), widening_err.to_string(), "{label}");

        let sf = SafetensorsFile::open(&path).expect("open: valid file");
        let first = sf.bf16_payload("t").expect_err("refused on first read");
        let again = sf
            .bf16_payload("t")
            .expect_err("a cached result keeps refusing");
        assert_eq!(first.to_string(), again.to_string(), "{label}");
    }
}

#[test]
fn bf16_payload_refuses_a_tensor_of_another_dtype() {
    let path = temp_path("lattice_weights_bf16_payload_dtype");
    write_single_f32_tensor(&path, "t", &[1.0, 2.0]);
    let sf = SafetensorsFile::open(&path).expect("open: valid file");
    let err = sf
        .bf16_payload("t")
        .expect_err("an F32 tensor is not a BF16 payload");
    assert!(
        matches!(err, InferenceError::InvalidSafetensors(_))
            && err.to_string().contains("F32")
            && err.to_string().contains("expected BF16"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn test_unsupported_dtype_tensor_fails_as_format_error_not_missing_tensor() {
    let path = temp_path("lattice_weights_unsupported_dtype");
    // I64 is a real safetensors dtype this crate does not materialize as
    // f32 — it must stay structurally tracked, not disappear.
    write_raw_tensor(&path, "counts", "I64", &[2], &8i64.to_le_bytes().repeat(2));

    let sf = SafetensorsFile::open(&path).expect("open: known dtype, valid extent");
    assert!(
        sf.has_tensor("counts"),
        "unsupported-dtype tensor must remain structurally tracked"
    );
    assert_eq!(sf.tensor_shape("counts"), Some(&[2usize][..]));

    let err = sf
        .get_f32_tensor("counts")
        .expect_err("requesting an I64 tensor as f32 must fail");
    assert!(
        matches!(err, InferenceError::InvalidSafetensors(_)),
        "must fail as a dtype/format error, not MissingTensor: {err:?}"
    );
    assert!(
        !matches!(err, InferenceError::MissingTensor(_)),
        "must not be reported as a missing tensor: {err:?}"
    );
}

#[test]
fn test_standard_sub_byte_and_fnuz_dtypes_are_tracked_structurally() {
    let cases = [
        ("F4", 2usize, 1usize),
        ("F6_E2M3", 4, 3),
        ("F6_E3M2", 4, 3),
        ("F8_E4M3FNUZ", 1, 1),
        ("F8_E5M2FNUZ", 1, 1),
    ];

    for (dtype, elements, bytes) in cases {
        let path = temp_path(&format!("lattice_weights_{dtype}"));
        write_raw_tensor(&path, "t", dtype, &[elements], &vec![0; bytes]);

        let sf = SafetensorsFile::open(&path)
            .unwrap_or_else(|err| panic!("{dtype} with valid extent must open: {err}"));
        assert!(sf.has_tensor("t"), "{dtype} tensor must remain tracked");
        assert_eq!(sf.tensor_dtype("t"), Some(dtype));
        let err = sf
            .get_f32_tensor("t")
            .expect_err("unsupported dtype must not materialize as f32");
        assert!(
            err.to_string().contains("unsupported dtype"),
            "unexpected {dtype} materialization error: {err}"
        );
    }
}

#[test]
fn test_sub_byte_dtype_requires_byte_aligned_shape() {
    let path = temp_path("lattice_weights_f4_unaligned");
    write_raw_tensor(&path, "t", "F4", &[1], &[]);

    let err = SafetensorsFile::open(&path)
        .expect_err("a one-element F4 tensor must not round down to zero bytes");
    assert!(
        err.to_string().contains("not byte-aligned"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_sub_byte_dtype_rejects_wrong_byte_extent() {
    let path = temp_path("lattice_weights_f4_bad_extent");
    write_raw_tensor(&path, "t", "F4", &[4], &[0]);

    let err = SafetensorsFile::open(&path).expect_err("four F4 elements require two payload bytes");
    assert!(
        err.to_string().contains("byte length mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_unrecognized_dtype_string_rejected_at_parse_time() {
    let path = temp_path("lattice_weights_bogus_dtype");
    let header = r#"{"t":{"dtype":"BANANA","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    let mut file = File::create(&path).expect("test setup: create safetensors file");
    file.write_all(&bytes)
        .expect("test setup: write safetensors bytes");
    drop(file);

    let err = SafetensorsFile::open(&path)
        .expect_err("an unrecognized dtype string must fail at open/parse time");
    assert!(
        err.to_string().contains("unrecognized dtype"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_f16_tensor_rejects_nan_bit_pattern_through_safetensors_path() {
    let path = temp_path("lattice_weights_f16_nan");
    // f16 quiet NaN: sign=0, exponent=0x1F, non-zero fraction.
    let nan_bits: u16 = 0x7E00;
    let one_bits: u16 = 0x3C00; // f16 1.0
    let mut raw = Vec::new();
    raw.extend_from_slice(&one_bits.to_le_bytes());
    raw.extend_from_slice(&nan_bits.to_le_bytes());
    write_raw_tensor(&path, "t", "F16", &[2], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("F16 NaN bit pattern must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
    assert!(
        err.to_string().contains("element index 1"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_f16_tensor_rejects_infinity_bit_pattern_through_safetensors_path() {
    let path = temp_path("lattice_weights_f16_inf");
    // f16 +infinity: sign=0, exponent=0x1F, fraction=0.
    let inf_bits: u16 = 0x7C00;
    let one_bits: u16 = 0x3C00;
    let mut raw = Vec::new();
    raw.extend_from_slice(&inf_bits.to_le_bytes());
    raw.extend_from_slice(&one_bits.to_le_bytes());
    write_raw_tensor(&path, "t", "F16", &[2], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("F16 +inf bit pattern must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_bf16_tensor_rejects_nan_bit_pattern_through_safetensors_path() {
    let path = temp_path("lattice_weights_bf16_nan");
    // bf16 quiet NaN: top 16 bits of the f32 QNaN 0x7FC0_0000.
    let nan_bits: u16 = 0x7FC0;
    let one_bits: u16 = 0x3F80; // bf16 1.0
    let mut raw = Vec::new();
    raw.extend_from_slice(&one_bits.to_le_bytes());
    raw.extend_from_slice(&nan_bits.to_le_bytes());
    write_raw_tensor(&path, "t", "BF16", &[2], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("BF16 NaN bit pattern must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
    assert!(
        err.to_string().contains("element index 1"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_bf16_tensor_rejects_infinity_bit_pattern_through_safetensors_path() {
    let path = temp_path("lattice_weights_bf16_inf");
    // bf16 +infinity: top 16 bits of f32 +inf, 0x7F80_0000.
    let inf_bits: u16 = 0x7F80;
    let one_bits: u16 = 0x3F80;
    let mut raw = Vec::new();
    raw.extend_from_slice(&inf_bits.to_le_bytes());
    raw.extend_from_slice(&one_bits.to_le_bytes());
    write_raw_tensor(&path, "t", "BF16", &[2], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("BF16 +inf bit pattern must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[cfg(feature = "f16")]
#[test]
fn test_f8_e4m3_tensor_decodes_through_safetensors_path() {
    let path = temp_path("lattice_weights_f8_e4m3_decode");
    // 0x00 = +0.0, 0x38 = 1.0, 0xb8 = -1.0, 0x7e = 448.0 (largest finite E4M3).
    let raw: [u8; 4] = [0x00, 0x38, 0xb8, 0x7e];
    write_raw_tensor(&path, "t", "F8_E4M3", &[raw.len()], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let (values, shape) = sf
        .get_f32_tensor("t")
        .expect("finite F8_E4M3 values must decode");
    assert_eq!(shape, &[raw.len()]);
    assert_eq!(values, &[0.0f32, 1.0, -1.0, 448.0]);
}

#[cfg(feature = "f16")]
#[test]
fn test_f8_e4m3_tensor_rejects_nan_bit_pattern_through_safetensors_path() {
    let path = temp_path("lattice_weights_f8_e4m3_nan");
    // E4M3FN's sole NaN encoding: exponent and mantissa both all-ones.
    let one_bits: u8 = 0x38;
    let nan_bits: u8 = 0x7f;
    write_raw_tensor(&path, "t", "F8_E4M3", &[2], &[one_bits, nan_bits]);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("F8_E4M3 NaN bit pattern must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
    assert!(
        err.to_string().contains("element index 1"),
        "unexpected error: {err}"
    );
}

#[cfg(feature = "f16")]
#[test]
fn test_f8_e5m2_tensor_decodes_through_safetensors_path() {
    let path = temp_path("lattice_weights_f8_e5m2_decode");
    // 0x00 = +0.0, 0x3c = 1.0, 0xbc = -1.0, 0x7b = 57344.0 (largest finite E5M2).
    let raw: [u8; 4] = [0x00, 0x3c, 0xbc, 0x7b];
    write_raw_tensor(&path, "t", "F8_E5M2", &[raw.len()], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let (values, shape) = sf
        .get_f32_tensor("t")
        .expect("finite F8_E5M2 values must decode");
    assert_eq!(shape, &[raw.len()]);
    assert_eq!(values, &[0.0f32, 1.0, -1.0, 57344.0]);
}

#[cfg(feature = "f16")]
#[test]
fn test_f8_e5m2_tensor_rejects_infinity_bit_pattern_through_safetensors_path() {
    let path = temp_path("lattice_weights_f8_e5m2_inf");
    // E5M2 +infinity: exponent all-ones, mantissa zero.
    let inf_bits: u8 = 0x7c;
    let one_bits: u8 = 0x3c;
    write_raw_tensor(&path, "t", "F8_E5M2", &[2], &[inf_bits, one_bits]);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let err = sf
        .get_f32_tensor("t")
        .expect_err("F8_E5M2 +inf bit pattern must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_f16_widening_preserves_finite_edge_values() {
    let path = temp_path("lattice_weights_f16_fused_finite");
    let bits = [0x0000u16, 0x8000, 0x0001, 0x3c00];
    let raw = bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    write_raw_tensor(&path, "t", "F16", &[bits.len()], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let (values, shape) = sf
        .get_f32_tensor("t")
        .expect("finite F16 values must widen");
    let expected = bits.map(crate::weights::half_bits::f16_bits_to_f32);
    assert_eq!(shape, &[bits.len()]);
    assert_eq!(
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert!(
        sf.tensors
            .get("t")
            .expect("tensor tracked")
            .validated
            .get()
            .is_some(),
        "widening must publish its fused validation result"
    );
}

#[test]
fn test_f16_widening_skips_second_finite_scan() {
    let path = temp_path("lattice_weights_f16_fused_scan");
    let bits = [0x0000u16, 0x8000, 0x0001, 0x3c00];
    let raw = bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    write_raw_tensor(&path, "t", "F16", &[bits.len()], &raw);

    crate::weights::ingress::reset_decoded_f32_finite_check_count();
    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    sf.get_f32_tensor("t")
        .expect("finite F16 values must widen");
    assert_eq!(
        crate::weights::ingress::decoded_f32_finite_check_count(),
        0,
        "fused F16 validation must not rescan widened values"
    );
}

#[test]
fn test_bf16_widening_preserves_finite_edge_values() {
    let path = temp_path("lattice_weights_bf16_fused_finite");
    let bits = [0x0000u16, 0x8000, 0x0001, 0x3f80];
    let raw = bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    write_raw_tensor(&path, "t", "BF16", &[bits.len()], &raw);

    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    let (values, shape) = sf
        .get_f32_tensor("t")
        .expect("finite BF16 values must widen");
    let expected = bits.map(crate::weights::half_bits::bf16_bits_to_f32);
    assert_eq!(shape, &[bits.len()]);
    assert_eq!(
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert!(
        sf.tensors
            .get("t")
            .expect("tensor tracked")
            .validated
            .get()
            .is_some(),
        "widening must publish its fused validation result"
    );
}

#[test]
fn test_bf16_widening_skips_second_finite_scan() {
    let path = temp_path("lattice_weights_bf16_fused_scan");
    let bits = [0x0000u16, 0x8000, 0x0001, 0x3f80];
    let raw = bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    write_raw_tensor(&path, "t", "BF16", &[bits.len()], &raw);

    crate::weights::ingress::reset_decoded_f32_finite_check_count();
    let sf = SafetensorsFile::open(&path).expect("open: header/extent are valid");
    sf.get_f32_tensor("t")
        .expect("finite BF16 values must widen");
    assert_eq!(
        crate::weights::ingress::decoded_f32_finite_check_count(),
        0,
        "fused BF16 validation must not rescan widened values"
    );
}

/// F16/BF16 -> f32 materialization must work in the DEFAULT feature build, with
/// no `f16` feature enabled: this is the regression a published checkpoint hits,
/// since most published safetensors checkpoints carry F16 or BF16 tensors and a
/// plain `cargo add lattice-inference` build must still load them.
///
/// Builds a tiny in-memory safetensors buffer (via `from_bytes`, no filesystem)
/// containing one F16 tensor and one BF16 tensor, and loads both through
/// `get_f32_tensor` -- the same public load path a real checkpoint goes through.
#[test]
fn f16_and_bf16_tensors_materialize_to_f32_in_default_feature_build() {
    // f16 bit patterns for 1.0 (0x3c00) and 2.0 (0x4000).
    let f16_bytes: [u8; 4] = [0x00, 0x3c, 0x00, 0x40];
    // bf16 bit patterns for 1.0 (0x3f80) and 2.0 (0x4000).
    let bf16_bytes: [u8; 4] = [0x80, 0x3f, 0x00, 0x40];

    let header = format!(
        r#"{{"f16_t":{{"dtype":"F16","shape":[2],"data_offsets":[0,{}]}},"bf16_t":{{"dtype":"BF16","shape":[2],"data_offsets":[{},{}]}}}}"#,
        f16_bytes.len(),
        f16_bytes.len(),
        f16_bytes.len() + bf16_bytes.len(),
    );
    let header_bytes = header.into_bytes();

    let mut buf = Vec::with_capacity(8 + header_bytes.len() + f16_bytes.len() + bf16_bytes.len());
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&f16_bytes);
    buf.extend_from_slice(&bf16_bytes);

    let file = SafetensorsFile::from_bytes(buf)
        .expect("in-memory safetensors buffer with F16/BF16 tensors must parse");

    let (f16_values, f16_shape) = file
        .get_f32_tensor("f16_t")
        .expect("F16 tensor must materialize to f32 without the f16 feature");
    assert_eq!(f16_shape, &[2]);
    assert_eq!(f16_values, &[1.0f32, 2.0]);

    let (bf16_values, bf16_shape) = file
        .get_f32_tensor("bf16_t")
        .expect("BF16 tensor must materialize to f32 without the f16 feature");
    assert_eq!(bf16_shape, &[2]);
    assert_eq!(bf16_values, &[1.0f32, 2.0]);
}

fn temp_dir(name: &str) -> TempDirGuard {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("invariant: system time is after UNIX_EPOCH")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{}_{}_{}", name, std::process::id(), nanos));
    fs::create_dir_all(&path).expect("test setup: create temp dir");
    TempDirGuard(path)
}

fn write_single_f32_tensor(path: &std::path::Path, name: &str, values: &[f32]) {
    let byte_len = std::mem::size_of_val(values);
    let header = format!(
        r#"{{"{name}":{{"dtype":"F32","shape":[{}],"data_offsets":[0,{byte_len}]}}}}"#,
        values.len()
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut file = File::create(path).expect("test setup: create shard");
    file.write_all(&bytes).expect("test setup: write shard");
}

#[test]
fn test_parse_safetensors_index_and_resolve_weight_map() {
    let dir = temp_dir("lattice_sharded_weights_test");
    let shard_a = dir.join("model-00001-of-00002.safetensors");
    let shard_b = dir.join("model-00002-of-00002.safetensors");
    write_single_f32_tensor(&shard_a, "tensor.a", &[1.0, 2.0]);
    write_single_f32_tensor(&shard_b, "tensor.b", &[3.0, 4.0, 5.0]);

    let index_path = dir.join("model.safetensors.index.json");
    fs::write(
        &index_path,
        r#"{
                "metadata": {"total_size": 20.0},
                "weight_map": {
                    "tensor.a": "model-00001-of-00002.safetensors",
                    "tensor.b": "model-00002-of-00002.safetensors"
                }
            }"#,
    )
    .expect("test setup: write index");

    let mut st = ShardedSafetensors::open_index(&index_path).expect("index parses");

    let (path_a, name_a) = st.resolve_weight("tensor.a").expect("tensor a resolves");
    assert_eq!(
        path_a.file_name().unwrap().to_string_lossy(),
        "model-00001-of-00002.safetensors"
    );
    assert_eq!(name_a, "tensor.a");

    let (a, a_shape) = st.get_f32_tensor_owned("tensor.a").expect("tensor a loads");
    let (b, b_shape) = st.get_f32_tensor_owned("tensor.b").expect("tensor b loads");
    assert_eq!(a_shape, vec![2]);
    assert_eq!(b_shape, vec![3]);
    assert_eq!(a, vec![1.0, 2.0]);
    assert_eq!(b, vec![3.0, 4.0, 5.0]);

    assert!(st.has_tensor("tensor.a").unwrap());
    assert!(!st.has_tensor("tensor.c").unwrap());

    // Test parse_index free function.
    let index = parse_index(&dir).expect("parse_index succeeds");
    assert_eq!(index.weight_map.len(), 2);

    // Test resolve_shard free function.
    // The returned path is the validated entry joined onto `model_dir`, not a
    // canonicalized one -- symlinks are followed at open time, not resolved here.
    let shard_a_path = resolve_shard(&index, &dir, "tensor.a").expect("resolve_shard tensor.a");
    assert_eq!(shard_a_path, shard_a);
    assert!(resolve_shard(&index, &dir, "tensor.missing").is_err());

    // Test load_sharded free function.
    let loaded = load_sharded(&dir).expect("load_sharded succeeds");
    assert_eq!(loaded.len(), 2);
    let ta = loaded.get("tensor.a").expect("tensor.a in loaded map");
    let tb = loaded.get("tensor.b").expect("tensor.b in loaded map");
    assert_eq!(ta.data, vec![1.0_f32, 2.0]);
    assert_eq!(ta.shape, vec![2]);
    assert_eq!(tb.data, vec![3.0_f32, 4.0, 5.0]);
    assert_eq!(tb.shape, vec![3]);
}

/// #1069: index-declared shard names are untrusted checkpoint content.
/// Plain names and subdirectory names beneath the model directory
/// resolve; anything that escapes the directory is rejected.
#[test]
fn contained_shard_path_accepts_entries_beneath_model_dir() {
    let dir = temp_dir("lattice_containment_ok_test");
    fs::write(dir.join("shard.safetensors"), b"x").expect("test setup");
    fs::create_dir_all(dir.join("sub")).expect("test setup");
    fs::write(dir.join("sub").join("nested.safetensors"), b"x").expect("test setup");

    let plain = contained_shard_path(&dir, "shard.safetensors").expect("plain name resolves");
    assert_eq!(
        plain.file_name().unwrap().to_string_lossy(),
        "shard.safetensors"
    );
    contained_shard_path(&dir, "sub/nested.safetensors")
        .expect("subdirectory entry beneath the model dir resolves");
}

/// #1069: an absolute index entry replaces the model directory entirely
/// in `Path::join`; it must be rejected even when the target exists.
#[test]
fn contained_shard_path_rejects_absolute_entry() {
    let dir = temp_dir("lattice_containment_abs_test");
    let outside = temp_dir("lattice_containment_abs_outside");
    let target = outside.join("outside.safetensors");
    fs::write(&target, b"x").expect("test setup");

    let err = contained_shard_path(&dir, &target.to_string_lossy())
        .expect_err("absolute entry must be rejected");
    assert!(
        matches!(err, InferenceError::InvalidSafetensors(_)),
        "unexpected error kind: {err}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("must stay within the model directory") || msg.contains("escapes model root"),
        "unexpected error: {err}"
    );
}

/// #1069: `..` components must not address files outside the model
/// directory, whether or not the target exists.
#[test]
fn contained_shard_path_rejects_parent_traversal() {
    let outer = temp_dir("lattice_containment_dotdot_test");
    let dir = outer.join("model");
    fs::create_dir_all(&dir).expect("test setup");
    fs::write(outer.join("secret.bin"), b"x").expect("test setup");

    let err =
        contained_shard_path(&dir, "../secret.bin").expect_err("parent traversal must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("must stay within the model directory") || msg.contains("escapes model root"),
        "unexpected error: {err}"
    );
    assert!(
        contained_shard_path(&dir, "../nonexistent.bin").is_err(),
        "traversal to a missing target must also fail"
    );
}

/// The HuggingFace hub cache stores snapshots as symlink farms:
/// `snapshots/<rev>/model.safetensors -> ../../blobs/<sha>`. Lattice
/// loads those directories directly (the e2e workflows point the
/// engine at `snapshot_download()` output), so index-entry validation
/// must stay lexical — a policy that resolved symlinks and required
/// the target beneath the model directory would reject every
/// hub-cache checkpoint. This pins the layout end-to-end through
/// `load_sharded`.
#[cfg(unix)]
#[test]
fn sharded_loader_follows_hub_cache_snapshot_layout() {
    let outer = temp_dir("lattice_containment_hub_layout_test");
    let blobs = outer.join("blobs");
    let dir = outer.join("snapshots").join("rev");
    fs::create_dir_all(&blobs).expect("test setup");
    fs::create_dir_all(&dir).expect("test setup");
    write_single_f32_tensor(&blobs.join("blob-sha"), "tensor.a", &[1.0, 2.0]);
    std::os::unix::fs::symlink(
        Path::new("../../blobs/blob-sha"),
        dir.join("model.safetensors"),
    )
    .expect("test setup");
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"metadata": {}, "weight_map": {"tensor.a": "model.safetensors"}}"#,
    )
    .expect("test setup: write index");

    let tensors =
        load_sharded(&dir).expect("hub-cache snapshot layout must load through the symlink");
    assert_eq!(tensors["tensor.a"].data, vec![1.0, 2.0]);
}

/// #1069 end-to-end: a sharded index whose weight_map entry escapes the
/// model directory fails to load through both sharded loaders.
#[test]
fn sharded_loaders_reject_escaping_index_entry() {
    let outer = temp_dir("lattice_containment_loader_test");
    let dir = outer.join("model");
    fs::create_dir_all(&dir).expect("test setup");
    // A structurally valid shard OUTSIDE the model dir — containment,
    // not file validity, must be what rejects it.
    let evil = outer.join("evil.safetensors");
    write_single_f32_tensor(&evil, "tensor.a", &[1.0, 2.0]);
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"metadata": {}, "weight_map": {"tensor.a": "../evil.safetensors"}}"#,
    )
    .expect("test setup: write index");

    let err = load_sharded(&dir).expect_err("load_sharded must reject the escaping entry");
    let msg = err.to_string();
    assert!(
        msg.contains("must stay within the model directory") || msg.contains("escapes model root"),
        "unexpected error: {err}"
    );

    let index_path = dir.join("model.safetensors.index.json");
    let mut st = ShardedSafetensors::open_index(&index_path).expect("index itself parses");
    let err = st
        .get_f32_tensor_owned("tensor.a")
        .expect_err("lazy loader must reject the escaping entry at access time");
    let msg = err.to_string();
    assert!(
        msg.contains("must stay within the model directory") || msg.contains("escapes model root"),
        "unexpected error: {err}"
    );
    let err = st
        .resolve_weight("tensor.a")
        .expect_err("resolve_weight must reject the escaping entry");
    let msg = err.to_string();
    assert!(
        msg.contains("must stay within the model directory") || msg.contains("escapes model root"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_load_sharded_multi_tensor_per_shard() {
    let dir = temp_dir("lattice_multi_tensor_shard_test");
    let shard = dir.join("model-00001-of-00001.safetensors");

    // Write one shard file containing two named tensors.
    let x_byte_count = 2_usize * std::mem::size_of::<f32>(); // 8
    let y_byte_count = 3_usize * std::mem::size_of::<f32>(); // 12
    let xy_end = x_byte_count + y_byte_count;
    let header = format!(
        r#"{{"tensor.x":{{"dtype":"F32","shape":[2],"data_offsets":[0,{x_byte_count}]}},"tensor.y":{{"dtype":"F32","shape":[3],"data_offsets":[{x_byte_count},{xy_end}]}}}}"#
    );
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in [1.0_f32, 2.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in [3.0_f32, 4.0, 5.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&shard, &bytes).expect("test setup: write two-tensor shard");

    let index_path = dir.join("model.safetensors.index.json");
    fs::write(
            &index_path,
            r#"{"weight_map":{"tensor.x":"model-00001-of-00001.safetensors","tensor.y":"model-00001-of-00001.safetensors"}}"#,
        )
        .expect("test setup: write index");

    let loaded = load_sharded(&dir).expect("load_sharded with multi-tensor shard succeeds");
    assert_eq!(loaded.len(), 2);
    let tx = loaded.get("tensor.x").expect("tensor.x in loaded map");
    let ty = loaded.get("tensor.y").expect("tensor.y in loaded map");
    assert_eq!(tx.data, vec![1.0_f32, 2.0]);
    assert_eq!(tx.shape, vec![2]);
    assert_eq!(ty.data, vec![3.0_f32, 4.0, 5.0]);
    assert_eq!(ty.shape, vec![3]);
}

#[test]
fn test_load_sharded_missing_shard_returns_err() {
    let dir = temp_dir("lattice_missing_shard_test");

    // Index references a shard file that is never written to disk.
    let index_path = dir.join("model.safetensors.index.json");
    fs::write(
        &index_path,
        r#"{"weight_map":{"tensor.a":"model-00001-of-00001.safetensors"}}"#,
    )
    .expect("test setup: write index");

    let result = load_sharded(&dir);
    assert!(
        result.is_err(),
        "load_sharded must return Err when indexed shard file is absent"
    );
}

// --- CrossEncoderWeights tests ---

#[test]
fn test_cross_encoder_weights_logit_known_value() {
    let weights = CrossEncoderWeights {
        classifier_weight: vec![1.0f32, 2.0, -1.0],
        classifier_bias: 0.5,
    };
    let pooled = [2.0f32, 1.0, 3.0];
    // logit = 1.0*2.0 + 2.0*1.0 + (-1.0)*3.0 + 0.5 = 2.0 + 2.0 - 3.0 + 0.5 = 1.5
    let logit = weights.logit(&pooled);
    assert!(
        (logit - 1.5f32).abs() < 1e-6,
        "expected logit=1.5, got {logit}"
    );
}

fn write_classifier_safetensors(
    path: &std::path::Path,
    weight_shape_header: &str,
    weight_values: &[f32],
    bias_value: f32,
    weight_end: usize,
) {
    let bias_end = weight_end + 4; // one f32
    let header = format!(
        r#"{{"classifier.weight":{{{weight_shape_header},"data_offsets":[0,{weight_end}]}},"classifier.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{weight_end},{bias_end}]}}}}"#
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in weight_values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&bias_value.to_le_bytes());
    let mut file = File::create(path).expect("test setup: create safetensors");
    file.write_all(&bytes)
        .expect("test setup: write safetensors");
}

#[test]
fn test_load_cross_encoder_weights_2d_shape() {
    // classifier.weight [1, 3] — standard HF BertForSequenceClassification shape
    let path = temp_path("lattice_ce_weights_2d");
    write_classifier_safetensors(
        &path,
        r#""dtype":"F32","shape":[1,3]"#,
        &[1.0f32, 2.0, 3.0],
        0.5,
        12,
    );
    let st = SafetensorsFile::open(&path).unwrap();
    let weights = st.load_cross_encoder_weights(3).unwrap();
    assert_eq!(weights.classifier_weight, vec![1.0f32, 2.0, 3.0]);
    assert!(
        (weights.classifier_bias - 0.5f32).abs() < 1e-6,
        "bias mismatch: {}",
        weights.classifier_bias
    );
}

#[test]
fn test_load_cross_encoder_weights_1d_shape() {
    // classifier.weight [3] — accepted as converted-checkpoint fallback
    let path = temp_path("lattice_ce_weights_1d");
    write_classifier_safetensors(
        &path,
        r#""dtype":"F32","shape":[3]"#,
        &[4.0f32, 5.0, 6.0],
        1.0,
        12,
    );
    let st = SafetensorsFile::open(&path).unwrap();
    let weights = st.load_cross_encoder_weights(3).unwrap();
    assert_eq!(weights.classifier_weight, vec![4.0f32, 5.0, 6.0]);
    assert!(
        (weights.classifier_bias - 1.0f32).abs() < 1e-6,
        "bias mismatch: {}",
        weights.classifier_bias
    );
}

#[test]
fn test_load_cross_encoder_weights_shape_mismatch_error() {
    // classifier.weight [1, 4] with hidden_size=3 must yield ShapeMismatch
    let path = temp_path("lattice_ce_weights_mismatch");
    write_classifier_safetensors(
        &path,
        r#""dtype":"F32","shape":[1,4]"#,
        &[1.0f32, 2.0, 3.0, 4.0],
        0.0,
        16,
    );
    let st = SafetensorsFile::open(&path).unwrap();
    let err = st
        .load_cross_encoder_weights(3)
        .expect_err("shape mismatch should be an error");
    assert!(
        matches!(err, InferenceError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn parse_index_rejects_oversized_index_file() {
    let root = temp_dir("lattice_index_oversized");
    let oversized = File::create(root.join("model.safetensors.index.json"))
        .expect("test setup: create oversized index");
    oversized
        .set_len(MAX_SAFETENSORS_INDEX_BYTES + 1)
        .expect("test setup: extend oversized index");
    let err = parse_index(&root).expect_err("an oversized safetensors index must be rejected");
    assert!(
        err.to_string().contains("MAX_SAFETENSORS_INDEX_BYTES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parse_index_rejects_a_directory_at_the_index_path() {
    let root = temp_dir("lattice_index_directory");
    fs::create_dir(root.join("model.safetensors.index.json"))
        .expect("test setup: create index directory");

    let err = parse_index(&root).expect_err("an index directory must be rejected");
    assert!(
        err.to_string().contains("is not a regular file"),
        "wrong error: {err}"
    );
}

#[cfg(unix)]
#[test]
fn parse_index_rejects_a_fifo_without_blocking() {
    use std::os::unix::ffi::OsStrExt;

    let root = temp_dir("lattice_index_fifo");
    let path = root.join("model.safetensors.index.json");
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: `c_path` is a valid, NUL-terminated path for the duration of the call.
    let result = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
    assert_eq!(
        result,
        0,
        "mkfifo failed: {}",
        std::io::Error::last_os_error()
    );

    let err = parse_index(&root).expect_err("a FIFO index must be rejected without blocking");
    assert!(
        err.to_string().contains("is not a regular file"),
        "wrong error: {err}"
    );
}

#[test]
fn parse_index_rejects_weight_map_over_entry_cap() {
    // FIX 6 regression: `weight_map` entry count must be capped incrementally during
    // deserialization -- before the whole map collapses into an owned `HashMap`.
    // MAX_WEIGHT_MAP_ENTRIES + 1 raw members, each mapping to the same short shard
    // name, must be rejected.
    let mut entries = String::with_capacity((MAX_WEIGHT_MAP_ENTRIES + 1) * 24);
    for i in 0..=MAX_WEIGHT_MAP_ENTRIES {
        if i > 0 {
            entries.push(',');
        }
        entries.push_str(&format!(r#""t{i}":"s""#));
    }
    let json = format!(r#"{{"metadata":{{}},"weight_map":{{{entries}}}}}"#);
    let err = serde_json::from_str::<SafetensorsIndex>(&json)
        .expect_err("weight_map over MAX_WEIGHT_MAP_ENTRIES must be rejected");
    assert!(
        err.to_string().contains("MAX_WEIGHT_MAP_ENTRIES"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parse_index_accepts_weight_map_at_entry_cap() {
    // A weight_map with exactly MAX_WEIGHT_MAP_ENTRIES entries must be accepted --
    // this is a boundary test, not a realistic-checkpoint test (real checkpoints use
    // far fewer entries; see `parse_index`'s doc comment).
    let mut entries = String::with_capacity(MAX_WEIGHT_MAP_ENTRIES * 24);
    for i in 0..MAX_WEIGHT_MAP_ENTRIES {
        if i > 0 {
            entries.push(',');
        }
        entries.push_str(&format!(r#""t{i}":"s""#));
    }
    let json = format!(r#"{{"metadata":{{}},"weight_map":{{{entries}}}}}"#);
    let index: SafetensorsIndex = serde_json::from_str(&json)
        .expect("weight_map at exactly MAX_WEIGHT_MAP_ENTRIES must be accepted");
    assert_eq!(index.weight_map.len(), MAX_WEIGHT_MAP_ENTRIES);
}

#[test]
fn open_manifest_entry_once_rejects_traversal_before_opening() {
    // The lexical check runs before the join, so a `../` entry is refused whether or
    // not it names a real file -- the fd is never opened. Symlink escape is NOT
    // tested here and NOT rejected: see the threat model on `contained_shard_path`
    // and `sharded_loader_follows_hub_cache_snapshot_layout`.
    let root = temp_dir("lattice_manifest_entry_traversal");
    let err = open_manifest_entry_once(&root, "../escape.safetensors")
        .expect_err("a `../` manifest entry must be rejected");
    assert!(
        err.to_string().contains("escapes the model directory"),
        "expected the lexical traversal guard to fire, got: {err}"
    );
}

// #1380: `open_manifest_entry_once` is a manifest-derived shard entry
// point -- one of the three the issue named -- that used to call plain
// `File::open` before any file-type check ran, so a FIFO planted at a
// shard path named by a checkpoint's manifest blocked this call
// indefinitely. Proves the fix the same way as the other loader entry
// points: the call must return (any return) inside a deadline, and the
// return must specifically reject the FIFO as non-regular.
#[test]
fn open_manifest_entry_once_rejects_a_fifo_without_blocking() {
    let root = temp_dir("lattice_manifest_entry_fifo");
    let shard_path = root.join("model.safetensors");
    let status = std::process::Command::new("mkfifo")
        .arg(&shard_path)
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo must succeed to set up this test");

    let (tx, rx) = std::sync::mpsc::channel();
    let probe_root = root.to_path_buf();
    std::thread::spawn(move || {
        let result = open_manifest_entry_once(&probe_root, "model.safetensors");
        let _ = tx.send(result);
    });
    let result = rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
        "open_manifest_entry_once did not return within 5s -- it blocked on the \
             planted FIFO, meaning the open-time regular-file guard regressed",
    );
    let err = result.expect_err("a FIFO shard must be rejected, not opened");
    assert!(
        err.to_string().contains("not a regular file"),
        "expected a not-a-regular-file refusal, got: {err}"
    );
}

#[test]
fn load_sharded_rejects_hostile_index_with_traversal_entry() {
    let root = temp_dir("lattice_containment_load_sharded");
    let escape_target = root.parent().expect("temp dir has a parent").join(format!(
        "lattice_containment_load_sharded_secret_{}.safetensors",
        std::process::id()
    ));
    write_single_f32_tensor(&escape_target, "tensor.a", &[1.0, 2.0]);

    let index_path = root.join("model.safetensors.index.json");
    fs::write(
        &index_path,
        format!(
            r#"{{"metadata": {{"total_size": 8.0}}, "weight_map": {{"tensor.a": "../{}"}}}}"#,
            escape_target.file_name().unwrap().to_str().unwrap()
        ),
    )
    .expect("test setup: write hostile index");

    let result = load_sharded(&root);

    fs::remove_file(&escape_target).ok();

    assert!(
        result.is_err(),
        "load_sharded must reject a `../` traversal entry in the index manifest"
    );
}
