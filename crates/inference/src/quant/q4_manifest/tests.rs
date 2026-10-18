use super::*;

#[test]
fn load_manifest_absent_file_is_none() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(load_manifest(tmp.path()).unwrap().is_none());
}

#[test]
fn load_manifest_bare_array_normalizes_to_quantize_q4_flavor() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(
        tmp.path().join("quantize_index.json"),
        r#"[{"name":"foo","file":"foo.q4","quantized":true,"shape":[2,2],"numel":4}]"#,
    )
    .unwrap();
    let manifest = load_manifest(tmp.path())
        .unwrap()
        .expect("manifest must load");
    assert_eq!(manifest.flavor, ManifestFlavor::QuantizeQ4);
    assert_eq!(manifest.quarot_seed, None);
    assert_eq!(manifest.tensors.len(), 1);
    assert_eq!(manifest.tensors[0].name, "foo");
}

#[test]
fn load_manifest_object_form_normalizes_to_quarot_flavor_with_seed() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(
            tmp.path().join("quantize_index.json"),
            r#"{"quarot_seed":42,"tensors":[{"name":"foo","file":"foo.q4","quantized":true,"shape":[2],"numel":2}]}"#,
        )
        .unwrap();
    let manifest = load_manifest(tmp.path())
        .unwrap()
        .expect("manifest must load");
    assert_eq!(manifest.flavor, ManifestFlavor::QuaRot);
    assert_eq!(manifest.quarot_seed, Some(42));
    assert_eq!(manifest.tensors.len(), 1);
}

#[test]
fn load_manifest_object_form_without_seed_key_is_none() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("quantize_index.json"), r#"{"tensors":[]}"#).unwrap();
    let manifest = load_manifest(tmp.path())
        .unwrap()
        .expect("manifest must load");
    assert_eq!(manifest.quarot_seed, None);
    assert!(manifest.tensors.is_empty());
}

#[test]
fn load_manifest_tolerates_entries_without_quantized_shape_numel() {
    // `doctor`'s pre-#655 parser only ever required `name`/`file` per
    // entry (it never reads `quantized`/`shape`/`numel`); that
    // tolerance is intentional and preserved here across both shapes,
    // even though the QuaRot seed loader requires the full entry
    // schema for the object form (see
    // `quant::quarot::convert::read_quarot_seed_from_index`).
    let bare = tempfile::tempdir().unwrap();
    fs::write(
        bare.path().join("quantize_index.json"),
        r#"[{"name":"foo","file":"foo.q4"}]"#,
    )
    .unwrap();
    let bare_manifest = load_manifest(bare.path())
        .unwrap()
        .expect("minimal bare-array entry must load for doctor's inventory");
    assert_eq!(bare_manifest.tensors[0].quantized, None);

    let object = tempfile::tempdir().unwrap();
    fs::write(
        object.path().join("quantize_index.json"),
        r#"{"quarot_seed":42,"tensors":[{"name":"foo","file":"foo.q4"}]}"#,
    )
    .unwrap();
    let object_manifest = load_manifest(object.path())
        .unwrap()
        .expect("minimal object-form entry must load for doctor's inventory");
    assert_eq!(object_manifest.tensors[0].shape, None);
    assert_eq!(object_manifest.tensors[0].numel, None);
}

#[test]
fn parse_manifest_malformed_bare_array_error_names_the_missing_field() {
    let err = parse_manifest(br#"[{"name": "x"}]"#, Path::new("quantize_index.json"))
        .expect_err("bare-array entry missing `file` must fail");
    assert!(
        matches!(err, Q4ManifestError::InvalidShape(_)),
        "wrong error variant: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("file"),
        "error must name the missing `file` field; got: {msg}"
    );
    assert!(
        !msg.contains("did not match any variant"),
        "error must not be the generic untagged-enum fallthrough; got: {msg}"
    );
}

#[test]
fn parse_manifest_malformed_object_tensors_error_names_tensors() {
    let err = parse_manifest(
        br#"{"quarot_seed": 42, "tensors": "not-an-array"}"#,
        Path::new("quantize_index.json"),
    )
    .expect_err("object-form manifest with non-array tensors must fail");
    assert!(
        matches!(err, Q4ManifestError::InvalidShape(_)),
        "wrong error variant: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("tensors") || msg.contains("sequence"),
        "error must point at the bad `tensors` value; got: {msg}"
    );
    assert!(
        !msg.contains("did not match any variant"),
        "error must not be the generic untagged-enum fallthrough; got: {msg}"
    );
}

#[test]
fn parse_manifest_non_array_non_object_root_is_rejected_with_shape_hint() {
    let err = parse_manifest(br#""just a string""#, Path::new("quantize_index.json"))
        .expect_err("scalar manifest root must fail");
    assert!(
        matches!(err, Q4ManifestError::InvalidShape(_)),
        "wrong error variant: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("either a bare array") && msg.contains("tensors"),
        "error must explain both accepted shapes; got: {msg}"
    );
}

#[test]
fn read_manifest_bytes_bounded_rejects_oversized_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("quantize_index.json");
    // One byte over the cap; write sparsely via set_len to avoid
    // actually allocating/writing 16 MiB+1 in the test.
    let f = fs::File::create(&path).unwrap();
    f.set_len(MAX_QUANTIZE_INDEX_LEN + 1).unwrap();
    drop(f);
    let err = read_manifest_bytes_bounded(&path)
        .expect_err("oversized quantize_index.json must be rejected");
    assert!(
        matches!(err, Q4ManifestError::TooLarge(_)),
        "wrong error variant: {err:?}"
    );
    assert!(
        err.to_string().contains("too large"),
        "error must name the size-cap failure; got: {err}"
    );
}

#[test]
fn read_manifest_bytes_bounded_rejects_truncated_json_via_parse_manifest() {
    // The bounded reader itself only bounds size; truncation-as-invalid-JSON
    // is caught one layer up, in `parse_manifest` / `load_manifest`.
    let tmp = tempfile::tempdir().unwrap();
    fs::write(
        tmp.path().join("quantize_index.json"),
        r#"{"quarot_seed":42,"tensors":[{"name":"foo","file":"foo.q4","quant"#,
    )
    .unwrap();
    let err =
        load_manifest(tmp.path()).expect_err("truncated quantize_index.json must be rejected");
    assert!(
        matches!(err, Q4ManifestError::InvalidJson(_)),
        "wrong error variant: {err:?}"
    );
}

#[test]
#[cfg(unix)]
fn read_manifest_bytes_bounded_rejects_dangling_symlink() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().unwrap();
    let link_path = tmp.path().join("quantize_index.json");
    symlink(tmp.path().join("does-not-exist"), &link_path).unwrap();

    let err = read_manifest_bytes_bounded(&link_path).expect_err(
        "a quantize_index.json symlink to a missing target must fail closed, not Ok(None)",
    );
    assert!(
        matches!(err, Q4ManifestError::Unreadable(_)),
        "wrong error variant: {err:?}"
    );
}

#[test]
#[cfg(unix)]
fn read_manifest_bytes_bounded_rejects_a_fifo_without_blocking() {
    use std::os::unix::ffi::OsStrExt;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("quantize_index.json");
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: `c_path` is a valid, NUL-terminated path for the duration of the call.
    let result = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
    assert_eq!(
        result,
        0,
        "mkfifo failed: {}",
        std::io::Error::last_os_error()
    );

    let err = read_manifest_bytes_bounded(&path)
        .expect_err("a FIFO must be rejected without opening a writer");
    assert!(matches!(err, Q4ManifestError::Unreadable(_)));
}

#[test]
#[cfg(unix)]
fn load_manifest_rejects_dangling_symlink() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().unwrap();
    let link_path = tmp.path().join("quantize_index.json");
    symlink(tmp.path().join("does-not-exist"), &link_path).unwrap();

    let err = load_manifest(tmp.path())
        .expect_err("a dangling quantize_index.json symlink must not be treated as absent");
    assert!(matches!(err, Q4ManifestError::Unreadable(_)));
}
