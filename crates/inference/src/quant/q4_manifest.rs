//! Shared bounded reader for `quantize_index.json`, the per-tensor manifest
//! written next to a Q4 checkpoint's `.q4`/`.f16` files.
//!
//! Two writers produce this file with different top-level JSON shapes:
//!
//! - `bin/quantize_q4.rs` ([`ManifestFlavor::QuantizeQ4`]): a bare top-level
//!   array of tensor entries.
//! - [`crate::quant::quarot::convert::convert_quarot_qwen35`] (ADR-051,
//!   [`ManifestFlavor::QuaRot`]): an object, `{"quarot_seed": u64 | null,
//!   "tensors": [...]}`, so a loader can recover the QuaRot rotation seed
//!   without parsing `config.json`.
//!
//! Before this module, both call sites re-derived the bounded-read/
//! size-cap contract independently (`lattice doctor`'s `inspect_q4_dir`
//! used an **unbounded** `std::fs::read`; QuaRot seed loading in
//! `quant::quarot::convert` had its own bounded reader) — see issue #655.
//! [`read_manifest_bytes_bounded`] is now the single implementation of that
//! contract, used by both callers: bounded, `Ok(None)` on a genuinely
//! *absent* manifest (a legitimate legacy/no-manifest fallback state), and
//! fail-closed (`Err`) on anything *present* but oversized, unreadable, or
//! a dangling symlink.
//!
//! Shape normalization, by contrast, is **not** fully unified: `doctor`'s
//! tensor inventory ([`parse_manifest`] / [`load_manifest`], via
//! [`Q4ManifestEntry`]) and QuaRot rotation-seed loading
//! (`quant::quarot::convert::read_quarot_seed_from_index`) have genuinely
//! different acceptance contracts. `doctor` only inventories tensors and
//! has always tolerated a minimal `{name, file}` entry (ignoring extra or
//! missing `quantized`/`shape`/`numel`); the seed loader has always
//! required every field of an object-form (`quarot_seed`-carrying)
//! manifest exactly, while treating *any* bare-array manifest as carrying
//! no seed without validating its entries at all (a bare array is
//! definitionally not the object shape a seed could live in). Those two
//! contracts are preserved exactly here rather than collapsed into one:
//! [`parse_manifest`] backs `doctor` only; the seed loader keeps its own
//! strict, array-shape-skips-validation parse in
//! `quant::quarot::convert` and calls only [`read_manifest_bytes_bounded`]
//! from this module.

use std::fs;
use std::io::Read;
use std::path::Path;

/// Bounded size cap for `quantize_index.json` reads (#504 remaining slice
/// 2). The index is a small per-tensor manifest (name/file/quantized/
/// shape/numel per output tensor, plus an optional rotation seed); even a
/// multi-thousand-tensor checkpoint stays well under this cap.
pub const MAX_QUANTIZE_INDEX_LEN: u64 = 16 * 1024 * 1024; // 16 MiB

/// Error surface for [`read_manifest_bytes_bounded`] / [`parse_manifest`] /
/// [`load_manifest`]. Each variant is a distinct failure class so tests
/// (and callers, via `matches!`) can assert the specific reason a manifest
/// was rejected rather than only `is_err()`.
///
/// Implements `Display`/`Error` and `From<Q4ManifestError> for String`, so
/// existing call sites that propagate errors as `String` via `?` are
/// unaffected — the conversion happens automatically at the `?` site.
#[derive(Debug)]
pub enum Q4ManifestError {
    /// The manifest path exists (as a directory entry — including a
    /// dangling symlink) but could not be stat'd, opened, or read: a
    /// permission error, a broken symlink target, or another I/O failure.
    /// Deliberately distinct from genuine absence: a broken installation
    /// must fail closed, not be treated as "no manifest".
    Unreadable(String),
    /// The file exceeds [`MAX_QUANTIZE_INDEX_LEN`], caught either at stat
    /// time or by the bounded `Take` read overrunning the cap (the latter
    /// guards a file swapped larger between stat and read).
    TooLarge(String),
    /// The file's bytes are not valid JSON at all.
    InvalidJson(String),
    /// Valid JSON, but it does not match either accepted manifest shape
    /// (bare array of tensor entries, or an object with a `tensors`
    /// array), or an entry within it is missing a required field.
    InvalidShape(String),
}

impl std::fmt::Display for Q4ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Q4ManifestError::Unreadable(s)
            | Q4ManifestError::TooLarge(s)
            | Q4ManifestError::InvalidJson(s)
            | Q4ManifestError::InvalidShape(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Q4ManifestError {}

impl From<Q4ManifestError> for String {
    fn from(e: Q4ManifestError) -> String {
        e.to_string()
    }
}

/// One tensor entry in `quantize_index.json`, normalized across both writer
/// shapes, for `doctor`'s tolerant tensor inventory. `quantized`/`shape`/
/// `numel` are `Option` because `doctor` only ever reads `name`/`file` and
/// has always tolerated a manifest (real or hand-written test fixture)
/// that omits the rest; both real writers (`quantize_q4`,
/// `convert_quarot_qwen35`) always emit the full set regardless.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Q4ManifestEntry {
    /// Source tensor name (dotted HF-style path, e.g.
    /// `model.language_model.layers.0.self_attn.q_proj.weight`).
    pub name: String,
    /// Output file name, relative to the manifest's directory.
    pub file: String,
    /// Whether the tensor was quantized to Q4 (`true`) or kept as f16
    /// (`false`).
    #[serde(default)]
    pub quantized: Option<bool>,
    /// Original tensor shape.
    #[serde(default)]
    pub shape: Option<Vec<usize>>,
    /// Original element count.
    #[serde(default)]
    pub numel: Option<usize>,
}

/// Which writer produced a `quantize_index.json`. Both flavors carry the
/// same tensor entry list; only [`ManifestFlavor::QuaRot`] can carry a
/// rotation seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestFlavor {
    /// `bin/quantize_q4.rs`: bare top-level JSON array of tensor entries.
    QuantizeQ4,
    /// `quant::quarot::convert::convert_quarot_qwen35` (ADR-051): JSON
    /// object `{"quarot_seed": ..., "tensors": [...]}`.
    QuaRot,
}

/// Normalized `quantize_index.json` contents for `doctor`'s tensor
/// inventory, independent of on-disk shape. Not used by QuaRot rotation-seed
/// loading — see the module doc for why.
#[derive(Debug, Clone)]
pub struct Q4Manifest {
    pub flavor: ManifestFlavor,
    /// QuaRot's residual-stream Hadamard rotation seed (ADR-051), read
    /// tolerantly here for inventory purposes only. `doctor` doesn't act on
    /// this field; the authoritative, strict seed read is
    /// `quant::quarot::convert::read_quarot_seed_from_index`.
    pub quarot_seed: Option<u64>,
    pub tensors: Vec<Q4ManifestEntry>,
}

/// Read `path`, bounded to [`MAX_QUANTIZE_INDEX_LEN`]. Returns `Ok(None)`
/// only when the manifest path itself is genuinely absent (`ENOENT` on the
/// path, not on a symlink target); any other failure — oversized,
/// permission-denied, a dangling symlink, or another I/O error — is `Err`.
///
/// Uses [`fs::symlink_metadata`] (does not follow symlinks) to decide
/// absence-vs-present, then [`fs::metadata`] (follows symlinks) to read the
/// target's size: a directory entry that exists but is a symlink to a
/// missing target is thereby distinguished from a path with no entry at
/// all — the former is a broken installation and must fail closed, not be
/// silently treated as "no manifest" (a real bug in an earlier version of
/// this reader). Stat-then-take, not a bare [`fs::read`], so a file swapped
/// for something huge between the stat and the read cannot cause an
/// unbounded allocation (#504 remaining slice 2).
pub fn read_manifest_bytes_bounded(path: &Path) -> Result<Option<Vec<u8>>, Q4ManifestError> {
    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(Q4ManifestError::Unreadable(format!(
                "{}: failed to stat quantize_index.json: {e}",
                path.display()
            )));
        }
    }
    // The entry exists (possibly a symlink); resolve it to find its real
    // size. Any failure here (broken symlink target, permission denied on
    // the target, ...) is a hard error, never `Ok(None)`.
    let metadata = fs::metadata(path).map_err(|e| {
        Q4ManifestError::Unreadable(format!(
            "{}: quantize_index.json entry exists but is unreadable \
             (broken symlink or permission error): {e}",
            path.display()
        ))
    })?;
    if metadata.len() > MAX_QUANTIZE_INDEX_LEN {
        return Err(Q4ManifestError::TooLarge(format!(
            "{}: quantize_index.json too large: {} bytes exceeds cap of {MAX_QUANTIZE_INDEX_LEN} bytes",
            path.display(),
            metadata.len()
        )));
    }
    let file = fs::File::open(path).map_err(|e| {
        Q4ManifestError::Unreadable(format!(
            "{}: failed to open quantize_index.json: {e}",
            path.display()
        ))
    })?;
    let mut buf = Vec::new();
    file.take(MAX_QUANTIZE_INDEX_LEN.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| {
            Q4ManifestError::Unreadable(format!(
                "{}: failed to read quantize_index.json: {e}",
                path.display()
            ))
        })?;
    if buf.len() as u64 > MAX_QUANTIZE_INDEX_LEN {
        return Err(Q4ManifestError::TooLarge(format!(
            "{}: quantize_index.json too large: read exceeds cap of {MAX_QUANTIZE_INDEX_LEN} bytes",
            path.display()
        )));
    }
    Ok(Some(buf))
}

/// Parse `bytes` (the contents of a `quantize_index.json`) into a
/// normalized [`Q4Manifest`] for `doctor`'s tolerant tensor inventory,
/// recognizing both writer shapes. **Not** used by QuaRot rotation-seed
/// loading (see module doc) — that reader keeps its own strict schema.
///
/// Uses a two-step parse (untyped [`serde_json::Value`] first, then the
/// matching schema) rather than a `#[serde(untagged)]` enum: untagged
/// deserialization collapses every schema error into "data did not match
/// any variant", which hides the actionable field/path detail (`missing
/// field \`file\``, `invalid type: string, expected a sequence`, ...) that
/// a diagnostic command like `doctor` needs in order to distinguish a
/// genuinely different shape from a corrupted manifest.
pub fn parse_manifest(bytes: &[u8], path: &Path) -> Result<Q4Manifest, Q4ManifestError> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| {
        Q4ManifestError::InvalidJson(format!("{} is not valid JSON: {e}", path.display()))
    })?;
    match value {
        serde_json::Value::Array(_) => serde_json::from_value::<Vec<Q4ManifestEntry>>(value)
            .map(|tensors| Q4Manifest {
                flavor: ManifestFlavor::QuantizeQ4,
                quarot_seed: None,
                tensors,
            })
            .map_err(|e| {
                Q4ManifestError::InvalidShape(format!(
                    "{}: invalid bare-array (quantize_q4) manifest: {e}",
                    path.display()
                ))
            }),
        serde_json::Value::Object(_) => {
            #[derive(serde::Deserialize)]
            struct WrappedManifest {
                #[serde(default)]
                quarot_seed: Option<u64>,
                tensors: Vec<Q4ManifestEntry>,
            }
            serde_json::from_value::<WrappedManifest>(value)
                .map(|w| Q4Manifest {
                    flavor: ManifestFlavor::QuaRot,
                    quarot_seed: w.quarot_seed,
                    tensors: w.tensors,
                })
                .map_err(|e| {
                    Q4ManifestError::InvalidShape(format!(
                        "{}: invalid object-form (quantize_quarot) manifest: {e}",
                        path.display()
                    ))
                })
        }
        _ => Err(Q4ManifestError::InvalidShape(format!(
            "{}: expected quantize_index.json to be either a bare array of tensor \
             entries (quantize_q4) or an object with a \"tensors\" array \
             (quantize_quarot)",
            path.display()
        ))),
    }
}

/// Read + parse `dir/quantize_index.json` in one bounded, fail-closed call,
/// for `doctor`'s tolerant tensor inventory. `Ok(None)` only when the
/// manifest file is genuinely absent (a Q4 directory without a manifest is
/// a valid, if degraded, input — `doctor` falls back to a directory scan
/// for that case).
pub fn load_manifest(dir: &Path) -> Result<Option<Q4Manifest>, Q4ManifestError> {
    let path = dir.join("quantize_index.json");
    let Some(bytes) = read_manifest_bytes_bounded(&path)? else {
        return Ok(None);
    };
    parse_manifest(&bytes, &path).map(Some)
}

#[cfg(test)]
mod tests {
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
    fn load_manifest_rejects_dangling_symlink() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let link_path = tmp.path().join("quantize_index.json");
        symlink(tmp.path().join("does-not-exist"), &link_path).unwrap();

        let err = load_manifest(tmp.path())
            .expect_err("a dangling quantize_index.json symlink must not be treated as absent");
        assert!(matches!(err, Q4ManifestError::Unreadable(_)));
    }
}
