//! Shared parsing and validation for `quantize_index.json`, the per-tensor
//! manifest written next to a Q4 checkpoint's `.q4`/`.f16` files.
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
//! Before this module, three independent call sites each re-derived the
//! shape-normalization and size-bounding logic (`lattice doctor`'s
//! `inspect_q4_dir`, QuaRot seed loading in `quant::quarot::convert`, and by
//! extension the Metal Q4 loader that calls into it) — see issue #655. One
//! reader here now backs all of them: [`read_manifest_bytes_bounded`] for the
//! bounded, fail-closed file read, and [`parse_manifest`] /
//! [`load_manifest`] for shape-normalized parsing into [`Q4Manifest`].

use std::fs;
use std::io::Read;
use std::path::Path;

/// Bounded size cap for `quantize_index.json` reads (#504 remaining slice
/// 2). The index is a small per-tensor manifest (name/file/quantized/
/// shape/numel per output tensor, plus an optional rotation seed); even a
/// multi-thousand-tensor checkpoint stays well under this cap.
pub const MAX_QUANTIZE_INDEX_LEN: u64 = 16 * 1024 * 1024; // 16 MiB

/// One tensor entry in `quantize_index.json`, normalized across both writer
/// shapes. `quantized`/`shape`/`numel` are `Option` because a hand-written
/// manifest (as used by some doctor-inventory call sites and test
/// fixtures) may omit them; both real writers (`quantize_q4`,
/// `convert_quarot_qwen35`) always emit the full set.
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

/// Normalized `quantize_index.json` contents, independent of on-disk shape.
#[derive(Debug, Clone)]
pub struct Q4Manifest {
    pub flavor: ManifestFlavor,
    /// QuaRot's residual-stream Hadamard rotation seed (ADR-051). Always
    /// `None` for [`ManifestFlavor::QuantizeQ4`]; may also be `None` for
    /// [`ManifestFlavor::QuaRot`] manifests written before the seed field
    /// was added.
    pub quarot_seed: Option<u64>,
    pub tensors: Vec<Q4ManifestEntry>,
}

/// Read `path`, bounded to [`MAX_QUANTIZE_INDEX_LEN`]. Returns `Ok(None)`
/// only when the file is genuinely absent; any other failure (oversized,
/// unreadable) is `Err`.
///
/// Stat-then-take, not a bare [`fs::read`], so a file swapped for something
/// huge between the stat and the read cannot cause an unbounded
/// allocation (#504 remaining slice 2).
pub fn read_manifest_bytes_bounded(path: &Path) -> Result<Option<Vec<u8>>, String> {
    let metadata = match fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(format!(
                "{}: failed to stat quantize_index.json: {e}",
                path.display()
            ));
        }
    };
    if metadata.len() > MAX_QUANTIZE_INDEX_LEN {
        return Err(format!(
            "{}: quantize_index.json too large: {} bytes exceeds cap of {MAX_QUANTIZE_INDEX_LEN} bytes",
            path.display(),
            metadata.len()
        ));
    }
    let file = fs::File::open(path).map_err(|e| {
        format!(
            "{}: failed to open quantize_index.json: {e}",
            path.display()
        )
    })?;
    let mut buf = Vec::new();
    file.take(MAX_QUANTIZE_INDEX_LEN.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| {
            format!(
                "{}: failed to read quantize_index.json: {e}",
                path.display()
            )
        })?;
    if buf.len() as u64 > MAX_QUANTIZE_INDEX_LEN {
        return Err(format!(
            "{}: quantize_index.json too large: read exceeds cap of {MAX_QUANTIZE_INDEX_LEN} bytes",
            path.display()
        ));
    }
    Ok(Some(buf))
}

/// Parse `bytes` (the contents of a `quantize_index.json`) into a
/// normalized [`Q4Manifest`], recognizing both writer shapes.
///
/// Uses a two-step parse (untyped [`serde_json::Value`] first, then the
/// matching schema) rather than a `#[serde(untagged)]` enum: untagged
/// deserialization collapses every schema error into "data did not match
/// any variant", which hides the actionable field/path detail (`missing
/// field \`file\``, `invalid type: string, expected a sequence`, ...) that
/// both a diagnostic command (`doctor`) and a fail-closed loader (QuaRot
/// seed detection) need in order to distinguish a genuinely different
/// shape from a corrupted manifest.
pub fn parse_manifest(bytes: &[u8], path: &Path) -> Result<Q4Manifest, String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;
    match value {
        serde_json::Value::Array(_) => serde_json::from_value::<Vec<Q4ManifestEntry>>(value)
            .map(|tensors| Q4Manifest {
                flavor: ManifestFlavor::QuantizeQ4,
                quarot_seed: None,
                tensors,
            })
            .map_err(|e| {
                format!(
                    "{}: invalid bare-array (quantize_q4) manifest: {e}",
                    path.display()
                )
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
                    format!(
                        "{}: invalid object-form (quantize_quarot) manifest: {e}",
                        path.display()
                    )
                })
        }
        _ => Err(format!(
            "{}: expected quantize_index.json to be either a bare array of tensor \
             entries (quantize_q4) or an object with a \"tensors\" array \
             (quantize_quarot)",
            path.display()
        )),
    }
}

/// Read + parse `dir/quantize_index.json` in one bounded, fail-closed call.
/// `Ok(None)` only when the manifest file is genuinely absent (a Q4
/// directory without a manifest is a valid, if degraded, input — callers
/// fall back to a directory scan or a legacy `config.json` field).
pub fn load_manifest(dir: &Path) -> Result<Option<Q4Manifest>, String> {
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
    fn parse_manifest_malformed_bare_array_error_names_the_missing_field() {
        let err = parse_manifest(br#"[{"name": "x"}]"#, Path::new("quantize_index.json"))
            .expect_err("bare-array entry missing `file` must fail");
        assert!(
            err.contains("file"),
            "error must name the missing `file` field; got: {err}"
        );
        assert!(
            !err.contains("did not match any variant"),
            "error must not be the generic untagged-enum fallthrough; got: {err}"
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
            err.contains("tensors") || err.contains("sequence"),
            "error must point at the bad `tensors` value; got: {err}"
        );
        assert!(
            !err.contains("did not match any variant"),
            "error must not be the generic untagged-enum fallthrough; got: {err}"
        );
    }

    #[test]
    fn parse_manifest_non_array_non_object_root_is_rejected_with_shape_hint() {
        let err = parse_manifest(br#""just a string""#, Path::new("quantize_index.json"))
            .expect_err("scalar manifest root must fail");
        assert!(
            err.contains("either a bare array") && err.contains("tensors"),
            "error must explain both accepted shapes; got: {err}"
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
            err.contains("too large"),
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
        assert!(
            load_manifest(tmp.path()).is_err(),
            "truncated quantize_index.json must be rejected"
        );
    }
}
