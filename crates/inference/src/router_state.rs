//! The on-disk router gate artifact: what a learned gate is when it is not in
//! memory (ADR-095 decision 3, ADR-094 decision 2 as amended 2026-09-21).
//!
//! A gate that does not survive a restart is not learning, so the gate is
//! written to a versioned artifact. Two things ride in that artifact besides
//! the network payload, and they are why this module exists rather than a
//! bare `fs::write` of `Network::to_bytes()`:
//!
//! **The adapter names the gate was trained on.** `AdapterRouter::route` maps
//! a gate output column to an adapter by position — the selected index is used
//! directly as `available[idx]` — and the `AdapterId` string is a label copied
//! into the result, never matched against anything the gate holds. So the
//! correspondence between columns and adapters is a contract owned entirely by
//! whoever builds the caller's slice, and nothing can check it: a wrong pairing
//! yields valid weights, no error, and the wrong adapter. Carrying the trained
//! names in the artifact is what makes that pairing checkable at all, and the
//! serving path refuses on mismatch rather than routing on an assumption.
//!
//! **A version, which is the counter and the content hash together.** The
//! counter orders refits and is what an operator pins; the hash is what makes
//! the counter mean something. A counter alone can be reused by a hand-edited
//! file, and a hash alone has no order, so neither is a version by itself.
//! Because the hash covers the names as well as the payload, editing a name
//! changes the version — which is the property the refusal above depends on,
//! since a name list that could be edited without a version change would
//! reintroduce the unverifiable pairing one level up.
//!
//! This module is deliberately free of the `mixture` gate. The serving path's
//! façade must be able to name a version in its refusal even on a build with
//! no router compiled in, and the gate payload is opaque bytes here — nothing
//! in this file needs a `lattice-fann` type to read or write it.
//!
//! Scope: this is the write and read half. Choosing WHICH version a server
//! loads is startup's job, not this module's; artifacts are kept beside each
//! other under their own versions precisely so that choice stays external.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::bounded_read::{BoundedReadError, read_bytes_bounded};

/// On-disk format revision for the manifest. Bumped only for a change that an
/// older reader would misread; a reader refuses a revision it does not know
/// rather than parsing a prefix of it.
pub const ROUTER_ARTIFACT_FORMAT: u32 = 1;

/// Size cap for the manifest read. The manifest is a small JSON object whose
/// only unbounded field is the adapter-name list.
pub const MAX_ROUTER_MANIFEST_LEN: u64 = 1024 * 1024;

/// Size cap for the gate payload read. A routing gate is a small fully
/// connected network; this is well above any gate the refit path produces and
/// exists so a corrupt or hostile length cannot drive an unbounded read.
pub const MAX_ROUTER_GATE_LEN: u64 = 64 * 1024 * 1024;

/// Every way reading or writing an artifact can fail, one variant per class so
/// a caller can branch on the reason rather than on `is_err()`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RouterArtifactError {
    /// The manifest or payload could not be read within its cap.
    #[error("reading {path}: {source}")]
    Read {
        /// The path whose read failed.
        path: PathBuf,
        /// The underlying bounded-read failure.
        source: BoundedReadErrorDisplay,
    },

    /// The manifest was present and readable but did not parse as JSON.
    #[error("manifest {path} did not parse: {message}")]
    Malformed {
        /// The manifest path.
        path: PathBuf,
        /// The parser's own message.
        message: String,
    },

    /// The manifest declares a format revision this build does not know.
    #[error(
        "manifest {path} declares format revision {found}, and this build knows \
         revision {known}; a newer artifact is not read as an older one"
    )]
    UnknownFormat {
        /// The manifest path.
        path: PathBuf,
        /// The revision the file declares.
        found: u32,
        /// The revision this build understands.
        known: u32,
    },

    /// The recomputed content hash does not match the manifest's. The payload,
    /// the name list, or both were changed after the artifact was written.
    #[error(
        "artifact version {version} in {dir} failed its content hash: manifest \
         records {recorded}, contents hash to {computed}. The gate payload or the \
         trained-adapter list was modified after this version was written, so the \
         column-to-adapter correspondence it claims cannot be trusted"
    )]
    HashMismatch {
        /// The directory holding the artifact.
        dir: PathBuf,
        /// The version counter that failed.
        version: u64,
        /// The hash the manifest carries.
        recorded: String,
        /// The hash the bytes on disk actually produce.
        computed: String,
    },

    /// Writing the artifact failed.
    #[error("writing {path}: {message}")]
    Write {
        /// The path whose write failed.
        path: PathBuf,
        /// The underlying error's message.
        message: String,
    },

    /// A version was asked for that this directory does not hold.
    #[error("no router artifact for version {version} in {dir}")]
    NotFound {
        /// The directory searched.
        dir: PathBuf,
        /// The version asked for.
        version: u64,
    },
}

/// `BoundedReadError` is crate-internal and not an `Error`, so it cannot be a
/// `#[source]` directly. This wrapper carries its rendered text, which is what
/// a reader of the refusal needs, without widening that type's visibility.
#[derive(Debug)]
pub struct BoundedReadErrorDisplay(String);

impl std::fmt::Display for BoundedReadErrorDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BoundedReadErrorDisplay {}

impl From<BoundedReadError> for BoundedReadErrorDisplay {
    fn from(err: BoundedReadError) -> Self {
        Self(match err {
            BoundedReadError::NotRegularFile => "not a regular file".to_string(),
            BoundedReadError::TooLarge { len, cap } => {
                format!("{len} bytes exceeds the {cap} byte cap")
            }
            BoundedReadError::Io(e) => e.to_string(),
        })
    }
}

/// A router gate as it exists on disk: the payload, the adapter names its
/// columns correspond to, and the counter half of its version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterArtifact {
    /// Monotonic refit counter. Orders versions and is what an operator pins.
    pub version: u64,
    /// The adapter names this gate was trained on, in gate column order. The
    /// serving path matches against this by name; the order here is how the
    /// artifact records which column was which, never how a caller must sort.
    pub adapter_names: Vec<String>,
    /// The serialized gate, opaque to this module.
    pub gate_bytes: Vec<u8>,
}

/// The JSON written beside the payload. Separate from [`RouterArtifact`] so
/// the in-memory type never carries a hash field that could disagree with its
/// own contents: the hash is computed from the artifact, never stored on it.
#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    format: u32,
    version: u64,
    adapter_names: Vec<String>,
    gate_len: u64,
    content_sha256: String,
}

impl RouterArtifact {
    /// The content hash: SHA-256 over the adapter names and the gate payload.
    ///
    /// Each name is length-prefixed before hashing. Concatenating them plainly
    /// would let two different lists hash identically — `["ab", "c"]` and
    /// `["a", "bc"]` produce the same bytes — and those two lists describe
    /// different column-to-adapter correspondences, which is exactly the
    /// confusion this hash exists to detect.
    pub fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update((self.adapter_names.len() as u64).to_le_bytes());
        for name in &self.adapter_names {
            hasher.update((name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
        }
        hasher.update((self.gate_bytes.len() as u64).to_le_bytes());
        hasher.update(&self.gate_bytes);
        hex_lower(&hasher.finalize())
    }

    /// The version as an operator reads it: counter and hash together, because
    /// neither half identifies a gate on its own.
    pub fn version_label(&self) -> String {
        let hash = self.content_hash();
        format!("{}:{}", self.version, &hash[..16])
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Path of the manifest for `version` in `dir`.
pub fn manifest_path(dir: &Path, version: u64) -> PathBuf {
    dir.join(format!("router-{version}.json"))
}

/// Path of the gate payload for `version` in `dir`.
pub fn gate_path(dir: &Path, version: u64) -> PathBuf {
    dir.join(format!("router-{version}.gate"))
}

/// Write `artifact` into `dir` under its own version, leaving any other
/// version already there untouched.
///
/// Versions live beside each other rather than overwriting a single current
/// file, so the previous gate is still on disk after a refit and a rollback
/// does not need a backup that someone remembered to take.
///
/// The payload is written before the manifest. A crash between the two leaves
/// a payload with no manifest, which every reader here treats as absent; the
/// reverse order would leave a manifest promising a payload that is not there,
/// which reads as corruption of a version that was never written.
pub fn write_artifact(
    dir: &Path,
    artifact: &RouterArtifact,
) -> Result<PathBuf, RouterArtifactError> {
    std::fs::create_dir_all(dir).map_err(|e| RouterArtifactError::Write {
        path: dir.to_path_buf(),
        message: e.to_string(),
    })?;

    let gate = gate_path(dir, artifact.version);
    std::fs::write(&gate, &artifact.gate_bytes).map_err(|e| RouterArtifactError::Write {
        path: gate.clone(),
        message: e.to_string(),
    })?;

    let manifest = Manifest {
        format: ROUTER_ARTIFACT_FORMAT,
        version: artifact.version,
        adapter_names: artifact.adapter_names.clone(),
        gate_len: artifact.gate_bytes.len() as u64,
        content_sha256: artifact.content_hash(),
    };
    let json = serde_json::to_vec_pretty(&manifest).map_err(|e| RouterArtifactError::Write {
        path: manifest_path(dir, artifact.version),
        message: e.to_string(),
    })?;

    let path = manifest_path(dir, artifact.version);
    std::fs::write(&path, &json).map_err(|e| RouterArtifactError::Write {
        path: path.clone(),
        message: e.to_string(),
    })?;
    Ok(path)
}

/// Read the artifact for `version` from `dir`, refusing anything whose
/// contents do not hash to what its manifest recorded.
pub fn read_artifact(dir: &Path, version: u64) -> Result<RouterArtifact, RouterArtifactError> {
    let manifest_file = manifest_path(dir, version);
    if !manifest_file.exists() {
        return Err(RouterArtifactError::NotFound {
            dir: dir.to_path_buf(),
            version,
        });
    }

    let raw = read_bytes_bounded(&manifest_file, MAX_ROUTER_MANIFEST_LEN).map_err(|e| {
        RouterArtifactError::Read {
            path: manifest_file.clone(),
            source: e.into(),
        }
    })?;
    let manifest: Manifest =
        serde_json::from_slice(&raw).map_err(|e| RouterArtifactError::Malformed {
            path: manifest_file.clone(),
            message: e.to_string(),
        })?;

    if manifest.format != ROUTER_ARTIFACT_FORMAT {
        return Err(RouterArtifactError::UnknownFormat {
            path: manifest_file,
            found: manifest.format,
            known: ROUTER_ARTIFACT_FORMAT,
        });
    }

    let gate_file = gate_path(dir, version);
    let gate_bytes = read_bytes_bounded(&gate_file, MAX_ROUTER_GATE_LEN).map_err(|e| {
        RouterArtifactError::Read {
            path: gate_file,
            source: e.into(),
        }
    })?;

    let artifact = RouterArtifact {
        version: manifest.version,
        adapter_names: manifest.adapter_names,
        gate_bytes,
    };

    let computed = artifact.content_hash();
    if computed != manifest.content_sha256 {
        return Err(RouterArtifactError::HashMismatch {
            dir: dir.to_path_buf(),
            version,
            recorded: manifest.content_sha256,
            computed,
        });
    }

    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> RouterArtifact {
        RouterArtifact {
            version: 7,
            adapter_names: vec!["legal".into(), "medical".into(), "code".into()],
            gate_bytes: vec![1, 2, 3, 4, 5],
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lattice-router-artifact-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_written_artifact_reads_back_byte_identical() {
        let dir = scratch("roundtrip");
        let original = artifact();
        write_artifact(&dir, &original).expect("write");
        let read = read_artifact(&dir, 7).expect("read");
        assert_eq!(read, original);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_modified_payload_is_refused_rather_than_loaded() {
        let dir = scratch("payload");
        write_artifact(&dir, &artifact()).expect("write");

        // One byte, the smallest change a corrupt or edited payload can be.
        let mut bytes = std::fs::read(gate_path(&dir, 7)).expect("read gate");
        bytes[0] ^= 0xff;
        std::fs::write(gate_path(&dir, 7), &bytes).expect("rewrite gate");

        match read_artifact(&dir, 7) {
            Err(RouterArtifactError::HashMismatch { version, .. }) => assert_eq!(version, 7),
            other => panic!("expected a hash mismatch, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn editing_an_adapter_name_changes_the_version_and_is_refused() {
        // The property the serving path's name match depends on: a name list
        // that could be edited without changing the version would reintroduce
        // the unverifiable pairing one level up, in the manifest instead of in
        // the caller's slice.
        let dir = scratch("names");
        write_artifact(&dir, &artifact()).expect("write");

        let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
        assert!(
            raw.contains("medical"),
            "fixture must contain the name it edits"
        );
        let edited = raw.replace("medical", "finance");
        std::fs::write(manifest_path(&dir, 7), edited).expect("rewrite manifest");

        match read_artifact(&dir, 7) {
            Err(RouterArtifactError::HashMismatch {
                recorded, computed, ..
            }) => {
                assert_ne!(recorded, computed);
            }
            other => panic!("expected a hash mismatch, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_hash_covers_the_names_so_two_name_lists_never_collide() {
        // Length-prefixing is what makes this true. Without it these two lists
        // hash the same bytes, and they describe different column-to-adapter
        // correspondences -- the exact confusion the hash exists to catch.
        let a = RouterArtifact {
            version: 1,
            adapter_names: vec!["ab".into(), "c".into()],
            gate_bytes: vec![9],
        };
        let b = RouterArtifact {
            version: 1,
            adapter_names: vec!["a".into(), "bc".into()],
            gate_bytes: vec![9],
        };
        assert_ne!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn reordering_the_names_changes_the_hash() {
        // Order is how the artifact records which column was which, so two
        // orders are two different claims and must not share a version.
        let mut reordered = artifact();
        reordered.adapter_names.swap(0, 2);
        assert_ne!(artifact().content_hash(), reordered.content_hash());
    }

    #[test]
    fn writing_a_new_version_leaves_the_previous_one_readable() {
        // Rollback needs the previous gate to still be on disk, without anyone
        // having remembered to take a backup.
        let dir = scratch("versions");
        write_artifact(&dir, &artifact()).expect("write v7");
        let next = RouterArtifact {
            version: 8,
            adapter_names: vec!["legal".into(), "medical".into(), "code".into()],
            gate_bytes: vec![6, 7, 8],
        };
        write_artifact(&dir, &next).expect("write v8");

        assert_eq!(
            read_artifact(&dir, 7).expect("v7 still readable"),
            artifact()
        );
        assert_eq!(read_artifact(&dir, 8).expect("v8 readable"), next);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unknown_format_revision_is_refused_not_parsed_as_this_one() {
        let dir = scratch("format");
        write_artifact(&dir, &artifact()).expect("write");
        let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
        let bumped = raw.replace(
            &format!("\"format\": {ROUTER_ARTIFACT_FORMAT}"),
            &format!("\"format\": {}", ROUTER_ARTIFACT_FORMAT + 1),
        );
        assert_ne!(bumped, raw, "the format field must have been rewritten");
        std::fs::write(manifest_path(&dir, 7), bumped).expect("rewrite");

        match read_artifact(&dir, 7) {
            Err(RouterArtifactError::UnknownFormat { found, known, .. }) => {
                assert_eq!(found, ROUTER_ARTIFACT_FORMAT + 1);
                assert_eq!(known, ROUTER_ARTIFACT_FORMAT);
            }
            other => panic!("expected an unknown-format refusal, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_absent_version_is_not_found_rather_than_a_read_failure() {
        // Absent and corrupt must not be one answer: a server that has never
        // been given a gate and one whose gate was truncated need different
        // operator responses.
        let dir = scratch("absent");
        write_artifact(&dir, &artifact()).expect("write");
        match read_artifact(&dir, 99) {
            Err(RouterArtifactError::NotFound { version, .. }) => assert_eq!(version, 99),
            other => panic!("expected NotFound, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_version_label_carries_both_halves() {
        let a = artifact();
        let label = a.version_label();
        assert!(label.starts_with("7:"), "counter half missing from {label}");
        assert_eq!(
            label.len(),
            "7:".len() + 16,
            "hash half wrong width in {label}"
        );
        let mut changed = a.clone();
        changed.gate_bytes.push(0);
        assert_ne!(a.version_label(), changed.version_label());
    }
}
