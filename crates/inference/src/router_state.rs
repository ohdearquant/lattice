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
pub const ROUTER_ARTIFACT_FORMAT: u32 = 3;

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

    /// A router directory was configured but holds no artifact at all.
    /// Distinct from not configuring one: the operator asked for a router and
    /// there is none, which is a different state from not asking.
    #[error(
        "--router-state {dir} holds no router artifact. Omit the flag to serve \
         without a router, or point it at a directory holding one"
    )]
    EmptyDirectory {
        /// The directory that was configured.
        dir: PathBuf,
    },

    /// A version was asked for that this directory does not hold.
    #[error("no router artifact for version {version} in {dir}")]
    NotFound {
        /// The directory searched.
        dir: PathBuf,
        /// The version asked for.
        version: u64,
    },

    /// A pin was given with no directory to resolve it in.
    ///
    /// Separate from `NotFound` on purpose. Both mean "the pinned version is
    /// not serving", and they have different remedies: this one is a missing
    /// flag, that one is a missing file. The operator reaching for a pin is
    /// mid-incident and the difference is the whole content of the message.
    #[error(
        "--router-pin {version} was given without --router-state; a pinned version names an \
         artifact in a directory, and no directory was configured"
    )]
    PinWithoutState {
        /// The version that was pinned.
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
    /// The representation the gate consumes. Inside the content hash, like
    /// the names: editing it is a new version, never a mutation of this one.
    pub representation: TrainedRepresentation,
    /// The serialized gate, opaque to this module.
    pub gate_bytes: Vec<u8>,
}

/// The representation a gate was trained on (ADR-094 decision 1, amended).
///
/// Recorded because "the gate was trained on that representation, so this
/// reuses it as trained" was an unwritten contract with no instrument -- the
/// same shape the adapter names were added to close. A mismatch here has no
/// symptom: the gate routes confidently on vectors it never saw, and the only
/// effect is worse selection, which is indistinguishable from a gate that did
/// not learn much.
///
/// `pooling` is what makes this necessary rather than tidy. The two
/// strategies produce DIFFERENT vectors of the SAME length, because the
/// dimension is the checkpoint's hidden size either way, so `input_width`
/// cannot stand in for it.
///
/// A refusal on mismatch, never a warrant of sameness on agreement: two
/// checkpoints can share a name, and fine-tuning changes the representation
/// without changing it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainedRepresentation {
    /// Identity of the embedding model whose output the gate consumes.
    pub embedding_model: String,
    /// Pooling strategy, spelled as `/v1/embeddings` spells it: `mean_visual`
    /// or `last_token`.
    pub pooling: String,
    /// Which text of the request was embedded, spelled as the serving path
    /// spells it: `last_user_message`.
    ///
    /// The third member of the representation, and the one that looks least
    /// like part of it. `embedding_model` and `pooling` say how a text becomes
    /// a vector; they say nothing about WHICH text. A gate trained on the last
    /// user message and served the whole rendered conversation agrees on the
    /// model, agrees on the pooling, and produces a vector of exactly the
    /// right width from different content -- the pooling problem one level
    /// out, with the same absence of any symptom.
    ///
    /// Recorded rather than fixed by convention for the reason the other two
    /// are: a convention has no instrument. The serving path has exactly one
    /// rule today, so this field cannot disagree with it yet, which is
    /// precisely when it is cheap to add and impossible to add later without
    /// invalidating every artifact already written.
    pub prompt_source: String,
    /// The gate's input width as RECORDED at write time.
    ///
    /// Already implied by the gate payload, and stored anyway so a startup
    /// refusal can name recorded against measured the way the hash check
    /// names recorded against computed. A disagreement between this and the
    /// loaded network is an artifact describing itself wrongly, which is a
    /// different fault from a server configured with the wrong embedding
    /// model, and the two have different remedies.
    pub input_width: u64,
}

/// The JSON written beside the payload. Separate from [`RouterArtifact`] so
/// the in-memory type never carries a hash field that could disagree with its
/// own contents: the hash is computed from the artifact, never stored on it.
#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    format: u32,
    version: u64,
    adapter_names: Vec<String>,
    representation: TrainedRepresentation,
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
        // The representation rides inside the hash for the reason the names
        // do: a field stored beside the artifact rather than within it can be
        // edited without producing a new version, which would make a pinned
        // version mean two different things at two different times.
        for field in [
            self.representation.embedding_model.as_str(),
            self.representation.pooling.as_str(),
            self.representation.prompt_source.as_str(),
        ] {
            hasher.update((field.len() as u64).to_le_bytes());
            hasher.update(field.as_bytes());
        }
        hasher.update(self.representation.input_width.to_le_bytes());
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
        representation: artifact.representation.clone(),
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

/// The versions present in `dir`, ascending.
///
/// A version counts as present when its MANIFEST is there. A payload with no
/// manifest is a crash between the two writes and is deliberately invisible
/// here, which is the same reading [`read_artifact`] gives it — the two must
/// agree, or a version would be listed and then refuse to load.
///
/// An unreadable directory is an error rather than an empty list: "no
/// artifacts here" and "I could not look" are different answers, and only one
/// of them means a server should start without a router.
pub fn versions(dir: &Path) -> Result<Vec<u64>, RouterArtifactError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(RouterArtifactError::Read {
                path: dir.to_path_buf(),
                source: BoundedReadErrorDisplay(e.to_string()),
            });
        }
    };

    let mut found = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| RouterArtifactError::Read {
            path: dir.to_path_buf(),
            source: BoundedReadErrorDisplay(e.to_string()),
        })?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(rest) = name.strip_prefix("router-") else {
            continue;
        };
        let Some(digits) = rest.strip_suffix(".json") else {
            continue;
        };
        // Parsed, never trimmed: `router-007.json` and `router-7.json` would
        // otherwise both claim version 7 and one would silently shadow the
        // other. Only the exact spelling this module writes is recognised.
        let Ok(version) = digits.parse::<u64>() else {
            continue;
        };
        if format!("{version}") == digits {
            found.push(version);
        }
    }
    found.sort_unstable();
    Ok(found)
}

/// What a startup should do about routing, given the configured directory.
///
/// This exists so the decision is a VALUE rather than a `process::exit` inside
/// a binary's argument match. A refusal that can only be produced by launching
/// a server is a refusal nobody tests, and this one has to distinguish three
/// states that are easy to collapse into two.
#[derive(Debug)]
pub enum StartupDisposition {
    /// No `--router-state` was given. The server runs without a router, and a
    /// request that omits `lora` selects the base model — today's behaviour,
    /// unchanged.
    NoRouter,
    /// A router directory was given and an artifact loaded from it.
    Loaded(Box<ResolvedRouter>),
}

/// A loaded gate together with how it was selected.
///
/// `pinned` is carried rather than derived because it cannot be derived. An
/// operator who pins version 7 during an incident needs to confirm the pin is
/// live, and the version number alone cannot tell them: a server reporting
/// version 7 is reporting the same number whether it is pinned there or
/// whether 7 simply happens to be the highest version written so far. The two
/// only diverge later, at the next refit and the next restart, which is
/// exactly when nobody is watching and exactly what a pin exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRouter {
    /// The gate artifact that was loaded.
    pub artifact: RouterArtifact,
    /// True when `--router-pin` selected this version, false when it was the
    /// highest version present.
    pub pinned: bool,
}

/// What `GET /v1/lora` reports about the serving gate, borrowed.
///
/// A borrowed view rather than a `ResolvedRouter` so that the reporting path
/// and the routing path can read one artifact. The serving state owns exactly
/// one copy of the gate; handing the reporter its own `ResolvedRouter` would
/// mean two, and two copies of a value that a later refit will replace is a
/// pairing nothing checks — the reported version and the routing version can
/// then disagree with no instrument able to say so.
#[derive(Debug, Clone, Copy)]
pub struct RouterReport<'a> {
    /// The artifact currently serving.
    pub artifact: &'a RouterArtifact,
    /// True when `--router-pin` selected this version.
    pub pinned: bool,
}

impl ResolvedRouter {
    /// Borrow this resolution as a report.
    pub fn report(&self) -> RouterReport<'_> {
        RouterReport {
            artifact: &self.artifact,
            pinned: self.pinned,
        }
    }
}

/// Decide what a startup does about routing.
///
/// `Ok(NoRouter)` only when no directory was configured. A configured
/// directory that is empty, unreadable, or holds an artifact that will not
/// load is an `Err`, and the caller is expected to stop rather than start
/// without a router: those two servers answer the same request differently,
/// and only one of them was asked for. Collapsing them is the failure this
/// function's shape exists to prevent — it is why `NoRouter` is unreachable
/// from any input other than `None`.
/// A pin with no directory refuses rather than being ignored. Ignoring it is
/// the dangerous reading: the operator believes a specific version is serving,
/// the server serves no router at all, and both facts are silent.
pub fn resolve_startup(
    dir: Option<&Path>,
    pin: Option<u64>,
) -> Result<StartupDisposition, RouterArtifactError> {
    let Some(dir) = dir else {
        return match pin {
            None => Ok(StartupDisposition::NoRouter),
            Some(version) => Err(RouterArtifactError::PinWithoutState { version }),
        };
    };
    let resolved = match pin {
        // A pinned version that is absent refuses. Falling back to the latest
        // is the failure this rejects: the pin is reached for precisely when
        // the latest is the thing misbehaving, so a silent fallback serves the
        // artifact the operator was trying to get away from.
        Some(version) => ResolvedRouter {
            artifact: read_artifact(dir, version)?,
            pinned: true,
        },
        None => match load_latest(dir)? {
            Some(artifact) => ResolvedRouter {
                artifact,
                pinned: false,
            },
            None => {
                return Err(RouterArtifactError::EmptyDirectory {
                    dir: dir.to_path_buf(),
                });
            }
        },
    };
    Ok(StartupDisposition::Loaded(Box::new(resolved)))
}

/// Load the highest version present in `dir`.
///
/// `Ok(None)` means the directory holds no artifact at all, which is a server
/// that has never been given a gate. Every other failure is an `Err`: a
/// directory that holds an artifact which will not load must stop a startup,
/// not degrade it to no-router, because those two states serve differently
/// and only one of them was asked for.
pub fn load_latest(dir: &Path) -> Result<Option<RouterArtifact>, RouterArtifactError> {
    match versions(dir)?.last() {
        None => Ok(None),
        Some(&version) => read_artifact(dir, version).map(Some),
    }
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
        representation: manifest.representation,
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

    fn representation() -> TrainedRepresentation {
        TrainedRepresentation {
            embedding_model: "gme-qwen35".into(),
            pooling: "mean_visual".into(),
            prompt_source: "last_user_message".into(),
            input_width: 8,
        }
    }

    fn artifact() -> RouterArtifact {
        RouterArtifact {
            version: 7,
            adapter_names: vec!["legal".into(), "medical".into(), "code".into()],
            representation: representation(),
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
            representation: representation(),
            gate_bytes: vec![9],
        };
        let b = RouterArtifact {
            version: 1,
            adapter_names: vec!["a".into(), "bc".into()],
            representation: representation(),
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
            representation: representation(),
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
    fn versions_lists_what_is_there_ascending_and_load_latest_takes_the_highest() {
        let dir = scratch("select");
        for v in [3u64, 11, 7] {
            write_artifact(
                &dir,
                &RouterArtifact {
                    version: v,
                    adapter_names: vec![format!("a{v}")],
                    representation: representation(),
                    gate_bytes: vec![v as u8],
                },
            )
            .expect("write");
        }
        assert_eq!(versions(&dir).expect("versions"), vec![3, 7, 11]);
        let latest = load_latest(&dir).expect("load").expect("some");
        assert_eq!(
            latest.version, 11,
            "11 must win over 7, not sort as a string"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_or_absent_directory_is_none_not_an_error() {
        // A server that has never been given a gate is a state, not a failure.
        let dir = scratch("empty");
        assert_eq!(versions(&dir).expect("absent dir"), Vec::<u64>::new());
        assert!(load_latest(&dir).expect("absent dir").is_none());
        std::fs::create_dir_all(&dir).expect("mkdir");
        assert!(load_latest(&dir).expect("empty dir").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_directory_holding_an_unloadable_artifact_errors_rather_than_reading_as_no_router() {
        // The distinction step 2 rests on: "no gate configured" and "the gate
        // you configured is broken" must not both start a server quietly.
        let dir = scratch("broken");
        write_artifact(&dir, &artifact()).expect("write");
        let mut bytes = std::fs::read(gate_path(&dir, 7)).expect("read");
        bytes[0] ^= 0xff;
        std::fs::write(gate_path(&dir, 7), &bytes).expect("corrupt");

        match load_latest(&dir) {
            Err(RouterArtifactError::HashMismatch { .. }) => {}
            Ok(None) => panic!("a corrupt artifact read as no-router, which is the bug"),
            other => panic!("expected a hash mismatch, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_payload_without_its_manifest_is_invisible_to_both_readers() {
        // The crash-between-writes state. `versions` and `read_artifact` must
        // agree, or a version gets listed and then refuses to load.
        let dir = scratch("halfwritten");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(gate_path(&dir, 5), [1, 2, 3]).expect("orphan payload");
        assert_eq!(versions(&dir).expect("versions"), Vec::<u64>::new());
        assert!(load_latest(&dir).expect("load").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_the_exact_spelling_this_module_writes_counts_as_a_version() {
        // `router-007.json` would otherwise claim version 7 and shadow the
        // real one, with the shadowing decided by directory order.
        let dir = scratch("spelling");
        write_artifact(&dir, &artifact()).expect("write v7");
        std::fs::write(dir.join("router-007.json"), "{}").expect("decoy");
        std::fs::write(dir.join("router-x.json"), "{}").expect("decoy");
        std::fs::write(dir.join("router-7.json.bak"), "{}").expect("decoy");
        assert_eq!(versions(&dir).expect("versions"), vec![7]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_flag_is_the_only_input_that_yields_no_router() {
        // The whole point of the resolver's shape: NoRouter is unreachable
        // from any configured directory, so "broken router" can never arrive
        // at a server as "no router".
        assert!(matches!(
            resolve_startup(None, None).expect("no flag"),
            StartupDisposition::NoRouter
        ));
    }

    #[test]
    fn a_configured_but_empty_directory_refuses_rather_than_starting_without_a_router() {
        let dir = scratch("startup-empty");
        std::fs::create_dir_all(&dir).expect("mkdir");
        match resolve_startup(Some(&dir), None) {
            Err(RouterArtifactError::EmptyDirectory { .. }) => {}
            Ok(StartupDisposition::NoRouter) => {
                panic!("a configured directory read as no-router, which is the state this refuses")
            }
            other => panic!("expected EmptyDirectory, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_configured_directory_that_does_not_exist_refuses() {
        let dir = scratch("startup-absent");
        match resolve_startup(Some(&dir), None) {
            Err(RouterArtifactError::EmptyDirectory { .. }) => {}
            other => panic!("expected a refusal for an absent configured dir, got {other:?}"),
        }
    }

    #[test]
    fn a_configured_directory_with_a_corrupt_artifact_refuses_with_the_artifacts_own_error() {
        // Not a generic "could not start": the operator needs to know the hash
        // failed, because the remedy differs from an empty directory's.
        let dir = scratch("startup-corrupt");
        write_artifact(&dir, &artifact()).expect("write");
        let mut bytes = std::fs::read(gate_path(&dir, 7)).expect("read");
        bytes[0] ^= 0xff;
        std::fs::write(gate_path(&dir, 7), &bytes).expect("corrupt");

        match resolve_startup(Some(&dir), None) {
            Err(RouterArtifactError::HashMismatch { version, .. }) => assert_eq!(version, 7),
            other => panic!("expected the artifact's own hash error, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_pin_with_no_directory_refuses_and_is_not_ignored() {
        // The dangerous reading is that a pin with nothing to resolve against
        // is harmless. It is the opposite: the operator believes a named
        // version is serving and the server has no router at all, with both
        // facts silent. Distinct from NotFound because the remedies differ --
        // a missing flag, not a missing file.
        match resolve_startup(None, Some(7)) {
            Err(RouterArtifactError::PinWithoutState { version }) => assert_eq!(version, 7),
            Ok(StartupDisposition::NoRouter) => {
                panic!("a pin was ignored, which is the state this refuses")
            }
            other => panic!("expected PinWithoutState, got {other:?}"),
        }
    }

    #[test]
    fn a_pin_selects_its_version_and_not_the_highest() {
        // The arm that separates a pin from a no-op. Two versions present and
        // the pin names the LOWER one, so a resolver that ignored the pin
        // would still return successfully with a valid artifact -- it would
        // just be the wrong one. Pinning to the highest version would pass
        // against both the correct and the broken implementation.
        let dir = scratch("startup-pin-selects");
        write_artifact(&dir, &artifact()).expect("write v7");
        write_artifact(
            &dir,
            &RouterArtifact {
                version: 9,
                adapter_names: vec!["legal".into()],
                representation: representation(),
                gate_bytes: vec![4, 2],
            },
        )
        .expect("write v9");
        match resolve_startup(Some(&dir), Some(7)).expect("load") {
            StartupDisposition::Loaded(r) => {
                assert_eq!(r.artifact.version, 7, "the pin did not select its version");
                assert!(r.pinned, "a pinned load must report itself pinned");
            }
            other => panic!("expected Loaded, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_absent_pinned_version_refuses_rather_than_falling_back_to_the_latest() {
        // A fallback here would serve precisely the artifact the operator was
        // pinning away from, under a flag that says otherwise. The directory
        // deliberately HOLDS a loadable artifact, so a resolver that fell back
        // would succeed and look healthy.
        let dir = scratch("startup-pin-absent");
        write_artifact(&dir, &artifact()).expect("write v7");
        match resolve_startup(Some(&dir), Some(99)) {
            Err(RouterArtifactError::NotFound { version, .. }) => assert_eq!(version, 99),
            Ok(StartupDisposition::Loaded(r)) => panic!(
                "fell back to version {} instead of refusing the absent pin",
                r.artifact.version
            ),
            other => panic!("expected NotFound, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_pinned_corrupt_artifact_refuses_with_the_hash_error_not_the_pin_error() {
        // A pin does not bypass verification, and the refusal must name the
        // real cause: the remedy for a corrupt artifact is not the remedy for
        // a mistyped version.
        let dir = scratch("startup-pin-corrupt");
        write_artifact(&dir, &artifact()).expect("write");
        let mut bytes = std::fs::read(gate_path(&dir, 7)).expect("read");
        bytes[0] ^= 0xff;
        std::fs::write(gate_path(&dir, 7), &bytes).expect("corrupt");
        match resolve_startup(Some(&dir), Some(7)) {
            Err(RouterArtifactError::HashMismatch { version, .. }) => assert_eq!(version, 7),
            other => panic!("expected HashMismatch, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_good_directory_loads_its_highest_version() {
        let dir = scratch("startup-good");
        write_artifact(&dir, &artifact()).expect("write v7");
        write_artifact(
            &dir,
            &RouterArtifact {
                version: 9,
                adapter_names: vec!["legal".into()],
                representation: representation(),
                gate_bytes: vec![4, 2],
            },
        )
        .expect("write v9");
        match resolve_startup(Some(&dir), None).expect("load") {
            StartupDisposition::Loaded(r) => {
                assert_eq!(r.artifact.version, 9);
                assert!(!r.pinned, "an unpinned load must not report itself pinned");
            }
            other => panic!("expected Loaded, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_representation_is_inside_the_content_hash() {
        // Leo's ruling, and the reason is the pin. A field stored BESIDE the
        // artifact could be edited without producing a new version, so a
        // pinned version would mean two different things at two different
        // times. Each field is varied alone, because a hash that covered only
        // one of them would still pass a test that changed all of them at once.
        //
        // The fields are ENUMERATED from the serialized struct rather than
        // listed here. The first version of this test listed three closures,
        // one per field, and when `prompt_source` was added the list stayed at
        // three: the new field rode outside the hash and this test passed. A
        // hand-written list of what a struct contains is a claim nobody
        // re-derives when they add to the struct, so it decays in exactly the
        // direction that reads as coverage.
        let base = artifact();
        let value =
            serde_json::to_value(&base.representation).expect("the representation serializes");
        let fields = value
            .as_object()
            .expect("a struct serializes to a JSON object");
        assert!(
            !fields.is_empty(),
            "no fields were enumerated, so the loop below asserts nothing"
        );

        for (name, original) in fields {
            let changed = match original {
                serde_json::Value::String(text) => serde_json::Value::String(format!("{text}-x")),
                serde_json::Value::Number(number) => {
                    let n = number
                        .as_u64()
                        .expect("a numeric field is an unsigned count");
                    serde_json::json!(n + 1)
                }
                other => panic!("field {name} has type {other:?}, which this arm cannot vary"),
            };
            let mut mutated = fields.clone();
            mutated.insert(name.clone(), changed);
            let representation: TrainedRepresentation =
                serde_json::from_value(serde_json::Value::Object(mutated))
                    .expect("the varied representation deserializes");

            let mut other = base.clone();
            other.representation = representation;
            assert_ne!(
                base.content_hash(),
                other.content_hash(),
                "representation field {name} changed without changing the hash"
            );
        }

        // The must-match control: an untouched copy hashes identically, so the
        // inequalities above are about the edits and not about instability.
        assert_eq!(base.content_hash(), base.clone().content_hash());
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
