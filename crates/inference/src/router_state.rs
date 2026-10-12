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

/// On-disk format revision written for an artifact that records its own
/// policy and lineage (ADR-096 decision 5). Bumped only for a change that an
/// older reader would misread; a reader refuses a revision it does not know
/// rather than parsing a prefix of it.
pub const ROUTER_ARTIFACT_FORMAT: u32 = 5;

/// The revision written before policy and lineage were recorded. Still read,
/// and still written for an artifact whose provenance is
/// [`RouterProvenance::Legacy`], because the revision is a property of what the
/// artifact carries rather than of the build that wrote it. ADR-096 decision 5
/// keeps these servable and explicitly uniform; they never acquire learner
/// eligibility by having their missing fields defaulted.
pub const ROUTER_ARTIFACT_FORMAT_LEGACY: u32 = 4;

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
         revisions {known}; a newer artifact is not read as an older one"
    )]
    UnknownFormat {
        /// The manifest path.
        path: PathBuf,
        /// The revision the file declares.
        found: u32,
        /// The revisions this build understands, as an operator-readable list.
        known: String,
    },

    /// A manifest's declared revision and its recorded policy disagree. Both
    /// directions are refused, and neither is repaired by inference: a
    /// revision-5 manifest missing its policy must not acquire one by default,
    /// which is how a legacy artifact would quietly become learner-eligible,
    /// and a revision-4 manifest carrying one was written by something that
    /// did not know what revision 4 means.
    #[error(
        "manifest {path} declares format revision {found} but {problem}; the \
         recorded policy and the declared revision are one statement, so this \
         pair is refused rather than one half being preferred"
    )]
    ProvenanceMismatch {
        /// The manifest path.
        path: PathBuf,
        /// The revision the file declares.
        found: u32,
        /// What disagreed with it, as a sentence fragment.
        problem: String,
    },

    /// The head record is present but could not be read or parsed. This is an
    /// error and never an absence: an absent head selects by counter, so a
    /// truncated head that read as absent would silently reinstate the rule
    /// the head exists to replace, and it would do it on the one artifact
    /// whose admission was interrupted.
    #[error("head record {path} is present but unreadable: {message}")]
    HeadUnreadable {
        /// The head record's path.
        path: PathBuf,
        /// The reader's or parser's own message.
        message: String,
    },

    /// The head names a version that is not in the directory. The artifact was
    /// removed under a live head, and serving the next-highest instead is the
    /// promotion the head exists to prevent.
    #[error(
        "head record {path} names version {version}, which is not present in \
         the directory; a head is a statement that a specific artifact was \
         admitted, so a missing one refuses rather than falling back"
    )]
    HeadNamesAbsentVersion {
        /// The head record's path.
        path: PathBuf,
        /// The version it names.
        version: u64,
    },

    /// The head names a version whose content hash is not the one it recorded.
    /// The counter was reused: a later write of the same number produced a
    /// different artifact, and the head is now pointing at something that was
    /// never admitted.
    #[error(
        "head record {path} names version {version} with content hash \
         {head_hash}, but that version now hashes to {artifact_hash}; the \
         admitted artifact was replaced under a live head"
    )]
    HeadDisagreesWithArtifact {
        /// The head record's path.
        path: PathBuf,
        /// The version named.
        version: u64,
        /// The hash the head recorded at admission.
        head_hash: String,
        /// The hash the artifact on disk has now.
        artifact_hash: String,
    },

    /// A pin naming a full version identity resolved to an artifact with a
    /// different one. The counter matched and the artifact did not.
    #[error(
        "--router-pin asked for {requested} and version {version} is \
         {found}; a pin that names an identity is refused rather than \
         served by counter alone"
    )]
    PinIdentityMismatch {
        /// The counter both halves agree on.
        version: u64,
        /// The identity the operator asked for.
        requested: String,
        /// The identity the artifact actually has.
        found: String,
    },

    /// A pin string is not a counter or a counter and hash prefix.
    #[error("--router-pin value {text:?} is not usable: {problem}")]
    PinMalformed {
        /// What the operator typed.
        text: String,
        /// Why it cannot be resolved.
        problem: String,
    },

    /// A recorded policy is not internally consistent. Checked at write and at
    /// read, because an artifact is pinned and served long after the process
    /// that produced it is gone.
    #[error("router policy is invalid: {problem}")]
    InvalidPolicy {
        /// What is wrong with it.
        problem: String,
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

    /// The manifest's own version counter disagrees with the one in its
    /// filename. Both are read: the directory scan lists versions by filename,
    /// and the loaded artifact reports the manifest's. A disagreement means an
    /// operator can pin one number and be served an artifact that calls itself
    /// another, so it is refused rather than resolved in either direction.
    #[error(
        "manifest {path} declares version {declared}, but its filename names          version {from_filename}; a pinned version must mean one artifact, so          this pair is refused rather than one half being preferred"
    )]
    VersionMismatch {
        /// The manifest path, whose name carries the other number.
        path: PathBuf,
        /// The counter the manifest's own field declares.
        declared: u64,
        /// The counter the filename declares.
        from_filename: u64,
    },

    /// The manifest's declared payload length disagrees with the payload's
    /// actual length. A declared size nothing compares is a field that cannot
    /// detect the truncation it exists to describe.
    #[error(
        "manifest {path} declares a {declared}-byte gate payload, but          {payload} holds {actual} bytes; the artifact is truncated or the          manifest belongs to a different payload"
    )]
    GateLengthMismatch {
        /// The manifest path.
        path: PathBuf,
        /// The payload path.
        payload: PathBuf,
        /// The length the manifest declares.
        declared: u64,
        /// The length the payload actually has.
        actual: u64,
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

/// Policy kinds a recorded artifact may name, spelled as `mixture::WeightPolicy`
/// spells them: `uniform` and `softmax` are its two variants. A closed set
/// rather than a free string: the serving side matches on it, and an
/// unrecognised spelling there would have to resolve to something, which is how
/// "unknown" becomes "uniform" without anyone deciding it.
pub const ROUTER_POLICY_KINDS: [&str; 2] = ["uniform", "softmax"];

/// Training states a recorded artifact may name. `bootstrap` is the explicitly
/// untrained baseline of ADR-096 decision 2; it is legitimate to serve and is
/// not evidence of learning, and the distinction only survives if it is written
/// down.
pub const ROUTER_TRAINING_STATES: [&str; 2] = ["bootstrap", "learned"];

/// The complete routing policy an artifact was admitted under.
///
/// ADR-096 decision 5: a version is a policy, not a network blob. Loading the
/// same gate bytes under a different temperature is a different behaviour, so
/// if the policy lived beside the artifact rather than inside its hash, one
/// version label would name two behaviours and a pin would stop meaning
/// anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouterPolicy {
    /// One of [`ROUTER_POLICY_KINDS`].
    pub kind: String,
    /// Softmax temperature. `Some` exactly when `kind` is `softmax`: a
    /// temperature on a uniform policy is a number nothing reads, and a
    /// softmax without one has no defined behaviour.
    pub tau: Option<f64>,
    /// The drop floor applied to the weight distribution (ADR-091 decision 2).
    pub epsilon: f64,
    /// Lower bound on `tau` (ADR-091 guard (a)). A temperature free to fall
    /// toward zero is an argmax with extra steps, so the bound is part of the
    /// policy rather than of the process that serves it.
    pub tau_floor: f64,
    /// Entropy below which a round counts as collapsed.
    pub entropy_floor: f64,
    /// How many consecutive collapsed rounds are tolerated before the guard
    /// fires.
    pub max_consecutive_collapsed: u32,
    /// One of [`ROUTER_TRAINING_STATES`].
    pub training_state: String,
    /// Identity of the objective this artifact was trained under. `None` for a
    /// bootstrap, which was not trained under one.
    pub objective_version: Option<String>,
}

/// Where an artifact came from: its schema, its parent, and the admission
/// record that let it be served.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterLineage {
    /// Identity of the ordered adapter schema (ADR-096 decision 1). Two
    /// artifacts sharing this describe the same columns in the same order over
    /// the same base model; a change to it ends the lineage.
    pub schema_id: String,
    /// The base model this lineage routes over. A gate trained against one
    /// base and served on another is routing on a representation produced by
    /// different weights, which no width or name check can see.
    pub base_model: String,
    /// The version label of the artifact this one was refit from. `None` for
    /// the first artifact in a lineage.
    pub parent: Option<String>,
    /// Identity of the admission record that admitted this candidate (ADR-096
    /// decision 5). `None` for a bootstrap, which is checked for structure and
    /// provenance rather than admitted on evidence.
    pub admission_record: Option<String>,
}

/// What an artifact records about itself beyond its gate and its names.
///
/// An enum rather than two `Option` fields, because the absent case has to be
/// unusable rather than defaultable. ADR-096 decision 5 is explicit that a
/// legacy artifact must not acquire learner eligibility by having its missing
/// fields filled in, and a pair of `Option`s invites exactly that at every
/// call site that reads them.
#[derive(Debug, Clone, PartialEq)]
pub enum RouterProvenance {
    /// Written before policy and lineage were recorded. Serves under the
    /// compiled-in uniform default and is never learner-eligible.
    Legacy,
    /// Records its own policy and lineage.
    ///
    /// Boxed because the recorded half is an order of magnitude larger than the
    /// legacy one, and every `RouterArtifact` pays the larger size whether or
    /// not it records anything. Legacy artifacts are the ones that exist today.
    Recorded(Box<RecordedProvenance>),
}

/// The policy and lineage a recorded artifact carries, kept behind one box so
/// the enum's two variants are close in size.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedProvenance {
    /// The policy this artifact was admitted under.
    pub policy: RouterPolicy,
    /// Where it came from.
    pub lineage: RouterLineage,
}

impl RouterProvenance {
    /// Build a recorded provenance without every call site naming the box.
    pub fn recorded(policy: RouterPolicy, lineage: RouterLineage) -> Self {
        Self::Recorded(Box::new(RecordedProvenance { policy, lineage }))
    }

    /// The recorded lineage, or `None` for a legacy artifact.
    pub fn lineage(&self) -> Option<&RouterLineage> {
        match self {
            Self::Legacy => None,
            Self::Recorded(r) => Some(&r.lineage),
        }
    }
}

impl RouterProvenance {
    /// The format revision an artifact with this provenance is written as. The
    /// revision is a property of what the artifact carries, not of the build
    /// that writes it, so a legacy artifact round-trips unchanged through a
    /// build that can write revision 5.
    pub fn format(&self) -> u32 {
        match self {
            Self::Legacy => ROUTER_ARTIFACT_FORMAT_LEGACY,
            Self::Recorded(_) => ROUTER_ARTIFACT_FORMAT,
        }
    }

    /// The recorded policy, or `None` for a legacy artifact. Deliberately not
    /// `unwrap_or(uniform)`: a caller that needs a policy must handle the
    /// artifact that does not name one.
    pub fn policy(&self) -> Option<&RouterPolicy> {
        match self {
            Self::Legacy => None,
            Self::Recorded(r) => Some(&r.policy),
        }
    }

    /// How this artifact was produced, as an operator reads it.
    ///
    /// A legacy artifact answers `"unrecorded"` rather than `"bootstrap"`.
    /// The two are different claims: a bootstrap was deliberately minted
    /// untrained and says so, while a legacy artifact says nothing at all and
    /// may well have been trained. Collapsing them would report a trained gate
    /// as a baseline, which is the reassuring direction.
    pub fn training_state(&self) -> &str {
        match self.policy() {
            None => "unrecorded",
            Some(p) => p.training_state.as_str(),
        }
    }

    /// Whether this artifact may be the parent of a learned refit. A legacy
    /// artifact may not: nothing recorded what it was serving under.
    pub fn is_learner_eligible(&self) -> bool {
        matches!(self.policy(), Some(p) if p.training_state == "learned"
            || p.training_state == "bootstrap")
    }
}

impl RouterPolicy {
    /// Refuse a policy whose fields do not describe a servable behaviour.
    ///
    /// Run at write AND at read. An artifact outlives the process that wrote
    /// it, is pinned by operators during incidents, and may be hand-edited;
    /// validating only on the way in leaves the read path trusting a file.
    pub fn validate(&self) -> Result<(), RouterArtifactError> {
        let bad = |problem: String| RouterArtifactError::InvalidPolicy { problem };
        if !ROUTER_POLICY_KINDS.contains(&self.kind.as_str()) {
            return Err(bad(format!(
                "kind {:?} is not one of {:?}",
                self.kind, ROUTER_POLICY_KINDS
            )));
        }
        if !ROUTER_TRAINING_STATES.contains(&self.training_state.as_str()) {
            return Err(bad(format!(
                "training_state {:?} is not one of {:?}",
                self.training_state, ROUTER_TRAINING_STATES
            )));
        }
        match (self.kind.as_str(), self.tau) {
            ("softmax", None) => {
                return Err(bad("a softmax policy records no tau".to_string()));
            }
            ("uniform", Some(tau)) => {
                return Err(bad(format!(
                    "a uniform policy records tau {tau}, which nothing reads"
                )));
            }
            _ => {}
        }
        // Every float is hashed by its bit pattern, and NaN has many of those
        // while comparing equal to none of them. A non-finite value here would
        // make a version label that cannot be reproduced from its own fields.
        for (name, value) in [
            ("epsilon", Some(self.epsilon)),
            ("tau_floor", Some(self.tau_floor)),
            ("entropy_floor", Some(self.entropy_floor)),
            ("tau", self.tau),
        ] {
            if let Some(v) = value
                && !v.is_finite()
            {
                return Err(bad(format!("{name} is {v}, which is not finite")));
            }
        }
        if let Some(tau) = self.tau
            && tau < self.tau_floor
        {
            return Err(bad(format!(
                "tau {tau} is below its own floor {}",
                self.tau_floor
            )));
        }
        if self.epsilon < 0.0 {
            return Err(bad(format!("epsilon {} is negative", self.epsilon)));
        }
        Ok(())
    }
}

/// A router gate as it exists on disk: the payload, the adapter names its
/// columns correspond to, the counter half of its version, and what it records
/// about its own policy and lineage.
///
/// `Eq` is deliberately absent: the recorded policy carries floats, and a total
/// equality over those would have to decide what two NaNs mean. Nothing in this
/// crate needs an artifact as a map key, and [`RouterPolicy::validate`] refuses
/// a non-finite value at both ends, so the question does not arise in practice
/// — but deriving `Eq` would be claiming it cannot.
#[derive(Debug, Clone, PartialEq)]
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
    /// What this artifact records about its own policy and lineage, and which
    /// format revision it is therefore written as. Inside the content hash.
    pub provenance: RouterProvenance,
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
    /// Which loader produced the embedding, spelled as that loader names
    /// itself: `qwen35-f16-decoder`.
    ///
    /// Recorded because the identity beside it is a NAME. A name distinguishes
    /// checkpoints an operator has distinguished; it says nothing about how
    /// the bytes were read, and two loaders reading the same directory can
    /// produce different vectors of the same width from the same text. There
    /// is one loader today, so this field cannot disagree with the server yet
    /// -- which is the only time a field like this can be added without
    /// invalidating every artifact already written.
    pub loader_format: String,
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
    /// Present exactly on a revision-5 manifest. Skipped when absent so a
    /// legacy artifact's manifest is byte-shaped as it always was, rather than
    /// gaining two `null`s that an older reader would have to ignore.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    policy: Option<RouterPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    lineage: Option<RouterLineage>,
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
            self.representation.loader_format.as_str(),
        ] {
            hasher.update((field.len() as u64).to_le_bytes());
            hasher.update(field.as_bytes());
        }
        hasher.update(self.representation.input_width.to_le_bytes());
        hasher.update((self.gate_bytes.len() as u64).to_le_bytes());
        hasher.update(&self.gate_bytes);

        // A legacy artifact hashes EXACTLY the bytes it hashed before this
        // revision existed, with nothing appended, not even a discriminant.
        // Every artifact already on disk carries a hash computed that way, and
        // a reader that appends anything for the legacy case would reject all
        // of them as corrupt. That property has its own test, because it is
        // the one this change could silently break.
        if let RouterProvenance::Recorded(r) = &self.provenance {
            let (policy, lineage) = (&r.policy, &r.lineage);
            // Domain separation so a recorded artifact can never collide with a
            // legacy one whose gate happens to end in these bytes.
            hasher.update(b"router-provenance-v1");
            for field in [
                policy.kind.as_str(),
                policy.training_state.as_str(),
                policy.objective_version.as_deref().unwrap_or(""),
                lineage.schema_id.as_str(),
                lineage.base_model.as_str(),
                lineage.parent.as_deref().unwrap_or(""),
                lineage.admission_record.as_deref().unwrap_or(""),
            ] {
                hasher.update((field.len() as u64).to_le_bytes());
                hasher.update(field.as_bytes());
            }
            // An absent optional string and an empty one are different claims:
            // "this lineage has no parent" is not "its parent is named by the
            // empty string". The length prefixes above cannot tell them apart,
            // so presence is hashed separately.
            for present in [
                policy.objective_version.is_some(),
                lineage.parent.is_some(),
                lineage.admission_record.is_some(),
                policy.tau.is_some(),
            ] {
                hasher.update([u8::from(present)]);
            }
            // Floats by bit pattern. `validate` has refused every non-finite
            // value, so each of these has exactly one representation.
            for f in [
                policy.tau.unwrap_or(0.0),
                policy.epsilon,
                policy.tau_floor,
                policy.entropy_floor,
            ] {
                hasher.update(f.to_bits().to_le_bytes());
            }
            hasher.update(policy.max_consecutive_collapsed.to_le_bytes());
        }
        hex_lower(&hasher.finalize())
    }

    /// The version as an operator reads it: counter and hash together, because
    /// neither half identifies a gate on its own.
    pub fn version_label(&self) -> String {
        let hash = self.content_hash();
        format!("{}:{}", self.version, &hash[..16])
    }
}

/// Names which of the two provenance fields a refusal is about, so the message
/// says `policy` rather than `a field`.
fn describe_provenance_fields(policy: bool, lineage: bool) -> &'static str {
    match (policy, lineage) {
        (true, true) => "a policy and a lineage",
        (true, false) => "a policy",
        (false, true) => "a lineage",
        (false, false) => "neither a policy nor a lineage",
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

/// Name of the committed head record inside a state directory.
///
/// It is not `router-<n>.json`, deliberately: `versions()` recognises only the
/// exact spelling it writes, so the head can never be mistaken for a version
/// by the scan, and the scan can never be widened into reading it by accident.
pub const ROUTER_HEAD_FILE: &str = "HEAD.json";

/// Cap on the head record, which holds three small fields.
pub const MAX_ROUTER_HEAD_LEN: u64 = 64 * 1024;

/// Path of the committed head record in `dir`.
pub fn head_path(dir: &Path) -> PathBuf {
    dir.join(ROUTER_HEAD_FILE)
}

/// What a state directory says is currently admitted.
///
/// The hash is here rather than only in the manifest because the head has to
/// name an artifact, not a filename. A counter can be written twice; the pair
/// cannot, and a disagreement between them says the admitted artifact was
/// replaced rather than that this one is fine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadRecord {
    /// Format revision of the artifact this head names.
    pub format: u32,
    /// The version counter it names.
    pub version: u64,
    /// The content hash that version had when it was admitted.
    pub content_sha256: String,
}

/// Read the committed head, if there is one.
///
/// `Ok(None)` means no head record exists, which is the format-4 state
/// directory ADR-095 described and is selected by counter. Every other outcome
/// is an `Err`, including a head that is present and unparseable: those two
/// answers pick different artifacts, so collapsing them would make an
/// interrupted admission look like a directory that had never been admitted
/// into.
pub fn read_head(dir: &Path) -> Result<Option<HeadRecord>, RouterArtifactError> {
    let path = head_path(dir);
    if !path.exists() {
        return Ok(None);
    }
    let raw = read_bytes_bounded(&path, MAX_ROUTER_HEAD_LEN).map_err(|e| {
        let display: BoundedReadErrorDisplay = e.into();
        RouterArtifactError::HeadUnreadable {
            path: path.clone(),
            message: display.to_string(),
        }
    })?;
    let head: HeadRecord =
        serde_json::from_slice(&raw).map_err(|e| RouterArtifactError::HeadUnreadable {
            path: path.clone(),
            message: e.to_string(),
        })?;
    Ok(Some(head))
}

/// Commit `artifact` as the directory's head.
///
/// Written to a temporary file in the same directory and renamed into place,
/// so a reader sees either the previous head or this one and never a partial
/// record. A same-directory rename is what makes that atomic; a temp file
/// elsewhere would be a copy across filesystems with no such guarantee.
///
/// The artifact must already be on disk. Committing a head for a version that
/// was never written produces a directory that refuses every unpinned startup,
/// which is worse than the rejection it was trying to record.
pub fn commit_head(dir: &Path, artifact: &RouterArtifact) -> Result<PathBuf, RouterArtifactError> {
    let manifest_file = manifest_path(dir, artifact.version);
    if !manifest_file.exists() {
        return Err(RouterArtifactError::HeadNamesAbsentVersion {
            path: head_path(dir),
            version: artifact.version,
        });
    }
    let head = HeadRecord {
        format: artifact.provenance.format(),
        version: artifact.version,
        content_sha256: artifact.content_hash(),
    };
    let json = serde_json::to_vec_pretty(&head).map_err(|e| RouterArtifactError::Write {
        path: head_path(dir),
        message: e.to_string(),
    })?;

    let path = head_path(dir);
    let staged = dir.join(format!("{ROUTER_HEAD_FILE}.staged-{}", artifact.version));
    std::fs::write(&staged, &json).map_err(|e| RouterArtifactError::Write {
        path: staged.clone(),
        message: e.to_string(),
    })?;
    std::fs::rename(&staged, &path).map_err(|e| {
        let _ = std::fs::remove_file(&staged);
        RouterArtifactError::Write {
            path: path.clone(),
            message: e.to_string(),
        }
    })?;
    Ok(path)
}

/// A `--router-pin` value: a counter, optionally with the hash prefix that
/// identifies which artifact that counter is supposed to be.
///
/// The bare counter is ADR-095's pin and stays exactly as written, which is
/// what a format-4 directory can answer. The pair is the new revision's full
/// version identity, and it is the form that survives a counter being reused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterPin {
    /// The version counter.
    pub version: u64,
    /// The hash prefix, when the operator named one.
    pub hash_prefix: Option<String>,
}

impl RouterPin {
    /// Parse `<version>` or `<version>:<hash-prefix>`, the same two spellings
    /// `version_label()` produces and an operator reads off a running server.
    pub fn parse(text: &str) -> Result<Self, RouterArtifactError> {
        let malformed = |problem: &str| RouterArtifactError::PinMalformed {
            text: text.to_string(),
            problem: problem.to_string(),
        };
        let (counter, hash_prefix) = match text.split_once(':') {
            None => (text, None),
            Some((counter, hash)) => {
                if hash.is_empty() {
                    return Err(malformed(
                        "a trailing colon with no hash names no identity; drop \
                         the colon to pin by counter",
                    ));
                }
                if !hash
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
                {
                    return Err(malformed(
                        "the hash half must be lowercase hex, as a version \
                         label prints it",
                    ));
                }
                (counter, Some(hash.to_string()))
            }
        };
        let version = counter
            .parse::<u64>()
            .map_err(|_| malformed("the counter half is not a version number"))?;
        Ok(Self {
            version,
            hash_prefix,
        })
    }

    /// The pin as the operator wrote it, for a refusal message.
    pub fn label(&self) -> String {
        match &self.hash_prefix {
            None => format!("{}", self.version),
            Some(hash) => format!("{}:{hash}", self.version),
        }
    }
}

impl std::str::FromStr for RouterPin {
    type Err = RouterArtifactError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text)
    }
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
    // Before anything is created. A refused policy must leave no payload
    // behind: `versions()` counts a version present when its MANIFEST exists,
    // so an orphan payload is invisible to every reader and would be silently
    // overwritten by the next write of the same counter.
    if let Some(policy) = artifact.provenance.policy() {
        policy.validate()?;
    }

    std::fs::create_dir_all(dir).map_err(|e| RouterArtifactError::Write {
        path: dir.to_path_buf(),
        message: e.to_string(),
    })?;

    let gate = gate_path(dir, artifact.version);
    std::fs::write(&gate, &artifact.gate_bytes).map_err(|e| RouterArtifactError::Write {
        path: gate.clone(),
        message: e.to_string(),
    })?;

    let (policy, lineage) = match &artifact.provenance {
        RouterProvenance::Legacy => (None, None),
        RouterProvenance::Recorded(r) => (Some(r.policy.clone()), Some(r.lineage.clone())),
    };
    let manifest = Manifest {
        format: artifact.provenance.format(),
        version: artifact.version,
        adapter_names: artifact.adapter_names.clone(),
        representation: artifact.representation.clone(),
        gate_len: artifact.gate_bytes.len() as u64,
        content_sha256: artifact.content_hash(),
        policy,
        lineage,
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
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedRouter {
    /// The gate artifact that was loaded.
    pub artifact: RouterArtifact,
    /// True when `--router-pin` selected this version, false when startup
    /// selection picked it: the committed head when the directory has one,
    /// and the highest counter present otherwise.
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
    /// The identity of the embedding model THIS server loaded.
    ///
    /// The server's own value, not the artifact's `trained_on`. Startup
    /// refuses when the two disagree, so a running server reports the same
    /// string either way -- which is exactly the reasoning that makes a
    /// co-located value look like the value it sits beside. If that check is
    /// ever relaxed or scoped, the report keeps answering the question an
    /// operator is actually asking, which is what this process embeds with.
    pub embedder: &'a str,
}

impl ResolvedRouter {
    /// Borrow this resolution as a report.
    ///
    /// The embedder identity is a parameter because a resolution does not
    /// have one: it is read from disk before this server's embedding model is
    /// known. Passing it in keeps the caller unable to report a gate without
    /// saying what it is being served against.
    pub fn report<'a>(&'a self, embedder: &'a str) -> RouterReport<'a> {
        RouterReport {
            artifact: &self.artifact,
            pinned: self.pinned,
            embedder,
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
    pin: Option<&RouterPin>,
) -> Result<StartupDisposition, RouterArtifactError> {
    let Some(dir) = dir else {
        return match pin {
            None => Ok(StartupDisposition::NoRouter),
            Some(pin) => Err(RouterArtifactError::PinWithoutState {
                version: pin.version,
            }),
        };
    };
    let resolved = match pin {
        // A pinned version that is absent refuses. Falling back to the latest
        // is the failure this rejects: the pin is reached for precisely when
        // the latest is the thing misbehaving, so a silent fallback serves the
        // artifact the operator was trying to get away from.
        //
        // The head is not consulted here, and that is the whole point of a
        // pin under the new revision: a head advancing while a pin is in force
        // changes the served gate for nobody until the pin is lifted.
        Some(pin) => {
            let artifact = read_artifact(dir, pin.version)?;
            if let Some(prefix) = &pin.hash_prefix {
                let hash = artifact.content_hash();
                if !hash.starts_with(prefix.as_str()) {
                    return Err(RouterArtifactError::PinIdentityMismatch {
                        version: pin.version,
                        requested: pin.label(),
                        found: artifact.version_label(),
                    });
                }
            }
            ResolvedRouter {
                artifact,
                pinned: true,
            }
        }
        None => match load_selected(dir)? {
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

/// What an unpinned startup serves from `dir`, under whichever of the two
/// selection rules the directory itself answers to (ADR-096 decision 5).
///
/// A head record present means the directory has been admitted into: it names
/// the artifact that passed, and a higher-numbered file beside it is a staged
/// or rejected candidate that must never be promoted by surviving a restart.
/// The head governs even when the artifact it names is older than a format-4
/// file sitting next to it, because a head is a statement that a specific
/// artifact was admitted and a counter is not. That format-4 version stays
/// reachable by pin, which is what keeps a rollback target available across
/// the revision boundary.
///
/// No head record means a format-4 directory, and ADR-095's rule applies
/// unchanged: the highest counter present wins. Nothing writes a head into
/// such a directory, so a reader looking for one would find an absence it
/// could not tell from a truncated write -- which is why `read_head` refuses
/// an unreadable head instead of reporting it absent.
pub fn load_selected(dir: &Path) -> Result<Option<RouterArtifact>, RouterArtifactError> {
    let Some(head) = read_head(dir)? else {
        return load_latest(dir);
    };
    if !manifest_path(dir, head.version).exists() {
        return Err(RouterArtifactError::HeadNamesAbsentVersion {
            path: head_path(dir),
            version: head.version,
        });
    }
    let artifact = read_artifact(dir, head.version)?;
    // The counter can be written twice; the pair cannot. A disagreement here
    // says the admitted artifact was replaced under a live head, which reads
    // on disk exactly like an ordinary refit and is not one.
    let hash = artifact.content_hash();
    if hash != head.content_sha256 {
        return Err(RouterArtifactError::HeadDisagreesWithArtifact {
            path: head_path(dir),
            version: head.version,
            head_hash: head.content_sha256,
            artifact_hash: hash,
        });
    }
    Ok(Some(artifact))
}

/// Load the highest version present in `dir`, by counter.
///
/// This is ADR-095's rule and stays exactly as written. It is the rule a
/// format-4 directory answers to; unpinned startup reaches it through
/// [`load_selected`], which applies it only when there is no committed head.
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

    let provenance = match (manifest.format, manifest.policy, manifest.lineage) {
        (ROUTER_ARTIFACT_FORMAT_LEGACY, None, None) => RouterProvenance::Legacy,
        (ROUTER_ARTIFACT_FORMAT, Some(policy), Some(lineage)) => {
            policy.validate()?;
            RouterProvenance::recorded(policy, lineage)
        }
        // The two halves disagree. Neither is repaired by inference: filling a
        // missing policy in would be exactly the defaulting ADR-096 decision 5
        // forbids, and dropping an unexpected one would serve a file while
        // ignoring what it says about itself.
        (ROUTER_ARTIFACT_FORMAT_LEGACY, policy, lineage) => {
            return Err(RouterArtifactError::ProvenanceMismatch {
                path: manifest_file,
                found: manifest.format,
                problem: format!(
                    "carries {}",
                    describe_provenance_fields(policy.is_some(), lineage.is_some())
                ),
            });
        }
        (ROUTER_ARTIFACT_FORMAT, policy, lineage) => {
            return Err(RouterArtifactError::ProvenanceMismatch {
                path: manifest_file,
                found: manifest.format,
                problem: format!(
                    "is missing {}",
                    describe_provenance_fields(policy.is_none(), lineage.is_none())
                ),
            });
        }
        (found, _, _) => {
            return Err(RouterArtifactError::UnknownFormat {
                path: manifest_file,
                found,
                known: format!("{ROUTER_ARTIFACT_FORMAT_LEGACY} and {ROUTER_ARTIFACT_FORMAT}"),
            });
        }
    };

    // The counter reaches this function twice by two different routes: as the
    // `version` argument, which `versions()` derived from the filename, and as
    // `manifest.version`, which the file declares. Neither is inside the
    // content hash, so the hash check below cannot see a disagreement between
    // them.
    if manifest.version != version {
        return Err(RouterArtifactError::VersionMismatch {
            path: manifest_file,
            declared: manifest.version,
            from_filename: version,
        });
    }

    let gate_file = gate_path(dir, version);
    let gate_bytes = read_bytes_bounded(&gate_file, MAX_ROUTER_GATE_LEN).map_err(|e| {
        RouterArtifactError::Read {
            path: gate_file,
            source: e.into(),
        }
    })?;

    // `gate_len` is written into every manifest and was, until this check,
    // read by nobody: the hash covers the payload's actual length, never the
    // declared one, so the two could disagree and both reads would pass.
    if manifest.gate_len != gate_bytes.len() as u64 {
        return Err(RouterArtifactError::GateLengthMismatch {
            path: manifest_file,
            payload: gate_path(dir, version),
            declared: manifest.gate_len,
            actual: gate_bytes.len() as u64,
        });
    }

    let artifact = RouterArtifact {
        version: manifest.version,
        adapter_names: manifest.adapter_names,
        representation: manifest.representation,
        gate_bytes,
        provenance,
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
            loader_format: "qwen35-f16-decoder".into(),
            input_width: 8,
        }
    }

    fn artifact() -> RouterArtifact {
        RouterArtifact {
            version: 7,
            adapter_names: vec!["legal".into(), "medical".into(), "code".into()],
            representation: representation(),
            gate_bytes: vec![1, 2, 3, 4, 5],
            provenance: RouterProvenance::Legacy,
        }
    }

    fn policy() -> RouterPolicy {
        RouterPolicy {
            kind: "softmax".into(),
            tau: Some(0.7),
            epsilon: 0.05,
            tau_floor: 0.1,
            entropy_floor: 0.35,
            max_consecutive_collapsed: 3,
            training_state: "learned".into(),
            objective_version: Some("whole-mixture-v1".into()),
        }
    }

    fn lineage() -> RouterLineage {
        RouterLineage {
            schema_id: "schema-abc123".into(),
            base_model: "qwen3.5-0.8b".into(),
            parent: Some("6:0123456789abcdef".into()),
            admission_record: Some("admission-42".into()),
        }
    }

    fn recorded() -> RouterArtifact {
        RouterArtifact {
            provenance: RouterProvenance::recorded(policy(), lineage()),
            ..artifact()
        }
    }

    /// The hash of a legacy artifact is the hash it had before recorded
    /// provenance existed, byte for byte.
    ///
    /// The constant is not a snapshot of what this code currently produces.
    /// It was derived independently, by reimplementing the PRE-CHANGE hash
    /// layout from the old source text in a different language and running it
    /// over this fixture. If it were a snapshot, this test would pass no matter
    /// what the legacy path did, and the property it exists to hold -- that
    /// every artifact already on disk still verifies -- would be untested.
    #[test]
    fn a_legacy_artifact_hashes_exactly_as_it_did_before_provenance_existed() {
        assert_eq!(
            artifact().content_hash(),
            "efbcc19e4dec67850a74df365728be02349fa5fef60c1fb2793c6da587cb6a53"
        );
        assert_eq!(
            artifact().provenance.format(),
            ROUTER_ARTIFACT_FORMAT_LEGACY
        );
    }

    #[test]
    fn a_recorded_artifact_round_trips_and_is_written_as_the_new_revision() {
        let dir = scratch("recorded");
        let original = recorded();
        write_artifact(&dir, &original).expect("write");

        let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
        assert!(
            raw.contains(&format!("\"format\": {ROUTER_ARTIFACT_FORMAT}")),
            "a recorded artifact must declare the new revision: {raw}"
        );

        assert_eq!(read_artifact(&dir, 7).expect("read"), original);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_legacy_manifest_carries_no_policy_keys_at_all() {
        // Not merely "no policy is read". An older build parses this file, and
        // two `null`s it has never seen are a different document from the one
        // it was written against.
        let dir = scratch("legacyshape");
        write_artifact(&dir, &artifact()).expect("write");
        let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
        assert!(
            !raw.contains("policy"),
            "legacy manifest mentions policy: {raw}"
        );
        assert!(
            !raw.contains("lineage"),
            "legacy manifest mentions lineage: {raw}"
        );
        assert!(raw.contains(&format!("\"format\": {ROUTER_ARTIFACT_FORMAT_LEGACY}")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_declared_revision_and_the_recorded_policy_must_agree_in_both_directions() {
        // Direction 1: the new revision without a policy. Defaulting one in is
        // exactly how a legacy gate would quietly become learner-eligible.
        let dir = scratch("missingpolicy");
        write_artifact(&dir, &recorded()).expect("write");
        let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read");
        let stripped: serde_json::Value = {
            let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse");
            v.as_object_mut().expect("object").remove("policy");
            v
        };
        std::fs::write(
            manifest_path(&dir, 7),
            serde_json::to_vec_pretty(&stripped).expect("serialize"),
        )
        .expect("rewrite");
        match read_artifact(&dir, 7) {
            Err(RouterArtifactError::ProvenanceMismatch { found, problem, .. }) => {
                assert_eq!(found, ROUTER_ARTIFACT_FORMAT);
                assert!(problem.contains("policy"), "{problem}");
            }
            other => panic!("expected a provenance mismatch, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);

        // Direction 2: the legacy revision carrying a policy. Dropping it would
        // serve the file while ignoring what it says about itself.
        let dir = scratch("extrapolicy");
        write_artifact(&dir, &artifact()).expect("write");
        let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read");
        let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        v.as_object_mut().expect("object").insert(
            "policy".into(),
            serde_json::to_value(policy()).expect("policy json"),
        );
        std::fs::write(
            manifest_path(&dir, 7),
            serde_json::to_vec_pretty(&v).expect("serialize"),
        )
        .expect("rewrite");
        match read_artifact(&dir, 7) {
            Err(RouterArtifactError::ProvenanceMismatch { found, problem, .. }) => {
                assert_eq!(found, ROUTER_ARTIFACT_FORMAT_LEGACY);
                assert!(problem.contains("policy"), "{problem}");
            }
            other => panic!("expected a provenance mismatch, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `validate()` runs on the READ path too, not only at write. An artifact
    /// outlives the process that wrote it and may be hand-edited, so a policy
    /// that was never servable must be refused when it is read back rather
    /// than trusted because the writer that produced it is gone.
    #[test]
    fn an_invalid_policy_is_refused_on_read_and_not_only_at_write() {
        let dir = scratch("readbadpolicy");
        write_artifact(&dir, &recorded()).expect("write");
        assert_eq!(
            read_artifact(&dir, 7).expect("control: the fixture reads"),
            recorded()
        );

        let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
        let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        v["policy"]["kind"] = serde_json::Value::String("argmax".into());
        std::fs::write(
            manifest_path(&dir, 7),
            serde_json::to_vec_pretty(&v).expect("serialize"),
        )
        .expect("rewrite");

        match read_artifact(&dir, 7) {
            Err(RouterArtifactError::InvalidPolicy { problem }) => {
                assert!(problem.contains("argmax"), "{problem}");
            }
            other => panic!("expected an invalid-policy refusal on read, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every recorded field is inside the hash, checked one field at a time.
    ///
    /// A single "the recorded artifact hashes differently from the legacy one"
    /// assertion would pass with all but one field left out, which is the
    /// defect this revision exists to avoid repeating: the last format put two
    /// declared fields outside the hash and nothing noticed for a day.
    #[test]
    fn every_recorded_field_changes_the_hash() {
        let base = recorded();
        let base_hash = base.content_hash();
        assert_ne!(base_hash, artifact().content_hash());

        type Mutation = Box<dyn Fn(&mut RouterPolicy, &mut RouterLineage)>;
        let mutate: Vec<(&str, Mutation)> = vec![
            (
                "kind",
                Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| {
                    p.kind = "uniform".into();
                    p.tau = None;
                }),
            ),
            (
                "tau",
                Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.tau = Some(0.9)),
            ),
            (
                "epsilon",
                Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.epsilon = 0.06),
            ),
            (
                "tau_floor",
                Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.tau_floor = 0.11),
            ),
            (
                "entropy_floor",
                Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.entropy_floor = 0.36),
            ),
            (
                "max_consecutive_collapsed",
                Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| {
                    p.max_consecutive_collapsed = 4
                }),
            ),
            (
                "training_state",
                Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| {
                    p.training_state = "bootstrap".into()
                }),
            ),
            (
                "objective_version",
                Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| {
                    p.objective_version = Some("whole-mixture-v2".into())
                }),
            ),
            (
                "objective_version absent",
                Box::new(|p: &mut RouterPolicy, _: &mut RouterLineage| p.objective_version = None),
            ),
            (
                "schema_id",
                Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| {
                    l.schema_id = "schema-def456".into()
                }),
            ),
            (
                "base_model",
                Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| {
                    l.base_model = "qwen3.5-4b".into()
                }),
            ),
            (
                "parent",
                Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| {
                    l.parent = Some("5:fedcba9876543210".into())
                }),
            ),
            (
                "parent absent",
                Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| l.parent = None),
            ),
            (
                "admission_record",
                Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| {
                    l.admission_record = Some("admission-43".into())
                }),
            ),
            (
                "admission_record absent",
                Box::new(|_: &mut RouterPolicy, l: &mut RouterLineage| l.admission_record = None),
            ),
        ];

        for (name, apply) in mutate {
            let mut p = policy();
            let mut l = lineage();
            apply(&mut p, &mut l);
            let changed = RouterArtifact {
                provenance: RouterProvenance::recorded(p, l),
                ..artifact()
            };
            assert_ne!(
                changed.content_hash(),
                base_hash,
                "changing {name} left the content hash unchanged, so that field is outside it"
            );
        }
    }

    /// A bare counter pin, which is ADR-095's spelling and the one these
    /// tests were written against.
    fn pin(version: u64) -> RouterPin {
        RouterPin::parse(&format!("{version}")).expect("a bare counter parses")
    }

    /// An absent optional string and an empty one are different claims, and the
    /// length prefixes alone cannot tell them apart.
    #[test]
    fn an_absent_optional_does_not_hash_as_an_empty_one() {
        let absent = RouterArtifact {
            provenance: RouterProvenance::recorded(
                policy(),
                RouterLineage {
                    parent: None,
                    ..lineage()
                },
            ),
            ..artifact()
        };
        let empty = RouterArtifact {
            provenance: RouterProvenance::recorded(
                policy(),
                RouterLineage {
                    parent: Some(String::new()),
                    ..lineage()
                },
            ),
            ..artifact()
        };
        assert_ne!(absent.content_hash(), empty.content_hash());
    }

    #[test]
    fn an_inconsistent_policy_is_refused_at_write_and_leaves_no_payload() {
        // The ordering matters as much as the refusal: `versions()` counts a
        // version present when its MANIFEST exists, so a payload written before
        // a refusal is invisible to every reader and silently overwritten by the
        // next write of the same counter.
        let dir = scratch("badpolicy");
        let bad = RouterArtifact {
            provenance: RouterProvenance::recorded(
                RouterPolicy {
                    tau: None,
                    ..policy()
                },
                lineage(),
            ),
            ..artifact()
        };
        match write_artifact(&dir, &bad) {
            Err(RouterArtifactError::InvalidPolicy { problem }) => {
                assert!(problem.contains("tau"), "{problem}");
            }
            other => panic!("expected an invalid-policy refusal, got {other:?}"),
        }
        assert!(
            !gate_path(&dir, 7).exists(),
            "a refused write left a gate payload behind"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_validation_covers_each_way_a_policy_can_be_unservable() {
        let cases: Vec<(&str, RouterPolicy)> = vec![
            (
                "kind",
                RouterPolicy {
                    kind: "argmax".into(),
                    ..policy()
                },
            ),
            (
                "training_state",
                RouterPolicy {
                    training_state: "trained".into(),
                    ..policy()
                },
            ),
            (
                "softmax with no tau",
                RouterPolicy {
                    tau: None,
                    ..policy()
                },
            ),
            (
                "uniform with a tau",
                RouterPolicy {
                    kind: "uniform".into(),
                    tau: Some(0.7),
                    ..policy()
                },
            ),
            (
                "non-finite tau",
                RouterPolicy {
                    tau: Some(f64::NAN),
                    ..policy()
                },
            ),
            (
                "non-finite epsilon",
                RouterPolicy {
                    epsilon: f64::INFINITY,
                    ..policy()
                },
            ),
            (
                "tau below its floor",
                RouterPolicy {
                    tau: Some(0.05),
                    ..policy()
                },
            ),
            (
                "negative epsilon",
                RouterPolicy {
                    epsilon: -0.01,
                    ..policy()
                },
            ),
        ];
        for (name, p) in cases {
            assert!(p.validate().is_err(), "{name} was accepted by validate()");
        }
        // The must-match control. Without it every arm above would pass against
        // a `validate` that rejected everything, including the policy this
        // module writes.
        policy()
            .validate()
            .expect("the fixture policy must validate");
    }

    /// A version of `recorded()` under a chosen counter, so a test can put two
    /// admitted artifacts in one directory.
    fn recorded_at(version: u64) -> RouterArtifact {
        RouterArtifact {
            version,
            ..recorded()
        }
    }

    /// With no head record the directory is a format-4 one and ADR-095's rule
    /// applies unchanged. The control is the same call on the same directory
    /// after a head is committed: if `load_selected` ignored the head, both
    /// halves would answer 9 and the test would pass while proving nothing.
    #[test]
    fn with_no_head_the_highest_counter_is_served_and_a_head_changes_that() {
        let dir = scratch("head-vs-counter");
        write_artifact(&dir, &recorded_at(3)).expect("write 3");
        write_artifact(&dir, &recorded_at(9)).expect("write 9");

        let by_counter = load_selected(&dir).expect("select").expect("an artifact");
        assert_eq!(by_counter.version, 9, "no head means the counter rule");

        commit_head(&dir, &recorded_at(3)).expect("commit head at 3");
        let by_head = load_selected(&dir).expect("select").expect("an artifact");
        assert_eq!(
            by_head.version, 3,
            "a head governs unpinned selection even against a higher counter"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A staged or rejected candidate is exactly this shape on disk: a higher
    /// counter present, not named by the head. It must never be promoted by
    /// surviving a restart, which is what the write-then-scan rule would do.
    #[test]
    fn a_higher_counter_not_named_by_the_head_is_never_promoted() {
        let dir = scratch("rejected-candidate");
        write_artifact(&dir, &recorded_at(1)).expect("write 1");
        commit_head(&dir, &recorded_at(1)).expect("commit head at 1");
        write_artifact(&dir, &recorded_at(2)).expect("write the candidate");

        for round in 0..2 {
            let served = load_selected(&dir).expect("select").expect("an artifact");
            assert_eq!(
                served.version, 1,
                "round {round}: the rejected candidate was promoted by a restart"
            );
        }

        // The control: the same directory once 2 is actually admitted.
        commit_head(&dir, &recorded_at(2)).expect("admit 2");
        assert_eq!(
            load_selected(&dir)
                .expect("select")
                .expect("an artifact")
                .version,
            2
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A head naming a version that is not there refuses. Falling back to the
    /// next-highest is the promotion the head exists to prevent, and it would
    /// happen at the moment someone deleted the admitted artifact by hand.
    #[test]
    fn a_head_naming_an_absent_version_refuses_rather_than_falling_back() {
        let dir = scratch("head-absent");
        write_artifact(&dir, &recorded_at(1)).expect("write 1");
        write_artifact(&dir, &recorded_at(2)).expect("write 2");
        commit_head(&dir, &recorded_at(2)).expect("commit head at 2");

        // The control, before the removal: this directory loads.
        assert_eq!(
            load_selected(&dir)
                .expect("select")
                .expect("an artifact")
                .version,
            2
        );

        std::fs::remove_file(manifest_path(&dir, 2)).expect("remove the admitted manifest");
        match load_selected(&dir) {
            Err(RouterArtifactError::HeadNamesAbsentVersion { version, .. }) => {
                assert_eq!(version, 2);
            }
            other => panic!("expected a refusal naming version 2, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The counter can be written twice and the pair cannot. A later write of
    /// the same number reads on disk exactly like an ordinary refit, and it is
    /// not one: the head is now pointing at an artifact that was never
    /// admitted.
    #[test]
    fn a_head_whose_hash_no_longer_matches_its_version_refuses() {
        let dir = scratch("head-hash");
        write_artifact(&dir, &recorded_at(4)).expect("write 4");
        commit_head(&dir, &recorded_at(4)).expect("commit head at 4");
        assert_eq!(
            load_selected(&dir)
                .expect("select")
                .expect("an artifact")
                .version,
            4,
            "control: the committed pair loads"
        );

        let replaced = RouterArtifact {
            gate_bytes: vec![9, 9, 9],
            ..recorded_at(4)
        };
        write_artifact(&dir, &replaced).expect("rewrite 4 with other content");
        match load_selected(&dir) {
            Err(RouterArtifactError::HeadDisagreesWithArtifact {
                version,
                head_hash,
                artifact_hash,
                ..
            }) => {
                assert_eq!(version, 4);
                assert_ne!(head_hash, artifact_hash);
                assert_eq!(artifact_hash, replaced.content_hash());
            }
            other => panic!("expected a hash disagreement, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An unreadable head is an error, never an absence. Reporting it absent
    /// would silently reinstate the counter rule on the one directory whose
    /// admission was interrupted, which is the directory where the counter
    /// rule is most likely to serve a rejected candidate.
    #[test]
    fn a_present_but_unparseable_head_is_an_error_not_an_absence() {
        let dir = scratch("head-garbage");
        write_artifact(&dir, &recorded_at(1)).expect("write 1");
        commit_head(&dir, &recorded_at(1)).expect("commit head at 1");
        write_artifact(&dir, &recorded_at(2)).expect("write 2");

        std::fs::write(head_path(&dir), b"{\"format\": 5, \"vers").expect("truncate the head");
        match read_head(&dir) {
            Err(RouterArtifactError::HeadUnreadable { path, .. }) => {
                assert_eq!(path, head_path(&dir));
            }
            other => panic!("expected an unreadable head, got {other:?}"),
        }
        assert!(
            matches!(
                load_selected(&dir),
                Err(RouterArtifactError::HeadUnreadable { .. })
            ),
            "a truncated head must not fall through to the counter rule and serve 2"
        );

        // The control: an ABSENT head on the same directory is not an error,
        // and it is what selects 2. The two states are distinguishable.
        std::fs::remove_file(head_path(&dir)).expect("remove the head");
        assert_eq!(read_head(&dir).expect("absent head"), None);
        assert_eq!(
            load_selected(&dir)
                .expect("select")
                .expect("an artifact")
                .version,
            2
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Committing a head for a version that was never written produces a
    /// directory that refuses every unpinned startup. Refusing at the commit
    /// is the only place that state can still be avoided.
    #[test]
    fn a_head_cannot_be_committed_for_a_version_that_is_not_on_disk() {
        let dir = scratch("head-commit-absent");
        write_artifact(&dir, &recorded_at(1)).expect("write 1");

        match commit_head(&dir, &recorded_at(2)) {
            Err(RouterArtifactError::HeadNamesAbsentVersion { version, .. }) => {
                assert_eq!(version, 2);
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(
            !head_path(&dir).exists(),
            "a refused commit must leave no head behind"
        );

        // The control: the version that IS on disk commits.
        commit_head(&dir, &recorded_at(1)).expect("commit head at 1");
        assert!(head_path(&dir).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The head is renamed into place, so a reader sees one record or the
    /// other and the staging file does not survive the commit.
    #[test]
    fn committing_a_head_replaces_the_previous_one_and_leaves_nothing_staged() {
        let dir = scratch("head-replace");
        write_artifact(&dir, &recorded_at(1)).expect("write 1");
        write_artifact(&dir, &recorded_at(2)).expect("write 2");
        commit_head(&dir, &recorded_at(1)).expect("commit 1");
        commit_head(&dir, &recorded_at(2)).expect("commit 2");

        let head = read_head(&dir).expect("read").expect("a head");
        assert_eq!(head.version, 2);
        assert_eq!(head.format, ROUTER_ARTIFACT_FORMAT);
        assert_eq!(head.content_sha256, recorded_at(2).content_hash());

        let staged: Vec<_> = std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".staged-"))
            .collect();
        assert!(staged.is_empty(), "staging files survived: {staged:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pin does not consult the head, so a head advancing while a pin is in
    /// force changes the served gate for nobody until the pin is lifted. The
    /// control is the unpinned call on the same directory, which does move.
    #[test]
    fn a_pin_disables_promotion_and_a_head_advance_does_not_reach_it() {
        let dir = scratch("pin-vs-head");
        write_artifact(&dir, &recorded_at(1)).expect("write 1");
        write_artifact(&dir, &recorded_at(2)).expect("write 2");
        commit_head(&dir, &recorded_at(1)).expect("commit 1");

        let pinned = |dir: &Path| match resolve_startup(Some(dir), Some(&pin(1))) {
            Ok(StartupDisposition::Loaded(r)) => {
                assert!(r.pinned);
                r.artifact.version
            }
            other => panic!("expected a pinned load, got {other:?}"),
        };
        assert_eq!(pinned(&dir), 1);

        commit_head(&dir, &recorded_at(2)).expect("advance the head to 2");
        assert_eq!(pinned(&dir), 1, "the head advance reached the pin");
        assert_eq!(
            load_selected(&dir)
                .expect("select")
                .expect("an artifact")
                .version,
            2,
            "control: the unpinned path did move, so the advance really happened"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pin naming a full identity is refused when that counter now holds a
    /// different artifact. The bare-counter pin in the same directory is the
    /// control: it still resolves, which is what makes the refusal a statement
    /// about the identity rather than about the version being unreadable.
    #[test]
    fn a_pin_naming_an_identity_refuses_when_the_counter_holds_another_artifact() {
        let dir = scratch("pin-identity");
        let original = recorded_at(5);
        write_artifact(&dir, &original).expect("write 5");
        let label = original.version_label();
        let requested = RouterPin::parse(&label).expect("a label parses as a pin");

        match resolve_startup(Some(&dir), Some(&requested)) {
            Ok(StartupDisposition::Loaded(r)) => assert_eq!(r.artifact.version_label(), label),
            other => panic!("control: the matching identity should load, got {other:?}"),
        }

        let replaced = RouterArtifact {
            gate_bytes: vec![7, 7],
            ..recorded_at(5)
        };
        write_artifact(&dir, &replaced).expect("rewrite 5");
        match resolve_startup(Some(&dir), Some(&requested)) {
            Err(RouterArtifactError::PinIdentityMismatch {
                version,
                requested: asked,
                found,
            }) => {
                assert_eq!(version, 5);
                assert_eq!(asked, label);
                assert_eq!(found, replaced.version_label());
            }
            other => panic!("expected an identity mismatch, got {other:?}"),
        }
        match resolve_startup(Some(&dir), Some(&pin(5))) {
            Ok(StartupDisposition::Loaded(r)) => assert_eq!(r.artifact.version, 5),
            other => panic!("control: a bare counter still resolves, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A format-4 artifact in a directory that has a head stays reachable by
    /// pin. That is what keeps a rollback target available across the revision
    /// boundary, and it is the one path by which a legacy artifact can still
    /// be served from such a directory.
    #[test]
    fn a_format_4_version_stays_reachable_by_pin_beneath_a_head() {
        let dir = scratch("legacy-under-head");
        let legacy = RouterArtifact {
            version: 8,
            ..artifact()
        };
        write_artifact(&dir, &legacy).expect("write the legacy artifact");
        write_artifact(&dir, &recorded_at(2)).expect("write the admitted one");
        commit_head(&dir, &recorded_at(2)).expect("commit the head at 2");

        assert_eq!(
            load_selected(&dir)
                .expect("select")
                .expect("an artifact")
                .version,
            2,
            "the head governs unpinned selection against the higher legacy counter"
        );
        match resolve_startup(Some(&dir), Some(&pin(8))) {
            Ok(StartupDisposition::Loaded(r)) => {
                assert_eq!(r.artifact.version, 8);
                assert_eq!(r.artifact.provenance, RouterProvenance::Legacy);
                assert!(r.pinned);
            }
            other => panic!("expected the legacy artifact by pin, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Both spellings and each way a pin can be unusable, with a must-match
    /// control in the same test so a parser that rejected everything would not
    /// read as thorough.
    #[test]
    fn pin_parsing_covers_each_spelling_and_each_refusal() {
        let bare = RouterPin::parse("7").expect("a bare counter");
        assert_eq!(bare.version, 7);
        assert_eq!(bare.hash_prefix, None);
        assert_eq!(bare.label(), "7");

        let full = RouterPin::parse("7:8363b3f04c702da3").expect("a full identity");
        assert_eq!(full.version, 7);
        assert_eq!(full.hash_prefix.as_deref(), Some("8363b3f04c702da3"));
        assert_eq!(full.label(), "7:8363b3f04c702da3");

        for (text, why) in [
            ("", "an empty pin names nothing"),
            ("seven", "a word is not a counter"),
            ("7:", "a trailing colon names no identity"),
            ("7:ZZZZ", "the hash half must be hex"),
            (
                "7:8363B3F0",
                "an uppercase hash is not the spelling we print",
            ),
            ("-1", "a negative counter is not a version"),
        ] {
            assert!(
                matches!(
                    RouterPin::parse(text),
                    Err(RouterArtifactError::PinMalformed { .. })
                ),
                "{text:?} should be refused: {why}"
            );
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
            provenance: RouterProvenance::Legacy,
        };
        let b = RouterArtifact {
            version: 1,
            adapter_names: vec!["a".into(), "bc".into()],
            representation: representation(),
            gate_bytes: vec![9],
            provenance: RouterProvenance::Legacy,
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
            provenance: RouterProvenance::Legacy,
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
        // Bump from the revision the fixture actually wrote, which is the
        // legacy one because `artifact()` records no provenance. Writing
        // `ROUTER_ARTIFACT_FORMAT` here would match nothing once the two
        // revisions diverged, and the replacement would silently be a no-op.
        let written = artifact().provenance.format();
        let bumped = raw.replace(
            &format!("\"format\": {written}"),
            &format!("\"format\": {}", ROUTER_ARTIFACT_FORMAT + 1),
        );
        assert_ne!(bumped, raw, "the format field must have been rewritten");
        std::fs::write(manifest_path(&dir, 7), bumped).expect("rewrite");

        match read_artifact(&dir, 7) {
            Err(RouterArtifactError::UnknownFormat { found, known, .. }) => {
                assert_eq!(found, ROUTER_ARTIFACT_FORMAT + 1);
                assert_eq!(
                    known,
                    format!("{ROUTER_ARTIFACT_FORMAT_LEGACY} and {ROUTER_ARTIFACT_FORMAT}")
                );
            }
            other => panic!("expected an unknown-format refusal, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_manifest_whose_version_disagrees_with_its_filename_is_refused() {
        // The counter is not inside the content hash, so this edit leaves the
        // hash valid: without an explicit comparison the file loads clean and
        // reports a version the operator never asked for. The control below is
        // the same directory reading fine before the edit, so a refusal here
        // cannot come from a broken fixture.
        let dir = scratch("versionskew");
        write_artifact(&dir, &artifact()).expect("write");
        assert_eq!(read_artifact(&dir, 7).expect("control reads"), artifact());

        let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
        let skewed = raw.replace("\"version\": 7", "\"version\": 3");
        assert_ne!(skewed, raw, "the version field must have been rewritten");
        std::fs::write(manifest_path(&dir, 7), skewed).expect("rewrite");

        match read_artifact(&dir, 7) {
            Err(RouterArtifactError::VersionMismatch {
                declared,
                from_filename,
                ..
            }) => {
                assert_eq!(declared, 3);
                assert_eq!(from_filename, 7);
            }
            other => panic!("expected a version mismatch, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_declared_payload_length_that_does_not_match_the_payload_is_refused() {
        // `gate_len` is written by every writer and, until it was compared,
        // read by nobody. It is outside the content hash too, so a manifest
        // that belongs to a different payload passes the hash check whenever
        // the payload it describes is the one on disk.
        let dir = scratch("gatelen");
        write_artifact(&dir, &artifact()).expect("write");
        assert_eq!(read_artifact(&dir, 7).expect("control reads"), artifact());

        let raw = std::fs::read_to_string(manifest_path(&dir, 7)).expect("read manifest");
        let skewed = raw.replace("\"gate_len\": 5", "\"gate_len\": 4");
        assert_ne!(skewed, raw, "the gate_len field must have been rewritten");
        std::fs::write(manifest_path(&dir, 7), skewed).expect("rewrite");

        match read_artifact(&dir, 7) {
            Err(RouterArtifactError::GateLengthMismatch {
                declared, actual, ..
            }) => {
                assert_eq!(declared, 4);
                assert_eq!(actual, 5);
            }
            other => panic!("expected a gate-length mismatch, got {other:?}"),
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
                    provenance: RouterProvenance::Legacy,
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
        match resolve_startup(None, Some(&pin(7))) {
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
                provenance: RouterProvenance::Legacy,
            },
        )
        .expect("write v9");
        match resolve_startup(Some(&dir), Some(&pin(7))).expect("load") {
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
        match resolve_startup(Some(&dir), Some(&pin(99))) {
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
        match resolve_startup(Some(&dir), Some(&pin(7))) {
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
                provenance: RouterProvenance::Legacy,
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
