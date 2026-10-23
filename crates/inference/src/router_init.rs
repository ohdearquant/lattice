//! Provisioning a router state directory (ADR-096 decision 2).
//!
//! `lattice router init` mints the first artifact a directory holds. Its
//! output is a **deterministic untrained bootstrap**: a gate whose logits are
//! zero, carrying the named `Uniform` policy with nothing learned. That is
//! legitimate to serve, because it is the baseline an operator asked for, and
//! it is not evidence of learning. Serving it and serving a learned gate are
//! different claims, so the artifact records which one it is and
//! `GET /v1/lora` reports it.
//!
//! The width is not a flag. ADR-096 decision 2 requires it to be measured
//! through the same loader contract the serving path uses, so the caller
//! passes the dimension count its own embedding model produced rather than a
//! number someone typed: a mistyped width produces an artifact that refuses at
//! every startup, and the refusal names the gate rather than the typo.
//!
//! The three refusals here exist because each permissive form fails toward
//! looking successful. An occupied directory is not initialised over. A
//! duplicate or empty adapter name is refused rather than written, because
//! the serving path matches columns to adapters BY NAME and a duplicate makes
//! one column unreachable while every check still passes. And initialising
//! does not enable learning: that is a separate explicit action, and nothing
//! in this module performs it.

use std::path::PathBuf;

use crate::router_state::{
    DEFAULT_ENTROPY_FLOOR, DEFAULT_MAX_CONSECUTIVE_COLLAPSED, DEFAULT_TAU_FLOOR,
    MIN_EXPLORATION_SCALE, RouterArtifactError, RouterPolicy, TrainedRepresentation,
};

// The two functions below are the only part of this module that needs a gate
// network, and `lattice-fann` arrives with either of these features. The
// types and the specification validation are compiled unconditionally, so a
// build without a gate backend still refuses a bad specification rather than
// having no opinion about one.
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
use crate::router_state::{RouterArtifact, RouterLineage, RouterProvenance, publish_bootstrap};
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
use std::path::Path;

/// The version counter a fresh directory's first artifact carries.
pub const BOOTSTRAP_VERSION: u64 = 1;

/// What goes wrong provisioning a state directory.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RouterInitError {
    /// The directory already holds artifacts. Initialising over them would
    /// discard a served history, and a mistyped destination is exactly the
    /// case where the permissive form looks like it worked.
    #[error(
        "router state directory {dir} already holds version(s) {present:?}; \
         `router init` does not overwrite, because the directory it would \
         overwrite is the one a running server is pinned into"
    )]
    Occupied {
        /// The directory asked for.
        dir: PathBuf,
        /// What is already there.
        present: Vec<u64>,
    },

    /// The ordered adapter specification is not usable as a column order.
    #[error("adapter specification is not usable: {problem}")]
    InvalidSpec {
        /// What is wrong with it.
        problem: String,
    },

    /// The zero gate could not be built or serialized.
    #[error("building the bootstrap gate failed: {message}")]
    Gate {
        /// The builder's own message.
        message: String,
    },

    /// Writing the artifact or committing the head failed.
    #[error(transparent)]
    Artifact(#[from] RouterArtifactError),
}

/// The policy a bootstrap carries: nothing learned, and either shape.
///
/// With `tau` absent this is the original uniform bootstrap, field for field.
/// Every guard field is present and zero rather than absent, because a uniform
/// policy has no collapse history to bound and no temperature to floor, and
/// recording that as an absence would make the artifact's policy shape depend
/// on how it was produced -- which is the one thing a hashed contract must not
/// do, because then two policies that serve identically hash differently.
///
/// With `tau` present the kind is `softmax` and the three guard fields carry
/// the router's own defaults rather than zero. That difference is the whole
/// point: on a softmax policy those fields ARE read, and zero in them is
/// ADR-091 decision 3's guard recorded as disabled. The uniform bootstrap's
/// zeros are correct because nothing reads them; copying them here would be
/// the same bytes meaning the opposite thing.
///
/// `training_state` stays `bootstrap` either way. A softmax bootstrap over a
/// zero gate has learned exactly as much as a uniform one -- nothing -- and
/// serves exactly the same mixture, because a softmax over equal logits IS
/// `1/k`. What changes is that the mixture is now a function of the scores, so
/// a perturbation of the scores can reach it. That is what makes the artifact
/// explorable, and it is not evidence of learning.
pub fn bootstrap_policy(tau: Option<f64>, exploration_scale: Option<f64>) -> RouterPolicy {
    let learned_shape = tau.is_some();
    RouterPolicy {
        kind: if learned_shape { "softmax" } else { "uniform" }.to_string(),
        // Refused by `validate()` on a uniform policy, and that is the point:
        // nothing reads a tau there, so recording one would be a number an
        // operator could change with no effect.
        tau,
        epsilon: 0.0,
        tau_floor: if learned_shape {
            f64::from(DEFAULT_TAU_FLOOR)
        } else {
            0.0
        },
        entropy_floor: if learned_shape {
            f64::from(DEFAULT_ENTROPY_FLOOR)
        } else {
            0.0
        },
        max_consecutive_collapsed: if learned_shape {
            DEFAULT_MAX_CONSECUTIVE_COLLAPSED as u32
        } else {
            0
        },
        training_state: "bootstrap".to_string(),
        // No objective was optimised. An empty string here would read as a
        // recorded objective with an unprintable name.
        objective_version: None,
        exploration_scale,
    }
}

/// The ordered specification a bootstrap is minted from.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapSpec {
    /// The adapter names in gate column order. This IS the schema: ADR-096
    /// decision 1 makes it immutable within a lineage.
    pub adapter_names: Vec<String>,
    /// The representation, with `input_width` measured through the serving
    /// path's own loader rather than declared.
    pub representation: TrainedRepresentation,
    /// The schema identity this lineage is named by.
    pub schema_id: String,
    /// The base model the adapters attach to.
    pub base_model: String,
    /// The softmax temperature, when the bootstrap is to be minted in the shape
    /// a learning run can use. Absent mints the original uniform bootstrap.
    pub tau: Option<f64>,
    /// The exploration scale the artifact records, when one is to be recorded.
    /// Requires `tau`: see [`BootstrapSpec::validate`].
    pub exploration_scale: Option<f64>,
}

impl BootstrapSpec {
    /// Refuse a specification that cannot be a column order.
    ///
    /// A duplicate name is the dangerous one: the serving path matches columns
    /// to adapters by name, so a duplicate leaves one column permanently
    /// unreachable while residency, width and hash checks all pass.
    pub fn validate(&self) -> Result<(), RouterInitError> {
        let bad = |problem: String| RouterInitError::InvalidSpec { problem };
        if self.adapter_names.is_empty() {
            return Err(bad("no adapter names were given".to_string()));
        }
        if let Some(blank) = self.adapter_names.iter().find(|n| n.trim().is_empty()) {
            return Err(bad(format!(
                "adapter name {blank:?} is empty, and a column matched by an \
                 empty name can never be routed to"
            )));
        }
        let mut seen = std::collections::BTreeSet::new();
        for name in &self.adapter_names {
            if !seen.insert(name.as_str()) {
                return Err(bad(format!(
                    "adapter name {name:?} appears twice; columns are matched \
                     by name, so one of them could never be selected"
                )));
            }
        }
        if self.representation.input_width == 0 {
            return Err(bad(
                "the measured input width is 0, which is what an embedding \
                 model that did not load produces"
                    .to_string(),
            ));
        }
        if self.schema_id.trim().is_empty() {
            return Err(bad("the schema id is empty".to_string()));
        }
        if self.base_model.trim().is_empty() {
            return Err(bad("the base model is empty".to_string()));
        }
        // The three checks below duplicate refusals `RouterPolicy::validate`
        // already makes, deliberately. That one runs at write time and names
        // the policy; this one runs here and names the flags the operator
        // typed, which is this module's stated rule for every other field.
        // Without it, `bootstrap_artifact` -- which does not write -- returns
        // an artifact whose policy is invalid, and the callers that never write
        // it (tests, and any future dry run) get no refusal at all.
        if let Some(tau) = self.tau {
            if !tau.is_finite() || tau <= 0.0 {
                return Err(bad(format!(
                    "tau {tau} must be finite and greater than 0.0"
                )));
            }
            if tau < f64::from(DEFAULT_TAU_FLOOR) {
                return Err(bad(format!(
                    "tau {tau} is below the floor {DEFAULT_TAU_FLOOR} this bootstrap records, \
                     so the artifact would refuse at every startup"
                )));
            }
        }
        if let Some(scale) = self.exploration_scale {
            if self.tau.is_none() {
                return Err(bad(format!(
                    "--exploration-scale requires --tau: a scale {scale} was given without a \
                     tau, so the policy would be uniform, the serving path would select every \
                     trained column, and a draw could not change a decision that is 1/k \
                     whatever the scores are"
                )));
            }
            if !scale.is_finite() || scale < MIN_EXPLORATION_SCALE {
                return Err(bad(format!(
                    "exploration scale {scale} must be finite and at least {MIN_EXPLORATION_SCALE}"
                )));
            }
        }
        Ok(())
    }
}

/// Build the untrained bootstrap artifact for `spec`, without writing it.
///
/// Deterministic: the same specification produces the same bytes and therefore
/// the same content hash, on any machine. A randomly initialised gate would
/// also be untrained, and would additionally be unreproducible -- two
/// operators provisioning the same lineage would get two artifacts that differ
/// in a way neither could explain.
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
pub fn bootstrap_artifact(spec: &BootstrapSpec) -> Result<RouterArtifact, RouterInitError> {
    use lattice_fann::{Activation, Layer, Network};

    spec.validate()?;

    // One linear layer of zeros: every column scores exactly 0 for every
    // input, so selection is whatever the uniform policy does with a tie. That
    // is the baseline being asked for, stated in the weights rather than in a
    // flag the serving path would have to honour.
    let layer = Layer::zeros(
        spec.representation.input_width as usize,
        spec.adapter_names.len(),
        Activation::Linear,
    )
    .map_err(|e| RouterInitError::Gate {
        message: e.to_string(),
    })?;
    let network = Network::new(vec![layer]).map_err(|e| RouterInitError::Gate {
        message: e.to_string(),
    })?;

    Ok(RouterArtifact {
        version: BOOTSTRAP_VERSION,
        adapter_names: spec.adapter_names.clone(),
        representation: spec.representation.clone(),
        gate_bytes: network.to_bytes(),
        provenance: RouterProvenance::recorded(
            bootstrap_policy(spec.tau, spec.exploration_scale),
            RouterLineage {
                schema_id: spec.schema_id.clone(),
                base_model: spec.base_model.clone(),
                // A bootstrap has no parent, and no admission record: it was
                // not admitted on evidence, it was asked for.
                parent: None,
                admission_record: None,
            },
        ),
    })
}

/// Provision `dir` with the bootstrap for `spec`, refusing an occupied one.
///
/// Publication is serialized and create-only. The bootstrap is constructed
/// internally, staged outside the selectable namespace, then its synced files
/// and committed HEAD are published in order. A crash before HEAD leaves no
/// selectable format-5 bootstrap; retry initialization in a fresh directory.
/// An error after HEAD replacement reports an uncertain commit and requires
/// recovery through normal startup selection before any activation.
#[cfg(any(feature = "mixture", feature = "metal-gpu"))]
pub fn init_directory(dir: &Path, spec: &BootstrapSpec) -> Result<PathBuf, RouterInitError> {
    publish_bootstrap(dir, spec)
}

#[cfg(test)]
mod tests;
