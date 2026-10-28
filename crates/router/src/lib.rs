//! Request-level LoRA adapter routing for Lattice.
//!
//! A router chooses, per request, which LoRA adapters a model serves with and how much weight
//! each one gets. This crate holds the gate and everything that keeps it honest across restarts:
//! the on-disk gate artifact ([`state`]), provisioning an untrained bootstrap ([`init`]), recorded
//! routing episodes ([`episode`]), the routing module itself (`mixture`, behind the `mixture`
//! feature), bounded refits from recorded feedback (`refit`) and held-out admission evidence
//! for a refit candidate (`admission`); the last two are behind the `router-learning` feature.

/// Registered held-out evidence for router candidate admission.
#[cfg(feature = "router-learning")]
pub mod admission;
pub mod episode;
/// Provisioning a router state directory with an untrained bootstrap
/// (ADR-096 decision 2).
pub mod init;
/// Adapter routing and mixture support. Requires the `mixture` feature.
#[cfg(feature = "mixture")]
pub mod mixture;
/// Bounded construction of router candidates from recorded feedback.
#[cfg(feature = "router-learning")]
pub mod refit;
/// The on-disk router gate artifact: version, trained-adapter names, payload.
/// Deliberately free of the `mixture` gate — see the module doc comment.
pub mod state;

/// Unsupported, version-coupled helpers for `lattice-serve`, reached through
/// `lattice-inference`'s hidden `serving_support` module. Not public API and not
/// covered by semver.
#[cfg(all(feature = "router-learning", feature = "serving-support"))]
#[doc(hidden)]
pub mod serving_support {
    use crate::mixture::AdapterRouter;
    use crate::refit::{RefitError, RefitSource};
    use crate::state::RouterArtifact;

    /// A field-for-field copy of `router` for refitting.
    pub fn refit_snapshot(router: &AdapterRouter) -> AdapterRouter {
        router.refit_snapshot()
    }

    /// A refit source for the serving router's recorded artifact and policy.
    ///
    /// # Errors
    /// Refuses an ineligible source: learning disabled, an artifact that is not
    /// learner eligible, or a recorded policy or lineage the refit cannot use.
    pub fn refit_source_from_serving(
        artifact: RouterArtifact,
        router: AdapterRouter,
        learning_enabled: bool,
        exploration_scale: Option<f64>,
    ) -> Result<RefitSource, RefitError> {
        RefitSource::from_serving(artifact, router, learning_enabled, exploration_scale)
    }
}
