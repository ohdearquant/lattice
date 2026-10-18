//! Gate-free façade over the `mixture` router (ADR-094 decision 2).
//!
//! The rule this module exists to keep is that a `cfg` never lands on the
//! serving call. Both bodies below are compiled from the same signatures, so a
//! build without `mixture` answers with a refusal a caller can read rather
//! than with an absent symbol.
//!
//! # Why this matches adapters by name
//!
//! `mixture::AdapterRouter::route` (feature `mixture`) maps a gate output column to an
//! adapter **by position**: the selected column index is used directly as
//! `available[idx]`, and the `AdapterId` is a label copied into the result,
//! never matched against anything the gate holds. The gate itself is a bare
//! network — weights and two widths — so nothing in it can say which adapter
//! any column was trained for.
//!
//! That makes the correspondence an unwritten contract owned entirely by the
//! caller, with no instrument that can check it: a gate trained with one
//! adapter at column 3 routes to whatever happens to sit at index 3 in the
//! list the caller passes. Valid weights, no error, wrong adapter.
//!
//! It stayed latent because every existing caller passes a fixed literal pool
//! in the scope where it is used, so position and name cannot disagree.
//! Serving is the first caller whose adapter set changes underneath it: an
//! operator can load or unload an adapter between two requests.
//!
//! So the artifact carries the names it was trained on, this façade builds the
//! router's `available` slice **in the artifact's order** — which is the gate's
//! column order — and it refuses when the resident set does not match. Matching
//! by name is chosen not because it is tidier but because it is the only shape
//! in which a wrong pairing is detectable at all.

use crate::router_state::RouterArtifact;
use crate::serve::ApiError;
use crate::serve::lora::{AdapterIndex, LoraSelection};

// The name-matching machinery below serves the `mixture` body and its tests.
// It is deliberately NOT `#[cfg(feature = "mixture")]` alone: `cargo test
// --workspace` at DEFAULT features is what `scripts/ci.sh` runs, and it is the
// only CI step these tests reach -- the inference mixture test step is
// filtered to `mixture::` and would never select `serve::routing::`. Gating on
// the feature alone would therefore compile the tests out of the one job that
// runs them, which is a worse outcome than the dead-code warning it silences.
// `test` is added rather than an `allow(dead_code)` because it states which
// builds actually use these, instead of suppressing the question.
/// Render a name list for a refusal message.
///
/// Still gated: the routability data is cfg-free, but only the refusal TEXT
/// built from it needs rendering, and that lives with the gate.
#[cfg(any(feature = "mixture", test))]
fn render(names: &[String]) -> String {
    if names.is_empty() {
        "(none)".to_string()
    } else {
        names.join(", ")
    }
}

/// Build the refusal for a gate whose trained set does not match residency.
///
/// Both lists are printed side by side because the operator's next question
/// after "refused" is always "which adapter moved", and a refusal that does
/// not answer it sends them to read two sources by hand.
#[cfg(any(feature = "mixture", test))]
fn mismatch(artifact: &[String], resident: &[String], detail: String) -> ApiError {
    ApiError::BadRequest {
        message: format!(
            "router gate does not match the resident adapters: {detail}. \
             gate was trained on [{}]; resident now [{}]",
            render(artifact),
            render(resident)
        ),
        code: "router_adapter_set_mismatch",
    }
}

/// Names in `a` that are absent from `b`, in `a`'s order.
fn missing(a: &[String], b: &[String]) -> Vec<String> {
    a.iter().filter(|n| !b.contains(n)).cloned().collect()
}

/// First name occurring more than once, if any.
fn first_duplicate(names: &[String]) -> Option<String> {
    names
        .iter()
        .enumerate()
        .find(|(i, n)| names[..*i].contains(n))
        .map(|(_, n)| n.clone())
}

/// Why the resident set can or cannot be routed, as data.
///
/// Data rather than a bool and a message, because multiple readers need this
/// and they must not diverge. `route` refuses on the name conditions below --
/// missing, unexpected, or a duplicate on either side -- because those are
/// what make the name-to-column pairing itself unresolvable.
///
/// `blend_refusal` is different: `route` does not check it (see its own field
/// doc), but `GET /v1/lora` and `routable()` both report it, so an operator
/// sees it coming rather than meeting it only as a 400 once a routed
/// request's blend actually executes, with the same planning check that
/// produced this reason. A surface that recomputed either kind of refusal
/// its own way would eventually disagree with what routing (and the blend)
/// actually do, and that disagreement has no symptom until a request
/// arrives.
///
/// It carries all three name conditions and not just the missing names, for
/// the same reason: routing refuses on any of them, so a report built from
/// one would claim routable over a set that refuses.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Routability {
    /// Trained names that are not resident.
    pub missing: Vec<String>,
    /// Resident names the gate was not trained on.
    pub unexpected: Vec<String>,
    /// A name the gate's trained list holds twice, whose columns therefore
    /// cannot be told apart.
    pub duplicate_trained: Option<String>,
    /// A name resident twice, which stops the name identifying one adapter.
    ///
    /// Kept separate from `duplicate_trained` because the two have different
    /// remedies -- one is a bad artifact, the other is a residency an operator
    /// can fix -- and a single field would make the report name neither.
    pub duplicate_resident: Option<String>,
    /// Why a blend of the FULL resident set would refuse at execution
    /// (issue #1735), or `None` when it would not.
    ///
    /// Independent of whether the trained set matches residency BY NAME: a
    /// set can pass every check above -- nothing missing, nothing
    /// unexpected, no duplicate on either side -- and still refuse to
    /// blend, because a routed request applies every resident adapter and
    /// the blend has its own rank and size budget. `route` does not refuse
    /// on this field: a routed request over such a set fails when the blend
    /// executes, with the same planning check that produced this reason.
    pub blend_refusal: Option<String>,
}

impl Routability {
    /// True when a request against this residency would succeed end to end
    /// right now: the name conditions above -- the ones `route` itself
    /// refuses on -- are all clear, AND the resulting blend would not
    /// refuse when it executes. `route` only checks the first half; this is
    /// the one place that also asks the second, which is why it can answer
    /// `false` on a set `route` alone would accept.
    pub fn routable(&self) -> bool {
        self.missing.is_empty()
            && self.unexpected.is_empty()
            && self.duplicate_trained.is_none()
            && self.duplicate_resident.is_none()
            && self.blend_refusal.is_none()
    }
}

/// Compare the artifact's trained set against residency, in both directions,
/// and carry the resident set's own blend feasibility (issue #1735) beside
/// it -- one refusal reason, however it arose.
pub fn routability(artifact: &RouterArtifact, resident: &AdapterIndex) -> Routability {
    let trained = &artifact.adapter_names;
    let live: Vec<String> = resident.adapters.iter().map(|a| a.name.clone()).collect();
    Routability {
        missing: missing(trained, &live),
        unexpected: missing(&live, trained),
        duplicate_trained: first_duplicate(trained),
        duplicate_resident: first_duplicate(&live),
        blend_refusal: resident.blend_refusal.clone(),
    }
}

/// Check the artifact's trained set against residency in BOTH directions and
/// return the adapter names in the gate's column order.
///
/// Refusing only one direction leaves the other silent: they are the same
/// unverifiable pairing seen from opposite sides. Differences of ORDER between
/// the two lists are resolved by name and are not a mismatch — position is the
/// thing this module exists to stop trusting.
#[cfg(any(feature = "mixture", test))]
fn trained_order(
    artifact: &RouterArtifact,
    resident: &AdapterIndex,
) -> Result<Vec<String>, ApiError> {
    let trained = &artifact.adapter_names;
    let live: Vec<String> = resident.adapters.iter().map(|a| a.name.clone()).collect();

    // DERIVED from the same value the surface reports, never re-checked here.
    // Two copies of this comparison would drift, and the drift would show up
    // as `GET /v1/lora` promising a route that the next request refuses.
    let state = routability(artifact, resident);

    // A duplicate name on either side makes the name-to-id resolution
    // ambiguous, and picking either candidate is exactly the silent
    // wrong-adapter selection this module exists to prevent. Not one of the
    // directions the ADR enumerates, because it is not a set difference: two
    // sides can agree as SETS and still be unresolvable.
    if let Some(dup) = state.duplicate_trained {
        return Err(mismatch(
            trained,
            &live,
            format!(
                "the gate's trained list names {dup:?} more than once, so its columns cannot be told apart"
            ),
        ));
    }
    if let Some(dup) = state.duplicate_resident {
        return Err(mismatch(
            trained,
            &live,
            format!(
                "{dup:?} is resident more than once, so the name does not identify one adapter"
            ),
        ));
    }
    if !state.missing.is_empty() {
        return Err(mismatch(
            trained,
            &live,
            format!(
                "[{}] the gate was trained on are not resident",
                render(&state.missing)
            ),
        ));
    }
    if !state.unexpected.is_empty() {
        return Err(mismatch(
            trained,
            &live,
            format!(
                "[{}] are resident but not in the gate's trained set",
                render(&state.unexpected)
            ),
        ));
    }

    Ok(trained.clone())
}

/// Resolve routed names back to the `u32` ids the serving contract speaks.
///
/// The two id types do not meet anywhere else: the router speaks
/// `AdapterId = String` and the contract speaks `LoraSelection.id: u32`.
/// Naming the trained set is what gives that translation a defined direction
/// instead of an implied one.
#[cfg(any(feature = "mixture", test))]
fn to_selections(
    routed: Vec<(String, f32)>,
    resident: &AdapterIndex,
) -> Result<Vec<LoraSelection>, ApiError> {
    routed
        .into_iter()
        .map(|(name, scale)| {
            let id = resident
                .adapters
                .iter()
                .find(|a| a.name == name)
                .map(|a| a.id)
                .ok_or_else(|| ApiError::Internal {
                    // Unreachable through `route`, which checks both
                    // directions first. Still an error rather than a
                    // `expect`: the alternative to a 500 here is a panic in a
                    // request handler.
                    message: format!("routed adapter {name:?} vanished from residency"),
                })?;
            serde_json::from_value::<LoraSelection>(serde_json::json!({"id": id, "scale": scale}))
                .map_err(|err| ApiError::Internal {
                    message: format!("could not build a selection for {name:?}: {err}"),
                })
        })
        .collect()
}

/// The serving path's router.
///
/// A concrete type with a `cfg`-selected body rather than a trait: one
/// implementor is not a trait's reason to exist, and a trait here would add a
/// dispatch seam whose only caller is the one it is wired into.
#[cfg(feature = "mixture")]
pub struct ServingRouter {
    router: crate::mixture::AdapterRouter,
    artifact: RouterArtifact,
    learning_enabled: bool,
    exploration_scale: Option<f64>,
}

/// Install the artifact's recorded policy on a freshly built router, or
/// refuse why it cannot be installed (ADR-096 decisions 3 and 5).
///
/// Returns whether learning is in force, which is not the same question as
/// whether the flag was given. Four outcomes, and the reasoning behind each
/// is below rather than in a comment on the single `match`:
///
/// A recorded learned policy WITHOUT `learning` fails the startup rather than
/// being served uniform. Serving it uniform would make one version label name
/// two behaviours, which is exactly what putting the policy inside the
/// artifact's content hash prevents, and an operator would read a learned
/// version number off a server that never ran it.
///
/// `learning` over an artifact recording a uniform policy, or recording no
/// policy at all, fails too, and those are two DIFFERENT refusals: nothing was
/// written down is not the same as uniform having been written down. Serving
/// uniform while reporting learning as enabled would be true of the flag and
/// false of the behaviour.
///
/// A recorded UNIFORM policy with `learning` off installs NOTHING and the
/// router keeps its own defaults, which is what this server serves today. Its
/// recorded guard fields are deliberately not installed: a bootstrap records
/// every one of them as zero because the hashed policy shape must not depend
/// on how the artifact was produced, so installing them would put a zeroed
/// `tau` floor on a server that never asked for one.
#[cfg(feature = "mixture")]
fn apply_recorded_policy(
    router: &mut crate::mixture::AdapterRouter,
    artifact: &RouterArtifact,
    learning: bool,
) -> Result<bool, ApiError> {
    let Some(recorded) = artifact.provenance.policy() else {
        if learning {
            return Err(ApiError::BadRequest {
                message: "learning was enabled but this artifact records no policy at all (it \
                          predates the policy field), so there is no learned policy to serve"
                    .to_string(),
                code: "router_no_recorded_policy",
            });
        }
        return Ok(false);
    };

    match (recorded.kind.as_str(), learning) {
        ("uniform", false) => Ok(false),
        ("uniform", true) => Err(ApiError::BadRequest {
            message: format!(
                "learning was enabled but artifact {} records a uniform policy, which has \
                 nothing learned to serve",
                artifact.version_label()
            ),
            code: "router_policy_not_learned",
        }),
        (_, false) => Err(ApiError::BadRequest {
            message: format!(
                "artifact {} records a {} policy, and this server did not enable learning; \
                 serving it uniform would make one version label name two behaviours",
                artifact.version_label(),
                recorded.kind
            ),
            code: "router_learning_not_enabled",
        }),
        (_, true) => {
            let tau = recorded.tau.ok_or_else(|| ApiError::Internal {
                message: "a softmax policy reached the serving path with no temperature; \
                          validation should have refused it"
                    .to_string(),
            })?;
            // Floors before the temperature they bound: installing `tau` first
            // would put it briefly under the router's default floor, and a
            // refusal ordering that depends on which setter ran first is one
            // nobody can reason about from the artifact.
            router.set_tau_floor(as_served_f32(recorded.tau_floor));
            router.set_entropy_floor(as_served_f32(recorded.entropy_floor));
            router.set_max_consecutive_collapsed(recorded.max_consecutive_collapsed as usize);
            router.set_epsilon(as_served_f32(recorded.epsilon));
            router.set_weight_policy(crate::mixture::WeightPolicy::Softmax {
                tau: as_served_f32(tau),
            });
            Ok(true)
        }
    }
}

/// The artifact stores its policy floats at the width its hash is computed
/// over; the router compares them against `f32` scores. Narrowed in one named
/// place so the conversion is visible rather than scattered through the
/// installer.
///
/// The narrowing is why [`ServingRouter::applied_policy`] can report a
/// temperature whose last bits differ from the recorded one. That is the
/// report doing its job: it says what the router is running, and what the
/// router is running is the narrowed value.
#[cfg(feature = "mixture")]
fn as_served_f32(value: f64) -> f32 {
    value as f32
}

#[cfg(feature = "mixture")]
impl ServingRouter {
    /// Capture the producing gate and policy for a bounded candidate round.
    ///
    /// Take a fresh snapshot for every round, then release the serving lock
    /// before building. Construction does not admit or activate a candidate.
    #[cfg(feature = "router-learning")]
    pub fn refit_source(
        &self,
    ) -> Result<crate::router_refit::RefitSource, crate::router_refit::RefitError> {
        crate::router_refit::RefitSource::from_serving(
            self.artifact.clone(),
            self.router.refit_snapshot(),
            self.learning_enabled,
            self.exploration_scale,
        )
    }

    /// Build a router from a loaded gate artifact.
    ///
    /// `learning` enables the artifact's recorded learned policy (ADR-096
    /// decision 3). It is off by default at the CLI, and the three refusals
    /// that make the flag meaningful are in `apply_recorded_policy`.
    pub fn new(artifact: RouterArtifact, learning: bool) -> Result<Self, ApiError> {
        let gate = lattice_fann::Network::from_bytes(&artifact.gate_bytes).map_err(|err| {
            ApiError::Internal {
                message: format!("router gate did not load: {err}"),
            }
        })?;

        // THE NAME LIST MUST DESCRIBE THE GATE IT SHIPS WITH. Everything above
        // treats `adapter_names` as the column labelling, so an artifact whose
        // list is a different length than the gate's output width is an
        // artifact lying about its own gate -- and it would pass every check
        // in this module, because they all compare the list against residency
        // and never against the network.
        //
        // The failure is silent by construction: `route` uses
        // `available.len().min(scores.len())` columns, so a gate wider than
        // the list simply ignores its trailing columns. Nothing errors, and
        // the adapters those columns were trained for never get selected.
        //
        // Equality rather than "at least": the list IS the labelling, so an
        // unnamed column is a column nothing can route to or verify.
        // RECORDED against MEASURED, named separately from the output-width
        // check below because the faults differ. This one says the artifact
        // describes itself wrongly; the server's embedding model is not
        // implicated and re-configuring it would not help.
        let recorded = artifact.representation.input_width;
        let measured = gate.num_inputs() as u64;
        if recorded != measured {
            return Err(ApiError::BadRequest {
                message: format!(
                    "router artifact records an input width of {recorded} but its gate takes \
                     {measured}; the artifact disagrees with the network it ships with"
                ),
                code: "router_artifact_input_width_mismatch",
            });
        }

        let declared = artifact.adapter_names.len();
        let outputs = gate.num_outputs();
        if declared != outputs {
            return Err(ApiError::BadRequest {
                message: format!(
                    "router artifact declares {declared} adapter name(s) but its gate has \
                     {outputs} output column(s); the name list is the column labelling, so \
                     they must agree"
                ),
                code: "router_artifact_width_mismatch",
            });
        }

        let mut router = crate::mixture::AdapterRouter::new(gate);
        let learning_enabled = apply_recorded_policy(&mut router, &artifact, learning)?;
        // Carried only when learning is enabled: the scale is the artifact's,
        // and on a server that did not enable the recorded policy nothing
        // draws, so reporting a scale would name one a request never ran at.
        let exploration_scale = learning_enabled
            .then(|| {
                artifact
                    .provenance
                    .policy()
                    .and_then(|policy| policy.exploration_scale)
            })
            .flatten();

        Ok(Self {
            router,
            artifact,
            learning_enabled,
            exploration_scale,
        })
    }

    /// The policy this router is serving under, which is not always the one
    /// its artifact records: a recorded learned policy that was not enabled is
    /// not applied, and reporting the recorded one would name a policy the
    /// server is not running. See [`crate::router_state::AppliedPolicy`].
    ///
    /// The values are read from the live router, so they are the narrowed
    /// `f32`s the router computes in rather than the `f64`s the artifact
    /// records; the two can differ in their last bits and this reports the one
    /// a request is served under.
    pub fn applied_policy(&self) -> crate::router_state::AppliedPolicy {
        let (kind, tau) = match self.router.weight_policy() {
            crate::mixture::WeightPolicy::Uniform => ("uniform", None),
            crate::mixture::WeightPolicy::Softmax { tau } => ("softmax", Some(f64::from(tau))),
        };
        crate::router_state::AppliedPolicy {
            kind: kind.to_string(),
            tau,
            epsilon: f64::from(self.router.epsilon()),
            exploration_scale: self.exploration_scale,
            learning_enabled: self.learning_enabled,
        }
    }

    /// Route one request against the currently resident adapters.
    ///
    /// `RouterError` renders into the message through `Display` rather than
    /// becoming a new error type: a move should not change the text a user
    /// sees, and a new enum would exist only to be converted at this boundary.
    ///
    /// `completion_id` seeds the draw when one happens. It is the id the
    /// response will carry, so the seed an episode records replays from the
    /// episode alone -- which is why the caller mints the id BEFORE routing
    /// rather than after: a seed from an id that does not exist yet would have
    /// to be random, and a random seed is recorded truthfully and reproducible
    /// by nobody.
    pub fn route(
        &mut self,
        resident: &AdapterIndex,
        context_vector: Vec<f32>,
        k: usize,
        completion_id: &str,
    ) -> Result<RoutedRequest, ApiError> {
        let order = trained_order(&self.artifact, resident)?;
        let draw = self
            .exploration_scale
            .map(|scale| crate::mixture::Exploration {
                // Narrowed for the same reason the report narrows: this is the
                // value the router computes in, and recording the f64 would
                // name a scale the draw did not use.
                scale: as_served_f32(scale),
                seed: crate::mixture::seed_from_completion_id(completion_id),
            });
        let outcome = self
            .router
            .route_explored(&context_vector, &order, k, draw)
            .map_err(|err| ApiError::BadRequest {
                message: format!("router refused: {err}"),
                code: "router_refused",
            })?;
        // Read from the policy in force inside this same lock acquisition, not
        // from the artifact: the artifact records what was admitted, the router
        // carries what a request runs under, and the installed policy is the
        // router's.
        let tau = match self.router.weight_policy() {
            crate::mixture::WeightPolicy::Uniform => None,
            crate::mixture::WeightPolicy::Softmax { tau } => Some(tau),
        };
        Ok(RoutedRequest {
            selections: to_selections(outcome.weights.clone(), resident)?,
            context_vector,
            selected_scores: outcome.selected_scores,
            sampled_scores: outcome.sampled_scores,
            pre_floor_weights: outcome.pre_floor_weights,
            applied_weights: outcome.weights,
            dropped: outcome.dropped,
            tau,
            policy_epoch: self.router.policy_epoch(),
            #[cfg(feature = "mixture")]
            exploration: outcome.exploration,
        })
    }

    /// The gate artifact this router is serving.
    pub fn artifact(&self) -> &RouterArtifact {
        &self.artifact
    }

    /// The policy epoch the gate is currently serving under.
    pub fn policy_epoch(&self) -> u64 {
        self.router.policy_epoch()
    }
}

/// One routing decision as the serving path needs it: what to apply, and what
/// the decision was made from.
///
/// The provenance travels with the selections rather than being fetched
/// afterwards, because "what did the gate score" has no answer once the next
/// request has run through the same router. ADR-096 decision 3 requires a
/// feedback event to carry the sampled logits and both weight vectors, and
/// there is exactly one moment at which all three exist.
///
/// Defined outside the `mixture` gate, like every other type here: a build
/// without a gate still has to name this in the signature it refuses from.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedRequest {
    /// The adapters to apply, resolved against the resident index.
    pub selections: Vec<LoraSelection>,
    /// The vector the gate consumed.
    ///
    /// Moved in rather than filled afterwards. A field a caller is expected to
    /// populate after construction is a field that is empty whenever someone
    /// forgets, and an episode carrying an empty context vector is one the
    /// learner cannot use and nothing rejects.
    pub context_vector: Vec<f32>,
    /// The gate scores `z` of the selected columns, by adapter name, in
    /// selection order.
    pub selected_scores: Vec<(String, f32)>,
    /// The logits the selection and the weight policy consumed, by adapter
    /// name, in selection order: `z` on the deterministic path and `z'` under
    /// a draw.
    pub sampled_scores: Vec<(String, f32)>,
    /// What the policy produced before the floor, by adapter name.
    pub pre_floor_weights: Vec<(String, f32)>,
    /// What the request ran with, by adapter name.
    pub applied_weights: Vec<(String, f32)>,
    /// Adapters the `epsilon` floor removed, by name.
    pub dropped: Vec<String>,
    /// The softmax temperature the mixture was formed at, or `None` for the
    /// uniform policy, which reads no temperature.
    ///
    /// Read off the policy in force when the decision was made, so an event
    /// records the temperature that produced its weights and not one installed
    /// since.
    pub tau: Option<f32>,
    /// The policy epoch the decision was made under.
    ///
    /// Read inside the same lock acquisition that produced the decision, and
    /// carried out with it. Read afterwards it would be a different question:
    /// a reload between the two reads would date this episode to a policy that
    /// did not produce it, and an off-epoch sample is discarded rather than
    /// refused, so nothing would ever report the mislabelling.
    pub policy_epoch: u64,
    /// The draw, when learning was enabled and the artifact recorded a scale.
    ///
    /// `None` is the ordinary case and means the decision was deterministic,
    /// not that a draw happened and went unrecorded. The two are different
    /// claims and a learner treats them differently: an absent record is an
    /// off-policy sample it must not attribute, while a present one carries the
    /// sampled logits `z'`, the scale `s`, the seed and the log density of `z'`
    /// that make attribution possible. There is no action probability: under
    /// this draw the policy is Gaussian on the logits, so what the learner
    /// scores is the density of `z'` (see
    /// [`crate::mixture::ExplorationOutcome::log_density`]).
    #[cfg(feature = "mixture")]
    pub exploration: Option<crate::mixture::ExplorationOutcome>,
}

/// The serving path's router — refusal body.
///
/// Compiled in rather than absent, so the missing configuration is a runtime
/// answer a caller can read. This is decision 1's shape and `lora_list`'s
/// shape, reused.
#[cfg(not(feature = "mixture"))]
pub struct ServingRouter {
    artifact: RouterArtifact,
}

#[cfg(not(feature = "mixture"))]
fn router_unsupported_build() -> ApiError {
    ApiError::BadRequest {
        message: "adapter routing requires a build with the `mixture` feature; this server was \
                  built without it and cannot run a gate"
            .to_string(),
        code: "router_unsupported_build",
    }
}

#[cfg(not(feature = "mixture"))]
impl ServingRouter {
    /// Refuse: this build has no gate implementation.
    pub fn new(_artifact: RouterArtifact, _learning: bool) -> Result<Self, ApiError> {
        Err(router_unsupported_build())
    }

    /// Refuse. Unreachable while `new` refuses, and kept so the two bodies
    /// present the same signatures — the point of the façade is that the
    /// serving call site does not change shape between builds.
    pub fn route(
        &mut self,
        _resident: &AdapterIndex,
        _context_vector: Vec<f32>,
        _k: usize,
        _completion_id: &str,
    ) -> Result<RoutedRequest, ApiError> {
        Err(router_unsupported_build())
    }

    /// The gate artifact this router would serve.
    pub fn artifact(&self) -> &RouterArtifact {
        &self.artifact
    }

    /// Unobservable: `new` always refuses in this build, so no value of this
    /// type exists to call it on. Present for the same reason `route` is --
    /// the façade's signatures do not change shape between builds -- and not
    /// a claim that this server serves epoch zero.
    pub fn policy_epoch(&self) -> u64 {
        0
    }

    /// Unobservable, for the same reason as [`policy_epoch`](Self::policy_epoch).
    /// Reports the uniform default this build would serve: it installed no
    /// policy, because `new` refuses before one could be installed.
    pub fn applied_policy(&self) -> crate::router_state::AppliedPolicy {
        crate::router_state::AppliedPolicy {
            kind: "uniform".to_string(),
            tau: None,
            epsilon: 0.0,
            exploration_scale: None,
            learning_enabled: false,
        }
    }
}

/// Which text of a request becomes the gate's context vector.
///
/// An enum with one variant rather than a bare string comparison, because the
/// point is that the serving path and the artifact name the same rule from one
/// definition. A second variant is the moment this earns itself; today its job
/// is to make the rule a value that can be compared at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptSource {
    /// The last user message, alone. Not the rendered conversation: that makes
    /// the vector drift with history length, so two requests asking the same
    /// question route differently depending on how long the chat has been
    /// going.
    LastUserMessage,
}

impl PromptSource {
    /// The rule this build's serving path actually follows.
    pub const SERVED: Self = Self::LastUserMessage;

    /// How the artifact spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LastUserMessage => "last_user_message",
        }
    }
}

/// The text a request contributes to the gate, under a given rule.
///
/// Takes `(is_user, text)` turns rather than a message type. The two binaries
/// hold different message structs and the engine's is Metal-gated, so naming
/// one here would either gate this module or pick a favourite; a trait for two
/// callers is the seam this module's own doc argues against. The caller states
/// which turns are the user's, in one visible expression.
///
/// It lives beside [`PromptSource`] so the rule has ONE definition: the
/// artifact records a name, `check_representation` compares it, and this
/// produces the text. Split apart, those three drift with no symptom, because
/// the vector is the right width either way.
///
/// `None` when the request carries no text under the rule. No substitute is
/// invented: routing on the system prompt instead would use text the gate
/// never saw and would look like it worked.
pub fn context_text<'a>(
    source: PromptSource,
    turns: impl DoubleEndedIterator<Item = (bool, &'a str)>,
) -> Option<&'a str> {
    match source {
        PromptSource::LastUserMessage => {
            let mut turns = turns;
            turns.rfind(|(is_user, _)| *is_user).map(|(_, text)| text)
        }
    }
}

/// Resolve the prompt-selection rule the gate was trained under.
pub fn trained_prompt_source(source: &str) -> Result<PromptSource, ApiError> {
    if source == PromptSource::LastUserMessage.as_str() {
        return Ok(PromptSource::LastUserMessage);
    }
    Err(ApiError::BadRequest {
        message: format!(
            "router artifact records prompt source {source:?}, which this build does not know how to reproduce; the gate would be routed on text it was not trained on"
        ),
        code: "router_artifact_unknown_prompt_source",
    })
}

/// Resolve the pooling the gate was trained under.
///
/// Parsed once, at startup, rather than per request: an unrecognised value is
/// a property of the artifact, so it is true of every request and there is no
/// reason for a caller to be the one who learns about it.
///
/// Public so `router init` validates its `--pooling` with the serving path's
/// OWN parser rather than a second list. A second list is the shape where a
/// build mints an artifact naming a rule it cannot reproduce, and every check
/// downstream passes until a request arrives.
pub fn trained_pooling(
    pooling: &str,
) -> Result<crate::forward::cpu_f16::PoolingStrategy, ApiError> {
    match pooling {
        "mean_visual" => Ok(crate::forward::cpu_f16::PoolingStrategy::MeanVisualTokens),
        "last_token" => Ok(crate::forward::cpu_f16::PoolingStrategy::LastToken),
        other => Err(ApiError::BadRequest {
            message: format!(
                "router artifact records pooling {other:?}, which this build does not know how to reproduce; the gate would be served a representation it was not trained on"
            ),
            code: "router_artifact_unknown_pooling",
        }),
    }
}

/// The identity recorded for a server's embedding model.
///
/// A type rather than a `&str` because the defect it prevents was exactly a
/// `&str`: the startup path had the chat model's id and the embedder's object
/// in the same scope and passed the chat model's, which every test agreed with
/// because the embedder used to be loaded from the served model's own
/// directory. The two are different strings the moment they can be different
/// directories, and nothing about the old signature said which one it wanted.
///
/// Derivation lives here rather than in the binary so it can be tested at all;
/// a `main.rs` that exits the process is not reachable from a unit test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedderIdentity(String);

impl EmbedderIdentity {
    /// The directory basename, or an explicit override.
    ///
    /// Mirrors how `served_model_id` is derived from `--model`/`--model-id`,
    /// weakness included: it is a NAME. It separates checkpoints an operator
    /// has separated, and a different checkpoint of the same family and width
    /// under a directory of the same name reads identical. The config cannot
    /// close that gap — it carries `model_type`, which is the family.
    pub fn resolve(dir: &std::path::Path, override_id: Option<&str>) -> Self {
        if let Some(id) = override_id {
            return Self(id.to_string());
        }
        // `file_name` is None for a path ending in `..` and for a bare root,
        // neither of which names a checkpoint; falling back keeps startup on
        // the refusal path (the artifact will not match "embedder") rather
        // than panicking on an operator's typo.
        Self(
            dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("embedder")
                .to_string(),
        )
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Check a gate artifact against the embedding model this server actually
/// loaded, and return the pooling the routing path must use.
///
/// This is the pairing [`ServingRouter::new`] cannot see. That constructor
/// checks the artifact against the gate it ships with, which says the artifact
/// describes itself correctly and nothing about whether this server can
/// reproduce the representation it names.
///
/// The identity compared is the embedder's: the `--embedding-model`
/// directory's basename, or `--embedding-model-id` when given. It is never the
/// chat model's id, and never the full path, which is machine-local and would
/// refuse a correct artifact for having been trained on a box that stored the
/// checkpoint elsewhere.
///
/// A name is weak evidence, and the refusal is deliberately one-directional
/// because of it: two checkpoints can share a name, and fine-tuning changes
/// the representation without changing the name. So a mismatch refuses, and a
/// match is not a warrant of sameness. The width comparison below is the half
/// that is actually checkable, and its passing is exactly what makes the
/// unchecked half look settled.
pub fn check_representation(
    artifact: &RouterArtifact,
    embedder_identity: &EmbedderIdentity,
    dimensions: usize,
) -> Result<crate::forward::cpu_f16::PoolingStrategy, ApiError> {
    // The EMBEDDER's identity, never the chat model's. Those were the same
    // string before ADR-094 decision 1 amendment 2, because the embedder was
    // loaded from the served model's own directory -- so this check passed by
    // CO-LOCATION rather than by agreement, and the moment the two could be
    // different directories it would have been comparing a gate's embedder
    // against a chat model's name without anything saying so.
    let trained_on = &artifact.representation.embedding_model;
    if trained_on != embedder_identity.as_str() {
        return Err(ApiError::BadRequest {
            message: format!(
                "router gate was trained on embedding model {trained_on:?} but this server embeds with {:?}; the gate would route on a representation it never saw",
                embedder_identity.as_str()
            ),
            code: "router_representation_model_mismatch",
        });
    }

    // The identity above is a NAME, so it separates checkpoints an operator
    // has separated and nothing more. This says how the bytes were READ: two
    // loaders over the same directory can produce different vectors of the
    // same width from the same text, and the width check below cannot see
    // that. One value exists today, which is exactly why it is cheap to
    // record now and impossible to add later without invalidating every
    // artifact already written.
    let trained_loader = &artifact.representation.loader_format;
    if trained_loader != crate::serve::embeddings::QWEN35_F16_DECODER_LOADER {
        return Err(ApiError::BadRequest {
            message: format!(
                "router gate was trained through the {trained_loader:?} embedding loader but this server embeds through {:?}",
                crate::serve::embeddings::QWEN35_F16_DECODER_LOADER
            ),
            code: "router_representation_loader_mismatch",
        });
    }

    let recorded = artifact.representation.input_width;
    let measured = dimensions as u64;
    if recorded != measured {
        return Err(ApiError::BadRequest {
            message: format!(
                "router gate takes a {recorded}-dimension context vector but this server's embedding model produces {measured} dimensions"
            ),
            code: "router_representation_width_mismatch",
        });
    }

    // Checked even though only one rule exists, and the check cannot fail
    // against an artifact this build wrote. It can fail against one written by
    // a build that had a second rule, which is the case that has no symptom:
    // same model, same pooling, same width, different text.
    let source = trained_prompt_source(&artifact.representation.prompt_source)?;
    if source != PromptSource::SERVED {
        return Err(ApiError::BadRequest {
            message: format!(
                "router gate was trained on the {} of a request but this server routes on the {}",
                source.as_str(),
                PromptSource::SERVED.as_str()
            ),
            code: "router_representation_prompt_source_mismatch",
        });
    }

    trained_pooling(&artifact.representation.pooling)
}

/// The serving state's router: one gate, the pooling it was trained under, and
/// what `GET /v1/lora` reports about it.
///
/// One artifact, borrowed by the reporter rather than copied for it. A second
/// copy would be a second thing a refit has to keep in step, with nothing able
/// to notice when it does not.
pub struct ServedRouter {
    router: std::sync::Mutex<ServingRouter>,
    pooling: crate::forward::cpu_f16::PoolingStrategy,
    pinned: bool,
    embedder: EmbedderIdentity,
}

impl ServedRouter {
    /// Build the serving router, checking it against this server's embedding
    /// model before anything is served.
    ///
    /// Both checks run here rather than at the first request for ADR-095
    /// decision 6's reason: "routing is configured" and "routing can run"
    /// become the same question, answered while the operator is still watching
    /// the process start.
    pub fn new(
        resolved: crate::router_state::ResolvedRouter,
        embedder_identity: &EmbedderIdentity,
        dimensions: usize,
        learning: bool,
    ) -> Result<Self, ApiError> {
        // Ordered from the most general refusal to the most specific, which
        // is the same rule the `/v1/lora` handlers follow. A build with no
        // gate implementation cannot route any artifact, so it answers first;
        // an artifact that disagrees with the gate it ships with is wrong on
        // every server, so it answers next; only then does this server's own
        // embedding model enter it. Reversed, an operator on a build without
        // `mixture` is told to go fix a representation that was never the
        // problem.
        let router = ServingRouter::new(resolved.artifact, learning)?;
        let pooling = check_representation(router.artifact(), embedder_identity, dimensions)?;
        Ok(Self {
            router: std::sync::Mutex::new(router),
            pooling,
            pinned: resolved.pinned,
            embedder: embedder_identity.clone(),
        })
    }

    /// The pooling the routing path must embed with.
    pub fn pooling(&self) -> crate::forward::cpu_f16::PoolingStrategy {
        self.pooling
    }

    /// Read the serving artifact for a report.
    ///
    /// A poisoned lock is recovered rather than propagated. The alternative is
    /// that one panic inside routing makes `GET /v1/lora` permanently
    /// unanswerable, which withholds the state an operator needs precisely
    /// when something has gone wrong. The artifact is immutable for the life
    /// of the process, so a reader cannot observe a half-written one.
    pub fn with_report<R>(&self, f: impl FnOnce(crate::router_state::RouterReport<'_>) -> R) -> R {
        let guard = self
            .router
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(crate::router_state::RouterReport {
            artifact: guard.artifact(),
            applied_policy: guard.applied_policy(),
            pinned: self.pinned,
            embedder: self.embedder.as_str(),
        })
    }

    /// Route one request.
    ///
    /// Unlike the report path, a poisoned lock refuses here. A panic inside
    /// the gate's forward pass leaves the router's own scratch state
    /// unaccounted for, and routing on it would produce a selection nobody can
    /// argue is correct -- which is the silent wrong-adapter outcome this
    /// module exists to prevent.
    pub fn route(
        &self,
        resident: &AdapterIndex,
        context_vector: Vec<f32>,
        k: usize,
        completion_id: &str,
    ) -> Result<RoutedRequest, ApiError> {
        let mut guard = self.router.lock().map_err(|_| ApiError::Internal {
            message: "the adapter router panicked on an earlier request and is no longer serving"
                .to_string(),
        })?;
        guard.route(resident, context_vector, k, completion_id)
    }

    /// The policy epoch the next request would route under.
    ///
    /// Tolerant of a poisoned lock, like [`with_report`](Self::with_report)
    /// and unlike [`route`](Self::route): this answers a question about state,
    /// and refusing to say which policy is loaded withholds it exactly when
    /// something has already gone wrong. Nothing routes on the value.
    ///
    /// Callers read this before taking the episode store's lock, never while
    /// holding it. The two locks are never nested in either order.
    pub fn policy_epoch(&self) -> u64 {
        let guard = self
            .router
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.policy_epoch()
    }
}

#[cfg(test)]
mod tests;
