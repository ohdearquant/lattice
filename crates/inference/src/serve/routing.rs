//! Gate-free façade over the `mixture` router (ADR-094 decision 2).
//!
//! The rule this module exists to keep is that a `cfg` never lands on the
//! serving call. Both bodies below are compiled from the same signatures, so a
//! build without `mixture` answers with a refusal a caller can read rather
//! than with an absent symbol.
//!
//! # Why this matches adapters by name
//!
//! [`crate::mixture::AdapterRouter::route`] maps a gate output column to an
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
#[cfg(any(feature = "mixture", test))]
fn missing(a: &[String], b: &[String]) -> Vec<String> {
    a.iter().filter(|n| !b.contains(n)).cloned().collect()
}

/// First name occurring more than once, if any.
#[cfg(any(feature = "mixture", test))]
fn first_duplicate(names: &[String]) -> Option<String> {
    names
        .iter()
        .enumerate()
        .find(|(i, n)| names[..*i].contains(n))
        .map(|(_, n)| n.clone())
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

    // A duplicate name on either side makes the name-to-id resolution
    // ambiguous, and picking either candidate is exactly the silent
    // wrong-adapter selection this module exists to prevent. Not one of the
    // directions the ADR enumerates, because it is not a set difference: two
    // sides can agree as SETS and still be unresolvable.
    if let Some(dup) = first_duplicate(trained) {
        return Err(mismatch(
            trained,
            &live,
            format!(
                "the gate's trained list names {dup:?} more than once, so its columns cannot be told apart"
            ),
        ));
    }
    if let Some(dup) = first_duplicate(&live) {
        return Err(mismatch(
            trained,
            &live,
            format!(
                "{dup:?} is resident more than once, so the name does not identify one adapter"
            ),
        ));
    }

    let absent = missing(trained, &live);
    if !absent.is_empty() {
        return Err(mismatch(
            trained,
            &live,
            format!(
                "[{}] the gate was trained on are not resident",
                render(&absent)
            ),
        ));
    }
    let unknown = missing(&live, trained);
    if !unknown.is_empty() {
        return Err(mismatch(
            trained,
            &live,
            format!(
                "[{}] are resident but not in the gate's trained set",
                render(&unknown)
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
}

#[cfg(feature = "mixture")]
impl ServingRouter {
    /// Build a router from a loaded gate artifact.
    pub fn new(artifact: RouterArtifact) -> Result<Self, ApiError> {
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

        Ok(Self {
            router: crate::mixture::AdapterRouter::new(gate),
            artifact,
        })
    }

    /// Route one request against the currently resident adapters.
    ///
    /// `RouterError` renders into the message through `Display` rather than
    /// becoming a new error type: a move should not change the text a user
    /// sees, and a new enum would exist only to be converted at this boundary.
    pub fn route(
        &mut self,
        resident: &AdapterIndex,
        context_vector: &[f32],
        k: usize,
    ) -> Result<Vec<LoraSelection>, ApiError> {
        let order = trained_order(&self.artifact, resident)?;
        let routed = self
            .router
            .route(context_vector, &order, k)
            .map_err(|err| ApiError::BadRequest {
                message: format!("router refused: {err}"),
                code: "router_refused",
            })?;
        to_selections(routed, resident)
    }

    /// The gate artifact this router is serving.
    pub fn artifact(&self) -> &RouterArtifact {
        &self.artifact
    }
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
    pub fn new(_artifact: RouterArtifact) -> Result<Self, ApiError> {
        Err(router_unsupported_build())
    }

    /// Refuse. Unreachable while `new` refuses, and kept so the two bodies
    /// present the same signatures — the point of the façade is that the
    /// serving call site does not change shape between builds.
    pub fn route(
        &mut self,
        _resident: &AdapterIndex,
        _context_vector: &[f32],
        _k: usize,
    ) -> Result<Vec<LoraSelection>, ApiError> {
        Err(router_unsupported_build())
    }

    /// The gate artifact this router would serve.
    pub fn artifact(&self) -> &RouterArtifact {
        &self.artifact
    }
}

/// Resolve the pooling the gate was trained under.
///
/// Parsed once, at startup, rather than per request: an unrecognised value is
/// a property of the artifact, so it is true of every request and there is no
/// reason for a caller to be the one who learns about it.
fn trained_pooling(pooling: &str) -> Result<crate::forward::cpu_f16::PoolingStrategy, ApiError> {
    match pooling {
        "mean_visual" => Ok(crate::forward::cpu_f16::PoolingStrategy::MeanVisualTokens),
        "last_token" => Ok(crate::forward::cpu_f16::PoolingStrategy::LastToken),
        other => Err(ApiError::BadRequest {
            message: format!(
                "router artifact records pooling {other:?}, which this build does not know how to                  reproduce; the gate would be served a representation it was not trained on"
            ),
            code: "router_artifact_unknown_pooling",
        }),
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
/// The identity compared is the served model id -- the name `/v1/models`
/// publishes and `--model-id` sets -- not the `--model` path, which is
/// machine-local and would refuse a correct artifact for having been trained
/// on a box that stored the checkpoint elsewhere.
///
/// A name is weak evidence, and the refusal is deliberately one-directional
/// because of it: two checkpoints can share a name, and fine-tuning changes
/// the representation without changing the name. So a mismatch refuses, and a
/// match is not a warrant of sameness. The width comparison below is the half
/// that is actually checkable, and its passing is exactly what makes the
/// unchecked half look settled.
pub fn check_representation(
    artifact: &RouterArtifact,
    served_model_id: &str,
    dimensions: usize,
) -> Result<crate::forward::cpu_f16::PoolingStrategy, ApiError> {
    let trained_on = &artifact.representation.embedding_model;
    if trained_on != served_model_id {
        return Err(ApiError::BadRequest {
            message: format!(
                "router gate was trained on embedding model {trained_on:?} but this server serves                  {served_model_id:?}; the gate would route on a representation it never saw"
            ),
            code: "router_representation_model_mismatch",
        });
    }

    let recorded = artifact.representation.input_width;
    let measured = dimensions as u64;
    if recorded != measured {
        return Err(ApiError::BadRequest {
            message: format!(
                "router gate takes a {recorded}-dimension context vector but this server's                  embedding model produces {measured} dimensions"
            ),
            code: "router_representation_width_mismatch",
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
        served_model_id: &str,
        dimensions: usize,
    ) -> Result<Self, ApiError> {
        // Ordered from the most general refusal to the most specific, which
        // is the same rule the `/v1/lora` handlers follow. A build with no
        // gate implementation cannot route any artifact, so it answers first;
        // an artifact that disagrees with the gate it ships with is wrong on
        // every server, so it answers next; only then does this server's own
        // embedding model enter it. Reversed, an operator on a build without
        // `mixture` is told to go fix a representation that was never the
        // problem.
        let router = ServingRouter::new(resolved.artifact)?;
        let pooling = check_representation(router.artifact(), served_model_id, dimensions)?;
        Ok(Self {
            router: std::sync::Mutex::new(router),
            pooling,
            pinned: resolved.pinned,
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
            pinned: self.pinned,
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
        context_vector: &[f32],
        k: usize,
    ) -> Result<Vec<LoraSelection>, ApiError> {
        let mut guard = self.router.lock().map_err(|_| ApiError::Internal {
            message: "the adapter router panicked on an earlier request and is no longer serving"
                .to_string(),
        })?;
        guard.route(resident, context_vector, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::lora::AdapterMetadata;

    fn resident(pairs: &[(u32, &str)]) -> AdapterIndex {
        AdapterIndex {
            adapters: pairs
                .iter()
                .map(|(id, name)| AdapterMetadata {
                    id: *id,
                    name: (*name).to_string(),
                    path: format!("/p/{name}.safetensors"),
                    rank: 8,
                    layers: 24,
                })
                .collect(),
            applied: Vec::new(),
        }
    }

    fn representation(input_width: u64) -> crate::router_state::TrainedRepresentation {
        crate::router_state::TrainedRepresentation {
            embedding_model: "gme-qwen35".into(),
            pooling: "mean_visual".into(),
            input_width,
        }
    }

    fn artifact(names: &[&str]) -> RouterArtifact {
        RouterArtifact {
            version: 1,
            adapter_names: names.iter().map(|n| (*n).to_string()).collect(),
            representation: representation(4),
            gate_bytes: vec![0],
        }
    }

    #[cfg(not(feature = "mixture"))]
    fn resolved(
        rep: crate::router_state::TrainedRepresentation,
    ) -> crate::router_state::ResolvedRouter {
        let mut art = artifact(&["technical"]);
        art.representation = rep;
        crate::router_state::ResolvedRouter {
            artifact: art,
            pinned: false,
        }
    }

    /// The pairing `ServingRouter::new` cannot see. It checks the artifact
    /// against its own gate, which says the artifact is self-consistent and
    /// nothing about whether this server can produce the vector it wants.
    #[test]
    fn a_gate_trained_on_another_embedding_model_refuses() {
        let err = check_representation(&artifact(&["technical"]), "some-other-model", 4)
            .expect_err("a different embedding model must refuse");
        let msg = err.message();
        assert!(
            msg.contains("gme-qwen35") && msg.contains("some-other-model"),
            "both identities must be named: {msg}"
        );
    }

    #[test]
    fn a_gate_whose_width_differs_from_this_servers_embedder_refuses() {
        let err = check_representation(&artifact(&["technical"]), "gme-qwen35", 1536)
            .expect_err("a width the embedder cannot produce must refuse");
        let msg = err.message();
        assert!(
            msg.contains('4') && msg.contains("1536"),
            "both widths must be named: {msg}"
        );
    }

    /// The half that is NOT checkable by width, which is why it is carried in
    /// the artifact at all: `MeanVisualTokens` and `LastToken` produce
    /// different vectors of the same length, so an unrecognised value here
    /// cannot be caught later by any comparison of dimensions.
    #[test]
    fn an_unrecognised_pooling_refuses_rather_than_defaulting() {
        let mut art = artifact(&["technical"]);
        art.representation.pooling = "cls".into();
        let err = check_representation(&art, "gme-qwen35", 4)
            .expect_err("an unknown pooling must refuse");
        assert!(
            err.message().contains("cls"),
            "the unrecognised value must be quoted: {}",
            err.message()
        );
    }

    #[test]
    fn a_matching_representation_returns_the_pooling_the_gate_was_trained_under() {
        let pooling = check_representation(&artifact(&["technical"]), "gme-qwen35", 4)
            .expect("a matching representation must resolve");
        assert_eq!(
            pooling,
            crate::forward::cpu_f16::PoolingStrategy::MeanVisualTokens,
            "the routing path must embed the way the gate was trained, not the way a request asks"
        );
    }

    /// The ordering `ServedRouter::new` owes an operator, on the build where
    /// the two answers differ. A representation error here would send them to
    /// re-train a gate on a server that cannot run any gate.
    #[cfg(not(feature = "mixture"))]
    #[test]
    fn a_build_without_a_gate_says_so_before_it_says_anything_about_the_artifact() {
        let mut rep = representation(4);
        rep.embedding_model = "some-other-model".into();
        // `match` rather than `expect_err`: a test's convenience is not a
        // reason to derive `Debug` on a type that holds a gate.
        let msg = match ServedRouter::new(resolved(rep), "gme-qwen35", 4) {
            Ok(_) => panic!("a build with no gate implementation must refuse"),
            Err(err) => err.message().to_string(),
        };
        assert!(
            msg.contains("mixture"),
            "the build is the answer, not the representation: {msg}"
        );
        assert!(
            !msg.contains("some-other-model"),
            "the artifact must not be blamed for a build problem: {msg}"
        );
    }

    /// The first arm ADR-094 decision 2 names. It fails against any
    /// implementation that kept a positional path: the two pools hold the same
    /// adapters under the same names and differ only in the order residency
    /// reports them, so a positional mapping pairs the gate's columns with
    /// different adapters in the two cases while a name-resolved one does not.
    #[test]
    fn two_pools_differing_only_in_order_resolve_to_the_same_column_order() {
        let art = artifact(&["technical", "legal", "medical"]);
        let a = trained_order(
            &art,
            &resident(&[(0, "technical"), (1, "legal"), (2, "medical")]),
        )
        .expect("pool a");
        let b = trained_order(
            &art,
            &resident(&[(2, "medical"), (0, "technical"), (1, "legal")]),
        )
        .expect("pool b reordered");
        assert_eq!(a, b, "residency order must not reach the gate's columns");
        assert_eq!(
            a,
            vec!["technical", "legal", "medical"],
            "the order handed to the gate is the ARTIFACT's, which is its column order"
        );
    }

    /// The second arm. It fails against an implementation that treats an
    /// absent name as a zero column, which is the shape that silently degrades
    /// instead of refusing.
    #[test]
    fn a_pool_missing_one_trained_name_refuses() {
        let art = artifact(&["technical", "legal", "medical"]);
        let err = trained_order(&art, &resident(&[(0, "technical"), (1, "legal")]))
            .expect_err("a missing trained adapter must refuse");
        let msg = err.message();
        assert!(
            msg.contains("medical"),
            "the absent name must be named: {msg}"
        );
        assert!(
            msg.contains("technical, legal, medical") && msg.contains("technical, legal"),
            "both lists must print side by side: {msg}"
        );
    }

    /// The other direction. Refusing only one leaves the other silent: they
    /// are the same unverifiable pairing seen from opposite sides.
    #[test]
    fn a_resident_adapter_absent_from_the_trained_set_refuses() {
        let art = artifact(&["technical", "legal"]);
        let err = trained_order(
            &art,
            &resident(&[(0, "technical"), (1, "legal"), (2, "surprise")]),
        )
        .expect_err("an untrained resident adapter must refuse");
        let msg = err.message();
        assert!(
            msg.contains("surprise"),
            "the extra name must be named: {msg}"
        );
        assert!(
            msg.contains("not in the gate's trained set"),
            "the direction must be stated: {msg}"
        );
    }

    /// Not one of the ADR's enumerated directions, and not reachable by set
    /// difference: two sides can agree AS SETS and still be unresolvable. A
    /// duplicate name makes name-to-id ambiguous, and picking either candidate
    /// is the silent wrong-adapter selection this module exists to prevent.
    #[test]
    fn a_duplicate_resident_name_refuses_even_though_the_sets_agree() {
        let art = artifact(&["technical", "legal"]);
        let pool = resident(&[(0, "technical"), (1, "legal"), (2, "legal")]);
        // The sets agree: every trained name is resident and every resident
        // name is trained. Only the multiplicity differs.
        assert!(missing(&art.adapter_names, &["technical".into(), "legal".into()]).is_empty());
        let err = trained_order(&art, &pool).expect_err("an ambiguous name must refuse");
        assert!(
            err.message().contains("does not identify one adapter"),
            "got {}",
            err.message()
        );
    }

    #[test]
    fn a_duplicate_trained_name_refuses_because_its_columns_cannot_be_told_apart() {
        let art = artifact(&["legal", "legal"]);
        let err = trained_order(&art, &resident(&[(0, "legal")])).expect_err("must refuse");
        assert!(
            err.message().contains("cannot be told apart"),
            "got {}",
            err.message()
        );
    }

    #[test]
    fn selections_carry_the_resident_id_not_the_position() {
        // The pool is deliberately ordered so that position and id disagree
        // for every entry: a to_selections that returned the index would pass
        // against an identity-ordered pool.
        let pool = resident(&[(40, "technical"), (7, "legal")]);
        let got = to_selections(vec![("legal".into(), 0.75)], &pool).expect("resolve");
        assert_eq!(got.len(), 1);
        assert_eq!(
            got[0].id, 7,
            "id must come from residency, not the position"
        );
        assert!((got[0].scale - 0.75).abs() < f32::EPSILON);
    }

    /// Recorded input width against the width the gate actually takes. A
    /// separate refusal from the output-width one because the fault differs:
    /// this says the artifact describes itself wrongly, and re-configuring the
    /// server's embedding model would not help.
    #[cfg(feature = "mixture")]
    #[test]
    fn an_artifact_that_misrecords_its_own_input_width_refuses() {
        use lattice_fann::{Activation, NetworkBuilder};

        let gate_bytes = NetworkBuilder::new()
            .input(4)
            .output(1, Activation::Linear)
            .build()
            .expect("gate must build")
            .to_bytes();

        // Control first: the truthful record constructs.
        assert!(
            ServingRouter::new(RouterArtifact {
                version: 1,
                adapter_names: vec!["technical".into()],
                representation: representation(4),
                gate_bytes: gate_bytes.clone(),
            })
            .is_ok(),
            "a truthful artifact must construct, or the refusal proves nothing"
        );

        match ServingRouter::new(RouterArtifact {
            version: 1,
            adapter_names: vec!["technical".into()],
            representation: representation(1024),
            gate_bytes,
        }) {
            Err(err) => {
                let m = err.message();
                assert!(
                    m.contains("1024") && m.contains('4'),
                    "the refusal must name recorded AND measured, got {m}"
                );
            }
            Ok(_) => panic!("a misrecorded input width must refuse"),
        }
    }

    /// A gate whose output width disagrees with the name list is refused at
    /// construction. Without this the disagreement is invisible: every other
    /// check in this module compares the list against RESIDENCY, never against
    /// the network, and `route` silently uses only the first `available.len()`
    /// columns -- so a wider gate routes fine while the adapters its trailing
    /// columns were trained for can never be selected.
    #[cfg(feature = "mixture")]
    #[test]
    fn an_artifact_whose_gate_is_wider_than_its_name_list_refuses() {
        use lattice_fann::{Activation, NetworkBuilder};

        let gate = |outputs: usize| {
            NetworkBuilder::new()
                .input(4)
                .output(outputs, Activation::Linear)
                .build()
                .expect("gate must build")
                .to_bytes()
        };

        // The must-match control FIRST: an agreeing pair constructs, so a
        // refusal below is about the width and not about the fixture.
        let ok = RouterArtifact {
            version: 1,
            adapter_names: vec!["technical".into(), "legal".into()],
            representation: representation(4),
            gate_bytes: gate(2),
        };
        assert!(
            ServingRouter::new(ok).is_ok(),
            "an agreeing artifact must construct, or the refusal arm proves nothing"
        );

        for (names, outputs) in [(2usize, 3usize), (3, 2)] {
            let artifact = RouterArtifact {
                version: 1,
                adapter_names: (0..names).map(|i| format!("a{i}")).collect(),
                representation: representation(4),
                gate_bytes: gate(outputs),
            };
            match ServingRouter::new(artifact) {
                Err(err) => {
                    let m = err.message();
                    assert!(
                        m.contains(&format!("{names} adapter name"))
                            && m.contains(&format!("{outputs} output")),
                        "the refusal must name both widths, got {m}"
                    );
                }
                Ok(_) => panic!("{names} names against a {outputs}-column gate must refuse"),
            }
        }
    }

    #[cfg(not(feature = "mixture"))]
    #[test]
    fn a_build_without_mixture_refuses_at_construction_rather_than_lacking_the_symbol() {
        // The whole point of the facade: the symbol exists and answers.
        match ServingRouter::new(artifact(&["technical"])) {
            Err(err) => assert!(err.message().contains("mixture"), "got {}", err.message()),
            Ok(_) => panic!("a build without `mixture` must refuse, not construct a router"),
        }
    }
}
