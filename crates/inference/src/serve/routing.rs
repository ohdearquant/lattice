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

    fn artifact(names: &[&str]) -> RouterArtifact {
        RouterArtifact {
            version: 1,
            adapter_names: names.iter().map(|n| (*n).to_string()).collect(),
            gate_bytes: vec![0],
        }
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
