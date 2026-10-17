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
        ..Default::default()
    }
}

fn identity(name: &str) -> EmbedderIdentity {
    EmbedderIdentity::resolve(std::path::Path::new("/models"), Some(name))
}

/// The identity is the EMBEDDER's, and the derivation is here rather than
/// in `main.rs` so that this test can exist at all.
#[test]
fn an_embedder_identity_is_the_directory_basename_unless_overridden() {
    use std::path::Path;
    assert_eq!(
        EmbedderIdentity::resolve(Path::new("/models/qwen3.5-0.8b"), None).as_str(),
        "qwen3.5-0.8b"
    );
    // A trailing slash names the same directory and must not change what
    // the gate is checked against.
    assert_eq!(
        EmbedderIdentity::resolve(Path::new("/models/qwen3.5-0.8b/"), None).as_str(),
        "qwen3.5-0.8b"
    );
    // The override wins, mirroring --model-id over the --model basename.
    assert_eq!(
        EmbedderIdentity::resolve(Path::new("/models/qwen3.5-0.8b"), Some("gme-qwen35")).as_str(),
        "gme-qwen35"
    );
    // A path that names no directory falls back rather than panicking, and
    // the fallback is a value no artifact will match, so startup lands on
    // the refusal instead of routing on a guess.
    assert_eq!(
        EmbedderIdentity::resolve(Path::new(".."), None).as_str(),
        "embedder"
    );
}

/// The loader is the half the identity cannot carry. Same name, same
/// width, same pooling, same text -- and a different reader of the bytes,
/// which no other field in the representation can express.
#[test]
fn a_gate_trained_through_another_embedding_loader_refuses() {
    let mut art = artifact(&["technical"]);
    art.representation.loader_format = "some-other-loader".into();
    let err = check_representation(&art, &identity("gme-qwen35"), 4)
        .expect_err("a loader the server does not use must refuse");
    assert_eq!(err.code(), "router_representation_loader_mismatch");
    assert!(
        err.message().contains("some-other-loader")
            && err
                .message()
                .contains(crate::serve::embeddings::QWEN35_F16_DECODER_LOADER),
        "the refusal names BOTH loaders, since which one moved is the question asked: {}",
        err.message()
    );
}

/// The agreeing case, so the arm above is not passing on a broken
/// comparison that refuses everything.
#[test]
fn a_gate_trained_through_this_servers_loader_is_accepted() {
    let art = artifact(&["technical"]);
    assert_eq!(
        art.representation.loader_format,
        crate::serve::embeddings::QWEN35_F16_DECODER_LOADER,
        "the fixture must carry the value the server actually uses, or the arm above proves nothing"
    );
    check_representation(&art, &identity("gme-qwen35"), 4)
        .expect("an artifact matching this server's loader must be accepted");
}

/// Condition (1). The cross-binary arm below compares two constants, so it
/// is worth nothing unless they hold different values: an arm that compares
/// a value against itself passes for every input and reads exactly like a
/// guard.
#[test]
fn the_two_loader_constants_are_different_values() {
    assert_ne!(
        crate::serve::embeddings::QWEN35_F16_DECODER_LOADER,
        crate::serve::embeddings::BERT_ENCODER_LOADER,
        "two binaries that read the bytes differently must not share one loader name"
    );
}

/// Condition (3). The refusal an operator actually meets: a gate trained
/// through `lattice_serve`'s BERT encoder, handed to the Qwen3.5 server at
/// startup.
///
/// Gated on `mixture` deliberately. Without that feature `ServingRouter::new`
/// answers first with the build refusal, which is the right ordering and the
/// wrong reason for this arm: the test would pass while proving nothing
/// about the loader.
#[cfg(feature = "mixture")]
#[test]
fn a_gate_trained_through_the_other_binarys_loader_refuses_at_startup() {
    use lattice_fann::{Activation, NetworkBuilder};

    let gate = || {
        NetworkBuilder::new()
            .input(4)
            .output(1, Activation::Linear)
            .build()
            .expect("gate must build")
            .to_bytes()
    };
    let resolved = |rep| crate::router_state::ResolvedRouter {
        artifact: RouterArtifact {
            version: 1,
            adapter_names: vec!["technical".into()],
            representation: rep,
            gate_bytes: gate(),
            provenance: crate::router_state::RouterProvenance::Legacy,
        },
        pinned: false,
    };

    // The must-match control FIRST, and it is load-bearing here: this
    // constructor refuses for several reasons before the loader is read, so
    // without it a green refusal arm could be a gate the constructor was
    // always going to reject.
    assert!(
        ServedRouter::new(
            resolved(representation(4)),
            &identity("gme-qwen35"),
            4,
            false
        )
        .is_ok(),
        "an artifact carrying this binary's own loader must start, or the arm below proves nothing"
    );

    let mut rep = representation(4);
    rep.loader_format = crate::serve::embeddings::BERT_ENCODER_LOADER.into();
    let Err(err) = ServedRouter::new(resolved(rep), &identity("gme-qwen35"), 4, false) else {
        panic!("a gate trained through the other binary's loader must refuse at startup")
    };
    assert_eq!(err.code(), "router_representation_loader_mismatch");
    assert!(
        err.message()
            .contains(crate::serve::embeddings::BERT_ENCODER_LOADER)
            && err
                .message()
                .contains(crate::serve::embeddings::QWEN35_F16_DECODER_LOADER),
        "the refusal names BOTH loaders, the artifact's and this binary's: {}",
        err.message()
    );
}

/// The epoch travels with the decision. Read afterwards instead, a reload
/// landing in between would date the episode to a policy that did not
/// produce it, and an off-epoch sample is discarded rather than refused --
/// so nothing would ever report the mislabelling.
///
/// Reaches the private router to make a policy change, because no serving
/// endpoint changes a running gate yet; when one lands it replaces this
/// lever rather than adding to it.
#[cfg(feature = "mixture")]
#[test]
fn a_routed_request_carries_the_epoch_it_was_decided_under() {
    use lattice_fann::{Activation, NetworkBuilder};

    let gate_bytes = NetworkBuilder::new()
        .input(4)
        .output(1, Activation::Linear)
        .build()
        .expect("gate must build")
        .to_bytes();
    let mut router = ServingRouter::new(
        RouterArtifact {
            version: 1,
            adapter_names: vec!["technical".into()],
            representation: representation(4),
            gate_bytes,
            provenance: crate::router_state::RouterProvenance::Legacy,
        },
        false,
    )
    .expect("a truthful artifact must construct");

    let pool = resident(&[(0, "technical")]);
    let before = router
        .route(&pool, vec![0.0; 4], 1, "chatcmpl-epoch-0")
        .expect("the gate must route its own adapter");
    router.router.set_epsilon(0.25);
    let after = router
        .route(&pool, vec![0.0; 4], 1, "chatcmpl-epoch-1")
        .expect("the gate must still route");

    assert_eq!(
        before.policy_epoch,
        crate::mixture::INITIAL_POLICY_EPOCH,
        "a router nobody has reconfigured serves the initial epoch"
    );
    assert_eq!(
        after.policy_epoch,
        before.policy_epoch + 1,
        "a decision made after a policy change belongs to the new epoch"
    );
}

fn representation(input_width: u64) -> crate::router_state::TrainedRepresentation {
    crate::router_state::TrainedRepresentation {
        embedding_model: "gme-qwen35".into(),
        pooling: "mean_visual".into(),
        prompt_source: crate::serve::routing::PromptSource::SERVED.as_str().into(),
        loader_format: crate::serve::embeddings::QWEN35_F16_DECODER_LOADER.into(),
        input_width,
    }
}

/// A gate with `outputs` linear columns of the right width, so a test can
/// build an artifact that constructs.
///
/// Zero weights and fixed, strictly descending biases, so the gate scores
/// the same values on every run and for every input.
/// `NetworkBuilder::build` seeds its weights randomly, which would make
/// whether a draw reorders the selection a property of the run rather than
/// of the seed: an assertion about reordering would then be flaky, and a
/// passing first run is exactly how that ships. Distinct biases also keep
/// the selected order a real ordering instead of a tie resolved by column
/// index.
#[cfg(feature = "mixture")]
fn servable_gate(outputs: usize) -> Vec<u8> {
    use lattice_fann::{Activation, Layer, Network};
    let mut layer =
        Layer::zeros(4, outputs, Activation::Linear).expect("the gate layer must build");
    for (column, bias) in layer.biases_mut().iter_mut().enumerate() {
        *bias = 0.1 * (outputs - column) as f32;
    }
    Network::new(vec![layer])
        .expect("gate must build")
        .to_bytes()
}

/// A learned policy: a softmax at a real temperature, carrying a scale and
/// a non-zero pair of guard fields so the install can be seen to happen.
#[cfg(feature = "mixture")]
fn learned_policy() -> crate::router_state::RouterPolicy {
    crate::router_state::RouterPolicy {
        kind: "softmax".into(),
        tau: Some(0.7),
        epsilon: 0.05,
        tau_floor: 0.1,
        entropy_floor: 0.35,
        max_consecutive_collapsed: 3,
        training_state: "learned".into(),
        objective_version: Some("whole-mixture-v1".into()),
        exploration_scale: Some(0.5),
    }
}

/// A uniform (bootstrap) policy. `epsilon` and the other guard fields are
/// recorded non-zero deliberately: the arm that says they are NOT installed
/// needs them to differ from the router's defaults.
#[cfg(feature = "mixture")]
fn uniform_policy() -> crate::router_state::RouterPolicy {
    crate::router_state::RouterPolicy {
        kind: "uniform".into(),
        tau: None,
        epsilon: 0.25,
        tau_floor: 0.5,
        entropy_floor: 0.75,
        max_consecutive_collapsed: 9,
        training_state: "bootstrap".into(),
        objective_version: None,
        exploration_scale: None,
    }
}

#[cfg(feature = "mixture")]
fn recorded_artifact(policy: crate::router_state::RouterPolicy) -> RouterArtifact {
    recorded_artifact_over(policy, 1)
}

#[cfg(feature = "mixture")]
fn recorded_artifact_over(
    policy: crate::router_state::RouterPolicy,
    outputs: usize,
) -> RouterArtifact {
    RouterArtifact {
        version: 1,
        adapter_names: (0..outputs).map(|i| format!("a{i}")).collect(),
        representation: representation(4),
        gate_bytes: servable_gate(outputs),
        provenance: crate::router_state::RouterProvenance::recorded(
            policy,
            crate::router_state::RouterLineage {
                schema_id: "schema-1".into(),
                base_model: "test-model".into(),
                parent: None,
                admission_record: None,
            },
        ),
    }
}

fn artifact(names: &[&str]) -> RouterArtifact {
    RouterArtifact {
        version: 1,
        adapter_names: names.iter().map(|n| (*n).to_string()).collect(),
        representation: representation(4),
        gate_bytes: vec![0],
        provenance: crate::router_state::RouterProvenance::Legacy,
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
    let err = check_representation(&artifact(&["technical"]), &identity("some-other-model"), 4)
        .expect_err("a different embedding model must refuse");
    let msg = err.message();
    assert!(
        msg.contains("gme-qwen35") && msg.contains("some-other-model"),
        "both identities must be named: {msg}"
    );
}

#[test]
fn a_gate_whose_width_differs_from_this_servers_embedder_refuses() {
    let err = check_representation(&artifact(&["technical"]), &identity("gme-qwen35"), 1536)
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
    let err = check_representation(&art, &identity("gme-qwen35"), 4)
        .expect_err("an unknown pooling must refuse");
    assert!(
        err.message().contains("cls"),
        "the unrecognised value must be quoted: {}",
        err.message()
    );
}

/// The third member of the representation, and the one a width check can
/// never stand in for: the same model and the same pooling over different
/// text produce a correctly-shaped vector of the wrong content.
#[test]
fn the_context_text_is_the_last_user_turn_not_the_conversation() {
    let turns = [
        (false, "you are helpful"),
        (true, "first question"),
        (false, "an answer"),
        (true, "the real question"),
    ];
    assert_eq!(
        context_text(PromptSource::SERVED, turns.into_iter()),
        Some("the real question"),
        "routing on the whole conversation makes the vector drift with history length, so the \
             same question routes differently depending on how long the chat has been running"
    );
}

#[test]
fn a_conversation_with_no_user_turn_yields_no_context_text() {
    assert_eq!(
        context_text(
            PromptSource::SERVED,
            [(false, "just a system prompt")].into_iter()
        ),
        None,
        "substituting another role's text would route on something the gate never saw"
    );
}

#[test]
fn an_unrecognised_prompt_source_refuses_rather_than_defaulting() {
    let mut art = artifact(&["technical"]);
    art.representation.prompt_source = "rendered_conversation".into();
    let err = check_representation(&art, &identity("gme-qwen35"), 4)
        .expect_err("an unknown prompt source must refuse");
    assert!(
        err.message().contains("rendered_conversation"),
        "the unrecognised value must be quoted: {}",
        err.message()
    );
}

#[test]
fn a_matching_representation_returns_the_pooling_the_gate_was_trained_under() {
    let pooling = check_representation(&artifact(&["technical"]), &identity("gme-qwen35"), 4)
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
    let msg = match ServedRouter::new(resolved(rep), &identity("gme-qwen35"), 4, false) {
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

/// The surface and the refusal must answer the same question. These pin
/// the predicate BOTH of them read, so a report claiming routable over a
/// set that routing declines is a failing test rather than a 400 nobody
/// predicted.
#[test]
fn a_matching_set_is_routable() {
    let state = routability(
        &artifact(&["technical", "legal"]),
        &resident(&[(1, "legal"), (0, "technical")]),
    );
    assert!(
        state.routable(),
        "order is resolved by name, so it is not a reason to refuse"
    );
    assert_eq!(state, Routability::default());
}

#[test]
fn routability_names_what_is_missing_and_what_is_unexpected() {
    let state = routability(
        &artifact(&["technical", "legal"]),
        &resident(&[(0, "technical"), (2, "medical")]),
    );
    assert!(!state.routable());
    assert_eq!(state.missing, vec!["legal".to_string()]);
    assert_eq!(state.unexpected, vec!["medical".to_string()]);
}

/// Issue #1735: a resident set matching the trained names exactly, with
/// no duplicate on either side, still refuses when the FULL resident set
/// cannot be blended -- the case `duplicate_*`/`missing`/`unexpected`
/// alone cannot see, because a routed request blends every resident
/// adapter, not just the ones a name comparison flags.
#[test]
fn a_blend_refusal_is_not_routable_even_though_the_names_agree() {
    let mut resident = resident(&[(0, "technical"), (1, "legal")]);
    resident.blend_refusal = Some("summed rank exceeds MAX_BLEND_RANK_TOTAL".to_string());
    let state = routability(&artifact(&["technical", "legal"]), &resident);
    assert!(
        state.missing.is_empty() && state.unexpected.is_empty(),
        "the names agree; only the blend budget refuses"
    );
    assert!(!state.routable(), "a blend refusal must not be routable");
    assert_eq!(
        state.blend_refusal.as_deref(),
        Some("summed rank exceeds MAX_BLEND_RANK_TOTAL")
    );

    resident.blend_refusal = None;
    let state = routability(&artifact(&["technical", "legal"]), &resident);
    assert!(
        state.routable(),
        "with no blend refusal, the names still agree"
    );
}

/// Reported separately from the set differences, because routing refuses
/// on it while the two sides agree AS SETS -- so a surface built from the
/// differences alone would print routable over a set that declines.
#[test]
fn a_duplicate_on_either_side_is_not_routable_even_though_the_sets_agree() {
    let dup_resident = routability(
        &artifact(&["technical"]),
        &resident(&[(0, "technical"), (1, "technical")]),
    );
    assert!(dup_resident.missing.is_empty() && dup_resident.unexpected.is_empty());
    assert!(
        !dup_resident.routable(),
        "the sets agree and it still refuses"
    );
    assert_eq!(
        dup_resident.duplicate_resident.as_deref(),
        Some("technical")
    );

    let dup_trained = routability(
        &artifact(&["technical", "technical"]),
        &resident(&[(0, "technical")]),
    );
    assert!(!dup_trained.routable());
    assert_eq!(dup_trained.duplicate_trained.as_deref(), Some("technical"));
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
        ServingRouter::new(
            RouterArtifact {
                version: 1,
                adapter_names: vec!["technical".into()],
                representation: representation(4),
                gate_bytes: gate_bytes.clone(),
                provenance: crate::router_state::RouterProvenance::Legacy,
            },
            false
        )
        .is_ok(),
        "a truthful artifact must construct, or the refusal proves nothing"
    );

    match ServingRouter::new(
        RouterArtifact {
            version: 1,
            adapter_names: vec!["technical".into()],
            representation: representation(1024),
            gate_bytes,
            provenance: crate::router_state::RouterProvenance::Legacy,
        },
        false,
    ) {
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
        provenance: crate::router_state::RouterProvenance::Legacy,
    };
    assert!(
        ServingRouter::new(ok, false).is_ok(),
        "an agreeing artifact must construct, or the refusal arm proves nothing"
    );

    for (names, outputs) in [(2usize, 3usize), (3, 2)] {
        let artifact = RouterArtifact {
            version: 1,
            adapter_names: (0..names).map(|i| format!("a{i}")).collect(),
            representation: representation(4),
            gate_bytes: gate(outputs),
            provenance: crate::router_state::RouterProvenance::Legacy,
        };
        match ServingRouter::new(artifact, false) {
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
    match ServingRouter::new(artifact(&["technical"]), false) {
        Err(err) => assert!(err.message().contains("mixture"), "got {}", err.message()),
        Ok(_) => panic!("a build without `mixture` must refuse, not construct a router"),
    }
}

/// A learned policy is not served quietly as uniform. Serving it uniform
/// would make one version label name two behaviours, and an operator would
/// read a learned version number off a server that never ran it.
#[cfg(feature = "mixture")]
#[test]
fn a_learned_artifact_is_refused_unless_learning_is_enabled() {
    match ServingRouter::new(recorded_artifact(learned_policy()), false) {
        Err(err) => assert_eq!(err.code(), "router_learning_not_enabled"),
        Ok(_) => panic!("a learned policy must not be served without the flag"),
    }
    // The control, and it is the whole test: the same artifact constructs
    // once the flag is given, so the refusal is about the flag and not
    // about the artifact.
    let enabled = ServingRouter::new(recorded_artifact(learned_policy()), true)
        .expect("the same artifact must construct with learning enabled");
    assert!(enabled.applied_policy().learning_enabled);
}

/// The flag asks for a learned policy. Over an artifact that records none,
/// reporting learning as enabled would be true of the flag and false of the
/// behaviour.
#[cfg(feature = "mixture")]
#[test]
fn enabling_learning_needs_something_learned_to_serve() {
    match ServingRouter::new(recorded_artifact(uniform_policy()), true) {
        Err(err) => assert_eq!(err.code(), "router_policy_not_learned"),
        Ok(_) => panic!("a uniform policy has nothing learned to enable"),
    }
    // A legacy artifact records no policy at all, which is a DIFFERENT
    // refusal from recording a uniform one: nothing was written down,
    // rather than uniform having been written down.
    let mut legacy = recorded_artifact(uniform_policy());
    legacy.provenance = crate::router_state::RouterProvenance::Legacy;
    match ServingRouter::new(legacy, true) {
        Err(err) => assert_eq!(err.code(), "router_no_recorded_policy"),
        Ok(_) => panic!("an artifact with no policy has nothing to enable"),
    }
    // Control: both of those artifacts serve with the flag off.
    assert!(ServingRouter::new(recorded_artifact(uniform_policy()), false).is_ok());
    let mut legacy_off = recorded_artifact(uniform_policy());
    legacy_off.provenance = crate::router_state::RouterProvenance::Legacy;
    assert!(ServingRouter::new(legacy_off, false).is_ok());
}

/// What the endpoint reports is what the router is running, field by field,
/// and the recorded values are the ones that arrive.
#[cfg(feature = "mixture")]
#[test]
fn an_enabled_router_serves_and_reports_the_recorded_policy() {
    let enabled = ServingRouter::new(recorded_artifact(learned_policy()), true)
        .expect("a learned artifact must construct with learning enabled");
    let applied = enabled.applied_policy();
    assert_eq!(applied.kind, "softmax");
    // Compared with a tolerance, not for equality: the router runs these at
    // `f32` and the report widens them back, so the last bits of a recorded
    // `f64` do not survive. The report is right and the exact comparison
    // would be the wrong question.
    assert!(
        applied.tau.is_some_and(|tau| (tau - 0.7).abs() < 1e-6),
        "{applied:?}"
    );
    assert_eq!(
        applied.exploration_scale,
        Some(0.5),
        "the scale is carried, not narrowed: nothing installs it on the router"
    );
    assert!((applied.epsilon - 0.05).abs() < 1e-6, "{applied:?}");
    assert!(applied.learning_enabled);
}

/// A uniform artifact's recorded guard fields are NOT installed, and this
/// is the arm that says so rather than a comment claiming it. A bootstrap
/// records every guard as zero because the hashed policy shape must not
/// depend on how the artifact was produced; installing those would put a
/// zeroed `tau` floor on a server that never asked for one. This fixture
/// records them NON-zero precisely so the arm can tell.
#[cfg(feature = "mixture")]
#[test]
fn a_uniform_artifacts_recorded_policy_is_not_installed() {
    let served = ServingRouter::new(recorded_artifact(uniform_policy()), false)
        .expect("a uniform artifact must construct");
    let applied = served.applied_policy();
    assert_eq!(applied.kind, "uniform");
    assert_eq!(applied.tau, None);
    assert_eq!(applied.exploration_scale, None);
    assert!(!applied.learning_enabled);
    assert_eq!(
        applied.epsilon, 0.0,
        "the artifact records epsilon 0.25; the router must still be at its own default"
    );
}

/// The draw: with learning enabled and a scale recorded, the routing call
/// draws, seeds from the completion id, records the outcome, and does NOT
/// overwrite the gate's own scores. With learning off the same gate serves
/// the same scores with no draw recorded.
#[cfg(feature = "mixture")]
#[test]
fn an_explored_decision_records_its_draw_and_keeps_the_gate_scores() {
    // ONE artifact, cloned. `NetworkBuilder` initialises randomly, so two
    // calls to the fixture produce two different gates, and any comparison
    // across them would be a comparison of two gates rather than of two
    // decisions.
    let artifact = recorded_artifact_over(learned_policy(), 2);
    let pool = resident(&[(0, "a0"), (1, "a1")]);
    let context = vec![0.25; 4];

    let mut router = ServingRouter::new(artifact.clone(), true)
        .expect("a learned artifact must construct with learning enabled");
    let routed = router
        .route(&pool, context.clone(), 2, "chatcmpl-1789991322-0")
        .expect("the gate must route");
    let draw = routed
        .exploration
        .as_ref()
        .expect("an enabled policy with a recorded scale must draw");

    assert_eq!(
        draw.seed,
        crate::mixture::seed_from_completion_id("chatcmpl-1789991322-0"),
        "the seed is the completion id's, so the episode replays from itself"
    );
    assert_eq!(
        f64::from(draw.scale),
        learned_policy()
            .exploration_scale
            .expect("the fixture records one"),
        "the scale served is the scale recorded"
    );
    assert!(
        draw.log_density.is_finite() && draw.log_density <= 0.0,
        "a log density must be finite and non-positive, got {}",
        draw.log_density
    );
    assert_ne!(
        draw.sampled_logits[..2].to_vec(),
        routed
            .selected_scores
            .iter()
            .map(|(_, score)| *score)
            .collect::<Vec<f32>>(),
        "a scale of 0.5 must actually perturb, or the comparison below is vacuous"
    );

    // The same gate with learning off: no draw, and the SAME score values.
    let mut deterministic = ServingRouter::new(
        RouterArtifact {
            provenance: crate::router_state::RouterProvenance::Legacy,
            ..artifact.clone()
        },
        false,
    )
    .expect("a legacy artifact serves the uniform default");
    let baseline = deterministic
        .route(&pool, context.clone(), 2, "chatcmpl-1789991322-0")
        .expect("the gate must route");
    assert!(
        baseline.exploration.is_none(),
        "a router with learning off records no draw"
    );
    // Compared as a MAPPING, not as a list: the VALUES are the gate's own
    // either way, but the ORDER is the drawn order under exploration and
    // the score order without it. Selection order is part of the action, so
    // it is right that the draw moves it, and a reader who takes
    // `selected_scores[0]` for "the gate's favourite" is wrong whenever a
    // draw happened.
    let as_map = |v: &[(String, f32)]| -> std::collections::BTreeMap<String, String> {
        v.iter()
            .map(|(name, score)| (name.clone(), format!("{score:.7}")))
            .collect()
    };
    assert_eq!(
        as_map(&baseline.selected_scores),
        as_map(&routed.selected_scores),
        "the draw must not overwrite the gate's own score values"
    );
    // Non-vacuousness, established over a sweep rather than on the pinned
    // id. Whether ONE seed happens to reorder a score gap is a fact about
    // that seed; the claim being made is that the draw CAN reorder, and a
    // single-id assertion of it is a coin flip written as a test.
    let mut reorderings = 0;
    for i in 0..32 {
        let mut probe = ServingRouter::new(artifact.clone(), true).expect("construct");
        let out = probe
            .route(&pool, vec![0.25; 4], 2, &format!("chatcmpl-sweep-{i}"))
            .expect("route");
        if out.selected_scores != baseline.selected_scores {
            reorderings += 1;
        }
    }
    assert!(
        reorderings > 0,
        "no draw in 32 reordered a 0.1 score gap at scale 0.5; the mapping \
             comparison above is then vacuous"
    );

    // Reproducibility: the same id replays, a different id does not.
    let mut again = ServingRouter::new(artifact.clone(), true).expect("construct");
    let replay = again
        .route(&pool, context.clone(), 2, "chatcmpl-1789991322-0")
        .expect("route");
    assert_eq!(replay.exploration, routed.exploration);
    let other = again
        .route(&pool, context, 2, "chatcmpl-1789991322-1")
        .expect("route");
    assert_ne!(
        other.exploration.as_ref().map(|e| e.seed),
        routed.exploration.as_ref().map(|e| e.seed),
        "a different completion draws differently"
    );
}
