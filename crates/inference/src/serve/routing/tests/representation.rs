use super::*;
use crate::serving_embeddings::{EmbeddingRepresentation, PooledEmbeddingService};

fn qwen_representation(dimensions: usize) -> EmbeddingRepresentation {
    EmbeddingRepresentation {
        loader_format: crate::serve::embeddings::QWEN35_F16_DECODER_LOADER,
        dimensions,
    }
}

/// Router artifacts already on disk record this exact string. The pooled
/// service now reports its own loader instead of the router reading a
/// constant, so a renamed value would refuse every artifact written before
/// the rename, with nothing else in the representation able to show why.
#[test]
fn the_pooled_service_reports_the_loader_recorded_router_artifacts_carry() {
    let model = crate::serve::embeddings::test_support::tiny_embedding_model();
    let representation = model.representation();
    assert_eq!(
        representation.loader_format.as_bytes(),
        b"qwen35-f16-decoder",
        "the pooled loader identifier is part of every recorded router artifact"
    );
    assert_eq!(
        representation.loader_format,
        crate::serve::embeddings::QWEN35_F16_DECODER_LOADER
    );
    assert_eq!(representation.dimensions, model.dimensions());
}

/// An artifact written before the service existed carries the old constant
/// in its representation, and it must still start.
#[test]
fn an_artifact_recorded_before_the_service_admits_through_its_representation() {
    let pooling = check_service_representation(
        &artifact(&["technical"]),
        &identity("gme-qwen35"),
        &qwen_representation(4),
    )
    .expect("an artifact recorded through the same loader must be accepted");
    assert_eq!(
        pooling,
        crate::forward::cpu_f16::PoolingStrategy::MeanVisualTokens
    );
}

/// The descriptor-aware check replaces the constant-reading one on the
/// startup path. Every refusal the old check gives, it must give with the
/// same code and the same words, or an operator meets a different startup
/// error for the same artifact.
#[test]
fn the_service_check_answers_every_case_exactly_as_the_constant_check_did() {
    let cases: Vec<(&str, RouterArtifact, &str, usize)> = vec![
        ("matching", artifact(&["technical"]), "gme-qwen35", 4),
        (
            "other model",
            artifact(&["technical"]),
            "some-other-model",
            4,
        ),
        ("other width", artifact(&["technical"]), "gme-qwen35", 1536),
        (
            "other loader",
            {
                let mut art = artifact(&["technical"]);
                art.representation.loader_format = "some-other-loader".into();
                art
            },
            "gme-qwen35",
            4,
        ),
        (
            "unknown pooling",
            {
                let mut art = artifact(&["technical"]);
                art.representation.pooling = "cls".into();
                art
            },
            "gme-qwen35",
            4,
        ),
        (
            "unknown prompt source",
            {
                let mut art = artifact(&["technical"]);
                art.representation.prompt_source = "whole_conversation".into();
                art
            },
            "gme-qwen35",
            4,
        ),
    ];
    let mut refusals = 0;
    for (name, art, embedder, dimensions) in &cases {
        let old = check_representation(art, &identity(embedder), *dimensions);
        let new = check_service_representation(
            art,
            &identity(embedder),
            &qwen_representation(*dimensions),
        );
        match (old, new) {
            (Ok(old), Ok(new)) => assert_eq!(old, new, "{name}: pooling differs"),
            (Err(old), Err(new)) => {
                refusals += 1;
                assert_eq!(old.code(), new.code(), "{name}: code differs");
                assert_eq!(old.message(), new.message(), "{name}: message differs");
            }
            (old, new) => panic!(
                "{name}: one check accepted and the other refused (old ok: {}, new ok: {})",
                old.is_ok(),
                new.is_ok()
            ),
        }
    }
    assert_eq!(
        refusals,
        cases.len() - 1,
        "every case except the matching one must refuse, or the comparison compared nothing"
    );
}

/// The loader now comes from the service, so a service reading the bytes
/// another way refuses an artifact recorded through this one, and the
/// refusal names both.
#[test]
fn a_service_reading_through_another_loader_refuses_a_qwen_artifact() {
    let representation = EmbeddingRepresentation {
        loader_format: crate::serve::embeddings::BERT_ENCODER_LOADER,
        dimensions: 4,
    };
    let err = check_service_representation(
        &artifact(&["technical"]),
        &identity("gme-qwen35"),
        &representation,
    )
    .expect_err("a service with a different loader must refuse");
    assert_eq!(err.code(), "router_representation_loader_mismatch");
    assert!(
        err.message()
            .contains(crate::serve::embeddings::BERT_ENCODER_LOADER)
            && err
                .message()
                .contains(crate::serve::embeddings::QWEN35_F16_DECODER_LOADER),
        "the refusal names both loaders: {}",
        err.message()
    );
}

#[test]
fn a_service_of_another_width_refuses_and_names_both_widths() {
    let err = check_service_representation(
        &artifact(&["technical"]),
        &identity("gme-qwen35"),
        &qwen_representation(1536),
    )
    .expect_err("a width the service cannot produce must refuse");
    assert_eq!(err.code(), "router_representation_width_mismatch");
    assert!(
        err.message().contains('4') && err.message().contains("1536"),
        "both widths must be named: {}",
        err.message()
    );
}

/// Same ordering the existing constructor owes an operator: a build with no
/// gate answers before anything about the artifact or the service.
#[cfg(not(feature = "mixture"))]
#[test]
fn a_build_without_a_gate_says_so_before_the_service_representation_is_read() {
    let mut rep = representation(4);
    rep.embedding_model = "some-other-model".into();
    let msg = match ServedRouter::new_with_representation(
        resolved(rep),
        &identity("gme-qwen35"),
        &qwen_representation(4),
        false,
    ) {
        Ok(_) => panic!("a build with no gate implementation must refuse"),
        Err(err) => err.message().to_string(),
    };
    assert!(msg.contains("mixture"), "the build is the answer: {msg}");
    assert!(
        !msg.contains("some-other-model"),
        "the artifact must not be blamed for a build problem: {msg}"
    );
}

/// The startup refusal through the new constructor, with its must-start
/// control first: the constructor refuses for several reasons before the
/// loader is read.
#[cfg(feature = "mixture")]
#[test]
fn a_service_through_another_loader_refuses_at_startup() {
    use lattice_fann::{Activation, NetworkBuilder};

    let resolved = || crate::router_state::ResolvedRouter {
        artifact: RouterArtifact {
            version: 1,
            adapter_names: vec!["technical".into()],
            representation: representation(4),
            gate_bytes: NetworkBuilder::new()
                .input(4)
                .output(1, Activation::Linear)
                .build()
                .expect("gate must build")
                .to_bytes(),
            provenance: crate::router_state::RouterProvenance::Legacy,
        },
        pinned: false,
    };

    assert!(
        ServedRouter::new_with_representation(
            resolved(),
            &identity("gme-qwen35"),
            &qwen_representation(4),
            false
        )
        .is_ok(),
        "an artifact recorded through the service's own loader must start"
    );

    let other = EmbeddingRepresentation {
        loader_format: crate::serve::embeddings::BERT_ENCODER_LOADER,
        dimensions: 4,
    };
    let Err(err) =
        ServedRouter::new_with_representation(resolved(), &identity("gme-qwen35"), &other, false)
    else {
        panic!("a service reading through another loader must refuse at startup")
    };
    assert_eq!(err.code(), "router_representation_loader_mismatch");
}
