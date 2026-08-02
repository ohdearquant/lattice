#![cfg(feature = "native")]

use lattice_embed::drift::{
    BaselineFixture, ModelDriftOutcome, compare_embeddings, load_baselines,
};

#[test]
fn baselines_are_loaded_in_filename_order() {
    let dir = tempfile::tempdir().unwrap();
    let second = BaselineFixture {
        model: "second".to_string(),
        texts: vec!["text".to_string()],
        embeddings: vec![vec![1.0, 0.0]],
    };
    let first = BaselineFixture {
        model: "first".to_string(),
        texts: vec!["text".to_string()],
        embeddings: vec![vec![1.0, 0.0]],
    };

    std::fs::write(
        dir.path().join("z_second.json"),
        serde_json::to_vec(&second).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("a_first.json"),
        serde_json::to_vec(&first).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.path().join("ignored.txt"), b"not JSON").unwrap();

    let loaded = load_baselines(dir.path()).unwrap();
    let models = loaded
        .iter()
        .map(|fixture| fixture.model.as_str())
        .collect::<Vec<_>>();
    assert_eq!(models, ["first", "second"]);
}

#[test]
fn comparison_reports_the_worst_vector() {
    let fixture = BaselineFixture {
        model: "test-model".to_string(),
        texts: vec!["stable".to_string(), "moved".to_string()],
        embeddings: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
    };
    let current = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let outcome = compare_embeddings(&fixture, &current).unwrap();
    assert_eq!(
        outcome,
        ModelDriftOutcome::Checked {
            max_one_minus_cos: 1.0,
            worst_index: 1,
        }
    );
}

#[test]
fn comparison_rejects_fixture_count_mismatch() {
    let fixture = BaselineFixture {
        model: "test-model".to_string(),
        texts: vec!["only text".to_string()],
        embeddings: Vec::new(),
    };

    let error = compare_embeddings(&fixture, &[]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("texts and embeddings counts differ")
    );
}
