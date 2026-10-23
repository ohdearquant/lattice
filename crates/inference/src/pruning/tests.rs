use super::*;

const TOL: f32 = 1e-5;

fn assert_close(a: f32, b: f32, msg: &str) {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1e-8);
    assert!(
        diff / scale < TOL,
        "{msg}: got {a}, expected {b}, diff={diff}"
    );
}

// ── cosine_similarity ─────────────────────────────────────────────────────

#[test]
fn test_cosine_identical() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    assert_close(cosine_similarity(&v, &v).unwrap(), 1.0, "identical vectors");
}

#[test]
fn test_cosine_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert_close(
        cosine_similarity(&a, &b).unwrap(),
        0.0,
        "orthogonal vectors",
    );
}

#[test]
fn test_cosine_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    assert_close(cosine_similarity(&a, &b).unwrap(), -1.0, "opposite vectors");
}

#[test]
fn test_cosine_known_angle() {
    // 45-degree angle: cos(π/4) = √2/2 ≈ 0.7071
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 1.0];
    let expected = 1.0 / 2.0_f32.sqrt();
    assert_close(
        cosine_similarity(&a, &b).unwrap(),
        expected,
        "45-degree angle",
    );
}

#[test]
fn test_cosine_zero_vector_returns_none() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 2.0, 3.0];
    assert_eq!(cosine_similarity(&a, &b), None, "zero input vector → None");
}

#[test]
fn test_cosine_both_zero_returns_none() {
    let z = vec![0.0; 4];
    assert_eq!(cosine_similarity(&z, &z), None, "both zero → None");
}

#[test]
fn test_cosine_high_dim() {
    // All-ones vectors in high dim should give cos = 1.0.
    let v = vec![1.0_f32; 896];
    assert_close(cosine_similarity(&v, &v).unwrap(), 1.0, "896-dim identical");
}

#[test]
fn test_cosine_numerical_stability() {
    // Near-identical large vectors should give cos ≈ 1.0, not > 1.0.
    let a = vec![1e6_f32; 1024];
    let mut b = a.clone();
    b[0] += 1e-2;
    let c = cosine_similarity(&a, &b).unwrap();
    assert!(c <= 1.0, "cosine should be clamped to ≤ 1.0, got {c}");
    assert!(
        c > 0.999,
        "near-identical vectors should have cos > 0.999, got {c}"
    );
}

#[test]
fn test_cosine_large_magnitude_finite() {
    // f32 squares of 1e20 overflow to inf; f64 accumulation must stay finite.
    let a = vec![1e20_f32, 0.0];
    let b = vec![1e20_f32, 0.0];
    let c = cosine_similarity(&a, &b);
    assert!(
        c.is_some(),
        "large identical vectors must return Some, not None"
    );
    assert!(
        c.unwrap().is_finite(),
        "large identical cosine must be finite, got {c:?}"
    );
    assert_close(
        c.unwrap(),
        1.0,
        "large identical vectors must give cos = 1.0",
    );

    let neg_b = vec![-1e20_f32, 0.0];
    let c2 = cosine_similarity(&a, &neg_b);
    assert!(c2.is_some(), "large opposite vectors must return Some");
    assert_close(
        c2.unwrap(),
        -1.0,
        "large opposite vectors must give cos = -1.0",
    );
}

#[test]
#[should_panic(expected = "vectors must have equal length")]
fn test_cosine_length_mismatch_panics() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0];
    let _ = cosine_similarity(&a, &b);
}

// ── BlockInfluence::from_vectors ──────────────────────────────────────────

#[test]
fn test_block_influence_identical() {
    let v = vec![1.0, 2.0, 3.0];
    let bi = BlockInfluence::from_vectors(0, &v, &v).unwrap();
    assert_close(bi.cosine_sim, 1.0, "identical cosine");
    assert!(
        bi.angular_distance < 1e-3,
        "identical angle should be near 0, got {}",
        bi.angular_distance
    );
    assert!(
        bi.influence < 1e-5,
        "identical influence should be near 0, got {}",
        bi.influence
    );
}

#[test]
fn test_block_influence_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let bi = BlockInfluence::from_vectors(5, &a, &b).unwrap();
    assert_close(bi.cosine_sim, 0.0, "orthogonal cosine");
    assert_close(
        bi.angular_distance,
        std::f32::consts::FRAC_PI_2,
        "orthogonal angle",
    );
    assert_close(bi.influence, 1.0, "orthogonal influence");
}

#[test]
fn test_block_influence_zero_vector_returns_none() {
    let zero = vec![0.0, 0.0, 0.0];
    let nonzero = vec![1.0, 0.0, 0.0];
    assert!(
        BlockInfluence::from_vectors(0, &zero, &nonzero).is_none(),
        "zero input → None"
    );
    assert!(
        BlockInfluence::from_vectors(0, &nonzero, &zero).is_none(),
        "zero output → None"
    );
}

// ── score_from_hidden_states ───────────────────────────────────────────────

#[test]
fn test_score_from_hidden_states() {
    let states = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0], // layer 0: no change → influence ≈ 0
        vec![0.0, 1.0, 0.0], // layer 1: orthogonal → influence ≈ 1
        vec![0.0, 0.7, 0.7], // layer 2: 45° rotation → influence ≈ 0.29
    ];
    let scores = score_from_hidden_states(&states);
    assert_eq!(scores.len(), 3);

    assert_close(scores[0].influence, 0.0, "layer 0 no change");
    assert_close(scores[1].influence, 1.0, "layer 1 orthogonal");

    // Layer 2: cos between (0,1,0) and (0,0.7,0.7)
    let expected_cos = 0.7 / (0.7_f32 * 0.7 + 0.7 * 0.7).sqrt();
    assert_close(scores[2].cosine_sim, expected_cos, "layer 2 cosine");
}

#[test]
fn test_score_from_hidden_states_single() {
    let states = vec![vec![1.0, 2.0]];
    assert!(score_from_hidden_states(&states).is_empty());
}

#[test]
fn test_score_from_hidden_states_zero_layer_omitted() {
    // A layer with zero input is omitted; later layers are still scored.
    let states = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0], // zero output → layer 0 omitted
        vec![0.0, 1.0, 0.0], // layer 1 has nonzero input and output
    ];
    let scores = score_from_hidden_states(&states);
    // Layer 0 omitted, layer 1 has nonzero input [0,0,0] → also None
    // (zero input from previous zero-output)
    assert_eq!(scores.len(), 0, "both layers omitted due to zero norms");
}

// ── pruning_rank ───────────────────────────────────────────────────────────

#[test]
fn test_pruning_rank_order() {
    let states = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0], // layer 0: influence ≈ 0 (most prunable)
        vec![0.0, 1.0, 0.0], // layer 1: influence ≈ 1 (least prunable)
        vec![0.0, 0.7, 0.7], // layer 2: intermediate
    ];
    let scores = score_from_hidden_states(&states);
    let rank = pruning_rank(&scores);

    assert_eq!(rank[0], 0, "most prunable first");
    assert_eq!(rank[2], 1, "least prunable last");
}

// ── BlockInfluenceAccumulator ──────────────────────────────────────────────

#[test]
fn test_accumulator_single_token_matches_from_vectors() {
    let a = vec![1.0_f32, 2.0, 3.0];
    let b = vec![2.0_f32, 3.0, 4.0];
    let direct = BlockInfluence::from_vectors(2, &a, &b).unwrap();

    let mut acc = BlockInfluenceAccumulator::new(2);
    acc.update(&a, &b);
    let averaged = acc.finalize().unwrap();

    assert_close(
        averaged.cosine_sim,
        direct.cosine_sim,
        "single-token acc == from_vectors",
    );
    assert_eq!(averaged.layer_idx, 2);
    assert_eq!(acc.count(), 1);
}

#[test]
fn test_accumulator_average_over_tokens() {
    // Two tokens: one identical pair (cos=1) and one orthogonal pair (cos=0).
    // Expected average cosine = 0.5, influence = 0.5.
    let identical_a = vec![1.0_f32, 0.0, 0.0];
    let orthogonal_b = vec![0.0_f32, 1.0, 0.0];

    let mut acc = BlockInfluenceAccumulator::new(0);
    acc.update(&identical_a, &identical_a); // cos = 1.0
    acc.update(&identical_a, &orthogonal_b); // cos = 0.0
    let bi = acc.finalize().unwrap();

    assert_eq!(acc.count(), 2);
    assert_close(bi.cosine_sim, 0.5, "average of 1.0 and 0.0");
    assert_close(bi.influence, 0.5, "influence = 1 - 0.5");
}

#[test]
fn test_accumulator_skips_zero_norm_tokens() {
    let zero = vec![0.0_f32; 4];
    let nonzero = vec![1.0_f32, 0.0, 0.0, 0.0];

    let mut acc = BlockInfluenceAccumulator::new(1);
    acc.update(&zero, &nonzero); // skipped
    acc.update(&nonzero, &zero); // skipped
    acc.update(&nonzero, &nonzero); // cos = 1.0, accepted

    assert_eq!(acc.count(), 1, "only non-zero-norm token counted");
    let bi = acc.finalize().unwrap();
    assert_close(bi.cosine_sim, 1.0, "only valid token was identical");
}

#[test]
fn test_accumulator_unequal_norm_tokens() {
    // Regression: accumulator must average per-token cosine values (paper formula),
    // NOT use sum(dot)/sum(norm_product) (ADR-060 pseudocode drift).
    //
    // Token 1: identical [1,0] pair → cos = 1.0, norm_product = 1.0
    // Token 2: orthogonal [100,0] vs [0,100] → cos = 0.0, norm_product = 10000.0
    //
    // Paper: mean(cos) = (1.0 + 0.0) / 2 = 0.5
    // Pseudocode drift: sum(dot) / sum(norms) = 0.0 / 10001.0 ≈ 0.0 ← WRONG
    let unit_x = vec![1.0_f32, 0.0];
    let _unit_y = [0.0_f32, 1.0];
    let big_x = vec![100.0_f32, 0.0];
    let big_y = vec![0.0_f32, 100.0];

    let mut acc = BlockInfluenceAccumulator::new(0);
    acc.update(&unit_x, &unit_x); // cos = 1.0
    acc.update(&big_x, &big_y); // cos = 0.0 (orthogonal, large norms)
    let bi = acc.finalize().unwrap();

    assert_eq!(acc.count(), 2);
    assert_close(
        bi.cosine_sim,
        0.5,
        "unequal-norm mean must be 0.5, not norm-weighted 0.0",
    );
    assert_close(bi.influence, 0.5, "influence = 1 - 0.5");
}

#[test]
fn test_accumulator_all_zero_returns_none() {
    let zero = vec![0.0_f32; 4];
    let mut acc = BlockInfluenceAccumulator::new(0);
    acc.update(&zero, &zero);
    assert_eq!(acc.count(), 0);
    assert!(acc.finalize().is_none(), "all-zero tokens → None");
}

#[test]
fn test_accumulator_empty_returns_none() {
    let acc = BlockInfluenceAccumulator::new(3);
    assert!(acc.finalize().is_none(), "no updates → None");
}

#[test]
fn test_accumulator_many_tokens_identical() {
    // 512 identical tokens → influence ≈ 0 regardless of count.
    let v = vec![1.0_f32; 896];
    let mut acc = BlockInfluenceAccumulator::new(0);
    for _ in 0..512 {
        acc.update(&v, &v);
    }
    let bi = acc.finalize().unwrap();
    assert_eq!(acc.count(), 512);
    assert_close(bi.cosine_sim, 1.0, "512 identical tokens");
    assert!(
        bi.influence < 1e-5,
        "influence near 0 for 512 identical, got {}",
        bi.influence
    );
}

#[test]
#[should_panic(expected = "vectors must have equal length")]
fn test_accumulator_length_mismatch_panics() {
    let mut acc = BlockInfluenceAccumulator::new(0);
    acc.update(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
}
