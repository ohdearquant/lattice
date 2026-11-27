//! The bf16 CPU weight store against the f32 store holding the same values.
//!
//! Both models are built from one seed, and every large matrix of the f32 model is first
//! rounded to a bf16 value, so the two stores hold identical numbers and differ only in how a
//! product reads them. The assertions are a tolerance well below any answer-changing
//! difference; whether the outputs are also bit-identical is printed as an observation.

use super::*;
use crate::model::qwen35::{AttentionWeights, FeedForwardWeights};
use lattice_compute::forward::cpu::weight_matrix::WeightMatrix;

fn large_matrices(weights: &mut ModelWeights) -> Vec<&mut WeightMatrix> {
    let mut matrices = vec![&mut weights.embed_tokens];
    if let Some(head) = weights.lm_head.as_mut() {
        matrices.push(head);
    }
    for (attn, common) in &mut weights.layers {
        match attn {
            AttentionWeights::Linear(w) => {
                matrices.extend([&mut w.in_proj_qkv, &mut w.in_proj_z, &mut w.out_proj]);
            }
            AttentionWeights::Full(w) => {
                matrices.extend([&mut w.q_proj, &mut w.k_proj, &mut w.v_proj, &mut w.o_proj]);
            }
        }
        if let FeedForwardWeights::Dense(ffn) = &mut common.ffn {
            matrices.extend([&mut ffn.gate_proj, &mut ffn.up_proj, &mut ffn.down_proj]);
        }
    }
    matrices
}

/// The f32 model with each large matrix rounded to bf16 values, and its twin holding those
/// matrices as bf16 bits.
fn f32_and_bf16_models(seed: u64) -> (Qwen35Model, Qwen35Model) {
    let mut f32_model = build_model(test_config(), seed);
    let mut bf16_model = build_model(test_config(), seed);
    let f32_matrices = large_matrices(&mut f32_model.weights);
    let bf16_matrices = large_matrices(&mut bf16_model.weights);
    assert_eq!(f32_matrices.len(), bf16_matrices.len());
    for (f32_matrix, bf16_matrix) in f32_matrices.into_iter().zip(bf16_matrices) {
        let values = f32_matrix
            .as_f32_mut()
            .expect("the test model is built as f32");
        let bits: Vec<u16> = values
            .iter()
            .map(|value| (value.to_bits() >> 16) as u16)
            .collect();
        for (value, &b) in values.iter_mut().zip(&bits) {
            *value = f32::from_bits(u32::from(b) << 16);
        }
        *bf16_matrix = WeightMatrix::Bf16(bits);
    }
    assert!(bf16_model.weights.embed_tokens.is_bf16());
    (f32_model, bf16_model)
}

/// Largest absolute difference, and whether every pair is bit-identical.
fn compare(label: &str, f32_values: &[f32], bf16_values: &[f32]) -> f32 {
    assert_eq!(f32_values.len(), bf16_values.len(), "{label}: lengths");
    let identical = f32_values
        .iter()
        .zip(bf16_values)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let max_diff = f32_values
        .iter()
        .zip(bf16_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("{label}: max |f32 - bf16| = {max_diff:e}, bit-identical = {identical}");
    max_diff
}

fn scale(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(1e-6_f32, f32::max)
}

#[test]
fn bf16_store_readout_matches_the_f32_store_on_the_same_values() {
    let (f32_model, bf16_model) = f32_and_bf16_models(0x5eed_0001);
    let tokens = [1_u32, 5, 2, 6, 3, 4, 6, 1, 2];
    let slots = [3, 8];
    let rows = [1_u32, 2, 3, 4, 5];
    let f32_out = f32_model
        .prefill_readout(PrefillInput::TokenIds(&tokens), &slots, &rows)
        .expect("f32 readout");
    let bf16_out = bf16_model
        .prefill_readout(PrefillInput::TokenIds(&tokens), &slots, &rows)
        .expect("bf16 readout");
    for (slot, (a, b)) in f32_out.iter().zip(&bf16_out).enumerate() {
        let diff = compare(&format!("readout slot {slot}"), a, b);
        assert!(diff <= 1e-5 * scale(a), "slot {slot}: {diff:e}");
    }
}

#[test]
fn bf16_store_option_scores_match_the_f32_store_on_the_same_values() {
    let (f32_model, bf16_model) = f32_and_bf16_models(0x5eed_0002);
    let first = [1_u32, 5, 2, 6, 3, 4, 6, 1];
    let second = [1_u32, 5, 2, 6, 3, 4, 6, 2, 3];
    let letters = [1_u32, 2, 3];
    let rows: [(&[u32], &[u32]); 2] = [(&first, &letters), (&second, &letters)];
    let f32_scores = f32_model
        .score_option_rows_shared_prefix(&rows)
        .expect("f32 scores");
    let bf16_scores = bf16_model
        .score_option_rows_shared_prefix(&rows)
        .expect("bf16 scores");
    for (row, (a, b)) in f32_scores.iter().zip(&bf16_scores).enumerate() {
        let diff = compare(&format!("option row {row} logits"), &a.logits, &b.logits);
        assert!(diff <= 1e-5 * scale(&a.logits), "row {row}: {diff:e}");
        let diff = compare(&format!("option row {row} probs"), &a.probs, &b.probs);
        assert!(diff <= 1e-5, "row {row} probs: {diff:e}");
    }
}
