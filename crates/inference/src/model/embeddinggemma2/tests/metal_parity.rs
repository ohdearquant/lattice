//! Whole-model parity of the Metal f32 path with the CPU path on synthetic checkpoints.
//!
//! Two checkpoints cover both layer kinds: the tiny one (heads of width 4 and 8) and a wide one
//! whose sliding layers use the released model's head width of 256 and whose full layers use 512.
//! Both CPU and GPU run f32 on the same weights, so the two paths differ only in summation order
//! and in the libm versus Metal implementations of `exp`, `tanh` and `rsqrt`.

use super::*;
use crate::forward::metal_embeddinggemma2::MetalEmbeddingGemma2State;
use crate::measurement::gpu_test_lock;

const WIDE_CONFIG: &str = r#"{
  "vocab_size": 64,
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_hidden_layers": 6,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "hidden_size_per_layer_input": 32,
  "embedding_dim": 48,
  "rms_norm_eps": 1e-6,
  "sliding_window": 3,
  "layer_types": ["sliding_attention", "sliding_attention", "full_attention",
                  "sliding_attention", "sliding_attention", "full_attention"],
  "hidden_activation": "gelu_pytorch_tanh",
  "per_layer_config": {
    "2": {"head_dim": 512, "num_key_value_heads": 1},
    "5": {"head_dim": 512, "num_key_value_heads": 1}
  },
  "rope_parameters": {
    "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
  }
}"#;

fn wide_fixture(seed: u64) -> Fixture {
    let cfg = EmbeddingGemma2Config::from_config_json_str(WIDE_CONFIG).expect("wide config parses");
    let list = synth_tensors(&cfg, PREFIX, seed);
    fixture_from(cfg, list)
}

fn ids_for(vocab: usize, len: usize, step: usize) -> Vec<u32> {
    (0..len).map(|i| ((i * step + 1) % vocab) as u32).collect()
}

fn enforce() -> bool {
    std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some()
}

/// True (after saying so) when this machine has no Metal device and the run is not enforcing one.
fn skip_without_device() -> bool {
    if ::metal::Device::system_default().is_some() {
        return false;
    }
    assert!(!enforce(), "Metal device required under enforcement");
    eprintln!("SKIP embeddinggemma2 Metal: no Metal device");
    true
}

/// The bound on the largest per-element difference between the two f32 paths: 1e-4 of the larger
/// of 1 and the largest CPU value. Summation order and libm-versus-Metal transcendental rounding
/// give differences near 1e-5 relative; the bound leaves headroom for the 24-layer-style
/// accumulation without hiding a wrong kernel, whose error is of the order of the values.
fn bound(cpu: &[f32]) -> f32 {
    1e-4 * cpu.iter().fold(1f32, |m, v| m.max(v.abs()))
}

fn assert_states_match(name: &str, metal: &[f32], cpu: &[f32]) {
    let diff = max_diff32(metal, cpu);
    let limit = bound(cpu);
    eprintln!("{name}: max abs diff {diff:e} (bound {limit:e})");
    assert!(
        diff <= limit,
        "{name}: Metal differs from CPU by {diff}, bound {limit}"
    );
}

#[test]
fn metal_token_states_match_cpu_on_both_synthetic_checkpoints() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let tiny = fixture();
    let wide = wide_fixture(13);
    for (name, fx, ids) in [
        ("tiny", &tiny, IDS.to_vec()),
        ("tiny long", &tiny, ids_for(32, 25, 7)),
        ("wide", &wide, ids_for(64, 23, 5)),
        ("wide short", &wide, ids_for(64, 2, 5)),
    ] {
        let mut state = MetalEmbeddingGemma2State::new(&fx.model).expect("state builds");
        let cpu = fx.model.token_states(&ids).expect("cpu forward");
        let metal = fx
            .model
            .token_states_metal(&mut state, &ids)
            .expect("metal forward");
        assert_states_match(name, &metal, &cpu);
    }
}

#[test]
fn metal_pooled_embeddings_match_cpu_at_every_width() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let tiny = fixture();
    let wide = wide_fixture(13);
    for (name, fx, ids) in [
        ("tiny", &tiny, IDS.to_vec()),
        ("wide", &wide, ids_for(64, 23, 5)),
    ] {
        let mut state = MetalEmbeddingGemma2State::new(&fx.model).expect("state builds");
        let dim = fx.cfg.embedding_dim;
        let widths = [dim, dim - 2, 5, 1];
        let cpu = fx
            .model
            .encode_ids_at_widths(&ids, &widths)
            .expect("cpu embeddings");
        let metal = fx
            .model
            .encode_ids_at_widths_metal(&mut state, &ids, &widths)
            .expect("metal embeddings");
        assert_eq!(metal.len(), widths.len());
        for ((m, c), width) in metal.iter().zip(&cpu).zip(widths) {
            assert_eq!(m.len(), width);
            let diff = max_diff32(m, c);
            eprintln!("{name} width {width}: max abs diff {diff:e}");
            assert!(diff <= 5e-5, "{name} width {width}: differs by {diff}");
            let norm = m.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "{name} width {width}: norm {norm}"
            );
        }
    }
}

#[test]
fn one_state_serves_sequences_of_changing_length() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let fx = wide_fixture(17);
    let mut state = MetalEmbeddingGemma2State::new(&fx.model).expect("state builds");
    // Growing, shrinking, growing past the first capacity, and a repeat of an earlier length
    // with different ids: stale activations from a longer call must not leak into a shorter one.
    for (len, step) in [(9usize, 5usize), (3, 7), (30, 3), (9, 11), (1, 1), (30, 13)] {
        let ids = ids_for(64, len, step);
        let cpu = fx.model.token_states(&ids).expect("cpu forward");
        let metal = fx
            .model
            .token_states_metal(&mut state, &ids)
            .expect("metal forward");
        assert_states_match(&format!("len {len} step {step}"), &metal, &cpu);
    }
}

#[test]
fn metal_refuses_a_foreign_state_and_invalid_inputs() {
    let _gpu = gpu_test_lock();
    if skip_without_device() {
        return;
    }
    let fx = wide_fixture(19);
    let other = wide_fixture(23);
    let mut state = MetalEmbeddingGemma2State::new(&fx.model).expect("state builds");
    let ids = ids_for(64, 6, 5);

    match other.model.token_states_metal(&mut state, &ids) {
        Err(InferenceError::InvalidInput(reason)) => {
            assert!(
                reason.contains("different model"),
                "wrong rejection: {reason}"
            );
        }
        result => panic!("a state from another model must be refused, got {result:?}"),
    }
    assert!(matches!(
        fx.model.token_states_metal(&mut state, &[]),
        Err(InferenceError::InvalidInput(_))
    ));
    assert!(matches!(
        fx.model.token_states_metal(&mut state, &[1, 64, 2]),
        Err(InferenceError::InvalidInput(_))
    ));
    for widths in [vec![0usize], vec![49]] {
        assert!(matches!(
            fx.model
                .encode_ids_at_widths_metal(&mut state, &ids, &widths),
            Err(InferenceError::InvalidInput(_))
        ));
    }
    // A non-finite weight is refused when the state is built, before any device access.
    let mut corrupt = wide_fixture(29);
    corrupt.model.weights.layers[1].gate_proj[3] = f32::NAN;
    let refused = MetalEmbeddingGemma2State::new(&corrupt.model);
    assert!(matches!(refused, Err(InferenceError::InvalidInput(_))));
    // A refused call leaves the state usable.
    let cpu = fx.model.token_states(&ids).expect("cpu forward");
    let metal = fx
        .model
        .token_states_metal(&mut state, &ids)
        .expect("metal forward after refusals");
    assert_states_match("after refusals", &metal, &cpu);
}
