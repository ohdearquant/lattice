//! Functional smoke test for issue #884: `train_micro_lora_with_gdn` must
//! actually populate GDN LoRA slots (the pre-fix code left `GdnLoraParams`
//! permanently empty — see `crates/tune/src/lora/train.rs`), and
//! `train_micro_lora`/`train_micro_lora_with_gdn(.., train_gdn: false)` must
//! remain byte-identical to the pre-#884 GQA-only behavior.
//!
//! Not run in CI: requires a real Qwen3.5-0.8B checkpoint on disk at
//! `$HOME/.lattice/models/qwen3.5-0.8b` (or `LATTICE_MODEL_DIR`), matching
//! the existing convention in `bench_backward_737.rs`.
use std::path::PathBuf;

use lattice_inference::model::qwen35::Qwen35Model;
use lattice_tune::lora::train::{
    MicroLoraConfig, TrainingPair, train_micro_lora, train_micro_lora_with_gdn,
};

fn model_dir() -> PathBuf {
    std::env::var("LATTICE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(".lattice")
                .join("models")
                .join("qwen3.5-0.8b")
        })
}

/// Deterministic xorshift64 token generator (mirrors `bench_backward_737.rs`).
fn synth_pairs(vocab: usize, n_pairs: usize, seq_len: usize) -> Vec<TrainingPair> {
    let mut state: u64 = 0x1234_5678_9abc_def1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n_pairs)
        .map(|_| {
            let tokens: Vec<u32> = (0..seq_len)
                .map(|_| (next() % vocab as u64) as u32)
                .collect();
            TrainingPair {
                tokens,
                completion_start: seq_len / 2,
            }
        })
        .collect()
}

fn gdn_config() -> MicroLoraConfig {
    MicroLoraConfig {
        rank: 4,
        alpha: 8.0,
        // Spans layers 19..=23: GQA(19), GDN(20,21,22), GQA(23) on the
        // Qwen3.5-0.8B [linear,linear,linear,full] hybrid layout — the same
        // window `bench_backward_737.rs` uses to exercise every mixer kind.
        first_layer: 19,
        last_layer: Some(23),
        steps: 1,
        learning_rate: 1e-3,
        max_seq_len: 64,
    }
}

const GDN_MODULES: [&str; 5] = [
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
];

#[test]
#[ignore = "requires a real Qwen3.5-0.8B checkpoint on disk; run explicitly for #884 verification"]
fn train_gdn_false_matches_train_micro_lora_exactly() {
    let model = Qwen35Model::from_safetensors(&model_dir())
        .unwrap_or_else(|e| panic!("failed to load model: {e}"));
    let vocab = model.config().vocab_size;
    let pairs = synth_pairs(vocab, 2, 48);
    let config = gdn_config();

    let baseline = train_micro_lora(&model, &pairs, &config).expect("train_micro_lora");
    let opted_out = train_micro_lora_with_gdn(&model, &pairs, &config, false)
        .expect("train_micro_lora_with_gdn(false)");

    assert_eq!(
        baseline.config().target_modules,
        opted_out.config().target_modules,
        "train_gdn=false must keep the GQA-only target_modules list"
    );
    for module in GDN_MODULES {
        assert!(
            !baseline.has_adapter(20, module) && !opted_out.has_adapter(20, module),
            "neither path should carry a GDN adapter when train_gdn is off"
        );
    }
    for ((layer, module), layer_w) in baseline.layers() {
        let other = opted_out
            .layers()
            .get(&(*layer, module.clone()))
            .unwrap_or_else(|| panic!("opted_out adapter missing ({layer}, {module})"));
        assert_eq!(
            layer_w.a, other.a,
            "layer {layer} module {module}: A factors diverged between train_micro_lora and \
             train_micro_lora_with_gdn(.., false) — GDN init must not perturb the GQA rng stream"
        );
        assert_eq!(
            layer_w.b, other.b,
            "layer {layer} module {module}: B factors diverged between train_micro_lora and \
             train_micro_lora_with_gdn(.., false)"
        );
    }
}

#[test]
#[ignore = "requires a real Qwen3.5-0.8B checkpoint on disk; run explicitly for #884 verification"]
fn train_gdn_true_populates_and_trains_all_five_gdn_modules() {
    let model = Qwen35Model::from_safetensors(&model_dir())
        .unwrap_or_else(|e| panic!("failed to load model: {e}"));
    let vocab = model.config().vocab_size;
    let pairs = synth_pairs(vocab, 2, 48);
    let config = gdn_config();

    let adapter = train_micro_lora_with_gdn(&model, &pairs, &config, true)
        .expect("train_micro_lora_with_gdn(true)");

    for module in ["q_proj", "v_proj"] {
        assert!(
            adapter
                .config()
                .target_modules
                .contains(&module.to_string())
        );
    }
    for module in GDN_MODULES {
        assert!(
            adapter
                .config()
                .target_modules
                .contains(&module.to_string()),
            "target_modules missing {module}"
        );
        for gdn_layer in [20usize, 21, 22] {
            assert!(
                adapter.has_adapter(gdn_layer, module),
                "adapter missing ({gdn_layer}, {module})"
            );
        }
    }
    // GQA layers must not pick up GDN modules, and layer 19/23 (GQA) must
    // not carry GDN adapters.
    for gqa_layer in [19usize, 23] {
        for module in GDN_MODULES {
            assert!(
                !adapter.has_adapter(gqa_layer, module),
                "GQA layer {gqa_layer} must not carry GDN module {module}"
            );
        }
    }

    // B factors start at zero; after >=1 Adam step with a nonzero gradient
    // at least one GDN slot's B array must have moved off zero — otherwise
    // this test would pass vacuously even if gradients/updates were wired
    // to a no-op.
    let mut any_nonzero_b = false;
    for gdn_layer in [20usize, 21, 22] {
        for module in GDN_MODULES {
            let layer_w = adapter
                .layers()
                .get(&(gdn_layer, module.to_string()))
                .expect("checked has_adapter above");
            if layer_w.b.iter().any(|&v| v != 0.0) {
                any_nonzero_b = true;
            }
            assert!(
                layer_w.a.iter().all(|v| v.is_finite()) && layer_w.b.iter().all(|v| v.is_finite()),
                "layer {gdn_layer} module {module}: non-finite LoRA factor after training"
            );
        }
    }
    assert!(
        any_nonzero_b,
        "no GDN B factor moved off zero after training — GDN gradients/Adam updates are not wired"
    );
}
