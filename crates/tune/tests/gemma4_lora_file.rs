//! Real-checkpoint test for the PEFT file-to-Gemma attachment boundary.
#![cfg(all(feature = "safetensors", feature = "inference-hook"))]

use lattice_inference::lora_hook::{LoraHook, LoraTargetGeometry};
use lattice_inference::model::gemma4_model::Gemma4Model;
use lattice_tune::lora::{LoraAdapter, LoraConfig, LoraLayer};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct CountedAdapter {
    inner: LoraAdapter,
    calls: Arc<AtomicUsize>,
}

impl LoraHook for CountedAdapter {
    fn apply(&self, layer: usize, module: &str, input: &[f32], output: &mut [f32]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        LoraHook::apply(&self.inner, layer, module, input, output);
    }
    fn is_active(&self, layer: usize, module: &str) -> bool {
        LoraHook::is_active(&self.inner, layer, module)
    }
    fn validate_geometry(&self, geometry: &dyn LoraTargetGeometry) -> Result<(), String> {
        LoraHook::validate_geometry(&self.inner, geometry)
    }
}

fn adapter(layer: usize, module: &str, input: usize, output: usize) -> LoraAdapter {
    let config = LoraConfig {
        rank: 1,
        alpha: 1.0,
        target_modules: vec![module.to_owned()],
        dtype: "f32".to_owned(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (layer, module.to_owned()),
        LoraLayer {
            a: vec![1.0; input],
            b: vec![0.0; output],
            d_in: input,
            d_out: output,
            rank: 1,
        },
    );
    LoraAdapter::new(config, layers).unwrap()
}

#[test]
#[ignore = "requires LATTICE_GEMMA4_MODEL_DIR with a Gemma 4 checkpoint"]
fn loaded_peft_adapter_refuses_shared_kv_at_attachment() {
    let path = PathBuf::from(
        std::env::var_os("LATTICE_GEMMA4_MODEL_DIR").expect("set LATTICE_GEMMA4_MODEL_DIR"),
    );
    let mut model = Gemma4Model::from_safetensors(&path).unwrap();
    let cfg = model.config();
    let shared = (0..cfg.num_hidden_layers)
        .find(|&layer| cfg.is_kv_shared_layer(layer))
        .expect("fixture must have a shared KV layer");
    let hidden = cfg.hidden_size;
    let q_width = cfg.num_attention_heads * cfg.attn_head_dim(0);
    let kv_width = cfg.num_key_value_heads * cfg.attn_head_dim(shared);
    let dir = tempfile::tempdir().unwrap();
    let valid_path = dir.path().join("q.safetensors");
    adapter(0, "q_proj", hidden, q_width)
        .save_safetensors(&valid_path, None)
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    model
        .set_lora(Box::new(CountedAdapter {
            inner: LoraAdapter::from_safetensors(&valid_path).unwrap(),
            calls: calls.clone(),
        }))
        .unwrap();
    for module in ["k_proj", "v_proj"] {
        let invalid_path = dir.path().join(format!("{module}.safetensors"));
        adapter(shared, module, hidden, kv_width)
            .save_safetensors(&invalid_path, None)
            .unwrap();
        let loaded = LoraAdapter::from_safetensors(&invalid_path).unwrap();
        let error = model.set_lora(Box::new(loaded)).err().unwrap().to_string();
        assert!(
            error.contains(&format!("layer {shared}")) && error.contains(module),
            "{error}"
        );
        let before = calls.load(Ordering::Relaxed);
        model.generate_greedy(&[0], 0, 2).unwrap();
        assert!(
            calls.load(Ordering::Relaxed) > before,
            "failed attachment replaced the prior adapter"
        );
    }
}
