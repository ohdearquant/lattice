//! Tiny synthetic EmbeddingGemma checkpoint for native identity tests.
use lattice_model_embeddinggemma2::config::EmbeddingGemma2Config;
use std::path::Path;

const PREFIX: &str = "language_model.";

const TINY_CONFIG: &str = r#"{
  "vocab_size": 32,
  "hidden_size": 8,
  "intermediate_size": 16,
  "num_hidden_layers": 4,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 4,
  "hidden_size_per_layer_input": 6,
  "embedding_dim": 64,
  "rms_norm_eps": 1e-6,
  "sliding_window": 2,
  "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
  "hidden_activation": "gelu_pytorch_tanh",
  "per_layer_config": {
    "1": {"head_dim": 8, "num_key_value_heads": 1},
    "3": {"head_dim": 8, "num_key_value_heads": 1}
  },
  "rope_parameters": {
    "full_attention": {"rope_theta": 1000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 100.0, "rope_type": "default"}
  }
}"#;

struct Lcg(u64);

impl Lcg {
    /// Uniform in `[-1, 1)`.
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / (1u64 << 23) as f32 - 1.0
    }
}

/// Truncates to bf16 precision, so the value survives a bf16 round trip exactly.
fn bf16(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0xFFFF_0000)
}

type NamedTensor = (String, Vec<usize>, Vec<f32>);

struct Synth {
    rng: Lcg,
    prefix: String,
    tensors: Vec<NamedTensor>,
}

impl Synth {
    fn add(&mut self, name: &str, shape: &[usize], offset: f32, scale: f32) {
        let n: usize = shape.iter().product();
        let values = (0..n)
            .map(|_| bf16(offset + scale * self.rng.next()))
            .collect();
        self.tensors
            .push((format!("{}{name}", self.prefix), shape.to_vec(), values));
    }
}

fn synth_tensors(cfg: &EmbeddingGemma2Config, prefix: &str, seed: u64) -> Vec<NamedTensor> {
    let mut s = Synth {
        rng: Lcg(seed),
        prefix: prefix.to_string(),
        tensors: Vec::new(),
    };
    let hidden = cfg.hidden_size;
    let per_layer = cfg.hidden_size_per_layer_input;
    s.add("embed_tokens.weight", &[cfg.vocab_size, hidden], 0.0, 1.0);
    s.add(
        "ple.per_layer_model_projection.weight",
        &[cfg.num_hidden_layers * per_layer, hidden],
        0.0,
        0.5,
    );
    s.add(
        "ple.per_layer_projection_norm.weight",
        &[per_layer],
        1.0,
        0.2,
    );
    for l in 0..cfg.num_hidden_layers {
        let shape = cfg.layer_shapes[l];
        let q_dim = cfg.num_attention_heads * shape.head_dim;
        let kv_dim = shape.num_key_value_heads * shape.head_dim;
        let lp = format!("layers.{l}.");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            s.add(&format!("{lp}{norm}.weight"), &[hidden], 1.0, 0.2);
        }
        s.add(
            &format!("{lp}self_attn.q_proj.weight"),
            &[q_dim, hidden],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}self_attn.k_proj.weight"),
            &[kv_dim, hidden],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}self_attn.v_proj.weight"),
            &[kv_dim, hidden],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}self_attn.o_proj.weight"),
            &[hidden, q_dim],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}self_attn.q_norm.weight"),
            &[shape.head_dim],
            1.0,
            0.2,
        );
        s.add(
            &format!("{lp}self_attn.k_norm.weight"),
            &[shape.head_dim],
            1.0,
            0.2,
        );
        let ff = cfg.intermediate_size;
        s.add(
            &format!("{lp}mlp.gate_proj.weight"),
            &[ff, hidden],
            0.0,
            0.5,
        );
        s.add(&format!("{lp}mlp.up_proj.weight"), &[ff, hidden], 0.0, 0.5);
        s.add(
            &format!("{lp}mlp.down_proj.weight"),
            &[hidden, ff],
            0.0,
            0.5,
        );
        s.add(&format!("{lp}layer_scalar"), &[1], 0.9, 0.05);
        s.add(
            &format!("{lp}ple_block.per_layer_input_gate.weight"),
            &[per_layer, hidden],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}ple_block.per_layer_projection.weight"),
            &[hidden, per_layer],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}ple_block.post_per_layer_input_norm.weight"),
            &[hidden],
            1.0,
            0.2,
        );
    }
    s.add("norm.weight", &[hidden], 1.0, 0.2);
    s.add(
        "embedding_projection.weight",
        &[cfg.embedding_dim, hidden],
        0.0,
        0.5,
    );
    s.tensors
}

/// A safetensors buffer holding BF16 tensors, except for the names in `f32_names`, which are
/// stored as F32.
fn safetensors_bytes(tensors: &[NamedTensor], f32_names: &[&str]) -> Vec<u8> {
    let mut entries = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = payload.len();
        let is_f32 = f32_names.contains(&name.as_str());
        for v in values {
            if is_f32 {
                payload.extend_from_slice(&v.to_le_bytes());
            } else {
                payload.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        let dtype = if is_f32 { "F32" } else { "BF16" };
        entries.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{start},{}]}}"#,
            dims.join(","),
            payload.len()
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    out
}

const TINY_TOKENIZER: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 4, "content": "<mask>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"},
  "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [{"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
    "special_tokens": {}
  },
  "decoder": {"type": "Sequence", "decoders": [
    {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
    {"type": "ByteFallback"},
    {"type": "Fuse"}
  ]},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": true,
    "byte_fallback": true,
    "ignore_merges": false,
    "vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4,
              "a": 5, "b": 6, "c": 7, "d": 8, "\u2581": 9, "ab": 10},
    "merges": [["a", "b"]]
  }
}"#;

pub(super) fn write_checkpoint(dir: &Path) -> Vec<u8> {
    let cfg = EmbeddingGemma2Config::from_config_json_str(TINY_CONFIG).unwrap();
    let weights = safetensors_bytes(&synth_tensors(&cfg, PREFIX, 7), &[]);
    std::fs::write(dir.join("config.json"), TINY_CONFIG).unwrap();
    std::fs::write(dir.join("tokenizer.json"), TINY_TOKENIZER).unwrap();
    std::fs::write(dir.join("model.safetensors"), &weights).unwrap();
    weights
}
