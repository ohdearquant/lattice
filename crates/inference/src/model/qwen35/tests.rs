use super::moe::moe_ffn_step;
use super::weights::{MoeLayerWeights, MoeRouter, RoutedExperts, SharedExpert};
use super::*;
use crate::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;

#[test]
fn test_partial_rope_only_rotates_first_quarter() {
    // Build a RoPE table for rope_dim=64 (head_dim=256, partial_rotary_factor=0.25)
    let rope_dim = 64;
    let rope = crate::rope::RopeTable::new(rope_dim, 128, 10_000_000.0);

    // Create a head vector of 256 dims
    let mut head = vec![1.0f32; 256];
    let original_tail = head[rope_dim..].to_vec();

    // Apply partial RoPE at position 10
    rope.apply(&mut head[..rope_dim], 10);

    // First rope_dim elements should be modified
    let rotated_part = &head[..rope_dim];
    // At position > 0 with non-trivial theta, at least some dims should change
    let any_changed = rotated_part.iter().any(|&v| (v - 1.0).abs() > 1e-6);
    assert!(
        any_changed,
        "RoPE should modify the rotary dimensions at non-zero position"
    );

    // Remaining dims should be unchanged
    assert_eq!(
        &head[rope_dim..],
        &original_tail[..],
        "Non-rotary dimensions must remain unchanged"
    );
}

#[test]
fn test_sample_greedy() {
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let cfg = crate::model::qwen35_config::GenerateConfig {
        temperature: 0.0,
        ..Default::default()
    };
    let mut rng = 1u64;
    let token = sample_token(&logits, &cfg, &[], &mut rng);
    assert_eq!(token, 3, "greedy should pick index with highest logit");
}

#[test]
fn test_sample_repetition_penalty() {
    // Token 3 has highest logit but gets penalized
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let cfg = crate::model::qwen35_config::GenerateConfig {
        temperature: 0.0,
        repetition_penalty: 100.0,
        ..Default::default()
    };
    let mut rng = 1u64;
    let token = sample_token(&logits, &cfg, &[3], &mut rng);
    assert_ne!(
        token, 3,
        "with heavy repetition penalty, should not pick the repeated token"
    );
}

#[test]
fn test_decode_tokens_roundtrip() {
    // Test the byte_encoder/decoder roundtrip
    let encoder = bytes_to_unicode();
    let mut decoder: std::collections::HashMap<char, u8> = std::collections::HashMap::new();
    for (b, &ch) in encoder.iter().enumerate() {
        decoder.insert(ch, b as u8);
    }

    // Every byte should roundtrip
    for byte_val in 0u8..=255 {
        let ch = encoder[byte_val as usize];
        let decoded = decoder[&ch];
        assert_eq!(
            byte_val, decoded,
            "byte {byte_val} -> char {ch:?} -> byte {decoded} should roundtrip"
        );
    }
}

#[test]
fn test_forward_scratch_decode_temporaries_are_reusable() {
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let q_proj_dim = 2 * cfg.full_q_dim();
    let q_dim = cfg.full_q_dim();
    let inter = cfg.intermediate_size;

    let mut scratch = ForwardScratch::new();
    scratch.ensure_decode_capacity(hidden, q_proj_dim, q_dim, inter);

    let input_ptr = scratch.input_tmp.as_ptr();
    let q_gate_ptr = scratch.q_and_gate.as_ptr();
    let gate_z_ptr = scratch.gate_z.as_ptr();
    let down_input_ptr = scratch.down_input.as_ptr();

    scratch.ensure_decode_capacity(hidden, q_proj_dim, q_dim, inter);

    assert_eq!(scratch.input_tmp.as_ptr(), input_ptr);
    assert_eq!(scratch.q_and_gate.as_ptr(), q_gate_ptr);
    assert_eq!(scratch.gate_z.as_ptr(), gate_z_ptr);
    assert_eq!(scratch.down_input.as_ptr(), down_input_ptr);
    assert!(scratch.input_tmp.len() >= hidden);
    assert!(scratch.q_and_gate.len() >= q_proj_dim);
    assert!(scratch.gate_z.len() >= q_dim);
    assert!(scratch.down_input.len() >= inter);
}

#[test]
fn test_qwen36_required_tensor_names_include_moe_and_lm_head() {
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen36_35b_a3b();
    let names = qwen_required_tensor_names(&cfg);

    // Untied lm_head
    assert!(names.contains(&"lm_head.weight".to_string()));

    // Layer 0 is linear attention
    assert!(
        names.contains(&"model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_string())
    );
    // Layer 3 is full attention
    assert!(names.contains(&"model.language_model.layers.3.self_attn.q_proj.weight".to_string()));

    // MoE-specific names appear for every layer
    assert!(names.contains(&"model.language_model.layers.0.mlp.gate.weight".to_string()));
    assert!(names.contains(&"model.language_model.layers.0.mlp.experts.gate_up_proj".to_string()));
    assert!(
        names.contains(&"model.language_model.layers.0.mlp.shared_expert_gate.weight".to_string())
    );

    // Dense FFN names must NOT appear (MoE config)
    assert!(!names.contains(&"model.language_model.layers.0.mlp.gate_proj.weight".to_string()));
}

#[test]
fn test_qwen36_27b_required_tensor_names_untied_lm_head() {
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen36_27b();
    let names = qwen_required_tensor_names(&cfg);
    assert!(
        names.iter().any(|n| n == "lm_head.weight"),
        "lm_head.weight must be separate — tie_word_embeddings=false"
    );
    assert!(names.iter().any(|n| n.contains("embed_tokens")));
    // dense FFN: gate_proj exists, no expert sharding
    assert!(names.iter().any(|n| n.contains("mlp.gate_proj")));
}

#[test]
fn test_qwen36_27b_required_tensor_names_language_model_prefix() {
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen36_27b();
    let names = qwen_required_tensor_names(&cfg);
    let layer_names: Vec<_> = names.iter().filter(|n| n.contains("layers.")).collect();
    assert!(!layer_names.is_empty());
    assert!(
        layer_names
            .iter()
            .all(|n| n.starts_with("model.language_model.")),
        "all layer weights must use model.language_model.* prefix"
    );
    // not MoE — no expert sharding names
    assert!(
        !names
            .iter()
            .any(|n| n.contains("experts") || n.contains("moe"))
    );
    assert!(!names.iter().any(|n| n.contains("shared_expert")));
}

#[test]
fn test_should_stop_token_includes_im_end() {
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen36_35b_a3b();
    let gen_cfg = crate::model::qwen35_config::GenerateConfig::default();
    assert!(should_stop_token(&cfg, &gen_cfg, cfg.eos_token_id));
    assert!(should_stop_token(&cfg, &gen_cfg, QWEN_CHAT_IM_END_TOKEN_ID));
    assert!(!should_stop_token(&cfg, &gen_cfg, 123));
}

#[test]
fn test_moe_router_construction_validates_shapes() {
    // 3 experts, hidden=2, top_k=2 — new signature: (gate, num_experts, num_experts_per_tok, hidden_size)
    let router = MoeRouter::new(
        vec![
            1.0, 0.0, // expert 0 row
            0.0, 1.0, // expert 1 row
            1.0, 1.0, // expert 2 row
        ],
        3,
        2,
        2,
    )
    .expect("router shape is valid");
    assert_eq!(router.num_experts, 3);
    assert_eq!(router.num_experts_per_tok, 2);
    assert_eq!(router.hidden_size, 2);

    // top_k > num_experts must error
    let err = MoeRouter::new(vec![1.0, 0.0], 1, 5, 2);
    assert!(err.is_err(), "top_k > num_experts should fail");

    // wrong gate length must error
    let err2 = MoeRouter::new(vec![1.0, 0.0, 0.5], 3, 1, 2);
    assert!(err2.is_err(), "wrong gate length should fail");
}

#[test]
fn test_shared_expert_shape_validation() {
    // hidden=3, intermediate=2: gate/up=[2,3] (len=6), down=[3,2] (len=6), gate_vec=[3]
    let expert = SharedExpert::new(vec![0.0; 6], vec![0.0; 6], vec![0.0; 6], vec![0.0; 3], 3, 2);
    assert!(expert.is_ok(), "valid shapes must succeed");

    // Bad gate_proj length
    let err = SharedExpert::new(
        vec![0.0; 5], // wrong: should be 6
        vec![0.0; 6],
        vec![0.0; 6],
        vec![0.0; 3],
        3,
        2,
    )
    .expect_err("bad gate shape should fail");
    assert!(
        err.to_string()
            .contains("mlp.shared_expert.gate_proj.weight"),
        "error should name the tensor: {err}"
    );
}

#[test]
fn test_moe_routing_numeric_determinism() {
    // Config: hidden=1, inter=1, shared_inter=1, num_experts=2, top_k=1
    //
    // Input x = [1.0].
    //
    // Gate weights [2, 1] = [1.0, -1.0]:
    //   logit[0] = 1.0 * 1.0 = 1.0
    //   logit[1] = 1.0 * (-1.0) = -1.0
    //
    // Softmax (max=1.0):
    //   raw = [exp(0), exp(-2)] = [1.0, ~0.13534]
    //   denom ~= 1.13534
    //   prob[0] ~= 0.88080, prob[1] ~= 0.11920
    //
    // Top-1: expert 0, renorm weight = 1.0 (trivially).
    //
    // Expert 0 gate_up_proj = [0.5, 2.0] (gate_w=[0.5], up_w=[2.0]):
    //   gate_out = 0.5, up_out = 2.0
    //   silu(0.5) = 0.5/(1+exp(-0.5)) ~= 0.31122
    //   intermediate = silu(0.5) * 2.0 ~= 0.62244
    //   down_proj=[3.0]: down_out[0] = 3.0 * 0.62244 ~= 1.86733
    // Expert 0 down_proj[1] = 0.0 → down_out[1] = 0.0 (but hidden=1, so only one output)
    //
    // Routed sum = 1.0 * [1.86733] (weight=1.0)
    //
    // Shared expert:
    //   shared_expert_gate=[2.0]: gate_logit = 2.0*1.0 = 2.0
    //   shared_gate = sigmoid(2.0) ~= 0.88080
    //   gate_proj=[0.5], up_proj=[2.0]: same as expert 0 → silu(0.5)*2.0
    //   down_proj=[1.0]: shared_out = silu(0.5)*2.0
    //   shared_contrib = sigmoid(2.0) * silu(0.5) * 2.0
    //
    // Final = 3.0*silu(0.5)*2.0 + sigmoid(2.0)*silu(0.5)*2.0
    //       = silu(0.5)*2.0*(3.0 + sigmoid(2.0))
    let hidden = 1usize;
    let inter = 1usize;
    let num_experts = 2usize;

    let router = MoeRouter::new(vec![1.0f32, -1.0f32], num_experts, 1, hidden).unwrap();

    let experts = RoutedExperts::new(
        vec![0.5f32, 2.0f32, 0.0f32, 0.0f32], // expert 0: [gate_w=0.5, up_w=2.0], expert 1: zeros
        vec![3.0f32, 0.0f32],                 // expert 0: down=[3.0], expert 1: [0.0]
        num_experts,
        hidden,
        inter,
    )
    .unwrap();

    let shared = SharedExpert::new(
        vec![0.5f32], // gate_proj [inter=1, hidden=1]
        vec![2.0f32], // up_proj   [inter=1, hidden=1]
        vec![1.0f32], // down_proj [hidden=1, inter=1]
        vec![2.0f32], // shared_expert_gate [hidden=1]
        hidden,
        inter,
    )
    .unwrap();

    let moe = MoeLayerWeights {
        router,
        experts,
        shared_expert: shared,
    };

    let mut scratch = ForwardScratch::new();
    scratch.ffn_out = vec![1.0f32]; // input token hidden state
    scratch.input_tmp = vec![0.0f32; hidden];
    scratch.expert_out = vec![0.0f32; hidden];
    scratch.gate_buf = vec![0.0f32; inter];
    scratch.up_buf = vec![0.0f32; inter];
    scratch.down_input = vec![0.0f32; inter];

    moe_ffn_step(&moe, &mut scratch, hidden);

    let silu_half = 0.5f32 / (1.0f32 + (-0.5f32).exp()); // silu(0.5) = 0.5 * sigmoid(0.5)
    let sig2 = 1.0f32 / (1.0f32 + (-2.0f32).exp()); // sigmoid(2.0)
    let expected = silu_half * 2.0f32 * (3.0f32 + sig2);

    assert!(
        (scratch.ffn_out[0] - expected).abs() < 1e-5,
        "moe_ffn_step numeric mismatch: expected {expected:.6}, got {:.6}",
        scratch.ffn_out[0]
    );
}

#[ignore = "requires local Qwen3.6 checkpoint"]
#[test]
fn test_qwen36_greedy_one_token_smoke() {
    let model_dir = std::env::var("LATTICE_QWEN36_MODEL_DIR")
        .expect("set LATTICE_QWEN36_MODEL_DIR to a Qwen3.6 checkpoint directory");
    let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
        .expect("Qwen3.6 MoE model should load successfully");
    let cfg = crate::model::qwen35_config::GenerateConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        ..Default::default()
    };
    let out = model.generate("Hello", &cfg).expect("generate succeeds");
    assert_eq!(
        out.prompt_tokens + out.generated_tokens,
        out.token_ids.len() + out.prompt_tokens
    );
}

// ---------------------------------------------------------------------------
// LoRA serving integration tests
//
// These verify the LoRA *serving* path: that `Qwen35Model::forward_step` invokes
// the `LoraHook` at every adapted projection (`q/k/v/o_proj`, `gate/up/down_proj`,
// and the GatedDeltaNet `in_proj_*`/`out_proj`) with buffer shapes matching the
// trait contract, and that an active adapter deterministically changes the
// output logits. The synthetic-model builder mirrors the one in
// `forward/batch_prefill.rs`'s test module — kept local rather than shared to
// avoid a cross-module test-support dependency.
// ---------------------------------------------------------------------------
mod lora_serving {
    use super::super::*;
    use crate::attention::gdn::{GatedDeltaNetState, GatedDeltaNetWeights};
    use crate::lora_hook::{LoraHook, NoopLoraHook};
    use crate::model::qwen35_config::{LayerType, Qwen35Config, compute_layer_types};
    use crate::rope::RopeTable;
    use crate::tokenizer::bpe::BpeTokenizer;
    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    /// Deterministic xorshift RNG → uniform noise in `[-scale, scale]`.
    fn rand_vec(state: &mut u64, len: usize, scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            out.push(((x >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale);
        }
        out
    }

    /// Minimal byte-level BPE tokenizer. `forward_step` never touches the
    /// tokenizer; the model struct just needs a value for the field.
    fn test_tokenizer() -> BpeTokenizer {
        let json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true },
  "post_processor": null,
  "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": { "<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, " ": 6 },
    "merges": []
  }
}"#;
        BpeTokenizer::from_tokenizer_json_str(json).expect("test tokenizer parses")
    }

    /// Compact 4-layer hybrid config: `layer_types = [linear, linear, linear, full]`,
    /// so a single `forward_step` exercises both attention paths plus the FFN.
    fn test_config() -> Qwen35Config {
        let num_hidden_layers = 4;
        let full_attention_interval = 4;
        Qwen35Config {
            hidden_size: 64,
            num_hidden_layers,
            vocab_size: 97,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: 4,
            linear_num_value_heads: Some(4),
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval,
            layer_types: compute_layer_types(num_hidden_layers, full_attention_interval),
            layer_mask: vec![true; num_hidden_layers],
            eos_token_id: 96,
            max_position_embeddings: 1024,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
        }
    }

    /// Build a synthetic `Qwen35Model` with deterministic random weights.
    fn build_model(cfg: Qwen35Config, seed: u64) -> Qwen35Model {
        let mut rng = seed | 1;
        let h = cfg.hidden_size;

        let embed_tokens = rand_vec(&mut rng, cfg.vocab_size * h, 0.02);
        let final_norm = rand_vec(&mut rng, h, 0.02);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_type in &cfg.layer_types {
            let common = CommonLayerWeights {
                input_layernorm: rand_vec(&mut rng, h, 0.02),
                post_attention_layernorm: rand_vec(&mut rng, h, 0.02),
                ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                    gate_proj: rand_vec(&mut rng, cfg.intermediate_size * h, 0.02),
                    up_proj: rand_vec(&mut rng, cfg.intermediate_size * h, 0.02),
                    down_proj: rand_vec(&mut rng, h * cfg.intermediate_size, 0.02),
                }),
            };
            let attn = match layer_type {
                LayerType::LinearAttention => {
                    let qkv_dim = cfg.linear_qkv_dim();
                    let output_dim = cfg.linear_output_dim();
                    let nh = cfg.linear_num_key_heads;
                    let kernel = cfg.linear_conv_kernel_dim;
                    AttentionWeights::Linear(GatedDeltaNetWeights {
                        in_proj_qkv: rand_vec(&mut rng, qkv_dim * h, 0.02),
                        in_proj_qkv_rows: qkv_dim,
                        in_proj_qkv_cols: h,
                        in_proj_z: rand_vec(&mut rng, output_dim * h, 0.02),
                        in_proj_z_rows: output_dim,
                        in_proj_z_cols: h,
                        in_proj_b: rand_vec(&mut rng, nh * h, 0.02),
                        in_proj_b_rows: nh,
                        in_proj_b_cols: h,
                        in_proj_a: rand_vec(&mut rng, nh * h, 0.02),
                        in_proj_a_rows: nh,
                        in_proj_a_cols: h,
                        a_log: rand_vec(&mut rng, nh, 0.02),
                        dt_bias: rand_vec(&mut rng, nh, 0.02),
                        conv1d_weight: rand_vec(&mut rng, qkv_dim * kernel, 0.02),
                        conv_dim: qkv_dim,
                        kernel_size: kernel,
                        norm_weight: rand_vec(&mut rng, output_dim, 0.02),
                        out_proj: rand_vec(&mut rng, h * output_dim, 0.02),
                        out_proj_rows: h,
                        out_proj_cols: output_dim,
                    })
                }
                LayerType::FullAttention => {
                    let q_dim = cfg.full_q_dim();
                    let kv_dim = cfg.full_kv_dim();
                    AttentionWeights::Full(FullAttentionLayerWeights {
                        q_proj: rand_vec(&mut rng, 2 * q_dim * h, 0.02),
                        k_proj: rand_vec(&mut rng, kv_dim * h, 0.02),
                        v_proj: rand_vec(&mut rng, kv_dim * h, 0.02),
                        o_proj: rand_vec(&mut rng, h * q_dim, 0.02),
                        q_norm: rand_vec(&mut rng, cfg.head_dim, 0.02),
                        k_norm: rand_vec(&mut rng, cfg.head_dim, 0.02),
                    })
                }
            };
            layers.push((attn, common));
        }

        let rope = RopeTable::new(
            cfg.rope_dim(),
            cfg.max_position_embeddings.min(8192),
            cfg.rope_theta,
        );

        Qwen35Model {
            config: cfg,
            weights: ModelWeights {
                embed_tokens,
                lm_head: None,
                final_norm,
                layers,
            },
            tokenizer: test_tokenizer(),
            rope,
            lora: Box::new(NoopLoraHook),
        }
    }

    /// Allocate the per-call mutable state `forward_step` needs.
    fn fresh_state(cfg: &Qwen35Config) -> (Vec<GatedDeltaNetState>, KvCache, ForwardScratch) {
        let gdn = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let kv = KvCache::new(cfg.num_full_attention_layers());
        (gdn, kv, ForwardScratch::new())
    }

    /// One recorded `LoraHook` call: `(layer_idx, module, x.len(), output.len())`.
    type CallLog = Vec<(usize, String, usize, usize)>;

    /// `LoraHook` that records every call into a shared [`CallLog`].
    struct SpyHook {
        calls: Arc<Mutex<CallLog>>,
    }
    impl LoraHook for SpyHook {
        fn apply(&self, layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]) {
            self.calls
                .lock()
                .unwrap()
                .push((layer_idx, module.to_string(), x.len(), output.len()));
        }
    }

    /// `LoraHook` that adds a constant delta to one specific `(layer, module)` output.
    struct DeltaHook {
        layer: usize,
        module: &'static str,
        delta: f32,
    }
    impl LoraHook for DeltaHook {
        fn apply(&self, layer_idx: usize, module: &str, _x: &[f32], output: &mut [f32]) {
            if layer_idx == self.layer && module == self.module {
                for o in output.iter_mut() {
                    *o += self.delta;
                }
            }
        }
    }

    #[test]
    fn lora_hook_invoked_at_every_adapted_projection() {
        let cfg = test_config();
        let mut model = build_model(cfg.clone(), 0xA11C_E5ED);

        let calls = Arc::new(Mutex::new(Vec::new()));
        model.set_lora(Box::new(SpyHook {
            calls: Arc::clone(&calls),
        }));

        let (mut gdn, mut kv, mut scratch) = fresh_state(&cfg);
        model.forward_step(1, 0, &mut gdn, &mut kv, &mut scratch);

        let recorded = calls.lock().unwrap();

        // Expected (layer, module) set, derived independently from the config.
        let mut expected: BTreeSet<(usize, String)> = BTreeSet::new();
        for (layer_idx, layer_type) in cfg.layer_types.iter().enumerate() {
            let attn_modules: &[&str] = match layer_type {
                LayerType::FullAttention => &["q_proj", "k_proj", "v_proj", "o_proj"],
                LayerType::LinearAttention => &[
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                ],
            };
            for m in attn_modules
                .iter()
                .chain(["gate_proj", "up_proj", "down_proj"].iter())
            {
                expected.insert((layer_idx, (*m).to_string()));
            }
        }

        let seen: BTreeSet<(usize, String)> = recorded
            .iter()
            .map(|(l, m, _, _)| (*l, m.clone()))
            .collect();
        assert_eq!(
            seen, expected,
            "LoRA hook must fire at exactly the adapted projections of every layer"
        );
        assert_eq!(
            recorded.len(),
            expected.len(),
            "each projection must invoke the hook exactly once per forward_step"
        );

        // Buffer shapes must match the trait contract: `x` is the projection
        // input, `output` is the projection output.
        let h = cfg.hidden_size;
        for (layer_idx, module, x_len, out_len) in recorded.iter() {
            let (exp_x, exp_out) = match module.as_str() {
                "q_proj" => (h, 2 * cfg.full_q_dim()),
                "k_proj" | "v_proj" => (h, cfg.full_kv_dim()),
                "o_proj" => (cfg.full_q_dim(), h),
                "gate_proj" | "up_proj" => (h, cfg.intermediate_size),
                "down_proj" => (cfg.intermediate_size, h),
                "in_proj_qkv" => (h, cfg.linear_qkv_dim()),
                "in_proj_z" => (h, cfg.linear_output_dim()),
                "in_proj_b" | "in_proj_a" => (h, cfg.linear_num_key_heads),
                "out_proj" => (cfg.linear_output_dim(), h),
                other => panic!("unexpected module name passed to LoraHook: {other}"),
            };
            assert_eq!(
                (*x_len, *out_len),
                (exp_x, exp_out),
                "layer {layer_idx} {module}: hook got (x={x_len}, out={out_len}), expected (x={exp_x}, out={exp_out})"
            );
        }
    }

    #[test]
    fn lora_adapter_changes_logits_deterministically() {
        let cfg = test_config();
        let mut model = build_model(cfg.clone(), 0x0D15_EA5E);

        let run = |model: &Qwen35Model| {
            let (mut gdn, mut kv, mut scratch) = fresh_state(&cfg);
            model.forward_step(2, 0, &mut gdn, &mut kv, &mut scratch);
            scratch.logits[..cfg.vocab_size].to_vec()
        };

        // Two NoopLoraHook runs must be bit-identical — this establishes that any
        // difference below is provably caused by the adapter, not nondeterminism.
        model.set_lora(Box::new(NoopLoraHook));
        let baseline_a = run(&model);
        let baseline_b = run(&model);
        assert_eq!(
            baseline_a, baseline_b,
            "forward_step must be deterministic under NoopLoraHook"
        );

        // An adapter targeting a real projection must move the logits.
        model.set_lora(Box::new(DeltaHook {
            layer: 3,
            module: "q_proj",
            delta: 0.5,
        }));
        let adapted = run(&model);
        assert_ne!(
            adapted, baseline_a,
            "an active LoRA adapter on a real projection must change the logits"
        );

        // An adapter keyed to a non-existent (layer, module) must be a no-op —
        // confirms forward_step only dispatches the hook to real projections.
        model.set_lora(Box::new(DeltaHook {
            layer: 999,
            module: "q_proj",
            delta: 0.5,
        }));
        let unmatched = run(&model);
        assert_eq!(
            unmatched, baseline_a,
            "an adapter keyed to a non-existent layer must not affect output"
        );
    }
}
