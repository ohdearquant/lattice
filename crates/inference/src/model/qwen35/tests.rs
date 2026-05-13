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
