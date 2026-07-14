//! Test-only tiny zero-weight `Qwen35Model` construction, gated behind
//! `cfg(any(test, feature = "test-utils"))` (ADR-080 C2, #816).
//!
//! `crates/inference/src/model/qwen35/generation.rs`'s `#[cfg(test)] mod
//! tests` builds an all-zero-weight tiny model for its own
//! mutation-sensitive `StopReason` tests, reaching this module through the
//! plain `cfg(test)` arm of the gate below. `crates/inference/src/bin/lattice.rs`'s
//! test module is a *separate* compilation unit that links against this
//! crate as an ordinary (non-test) dependency, so it cannot see anything
//! gated on `cfg(test)` alone; it reaches this same module through the
//! `feature = "test-utils"` arm instead. The HTTP-level disconnect test that
//! `lattice.rs`'s `chat_completions` handler needs a real (but tiny, fast,
//! deterministic) `ModelBackend::Cpu` to exercise the actual streaming
//! composition end to end, so this function is the one library-side seam
//! that makes it constructible from the bin's test module too -- via
//! `--features test-utils`, never a default feature.

use super::Qwen35Model;
use super::weights::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
    FullAttentionLayerWeights, ModelWeights,
};
use crate::attention::gdn::GatedDeltaNetWeights;
use crate::lora_hook::NoopLoraHook;
use crate::model::qwen35_config::{LayerType, Qwen35Config, compute_layer_types};
use crate::rope::RopeTable;
use crate::tokenizer::bpe::BpeTokenizer;

const DEFAULT_TINY_TOK_JSON: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5," ":6},"merges":[]}
}"#;

/// An all-zero-weight, 4-layer, 64-hidden-dim `Qwen35Model` with a 7-token
/// vocab. All-zero weights make every forward pass emit all-zero logits, so
/// greedy sampling deterministically always picks token 0 (`"<unk>"`) --
/// useful for tests that need *some* real, non-empty, deterministic
/// generation output without loading real model weights.
pub fn tiny_zero_model() -> Qwen35Model {
    tiny_zero_model_with_tokenizer(DEFAULT_TINY_TOK_JSON)
}

/// Same as [`tiny_zero_model`], with a caller-supplied tokenizer JSON (e.g.
/// to add a special token like `</think>` at a known id).
pub fn tiny_zero_model_with_tokenizer(tok_json: &str) -> Qwen35Model {
    const H: usize = 64;
    const VOCAB: usize = 97;
    const I: usize = 128;
    const NUM_LAYERS: usize = 4;
    const FULL_INTERVAL: usize = 4;
    const HEAD_DIM: usize = 16;
    const LINEAR_KH: usize = 4;
    const KERNEL: usize = 4;

    let cfg = Qwen35Config {
        hidden_size: H,
        num_hidden_layers: NUM_LAYERS,
        vocab_size: VOCAB,
        intermediate_size: I,
        rms_norm_eps: 1e-6,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: HEAD_DIM,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        rope_parameters: None,
        linear_num_key_heads: LINEAR_KH,
        linear_num_value_heads: Some(LINEAR_KH),
        linear_key_head_dim: HEAD_DIM,
        linear_value_head_dim: HEAD_DIM,
        linear_conv_kernel_dim: KERNEL,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: FULL_INTERVAL,
        layer_types: compute_layer_types(NUM_LAYERS, FULL_INTERVAL),
        layer_mask: vec![true; NUM_LAYERS],
        eos_token_id: (VOCAB - 1) as u32,
        max_position_embeddings: 1024,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
    };

    let z = |len: usize| vec![0.0_f32; len];
    let qkv_dim = cfg.linear_qkv_dim();
    let out_dim = cfg.linear_output_dim();
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();

    let mut layers = Vec::with_capacity(NUM_LAYERS);
    for lt in &cfg.layer_types {
        let common = CommonLayerWeights {
            input_layernorm: z(H),
            post_attention_layernorm: z(H),
            ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                gate_proj: z(I * H),
                up_proj: z(I * H),
                down_proj: z(H * I),
            }),
        };
        let attn = match lt {
            LayerType::LinearAttention => AttentionWeights::Linear(GatedDeltaNetWeights {
                in_proj_qkv: z(qkv_dim * H),
                in_proj_qkv_rows: qkv_dim,
                in_proj_qkv_cols: H,
                in_proj_z: z(out_dim * H),
                in_proj_z_rows: out_dim,
                in_proj_z_cols: H,
                in_proj_b: z(LINEAR_KH * H),
                in_proj_b_rows: LINEAR_KH,
                in_proj_b_cols: H,
                in_proj_a: z(LINEAR_KH * H),
                in_proj_a_rows: LINEAR_KH,
                in_proj_a_cols: H,
                a_log: z(LINEAR_KH),
                dt_bias: z(LINEAR_KH),
                conv1d_weight: z(qkv_dim * KERNEL),
                conv_dim: qkv_dim,
                kernel_size: KERNEL,
                norm_weight: z(out_dim),
                out_proj: z(H * out_dim),
                out_proj_rows: H,
                out_proj_cols: out_dim,
            }),
            LayerType::FullAttention => AttentionWeights::Full(FullAttentionLayerWeights {
                q_proj: z(2 * q_dim * H),
                k_proj: z(kv_dim * H),
                v_proj: z(kv_dim * H),
                o_proj: z(H * q_dim),
                q_norm: z(HEAD_DIM),
                k_norm: z(HEAD_DIM),
            }),
        };
        layers.push((attn, common));
    }

    let tokenizer = BpeTokenizer::from_tokenizer_json_str(tok_json).expect("test tokenizer parses");
    let rope = RopeTable::new(
        cfg.rope_dim(),
        cfg.max_position_embeddings.min(8192),
        cfg.rope_theta,
    );

    Qwen35Model {
        config: cfg.clone(),
        weights: ModelWeights {
            embed_tokens: z(VOCAB * H),
            lm_head: None,
            final_norm: z(H),
            layers,
        },
        tokenizer,
        rope,
        lora: Box::new(NoopLoraHook),
    }
}
