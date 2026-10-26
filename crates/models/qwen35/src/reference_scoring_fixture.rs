use crate::model::qwen35::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
    FullAttentionLayerWeights, ModelWeights,
};
use lattice_arch::qwen35_config::{LayerType, Qwen35Config};

pub(crate) fn tiny_hybrid_fixture() -> (Qwen35Config, ModelWeights) {
    use lattice_compute::attention::gdn::GatedDeltaNetWeights;
    // Metal FlashAttention decode requires head_dim=256 (see
    // `validate_flash_decode_shape`). Hidden size = num_attention_heads * head_dim
    // so the Q/K/V projections fit the per-head layout used by the decode kernel.
    let hidden = 512usize;
    let intermediate = 64usize;
    let vocab = 32usize;
    let num_kh = 1usize;
    let kd = 16usize;
    // vd * num_kh = GDN out_dim = the contraction dim K of the out_proj Q8
    // GEMM, which the GEMV/GEMM kernel requires to be a multiple of 32 (real
    // configs use vd=128). 16 made the GDN-only GPU tests crash on real Apple
    // hardware (and silently skip in paravirtual CI where no Metal device exists).
    let vd = 32usize;
    let qkv_dim = num_kh * kd * 2 + num_kh * vd; // Q+K+V
    let out_dim = num_kh * vd;
    let ks = 4usize;
    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 4,
        vocab_size: vocab,
        intermediate_size: intermediate,
        rms_norm_eps: 1e-6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        rope_parameters: None,
        linear_num_key_heads: num_kh,
        linear_num_value_heads: Some(num_kh),
        linear_key_head_dim: kd,
        linear_value_head_dim: vd,
        linear_conv_kernel_dim: ks,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        full_attention_interval: 4,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ],
        layer_mask: vec![true; 4],
        eos_token_id: (vocab - 1) as u32,
        max_position_embeddings: 128,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let make_common = || CommonLayerWeights {
        input_layernorm: vec![1.0; hidden],
        post_attention_layernorm: vec![1.0; hidden],
        ffn: FeedForwardWeights::Dense(DenseFfnWeights {
            gate_proj: vec![0.0; intermediate * hidden],
            up_proj: vec![0.0; intermediate * hidden],
            down_proj: vec![0.0; hidden * intermediate],
        }),
    };

    let make_gdn = || {
        AttentionWeights::Linear(GatedDeltaNetWeights {
            in_proj_qkv: vec![0.0; qkv_dim * hidden],
            in_proj_qkv_rows: qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z: vec![0.0; out_dim * hidden],
            in_proj_z_rows: out_dim,
            in_proj_z_cols: hidden,
            in_proj_b: vec![0.0; num_kh * hidden],
            in_proj_b_rows: num_kh,
            in_proj_b_cols: hidden,
            in_proj_a: vec![0.0; num_kh * hidden],
            in_proj_a_rows: num_kh,
            in_proj_a_cols: hidden,
            a_log: vec![-1.0; num_kh],
            dt_bias: vec![0.0; num_kh],
            conv1d_weight: vec![0.0; qkv_dim * ks],
            conv_dim: qkv_dim,
            kernel_size: ks,
            norm_weight: vec![1.0; out_dim],
            out_proj: vec![0.0; hidden * out_dim],
            out_proj_rows: hidden,
            out_proj_cols: out_dim,
        })
    };

    let make_full = || {
        AttentionWeights::Full(FullAttentionLayerWeights {
            q_proj: vec![0.0; 2 * cfg.full_q_dim() * hidden],
            k_proj: vec![0.0; cfg.full_kv_dim() * hidden],
            v_proj: vec![0.0; cfg.full_kv_dim() * hidden],
            o_proj: vec![0.0; hidden * cfg.full_q_dim()],
            q_norm: vec![1.0; cfg.head_dim],
            k_norm: vec![1.0; cfg.head_dim],
        })
    };

    let mut embed = vec![0.0f32; vocab * hidden];
    for tok in 0..vocab {
        embed[tok * hidden] = if tok % 2 == 0 { 1.0 } else { -1.0 };
    }

    let weights = ModelWeights {
        embed_tokens: embed,
        lm_head: None,
        final_norm: vec![1.0; hidden],
        layers: vec![
            (make_gdn(), make_common()),
            (make_gdn(), make_common()),
            (make_gdn(), make_common()),
            (make_full(), make_common()),
        ],
    };
    (cfg, weights)
}

pub(crate) fn option_scoring_fixture() -> (Qwen35Config, ModelWeights) {
    let (cfg, mut weights) = tiny_hybrid_fixture();
    let hidden = cfg.hidden_size;
    let q_dim = cfg.full_q_dim();
    for token in 0..cfg.vocab_size {
        let row = &mut weights.embed_tokens[token * hidden..(token + 1) * hidden];
        row[0] = (token % 5) as f32 - 2.0;
        row[1] = (token % 3) as f32 - 1.0;
    }
    let Some((AttentionWeights::Full(full), _)) = weights.layers.last_mut() else {
        panic!("the last fixture layer is full attention");
    };
    full.v_proj[0] = 1.0;
    // o_proj is [hidden, q_dim] row-major: row 1, column 0 writes head 0 channel 0 into hidden[1].
    full.o_proj[q_dim] = 1.0;
    (cfg, weights)
}
