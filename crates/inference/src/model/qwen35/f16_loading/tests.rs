use crate::model::qwen35_config::Qwen35Config;

fn estimate_f32_model_bytes(cfg: &Qwen35Config) -> usize {
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let value_heads = cfg.linear_num_value_heads();
    let kernel_size = cfg.linear_conv_kernel_dim;

    let mut bytes = 0usize;

    bytes += cfg.vocab_size * hidden * 4;
    bytes += hidden * 4;

    for i in 0..cfg.num_hidden_layers {
        bytes += hidden * 4;
        bytes += hidden * 4;
        bytes += intermediate * hidden * 4;
        bytes += intermediate * hidden * 4;
        bytes += hidden * intermediate * 4;

        if cfg.is_full_attention(i) {
            bytes += (2 * q_dim * hidden) * 4;
            bytes += (kv_dim * hidden) * 4;
            bytes += (kv_dim * hidden) * 4;
            bytes += (hidden * q_dim) * 4;
            bytes += cfg.head_dim * 4;
            bytes += cfg.head_dim * 4;
        } else {
            bytes += (qkv_dim * hidden) * 4;
            bytes += (output_dim * hidden) * 4;
            bytes += (value_heads * hidden) * 4;
            bytes += (value_heads * hidden) * 4;
            bytes += value_heads * 4;
            bytes += value_heads * 4;
            bytes += (qkv_dim * kernel_size) * 4;
            bytes += output_dim * 4;
            bytes += (hidden * output_dim) * 4;
        }
    }

    bytes
}

fn estimate_f16_model_bytes(cfg: &Qwen35Config) -> usize {
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let value_heads = cfg.linear_num_value_heads();
    let kernel_size = cfg.linear_conv_kernel_dim;

    let mut bytes = 0usize;

    bytes += cfg.vocab_size * hidden * 2;
    bytes += hidden * 4;

    for i in 0..cfg.num_hidden_layers {
        bytes += hidden * 4;
        bytes += hidden * 4;
        bytes += intermediate * hidden * 2;
        bytes += intermediate * hidden * 2;
        bytes += hidden * intermediate * 2;

        if cfg.is_full_attention(i) {
            bytes += (2 * q_dim * hidden) * 2;
            bytes += (kv_dim * hidden) * 2;
            bytes += (kv_dim * hidden) * 2;
            bytes += (hidden * q_dim) * 2;
            bytes += cfg.head_dim * 4;
            bytes += cfg.head_dim * 4;
        } else {
            bytes += (qkv_dim * hidden) * 2;
            bytes += (output_dim * hidden) * 2;
            bytes += (value_heads * hidden) * 2;
            bytes += (value_heads * hidden) * 2;
            bytes += value_heads * 4;
            bytes += value_heads * 4;
            bytes += (qkv_dim * kernel_size) * 4;
            bytes += output_dim * 4;
            bytes += (hidden * output_dim) * 2;
        }
    }

    bytes
}

#[test]
fn test_f16_memory_savings() {
    let cfg = Qwen35Config::qwen35_2b();
    let f32_bytes = estimate_f32_model_bytes(&cfg);
    let f16_bytes = estimate_f16_model_bytes(&cfg);

    let ratio = f16_bytes as f64 / f32_bytes as f64;
    assert!(
        (ratio - 0.5).abs() < 0.05,
        "f16/f32 ratio should be ~0.5, got {ratio}"
    );
}
