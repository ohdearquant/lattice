use super::*;

#[test]
fn qwen35_target_preserves_dense_preset_layout() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let target = qwen35_rotation_target(&cfg);
    assert_eq!(target.num_hidden_layers(), 24);
    assert_eq!(target.num_attention_heads(), 8);
    assert_eq!(target.intermediate_size(), 3584);
    assert_eq!(target.full_attention_layers(), &[3, 7, 11, 15, 19, 23]);
    assert!(target.dense_mlp());
    for idx in 0..24 {
        assert_eq!(
            target.is_full_attention(idx),
            [3, 7, 11, 15, 19, 23].contains(&idx)
        );
        assert_eq!(
            target.layer_tensor_prefix(idx),
            format!("model.language_model.layers.{idx}")
        );
    }
}

#[test]
fn qwen35_target_preserves_expert_preset_layout() {
    let cfg = Qwen35Config::qwen36_35b_a3b();
    let target = qwen35_rotation_target(&cfg);
    assert_eq!(target.num_hidden_layers(), 40);
    assert_eq!(target.num_attention_heads(), 16);
    assert_eq!(target.intermediate_size(), 6144);
    assert_eq!(
        target.full_attention_layers(),
        &[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    );
    assert!(!target.dense_mlp());
    for idx in 0..40 {
        assert_eq!(
            target.is_full_attention(idx),
            [3, 7, 11, 15, 19, 23, 27, 31, 35, 39].contains(&idx)
        );
        assert_eq!(
            target.layer_tensor_prefix(idx),
            format!("model.language_model.layers.{idx}")
        );
    }
}
