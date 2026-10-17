use super::*;

#[test]
fn multimodal_input_valid() {
    let input = MultimodalInput {
        patch_embeddings: vec![0.0f32; 196 * 2048],
        raw_patches: 784,
        visual_tokens: 196,
        d_model: 2048,
        text_tokens: vec![1, 2, 3],
    };
    input.validate().expect("valid input");
    assert_eq!(input.total_sequence_len(), 199); // 196 + 3
}

#[test]
fn multimodal_input_wrong_embed_len() {
    let input = MultimodalInput {
        patch_embeddings: vec![0.0f32; 100], // wrong: should be 196*2048
        raw_patches: 784,
        visual_tokens: 196,
        d_model: 2048,
        text_tokens: vec![],
    };
    assert!(input.validate().is_err());
}

#[test]
fn multimodal_input_zero_d_model() {
    let input = MultimodalInput {
        patch_embeddings: vec![],
        raw_patches: 784,
        visual_tokens: 196,
        d_model: 0,
        text_tokens: vec![],
    };
    assert!(input.validate().is_err());
}

#[test]
fn multimodal_input_zero_visual_tokens() {
    let input = MultimodalInput {
        patch_embeddings: vec![],
        raw_patches: 0,
        visual_tokens: 0,
        d_model: 2048,
        text_tokens: vec![1, 2],
    };
    assert!(input.validate().is_err());
}

#[test]
fn total_sequence_len() {
    let input = MultimodalInput {
        patch_embeddings: vec![0.0f32; 196 * 512],
        raw_patches: 784,
        visual_tokens: 196,
        d_model: 512,
        text_tokens: vec![10, 20, 30, 40],
    };
    assert_eq!(input.total_sequence_len(), 200);
}
