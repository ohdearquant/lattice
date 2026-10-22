use super::*;
use approx::assert_relative_eq;

#[test]
fn test_mean_pool_respects_attention_mask() {
    let hidden_states = vec![
        1.0, 2.0, // token 0
        3.0, 4.0, // token 1
        100.0, 200.0, // padding token
    ];
    let attention_mask = vec![1, 1, 0];
    let pooled = mean_pool(&hidden_states, &attention_mask, 3, 2);
    assert_eq!(pooled, vec![2.0, 3.0]);
}

#[test]
fn test_mean_pool_all_masked() {
    let hidden_states = vec![1.0, 2.0, 3.0, 4.0];
    let attention_mask = vec![0, 0];
    let pooled = mean_pool(&hidden_states, &attention_mask, 2, 2);
    assert_eq!(pooled, vec![0.0, 0.0]);
}

#[test]
fn test_mean_pool_simd_matches_scalar() {
    // Use hidden_size=384 (BGE-small) to exercise SIMD paths.
    let hidden_size = 384;
    let seq_len = 32;
    let hidden_states = make_deterministic_vec(seq_len * hidden_size, 0xB001);
    // Mix of 1s and 0s in the mask
    let attention_mask: Vec<u32> = (0..seq_len).map(|i| if i < 24 { 1 } else { 0 }).collect();

    let simd_result = mean_pool(&hidden_states, &attention_mask, seq_len, hidden_size);
    let scalar_result = mean_pool_scalar(&hidden_states, &attention_mask, seq_len, hidden_size);

    assert_eq!(simd_result.len(), scalar_result.len());
    for i in 0..hidden_size {
        assert_relative_eq!(simd_result[i], scalar_result[i], epsilon = 1e-5);
    }
}

#[test]
fn test_l2_normalize_unit_length() {
    let mut v = vec![3.0, 4.0];
    l2_normalize(&mut v);
    assert_relative_eq!(v[0], 0.6, epsilon = 1e-6);
    assert_relative_eq!(v[1], 0.8, epsilon = 1e-6);
    let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
}

#[test]
fn test_l2_normalize_zero_vector() {
    let mut v = vec![0.0; 384];
    l2_normalize(&mut v);
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn test_l2_normalize_simd_matches_scalar() {
    // Use 384-dim vector to exercise SIMD paths.
    let mut simd_v = make_deterministic_vec(384, 0xA2B0);
    let mut scalar_v = simd_v.clone();

    l2_normalize(&mut simd_v);
    l2_normalize_scalar(&mut scalar_v);

    for i in 0..384 {
        assert_relative_eq!(simd_v[i], scalar_v[i], epsilon = 1e-5);
    }

    // Verify unit norm
    let norm: f32 = simd_v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
}

// --- cls_pool tests ---

#[test]
fn test_cls_pool_extracts_first_token() {
    // seq_len=2, hidden_size=3: CLS vector is the first hidden_size floats
    let hidden_states = vec![
        0.1f32, 0.2, 0.3, // token 0 ([CLS])
        0.4, 0.5, 0.6, // token 1
    ];
    let cls = cls_pool(&hidden_states, 2, 3);
    assert_eq!(cls, vec![0.1f32, 0.2, 0.3]);
    // non-CLS tokens must not appear in output
    assert_eq!(cls.len(), 3);
}

#[test]
fn test_cls_pool_single_token() {
    let hidden_states = vec![1.0f32, 2.0, 3.0];
    let cls = cls_pool(&hidden_states, 1, 3);
    assert_eq!(cls, vec![1.0f32, 2.0, 3.0]);
}

#[test]
fn test_cls_pool_longer_sequence_ignores_later_tokens() {
    // Even with many tokens, only first hidden_size values are returned
    let hidden_size = 2;
    let seq_len = 4;
    let hidden_states: Vec<f32> = (0..seq_len * hidden_size).map(|i| i as f32 * 0.1).collect();
    let cls = cls_pool(&hidden_states, seq_len, hidden_size);
    assert_eq!(cls, vec![0.0f32, 0.1]);
}

#[test]
#[should_panic(expected = "seq_len must be non-zero")]
fn test_cls_pool_zero_seq_len_panics() {
    cls_pool(&[], 0, 4);
}

/// Deterministic pseudo-random vector for reproducible tests.
fn make_deterministic_vec(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed ^ (len as u32).wrapping_mul(0x9E37_79B9);
    if state == 0 {
        state = 0xA341_316C;
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let unit = state as f32 / u32::MAX as f32;
        out.push(unit * 0.04 - 0.02);
    }
    out
}

#[test]
#[should_panic(expected = "hidden_states length must match seq_len * hidden_size")]
fn last_token_pool_rejects_short_hidden_states() {
    // seq_len=2, hidden_size=4 needs 8 elements; pass 4.
    let _ = last_token_pool(&[0.0; 4], &[1, 1], 2, 4);
}

#[test]
#[should_panic(expected = "seq_len must be non-zero")]
fn last_token_pool_rejects_zero_seq_len() {
    let _ = last_token_pool(&[], &[], 0, 4);
}
