use super::*;

#[test]
fn add_bias_gelu_accepts_valid_lengths() {
    let dim = 4;
    let bias = vec![0.1, -0.2, 0.3, -0.4];
    let mut x = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0, 1.5]; // 2 rows × dim
    add_bias_gelu(&mut x, &bias, dim);
    assert_eq!(x.len(), 8);
    assert!(x.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "bias length must equal dim")]
fn add_bias_gelu_rejects_short_bias() {
    // dim=4 needs bias.len()==4; a short bias would OOB the unsafe SIMD kernel.
    let mut x = vec![0.0; 8];
    let bias = vec![0.0; 3];
    add_bias_gelu(&mut x, &bias, 4);
}

#[test]
#[should_panic(expected = "bias length must equal dim")]
fn add_bias_rejects_short_bias() {
    // A short bias previously truncated silently (chunks_exact + zip),
    // leaving the trailing lanes unbiased. Fail closed in release now.
    let mut x = vec![0.0; 8];
    let bias = vec![0.0; 3];
    add_bias(&mut x, &bias, 4);
}

#[test]
#[should_panic(expected = "multiple of dim")]
fn add_bias_rejects_ragged_x() {
    // x.len() not a multiple of dim previously dropped the remainder row.
    let mut x = vec![0.0; 7];
    let bias = vec![0.0; 4];
    add_bias(&mut x, &bias, 4);
}

#[test]
fn add_bias_accepts_valid_lengths() {
    let dim = 4;
    let bias = vec![0.1, -0.2, 0.3, -0.4];
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 rows × dim
    add_bias(&mut x, &bias, dim);
    assert_eq!(x, vec![1.1, 1.8, 3.3, 3.6, 5.1, 5.8, 7.3, 7.6]);
}
