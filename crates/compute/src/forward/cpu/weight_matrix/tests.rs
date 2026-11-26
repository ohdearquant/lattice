use super::*;

fn bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}

#[test]
fn weight_matrix_reports_length_store_and_held_bytes() {
    let f = WeightMatrix::from(vec![1.0f32; 6]);
    let b = WeightMatrix::Bf16(bits(&[1.0; 6]));
    assert_eq!((f.len(), f.is_bf16(), f.held_bytes()), (6, false, 24));
    assert_eq!((b.len(), b.is_bf16(), b.held_bytes()), (6, true, 12));
    assert!(WeightMatrix::default().is_empty());
}

#[test]
fn weight_matrix_reads_the_same_values_from_either_store() {
    let values = [1.0f32, -2.5, 0.375, 96.0, -0.0078125, 3.0];
    let f = WeightMatrix::from(values.to_vec());
    let b = WeightMatrix::Bf16(bits(&values));
    assert!(matches!(f.to_f32(), Cow::Borrowed(_)));
    assert_eq!(f.to_f32().as_ref(), values.as_slice());
    assert_eq!(b.to_f32().as_ref(), values.as_slice());
    assert_eq!(f.as_f32(), Some(values.as_slice()));
    assert_eq!(b.as_f32(), None);

    for row in 0..2 {
        let (mut from_f, mut from_b) = ([0.0f32; 3], [0.0f32; 3]);
        f.copy_row_into(row, &mut from_f);
        b.copy_row_into(row, &mut from_b);
        assert_eq!(from_f, values[row * 3..row * 3 + 3]);
        assert_eq!(from_b, values[row * 3..row * 3 + 3]);
    }
}

#[test]
fn weight_matrix_multiplies_either_store_and_leaves_the_scratch_alone_at_f32() {
    let (m, k, n) = (2, 3, 2);
    let a = [1.0f32, 2.0, 3.0, -1.0, 0.5, 0.25];
    let w = [0.5f32, -1.0, 2.0, 4.0, 0.25, -0.5];
    let mut want = [0.0f32; 4];
    matmul_bt(&a, &w, &mut want, m, k, n);

    let mut scratch = Bf16MatmulScratch::new();
    let mut got_f = [f32::NAN; 4];
    WeightMatrix::from(w.to_vec()).matmul_bt(&a, &mut got_f, m, k, n, &mut scratch);
    assert_eq!(got_f, want);
    assert_eq!(scratch.held_elements(), 0);

    let mut got_b = [f32::NAN; 4];
    WeightMatrix::Bf16(bits(&w)).matmul_bt(&a, &mut got_b, m, k, n, &mut scratch);
    assert_eq!(got_b, want);
    assert!(scratch.held_elements() > 0);
}

#[test]
fn weight_matrix_writes_only_through_the_f32_store() {
    let mut f = WeightMatrix::from(vec![0.0f32; 2]);
    if let Some(values) = f.as_f32_mut() {
        values[1] = 7.0;
    }
    assert_eq!(f.as_f32(), Some([0.0, 7.0].as_slice()));
    assert!(WeightMatrix::Bf16(vec![0; 2]).as_f32_mut().is_none());
}

#[test]
#[should_panic]
fn weight_matrix_rejects_a_row_past_the_end() {
    let mut dst = [0.0f32; 3];
    WeightMatrix::Bf16(vec![0; 6]).copy_row_into(2, &mut dst);
}
