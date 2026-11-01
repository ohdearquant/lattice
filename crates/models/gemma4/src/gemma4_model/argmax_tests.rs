use super::argmax;

#[test]
fn greedy_argmax_keeps_first_tie_and_skips_nan() {
    assert_eq!(argmax(&[0.0, 2.0, 2.0]), 1);
    assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), 2);
    assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
    assert_eq!(argmax(&[]), 0);
}

#[test]
fn greedy_argmax_matches_previous_loop() {
    fn previous(logits: &[f32]) -> u32 {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx as u32
    }
    let cases: &[&[f32]] = &[
        &[],
        &[1.0, 2.0, 2.0],
        &[1.0, f32::NAN, 2.0],
        &[f32::NAN, f32::NAN],
        &[f32::NEG_INFINITY, f32::NEG_INFINITY],
    ];
    for logits in cases {
        assert_eq!(argmax(logits), previous(logits));
    }
}
