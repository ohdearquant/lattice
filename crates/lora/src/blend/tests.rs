use super::*;

#[test]
fn concatenates_unequal_ranks_with_effective_weights() {
    let a1 = [1.0, 2.0];
    let b1 = [1.3, -0.5];
    let a2 = [3.0, 4.0, 5.0, 6.0];
    let b2 = [0.25, 0.5, 1.5, 2.0];
    let effective = 1.1_f32 * 1.2_f32;
    let inputs = [
        (&a1[..], &b1[..], 1, effective),
        (&a2[..], &b2[..], 2, -0.7),
    ];
    let out = blend_effective_weights("fixture", inputs.into_iter(), 3, 2, 2).unwrap();
    assert_eq!(out.a, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(out.b[0].to_bits(), 0x3fdba5e3);
    assert_eq!(out.b[1].to_bits(), (-0.7_f32 * 0.25).to_bits());
    assert_eq!(out.b[2].to_bits(), (-0.7_f32 * 0.5).to_bits());
    assert_eq!(out.b[3].to_bits(), (effective * -0.5).to_bits());
    assert_eq!(out.b[4].to_bits(), (-0.7_f32 * 1.5).to_bits());
    assert_eq!(out.b[5].to_bits(), (-0.7_f32 * 2.0).to_bits());
}

#[test]
fn rank_zero_empty_factors_are_valid_but_positive_rank_placeholders_are_not() {
    let entries = [(&[][..], &[][..], 0, 0.5)];
    let out = blend_effective_weights("fixture", entries.into_iter(), 0, 2, 2).unwrap();
    assert!(out.a.is_empty());
    assert!(out.b.is_empty());
    let entries = [(&[][..], &[][..], 1, 0.5)];
    let err = blend_effective_weights("fixture", entries.into_iter(), 1, 2, 2).unwrap_err();
    assert_eq!(
        err.message(),
        "fixture: entry 0 A slice length 0 does not match rank*d_in=1*2=2"
    );
}
