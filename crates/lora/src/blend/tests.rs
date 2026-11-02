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

#[test]
fn slice_bridges_match_iterator_and_preserve_validation_order() {
    let first = LoraLayer {
        a: vec![1.0, 2.0],
        b: vec![1.3, -0.5],
        rank: 1,
        d_in: 2,
        d_out: 2,
    };
    let second = LoraLayer {
        a: vec![3.0, 4.0, 5.0, 6.0],
        b: vec![0.25, 0.5, 1.5, 2.0],
        rank: 2,
        d_in: 2,
        d_out: 2,
    };
    let make_data = |layer: &LoraLayer| LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: layer.a.clone(),
        b: layer.b.clone(),
        rank: layer.rank,
        d_in: layer.d_in,
        d_out: layer.d_out,
    };
    let data_first = make_data(&first);
    let data_second = make_data(&second);
    let layers = [(&first, 1.1_f32 * 1.2), (&second, -0.7)];
    let data = [(&data_first, 1.1_f32 * 1.2), (&data_second, -0.7)];
    let reference = blend_effective_weights(
        "fixture",
        layers
            .iter()
            .map(|(layer, weight)| (layer.a.as_slice(), layer.b.as_slice(), layer.rank, *weight)),
        3,
        2,
        2,
    )
    .unwrap();
    for actual in [
        blend_layer_effective_weights("fixture", &layers, 3, 2, 2).unwrap(),
        blend_layer_data_effective_weights("fixture", &data, 3, 2, 2).unwrap(),
    ] {
        assert_eq!(actual.a, reference.a);
        assert_eq!(
            actual.b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference.b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    let invalid = LoraLayer {
        a: vec![],
        b: vec![],
        rank: 1,
        d_in: 2,
        d_out: 2,
    };
    let invalid_data = make_data(&invalid);
    for rank_total in [0, 1] {
        let reference = blend_effective_weights(
            "fixture",
            [(&invalid.a[..], &invalid.b[..], invalid.rank, 1.0)].into_iter(),
            rank_total,
            2,
            2,
        )
        .unwrap_err()
        .message();
        assert_eq!(
            blend_layer_effective_weights("fixture", &[(&invalid, 1.0)], rank_total, 2, 2)
                .unwrap_err()
                .message(),
            reference
        );
        assert_eq!(
            blend_layer_data_effective_weights(
                "fixture",
                &[(&invalid_data, 1.0)],
                rank_total,
                2,
                2,
            )
            .unwrap_err()
            .message(),
            reference
        );
    }
}

#[test]
fn slice_bridges_accept_zero_rank() {
    let layer = LoraLayer {
        a: vec![],
        b: vec![],
        rank: 0,
        d_in: 2,
        d_out: 2,
    };
    let data = LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: vec![],
        b: vec![],
        rank: 0,
        d_in: 2,
        d_out: 2,
    };
    let host = blend_layer_effective_weights("fixture", &[(&layer, 0.5)], 0, 2, 2).unwrap();
    let device = blend_layer_data_effective_weights("fixture", &[(&data, 0.5)], 0, 2, 2).unwrap();
    assert!(host.a.is_empty() && host.b.is_empty());
    assert!(device.a.is_empty() && device.b.is_empty());
}

#[test]
fn b_rows_match_original_scalar_loop_at_small_and_large_ranks() {
    for ranks in [&[0, 1, 2][..], &[8, 0, 16][..]] {
        let d_in = 3;
        let d_out = 5;
        let rank_total: usize = ranks.iter().sum();
        let layers: Vec<LoraLayer> = ranks
            .iter()
            .enumerate()
            .map(|(idx, &rank)| LoraLayer {
                a: (0..rank * d_in).map(|i| i as f32 - 2.0).collect(),
                b: (0..d_out * rank)
                    .map(|i| (i as f32 - 17.0) * (idx as f32 + 0.125))
                    .collect(),
                rank,
                d_in,
                d_out,
            })
            .collect();
        let entries: Vec<_> = layers
            .iter()
            .enumerate()
            .map(|(i, layer)| (layer, [0.0, -0.7, 1.1][i]))
            .collect();
        let mut expected = vec![0.0_f32; d_out * rank_total];
        let mut col_offset = 0;
        for (layer, weight) in &entries {
            for row in 0..d_out {
                for col in 0..layer.rank {
                    expected[row * rank_total + col_offset + col] =
                        *weight * layer.b[row * layer.rank + col];
                }
            }
            col_offset += layer.rank;
        }
        let actual =
            blend_layer_effective_weights("fixture", &entries, rank_total, d_in, d_out).unwrap();
        assert_eq!(
            actual.b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
