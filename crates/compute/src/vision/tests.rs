use super::*;

#[test]
fn vision_layer_norm_constant_rows_preserve_bias() {
    let constant = f32::from_bits(0x4280_0003);
    for hidden in [16, 1, 31, 896] {
        let weight: Vec<_> = (0..hidden).map(|i| 1.0 + (i % 5) as f32 * 0.25).collect();
        let bias: Vec<_> = (0..hidden).map(|i| 0.25 + (i % 7) as f32 * 0.125).collect();
        for value in [constant, -constant] {
            let mut output = vec![value; hidden];
            layer_norm(&mut output, &weight, &bias, 1e-5);
            for (index, (&got, &want)) in output.iter().zip(&bias).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "vision constant bias hidden={hidden} index={index}"
                );
            }
        }
    }
}

#[test]
fn vision_layer_norm_rounded_mean_matches_analytic() {
    for hidden in [896, 898] {
        let weight = vec![1.0; hidden];
        let bias = vec![0.0; hidden];
        for low in [33_554_448.0f32, 33_554_452.0] {
            let mut output: Vec<_> = (0..hidden).map(|i| low + (i % 2) as f32 * 4.0).collect();
            layer_norm(&mut output, &weight, &bias, 1e-5);
            for (index, &got) in output.iter().enumerate() {
                let want = if index % 2 == 0 { -1.0 } else { 1.0 };
                assert!(
                    (got - want).abs() <= 1e-5,
                    "vision rounded mean hidden={hidden} index={index}: got={got}, want={want}"
                );
            }
        }
    }
}

#[test]
fn vision_layer_norm_ordinary_affine_matches_reference() {
    for hidden in [16, 31, 896, 898] {
        let input: Vec<_> = (0..hidden)
            .map(|i| (((i * 13) % 31) as f32 - 15.0) * 0.125)
            .collect();
        let weight: Vec<_> = (0..hidden).map(|i| (i % 9) as f32 * 0.25 - 0.75).collect();
        let bias: Vec<_> = (0..hidden).map(|i| (i % 7) as f32 * 0.125 - 0.25).collect();
        let mean = input.iter().map(|&value| f64::from(value)).sum::<f64>() / hidden as f64;
        let variance = input
            .iter()
            .map(|&value| {
                let centered = f64::from(value) - mean;
                centered * centered
            })
            .sum::<f64>()
            / hidden as f64;
        let mut output = input.clone();
        layer_norm(&mut output, &weight, &bias, 1e-5);
        for (index, (&value, &got)) in input.iter().zip(&output).enumerate() {
            let want = (f64::from(value) - mean) / (variance + f64::from(1e-5f32)).sqrt()
                * f64::from(weight[index])
                + f64::from(bias[index]);
            let limit = 1e-4 * f64::from(got).abs().max(want.abs()) + 1e-5;
            assert!(
                got.is_finite() && (f64::from(got) - want).abs() <= limit,
                "vision ordinary hidden={hidden} index={index}: got={got}, reference={want}"
            );
        }
        eprintln!(
            "vision ordinary arch={} hidden={hidden} bits={:?}",
            std::env::consts::ARCH,
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn vision_layer_norm_correction_preserves_infinite_variance() {
    let low = f32::from_bits(0x7180_0000);
    let mut output: Vec<_> = (0..64)
        .map(|i| if i % 2 == 0 { low } else { low.next_up() })
        .collect();
    let weight = vec![1.0; output.len()];
    let bias = vec![0.0; output.len()];
    layer_norm(&mut output, &weight, &bias, 1e-5);
    assert!(
        output.iter().all(|&value| value == 0.0),
        "vision correction must preserve infinite variance"
    );
}

// ADR-080 C1 / #741: `softmax_inplace` previously had NO fail-closed guard
// at all — a NaN or `+inf` logit divided every element by a NaN/Inf `sum`,
// propagating NaN through the whole row instead of zeroing it. RED before
// the fix: these assertions failed with a NaN-populated row.
#[test]
fn softmax_inplace_nan_fails_closed() {
    let mut x = vec![1.0f32, f32::NAN, 2.0, 0.5];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn softmax_inplace_pos_inf_fails_closed() {
    let mut x = vec![1.0f32, f32::INFINITY, 2.0];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn softmax_inplace_all_neg_inf_fails_closed() {
    let mut x = vec![f32::NEG_INFINITY; 4];
    softmax_inplace(&mut x);
    assert!(x.iter().all(|&v| v == 0.0), "expected all-zero row: {x:?}");
}

#[test]
fn softmax_inplace_normal_row_still_normalizes() {
    let mut x = vec![1.0f32, 2.0, 3.0];
    softmax_inplace(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "row must sum to ~1: {x:?}");
    assert!(x.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
#[should_panic(expected = "a too short for m*k")]
fn vit_batch_matvec_rejects_short_x() {
    let a = [0.0f32; 4]; // rows*cols = 4
    let x = [0.0f32; 3]; // needs n*cols = 4
    batch_matvec(&a, &x, 2, 2, 2);
}

#[test]
#[should_panic(expected = "b too short for n*k")]
fn vit_batch_matvec_rejects_short_a() {
    let a = [0.0f32; 3]; // needs rows*cols = 4
    let x = [0.0f32; 4];
    batch_matvec(&a, &x, 2, 2, 2);
}

#[test]
#[should_panic(expected = "vit_batch_matvec: output shape overflow: n*rows")]
fn vit_batch_matvec_rejects_output_overflow() {
    let a = [0.0f32; 4];
    let x = [0.0f32; 4];
    batch_matvec(&a, &x, usize::MAX, 2, 2);
}

#[test]
fn erf_matches_known_values() {
    // Reference values from a standard erf table.
    assert!((erf(0.0) - 0.0).abs() < 1e-6);
    assert!((erf(1.0) - 0.842_700_79).abs() < 1e-6);
    assert!((erf(-1.0) + 0.842_700_79).abs() < 1e-6);
    assert!((erf(2.0) - 0.995_322_3).abs() < 1e-6);
}

#[test]
fn gelu_exact_matches_known_values() {
    // gelu_exact(0) = 0; gelu_exact is odd-ish but not exactly odd.
    assert!((gelu_exact(0.0)).abs() < 1e-6);
    // gelu_exact(1.0) ~= 0.8413447
    assert!((gelu_exact(1.0) - 0.841_344_7).abs() < 1e-4);
    // gelu_exact(-1.0) ~= -0.1586553
    assert!((gelu_exact(-1.0) + 0.158_655_3).abs() < 1e-4);
}
