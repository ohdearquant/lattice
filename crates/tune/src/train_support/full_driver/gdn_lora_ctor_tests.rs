use super::*;

/// Use asymmetric head counts to expose GDN B-factor sizing drift.
/// See [`docs/design.md`](../../../docs/design.md#train_grad_full-details) for the value-head invariant.
fn asymmetric_gdn_dims() -> GdnDims {
    let key_dim = 8;
    let value_dim = 8;
    let num_kh = 2;
    let value_heads = 4;
    GdnDims {
        num_kh,
        value_heads,
        key_dim,
        value_dim,
        qkv_dim: 2 * key_dim * num_kh + value_heads * value_dim,
        output_dim: value_heads * value_dim,
        kernel_size: 3,
        scale: 1.0 / (key_dim as f32).sqrt(),
    }
}

#[test]
fn gradcheck_gdn_loras_sizes_b_b_and_b_a_by_value_heads() {
    let gd = asymmetric_gdn_dims();
    let hidden = 16;
    let rank = 3;
    let loras = gradcheck_gdn_loras(2, rank, hidden, &gd, 0x1234_5678);
    assert_eq!(loras.len(), 2);
    for lora in &loras {
        assert_eq!(
            lora.b_b.len(),
            gd.value_heads * rank,
            "b_b must be sized by value_heads ({}), not num_kh ({})",
            gd.value_heads,
            gd.num_kh
        );
        assert_eq!(
            lora.b_a.len(),
            gd.value_heads * rank,
            "b_a must be sized by value_heads ({}), not num_kh ({})",
            gd.value_heads,
            gd.num_kh
        );
        // Non-zero: gradcheck mode fills both A and B with random noise.
        assert!(lora.b_b.iter().any(|&v| v != 0.0));
        assert!(lora.b_a.iter().any(|&v| v != 0.0));
    }
}

#[test]
fn zero_b_gdn_loras_sizes_b_b_and_b_a_by_value_heads() {
    let gd = asymmetric_gdn_dims();
    let hidden = 16;
    let rank = 3;
    let init_amp = 1.0 / (hidden as f32).sqrt();
    let loras = zero_b_gdn_loras(2, rank, hidden, &gd, 0xFEED_FACE, init_amp);
    assert_eq!(loras.len(), 2);
    for lora in &loras {
        assert_eq!(
            lora.b_b.len(),
            gd.value_heads * rank,
            "b_b must be sized by value_heads ({}), not num_kh ({})",
            gd.value_heads,
            gd.num_kh
        );
        assert_eq!(
            lora.b_a.len(),
            gd.value_heads * rank,
            "b_a must be sized by value_heads ({}), not num_kh ({})",
            gd.value_heads,
            gd.num_kh
        );
        // Training-mode: B is zero at init (delta=0 reproduces the base).
        assert!(lora.b_b.iter().all(|&v| v == 0.0));
        assert!(lora.b_a.iter().all(|&v| v == 0.0));
        // A is non-zero random.
        assert!(lora.a_b.iter().any(|&v| v != 0.0));
        assert!(lora.a_a.iter().any(|&v| v != 0.0));
    }
}
