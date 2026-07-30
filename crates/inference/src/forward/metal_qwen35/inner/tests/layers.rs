#[test]
fn metal_gdn_state_pairs_buffers_with_explicit_geometry() {
    let device = metal::Device::system_default()
        .expect("typed Metal GDN state gate requires a real Metal device");
    let _guard = gpu_test_lock();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.layer_mask[1] = false;
    let allocated_layers = cfg.num_active_linear_attention_layers();

    let active_state = MetalGdnState::new(&device, &cfg, allocated_layers);
    let geometry = active_state.geometry();

    assert_eq!(geometry.architectural_layers, 3);
    assert_eq!(geometry.active_layers, 2);
    assert_eq!(geometry.allocated_layers, 2);
    assert_eq!(active_state.len(), 2);
    assert_eq!(active_state.precision(), MetalGdnStatePrecision::F32);
    assert_eq!(geometry.qkv_dim, cfg.linear_qkv_dim());
    assert_eq!(
        geometry.conv_history,
        cfg.linear_conv_kernel_dim.saturating_sub(1)
    );
    assert_eq!(geometry.value_heads, cfg.linear_num_value_heads());
    assert_eq!(geometry.key_dim, cfg.linear_key_head_dim);
    assert_eq!(geometry.value_dim, cfg.linear_value_head_dim);

    for (index, layer) in active_state.layers().enumerate() {
        assert_eq!(layer.conv_buffer().label(), format!("gdn_conv_{index}"));
        assert_eq!(layer.s_matrix().label(), format!("gdn_s_{index}"));
        assert_eq!(
            layer.conv_buffer().length(),
            (geometry.conv_elements_per_layer() * std::mem::size_of::<f32>()) as u64
        );
        assert_eq!(
            layer.s_matrix().length(),
            (geometry.matrix_elements_per_layer() * std::mem::size_of::<f32>()) as u64
        );
    }

    let in_memory_state =
        MetalQwen35State::new(&weights, &cfg, 32).expect("pruned tiny hybrid fixture");
    let in_memory_geometry = in_memory_state.session.gdn_gpu_state.geometry();
    assert_eq!(in_memory_geometry.architectural_layers, 3);
    assert_eq!(in_memory_geometry.active_layers, 2);
    assert_eq!(in_memory_geometry.allocated_layers, 3);
    assert_eq!(in_memory_state.session.gdn_gpu_state.len(), 3);
}

#[test]
fn metal_gdn_state_layer_index_drives_each_real_recurrence_pair() {
    metal::Device::system_default()
        .expect("typed Metal GDN state recurrence gate requires a real Metal device");
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    for (index, layer) in state.session.gdn_gpu_state.layers().enumerate() {
        for (buffer, seed) in [
            (layer.conv_buffer(), -(index as f32 + 1.0)),
            (layer.s_matrix(), index as f32 + 1.0),
        ] {
            let elements = (buffer.length() / std::mem::size_of::<f32>() as u64) as usize;
            // SAFETY: StorageModeShared; no command buffer is in flight in this test.
            unsafe {
                let values =
                    std::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), elements);
                values.fill(seed);
            }
        }
    }

    let read_layers = |state: &MetalQwen35State| {
        state
            .session
            .gdn_gpu_state
            .layers()
            .map(|layer| {
                [layer.s_matrix(), layer.conv_buffer()].map(|buffer| {
                    let elements =
                        (buffer.length() / std::mem::size_of::<f32>() as u64) as usize;
                    // SAFETY: StorageModeShared; the preceding command buffer completed.
                    unsafe {
                        std::slice::from_raw_parts(buffer.contents().cast::<f32>(), elements)
                            .to_vec()
                    }
                })
            })
            .collect::<Vec<_>>()
    };

    let before = read_layers(&state);
    let _ = state.forward_step_gdn_only(1, 0);
    let after = read_layers(&state);

    assert_eq!(before.len(), state.session.gdn_gpu_state.len());
    assert_eq!(after.len(), before.len());
    for (index, (before_layer, after_layer)) in before.iter().zip(&after).enumerate() {
        assert_ne!(
            after_layer[0], before_layer[0],
            "real recurrence must update layer {index}'s paired S matrix"
        );
        assert_ne!(
            after_layer[1], before_layer[1],
            "real recurrence must update layer {index}'s paired conv history"
        );
    }
}

#[test]
fn forward_step_gdn_only_does_not_advance_kv_cache() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let seq_len_before = state.session.kv_cache.seq_len;
    let logits = state.forward_step_gdn_only(0, 0);

    assert_eq!(
        state.session.kv_cache.seq_len, seq_len_before,
        "forward_step_gdn_only must not advance KV cache"
    );
    assert_eq!(
        logits.len(),
        cfg.vocab_size,
        "forward_step_gdn_only must return vocab_size logits"
    );
}

#[test]
fn forward_step_gdn_only_returns_finite_logits() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let logits = state.forward_step_gdn_only(1, 0);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "all GDN-only logits must be finite"
    );
}

/// Mutation-sensitive guard for the GDN decode decay-gate clamp.
///
/// When `a_log > ~88`, the kernel's `exp(a_log)` overflows to `+inf`; when
/// `alpha + dt_bias` drives softplus to exactly `0.0`, the *unclamped*
/// product is `inf * 0 = NaN`, which poisons the recurrent state and every
/// subsequent logit (coherent-early / garbage-late on long generations).
/// The fix mirrors the CPU `compute_decay_gate` clamp (`exp(a_log).min(MAX)`).
/// This test FAILS (non-finite logits) if the decode clamp is reverted and
/// PASSES once it is present.
#[test]
fn forward_step_gdn_only_decay_gate_clamps_overflow() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, mut weights) = tiny_hybrid_fixture();

    // Drive every GDN head into the overflow corner: a_log huge (exp -> +inf)
    // and dt_bias very negative (softplus -> 0). Unclamped: inf * 0 = NaN.
    for (attn, _) in weights.layers.iter_mut() {
        if let AttentionWeights::Linear(gdn) = attn {
            for a in gdn.a_log.iter_mut() {
                *a = 100.0;
            }
            for b in gdn.dt_bias.iter_mut() {
                *b = -200.0;
            }
        }
    }

    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let logits = state.forward_step_gdn_only(1, 0);

    assert!(
        logits.iter().all(|v| v.is_finite()),
        "GDN decode decay-gate must clamp exp(a_log) to finite; got non-finite \
             logits (inf * 0 = NaN poison) — the decode clamp was reverted"
    );
}
