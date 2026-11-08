use super::*;

fn tiny_gdn_dims() -> GdnDims {
    // hidden=16, num_kh=2, value_heads=4, key_dim=4, value_dim=4 — matches
    // the shapes exercised by gdn_backward's own multi-head gradcheck fixture.
    GdnDims {
        num_kh: 2,
        value_heads: 4,
        key_dim: 4,
        value_dim: 4,
        qkv_dim: 2 * (2 * 4) + 4 * 4, // 2*key_dim*num_kh + value_heads*value_dim = 32
        output_dim: 4 * 4,            // value_heads * value_dim = 16
        kernel_size: 3,
        scale: 0.5,
    }
}

#[test]
fn gdn_lora_params_zeros_shapes_match_gdn_dims() {
    let gd = tiny_gdn_dims();
    let hidden = 16;
    let rank = 3;
    let p = GdnLoraParams::zeros(rank, hidden, &gd).expect("zeros should not overflow");

    assert_eq!(p.a_qkv.len(), rank * hidden);
    assert_eq!(p.b_qkv.len(), gd.qkv_dim * rank);
    assert_eq!(p.a_z.len(), rank * hidden);
    assert_eq!(p.b_z.len(), gd.output_dim * rank);
    assert_eq!(p.a_b.len(), rank * hidden);
    assert_eq!(p.b_b.len(), gd.value_heads * rank);
    assert_eq!(p.a_a.len(), rank * hidden);
    assert_eq!(p.b_a.len(), gd.value_heads * rank);
    assert_eq!(p.a_out.len(), rank * gd.output_dim);
    assert_eq!(p.b_out.len(), hidden * rank);
    // Zero-initialised.
    assert!(p.a_qkv.iter().all(|&v| v == 0.0));
    assert!(p.b_out.iter().all(|&v| v == 0.0));
}

#[test]
fn gdn_lora_params_zeros_rejects_overflowing_rank() {
    let gd = tiny_gdn_dims();
    // rank so large that rank * hidden overflows usize on any platform.
    let err = GdnLoraParams::zeros(usize::MAX / 2, 16, &gd);
    assert!(err.is_err(), "overflowing rank*hidden should be rejected");
}

/// Run `f` with the `TrainCtx` the GDN optimizer arms share, on the 0.8B preset.
///
/// A callback rather than a return, because `TrainCtx` borrows the geometry,
/// the config and the slot-layer slices it is built from; returning one would
/// hand back references to this function's locals.
fn with_gdn_train_ctx<R>(rank: usize, f: impl FnOnce(&TrainCtx<'_>) -> R) -> R {
    let cfg = Qwen35Config::qwen35_0_8b();
    let dims = Dims {
        hidden: cfg.hidden_size,
        vocab: cfg.vocab_size,
        num_q_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rope_dim: cfg.rope_dim(),
        inter: cfg.intermediate_size,
        q_dim: cfg.full_q_dim(),
        kv_dim: cfg.full_kv_dim(),
        eps: cfg.rms_norm_eps,
    };
    let real_gdn_dims = GdnDims::from_cfg(&cfg);
    assert!(
        !cfg.is_full_attention(20),
        "layer 20 must be GDN on the 0.8B preset"
    );
    let gqa_layers: [usize; 0] = [];
    let gdn_layers = [20usize];
    let train = TrainCtx::try_new(
        TapeGeometry::new(&dims, &real_gdn_dims, &cfg),
        rank,
        (rank as f32) * 2.0,
        SlotLayout::new(&gqa_layers, &gdn_layers),
        AdamConfig::new(0.1, 0.9, 0.999, 1e-8),
    )
    .expect("valid TrainCtx");
    f(&train)
}

/// Gradients large enough that a single Adam step cannot leave a parameter at
/// its zero init. Every one of the ten arrays is fed, so an array that does
/// not move did not move because it was SKIPPED.
fn all_ten_grads(rank: usize, hidden: usize, gd: &GdnDims) -> GdnLoraParams {
    let mut g = GdnLoraParams::zeros(rank, hidden, gd).unwrap();
    for arr in [
        &mut g.a_qkv,
        &mut g.b_qkv,
        &mut g.a_z,
        &mut g.b_z,
        &mut g.a_b,
        &mut g.b_b,
        &mut g.a_a,
        &mut g.b_a,
        &mut g.a_out,
        &mut g.b_out,
    ] {
        for (i, v) in arr.iter_mut().enumerate() {
            *v = 0.05 * (i as f32 + 1.0);
        }
    }
    g
}

/// THE ZERO PROOF. Three steps under the default selection, with a non-zero
/// gradient on every one of the ten arrays, and `b_a`/`b_b` are still exactly
/// `0.0` afterwards.
///
/// This is what makes leaving `in_proj_a`/`in_proj_b` out of the saved adapter
/// LOSSLESS rather than approximate: a LoRA layer contributes
/// `scale * B @ (A @ x)`, `B` is zero-initialised, and an unstepped `B` stays
/// exactly zero, so the omitted module's delta is exactly zero -- not small.
/// The claim is measured on the tensors here rather than argued from control
/// flow, which is the only form that survives a refactor of the step macro.
#[test]
fn served_selection_leaves_the_fused_kernel_factors_exactly_zero() {
    let gd = tiny_gdn_dims();
    let hidden = 16;
    let rank = 3;
    let mut gdn_loras = vec![GdnLoraParams::zeros(rank, hidden, &gd).unwrap()];
    let gdn_grads = vec![all_ten_grads(rank, hidden, &gd)];
    with_gdn_train_ctx(rank, |train| {
        let mut adam = AdamState::new();
        for _ in 0..3 {
            apply_gdn_adam_updates(
                &mut adam,
                &mut gdn_loras,
                &gdn_grads,
                train,
                GdnModuleSelection::Served,
            );
        }
    });

    let p = &gdn_loras[0];
    for (name, arr) in [("b_b", &p.b_b), ("b_a", &p.b_a)] {
        assert!(
            arr.iter().all(|&v| v == 0.0),
            "{name} must be exactly zero after 3 steps under Served, got {arr:?}"
        );
    }
    for (name, arr) in [("a_b", &p.a_b), ("a_a", &p.a_a)] {
        assert!(
            arr.iter().all(|&v| v == 0.0),
            "{name} must be untouched under Served, got {arr:?}"
        );
    }
    // THE CONTROL, in the same arm: the four served arrays DID move under the
    // same gradients. Without this the test would pass against an optimizer
    // that stepped nothing at all.
    for (name, arr) in [
        ("a_qkv", &p.a_qkv),
        ("b_qkv", &p.b_qkv),
        ("a_out", &p.a_out),
        ("b_out", &p.b_out),
    ] {
        assert!(
            arr.iter().all(|&v| v != 0.0),
            "{name} must have moved under Served, got {arr:?}"
        );
    }
}

/// THE MUTATION. Identical inputs under `All`, and the four fused-kernel
/// arrays move. This is what proves the arm above is measuring the selection
/// rather than a gradient that happened to be zero, a shape that happened to
/// be empty, or an optimizer that stepped nothing.
#[test]
fn all_selection_moves_the_fused_kernel_factors() {
    let gd = tiny_gdn_dims();
    let hidden = 16;
    let rank = 3;
    let mut gdn_loras = vec![GdnLoraParams::zeros(rank, hidden, &gd).unwrap()];
    let gdn_grads = vec![all_ten_grads(rank, hidden, &gd)];
    with_gdn_train_ctx(rank, |train| {
        let mut adam = AdamState::new();
        for _ in 0..3 {
            apply_gdn_adam_updates(
                &mut adam,
                &mut gdn_loras,
                &gdn_grads,
                train,
                GdnModuleSelection::All,
            );
        }
    });

    let p = &gdn_loras[0];
    for (name, arr) in [
        ("a_b", &p.a_b),
        ("b_b", &p.b_b),
        ("a_a", &p.a_a),
        ("b_a", &p.b_a),
    ] {
        assert!(
            arr.iter().all(|&v| v != 0.0),
            "{name} must move under All, got {arr:?}"
        );
    }
}

/// THE SET PROOF. The default selection names only modules the Metal forward
/// can apply, and it reads them from the inference crate's constant rather
/// than from a list maintained here.
#[test]
fn served_selection_is_exactly_the_modules_metal_can_load() {
    use lattice_inference::lora_hook::{
        GDN_LORA_MODULES, GDN_LORA_MODULES_SERVABLE, gdn_lora_module_is_servable,
    };

    assert_eq!(
        GdnModuleSelection::Served.modules(),
        GDN_LORA_MODULES_SERVABLE.as_slice()
    );
    assert_eq!(
        GdnModuleSelection::All.modules(),
        GDN_LORA_MODULES.as_slice()
    );
    assert!(GdnModuleSelection::default() == GdnModuleSelection::Served);

    for m in GdnModuleSelection::Served.modules() {
        assert!(
            gdn_lora_module_is_servable(m),
            "Served names {m}, which the Metal forward refuses"
        );
    }
    // And the difference is exactly the two fused-kernel projections, so this
    // arm fails if either constant grows a module the other does not know.
    let excluded: Vec<&str> = GDN_LORA_MODULES
        .iter()
        .copied()
        .filter(|m| !gdn_lora_module_is_servable(m))
        .collect();
    assert_eq!(excluded, vec!["in_proj_b", "in_proj_a"]);
    assert!(!GdnModuleSelection::Served.trains_fused_kernel_projections());
    assert!(GdnModuleSelection::All.trains_fused_kernel_projections());
}

#[test]
fn apply_gdn_adam_updates_moves_nonzero_grad_params() {
    let gd = tiny_gdn_dims();
    let hidden = 16;
    let rank = 3;
    let mut gdn_loras = vec![GdnLoraParams::zeros(rank, hidden, &gd).unwrap()];
    // Non-zero, non-uniform gradient so the Adam step has a definite sign.
    let mut grads = GdnLoraParams::zeros(rank, hidden, &gd).unwrap();
    for (i, v) in grads.a_qkv.iter_mut().enumerate() {
        *v = 0.01 * (i as f32 + 1.0);
    }
    for (i, v) in grads.b_out.iter_mut().enumerate() {
        *v = -0.02 * (i as f32 + 1.0);
    }
    let gdn_grads = vec![grads];

    // apply_gdn_adam_updates only reads `train`'s gdn slot layout and Adam
    // hyperparameters (not its geometry), so a real preset's consistent
    // Dims/GdnDims/Qwen35Config triple is enough here — it need not match
    // `gd` (the tiny fixture the GdnLoraParams shapes above were built
    // from).
    let cfg = Qwen35Config::qwen35_0_8b();
    let dims = Dims {
        hidden: cfg.hidden_size,
        vocab: cfg.vocab_size,
        num_q_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rope_dim: cfg.rope_dim(),
        inter: cfg.intermediate_size,
        q_dim: cfg.full_q_dim(),
        kv_dim: cfg.full_kv_dim(),
        eps: cfg.rms_norm_eps,
    };
    let real_gdn_dims = GdnDims::from_cfg(&cfg);
    let gqa_layers: [usize; 0] = [];
    // Layer 7 was a full-attention (GQA) layer on the 0.8B preset — an
    // arbitrary index was fine before TrainCtx validated per-slot mixer
    // kind. Now the fixture must name an actual GatedDeltaNet layer.
    assert!(
        !cfg.is_full_attention(20),
        "layer 20 must be GDN on the 0.8B preset"
    );
    let gdn_layers = [20usize];
    let train = TrainCtx::try_new(
        TapeGeometry::new(&dims, &real_gdn_dims, &cfg),
        rank,
        (rank as f32) * 2.0,
        SlotLayout::new(&gqa_layers, &gdn_layers),
        AdamConfig::new(0.1, 0.9, 0.999, 1e-8),
    )
    .expect("valid TrainCtx");

    let mut adam = AdamState::new();
    apply_gdn_adam_updates(
        &mut adam,
        &mut gdn_loras,
        &gdn_grads,
        &train,
        GdnModuleSelection::All,
    );

    // Adam moves params opposite the gradient sign: positive grad -> param
    // decreases from its zero init; negative grad -> param increases.
    assert!(
        gdn_loras[0].a_qkv.iter().all(|&v| v < 0.0),
        "a_qkv should move negative under positive grad: {:?}",
        gdn_loras[0].a_qkv
    );
    assert!(
        gdn_loras[0].b_out.iter().all(|&v| v > 0.0),
        "b_out should move positive under negative grad: {:?}",
        gdn_loras[0].b_out
    );
    // Untouched (zero-grad) arrays stay at their zero init.
    assert!(gdn_loras[0].a_z.iter().all(|&v| v == 0.0));
}

fn qwen35_0_8b_dims(
    cfg: &Qwen35Config,
    gqa_layers: &[usize],
    gdn_layers: &[usize],
) -> (Dims, GdnDims) {
    let dims = Dims {
        hidden: cfg.hidden_size,
        vocab: cfg.vocab_size,
        num_q_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rope_dim: cfg.rope_dim(),
        inter: cfg.intermediate_size,
        q_dim: cfg.full_q_dim(),
        kv_dim: cfg.full_kv_dim(),
        eps: cfg.rms_norm_eps,
    };
    assert!(gqa_layers.iter().all(|&l| cfg.is_full_attention(l)));
    assert!(gdn_layers.iter().all(|&l| !cfg.is_full_attention(l)));
    let gdn_dims = GdnDims::from_cfg(cfg);
    (dims, gdn_dims)
}

fn ramp(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| scale * ((i % 7) as f32 - 3.0 + 0.5))
        .collect()
}

fn ramp_lora(p: &LoraParams, scale: f32) -> LoraParams {
    LoraParams {
        a_q: ramp(p.a_q.len(), scale),
        b_q: ramp(p.b_q.len(), -scale),
        a_v: ramp(p.a_v.len(), scale * 0.5),
        b_v: ramp(p.b_v.len(), -scale * 0.5),
    }
}

fn ramp_gdn(g: &GdnLoraParams, scale: f32) -> GdnLoraParams {
    let mut out = g.clone();
    for (field, s) in [
        (&mut out.a_qkv, 1.0f32),
        (&mut out.b_qkv, -1.0),
        (&mut out.a_z, 0.5),
        (&mut out.b_z, -0.5),
        (&mut out.a_b, 0.25),
        (&mut out.b_b, -0.25),
        (&mut out.a_a, 0.75),
        (&mut out.b_a, -0.75),
        (&mut out.a_out, 1.5),
        (&mut out.b_out, -1.5),
    ] {
        *field = ramp(field.len(), scale * s);
    }
    out
}

#[test]
fn adam_update_key_lengths_match_the_keys_one_update_creates() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let (gqa_layers, gdn_layers, rank) = ([19usize], [20usize, 21], 4usize);
    let (dims, gdn_dims) = qwen35_0_8b_dims(&cfg, &gqa_layers, &gdn_layers);
    let train = TrainCtx::try_new(
        TapeGeometry::new(&dims, &gdn_dims, &cfg),
        rank,
        8.0,
        SlotLayout::new(&gqa_layers, &gdn_layers),
        AdamConfig::new(0.01, 0.9, 0.999, 1e-8),
    )
    .expect("valid TrainCtx");
    for selection in [GdnModuleSelection::Served, GdnModuleSelection::All] {
        let mut loras = vec![LoraParams::zeros(rank, &dims).unwrap()];
        let mut gdn_loras: Vec<GdnLoraParams> = gdn_layers
            .iter()
            .map(|_| GdnLoraParams::zeros(rank, dims.hidden, &gdn_dims).unwrap())
            .collect();
        let grads = vec![ramp_lora(&loras[0], 0.01)];
        let gdn_grads: Vec<GdnLoraParams> = gdn_loras.iter().map(|g| ramp_gdn(g, 0.01)).collect();
        let mut adam = AdamState::new();
        apply_adam_updates(&mut adam, &mut loras, &grads, &train);
        apply_gdn_adam_updates(&mut adam, &mut gdn_loras, &gdn_grads, &train, selection);
        assert_eq!(
            adam.key_lengths(),
            adam_update_key_lengths(&loras, &gdn_loras, &train, selection),
            "{selection:?}"
        );
    }
}

#[test]
fn adam_state_bytes_round_trip_and_continue_bit_for_bit() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let (gqa_layers, gdn_layers, rank) = ([19usize], [20usize], 4usize);
    let (dims, gdn_dims) = qwen35_0_8b_dims(&cfg, &gqa_layers, &gdn_layers);
    let train = TrainCtx::try_new(
        TapeGeometry::new(&dims, &gdn_dims, &cfg),
        rank,
        8.0,
        SlotLayout::new(&gqa_layers, &gdn_layers),
        AdamConfig::new(0.01, 0.9, 0.999, 1e-8),
    )
    .expect("valid TrainCtx");
    let selection = GdnModuleSelection::Served;
    let mut loras = vec![LoraParams::zeros(rank, &dims).unwrap()];
    let mut gdn_loras = vec![GdnLoraParams::zeros(rank, dims.hidden, &gdn_dims).unwrap()];
    let grads_1 = vec![ramp_lora(&loras[0], 0.01)];
    let gdn_grads_1 = vec![ramp_gdn(&gdn_loras[0], 0.01)];
    let grads_2 = vec![ramp_lora(&loras[0], -0.03)];
    let gdn_grads_2 = vec![ramp_gdn(&gdn_loras[0], 0.02)];

    let mut adam = AdamState::new();
    for _ in 0..2 {
        apply_adam_updates(&mut adam, &mut loras, &grads_1, &train);
        apply_gdn_adam_updates(&mut adam, &mut gdn_loras, &gdn_grads_1, &train, selection);
    }
    let bytes = adam.to_bytes();
    let mut restored = AdamState::from_bytes(&bytes).expect("decode");
    assert_eq!(restored.to_bytes(), bytes);
    assert_eq!(restored.t, adam.t);

    let (mut loras_a, mut gdn_a) = (loras.clone(), gdn_loras.clone());
    let (mut loras_b, mut gdn_b) = (loras, gdn_loras);
    apply_adam_updates(&mut adam, &mut loras_a, &grads_2, &train);
    apply_gdn_adam_updates(&mut adam, &mut gdn_a, &gdn_grads_2, &train, selection);
    apply_adam_updates(&mut restored, &mut loras_b, &grads_2, &train);
    apply_gdn_adam_updates(&mut restored, &mut gdn_b, &gdn_grads_2, &train, selection);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&loras_a[0].a_q), bits(&loras_b[0].a_q));
    assert_eq!(bits(&loras_a[0].b_v), bits(&loras_b[0].b_v));
    assert_eq!(bits(&gdn_a[0].a_qkv), bits(&gdn_b[0].a_qkv));
    assert_eq!(bits(&gdn_a[0].b_out), bits(&gdn_b[0].b_out));
    assert_eq!(adam.to_bytes(), restored.to_bytes());
}

#[test]
fn adam_state_from_bytes_refuses_damaged_input() {
    let mut adam = AdamState::new();
    let mut p = vec![0.0f32; 3];
    adam.step(
        "k",
        &mut p,
        &[0.1, -0.2, 0.3],
        0.01,
        0.9,
        0.999,
        1e-8,
        0.0,
        false,
    );
    let good = adam.to_bytes();
    assert!(AdamState::from_bytes(&good).is_ok());
    for cut in [0, 3, 4, 10, good.len() - 1] {
        assert!(
            AdamState::from_bytes(&good[..cut]).is_err(),
            "truncated at {cut}"
        );
    }
    let mut trailing = good.clone();
    trailing.push(0);
    assert!(AdamState::from_bytes(&trailing).is_err());
    // Step count of key "k" sits after the 4-byte count, 4-byte name length and 1-byte name.
    let mut zero_t = good.clone();
    zero_t[9..17].copy_from_slice(&0u64.to_le_bytes());
    assert!(AdamState::from_bytes(&zero_t).is_err());
    let mut huge_n = good.clone();
    huge_n[17..25].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(AdamState::from_bytes(&huge_n).is_err());
    let mut repeated = 2u32.to_le_bytes().to_vec();
    repeated.extend_from_slice(&good[4..]);
    repeated.extend_from_slice(&good[4..]);
    assert!(AdamState::from_bytes(&repeated).is_err());
}
