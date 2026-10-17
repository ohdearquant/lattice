use super::*;

/// A geometry triple that satisfies `TrainCtx::try_new`'s consistency
/// check by construction: `Dims`/`GdnDims` are derived from the same
/// `Qwen35Config` the way every real call site (`train.rs`,
/// `train_grad_full.rs`) builds them.
struct Geometry {
    cfg: Qwen35Config,
    dims: Dims,
    gdn_dims: GdnDims,
}

fn valid_geometry() -> Geometry {
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
    let gdn_dims = GdnDims::from_cfg(&cfg);
    Geometry {
        cfg,
        dims,
        gdn_dims,
    }
}

fn valid_adam() -> AdamConfig {
    AdamConfig::new(1e-3, 0.9, 0.999, 1e-8)
}

/// `TrainCtx` intentionally has no `Debug` impl (it borrows a `Qwen35Config`
/// among other things, and nothing needs to print one), so
/// `Result::unwrap_err` isn't available — extract the error message by hand.
fn expect_err(result: Result<TrainCtx<'_>>) -> String {
    match result {
        Ok(_) => panic!("expected TrainCtx::try_new to reject this input"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn try_new_accepts_valid_inputs() {
    let g = valid_geometry();
    let gqa = [19usize, 23];
    let gdn = [20usize, 21, 22];
    let train = TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    )
    .expect("valid geometry/rank/slots/adam must construct a TrainCtx");
    assert_eq!(train.num_gqa_slots(), 2);
    assert_eq!(train.num_gdn_slots(), 3);
    assert_eq!(train.rank(), 8);
    assert!((train.scale() - 2.0).abs() < 1e-6); // alpha / rank = 16 / 8
}

/// An out-of-range GQA slot layer (>= num_hidden_layers)
/// must be rejected — without this guard the tape indexes
/// `model.weights.layers[idx]` (or an equivalent slot array) out of bounds.
#[test]
fn try_new_rejects_out_of_range_slot_layer() {
    let g = valid_geometry();
    let gqa = [g.cfg.num_hidden_layers]; // one past the last valid index
    let gdn: [usize; 0] = [];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("out of range"),
        "expected out-of-range rejection; got: {err}"
    );
}

/// A duplicate layer index within one slot list must
/// be rejected — two slots claiming the same layer would silently alias
/// the same LoRA weights to two different gradient accumulators.
#[test]
fn try_new_rejects_duplicate_slot_layer() {
    let g = valid_geometry();
    let gqa = [19usize, 19];
    let gdn: [usize; 0] = [];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("duplicate"),
        "expected duplicate-layer rejection; got: {err}"
    );
}

/// A layer index claimed by both the GQA and GDN
/// slot lists is a mixer-kind conflict — the same layer cannot be
/// trained as both mixer kinds in one materialised tape. (Layer 19 is
/// full-attention on the 0.8B preset, so this also exercises the
/// one-sided "gdn slot layer is actually GQA" rejection below — either
/// way, `TrainCtx` must not construct.)
#[test]
fn try_new_rejects_mixer_kind_slot_index_mismatch() {
    let g = valid_geometry();
    let gqa = [19usize];
    let gdn = [19usize];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("mixer-kind"),
        "expected mixer-kind/slot-index mismatch rejection; got: {err}"
    );
}

/// Mutation-sensitive, one-sided misclassification: layer 20 is
/// GatedDeltaNet (not full-attention) on the 0.8B preset, so it must be
/// rejected when claimed as a GQA slot even though nothing else conflicts.
#[test]
fn try_new_rejects_gqa_slot_layer_wrong_kind() {
    let g = valid_geometry();
    assert!(
        !g.cfg.is_full_attention(20),
        "fixture assumption: layer 20 must be GDN on the 0.8B preset"
    );
    let gqa = [20usize];
    let gdn: [usize; 0] = [];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("mixer-kind"),
        "expected gqa-slot-wrong-kind rejection; got: {err}"
    );
}

/// Mutation-sensitive, one-sided misclassification: layer 19 is
/// full-attention (GQA) on the 0.8B preset, so it must be rejected when
/// claimed as a GDN slot even though nothing else conflicts.
#[test]
fn try_new_rejects_gdn_slot_layer_wrong_kind() {
    let g = valid_geometry();
    assert!(
        g.cfg.is_full_attention(19),
        "fixture assumption: layer 19 must be full-attention on the 0.8B preset"
    );
    let gqa: [usize; 0] = [];
    let gdn = [19usize];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("mixer-kind"),
        "expected gdn-slot-wrong-kind rejection; got: {err}"
    );
}

/// This exact swap was unrejected pre-fix —
/// `gqa_layers = [20]` (actually GDN), `gdn_layers = [19]` (actually
/// GQA) — both wrong, together. Must be rejected; before this fix
/// `TrainCtx::try_new(...).is_ok()` on this input.
#[test]
fn try_new_rejects_gqa_gdn_slot_swap_19_20() {
    let g = valid_geometry();
    let gqa = [20usize];
    let gdn = [19usize];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("mixer-kind"),
        "expected 19/20 swap rejection; got: {err}"
    );
}

#[test]
fn try_new_rejects_zero_rank() {
    let g = valid_geometry();
    let gqa = [19usize];
    let gdn: [usize; 0] = [];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
        0,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("rank"),
        "expected rank-zero rejection; got: {err}"
    );
}

/// Non-finite alpha must be rejected before it can
/// propagate into `scale = alpha / rank` and poison every LoRA-scaled
/// forward/backward computation with NaN/inf.
#[test]
fn try_new_rejects_non_finite_alpha() {
    let g = valid_geometry();
    let gqa = [19usize];
    let gdn: [usize; 0] = [];
    for bad_alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            8,
            bad_alpha,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("alpha"),
            "expected alpha rejection for {bad_alpha}; got: {err}"
        );
    }
}

/// Non-finite Adam hyperparameters (learning rate,
/// beta1, beta2, epsilon) must each be rejected — any one of them
/// propagates NaN/inf into every optimizer step.
#[test]
fn try_new_rejects_non_finite_adam_hyperparams() {
    let g = valid_geometry();
    let gqa = [19usize];
    let gdn: [usize; 0] = [];
    let cases: [(&str, AdamConfig); 4] = [
        ("learning_rate", AdamConfig::new(f32::NAN, 0.9, 0.999, 1e-8)),
        ("beta1", AdamConfig::new(1e-3, f32::INFINITY, 0.999, 1e-8)),
        ("beta2", AdamConfig::new(1e-3, 0.9, f32::NEG_INFINITY, 1e-8)),
        ("epsilon", AdamConfig::new(1e-3, 0.9, 0.999, f32::NAN)),
    ];
    for (name, adam) in cases {
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            adam,
        ));
        assert!(err.contains(name), "expected {name} rejection; got: {err}");
    }
}

/// `Dims` built from a different model than
/// `TapeGeometry.model` must be rejected — this is the geometry-
/// consistency check the ten-argument signatures had no way to enforce.
#[test]
fn try_new_rejects_dims_model_mismatch() {
    let g = valid_geometry();
    let mut mismatched_dims = g.dims;
    mismatched_dims.hidden += 1;
    let gqa = [19usize];
    let gdn: [usize; 0] = [];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&mismatched_dims, &g.gdn_dims, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("Dims"),
        "expected Dims/model mismatch rejection; got: {err}"
    );
}

/// `GdnDims` built from a different model than
/// `TapeGeometry.model` must be rejected, mirroring the `Dims` check.
#[test]
fn try_new_rejects_gdn_dims_model_mismatch() {
    let g = valid_geometry();
    let mut mismatched_gdn = g.gdn_dims;
    mismatched_gdn.value_heads += 1;
    let gqa = [19usize];
    let gdn: [usize; 0] = [];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &mismatched_gdn, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("GdnDims"),
        "expected GdnDims/model mismatch rejection; got: {err}"
    );
}

/// NaN bypasses `(a - b).abs() > 1e-9` (NaN
/// comparisons are always false), so a non-finite `Dims.eps` would
/// silently pass the tolerance check and poison every RMSNorm call in
/// the tape. The finite check ahead of the tolerance comparison must
/// catch it explicitly.
#[test]
fn try_new_rejects_non_finite_dims_eps() {
    let g = valid_geometry();
    let mut nan_dims = g.dims;
    nan_dims.eps = f32::NAN;
    let gqa = [19usize];
    let gdn: [usize; 0] = [];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&nan_dims, &g.gdn_dims, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("eps"),
        "expected non-finite Dims.eps rejection; got: {err}"
    );
}

/// Same NaN-bypasses-tolerance hole as `Dims.eps`,
/// for `GdnDims.scale` — a non-finite scale would poison every
/// GatedDeltaNet forward/backward call in the tape.
#[test]
fn try_new_rejects_non_finite_gdn_scale() {
    let g = valid_geometry();
    let mut nan_gdn = g.gdn_dims;
    nan_gdn.scale = f32::NAN;
    let gqa = [19usize];
    let gdn: [usize; 0] = [];
    let err = expect_err(TrainCtx::try_new(
        TapeGeometry::new(&g.dims, &nan_gdn, &g.cfg),
        8,
        16.0,
        SlotLayout::new(&gqa, &gdn),
        valid_adam(),
    ));
    assert!(
        err.contains("scale"),
        "expected non-finite GdnDims.scale rejection; got: {err}"
    );
}
