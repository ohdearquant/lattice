//! LoRA mixture admission and batched prefill with a loaded adapter.

use super::*;
// -----------------------------------------------------------------
// `generate_with_lora_mixture` must reject an empty inner layer
// slice before it unloads the currently loaded adapter — otherwise
// the request errors out *after* destroying a previously loaded
// adapter, contradicting this method's own admission-error promise
// ("Admission errors are returned before the existing adapter or
// prefix cache is changed"). Loads a real adapter first so the
// destructive path (`unload_lora_adapter`) has something to destroy,
// then asserts it is untouched after the rejected call.
// -----------------------------------------------------------------
#[test]
fn generate_with_lora_mixture_rejects_empty_inner_slice_without_unloading() {
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");

    state
        .load_lora_adapter(vec![make_valid_layer(cfg.hidden_size, 1)], 1.0, None)
        .expect("valid LoRA adapter loads");
    assert!(
        state.has_lora_adapter(),
        "precondition: an adapter must be loaded before the rejected call"
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: (0..cfg.vocab_size as u32).collect(),
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let empty: &[LoraLayerData] = &[];
    let result = state.generate_with_lora_mixture(&[(empty, 1.0)], "a", &tokenizer, &gen_cfg);
    assert!(
        result.is_err(),
        "an empty inner layer slice must be rejected"
    );
    assert!(
        state.has_lora_adapter(),
        "the previously loaded adapter must survive a rejected request, \
                 not be destroyed by the unconditional unload before blending"
    );
}

#[test]
fn load_lora_adapter_rejects_out_of_range_layer_idx() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    // cfg.num_hidden_layers == 1, so layer_idx=1 is out of range
    let mut layer = make_valid_layer(hidden, rank);
    layer.layer_idx = cfg.num_hidden_layers; // == num_hidden_layers, out of range
    let result = state.load_lora_adapter(vec![layer], 1.0, None);
    assert!(result.is_err(), "out-of-range layer_idx must return Err");
}

#[test]
fn load_lora_adapter_rejects_wrong_a_length_mismatched_dims() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    // Declare rank=4, d_in=hidden, but supply A with wrong inner dim.
    // o_proj: d_in = full_q_dim() = hidden, d_out = hidden in tiny fixture.
    let layer = LoraLayerData {
        layer_idx: 0,
        module: "o_proj".into(),
        a: vec![0.0f32; rank * (hidden + 1)], // extra column — length mismatch
        b: vec![0.0f32; hidden * rank],
        rank,
        d_in: hidden,
        d_out: hidden,
    };
    let result = state.load_lora_adapter(vec![layer], 1.0, None);
    assert!(result.is_err(), "A length != rank*d_in must return Err");
}

#[test]
fn load_lora_adapter_rejects_wrong_b_length_mismatched_dims() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    // o_proj: d_in = full_q_dim() = hidden, d_out = hidden in tiny fixture.
    let layer = LoraLayerData {
        layer_idx: 0,
        module: "o_proj".into(),
        a: vec![0.0f32; rank * hidden],
        b: vec![0.0f32; (hidden + 1) * rank], // extra row — length mismatch
        rank,
        d_in: hidden,
        d_out: hidden,
    };
    let result = state.load_lora_adapter(vec![layer], 1.0, None);
    assert!(result.is_err(), "B length != d_out*rank must return Err");
}

#[test]
fn load_lora_adapter_rejects_quarot_with_short_a() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    // Give the fixture a base QuaRot seed matching the adapter's `Some(42)`
    // below, so this test exercises the quarot rotation path itself rather
    // than tripping the (adapter-seed, no-base-seed) refusal added for the
    // fail-closed seed resolution in `load_lora_adapter`.
    state.engine.base_quarot_seed = Some(42);
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let mut layer = make_valid_layer(hidden, rank);
    layer.a.pop(); // short A triggers validation error
    // quarot_seed = Some triggers rotate_adapter_for_quarot path, but
    // validation after that catches the mismatch.
    let result = state.load_lora_adapter(vec![layer], 1.0, Some(42));
    // Assert on the specific length-mismatch message, not merely `is_err()`:
    // a missing base seed ALSO returns an `Err` (the seed-resolution refusal
    // added ahead of this validation), so a bare `is_err()` check cannot tell
    // "reached the length check" apart from "refused before reaching it" and
    // would stay green even if the `state.engine.base_quarot_seed = Some(42)`
    // line above were dropped.
    let message = result
        .expect_err("quarot path: short A must return Err")
        .to_string();
    assert!(
        message.contains("A length"),
        "must fail on the A-length check specifically: {message}"
    );
}

#[test]
fn load_lora_adapter_rejects_quarot_with_short_b() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    // See the matching comment in `load_lora_adapter_rejects_quarot_with_short_a`.
    state.engine.base_quarot_seed = Some(42);
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let mut layer = make_valid_layer(hidden, rank);
    layer.b.pop();
    let result = state.load_lora_adapter(vec![layer], 1.0, Some(42));
    // Unlike short_a: `o_proj` is an output-side module, so B (not A) is the
    // side `rotate_adapter_for_quarot` actually rewrites. Its own internal
    // shape check (`absorb_output_rotation`) rejects the short buffer while
    // applying the rotation, before the generic post-rotation length loop
    // ever runs -- so the message is NOT "B length ...", it is
    // absorb_output_rotation's own "weight length ... != rows*cols ...".
    // Measured: this test failed with exactly that message before this
    // assertion was corrected to match it.
    let message = result
        .expect_err("quarot path: short B must return Err")
        .to_string();
    assert!(
        message.contains("absorb_output_rotation"),
        "must fail inside the output-side rotation's own shape check: {message}"
    );
}

// ── QuaRot seed resolution tests (fail-closed default) ──────────────
// These cover the seed-resolution rule added to `load_lora_adapter`:
// an adapter that names no seed of its own is rotated with the base's
// own seed rather than left unrotated, and a seed mismatch refuses.

#[test]
fn load_lora_adapter_none_seed_resolves_to_base_seed_not_no_rotation() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let base_seed = 42u64;

    // Non-zero, distinguishable A/B data -- `make_valid_layer`'s all-zero
    // buffers would rotate to themselves and could not show a difference.
    let layer = || LoraLayerData {
        layer_idx: 0,
        module: "o_proj".into(),
        a: (0..rank * hidden)
            .map(|i| (i as f32) * 0.001 + 0.1)
            .collect(),
        b: (0..hidden * rank)
            .map(|i| (i as f32) * 0.002 + 0.2)
            .collect(),
        rank,
        d_in: hidden,
        d_out: hidden,
    };

    // QS arm: base is QuaRot-rotated (base_quarot_seed = Some); the adapter
    // passes no seed of its own (`None`) and must be rotated with the base's
    // seed rather than skipped.
    let mut state_qs = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    state_qs.engine.base_quarot_seed = Some(base_seed);
    state_qs
        .load_lora_adapter(vec![layer()], 1.0, None)
        .expect("None adapter seed against a rotated base must succeed");

    // QN arm: base is not rotated (base_quarot_seed = None, the default from
    // `MetalQwen35State::new`); nothing should be rotated.
    let mut state_qn = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    state_qn
        .load_lora_adapter(vec![layer()], 1.0, None)
        .expect("None adapter seed against an unrotated base must succeed");

    let read_ab = |state: &MetalQwen35State| -> (Vec<f32>, Vec<f32>) {
        let adapter = state.lora.as_ref().expect("adapter loaded");
        let proj = adapter
            .get_projection(0, "o_proj")
            .expect("o_proj projection present");
        unsafe {
            (
                read_buffer(&proj.a_buf, rank * hidden),
                read_buffer(&proj.b_buf, hidden * rank),
            )
        }
    };
    let (a_qs, b_qs) = read_ab(&state_qs);
    let (a_qn, b_qn) = read_ab(&state_qn);

    // QN must be byte-identical to the raw input: no rotation happened.
    let raw = layer();
    assert_eq!(a_qn, raw.a, "QN: A must be unrotated");
    assert_eq!(b_qn, raw.b, "QN: B must be unrotated");

    // QS must differ from QN: a `None` adapter seed against a rotated base
    // must NOT resolve the same way as against an unrotated one. If rule 2's
    // `(None, base)` arm regresses back to a skip, this fails because QS
    // becomes byte-identical to QN.
    assert_ne!(
        (a_qs.clone(), b_qs.clone()),
        (a_qn, b_qn),
        "QS (None resolved to the base seed) must differ from QN (no rotation)"
    );

    // And QS must resolve to EXACTLY the base's seed, not merely "some"
    // rotation: rotating an identical copy directly with `base_seed` must
    // land on the same buffers `load_lora_adapter` produced.
    let mut direct = [layer()];
    {
        use crate::quant::quarot::lora::{LoraLayerMut, rotate_adapter_for_quarot};
        use crate::quant::quarot::plan::RotationPlan;

        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let lora_muts: Vec<LoraLayerMut<'_>> = direct
            .iter_mut()
            .map(|l| LoraLayerMut {
                layer_idx: l.layer_idx,
                module: &l.module,
                a: &mut l.a,
                b: &mut l.b,
                rank: l.rank,
                d_in: l.d_in,
                d_out: l.d_out,
            })
            .collect();
        rotate_adapter_for_quarot(lora_muts, base_seed, hidden, &plan)
            .expect("direct rotation with the base seed succeeds");
    }
    assert_eq!(
        a_qs, direct[0].a,
        "QS A must match direct rotation with the base seed"
    );
    assert_eq!(
        b_qs, direct[0].b,
        "QS B must match direct rotation with the base seed"
    );
}

#[test]
fn load_lora_adapter_refuses_seed_mismatch_and_seed_without_rotated_base() {
    let _gpu_guard = gpu_test_lock();
    let Some(_dev) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let hidden = cfg.hidden_size;
    let rank = 4usize;

    // (Some(adapter_seed), Some(base_seed)) with adapter_seed != base_seed:
    // refuse, naming both values.
    let mut state_mismatch = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    state_mismatch.engine.base_quarot_seed = Some(7);
    let err = state_mismatch
        .load_lora_adapter(vec![make_valid_layer(hidden, rank)], 1.0, Some(9))
        .expect_err("mismatched adapter/base QuaRot seeds must refuse");
    let message = err.to_string();
    assert!(
        message.contains('9') && message.contains('7'),
        "error must name both the adapter seed and the base seed: {message}"
    );

    // (Some(adapter_seed), None): refuse -- rotating an adapter for a base
    // that is not QuaRot-rotated. `base_quarot_seed` defaults to `None` from
    // `MetalQwen35State::new`, so no explicit assignment is needed here.
    let mut state_unrotated = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let err = state_unrotated
        .load_lora_adapter(vec![make_valid_layer(hidden, rank)], 1.0, Some(9))
        .expect_err("a seeded adapter against an unrotated base must refuse");
    let message = err.to_string();
    assert!(
        message.contains('9') && message.to_lowercase().contains("base is not rotated"),
        "error must name the adapter seed and state the base is not rotated: {message}"
    );
}

// ── shape / semantic validation tests ────────────────────────────────
// These tests require a Metal device — they return early if none is available.

#[test]
fn load_lora_adapter_rejects_wrong_shape_for_module() {
    let _gpu_guard = gpu_test_lock();
    // q_proj expects d_out = 2 * full_q_dim() = 1024, not hidden = 512
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let layers = vec![LoraLayerData {
        layer_idx: 0,
        module: "q_proj".into(),
        a: vec![0.0; rank * hidden],
        b: vec![0.0; hidden * rank], // d_out=hidden is wrong for q_proj
        rank,
        d_in: hidden,
        d_out: hidden,
    }];
    let result = state.load_lora_adapter(layers, 1.0, None);
    assert!(result.is_err(), "should reject wrong d_out for q_proj");
}

#[test]
fn load_lora_adapter_rejects_gdn_module_on_full_attention_layer() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let qkv_dim = cfg.linear_qkv_dim();
    let layers = vec![LoraLayerData {
        layer_idx: 0,
        module: "in_proj_qkv".into(),
        a: vec![0.0; rank * hidden],
        b: vec![0.0; qkv_dim * rank],
        rank,
        d_in: hidden,
        d_out: qkv_dim,
    }];
    let result = state.load_lora_adapter(layers, 1.0, None);
    assert!(
        result.is_err(),
        "should reject GDN module on full-attention layer"
    );
}

#[test]
fn load_lora_adapter_rejects_unknown_module() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let layers = vec![LoraLayerData {
        layer_idx: 0,
        module: "nonexistent_proj".into(),
        a: vec![0.0; rank * hidden],
        b: vec![0.0; hidden * rank],
        rank,
        d_in: hidden,
        d_out: hidden,
    }];
    let result = state.load_lora_adapter(layers, 1.0, None);
    assert!(result.is_err(), "should reject unknown module name");
}

#[test]
fn load_lora_adapter_rejects_duplicate_projection() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let d_out = cfg.intermediate_size; // gate_proj: hidden -> intermediate
    // Construct two identical layers to trigger duplicate detection.
    let make_gate = || LoraLayerData {
        layer_idx: 0,
        module: "gate_proj".into(),
        a: vec![0.0; rank * hidden],
        b: vec![0.0; d_out * rank],
        rank,
        d_in: hidden,
        d_out,
    };
    let layers = vec![make_gate(), make_gate()];
    let result = state.load_lora_adapter(layers, 1.0, None);
    assert!(result.is_err(), "should reject duplicate layer/module");
}

#[test]
fn load_lora_adapter_rejects_empty_layers() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let result = state.load_lora_adapter(vec![], 1.0, None);
    assert!(result.is_err(), "should reject empty layers");
}

/// A second `load_lora_adapter` while one is already loaded is
/// rejected, and the adapter that was already there is still loaded
/// afterwards.
///
/// The second half is the part worth a test. The rejection is written
/// as an early return before any mutation of `self`, so "the previous
/// adapter keeps serving" is a property of where that check sits in
/// the function, not of anything the check itself does -- exactly the
/// kind of property that survives a refactor silently until something
/// reads it. Every sibling test above asserts only `is_err()`; this
/// one asserts the state as well.
///
/// `has_lora_adapter` reports presence, not identity, so what this
/// pins is that the rejected load left an adapter loaded. Proving it
/// is the *same* adapter would need an observable the type does not
/// expose today; the mutation this guards against -- moving the check
/// below the first line that touches `self.lora` -- is caught by
/// presence alone, because that line assigns.
#[test]
fn load_lora_adapter_rejects_second_load_and_keeps_the_first() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");

    state
        .load_lora_adapter(vec![make_valid_layer(cfg.hidden_size, 1)], 1.0, None)
        .expect("first load should succeed");
    assert!(state.has_lora_adapter(), "first load should have landed");

    let err = state
        .load_lora_adapter(vec![make_valid_layer(cfg.hidden_size, 2)], 1.0, None)
        .expect_err("a second load must be rejected");
    assert!(
        err.to_string()
            .contains("LoRA adapter already loaded; call unload_lora_adapter first"),
        "the rejection should name the remedy, got: {err}"
    );
    assert!(
        state.has_lora_adapter(),
        "the rejected load must leave the previous adapter loaded"
    );

    state.unload_lora_adapter();
    assert!(!state.has_lora_adapter());
    state
        .load_lora_adapter(vec![make_valid_layer(cfg.hidden_size, 2)], 1.0, None)
        .expect("after an unload the same load succeeds");
}

#[test]
fn load_lora_adapter_rejects_non_finite_scale() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let d_out = cfg.intermediate_size;
    let make_layer = || {
        let state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
        let layers = vec![LoraLayerData {
            layer_idx: 0,
            module: "gate_proj".into(),
            a: vec![0.0; rank * hidden],
            b: vec![0.0; d_out * rank],
            rank,
            d_in: hidden,
            d_out,
        }];
        (state, layers)
    };
    let (mut state, layers) = make_layer();
    assert!(
        state.load_lora_adapter(layers, f32::NAN, None).is_err(),
        "should reject NaN scale"
    );
    let (mut state, layers) = make_layer();
    assert!(
        state
            .load_lora_adapter(layers, f32::INFINITY, None)
            .is_err(),
        "should reject +Inf scale"
    );
    let (mut state, layers) = make_layer();
    assert!(
        state
            .load_lora_adapter(layers, f32::NEG_INFINITY, None)
            .is_err(),
        "should reject -Inf scale"
    );
}

#[test]
fn load_lora_adapter_rejects_in_proj_b_and_in_proj_a() {
    let _gpu_guard = gpu_test_lock();
    use crate::model::qwen35_config::Qwen35Config;
    // Test the actual GDN-layer rejection branch (is_full=false)
    let mut gdn_cfg = Qwen35Config::qwen35_0_8b();
    gdn_cfg.num_hidden_layers = 1;
    gdn_cfg.layer_types = vec![LayerType::LinearAttention];
    gdn_cfg.layer_mask = vec![true];

    let result_b = MetalQwen35State::expected_lora_shape(&gdn_cfg, 0, "in_proj_b");
    assert!(result_b.is_err(), "in_proj_b must be rejected on GDN layer");
    let err_msg = result_b.unwrap_err().to_string();
    assert!(
        err_msg.contains("not yet supported"),
        "error should mention 'not yet supported', got: {err_msg}"
    );

    let result_a = MetalQwen35State::expected_lora_shape(&gdn_cfg, 0, "in_proj_a");
    assert!(result_a.is_err(), "in_proj_a must be rejected on GDN layer");

    // Also verify they're rejected on full-attention layers (as unknown)
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    let hidden = cfg.hidden_size;
    let rank = 4usize;
    let layers = vec![LoraLayerData {
        layer_idx: 0,
        module: "in_proj_b".into(),
        a: vec![0.0; rank * hidden],
        b: vec![0.0; hidden * rank],
        rank,
        d_in: hidden,
        d_out: hidden,
    }];
    assert!(
        state.load_lora_adapter(layers, 1.0, None).is_err(),
        "in_proj_b rejected via loader on full-attention layer too"
    );
}

#[test]
fn expected_lora_shape_preserves_metal_in_proj_capability_filter_without_device() {
    use crate::model::qwen35_config::Qwen35Config;

    let mut gdn_cfg = Qwen35Config::qwen35_0_8b();
    gdn_cfg.num_hidden_layers = 1;
    gdn_cfg.layer_types = vec![LayerType::LinearAttention];
    gdn_cfg.layer_mask = vec![true];

    for module in ["in_proj_b", "in_proj_a"] {
        let err = MetalQwen35State::expected_lora_shape(&gdn_cfg, 0, module)
            .expect_err("Metal cannot apply alpha/beta LoRA inside the fused GDN kernel");
        assert_eq!(
            err.to_string(),
            format!(
                "Inference error: module '{module}' is not yet supported in Metal forward \
                         (consumed inside fused GDN recurrence kernel); target other GDN modules instead"
            )
        );
    }

    let full_cfg = Qwen35Config::qwen35_0_8b();
    for module in ["in_proj_b", "in_proj_a"] {
        let err = MetalQwen35State::expected_lora_shape(&full_cfg, 3, module)
            .expect_err("alpha/beta modules remain invalid on full-attention layers");
        assert_eq!(
            err.to_string(),
            format!("Inference error: unknown LoRA module '{module}'")
        );
    }
}

#[test]
fn lora_prefill_matches_sequential_with_adapter() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let rank = 4usize;
    // Use gate+up+down together so the delta survives through silu_mul and
    // produces observable logit change (single-module LoRA is absorbed by
    // zero downstream weights in this fixture). Larger values (0.1, scale=10)
    // ensure the cascaded delta is above measurement noise after 3 matmuls.
    let make_layers = || {
        vec![
            LoraLayerData {
                layer_idx: 0,
                module: "gate_proj".into(),
                a: vec![0.1; rank * hidden],
                b: vec![0.1; inter * rank],
                rank,
                d_in: hidden,
                d_out: inter,
            },
            LoraLayerData {
                layer_idx: 0,
                module: "up_proj".into(),
                a: vec![0.1; rank * hidden],
                b: vec![0.1; inter * rank],
                rank,
                d_in: hidden,
                d_out: inter,
            },
            LoraLayerData {
                layer_idx: 0,
                module: "down_proj".into(),
                a: vec![0.1; rank * inter],
                b: vec![0.1; hidden * rank],
                rank,
                d_in: inter,
                d_out: hidden,
            },
        ]
    };

    // Sequential path with adapter
    let mut state_seq = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    state_seq
        .load_lora_adapter(make_layers(), 10.0, None)
        .unwrap();
    let _ = state_seq.forward_step(42, 0);
    let logits_seq = state_seq.forward_step(7, 1);

    // Batched prefill path with adapter
    let mut state_pre = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
    state_pre
        .load_lora_adapter(make_layers(), 10.0, None)
        .unwrap();
    let logits_pre = state_pre.forward_prefill(&[42, 7]);

    // 1. Batched prefill+adapter must match sequential+adapter
    assert_eq!(logits_seq.len(), logits_pre.len());
    let max_diff: f32 = logits_seq
        .iter()
        .zip(logits_pre.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "prefill with adapter must match sequential: max_diff={max_diff}"
    );

    // Observability (adapter changes logits) is proven separately by the
    // load_adapter_and_dispatch_lora_if_active test which verifies GPU LoRA
    // GEMV on individual buffers. Full-forward observability requires a
    // non-zero-weight fixture (deferred to step 5 e2e PPL validation).
}

// -----------------------------------------------------------------------
// Batched prefill with a loaded LoRA adapter
// -----------------------------------------------------------------------

/// Deterministic, non-constant, nonzero values in `[-amp, amp]`.
fn lora_test_values(len: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            let value = (unit - 0.5) * 2.0 * amp;
            if value == 0.0 { amp } else { value }
        })
        .collect()
}

/// Adapter data for every servable Metal module of the listed ORIGINAL layers.
fn lora_synthetic_layers(
    cfg: &Qwen35Config,
    rank: usize,
    amp: f32,
    layers: &[usize],
) -> Vec<LoraLayerData> {
    let mut out = Vec::new();
    for &layer_idx in layers {
        let attention: &[&str] = if cfg.is_full_attention(layer_idx) {
            &["q_proj", "k_proj", "v_proj", "o_proj"]
        } else {
            &crate::lora_hook::GDN_LORA_MODULES_SERVABLE
        };
        for (n, module) in attention
            .iter()
            .chain(["gate_proj", "up_proj", "down_proj"].iter())
            .enumerate()
        {
            let (d_in, d_out) = MetalQwen35State::expected_lora_shape(cfg, layer_idx, module)
                .expect("servable module has a shape");
            let seed = (layer_idx * 16 + n) as u64 + 1;
            out.push(LoraLayerData {
                layer_idx,
                module: (*module).into(),
                a: lora_test_values(rank * d_in, seed, amp),
                b: lora_test_values(d_out * rank, seed + 1000, amp),
                rank,
                d_in,
                d_out,
            });
        }
    }
    out
}

/// Returns false (after saying so) when no Metal device exists; fails instead
/// when `LATTICE_METAL_TEST_ENFORCE` demands one.
fn lora_batched_test_device(test: &str) -> bool {
    if Device::system_default().is_some() {
        return true;
    }
    eprintln!("[METAL_TEST_SKIP] context={test} reason=no_metal_device");
    assert!(
        std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_none(),
        "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present ({test})"
    );
    false
}

fn lora_max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "row lengths must match");
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

fn lora_gdn_state_diff(
    left: &crate::attention::gdn::GdnSnapshot,
    right: &crate::attention::gdn::GdnSnapshot,
) -> f32 {
    assert_eq!(left.len(), right.len(), "GDN layer counts must match");
    left.iter()
        .zip(right)
        .map(|((s_l, c_l), (s_r, c_r))| {
            lora_max_abs_diff(s_l, s_r).max(lora_max_abs_diff(c_l, c_r))
        })
        .fold(0.0_f32, f32::max)
}

struct LoraArm {
    logits: Vec<f32>,
    hidden: Vec<f32>,
    followup: Vec<f32>,
    gdn: crate::attention::gdn::GdnSnapshot,
    proof: PathProofSnapshot,
}

/// Fresh state (optionally with an adapter), one prefill, a GDN snapshot, then one
/// follow-up `forward_step` at `next_pos`.
fn lora_run_arm(
    weights: &ModelWeights,
    cfg: &Qwen35Config,
    max_cache: usize,
    adapter: Option<Vec<LoraLayerData>>,
    next_pos: usize,
    prefill: impl FnOnce(&mut MetalQwen35State) -> (Vec<f32>, Vec<f32>),
) -> LoraArm {
    use crate::speculative::MtpTargetVerifier as _;
    let mut state = MetalQwen35State::new(weights, cfg, max_cache).expect("tiny fixture");
    if let Some(layers) = adapter {
        state
            .load_lora_adapter(layers, 1.0, None)
            .expect("adapter loads");
    }
    state.path_proof_enabled = true;
    state.reset_path_proof_counters();
    let (logits, hidden) = prefill(&mut state);
    let gdn = state.snapshot_gdn_states();
    let proof = state.path_proof_snapshot();
    let followup = state.forward_step(7, next_pos);
    LoraArm {
        logits,
        hidden,
        followup,
        gdn,
        proof,
    }
}

struct LoraArmDiff {
    logits: f32,
    hidden: f32,
    followup: f32,
    gdn: f32,
}

fn lora_arm_diff(left: &LoraArm, right: &LoraArm) -> LoraArmDiff {
    LoraArmDiff {
        logits: lora_max_abs_diff(&left.logits, &right.logits),
        hidden: if left.hidden.is_empty() && right.hidden.is_empty() {
            0.0
        } else {
            lora_max_abs_diff(&left.hidden, &right.hidden)
        },
        followup: lora_max_abs_diff(&left.followup, &right.followup),
        gdn: lora_gdn_state_diff(&left.gdn, &right.gdn),
    }
}

/// Sequential-vs-batched parity with an adapter, bounded by the same pair's
/// disagreement with no adapter loaded, plus proof that the batched arm ran the
/// batched scheduler and that the adapter is visible in the output.
///
/// `seq_*` / `batch_*` are the adapter arms; `base_*` the adapter-free pair.
fn assert_lora_batched_parity(
    label: &str,
    seq_lora: &LoraArm,
    batch_lora: &LoraArm,
    seq_base: &LoraArm,
    batch_base: &LoraArm,
) {
    const TOL: f32 = 1e-4;
    let lora = lora_arm_diff(seq_lora, batch_lora);
    let base = lora_arm_diff(seq_base, batch_base);
    eprintln!(
        "{label}: adapter pair logits={:e} hidden={:e} followup={:e} gdn={:e}; \
                 base pair logits={:e} hidden={:e} followup={:e} gdn={:e}",
        lora.logits,
        lora.hidden,
        lora.followup,
        lora.gdn,
        base.logits,
        base.hidden,
        base.followup,
        base.gdn
    );
    assert!(
        batch_lora.proof.prefill_kv_batch > 0 && seq_lora.proof.prefill_kv_batch == 0,
        "{label}: the adapter arm must use the batched scheduler and the reference \
                 arm the per-token path (batched={}, sequential={})",
        batch_lora.proof.prefill_kv_batch,
        seq_lora.proof.prefill_kv_batch
    );
    assert!(
        lora.logits <= base.logits + TOL,
        "{label}: logits {} vs base-pair {}",
        lora.logits,
        base.logits
    );
    assert!(
        lora.hidden <= base.hidden + TOL,
        "{label}: hidden {} vs base-pair {}",
        lora.hidden,
        base.hidden
    );
    assert!(
        lora.followup <= base.followup + TOL,
        "{label}: follow-up logits {} vs base-pair {}",
        lora.followup,
        base.followup
    );
    assert!(
        lora.gdn <= base.gdn + TOL,
        "{label}: GDN state {} vs base-pair {}",
        lora.gdn,
        base.gdn
    );
    assert_eq!(
        argmax_f32(&seq_lora.logits),
        argmax_f32(&batch_lora.logits),
        "{label}: argmax"
    );
    assert_eq!(
        argmax_f32(&seq_lora.followup),
        argmax_f32(&batch_lora.followup),
        "{label}: follow-up argmax"
    );
    // The adapter must be visible: far above the disagreement it is bounded by.
    let visible = lora_max_abs_diff(&batch_lora.logits, &batch_base.logits);
    let bound = base.logits + TOL;
    eprintln!("{label}: adapter-on vs adapter-off logits diff={visible:e} (bound {bound:e})");
    assert!(
        visible > 100.0 * bound,
        "{label}: adapter effect {visible} is not > 100x the parity bound {bound}"
    );
    // The state the later steps read must carry the adapter too, so the state
    // comparison above is not between two arms that both ignored it.
    let gdn_visible = lora_gdn_state_diff(&batch_lora.gdn, &batch_base.gdn);
    eprintln!("{label}: adapter-on vs adapter-off GDN state diff={gdn_visible:e}");
    assert!(
        gdn_visible > 100.0 * (base.gdn + TOL),
        "{label}: adapter effect on GDN state {gdn_visible} is not > 100x the bound"
    );
    if !batch_lora.hidden.is_empty() {
        let hidden_visible = lora_max_abs_diff(&batch_lora.hidden, &batch_base.hidden);
        eprintln!("{label}: adapter-on vs adapter-off hidden diff={hidden_visible:e}");
        assert!(
            hidden_visible > 100.0 * (base.hidden + TOL),
            "{label}: adapter effect on hidden {hidden_visible} is not > 100x the bound"
        );
    }
}

fn lora_sequential_prefill(
    tokens: &[u32],
    start: usize,
) -> impl FnOnce(&mut MetalQwen35State) -> (Vec<f32>, Vec<f32>) + '_ {
    move |state| {
        let mut last = Vec::new();
        for (offset, &token) in tokens.iter().enumerate() {
            last = state.forward_step(token, start + offset);
        }
        (last, Vec::new())
    }
}

const LORA_PARITY_TOKENS: [u32; 10] = [3, 8, 1, 12, 5, 6, 9, 2, 17, 30];

#[test]
fn lora_rows_dispatch_matches_cpu_reference_across_rows_ranks_and_strides() {
    let _gpu_guard = gpu_test_lock();
    if !lora_batched_test_device(
        "lora_rows_dispatch_matches_cpu_reference_across_rows_ranks_and_strides",
    ) {
        return;
    }
    let (cfg, weights) = tiny_hybrid_fixture();
    let layer = 3usize;
    let module = "q_proj";
    let (d_in, d_out) = MetalQwen35State::expected_lora_shape(&cfg, layer, module)
        .expect("q_proj shape on the full-attention layer");
    let scale = 1.0f32;
    const GUARD: usize = 16;
    const SENTINEL: f32 = -7777.0;
    for rank in [8usize, 72] {
        let mut state = MetalQwen35State::new(&weights, &cfg, 40).expect("tiny fixture");
        let a = lora_test_values(rank * d_in, 11 + rank as u64, 0.5);
        let b = lora_test_values(d_out * rank, 23 + rank as u64, 0.5);
        state
            .load_lora_adapter(
                vec![LoraLayerData {
                    layer_idx: layer,
                    module: module.into(),
                    a: a.clone(),
                    b: b.clone(),
                    rank,
                    d_in,
                    d_out,
                }],
                scale,
                None,
            )
            .expect("adapter loads");
        for rows in [1usize, 3, 33] {
            let x_stride = d_in + 7;
            let y_stride = d_out + 5;
            let xs = lora_test_values(rows * d_in, 101 + rows as u64, 1.0);
            let y_init = lora_test_values(rows * d_out, 307 + rows as u64, 1.0);
            let mut x_host = vec![SENTINEL; GUARD + rows * x_stride + GUARD];
            let mut y_host = vec![SENTINEL; GUARD + rows * y_stride + GUARD];
            for t in 0..rows {
                let xo = GUARD + t * x_stride;
                x_host[xo..xo + d_in].copy_from_slice(&xs[t * d_in..(t + 1) * d_in]);
                let yo = GUARD + t * y_stride;
                y_host[yo..yo + d_out].copy_from_slice(&y_init[t * d_out..(t + 1) * d_out]);
            }
            let make_buf = |data: &[f32]| -> Buffer {
                state.engine.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    (data.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            let x_buf = make_buf(&x_host);
            let y_buf = make_buf(&y_host);

            let cmd = state.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            state.dispatch_lora_rows_if_active(
                enc,
                &x_buf,
                (GUARD * 4) as u64,
                x_stride as u32,
                &y_buf,
                (GUARD * 4) as u64,
                y_stride as u32,
                rows as u32,
                layer,
                module,
            );
            // An absent projection must leave the output untouched.
            state.dispatch_lora_rows_if_active(
                enc,
                &x_buf,
                (GUARD * 4) as u64,
                x_stride as u32,
                &y_buf,
                (GUARD * 4) as u64,
                y_stride as u32,
                rows as u32,
                layer,
                "k_proj",
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            // SAFETY: StorageModeShared buffer, command buffer completed, length
            // matches the allocation.
            let y_gpu = unsafe { read_buffer(&y_buf, y_host.len()) };

            let mut expected = y_host.clone();
            for t in 0..rows {
                let x_row = &xs[t * d_in..(t + 1) * d_in];
                let inter: Vec<f64> = (0..rank)
                    .map(|j| {
                        (0..d_in)
                            .map(|k| a[j * d_in + k] as f64 * x_row[k] as f64)
                            .sum()
                    })
                    .collect();
                for o in 0..d_out {
                    let dot: f64 = (0..rank).map(|j| b[o * rank + j] as f64 * inter[j]).sum();
                    expected[GUARD + t * y_stride + o] += (scale as f64 * dot) as f32;
                }
            }
            let rows_differ = (1..rows).any(|t| {
                expected[GUARD + t * y_stride..GUARD + t * y_stride + d_out]
                    != expected[GUARD..GUARD + d_out]
            });
            assert!(rows == 1 || rows_differ, "fixture rows must differ");
            for (i, (&g, &e)) in y_gpu.iter().zip(&expected).enumerate() {
                let tol = 1e-3 + 1e-4 * e.abs();
                assert!(
                    (g - e).abs() <= tol,
                    "rank={rank} rows={rows} element {i}: gpu {g} vs cpu {e} \
                             (sentinel regions and stride padding must stay untouched)"
                );
            }
        }
    }
}

#[test]
fn lora_batched_prefill_matches_sequential_on_hybrid_fixture() {
    let _gpu_guard = gpu_test_lock();
    if !lora_batched_test_device("lora_batched_prefill_matches_sequential_on_hybrid_fixture") {
        return;
    }
    let (cfg, weights) = tiny_hybrid_fixture();
    let tokens = LORA_PARITY_TOKENS;
    let n = tokens.len();
    let layers = || Some(lora_synthetic_layers(&cfg, 8, 0.3, &[0, 1, 2, 3]));
    let batch = |state: &mut MetalQwen35State| (state.forward_prefill(&tokens), Vec::new());
    let seq_base = lora_run_arm(
        &weights,
        &cfg,
        32,
        None,
        n,
        lora_sequential_prefill(&tokens, 0),
    );
    let batch_base = lora_run_arm(&weights, &cfg, 32, None, n, batch);
    let seq_lora = lora_run_arm(
        &weights,
        &cfg,
        32,
        layers(),
        n,
        lora_sequential_prefill(&tokens, 0),
    );
    let batch_lora = lora_run_arm(&weights, &cfg, 32, layers(), n, batch);
    assert_lora_batched_parity(
        "hybrid forward_prefill",
        &seq_lora,
        &batch_lora,
        &seq_base,
        &batch_base,
    );
}

#[test]
fn lora_batched_prefill_with_hidden_matches_sequential() {
    let _gpu_guard = gpu_test_lock();
    if !lora_batched_test_device("lora_batched_prefill_with_hidden_matches_sequential") {
        return;
    }
    let (cfg, weights) = tiny_hybrid_fixture();
    let tokens = LORA_PARITY_TOKENS;
    let n = tokens.len();
    let layers = || Some(lora_synthetic_layers(&cfg, 8, 0.3, &[0, 1, 2, 3]));
    let seq = |state: &mut MetalQwen35State| {
        for (pos, &token) in tokens[..n - 1].iter().enumerate() {
            state.forward_step(token, pos);
        }
        state
            .forward_step_with_hidden(tokens[n - 1], n - 1)
            .expect("hidden step")
    };
    let batch = |state: &mut MetalQwen35State| {
        state
            .forward_prefill_with_hidden(&tokens)
            .expect("hidden prefill")
    };
    let seq_base = lora_run_arm(&weights, &cfg, 32, None, n, seq);
    let batch_base = lora_run_arm(&weights, &cfg, 32, None, n, batch);
    let seq_lora = lora_run_arm(&weights, &cfg, 32, layers(), n, seq);
    let batch_lora = lora_run_arm(&weights, &cfg, 32, layers(), n, batch);
    assert!(batch_lora.hidden.iter().any(|&v| v != 0.0));
    assert_lora_batched_parity(
        "hybrid forward_prefill_with_hidden",
        &seq_lora,
        &batch_lora,
        &seq_base,
        &batch_base,
    );
}

#[test]
fn lora_batched_prefill_from_nonzero_start_matches_sequential() {
    let _gpu_guard = gpu_test_lock();
    if !lora_batched_test_device("lora_batched_prefill_from_nonzero_start_matches_sequential") {
        return;
    }
    let (cfg, weights) = tiny_hybrid_fixture();
    let tokens = LORA_PARITY_TOKENS;
    let n = tokens.len();
    let split = 4usize;
    let layers = || Some(lora_synthetic_layers(&cfg, 8, 0.3, &[0, 1, 2, 3]));
    let batch = |state: &mut MetalQwen35State| {
        state.forward_prefill(&tokens[..split]);
        let logits = state
            .forward_prefill_from(&tokens[split..], split, false)
            .expect("suffix prefill");
        (logits, Vec::new())
    };
    let seq_base = lora_run_arm(
        &weights,
        &cfg,
        32,
        None,
        n,
        lora_sequential_prefill(&tokens, 0),
    );
    let batch_base = lora_run_arm(&weights, &cfg, 32, None, n, batch);
    let seq_lora = lora_run_arm(
        &weights,
        &cfg,
        32,
        layers(),
        n,
        lora_sequential_prefill(&tokens, 0),
    );
    let batch_lora = lora_run_arm(&weights, &cfg, 32, layers(), n, batch);
    assert_lora_batched_parity(
        "hybrid forward_prefill_from",
        &seq_lora,
        &batch_lora,
        &seq_base,
        &batch_base,
    );
}

#[test]
fn lora_batched_prefill_across_a_chunk_boundary_matches_sequential() {
    let _gpu_guard = gpu_test_lock();
    if !lora_batched_test_device("lora_batched_prefill_across_a_chunk_boundary_matches_sequential")
    {
        return;
    }
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.max_position_embeddings = 640;
    let max_cache = 600usize;
    let tokens: Vec<u32> = (0..520u32).map(|i| (i * 7 + 3) % 31).collect();
    let n = tokens.len();
    let layers = || Some(lora_synthetic_layers(&cfg, 4, 0.3, &[0, 1, 2, 3]));
    let batch = |state: &mut MetalQwen35State| {
        assert!(
            n > state.session.max_prefill,
            "the prompt must span more than one prefill chunk"
        );
        (state.forward_prefill(&tokens), Vec::new())
    };
    let seq_base = lora_run_arm(
        &weights,
        &cfg,
        max_cache,
        None,
        n,
        lora_sequential_prefill(&tokens, 0),
    );
    let batch_base = lora_run_arm(&weights, &cfg, max_cache, None, n, batch);
    let seq_lora = lora_run_arm(
        &weights,
        &cfg,
        max_cache,
        layers(),
        n,
        lora_sequential_prefill(&tokens, 0),
    );
    let batch_lora = lora_run_arm(&weights, &cfg, max_cache, layers(), n, batch);
    assert_lora_batched_parity(
        "hybrid two-chunk forward_prefill",
        &seq_lora,
        &batch_lora,
        &seq_base,
        &batch_base,
    );
}

#[test]
fn lora_batched_prefill_keys_the_adapter_by_original_layer_under_layer_mask() {
    let _gpu_guard = gpu_test_lock();
    if !lora_batched_test_device(
        "lora_batched_prefill_keys_the_adapter_by_original_layer_under_layer_mask",
    ) {
        return;
    }
    let (mut cfg, mut weights) = tiny_hybrid_fixture();
    // Original layer 0 is pruned: the engine keeps three layers, so original
    // layers 2 and 3 sit at compact indices 1 and 2.
    cfg.layer_mask = vec![false, true, true, true];
    weights.layers.remove(0);
    let tokens = LORA_PARITY_TOKENS;
    let n = tokens.len();
    let layers = || Some(lora_synthetic_layers(&cfg, 8, 0.3, &[2, 3]));
    let batch = |state: &mut MetalQwen35State| (state.forward_prefill(&tokens), Vec::new());
    let seq_base = lora_run_arm(
        &weights,
        &cfg,
        32,
        None,
        n,
        lora_sequential_prefill(&tokens, 0),
    );
    let batch_base = lora_run_arm(&weights, &cfg, 32, None, n, batch);
    let seq_lora = lora_run_arm(
        &weights,
        &cfg,
        32,
        layers(),
        n,
        lora_sequential_prefill(&tokens, 0),
    );
    let batch_lora = lora_run_arm(&weights, &cfg, 32, layers(), n, batch);
    assert_lora_batched_parity(
        "masked-layer forward_prefill",
        &seq_lora,
        &batch_lora,
        &seq_base,
        &batch_base,
    );
}

#[test]
fn lora_on_moe_model_keeps_every_prefill_entry_on_the_per_token_path() {
    with_moe_prefill_state(16, |cfg, state| {
        let (d_in, d_out) = MetalQwen35State::expected_lora_shape(cfg, 0, "q_proj")
            .expect("q_proj shape on the full-attention layer");
        let rank = 4usize;
        state
            .load_lora_adapter(
                vec![LoraLayerData {
                    layer_idx: 0,
                    module: "q_proj".into(),
                    a: lora_test_values(rank * d_in, 5, 0.3),
                    b: lora_test_values(d_out * rank, 6, 0.3),
                    rank,
                    d_in,
                    d_out,
                }],
                1.0,
                None,
            )
            .expect("adapter loads on the MoE fixture");
        let tokens = [1u32, 3, 5];

        state.reset_state();
        let mut reference = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            reference = state.forward_step(token, pos);
        }

        state.reset_state();
        let prefill = state.forward_prefill(&tokens);
        assert_forward_rows_close("forward_prefill", &reference, &prefill);

        state.reset_state();
        let (with_hidden, _) = state
            .forward_prefill_with_hidden(&tokens)
            .expect("hidden prefill takes the per-token path");
        assert_forward_rows_close("forward_prefill_with_hidden", &reference, &with_hidden);

        state.reset_state();
        let from = state
            .forward_prefill_from(&tokens, 0, false)
            .expect("suffix prefill takes the per-token path");
        assert_forward_rows_close("forward_prefill_from", &reference, &from);
    });
}
