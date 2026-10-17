use super::tests::{assert_allclose, build_random_model, run_sequential_prefill, tiny_test_config};
use super::*;

fn expected_lengths(cfg: &Qwen35Config, seq_len: usize) -> ([usize; 5], [usize; 5], [usize; 3]) {
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let value_heads = cfg.linear_num_value_heads.unwrap_or(32);
    let output = value_heads * cfg.linear_value_head_dim;
    let qkv = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + output;
    (
        [
            seq_len * 2 * q,
            seq_len * q,
            seq_len * kv,
            seq_len * kv,
            seq_len * q,
        ],
        [
            seq_len * qkv,
            seq_len * output,
            seq_len * value_heads,
            seq_len * value_heads,
            seq_len * output,
        ],
        [
            seq_len * cfg.intermediate_size,
            seq_len * cfg.intermediate_size,
            seq_len * cfg.hidden_size,
        ],
    )
}

fn asymmetric_config() -> Qwen35Config {
    let mut cfg = tiny_test_config();
    cfg.hidden_size = 19;
    cfg.intermediate_size = 23;
    cfg.num_attention_heads = 6;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 7;
    cfg.linear_num_key_heads = 2;
    cfg.linear_num_value_heads = Some(3);
    cfg.linear_key_head_dim = 5;
    cfg.linear_value_head_dim = 7;
    cfg
}

fn persistent_vectors(scratch: &PrefillScratch) -> [&Vec<f32>; 12] {
    [
        &scratch.hidden,
        &scratch.residual,
        &scratch.attn_out,
        &scratch.logits,
        &scratch.gdn_conv_token,
        &scratch.gdn_output_heads,
        &scratch.gdn_q_tmp,
        &scratch.gdn_k_tmp,
        &scratch.gdn_kv_mem,
        &scratch.gdn_delta,
        &scratch.attn_acc,
        &scratch.attn_tile_scores,
    ]
}

fn poison_scratch(scratch: &mut PrefillScratch) {
    scratch.phase.values.fill(f32::NAN);
    for values in [
        &mut scratch.hidden,
        &mut scratch.residual,
        &mut scratch.attn_out,
        &mut scratch.logits,
        &mut scratch.gdn_conv_token,
        &mut scratch.gdn_output_heads,
        &mut scratch.gdn_q_tmp,
        &mut scratch.gdn_k_tmp,
        &mut scratch.gdn_kv_mem,
        &mut scratch.gdn_delta,
        &mut scratch.attn_acc,
        &mut scratch.attn_tile_scores,
    ] {
        values.fill(f32::NAN);
    }
}

type VectorSnapshot = (Vec<u32>, usize, usize);

fn vector_snapshot(values: &Vec<f32>) -> VectorSnapshot {
    (
        values.iter().map(|v| v.to_bits()).collect(),
        values.capacity(),
        values.as_ptr() as usize,
    )
}

#[derive(Debug, PartialEq, Eq)]
struct ScratchSnapshot {
    layout: PhaseLayout,
    vectors: Vec<VectorSnapshot>,
    tiling: TiledAttentionConfig,
}

fn scratch_snapshot(scratch: &PrefillScratch) -> ScratchSnapshot {
    let mut vectors = vec![vector_snapshot(&scratch.phase.values)];
    vectors.extend(persistent_vectors(scratch).into_iter().map(vector_snapshot));
    ScratchSnapshot {
        layout: scratch.phase.layout,
        vectors,
        tiling: scratch.tiled_config.clone(),
    }
}

#[test]
fn phase_layout_matches_independent_full_gdn_ffn_lengths() {
    let mut ffn_dominant = asymmetric_config();
    ffn_dominant.intermediate_size = 1000;
    for cfg in [tiny_test_config(), asymmetric_config(), ffn_dominant] {
        for seq_len in [0, 1, 7, 31] {
            let layout = PhaseLayout::new(&cfg, seq_len).unwrap();
            let (full, gdn, ffn) = expected_lengths(&cfg, seq_len);
            assert_eq!(layout.full, full);
            assert_eq!(layout.gdn, gdn);
            assert_eq!(layout.ffn, ffn);
            let totals = [
                full.into_iter().sum::<usize>(),
                gdn.into_iter().sum(),
                ffn.into_iter().sum(),
            ];
            assert_eq!(layout.len, *totals.iter().max().unwrap());
            if seq_len > 0 {
                assert!(layout.len < totals.into_iter().sum());
            }
        }
    }

    let addressable = isize::MAX as usize / size_of::<f32>();
    assert_eq!(checked_f32_len(addressable).unwrap(), addressable);
    assert!(checked_f32_len(addressable + 1).is_err());
    assert!(checked_phase_sum(&[usize::MAX, 1]).is_err());
    assert!(checked_phase_sum(&[addressable, 1]).is_err());
    let cfg = asymmetric_config();
    let (full, _, _) = expected_lengths(&cfg, 1);
    let full_per_token: usize = full.into_iter().sum();
    let seq_len = addressable / full_per_token + 1;
    assert!(full.into_iter().all(|len| seq_len * len <= addressable));
    assert!(PhaseLayout::new(&cfg, seq_len).is_err());
    let mut bad_logits = cfg;
    bad_logits.vocab_size = addressable + 1;
    assert!(PhaseLayout::new(&bad_logits, 0).is_err());
}

#[test]
fn new_scratch_capacity_uses_largest_phase_not_the_sum() {
    let cfg = tiny_test_config();
    let seq_len = 2048;
    let (full, gdn, ffn) = expected_lengths(&cfg, seq_len);
    let phase_totals = [
        full.into_iter().sum::<usize>(),
        gdn.into_iter().sum(),
        ffn.into_iter().sum(),
    ];
    let expected = *phase_totals.iter().max().unwrap();
    let old_total: usize = phase_totals.into_iter().sum();
    assert!(expected > 500_000);
    let mut scratch = PrefillScratch::new(&cfg);
    assert_eq!(scratch.phase.values.capacity(), 0);
    scratch.ensure_capacity(&cfg, seq_len).unwrap();
    assert_eq!(scratch.phase.values.len(), expected);
    assert!(scratch.phase.values.capacity() >= expected);
    assert!(scratch.phase.values.capacity() < old_total);

    let value_heads = cfg.linear_num_value_heads.unwrap();
    let output = value_heads * cfg.linear_value_head_dim;
    let qkv = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + output;
    let expected_persistent = [
        seq_len * cfg.hidden_size,
        seq_len * cfg.hidden_size,
        seq_len * cfg.hidden_size,
        cfg.vocab_size,
        qkv,
        output,
        cfg.linear_key_head_dim,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
        cfg.linear_value_head_dim,
        cfg.head_dim,
        scratch.tiled_config.tile_size_kv.max(1),
    ];
    for (values, expected) in persistent_vectors(&scratch)
        .into_iter()
        .zip(expected_persistent)
    {
        assert_eq!(values.len(), expected);
        assert!(values.capacity() >= expected);
    }
}

fn assert_view_offsets<const N: usize>(views: [&[f32]; N], lengths: [usize; N], base: usize) {
    let mut offset = 0;
    for (index, (view, len)) in views.into_iter().zip(lengths).enumerate() {
        assert_eq!(view.len(), len, "view {index} length");
        assert_eq!(
            view.as_ptr() as usize,
            base + offset * size_of::<f32>(),
            "view {index} start offset"
        );
        offset += len;
    }
}

#[test]
fn phase_views_start_at_layout_offsets_and_match_expected_lengths() {
    let cfg = asymmetric_config();
    let mut scratch = PrefillScratch::new(&cfg);
    scratch.ensure_capacity(&cfg, 17).unwrap();
    scratch.ensure_capacity(&cfg, 5).unwrap();
    let (full, gdn, ffn) = expected_lengths(&cfg, 5);
    let base = scratch.phase.values.as_ptr() as usize;

    {
        let view = scratch.phase.full();
        assert_view_offsets(
            [
                view.q_batch,
                view.gate_z_batch,
                view.k_batch,
                view.v_batch,
                view.context_batch,
            ],
            full,
            base,
        );
    }

    {
        let view = scratch.phase.gdn();
        assert_view_offsets(
            [
                view.gdn_qkv_batch,
                view.gdn_z_batch,
                view.gdn_beta_batch,
                view.gdn_alpha_batch,
                view.gdn_out_batch,
            ],
            gdn,
            base,
        );
    }

    {
        let view = scratch.phase.ffn();
        assert_view_offsets([view.gate_batch, view.up_batch, view.ffn_batch], ffn, base);
    }
}

#[test]
fn phase_arena_grows_then_reuses_capacity_and_pointer_when_prompts_shrink() {
    let cfg = tiny_test_config();
    let mut scratch = PrefillScratch::new(&cfg);
    scratch.ensure_capacity(&cfg, 3).unwrap();
    let first_capacity = scratch.phase.values.capacity();
    scratch.ensure_capacity(&cfg, 29).unwrap();
    assert!(scratch.phase.values.capacity() > first_capacity);
    let capacity = scratch.phase.values.capacity();
    let pointer = scratch.phase.values.as_ptr();
    let highwater_len = scratch.phase.values.len();
    scratch.phase.values.fill(-77.0);
    let original = scratch.phase.values.clone();
    for seq_len in [1, 7, 0, 19, 29] {
        scratch.ensure_capacity(&cfg, seq_len).unwrap();
        assert_eq!(
            scratch.phase.layout,
            PhaseLayout::new(&cfg, seq_len).unwrap()
        );
        assert_eq!(scratch.phase.values.as_ptr(), pointer);
        assert_eq!(scratch.phase.values.capacity(), capacity);
        assert_eq!(scratch.phase.values.len(), highwater_len);
        assert_eq!(scratch.phase.values, original);
    }
}

fn assert_finite_signal(name: &str, values: &[f32]) {
    assert!(values.iter().all(|v| v.is_finite()), "{name}: nonfinite");
    assert!(values.iter().any(|v| *v != 0.0), "{name}: no signal");
}

#[test]
fn poisoned_reused_scratch_matches_sequential_mixed_prefill() {
    let cfg = tiny_test_config();
    assert_eq!(cfg.num_linear_attention_layers(), 6);
    assert_eq!(cfg.num_full_attention_layers(), 2);
    let model = build_random_model(cfg.clone(), 0x3a17_4b26_5c35_6d44);
    assert!(model.weights.layers.iter().all(|(_, common)| {
        matches!(&common.ffn, FeedForwardWeights::Dense(dense) if dense.gate_proj.iter().any(|v| *v != 0.0)
            && dense.up_proj.iter().any(|v| *v != 0.0)
            && dense.down_proj.iter().any(|v| *v != 0.0))
    }));
    let mut scratch = PrefillScratch::new(&cfg);
    scratch.ensure_capacity(&cfg, 101).unwrap();
    let arena_pointer = scratch.phase.values.as_ptr();
    let arena_capacity = scratch.phase.values.capacity();
    let persistent_capacities: Vec<_> = persistent_vectors(&scratch)
        .into_iter()
        .map(|values| (values.as_ptr(), values.capacity()))
        .collect();

    for seq_len in [1, 3, 9, 2, 17, 5, 65, 97] {
        let prompt: Vec<_> = (0..seq_len)
            .map(|i| ((i * 13 + 7) % cfg.vocab_size) as u32)
            .collect();
        let (expected_logits, expected_states, expected_kv) =
            run_sequential_prefill(&model, &prompt);
        let mut states: Vec<_> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(&cfg))
            .collect();
        let mut kv = KvCache::new(cfg.num_full_attention_layers());
        poison_scratch(&mut scratch);
        let logits = model
            .prefill_prompt(
                &prompt,
                &mut states,
                &mut kv,
                &mut scratch,
                model.lora.as_ref(),
            )
            .unwrap();
        assert_eq!(kv.seq_len, seq_len);
        assert_eq!(expected_kv.seq_len, seq_len);
        assert_finite_signal("logits", &logits);
        assert_allclose("poisoned logits", &expected_logits, &logits, 1e-5);
        assert_eq!(kv.k.len(), expected_kv.k.len());
        assert_eq!(kv.v.len(), expected_kv.v.len());
        for layer in 0..kv.k.len() {
            assert_eq!(kv.k[layer].len(), seq_len * cfg.full_kv_dim());
            assert_eq!(kv.v[layer].len(), seq_len * cfg.full_kv_dim());
            assert_finite_signal("cached K", &kv.k[layer]);
            assert_finite_signal("cached V", &kv.v[layer]);
            assert_allclose("poisoned K", &expected_kv.k[layer], &kv.k[layer], 1e-5);
            assert_allclose("poisoned V", &expected_kv.v[layer], &kv.v[layer], 1e-5);
        }
        assert_eq!(states.len(), expected_states.len());
        for (actual, expected) in states.iter().zip(&expected_states) {
            assert_finite_signal("GDN state", &actual.s_matrices);
            assert_finite_signal("GDN convolution", &actual.conv_buffer);
            assert_allclose(
                "poisoned GDN",
                &expected.s_matrices,
                &actual.s_matrices,
                1e-5,
            );
            assert_allclose(
                "poisoned convolution",
                &expected.conv_buffer,
                &actual.conv_buffer,
                1e-5,
            );
        }
        assert_eq!(scratch.phase.values.as_ptr(), arena_pointer);
        assert_eq!(scratch.phase.values.capacity(), arena_capacity);
        for (values, (pointer, capacity)) in persistent_vectors(&scratch)
            .into_iter()
            .zip(&persistent_capacities)
        {
            assert_eq!(values.as_ptr(), *pointer);
            assert_eq!(values.capacity(), *capacity);
        }
    }
}

#[test]
fn scratch_dimension_overflow_refuses_without_mutating_reused_buffers() {
    let cfg = tiny_test_config();
    let mutations: &[fn(&mut Qwen35Config)] = &[
        |cfg| cfg.hidden_size = usize::MAX,
        |cfg| cfg.intermediate_size = usize::MAX,
        |cfg| cfg.num_attention_heads = usize::MAX,
        |cfg| cfg.num_key_value_heads = usize::MAX,
        |cfg| cfg.linear_num_key_heads = usize::MAX,
        |cfg| cfg.linear_num_value_heads = Some(usize::MAX),
        |cfg| cfg.linear_key_head_dim = usize::MAX,
        |cfg| cfg.linear_value_head_dim = usize::MAX,
        |cfg| cfg.vocab_size = usize::MAX,
        |cfg| cfg.hidden_size = isize::MAX as usize / size_of::<f32>() + 1,
        |cfg| {
            cfg.num_attention_heads = 1;
            cfg.num_key_value_heads = 1;
            cfg.head_dim = (1 << 56) + 1;
        },
    ];
    for mutate in mutations {
        let mut scratch = PrefillScratch::new(&cfg);
        scratch.ensure_capacity(&cfg, 9).unwrap();
        poison_scratch(&mut scratch);
        let before = scratch_snapshot(&scratch);
        let mut bad_cfg = cfg.clone();
        mutate(&mut bad_cfg);
        assert!(matches!(
            scratch.ensure_capacity(&bad_cfg, 2),
            Err(InferenceError::InvalidInput(_))
        ));
        assert_eq!(scratch_snapshot(&scratch), before);
    }

    let mut scratch = PrefillScratch::new(&cfg);
    scratch.ensure_capacity(&cfg, 9).unwrap();
    poison_scratch(&mut scratch);
    let before = scratch_snapshot(&scratch);
    assert!(matches!(
        scratch.ensure_capacity(&cfg, usize::MAX),
        Err(InferenceError::InvalidInput(_))
    ));
    assert_eq!(scratch_snapshot(&scratch), before);

    scratch.tiled_config.tile_size_kv = usize::MAX;
    let before = scratch_snapshot(&scratch);
    assert!(matches!(
        scratch.ensure_capacity(&cfg, 2),
        Err(InferenceError::InvalidInput(_))
    ));
    assert_eq!(scratch_snapshot(&scratch), before);
}

#[test]
fn prefill_dimension_overflow_refuses_before_state_cache_or_scratch_mutation() {
    let cfg = tiny_test_config();
    let mut model = build_random_model(cfg.clone(), 0x8162_7344_9556_b778);
    let mut scratch = PrefillScratch::new(&cfg);
    scratch.ensure_capacity(&cfg, 9).unwrap();
    poison_scratch(&mut scratch);
    let before_scratch = scratch_snapshot(&scratch);
    let mut states: Vec<_> = (0..cfg.num_linear_attention_layers())
        .map(|_| GatedDeltaNetState::new(&cfg))
        .collect();
    for state in &mut states {
        state.s_matrices.fill(3.25);
        state.conv_buffer.fill(-4.5);
    }
    let before_states: Vec<_> = states.iter().map(GatedDeltaNetState::snapshot).collect();
    let mut kv = KvCache::new(cfg.num_full_attention_layers());
    for values in kv.k.iter_mut().chain(&mut kv.v) {
        values.reserve_exact(1024);
    }
    let before_kv: Vec<_> = kv.k.iter().chain(&kv.v).map(vector_snapshot).collect();
    model.config.hidden_size = usize::MAX;
    assert!(matches!(
        model.prefill_prompt(
            &[1, 2],
            &mut states,
            &mut kv,
            &mut scratch,
            model.lora.as_ref()
        ),
        Err(InferenceError::InvalidInput(_))
    ));
    assert_eq!(scratch_snapshot(&scratch), before_scratch);
    assert_eq!(
        states
            .iter()
            .map(GatedDeltaNetState::snapshot)
            .collect::<Vec<_>>(),
        before_states
    );
    assert_eq!(kv.seq_len, 0);
    assert_eq!(
        kv.k.iter()
            .chain(&kv.v)
            .map(vector_snapshot)
            .collect::<Vec<_>>(),
        before_kv
    );

    model.config = cfg;
    let addressable = isize::MAX as usize / size_of::<f32>();
    model.config.num_key_value_heads = addressable / (2 * model.config.head_dim) + 1;
    let kv_dim = model.config.num_key_value_heads * model.config.head_dim;
    assert!(kv_dim <= addressable);
    assert!(2 * kv_dim > addressable);
    assert!(matches!(
        model.prefill_tokens_batched_for_generate(&[1, 2], &mut states, &mut kv),
        Err(InferenceError::InvalidInput(_))
    ));
    assert_eq!(scratch_snapshot(&scratch), before_scratch);
    assert_eq!(
        states
            .iter()
            .map(GatedDeltaNetState::snapshot)
            .collect::<Vec<_>>(),
        before_states
    );
    assert_eq!(kv.seq_len, 0);
    assert_eq!(
        kv.k.iter()
            .chain(&kv.v)
            .map(vector_snapshot)
            .collect::<Vec<_>>(),
        before_kv
    );
}

#[test]
fn allclose_rejects_nonfinite_values_in_either_operand() {
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (lhs, rhs) in [
            ([nonfinite], [1.0]),
            ([1.0], [nonfinite]),
            ([nonfinite], [nonfinite]),
        ] {
            assert!(
                std::panic::catch_unwind(|| assert_allclose("nonfinite control", &lhs, &rhs, 1e-5))
                    .is_err()
            );
        }
    }
}
