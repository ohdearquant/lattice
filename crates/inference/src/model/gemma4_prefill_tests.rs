use super::*;
use crate::model::gemma4_config::Gemma4LayerType;
use crate::model::gemma4_ops::gemma4_rope_inv_freq;
use crate::model::gemma4_weights::{
    Gemma4LayerWeights, Gemma4Weights, per_layer_embeddings_from_bits,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::mem::size_of;
use std::path::PathBuf;

const ABS: f32 = 5e-5;
const REL: f32 = 5e-5;
const OWNERS: [usize; 4] = [0, 1, 2, 3];
const SLOT_MAP: [usize; 8] = [0, 1, 2, 3, 2, 3, 2, 3];

fn close(actual: &[f32], expected: &[f32], abs: f32, rel: f32) {
    assert_eq!(actual.len(), expected.len(), "different active lengths");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "actual[{index}] is {actual}");
        assert!(expected.is_finite(), "expected[{index}] is {expected}");
        let bound = abs + rel * actual.abs().max(expected.abs());
        assert!(
            (actual - expected).abs() <= bound,
            "index {index}: actual={actual}, expected={expected}, bound={bound}"
        );
    }
}

fn pattern(length: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((state >> 16) % 31) as i32 - 15) as f32 / 64.0
        })
        .collect()
}

fn norm(length: usize, seed: u32) -> Vec<f32> {
    pattern(length, seed)
        .into_iter()
        .map(|value| 1.0 + value / 4.0)
        .collect()
}

fn fixture() -> Gemma4Model {
    let mut model = super::super::tiny_zero_model_with_vocab(29);
    let mut cfg = model.config.clone();
    cfg.hidden_size = 12;
    cfg.vocab_size = 29;
    cfg.num_hidden_layers = 8;
    cfg.intermediate_size = 11;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 4;
    cfg.global_head_dim = 8;
    cfg.sliding_window = 5;
    cfg.rope_theta = 1_000_000.0;
    cfg.rope_local_base_freq = 10_000.0;
    cfg.partial_rotary_factor = 0.5;
    cfg.layer_types = (0..8)
        .map(|layer| {
            if layer % 2 == 0 {
                Gemma4LayerType::SlidingAttention
            } else {
                Gemma4LayerType::FullAttention
            }
        })
        .collect();
    cfg.num_kv_shared_layers = 4;
    cfg.use_double_wide_mlp_raw = true;
    cfg.hidden_size_per_layer_input = 3;
    cfg.max_position_embeddings = 4096;
    let h = cfg.hidden_size;
    let p = cfg.hidden_size_per_layer_input;
    let packed = cfg.num_hidden_layers * p;
    let layers = (0..cfg.num_hidden_layers)
        .map(|layer| {
            let seed = 101 + layer as u32 * 71;
            let d = cfg.attn_head_dim(layer);
            let q = cfg.num_attention_heads * d;
            let kv = cfg.num_key_value_heads * d;
            let i = cfg.mlp_intermediate_size(layer);
            let shared = cfg.is_kv_shared_layer(layer);
            Gemma4LayerWeights {
                input_layernorm: norm(h, seed),
                post_attention_layernorm: norm(h, seed + 1),
                pre_feedforward_layernorm: norm(h, seed + 2),
                post_feedforward_layernorm: norm(h, seed + 3),
                post_per_layer_input_norm: norm(h, seed + 4),
                layer_scalar: 0.875 + layer as f32 / 64.0,
                per_layer_input_gate: pattern(p * h, seed + 5),
                per_layer_projection: pattern(h * p, seed + 6),
                q_proj: pattern(q * h, seed + 7),
                o_proj: pattern(h * q, seed + 8),
                q_norm: norm(d, seed + 9),
                k_proj: (!shared).then(|| pattern(kv * h, seed + 10)),
                v_proj: (!shared).then(|| pattern(kv * h, seed + 11)),
                k_norm: (!shared).then(|| norm(d, seed + 12)),
                gate_proj: pattern(i * h, seed + 13),
                up_proj: pattern(i * h, seed + 14),
                down_proj: pattern(h * i, seed + 15),
            }
        })
        .collect();
    model.weights = Gemma4Weights {
        embed_tokens: pattern(cfg.vocab_size * h, 17),
        norm: norm(h, 19),
        per_layer_model_projection: pattern(packed * h, 23),
        per_layer_projection_norm: norm(p, 29),
        layers,
    };
    let bits: Vec<u16> = (0..cfg.vocab_size)
        .flat_map(|token| {
            (0..packed).map(move |column| {
                let value = ((token * 17 + column * 7) % 31) as i32 - 15;
                ((value as f32 / 64.0).to_bits() >> 16) as u16
            })
        })
        .collect();
    model.per_layer_embeddings = per_layer_embeddings_from_bits(cfg.vocab_size, packed, &bits);
    model.local_inv_freq = gemma4_rope_inv_freq(cfg.head_dim, cfg.rope_local_base_freq, None);
    model.global_inv_freq = gemma4_rope_inv_freq(
        cfg.global_head_dim,
        cfg.rope_theta,
        Some(cfg.partial_rotary_factor),
    );
    model.config = cfg;
    model
}

fn ids(length: usize, offset: usize) -> Vec<u32> {
    (0..length)
        .map(|index| ((index * 11 + offset * 7 + index / 3) % 29) as u32)
        .collect()
}

fn alias_map(cache: &Gemma4KvCache) {
    assert_eq!(cache.num_slots(), OWNERS.len());
    for (layer, &owner) in SLOT_MAP.iter().enumerate() {
        assert_eq!(cache.layer_slot(layer).unwrap(), owner);
        assert_eq!(cache.is_kv_shared_layer(layer).unwrap(), layer >= 4);
        assert_eq!(
            cache.k_view(layer).unwrap().as_ptr(),
            cache.k_view(owner).unwrap().as_ptr()
        );
        assert_eq!(
            cache.v_view(layer).unwrap().as_ptr(),
            cache.v_view(owner).unwrap().as_ptr()
        );
    }
    for (index, &left) in OWNERS.iter().enumerate() {
        for &right in &OWNERS[index + 1..] {
            assert_ne!(
                cache.k_view(left).unwrap().as_ptr(),
                cache.k_view(right).unwrap().as_ptr()
            );
        }
    }
}

fn cache_close(
    actual: &Gemma4KvCache,
    expected: &Gemma4KvCache,
    layers: usize,
    abs: f32,
    rel: f32,
) {
    assert_eq!(actual.num_slots(), expected.num_slots());
    for layer in 0..layers {
        let owner = actual.layer_slot(layer).unwrap();
        assert_eq!(
            actual.k_view(layer).unwrap().as_ptr(),
            actual.k_view(owner).unwrap().as_ptr()
        );
        assert_eq!(
            actual.v_view(layer).unwrap().as_ptr(),
            actual.v_view(owner).unwrap().as_ptr()
        );
        assert_eq!(
            actual.layer_slot(layer).unwrap(),
            expected.layer_slot(layer).unwrap()
        );
        assert_eq!(
            actual.seq_len(layer).unwrap(),
            expected.seq_len(layer).unwrap()
        );
        close(
            actual.k_view(layer).unwrap(),
            expected.k_view(layer).unwrap(),
            abs,
            rel,
        );
        close(
            actual.v_view(layer).unwrap(),
            expected.v_view(layer).unwrap(),
            abs,
            rel,
        );
    }
}

#[derive(Default)]
struct Observed {
    projections: Vec<(Projection, Option<usize>, usize, usize, usize)>,
    ffns: Vec<(usize, usize, usize)>,
    appends: Vec<(usize, usize)>,
    layers: Vec<(usize, Vec<f32>)>,
}

impl WorkObserver for Observed {
    fn projection(
        &mut self,
        stage: Projection,
        layer: Option<usize>,
        rows: usize,
        input: usize,
        output: usize,
    ) {
        self.projections.push((stage, layer, rows, input, output));
    }

    fn ffn(&mut self, layer: usize, rows: usize, intermediate: usize) {
        self.ffns.push((layer, rows, intermediate));
    }

    fn append(&mut self, layer: usize, rows: usize) {
        self.appends.push((layer, rows));
    }

    fn layer_output(&mut self, layer: usize, last_hidden: &[f32]) {
        self.layers.push((layer, last_hidden.to_vec()));
    }
}

fn buffers(s: &Gemma4PrefillScratch) -> [&Vec<f32>; 20] {
    [
        &s.hidden,
        &s.residual,
        &s.normed,
        &s.branch_out,
        &s.q,
        &s.context,
        &s.k,
        &s.v,
        &s.mlp_gate,
        &s.mlp_up,
        &s.ple_gate,
        &s.ple_inputs,
        &s.ple_context,
        &s.cos_local,
        &s.sin_local,
        &s.cos_global,
        &s.sin_global,
        &s.scores,
        &s.v_norm_ones,
        &s.final_logits,
    ]
}

fn storage(s: &Gemma4PrefillScratch) -> Vec<(usize, usize, usize)> {
    buffers(s)
        .into_iter()
        .map(|v| (v.as_ptr() as usize, v.len(), v.capacity()))
        .collect()
}

fn layout(s: &Gemma4PrefillScratch) -> [usize; 12] {
    let l = &s.layout;
    [
        l.rows,
        l.hidden_len,
        l.q_len,
        l.kv_len,
        l.mlp_len,
        l.gate_len,
        l.ple_len,
        l.local_rope_len,
        l.global_rope_len,
        l.scores_len,
        l.v_norm_len,
        l.logits_len,
    ]
}

fn bits(s: &Gemma4PrefillScratch) -> Vec<Vec<u32>> {
    buffers(s)
        .into_iter()
        .map(|v| v.iter().map(|x| x.to_bits()).collect())
        .collect()
}

fn poison(s: &mut Gemma4PrefillScratch) {
    for v in [
        &mut s.hidden,
        &mut s.residual,
        &mut s.normed,
        &mut s.branch_out,
        &mut s.q,
        &mut s.context,
        &mut s.k,
        &mut s.v,
        &mut s.mlp_gate,
        &mut s.mlp_up,
        &mut s.ple_gate,
        &mut s.ple_inputs,
        &mut s.ple_context,
        &mut s.cos_local,
        &mut s.sin_local,
        &mut s.cos_global,
        &mut s.sin_global,
        &mut s.scores,
        &mut s.v_norm_ones,
        &mut s.final_logits,
    ] {
        v.fill(f32::NAN);
    }
}

fn two_decodes(
    model: &Gemma4Model,
    position: usize,
    cache: &mut Gemma4KvCache,
    scratch: &mut Gemma4Scratch,
    expected_cache: &mut Gemma4KvCache,
    expected: &mut Gemma4Scratch,
    abs: f32,
    rel: f32,
) {
    for step in 0..2 {
        close(&scratch.logits, &expected.logits, abs, rel);
        let token = super::super::argmax(&expected.logits);
        assert_eq!(super::super::argmax(&scratch.logits), token);
        model
            .forward_step(token, position + step, cache, scratch, &[])
            .unwrap();
        model
            .forward_step(token, position + step, expected_cache, expected, &[])
            .unwrap();
        close(&scratch.logits, &expected.logits, abs, rel);
        assert_eq!(
            super::super::argmax(&scratch.logits),
            super::super::argmax(&expected.logits)
        );
        cache_close(
            cache,
            expected_cache,
            model.config.num_hidden_layers,
            abs,
            rel,
        );
    }
}

fn kernel_pair(
    model: &Gemma4Model,
    prompt: &[u32],
    prefix: &[u32],
    batch: &mut Gemma4PrefillScratch,
) {
    let cfg = &model.config;
    let mut cache = model.new_cache(prefix.len() + prompt.len() + 2).unwrap();
    let mut reference_cache = model.new_cache(prefix.len() + prompt.len() + 2).unwrap();
    let mut single = Gemma4Scratch::new(cfg);
    let mut reference = Gemma4Scratch::new(cfg);
    for (position, &token) in prefix.iter().enumerate() {
        model
            .forward_step(token, position, &mut cache, &mut single, &[])
            .unwrap();
        model
            .forward_step(token, position, &mut reference_cache, &mut reference, &[])
            .unwrap();
    }
    let captures: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let mut position = prefix.len();
    for (chunk_index, chunk) in prompt.chunks(64).enumerate() {
        let mut raw_k = Vec::new();
        let mut raw_v = Vec::new();
        for &owner in &OWNERS {
            let width = cfg.num_key_value_heads * cfg.attn_head_dim(owner);
            let k = reference_cache.k_view(owner).unwrap();
            let v = reference_cache.v_view(owner).unwrap();
            let keep = if cfg.is_global_layer(owner) {
                k.len()
            } else {
                k.len().min((cfg.sliding_window - 1) * width)
            };
            raw_k.push(k[k.len() - keep..].to_vec());
            raw_v.push(v[v.len() - keep..].to_vec());
        }
        let mut trace = Vec::new();
        for (row, &token) in chunk.iter().enumerate() {
            trace = model
                .forward_step(
                    token,
                    position + row,
                    &mut reference_cache,
                    &mut reference,
                    if row + 1 == chunk.len() {
                        &captures
                    } else {
                        &[]
                    },
                )
                .unwrap();
            for (slot, &owner) in OWNERS.iter().enumerate() {
                let width = cfg.num_key_value_heads * cfg.attn_head_dim(owner);
                let k = reference_cache.k_view(owner).unwrap();
                let v = reference_cache.v_view(owner).unwrap();
                raw_k[slot].extend_from_slice(&k[k.len() - width..]);
                raw_v[slot].extend_from_slice(&v[v.len() - width..]);
            }
        }
        let final_chunk = (chunk_index + 1) * 64 >= prompt.len();
        let saved_logits: Vec<u32> = single.logits.iter().map(|v| v.to_bits()).collect();
        let mut observed = Observed::default();
        assert!(
            forward_chunk_with_observer(
                model,
                chunk,
                position,
                &mut cache,
                batch,
                &mut single,
                final_chunk,
                &|| false,
                &mut observed
            )
            .unwrap()
        );
        assert_eq!(observed.layers.len(), trace.len());
        for ((layer, actual), (expected_layer, expected)) in observed.layers.iter().zip(&trace) {
            assert_eq!(layer, expected_layer);
            close(actual, expected, ABS, REL);
        }
        for (slot, &owner) in OWNERS.iter().enumerate() {
            close(cache.k_view(owner).unwrap(), &raw_k[slot], ABS, REL);
            close(cache.v_view(owner).unwrap(), &raw_v[slot], ABS, REL);
        }
        if final_chunk {
            close(&single.logits, &reference.logits, ABS, REL);
        } else {
            assert_eq!(
                single
                    .logits
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                saved_logits
            );
        }
        cache.finish_prefill_chunk();
        cache_close(&cache, &reference_cache, cfg.num_hidden_layers, ABS, REL);
        alias_map(&cache);
        position += chunk.len();
    }
    assert!(single.logits.iter().any(|&v| v != 0.0));
    two_decodes(
        model,
        position,
        &mut cache,
        &mut single,
        &mut reference_cache,
        &mut reference,
        ABS,
        REL,
    );
}

#[test]
fn gemma_batched_prefill_matches_scalar_layers_logits_cache_and_two_decodes() {
    let model = fixture();
    for prefix_len in [0, 11] {
        for length in [1, 4, 5, 6, 7, 63, 64, 65, 127, 128, 129] {
            kernel_pair(
                &model,
                &ids(length, 3),
                &ids(prefix_len, 1),
                &mut Gemma4PrefillScratch::default(),
            );
        }
    }
}

fn controller_pair(
    model: &Gemma4Model,
    prompt: &[u32],
    abs_logits: f32,
    abs_cache: f32,
    relative: f32,
) {
    let mut cache = model.new_cache(prompt.len() + 2).unwrap();
    let mut expected_cache = model.new_cache(prompt.len() + 2).unwrap();
    let mut actual = Gemma4Scratch::new(&model.config);
    let mut expected = Gemma4Scratch::new(&model.config);
    let mut position = 0;
    model
        .prefill_prompt(prompt, &mut position, &mut cache, &mut actual, &|| false)
        .unwrap();
    assert_eq!(position, prompt.len());
    for (index, &token) in prompt.iter().enumerate() {
        model
            .forward_step(token, index, &mut expected_cache, &mut expected, &[])
            .unwrap();
    }
    close(&actual.logits, &expected.logits, abs_logits, relative);
    assert_eq!(
        super::super::argmax(&actual.logits),
        super::super::argmax(&expected.logits)
    );
    cache_close(
        &cache,
        &expected_cache,
        model.config.num_hidden_layers,
        abs_cache,
        relative,
    );
    for step in 0..2 {
        let token = super::super::argmax(&expected.logits);
        assert_eq!(super::super::argmax(&actual.logits), token);
        model
            .forward_step(token, position + step, &mut cache, &mut actual, &[])
            .unwrap();
        model
            .forward_step(
                token,
                position + step,
                &mut expected_cache,
                &mut expected,
                &[],
            )
            .unwrap();
        close(&actual.logits, &expected.logits, abs_logits, relative);
        assert_eq!(
            super::super::argmax(&actual.logits),
            super::super::argmax(&expected.logits)
        );
        cache_close(
            &cache,
            &expected_cache,
            model.config.num_hidden_layers,
            abs_cache,
            relative,
        );
    }
}

#[test]
fn gemma_batched_prefill_controller_matches_scalar_across_chunk_boundaries() {
    let model = fixture();
    for length in [1, 63, 64, 65, 127, 128, 129] {
        controller_pair(&model, &ids(length, 2), ABS, ABS, REL);
    }
}

#[test]
fn gemma_batched_prefill_uses_multirow_work_and_one_final_vocab_projection() {
    let model = fixture();
    let cfg = &model.config;
    let mut cache = model.new_cache(16).unwrap();
    let mut single = Gemma4Scratch::new(cfg);
    single.logits.fill(1234.0);
    let mut batch = Gemma4PrefillScratch::default();
    for (position, length, final_chunk) in [(0, 9, false), (9, 3, true)] {
        let mut observed = Observed::default();
        assert!(
            forward_chunk_with_observer(
                &model,
                &ids(length, position),
                position,
                &mut cache,
                &mut batch,
                &mut single,
                final_chunk,
                &|| false,
                &mut observed
            )
            .unwrap()
        );
        let mut expected = vec![(Projection::PerLayerContext, None, length, 12, 24)];
        for layer in 0..8 {
            let d = cfg.attn_head_dim(layer);
            expected.push((Projection::Query, Some(layer), length, 12, 4 * d));
            if layer < 4 {
                expected.push((Projection::Key, Some(layer), length, 12, 2 * d));
                expected.push((Projection::Value, Some(layer), length, 12, 2 * d));
            }
            expected.push((Projection::AttentionOutput, Some(layer), length, 4 * d, 12));
            expected.push((Projection::PerLayerGate, Some(layer), length, 12, 3));
            expected.push((Projection::PerLayerOutput, Some(layer), length, 3, 12));
        }
        if final_chunk {
            expected.push((Projection::FinalVocabulary, None, 1, 12, 29));
        }
        assert_eq!(observed.projections, expected);
        assert_eq!(
            observed.ffns,
            (0..8)
                .map(|l| (l, length, if l < 4 { 11 } else { 22 }))
                .collect::<Vec<_>>()
        );
        assert_eq!(observed.appends, OWNERS.map(|l| (l, length)));
        assert_eq!(
            observed.layers.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        if !final_chunk {
            assert!(single.logits.iter().all(|&v| v == 1234.0));
        }
        cache.finish_prefill_chunk();
    }
    assert!(single.logits.iter().all(|v| v.is_finite()));
    assert!(single.logits.iter().any(|&v| v != 0.0 && v != 1234.0));

    struct CancelAfterVocabulary<'a>(&'a std::cell::Cell<bool>);
    impl WorkObserver for CancelAfterVocabulary<'_> {
        fn projection(
            &mut self,
            stage: Projection,
            _layer: Option<usize>,
            _rows: usize,
            _input: usize,
            _output: usize,
        ) {
            if stage == Projection::FinalVocabulary {
                self.0.set(true);
            }
        }
    }
    let reached = std::cell::Cell::new(false);
    let mut observer = CancelAfterVocabulary(&reached);
    let mut cache = model.new_cache(16).unwrap();
    let mut single = Gemma4Scratch::new(cfg);
    single.hidden.fill(137.0);
    single.logits.fill(149.0);
    let output_before = (single.hidden.clone(), single.logits.clone());
    assert!(
        !forward_chunk_with_observer(
            &model,
            &ids(9, 2),
            0,
            &mut cache,
            &mut Gemma4PrefillScratch::default(),
            &mut single,
            true,
            &|| reached.get(),
            &mut observer,
        )
        .unwrap()
    );
    assert!(
        reached.get(),
        "must compute the actual candidate vocabulary"
    );
    assert_eq!((single.hidden, single.logits), output_before);
}

#[test]
fn gemma_batched_prefill_reuses_poisoned_scratch_across_grow_and_shrink() {
    let model = fixture();
    let mut batch = Gemma4PrefillScratch::default();
    batch.prepare(&model.config, 64, 75).unwrap();
    let stable = storage(&batch);
    for length in [64, 1, 17, 3, 63] {
        poison(&mut batch);
        kernel_pair(&model, &ids(length, length), &ids(11, 4), &mut batch);
        assert_eq!(
            storage(&batch),
            stable,
            "unexpected shrink/grow at {length}"
        );
        assert!(batch.v_norm_ones[..8].iter().all(|&v| v == 1.0));
    }
}

#[test]
fn gemma_batched_prefill_prepare_refuses_geometry_before_mutation() {
    let model = fixture();
    let mut scratch = Gemma4PrefillScratch::default();
    scratch.prepare(&model.config, 7, 29).unwrap();
    assert_eq!(
        layout(&scratch),
        [7, 84, 224, 112, 154, 21, 168, 28, 56, 29, 8, 29]
    );
    poison(&mut scratch);
    let before = (layout(&scratch), storage(&scratch), bits(&scratch));
    for case in 0..9 {
        let mut cfg = model.config.clone();
        let mut rows = 1;
        let mut context = 1;
        match case {
            0 => rows = usize::MAX,
            1 => rows = isize::MAX as usize / (32 * size_of::<f32>()) + 1,
            2 => context = isize::MAX as usize / size_of::<f32>() + 1,
            3 => cfg.hidden_size_per_layer_input = usize::MAX / 8 + 1,
            4 => cfg.intermediate_size = usize::MAX / 2 + 1,
            5 => cfg.num_attention_heads = usize::MAX - 1,
            6 => cfg.hidden_size = usize::MAX,
            7 => cfg.head_dim = 3,
            8 => cfg.sliding_window = 0,
            _ => unreachable!(),
        }
        if rows > context {
            context = rows;
        }
        assert!(
            matches!(
                scratch.prepare(&cfg, rows, context),
                Err(InferenceError::InvalidInput(_))
            ),
            "case {case}"
        );
        assert_eq!(
            (layout(&scratch), storage(&scratch), bits(&scratch)),
            before,
            "case {case} mutated scratch"
        );
    }
}

#[test]
fn gemma_batched_prefill_kernel_preflight_preserves_all_caller_state() {
    for case in 0..6 {
        let mut model = fixture();
        let mut cfg = model.config.clone();
        if case == 5 {
            cfg.num_key_value_heads = 1;
        }
        let mut cache = Gemma4KvCache::new(&cfg, 16).unwrap();
        let position = usize::from(case == 5);
        if case >= 4 {
            for layer in OWNERS {
                let width = cfg.num_key_value_heads * cfg.attn_head_dim(layer);
                cache
                    .append_kv(layer, &pattern(width, 41), &pattern(width, 47))
                    .unwrap();
            }
        }
        let mut single = Gemma4Scratch::new(&model.config);
        single.hidden.fill(71.0);
        single.logits.fill(73.0);
        match case {
            0 => {
                model.weights.layers[7].q_proj.pop();
            }
            1 => {
                model.weights.norm.pop();
            }
            2 => {
                single.hidden.pop();
            }
            3 => {
                single.logits.pop();
            }
            4 | 5 => {}
            _ => unreachable!(),
        }
        let cache_state = |cache: &Gemma4KvCache| {
            (0..8)
                .map(|layer| {
                    (
                        cache.layer_slot(layer).unwrap(),
                        cache.seq_len(layer).unwrap(),
                        cache.k_view(layer).unwrap().as_ptr() as usize,
                        cache.v_view(layer).unwrap().as_ptr() as usize,
                        cache
                            .k_view(layer)
                            .unwrap()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        cache
                            .v_view(layer)
                            .unwrap()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let cache_before = cache_state(&cache);
        let output_before = (
            single
                .hidden
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            single
                .logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        );
        let mut batch = Gemma4PrefillScratch::default();
        batch.prepare(&model.config, 1, 1).unwrap();
        poison(&mut batch);
        let scratch_before = (layout(&batch), storage(&batch), bits(&batch));
        let mut observed = Observed::default();
        let error = forward_chunk_with_observer(
            &model,
            &ids(9, 2),
            position,
            &mut cache,
            &mut batch,
            &mut single,
            true,
            &|| false,
            &mut observed,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                InferenceError::InvalidInput(_) | InferenceError::ShapeMismatch { .. }
            ),
            "case {case}: {error}"
        );
        assert_eq!(cache_state(&cache), cache_before, "case {case} cache");
        assert_eq!(
            (
                single
                    .hidden
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                single
                    .logits
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
            ),
            output_before,
            "case {case} output"
        );
        assert_eq!(
            (layout(&batch), storage(&batch), bits(&batch)),
            scratch_before,
            "case {case} scratch"
        );
        assert!(observed.projections.is_empty(), "case {case}");
        assert!(observed.ffns.is_empty(), "case {case}");
        assert!(observed.appends.is_empty(), "case {case}");
        assert!(observed.layers.is_empty(), "case {case}");
    }
}

#[test]
fn gemma_batched_prefill_refuses_bad_positions_without_cache_or_logits() {
    let model = fixture();
    let mut positions = vec![usize::MAX];
    if let Ok(position) = usize::try_from(u64::from(u32::MAX) + 1) {
        positions.push(position);
    }
    for position in positions {
        let mut cache = model.new_cache(16).unwrap();
        let mut single = Gemma4Scratch::new(&model.config);
        single.logits.fill(91.0);
        let mut batch = Gemma4PrefillScratch::default();
        let error = forward_chunk(
            &model,
            &[2, 3],
            position,
            &mut cache,
            &mut batch,
            &mut single,
            true,
            &|| false,
        )
        .unwrap_err();
        assert!(matches!(error, InferenceError::InvalidInput(_)));
        assert!(single.logits.iter().all(|&v| v == 91.0));
        for layer in 0..8 {
            assert_eq!(cache.seq_len(layer).unwrap(), 0);
        }
    }
}

#[test]
fn gemma_batched_prefill_attention_excludes_future_and_expired_poison() {
    let query = [0.5, -1.0];
    for visible in [0..1, 1..4, 2..3] {
        let mut k = vec![f32::NAN; 5 * 4];
        let mut v = vec![f32::INFINITY; 5 * 4];
        let mut scores64 = Vec::new();
        let mut values = Vec::new();
        for row in visible.clone() {
            let key = [row as f32 / 4.0, (row as f32 - 2.0) / 3.0];
            let value = [2.0 - row as f32, row as f32 * 1.25 + 0.5];
            k[row * 4 + 2..row * 4 + 4].copy_from_slice(&key);
            v[row * 4 + 2..row * 4 + 4].copy_from_slice(&value);
            scores64.push(
                f64::from(query[0]) * f64::from(key[0]) + f64::from(query[1]) * f64::from(key[1]),
            );
            values.push(value);
        }
        let max = scores64.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = scores64.iter().map(|x| (x - max).exp()).collect();
        let denominator: f64 = exp.iter().sum();
        let expected: Vec<f32> = (0..2)
            .map(|d| {
                exp.iter()
                    .zip(&values)
                    .map(|(w, value)| w / denominator * f64::from(value[d]))
                    .sum::<f64>() as f32
            })
            .collect();
        let mut scores = vec![f32::NAN; visible.len()];
        let mut context = [f32::NAN; 2];
        attention_row(&query, &k, &v, 4, 2, visible, &mut scores, &mut context);
        close(&context, &expected, ABS, REL);
        assert!(scores.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn gemma_batched_prefill_attention_preserves_visible_nonfinite_policy() {
    for middle in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut scores = [f32::NAN; 3];
        let mut context = [f32::NAN];
        attention_row(
            &[1.0],
            &[0.0, middle, 0.0, f32::NAN],
            &[2.0, 5.0, 9.0, f32::NAN],
            1,
            0,
            0..3,
            &mut scores,
            &mut context,
        );
        if middle == f32::NEG_INFINITY {
            assert_eq!(context, [5.5]);
        } else {
            assert_eq!(scores, [0.0, 0.0, 1.0]);
            assert_eq!(context, [9.0]);
        }
    }
    let mut scores = [f32::NAN; 3];
    let mut context = [f32::NAN];
    attention_row(
        &[1.0],
        &[f32::NEG_INFINITY; 3],
        &[2.0, 5.0, 9.0],
        1,
        0,
        0..3,
        &mut scores,
        &mut context,
    );
    assert_eq!(scores, [0.0, 0.0, 1.0]);
    assert_eq!(context, [9.0]);
    for poisoned_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        attention_row(
            &[1.0],
            &[0.0, f32::NEG_INFINITY, 0.0],
            &[2.0, poisoned_value, 6.0],
            1,
            0,
            0..3,
            &mut scores,
            &mut context,
        );
        assert_eq!(scores, [0.5, 0.0, 0.5]);
        assert!(
            context[0].is_nan(),
            "visible zero-weight nonfinite product must be evaluated"
        );
    }
}

fn scalar_logits(model: &Gemma4Model) -> Vec<f32> {
    let mut cache = model.new_cache(32).unwrap();
    let mut scratch = Gemma4Scratch::new(&model.config);
    for (position, token) in ids(13, 8).into_iter().enumerate() {
        model
            .forward_step(token, position, &mut cache, &mut scratch, &[])
            .unwrap();
    }
    assert!(scratch.logits.iter().all(|v| v.is_finite()));
    scratch.logits
}

#[test]
fn gemma_batched_prefill_fixture_has_observable_ple_ffn_and_layer_scalars() {
    let expected = scalar_logits(&fixture());
    for mutation in 0..3 {
        let mut model = fixture();
        match mutation {
            0 => {
                for layer in &mut model.weights.layers {
                    layer.per_layer_input_gate.fill(0.0);
                }
            }
            1 => {
                for layer in &mut model.weights.layers[4..] {
                    for row in layer.down_proj.chunks_mut(22) {
                        row[11..].fill(0.0);
                    }
                }
            }
            2 => {
                for layer in &mut model.weights.layers {
                    layer.layer_scalar = 1.0;
                }
            }
            _ => unreachable!(),
        }
        let actual = scalar_logits(&model);
        assert!(
            actual
                .iter()
                .zip(&expected)
                .any(|(a, b)| (a - b).abs() > ABS + REL * a.abs().max(b.abs())),
            "fixture mutation {mutation} is inert"
        );
    }
    assert_ne!(
        fixture().weights.layers[0].k_proj,
        fixture().weights.layers[2].k_proj
    );
    assert_ne!(
        fixture().weights.layers[1].k_proj,
        fixture().weights.layers[3].k_proj
    );
}

#[test]
fn gemma_batched_prefill_comparison_rejects_nonfinite_operands() {
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(std::panic::catch_unwind(|| close(&[poison], &[0.0], ABS, REL)).is_err());
        assert!(std::panic::catch_unwind(|| close(&[0.0], &[poison], ABS, REL)).is_err());
    }
    close(&[1.0, -2.0], &[1.0, -2.0], ABS, REL);
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RealCorpus {
    cases: Vec<RealCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RealCase {
    name: String,
    input_ids: Vec<u32>,
}

fn real_controller_parity(lengths: &[usize]) {
    let corpus_path = PathBuf::from(
        std::env::var_os("LATTICE_GEMMA4_BENCH_CORPUS")
            .expect("selected checkpoint parity requires an explicit real token-ID corpus"),
    );
    let bytes = std::fs::read(&corpus_path).expect("read real token-ID corpus");
    let corpus: RealCorpus =
        serde_json::from_slice(&bytes).expect("strict real token-ID corpus JSON");
    assert!((2..=8).contains(&corpus.cases.len()));
    let model_dir = PathBuf::from(
        std::env::var_os("LATTICE_GEMMA4_MODEL_DIR")
            .expect("selected checkpoint parity requires a real model directory"),
    );
    let model_dir = model_dir
        .canonicalize()
        .expect("existing real Gemma checkpoint directory");
    let model = Gemma4Model::from_safetensors(&model_dir).expect("load real Gemma checkpoint");
    let mut names = std::collections::HashSet::new();
    for case in &corpus.cases {
        assert!(!case.name.is_empty() && names.insert(&case.name));
        assert!((1..=16_384).contains(&case.input_ids.len()));
        assert!(
            case.input_ids
                .iter()
                .all(|&id| (id as usize) < model.config.vocab_size)
        );
        assert!(case.input_ids.len() + 2 <= model.config.max_position_embeddings);
    }
    println!(
        "GEMMA_PREFILL_PARITY corpus_sha256={:x} model_dir={model_dir:?} config_sha256={:x} logits_abs=0.01 cache_abs=0.001 relative=0 decode_steps=2",
        Sha256::digest(&bytes),
        Sha256::digest(std::fs::read(model_dir.join("config.json")).unwrap())
    );
    for &length in lengths {
        let selected: Vec<_> = corpus
            .cases
            .iter()
            .filter(|c| c.input_ids.len() == length)
            .collect();
        assert!(
            !selected.is_empty(),
            "corpus must contain a real case with exactly {length} token IDs"
        );
        for case in selected {
            let mut digest = Sha256::new();
            for &token in &case.input_ids {
                digest.update(token.to_le_bytes());
            }
            println!(
                "GEMMA_PREFILL_PARITY_CASE name={:?} tokens={length} token_ids_sha256={:x}",
                case.name,
                digest.finalize()
            );
            controller_pair(&model, &case.input_ids, 1e-2, 1e-3, 0.0);
        }
    }
    println!("GEMMA_PREFILL_PARITY_COMPLETE lengths={lengths:?}");
}

#[test]
#[ignore = "real checkpoint and explicit real token-ID corpus required; selected invocation fails closed"]
fn gemma_batched_prefill_real_controller_matches_scalar_at_511_512_513() {
    real_controller_parity(&[511, 512, 513]);
}

#[test]
#[ignore = "real checkpoint and explicit 2512-token real corpus case required; selected invocation fails closed"]
fn gemma_batched_prefill_real_controller_matches_scalar_at_2512() {
    real_controller_parity(&[2512]);
}
