use super::tests::{equiv_weights, pseudo_random_vec};
use super::*;
use std::mem::size_of;

pub fn original_gemma4_rope_inv_freq(
    head_dim: usize,
    theta: f64,
    partial_rotary_factor: Option<f32>,
) -> Vec<f32> {
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "head_dim must be a positive even number"
    );
    let half = head_dim / 2;
    let rope_angles = match partial_rotary_factor {
        Some(factor) => ((factor * head_dim as f32) / 2.0) as usize,
        None => half,
    };
    (0..half)
        .map(|i| {
            if i < rope_angles {
                (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32
            } else {
                0.0
            }
        })
        .collect()
}

fn original_mrope_cos_sin_table(
    cfg: &Ernie45Config,
    positions: &[[u32; 3]],
) -> (Vec<f32>, Vec<f32>) {
    let inv_freq = original_gemma4_rope_inv_freq(cfg.head_dim, cfg.rope_theta, None);
    let s = positions.len();
    let head_dim = cfg.head_dim;
    let half = head_dim / 2;
    // Lane -> position-row map: the half dimension's `mrope_section`
    // (`[16, 24, 24]` pinned) is repeated twice to cover the stride-half
    // pairing, so lane `j` reads the position row of the section it falls
    // in — six section-sized blocks running `t, h, w, t, h, w` across the
    // 128 lanes, not one row per lane — and always uses frequency
    // `j % half`. The repetition is what keeps paired lanes `j` and
    // `j + half` on the same row.
    let mut lane_row = vec![0usize; head_dim];
    let mut j = 0;
    for _rep in 0..2 {
        for (row, &size) in cfg.rope_scaling.mrope_section.iter().enumerate() {
            for _ in 0..size {
                lane_row[j] = row;
                j += 1;
            }
        }
    }
    // Config validation guarantees `sum(mrope_section) == head_dim / 2`,
    // so the doubled sections tile the lanes exactly.
    assert_eq!(j, head_dim);
    let mut cos = vec![0f32; s * head_dim];
    let mut sin = vec![0f32; s * head_dim];
    for (t, pos) in positions.iter().enumerate() {
        for j in 0..head_dim {
            let angle = pos[lane_row[j]] as f32 * inv_freq[j % half];
            let (sn, cs) = angle.sin_cos();
            cos[t * head_dim + j] = cs;
            sin[t * head_dim + j] = sn;
        }
    }
    (cos, sin)
}

impl Ernie45Model {
    pub fn original_kv_decode_step(
        &self,
        embeds: &[f32],
        position: [u32; 3],
        cache: &mut Ernie45KvCache,
    ) -> Result<Vec<f32>, InferenceError> {
        let h = self.cfg.hidden_size;
        let len = cache.len();
        if len == 0 {
            return Err(InferenceError::Inference(
                "ernie45 kv step: cache is empty; run kv_prefill first".into(),
            ));
        }
        if len == cache.capacity {
            return Err(InferenceError::Inference(format!(
                "ernie45 kv step: cache is full ({} tokens at capacity {capacity}); the sequence would overflow",
                len,
                capacity = cache.capacity
            )));
        }
        let kv_dim = self.cfg.kv_dim()?;
        self.original_check_cache_shape(cache, kv_dim)?;
        if embeds.len() != h {
            return Err(InferenceError::InvalidInput(format!(
                "ernie45 kv step: embeds has {} values; expected 1 token x {} hidden",
                embeds.len(),
                h
            )));
        }
        if !embeds.iter().all(|v| v.is_finite()) {
            return Err(InferenceError::Inference(
                "ernie45 kv step: embeds carry a non-finite value".into(),
            ));
        }
        self.original_kv_decode_core(embeds, position, cache, kv_dim)
    }

    fn original_kv_decode_core(
        &self,
        embeds: &[f32],
        position: [u32; 3],
        cache: &mut Ernie45KvCache,
        kv_dim: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.cfg;
        let w = &self.weights;
        let h = cfg.hidden_size;
        let len = cache.len();
        // The caller checked len < capacity, so len + 1 cannot wrap.
        let keys = len + 1;

        let q_dim = cfg.q_dim()?;
        let s_q_dim = checked_product(1, q_dim, "kv step query buffer size")?;
        let s_kv_dim = checked_product(1, kv_dim, "kv step key/value buffer size")?;
        let s_h = checked_product(1, h, "kv step hidden buffer size")?;
        let s_intermediate =
            checked_product(1, cfg.intermediate_size, "kv step intermediate buffer size")?;
        let s_keys = checked_product(1, keys, "kv step attention score buffer size")?;
        let keys_hd = checked_product(keys, cfg.head_dim, "kv step key head buffer size")?;
        let hd_keys = checked_product(cfg.head_dim, keys, "kv step transposed value buffer size")?;
        let hd = cfg.head_dim;

        let mut x = embeds.to_vec();
        let (cos, sin) = original_mrope_cos_sin_table(cfg, &[position]);

        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let groups = heads / kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let cap_kv = checked_product(cache.capacity, kv_dim, "kv cache per-layer capacity")?;

        let mut normed = vec![0f32; s_h];
        let mut q = vec![0f32; s_q_dim];
        let mut k = vec![0f32; s_kv_dim];
        let mut v = vec![0f32; s_kv_dim];
        let mut attn_out = vec![0f32; s_q_dim];
        let mut proj = vec![0f32; s_h];
        let mut gate = vec![0f32; s_intermediate];
        let mut up = vec![0f32; s_intermediate];
        let mut scores = vec![0f32; s_keys];
        let mut q_h = vec![0f32; hd];
        let mut k_h = vec![0f32; keys_hd];
        let mut v_t = vec![0f32; hd_keys];
        let mut out_h = vec![0f32; hd];

        for (layer_idx, layer) in w.layers.iter().enumerate() {
            // Attention block: same norm/projection/RoPE calls as
            // `forward_embeds_core` at s == 1.
            normed.copy_from_slice(&x);
            rms_norm(&mut normed, &layer.input_layernorm, h, cfg.rms_norm_eps);
            matmul_bt(&normed, &layer.q_proj, &mut q, 1, h, q_dim);
            matmul_bt(&normed, &layer.k_proj, &mut k, 1, h, kv_dim);
            matmul_bt(&normed, &layer.v_proj, &mut v, 1, h, kv_dim);
            gemma4_apply_rope(&mut q, &cos, &sin, 1, heads, hd);
            gemma4_apply_rope(&mut k, &cos, &sin, 1, kv_heads, hd);

            // Append this token's rows to the cache at row `len`, in the
            // same `[token, kv_head, head_dim]` layout the kernel's own
            // `k` / `v` buffers use.
            let row_base = checked_product(
                checked_product(layer_idx, cache.capacity, "kv cache layer offset")?,
                kv_dim,
                "kv cache row offset",
            )? + checked_product(len, kv_dim, "kv cache row offset")?;
            cache.layer_k[row_base..row_base + kv_dim].copy_from_slice(&k);
            cache.layer_v[row_base..row_base + kv_dim].copy_from_slice(&v);

            // Causal GQA attention over the cached keys. Every cached key
            // is at or before this token's position, so there is no
            // masking row to zero (the uncached path's `row[tq + 1..]`
            // tail). The exact `f32::exp` softmax and the one-KV-head-at-a-
            // time `k_h` / `v_t` scratch copies are the prefill kernel's.
            let layer_k = &cache.layer_k[layer_idx * cap_kv..][..keys * kv_dim];
            let layer_v = &cache.layer_v[layer_idx * cap_kv..][..keys * kv_dim];
            for kvh in 0..kv_heads {
                for tk in 0..keys {
                    let k_row = &layer_k[tk * kv_dim + kvh * hd..][..hd];
                    k_h[tk * hd..(tk + 1) * hd].copy_from_slice(k_row);
                    for d in 0..hd {
                        v_t[d * keys + tk] = layer_v[tk * kv_dim + kvh * hd + d];
                    }
                }
                for qh in kvh * groups..(kvh + 1) * groups {
                    let q_row = &q[qh * hd..(qh + 1) * hd];
                    q_h.copy_from_slice(q_row);
                    matmul_bt(&q_h, &k_h, &mut scores, 1, hd, keys);
                    for score in &mut scores[..keys] {
                        *score *= scale;
                    }
                    softmax_row_fail_closed(&mut scores[..keys]);
                    matmul_bt(&scores, &v_t, &mut out_h, 1, keys, hd);
                    attn_out[qh * hd..(qh + 1) * hd].copy_from_slice(&out_h);
                }
            }
            matmul_bt(&attn_out, &layer.o_proj, &mut proj, 1, q_dim, h);
            for (xi, pi) in x.iter_mut().zip(proj.iter()) {
                *xi += pi;
            }

            // MLP block: identical calls to `forward_embeds_core` at s == 1.
            normed.copy_from_slice(&x);
            rms_norm(
                &mut normed,
                &layer.post_attention_layernorm,
                h,
                cfg.rms_norm_eps,
            );
            matmul_bt(
                &normed,
                &layer.gate_proj,
                &mut gate,
                1,
                h,
                cfg.intermediate_size,
            );
            matmul_bt(
                &normed,
                &layer.up_proj,
                &mut up,
                1,
                h,
                cfg.intermediate_size,
            );
            silu_inplace(&mut gate);
            for (g, &u) in gate.iter_mut().zip(up.iter()) {
                *g *= u;
            }
            matmul_bt(
                &gate,
                &layer.down_proj,
                &mut proj,
                1,
                cfg.intermediate_size,
                h,
            );
            for (xi, pi) in x.iter_mut().zip(proj.iter()) {
                *xi += pi;
            }
        }

        let mut final_normed = x;
        rms_norm(&mut final_normed, &w.final_norm, h, cfg.rms_norm_eps);
        let mut logits = vec![0f32; cfg.vocab_size];
        matmul_bt(&final_normed, &w.lm_head, &mut logits, 1, h, cfg.vocab_size);
        cache.len = len + 1;
        Ok(logits)
    }

    fn original_check_cache_shape(
        &self,
        cache: &Ernie45KvCache,
        kv_dim: usize,
    ) -> Result<(), InferenceError> {
        if cache.layers != self.cfg.num_hidden_layers || cache.kv_dim != kv_dim {
            return Err(InferenceError::Inference(format!(
                "ernie45 kv: cache was built for {} layers x kv_dim {}, this model has {} layers x kv_dim {}",
                cache.layers, cache.kv_dim, self.cfg.num_hidden_layers, kv_dim
            )));
        }
        Ok(())
    }
}

fn config(
    h: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    layers: usize,
) -> Ernie45Config {
    let half = head_dim / 2;
    Ernie45Config {
        hidden_size: h,
        intermediate_size: intermediate,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: kv_heads,
        head_dim,
        vocab_size: vocab,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: Ernie45RopeScaling {
            mrope_section: vec![half.div_ceil(3), (half + 1) / 3, half / 3],
        },
        tie_word_embeddings: false,
        use_bias: false,
    }
}

fn model(cfg: &Ernie45Config) -> Ernie45Model {
    Ernie45Model::new(cfg.clone(), equiv_weights(cfg)).unwrap()
}

fn positions(length: usize) -> Vec<[u32; 3]> {
    (0..length as u32)
        .map(|p| [p * 3 + 1, p * 5 + 2, p * 7 + 4])
        .collect()
}

fn cache(model: &Ernie45Model, prefix: usize, capacity: usize, seed: u64) -> Ernie45KvCache {
    let cfg = model.config();
    let mut cache =
        Ernie45KvCache::new(cfg.num_hidden_layers, cfg.kv_dim().unwrap(), capacity).unwrap();
    model
        .kv_prefill(
            &pseudo_random_vec(prefix * cfg.hidden_size, seed),
            &positions(prefix),
            &mut cache,
        )
        .unwrap();
    cache
}

fn clone_cache(cache: &Ernie45KvCache) -> Ernie45KvCache {
    Ernie45KvCache {
        layers: cache.layers,
        capacity: cache.capacity,
        kv_dim: cache.kv_dim,
        len: cache.len,
        layer_k: cache.layer_k.clone(),
        layer_v: cache.layer_v.clone(),
    }
}

fn float_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn assert_finite_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|v| v.is_finite()), "actual is nonfinite");
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "old oracle is nonfinite"
    );
    assert_eq!(float_bits(actual), float_bits(expected));
}

#[derive(Debug, PartialEq, Eq)]
struct CacheState {
    dimensions: [usize; 4],
    k: Vec<u32>,
    v: Vec<u32>,
}

fn cache_state(cache: &Ernie45KvCache) -> CacheState {
    CacheState {
        dimensions: [cache.layers, cache.capacity, cache.kv_dim, cache.len],
        k: float_bits(&cache.layer_k),
        v: float_bits(&cache.layer_v),
    }
}

fn cache_storage(cache: &Ernie45KvCache) -> [(usize, usize, usize); 2] {
    [
        (
            cache.layer_k.as_ptr() as usize,
            cache.layer_k.len(),
            cache.layer_k.capacity(),
        ),
        (
            cache.layer_v.as_ptr() as usize,
            cache.layer_v.len(),
            cache.layer_v.capacity(),
        ),
    ]
}

fn assert_cache(actual: &Ernie45KvCache, expected: &Ernie45KvCache) {
    assert_eq!(cache_state(actual), cache_state(expected));
    for layer in 0..actual.layers {
        let start = layer * actual.capacity * actual.kv_dim;
        let end = start + actual.len * actual.kv_dim;
        assert!(actual.layer_k[start..end].iter().all(|v| v.is_finite()));
        assert!(actual.layer_v[start..end].iter().all(|v| v.is_finite()));
    }
}

fn float_buffers(s: &Ernie45DecodeScratch) -> [&Vec<f32>; 18] {
    [
        &s.x,
        &s.normed,
        &s.q,
        &s.k,
        &s.v,
        &s.attn_out,
        &s.proj,
        &s.gate,
        &s.up,
        &s.scores,
        &s.q_h,
        &s.k_h,
        &s.v_t,
        &s.out_h,
        &s.logits,
        &s.inv_freq,
        &s.cos,
        &s.sin,
    ]
}

fn storage(s: &Ernie45DecodeScratch) -> Vec<(usize, usize, usize)> {
    let mut slots: Vec<_> = float_buffers(s)
        .into_iter()
        .map(|v| (v.as_ptr() as usize, v.len(), v.capacity()))
        .collect();
    slots.insert(
        16,
        (
            s.lane_row.as_ptr() as usize,
            s.lane_row.len(),
            s.lane_row.capacity(),
        ),
    );
    slots
}

#[derive(Debug, PartialEq, Eq)]
struct ScratchState {
    storage: Vec<(usize, usize, usize)>,
    floats: Vec<Vec<u32>>,
    lane_row: Vec<usize>,
    calls: usize,
}

fn scratch_state(s: &Ernie45DecodeScratch) -> ScratchState {
    ScratchState {
        storage: storage(s),
        floats: float_buffers(s)
            .into_iter()
            .map(|v| float_bits(v))
            .collect(),
        lane_row: s.lane_row.clone(),
        calls: s.test_decode_calls(),
    }
}

fn poison(s: &mut Ernie45DecodeScratch) {
    for values in [
        &mut s.x,
        &mut s.normed,
        &mut s.q,
        &mut s.k,
        &mut s.v,
        &mut s.attn_out,
        &mut s.proj,
        &mut s.gate,
        &mut s.up,
        &mut s.scores,
        &mut s.q_h,
        &mut s.k_h,
        &mut s.v_t,
        &mut s.out_h,
        &mut s.logits,
        &mut s.inv_freq,
        &mut s.cos,
        &mut s.sin,
    ] {
        values.fill(f32::NAN);
    }
    s.lane_row.fill(usize::MAX);
}

fn poison_inactive(cache: &mut Ernie45KvCache) {
    for layer in 0..cache.layers {
        let start = (layer * cache.capacity + cache.len) * cache.kv_dim;
        let end = (layer + 1) * cache.capacity * cache.kv_dim;
        cache.layer_k[start..end].fill(f32::NAN);
        cache.layer_v[start..end].fill(f32::NAN);
    }
}

#[test]
fn ernie_decode_scratch_matches_frozen_allocating_core_full_logits_and_cache() {
    for cfg in [
        config(8, 11, 4, 2, 4, 7, 2),
        config(13, 17, 3, 3, 6, 9, 2),
        config(9, 7, 4, 1, 8, 11, 3),
        config(33, 21, 4, 2, 32, 17, 2),
    ] {
        let model = model(&cfg);
        for prefix in [1, 3, 65] {
            let capacity = prefix + 4;
            let mut actual = cache(&model, prefix, capacity, 0x1234 + prefix as u64);
            let mut expected = clone_cache(&actual);
            let mut scratch = Ernie45DecodeScratch::new(&cfg, capacity).unwrap();
            let stable = storage(&scratch);
            for step in 0..3 {
                poison(&mut scratch);
                let row = pseudo_random_vec(cfg.hidden_size, 0x7856 + step as u64);
                let position = [prefix as u32 + step * 3 + 5, step * 7 + 11, step * 13 + 17];
                let old = model
                    .original_kv_decode_step(&row, position, &mut expected)
                    .unwrap();
                let new = model
                    .kv_decode_step_with_scratch(&row, position, &mut actual, &mut scratch)
                    .unwrap();
                assert_eq!(new.len(), cfg.vocab_size);
                assert!(old.iter().any(|v| *v != 0.0));
                assert_finite_bits(new, &old);
                assert_cache(&actual, &expected);
                assert_eq!(actual.len, prefix + step as usize + 1);
                assert_eq!(scratch.test_decode_calls(), step as usize + 1);
                assert_eq!(storage(&scratch), stable);
            }
        }
    }
}

#[test]
fn ernie_decode_scratch_reuses_all_nineteen_buffers_with_poisoned_tails() {
    let cfg = config(13, 17, 4, 2, 6, 9, 2);
    let model = model(&cfg);
    let capacity = 131;
    let mut actual = cache(&model, 65, capacity, 17);
    let mut scratch = Ernie45DecodeScratch::new(&cfg, capacity).unwrap();
    let stable = storage(&scratch);
    assert_eq!(stable.len(), 19);
    let pairs: Vec<_> = stable.iter().map(|&(ptr, _, cap)| (ptr, cap)).collect();
    assert_eq!(scratch.test_storage_snapshot(), pairs);
    for (round, prefix) in [65, 1, 33, 3].into_iter().enumerate() {
        if round != 0 {
            actual.clear();
            model
                .kv_prefill(
                    &pseudo_random_vec(prefix * cfg.hidden_size, 91 + round as u64),
                    &positions(prefix),
                    &mut actual,
                )
                .unwrap();
        }
        poison_inactive(&mut actual);
        let mut expected = clone_cache(&actual);
        poison(&mut scratch);
        let row = pseudo_random_vec(cfg.hidden_size, 109 + round as u64);
        let position = [
            211 + round as u32,
            17 + round as u32 * 3,
            39 + round as u32 * 5,
        ];
        let old = model
            .original_kv_decode_step(&row, position, &mut expected)
            .unwrap();
        let new = model
            .kv_decode_step_with_scratch(&row, position, &mut actual, &mut scratch)
            .unwrap();
        assert_finite_bits(new, &old);
        assert_cache(&actual, &expected);
        assert_eq!(scratch.test_decode_calls(), round + 1);
        assert_eq!(storage(&scratch), stable);
        assert_eq!(scratch.test_storage_snapshot(), pairs);
    }
}

#[test]
fn ernie_decode_scratch_recomputes_rope_for_changed_model_semantics() {
    let cfg = config(13, 17, 4, 2, 8, 11, 2);
    let first = model(&cfg);
    let mut scratch = Ernie45DecodeScratch::new(&cfg, 16).unwrap();
    let stable = storage(&scratch);
    let row = pseudo_random_vec(cfg.hidden_size, 421);
    let mut first_cache = cache(&first, 3, 16, 331);
    first
        .kv_decode_step_with_scratch(&row, [7, 11, 19], &mut first_cache, &mut scratch)
        .unwrap();
    let mut changed = cfg.clone();
    changed.rope_theta = 500_000.0;
    changed.rope_scaling.mrope_section = vec![1, 2, 1];
    let mut weights = equiv_weights(&changed);
    weights.layers[0].q_proj[0] += 0.125;
    let second = Ernie45Model::new(changed.clone(), weights).unwrap();
    let mut actual = cache(&second, 3, 16, 337);
    let mut expected = clone_cache(&actual);
    let mut fresh_cache = clone_cache(&actual);
    let position = [23, 41, 67];
    let old = second
        .original_kv_decode_step(&row, position, &mut expected)
        .unwrap();
    let fresh = second
        .kv_decode_step_with_scratch(
            &row,
            position,
            &mut fresh_cache,
            &mut Ernie45DecodeScratch::new(&changed, 16).unwrap(),
        )
        .unwrap()
        .to_vec();
    let reused = second
        .kv_decode_step_with_scratch(&row, position, &mut actual, &mut scratch)
        .unwrap();
    assert_finite_bits(reused, &old);
    assert_finite_bits(&fresh, &old);
    assert_cache(&actual, &expected);
    assert_cache(&fresh_cache, &expected);
    assert_eq!(storage(&scratch), stable);
    assert_eq!(scratch.test_decode_calls(), 2);
}

#[test]
fn ernie_decode_scratch_rejects_mismatched_geometry_before_mutation() {
    let cfg = config(8, 11, 4, 2, 4, 7, 2);
    let model = model(&cfg);
    for case in 0..8 {
        let mut other = cfg.clone();
        let mut keys = 8;
        match case {
            0 => other.hidden_size += 1,
            1 => other.intermediate_size += 1,
            2 => other.num_hidden_layers += 1,
            3 => other.num_attention_heads += 2,
            4 => other.num_key_value_heads = 1,
            5 => {
                other.head_dim = 6;
                other.rope_scaling.mrope_section = vec![1, 1, 1];
            }
            6 => other.vocab_size += 1,
            7 => keys = 2,
            _ => unreachable!(),
        }
        let mut scratch = Ernie45DecodeScratch::new(&other, keys).unwrap();
        poison(&mut scratch);
        let before = scratch_state(&scratch);
        let mut actual = cache(&model, 3, 8, 203);
        let cache_before = cache_state(&actual);
        let storage_before = cache_storage(&actual);
        assert!(
            model
                .kv_decode_step_with_scratch(
                    &pseudo_random_vec(cfg.hidden_size, 307),
                    [7, 11, 19],
                    &mut actual,
                    &mut scratch
                )
                .is_err(),
            "case {case}"
        );
        assert_eq!(scratch_state(&scratch), before, "case {case}");
        assert_eq!(cache_state(&actual), cache_before, "case {case}");
        assert_eq!(cache_storage(&actual), storage_before, "case {case}");
    }
}

#[test]
fn ernie_decode_scratch_preserves_original_preflight_error_ordering() {
    let cfg = config(8, 11, 4, 2, 4, 7, 2);
    let model = model(&cfg);
    for case in 0..7 {
        let mut original =
            Ernie45KvCache::new(cfg.num_hidden_layers, cfg.kv_dim().unwrap(), 4).unwrap();
        original.layer_k.fill(71.0);
        original.layer_v.fill(73.0);
        original.len = 1;
        let mut row = vec![0.25; cfg.hidden_size];
        match case {
            0 => {
                original.len = 0;
                row.clear();
            }
            1 => {
                original.len = original.capacity;
                original.layers += 1;
                row.clear();
            }
            2 => {
                original.layers += 1;
                row.clear();
            }
            3 => {
                let _ = row.pop();
                row[0] = f32::NAN;
            }
            4 => row[0] = f32::NAN,
            5 => row[0] = f32::INFINITY,
            6 => row[0] = f32::NEG_INFINITY,
            _ => unreachable!(),
        }
        let mut old_cache = clone_cache(&original);
        let old = model
            .original_kv_decode_step(&row, [7, 11, 19], &mut old_cache)
            .unwrap_err();
        assert_eq!(cache_state(&old_cache), cache_state(&original));
        let mut public_cache = clone_cache(&original);
        let public = model
            .kv_decode_step(&row, [7, 11, 19], &mut public_cache)
            .unwrap_err();
        assert_eq!(
            std::mem::discriminant(&public),
            std::mem::discriminant(&old)
        );
        assert_eq!(public.to_string(), old.to_string());
        assert_eq!(cache_state(&public_cache), cache_state(&original));
        let mut other = cfg.clone();
        other.hidden_size += 1;
        let mut scratch = Ernie45DecodeScratch::new(&other, 4).unwrap();
        poison(&mut scratch);
        let before = scratch_state(&scratch);
        let storage_before = cache_storage(&original);
        let cache_before = cache_state(&original);
        let error = model
            .kv_decode_step_with_scratch(&row, [7, 11, 19], &mut original, &mut scratch)
            .unwrap_err();
        assert_eq!(std::mem::discriminant(&error), std::mem::discriminant(&old));
        assert_eq!(error.to_string(), old.to_string(), "case {case}");
        assert_eq!(scratch_state(&scratch), before, "case {case}");
        assert_eq!(cache_state(&original), cache_before, "case {case}");
        assert_eq!(cache_storage(&original), storage_before, "case {case}");
    }
}

#[test]
fn ernie_decode_scratch_constructor_refuses_invalid_or_unaddressable_geometry() {
    let cfg = config(13, 17, 4, 2, 6, 9, 2);
    let keys = 131;
    let scratch = Ernie45DecodeScratch::new(&cfg, keys).unwrap();
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let d = cfg.head_dim;
    let expected = [
        h,
        h,
        q,
        kv,
        kv,
        q,
        h,
        cfg.intermediate_size,
        cfg.intermediate_size,
        keys,
        d,
        keys * d,
        d * keys,
        d,
        cfg.vocab_size,
        d / 2,
        d,
        d,
        d,
    ];
    let actual = storage(&scratch);
    assert_eq!(actual.len(), expected.len());
    for (&(_, len, cap), expected) in actual.iter().zip(expected) {
        assert_eq!(len, expected);
        assert!(cap >= len);
    }
    assert_eq!(scratch.test_decode_calls(), 0);
    for keys in [
        0,
        usize::MAX,
        isize::MAX as usize / (size_of::<f32>() * d) + 1,
    ] {
        assert!(Ernie45DecodeScratch::new(&cfg, keys).is_err());
    }
    for case in 0..9 {
        let mut invalid = cfg.clone();
        match case {
            0 => invalid.hidden_size = 0,
            1 => invalid.intermediate_size = usize::MAX,
            2 => invalid.num_hidden_layers = 0,
            3 => invalid.num_attention_heads = 3,
            4 => invalid.num_key_value_heads = 0,
            5 => invalid.head_dim = 3,
            6 => invalid.vocab_size = usize::MAX,
            7 => invalid.rope_scaling.mrope_section = vec![usize::MAX, 1],
            8 => invalid.rope_scaling.mrope_section = vec![1, 1, 1, 1],
            _ => unreachable!(),
        }
        assert!(
            Ernie45DecodeScratch::new(&invalid, 1).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn ernie_decode_scratch_public_wrapper_matches_frozen_owned_outputs() {
    let cfg = config(13, 17, 3, 3, 6, 9, 2);
    let model = model(&cfg);
    let mut actual = cache(&model, 3, 8, 97);
    let mut expected = clone_cache(&actual);
    let row = pseudo_random_vec(cfg.hidden_size, 121);
    let first = model
        .kv_decode_step(&row, [17, 21, 35], &mut actual)
        .unwrap();
    let first_bits = float_bits(&first);
    let old = model
        .original_kv_decode_step(&row, [17, 21, 35], &mut expected)
        .unwrap();
    assert_finite_bits(&first, &old);
    assert_cache(&actual, &expected);
    let second = model
        .kv_decode_step(&row, [19, 33, 51], &mut actual)
        .unwrap();
    let old = model
        .original_kv_decode_step(&row, [19, 33, 51], &mut expected)
        .unwrap();
    assert_finite_bits(&second, &old);
    assert_eq!(float_bits(&first), first_bits, "owned first result changed");
    assert_cache(&actual, &expected);
}

#[test]
fn ernie_decode_scratch_bit_comparison_rejects_nonfinite_inputs() {
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            std::panic::catch_unwind(|| assert_finite_bits(&[nonfinite], &[nonfinite])).is_err()
        );
        assert!(std::panic::catch_unwind(|| assert_finite_bits(&[nonfinite], &[1.0])).is_err());
        assert!(std::panic::catch_unwind(|| assert_finite_bits(&[1.0], &[nonfinite])).is_err());
    }
    assert_finite_bits(&[1.0, -2.0], &[1.0, -2.0]);
}
