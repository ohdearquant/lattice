//! Private synthetic-only experiment; no cache or adapter identity API.

use super::{build_model, test_config};
use crate::attention::gdn::{GatedDeltaNetState, qwen35_gdn_geometry};
use crate::model::qwen35::cache::{ForwardScratch, KvCache};
use crate::model::qwen35::eval::log_softmax_nll;
use crate::model::qwen35::model::Qwen35Model;
use crate::model::qwen35::weights::AttentionWeights;
use lattice_arch::qwen35_config::LayerType;

struct FixedFixture {
    model: Qwen35Model,
}

impl FixedFixture {
    fn new() -> Self {
        // build_model installs NoopLoraHook; no arbitrary model or mutable access is exposed.
        let mut model = build_model(test_config(), 0x3141_5926);
        for (attention, _) in &mut model.weights.layers {
            match attention {
                AttentionWeights::Linear(w) => {
                    for values in [
                        &mut w.in_proj_qkv,
                        &mut w.in_proj_z,
                        &mut w.out_proj,
                        &mut w.conv1d_weight,
                    ] {
                        for x in values {
                            *x *= 5.0;
                        }
                    }
                    w.norm_weight.fill(1.0);
                }
                AttentionWeights::Full(w) => {
                    for values in [&mut w.q_proj, &mut w.k_proj, &mut w.v_proj, &mut w.o_proj] {
                        for x in values {
                            *x *= 5.0;
                        }
                    }
                }
            }
        }
        Self { model }
    }
}

struct Prefix<'a> {
    model: &'a FixedFixture,
    kv: KvCache,
    gdn: Vec<GatedDeltaNetState>,
    logits: Vec<f32>,
}

impl<'a> Prefix<'a> {
    fn capture(model: &'a FixedFixture, doc: &[u32]) -> Result<Self, &'static str> {
        let cfg = &model.model.config;
        if doc.is_empty()
            || doc.len() >= model.model.max_context()
            || doc.iter().any(|&token| token as usize >= cfg.vocab_size)
            || cfg.layer_types.len() != cfg.num_hidden_layers
            || cfg.layer_mask.len() != cfg.num_hidden_layers
            || cfg.layer_mask.iter().any(|&enabled| !enabled)
            || model.model.weights.layers.len() != cfg.num_hidden_layers
            || cfg.layer_types.iter().zip(&model.model.weights.layers).any(
                |(kind, (weights, _))| {
                    !matches!(
                        (kind, weights),
                        (LayerType::LinearAttention, AttentionWeights::Linear(_))
                            | (LayerType::FullAttention, AttentionWeights::Full(_))
                    )
                },
            )
        {
            return Err("unsupported prefix or model state");
        }
        let geometry = qwen35_gdn_geometry(cfg).map_err(|_| "invalid GDN geometry")?;
        let mut gdn: Vec<_> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::from_geometry(&geometry))
            .collect();
        let mut kv = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();
        for (position, &token) in doc.iter().enumerate() {
            model
                .model
                .forward_step(token, position, &mut gdn, &mut kv, &mut scratch);
            kv.seq_len += 1;
        }
        let span = doc
            .len()
            .checked_mul(cfg.full_kv_dim())
            .ok_or("KV overflow")?;
        if kv.k.len() != cfg.num_full_attention_layers()
            || kv.v.len() != kv.k.len()
            || kv.k.iter().chain(&kv.v).any(|layer| layer.len() != span)
            || gdn.len() != cfg.num_linear_attention_layers()
            || scratch.logits.len() < cfg.vocab_size
        {
            return Err("incomplete continuation state");
        }
        let template = GatedDeltaNetState::from_geometry(&geometry);
        if gdn.iter().any(|state| {
            state.s_matrices.len() != template.s_matrices.len()
                || state.conv_buffer.len() != template.conv_buffer.len()
        }) {
            return Err("incomplete recurrent state");
        }
        Ok(Self {
            model,
            kv,
            gdn,
            logits: scratch.logits[..cfg.vocab_size].to_vec(),
        })
    }

    fn copy_state(&self) -> (KvCache, Vec<GatedDeltaNetState>) {
        (
            KvCache {
                k: self.kv.k.clone(),
                v: self.kv.v.clone(),
                seq_len: self.kv.seq_len,
            },
            self.gdn.clone(),
        )
    }

    fn score(&self, query: &[u32]) -> Result<Vec<f32>, &'static str> {
        let model = &self.model.model;
        if query.is_empty()
            || self
                .kv
                .seq_len
                .checked_add(query.len())
                .is_none_or(|n| n > model.max_context())
            || query
                .iter()
                .any(|&token| token as usize >= model.config.vocab_size)
        {
            return Err("unsupported query");
        }
        let (mut kv, mut gdn) = self.copy_state();
        let mut logits = self.logits.clone();
        let mut scratch = ForwardScratch::new();
        let mut nlls = Vec::with_capacity(query.len());
        for (i, &token) in query.iter().enumerate() {
            nlls.push(log_softmax_nll(&logits, token as usize));
            if i + 1 < query.len() {
                let pos = kv.seq_len;
                model.forward_step(token, pos, &mut gdn, &mut kv, &mut scratch);
                kv.seq_len += 1;
                logits.copy_from_slice(&scratch.logits[..model.config.vocab_size]);
            }
        }
        Ok(nlls)
    }

    fn advance_all(&self, query: &[u32]) -> (KvCache, Vec<GatedDeltaNetState>, Vec<f32>) {
        let (mut kv, mut gdn) = self.copy_state();
        let mut scratch = ForwardScratch::new();
        for &token in query {
            let pos = kv.seq_len;
            self.model
                .model
                .forward_step(token, pos, &mut gdn, &mut kv, &mut scratch);
            kv.seq_len += 1;
        }
        let logits = scratch.logits[..self.model.model.config.vocab_size].to_vec();
        (kv, gdn, logits)
    }

    fn after_first_logits(
        &self,
        token: u32,
        mut kv: KvCache,
        mut gdn: Vec<GatedDeltaNetState>,
    ) -> Vec<f32> {
        let mut scratch = ForwardScratch::new();
        let pos = kv.seq_len;
        self.model
            .model
            .forward_step(token, pos, &mut gdn, &mut kv, &mut scratch);
        scratch.logits[..self.model.model.config.vocab_size].to_vec()
    }
}

fn cold(model: &FixedFixture, doc: &[u32], query: &[u32]) -> Vec<f32> {
    let mut all = doc.to_vec();
    all.extend_from_slice(query);
    model.model.compute_token_nlls(&all).unwrap()[doc.len() - 1..].to_vec()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[derive(Debug, PartialEq, Eq)]
struct StateBits {
    cursor: usize,
    k: Vec<Vec<u32>>,
    v: Vec<Vec<u32>>,
    gdn: Vec<(Vec<u32>, Vec<u32>)>,
    logits: Vec<u32>,
}

fn state_bits(kv: &KvCache, gdn: &[GatedDeltaNetState], logits: &[f32]) -> StateBits {
    StateBits {
        cursor: kv.seq_len,
        k: kv.k.iter().map(|v| bits(v)).collect(),
        v: kv.v.iter().map(|v| bits(v)).collect(),
        gdn: gdn
            .iter()
            .map(|s| (bits(&s.s_matrices), bits(&s.conv_buffer)))
            .collect(),
        logits: bits(logits),
    }
}

fn snapshot(prefix: &Prefix<'_>) -> StateBits {
    state_bits(&prefix.kv, &prefix.gdn, &prefix.logits)
}

// Both paths advance the cursor after every token at this comparison boundary;
// compute_token_nlls returns NLLs only and leaves its last cursor unadvanced.
fn cold_final_state(model: &FixedFixture, tokens: &[u32]) -> StateBits {
    let cfg = &model.model.config;
    let geometry = qwen35_gdn_geometry(cfg).unwrap();
    let mut gdn: Vec<_> = (0..cfg.num_linear_attention_layers())
        .map(|_| GatedDeltaNetState::from_geometry(&geometry))
        .collect();
    let mut kv = KvCache::new(cfg.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();
    for &token in tokens {
        let pos = kv.seq_len;
        model
            .model
            .forward_step(token, pos, &mut gdn, &mut kv, &mut scratch);
        kv.seq_len += 1;
    }
    state_bits(&kv, &gdn, &scratch.logits[..cfg.vocab_size])
}

#[test]
fn two_suffixes_reuse_one_immutable_hybrid_prefix_exactly() {
    let model = FixedFixture::new();
    let doc = [1, 2, 3];
    let a = [4, 5, 7];
    let b = [6];
    let prefix = Prefix::capture(&model, &doc).unwrap();
    assert_eq!(prefix.kv.seq_len, 3);
    assert_eq!(prefix.kv.k.len(), 1);
    assert_eq!(prefix.gdn.len(), 3);
    let original = snapshot(&prefix);
    let a_cold = cold(&model, &doc, &a);
    let b_cold = cold(&model, &doc, &b);
    assert!(a_cold.iter().chain(&b_cold).all(|nll| nll.is_finite()));
    assert!(prefix.logits.iter().all(|logit| logit.is_finite()));
    assert_eq!(bits(&prefix.score(&a).unwrap()), bits(&a_cold));
    assert_eq!(snapshot(&prefix), original);
    assert_eq!(bits(&prefix.score(&b).unwrap()), bits(&b_cold));
    assert_eq!(snapshot(&prefix), original);
    assert_eq!(bits(&prefix.score(&b).unwrap()), bits(&b_cold));
    assert_eq!(bits(&prefix.score(&a).unwrap()), bits(&a_cold));
    assert_eq!(snapshot(&prefix), original);
    for query in [&a[..], &b[..]] {
        let (kv, gdn, logits) = prefix.advance_all(query);
        let mut all = doc.to_vec();
        all.extend_from_slice(query);
        assert_eq!(
            state_bits(&kv, &gdn, &logits),
            cold_final_state(&model, &all)
        );
    }
    assert_eq!(snapshot(&prefix), original);
    let mean = |values: &[f32]| {
        -(values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64) as f32
    };
    assert_eq!(
        mean(&prefix.score(&a).unwrap()).to_bits(),
        mean(&a_cold).to_bits()
    );
    assert_eq!(
        mean(&prefix.score(&b).unwrap()).to_bits(),
        mean(&b_cold).to_bits()
    );
}

#[test]
fn invalid_prefix_and_query_are_refused() {
    let model = FixedFixture::new();
    assert!(Prefix::capture(&model, &[]).is_err());
    assert!(Prefix::capture(&model, &[28]).is_err());
    assert!(Prefix::capture(&model, &vec![1; model.model.max_context()]).is_err());
    let prefix = Prefix::capture(&model, &[1, 2, 3]).unwrap();
    assert!(prefix.score(&[]).is_err());
    assert!(prefix.score(&[28]).is_err());
    assert!(prefix.score(&vec![1; model.model.max_context()]).is_err());
    let mut unsupported = FixedFixture::new();
    unsupported.model.config.layer_mask[0] = false;
    assert!(Prefix::capture(&unsupported, &[1]).is_err());
    let mut mismatched = FixedFixture::new();
    mismatched.model.config.layer_types[0] = LayerType::FullAttention;
    assert!(Prefix::capture(&mismatched, &[1]).is_err());
}

fn assert_changed(reference: &[f32], changed: &[f32], resource: &str) {
    let max = reference
        .iter()
        .zip(changed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max > 1e-7,
        "{resource} mutation had insufficient sensitivity: {max}"
    );
}

#[test]
fn corrupting_kv_changes_next_query_logits() {
    let model = FixedFixture::new();
    let prefix = Prefix::capture(&model, &[1, 2, 3]).unwrap();
    let (kv, gdn) = prefix.copy_state();
    let reference = prefix.after_first_logits(4, kv, gdn);
    let (mut kv, gdn) = prefix.copy_state();
    assert!(kv.v.iter().flatten().any(|value| *value != 0.0));
    for layer in kv.k.iter_mut().chain(&mut kv.v) {
        layer.fill(0.0);
    }
    assert_changed(&reference, &prefix.after_first_logits(4, kv, gdn), "KV");
}

#[test]
fn corrupting_s_changes_next_query_logits() {
    let model = FixedFixture::new();
    let prefix = Prefix::capture(&model, &[1, 2, 3]).unwrap();
    let (kv, gdn) = prefix.copy_state();
    let reference = prefix.after_first_logits(4, kv, gdn);
    let (kv, mut gdn) = prefix.copy_state();
    assert!(
        gdn.iter()
            .flat_map(|state| &state.s_matrices)
            .any(|value| *value != 0.0)
    );
    for state in &mut gdn {
        state.s_matrices.fill(0.0);
    }
    assert_changed(&reference, &prefix.after_first_logits(4, kv, gdn), "S");
}

#[test]
fn corrupting_conv_changes_next_query_logits() {
    let model = FixedFixture::new();
    let prefix = Prefix::capture(&model, &[1, 2, 3]).unwrap();
    let (kv, gdn) = prefix.copy_state();
    let reference = prefix.after_first_logits(4, kv, gdn);
    let (kv, mut gdn) = prefix.copy_state();
    assert!(
        gdn.iter()
            .flat_map(|state| &state.conv_buffer)
            .any(|value| *value != 0.0)
    );
    for state in &mut gdn {
        state.conv_buffer.fill(0.0);
    }
    assert_changed(&reference, &prefix.after_first_logits(4, kv, gdn), "conv");
}

#[test]
fn corrupting_final_doc_logits_changes_one_token_score() {
    let model = FixedFixture::new();
    let prefix = Prefix::capture(&model, &[1, 2, 3]).unwrap();
    let expected = prefix.score(&[6]).unwrap()[0];
    let mut bad_logits = prefix.logits.clone();
    bad_logits[6] += 10.0;
    let actual = log_softmax_nll(&bad_logits, 6);
    assert!((expected - actual).abs() > 1e-5);
}
