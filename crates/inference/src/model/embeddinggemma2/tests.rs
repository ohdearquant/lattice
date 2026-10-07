//! Tests for the EmbeddingGemma 2 text encoder.
//!
//! Every numeric test compares the model with a plain f64 implementation written in this
//! file from the reference model's definition. The reference helpers below call nothing from
//! the code under test: no shared kernels, no shared rope or mask logic.

use super::*;
use std::collections::HashMap;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod metal_parity;

const PREFIX: &str = "language_model.";

/// 4 layers (sliding, full, sliding, full), a sliding window of 2 and distinct rope bases so a
/// swap between layer types shows up. Sliding layers use 4 heads of width 4 over 2 key/value
/// heads; full layers use width 8 over 1.
const TINY_CONFIG: &str = r#"{
  "vocab_size": 32,
  "hidden_size": 8,
  "intermediate_size": 16,
  "num_hidden_layers": 4,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 4,
  "hidden_size_per_layer_input": 6,
  "embedding_dim": 10,
  "rms_norm_eps": 1e-6,
  "sliding_window": 2,
  "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
  "hidden_activation": "gelu_pytorch_tanh",
  "per_layer_config": {
    "1": {"head_dim": 8, "num_key_value_heads": 1},
    "3": {"head_dim": 8, "num_key_value_heads": 1}
  },
  "rope_parameters": {
    "full_attention": {"rope_theta": 1000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 100.0, "rope_type": "default"}
  }
}"#;

const IDS: [u32; 9] = [3, 17, 5, 5, 28, 1, 9, 22, 14];

// ---------------------------------------------------------------------------
// Synthetic checkpoint
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    /// Uniform in `[-1, 1)`.
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / (1u64 << 23) as f32 - 1.0
    }
}

/// Truncates to bf16 precision, so the value survives a bf16 round trip exactly.
fn bf16(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0xFFFF_0000)
}

fn rand_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| scale * rng.next()).collect()
}

type NamedTensor = (String, Vec<usize>, Vec<f32>);

struct Synth {
    rng: Lcg,
    prefix: String,
    tensors: Vec<NamedTensor>,
}

impl Synth {
    fn add(&mut self, name: &str, shape: &[usize], offset: f32, scale: f32) {
        let n: usize = shape.iter().product();
        let values = (0..n)
            .map(|_| bf16(offset + scale * self.rng.next()))
            .collect();
        self.tensors
            .push((format!("{}{name}", self.prefix), shape.to_vec(), values));
    }
}

fn synth_tensors(cfg: &EmbeddingGemma2Config, prefix: &str, seed: u64) -> Vec<NamedTensor> {
    let mut s = Synth {
        rng: Lcg(seed),
        prefix: prefix.to_string(),
        tensors: Vec::new(),
    };
    let hidden = cfg.hidden_size;
    let per_layer = cfg.hidden_size_per_layer_input;
    s.add("embed_tokens.weight", &[cfg.vocab_size, hidden], 0.0, 1.0);
    s.add(
        "ple.per_layer_model_projection.weight",
        &[cfg.num_hidden_layers * per_layer, hidden],
        0.0,
        0.5,
    );
    s.add(
        "ple.per_layer_projection_norm.weight",
        &[per_layer],
        1.0,
        0.2,
    );
    for l in 0..cfg.num_hidden_layers {
        let shape = cfg.layer_shapes[l];
        let q_dim = cfg.num_attention_heads * shape.head_dim;
        let kv_dim = shape.num_key_value_heads * shape.head_dim;
        let lp = format!("layers.{l}.");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            s.add(&format!("{lp}{norm}.weight"), &[hidden], 1.0, 0.2);
        }
        s.add(
            &format!("{lp}self_attn.q_proj.weight"),
            &[q_dim, hidden],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}self_attn.k_proj.weight"),
            &[kv_dim, hidden],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}self_attn.v_proj.weight"),
            &[kv_dim, hidden],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}self_attn.o_proj.weight"),
            &[hidden, q_dim],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}self_attn.q_norm.weight"),
            &[shape.head_dim],
            1.0,
            0.2,
        );
        s.add(
            &format!("{lp}self_attn.k_norm.weight"),
            &[shape.head_dim],
            1.0,
            0.2,
        );
        let ff = cfg.intermediate_size;
        s.add(
            &format!("{lp}mlp.gate_proj.weight"),
            &[ff, hidden],
            0.0,
            0.5,
        );
        s.add(&format!("{lp}mlp.up_proj.weight"), &[ff, hidden], 0.0, 0.5);
        s.add(
            &format!("{lp}mlp.down_proj.weight"),
            &[hidden, ff],
            0.0,
            0.5,
        );
        s.add(&format!("{lp}layer_scalar"), &[1], 0.9, 0.05);
        s.add(
            &format!("{lp}ple_block.per_layer_input_gate.weight"),
            &[per_layer, hidden],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}ple_block.per_layer_projection.weight"),
            &[hidden, per_layer],
            0.0,
            0.5,
        );
        s.add(
            &format!("{lp}ple_block.post_per_layer_input_norm.weight"),
            &[hidden],
            1.0,
            0.2,
        );
    }
    s.add("norm.weight", &[hidden], 1.0, 0.2);
    s.add(
        "embedding_projection.weight",
        &[cfg.embedding_dim, hidden],
        0.0,
        0.5,
    );
    s.tensors
}

/// A safetensors buffer holding BF16 tensors, except for the names in `f32_names`, which are
/// stored as F32.
fn safetensors_bytes(tensors: &[NamedTensor], f32_names: &[&str]) -> Vec<u8> {
    let mut entries = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = payload.len();
        let is_f32 = f32_names.contains(&name.as_str());
        for v in values {
            if is_f32 {
                payload.extend_from_slice(&v.to_le_bytes());
            } else {
                payload.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        let dtype = if is_f32 { "F32" } else { "BF16" };
        entries.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{start},{}]}}"#,
            dims.join(","),
            payload.len()
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    out
}

fn tiny_cfg() -> EmbeddingGemma2Config {
    EmbeddingGemma2Config::from_config_json_str(TINY_CONFIG).expect("tiny config parses")
}

struct Fixture {
    cfg: EmbeddingGemma2Config,
    tensors: HashMap<String, Vec<f32>>,
    model: EmbeddingGemma2Model,
}

impl Fixture {
    fn w(&self, name: &str) -> &[f32] {
        let key = format!("{PREFIX}{name}");
        self.tensors
            .get(&key)
            .unwrap_or_else(|| panic!("fixture has no tensor {key}"))
    }
}

fn fixture() -> Fixture {
    let cfg = tiny_cfg();
    let list = synth_tensors(&cfg, PREFIX, 7);
    fixture_from(cfg, list)
}

/// A fixture whose per-layer signal is tiny, so the RMSNorm epsilon is comparable to the mean
/// square it normalizes. Away from that regime a norm cancels any constant factor in front of it,
/// so the `hidden^-0.5` factor of the per-layer projection cannot show up in the output.
fn fixture_with_small_per_layer_signal() -> Fixture {
    let cfg = tiny_cfg();
    let mut list = synth_tensors(&cfg, PREFIX, 7);
    for (name, _, values) in &mut list {
        let factor = if name.ends_with("ple.per_layer_model_projection.weight") {
            SMALL_PLE_PROJECTION
        } else if name.ends_with("ple_block.per_layer_projection.weight") {
            SMALL_PLE_BLOCK_PROJECTION
        } else {
            continue;
        };
        for v in values.iter_mut() {
            *v = bf16(*v * factor);
        }
    }
    fixture_from(cfg, list)
}

const SMALL_PLE_PROJECTION: f32 = 2.6e-4;
const SMALL_PLE_BLOCK_PROJECTION: f32 = 3e-3;

fn fixture_from(cfg: EmbeddingGemma2Config, list: Vec<NamedTensor>) -> Fixture {
    let mut file = SafetensorsFile::from_bytes(safetensors_bytes(&list, &[]))
        .expect("synthetic checkpoint is valid safetensors");
    let weights = load_weights(&mut file, &cfg).expect("synthetic checkpoint loads");
    let tensors = list.into_iter().map(|(n, _, v)| (n, v)).collect();
    Fixture {
        cfg: cfg.clone(),
        tensors,
        model: EmbeddingGemma2Model {
            cfg,
            weights,
            tokenizer: None,
            max_tokens: Some(EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS),
        },
    }
}

// ---------------------------------------------------------------------------
// Scalar reference (f64)
// ---------------------------------------------------------------------------

const EPS: f64 = 1e-6;

/// `x [rows, inn] * w[out, inn]^T`.
fn lin(x: &[f64], rows: usize, inn: usize, w: &[f32], out: usize) -> Vec<f64> {
    assert_eq!(x.len(), rows * inn);
    assert_eq!(w.len(), out * inn);
    let mut y = vec![0.0; rows * out];
    for r in 0..rows {
        for o in 0..out {
            let mut s = 0.0;
            for i in 0..inn {
                s += x[r * inn + i] * f64::from(w[o * inn + i]);
            }
            y[r * out + o] = s;
        }
    }
    y
}

/// `x * (mean(x^2) + eps)^-0.5 * weight`, with the weight used as given.
fn rms(x: &[f64], weight: Option<&[f32]>) -> Vec<f64> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = (mean_sq + EPS).powf(-0.5);
    x.iter()
        .enumerate()
        .map(|(i, v)| v * inv * weight.map_or(1.0, |w| f64::from(w[i])))
        .collect()
}

fn rms_rows(x: &[f64], width: usize, weight: Option<&[f32]>) -> Vec<f64> {
    x.chunks(width).flat_map(|row| rms(row, weight)).collect()
}

fn gelu_tanh(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// `x * cos + rotate_half(x) * sin` for one head vector at position `pos`.
fn rope(x: &mut [f64], pos: usize, theta: f64) {
    let hd = x.len();
    let half = hd / 2;
    let original = x.to_vec();
    for i in 0..half {
        let angle = pos as f64 * theta.powf(-(2.0 * i as f64) / hd as f64);
        let (s, c) = angle.sin_cos();
        // rotate_half(x) = cat(-x[half..], x[..half])
        x[i] = original[i] * c - original[half + i] * s;
        x[half + i] = original[half + i] * c + original[i] * s;
    }
}

fn to64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

/// Per-layer signal for every layer, laid out `[t, layers, per_layer]` as the reference reshapes it.
fn ref_ple(fx: &Fixture, emb: &[f64], t: usize) -> Vec<f64> {
    ref_ple_scaled(fx, emb, t, (fx.cfg.hidden_size as f64).powf(-0.5))
}

fn ref_ple_scaled(fx: &Fixture, emb: &[f64], t: usize, scale: f64) -> Vec<f64> {
    let cfg = &fx.cfg;
    let (hidden, p, nl) = (
        cfg.hidden_size,
        cfg.hidden_size_per_layer_input,
        cfg.num_hidden_layers,
    );
    let mut proj = lin(
        emb,
        t,
        hidden,
        fx.w("ple.per_layer_model_projection.weight"),
        nl * p,
    );
    for v in proj.iter_mut() {
        *v *= scale;
    }
    rms_rows(&proj, p, Some(fx.w("ple.per_layer_projection_norm.weight")))
}

fn layer_signal(fx: &Fixture, ple: &[f64], t: usize, l: usize) -> Vec<f64> {
    let (p, nl) = (fx.cfg.hidden_size_per_layer_input, fx.cfg.num_hidden_layers);
    (0..t)
        .flat_map(|r| ple[(r * nl + l) * p..(r * nl + l) * p + p].to_vec())
        .collect()
}

/// One encoder layer on `h [t, hidden]` with this layer's per-layer signal `pli [t, per_layer]`.
fn ref_layer(fx: &Fixture, l: usize, h: &[f64], pli: &[f64]) -> Vec<f64> {
    let cfg = &fx.cfg;
    let hidden = cfg.hidden_size;
    let p = cfg.hidden_size_per_layer_input;
    let heads = cfg.num_attention_heads;
    let shape = cfg.layer_shapes[l];
    let (hd, kvh) = (shape.head_dim, shape.num_key_value_heads);
    let t = h.len() / hidden;
    let kind = cfg.layer_types[l];
    let theta = match kind {
        EmbeddingGemma2LayerKind::Sliding => cfg.rope_theta_sliding,
        EmbeddingGemma2LayerKind::Full => cfg.rope_theta_full,
    };
    let g = |n: &str| fx.w(&format!("layers.{l}.{n}"));

    // Attention.
    let x = rms_rows(h, hidden, Some(g("input_layernorm.weight")));
    let q = lin(&x, t, hidden, g("self_attn.q_proj.weight"), heads * hd);
    let k = lin(&x, t, hidden, g("self_attn.k_proj.weight"), kvh * hd);
    let v = lin(&x, t, hidden, g("self_attn.v_proj.weight"), kvh * hd);
    let mut qn = vec![0.0; q.len()];
    let mut kn = vec![0.0; k.len()];
    let mut vn = vec![0.0; v.len()];
    for r in 0..t {
        for hh in 0..heads {
            let b = (r * heads + hh) * hd;
            let mut head = rms(&q[b..b + hd], Some(g("self_attn.q_norm.weight")));
            rope(&mut head, r, theta);
            qn[b..b + hd].copy_from_slice(&head);
        }
        for hh in 0..kvh {
            let b = (r * kvh + hh) * hd;
            let mut head = rms(&k[b..b + hd], Some(g("self_attn.k_norm.weight")));
            rope(&mut head, r, theta);
            kn[b..b + hd].copy_from_slice(&head);
            vn[b..b + hd].copy_from_slice(&rms(&v[b..b + hd], None));
        }
    }
    let mut ctx = vec![0.0; t * heads * hd];
    for hh in 0..heads {
        let kv = hh / (heads / kvh);
        for i in 0..t {
            let mut js = Vec::new();
            let mut scores = Vec::new();
            for j in 0..t {
                let allowed = match kind {
                    EmbeddingGemma2LayerKind::Full => true,
                    EmbeddingGemma2LayerKind::Sliding => {
                        (i as i64 - j as i64).abs() <= cfg.sliding_window as i64
                    }
                };
                if allowed {
                    let qb = (i * heads + hh) * hd;
                    let kb = (j * kvh + kv) * hd;
                    let dot: f64 = (0..hd).map(|d| qn[qb + d] * kn[kb + d]).sum();
                    js.push(j);
                    scores.push(dot);
                }
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for (&j, e) in js.iter().zip(&exps) {
                let vb = (j * kvh + kv) * hd;
                for d in 0..hd {
                    ctx[(i * heads + hh) * hd + d] += e / sum * vn[vb + d];
                }
            }
        }
    }
    let attn = lin(&ctx, t, heads * hd, g("self_attn.o_proj.weight"), hidden);
    let attn = rms_rows(&attn, hidden, Some(g("post_attention_layernorm.weight")));
    let h1: Vec<f64> = h.iter().zip(&attn).map(|(a, b)| a + b).collect();

    // Feed-forward.
    let x = rms_rows(&h1, hidden, Some(g("pre_feedforward_layernorm.weight")));
    let ff = cfg.intermediate_size;
    let gate = lin(&x, t, hidden, g("mlp.gate_proj.weight"), ff);
    let up = lin(&x, t, hidden, g("mlp.up_proj.weight"), ff);
    let act: Vec<f64> = gate
        .iter()
        .zip(&up)
        .map(|(a, b)| gelu_tanh(*a) * b)
        .collect();
    let mlp = lin(&act, t, ff, g("mlp.down_proj.weight"), hidden);
    let mlp = rms_rows(&mlp, hidden, Some(g("post_feedforward_layernorm.weight")));
    let h2: Vec<f64> = h1.iter().zip(&mlp).map(|(a, b)| a + b).collect();

    // Per-layer input block, then the layer scalar.
    let gate = lin(
        &h2,
        t,
        hidden,
        g("ple_block.per_layer_input_gate.weight"),
        p,
    );
    let gated: Vec<f64> = gate
        .iter()
        .zip(pli)
        .map(|(a, b)| gelu_tanh(*a) * b)
        .collect();
    let back = lin(
        &gated,
        t,
        p,
        g("ple_block.per_layer_projection.weight"),
        hidden,
    );
    let back = rms_rows(
        &back,
        hidden,
        Some(g("ple_block.post_per_layer_input_norm.weight")),
    );
    let scalar = f64::from(g("layer_scalar")[0]);
    h2.iter()
        .zip(&back)
        .map(|(a, b)| (a + b) * scalar)
        .collect()
}

fn ref_embeddings(fx: &Fixture, ids: &[u32]) -> Vec<f64> {
    let hidden = fx.cfg.hidden_size;
    let table = fx.w("embed_tokens.weight");
    let scale = (hidden as f64).sqrt();
    ids.iter()
        .flat_map(|&id| {
            table[id as usize * hidden..(id as usize + 1) * hidden]
                .iter()
                .map(move |&v| f64::from(v) * scale)
        })
        .collect()
}

/// Per-token states `[t, embedding_dim]`.
fn ref_states(fx: &Fixture, ids: &[u32]) -> Vec<f64> {
    ref_states_with_ple_scale(fx, ids, (fx.cfg.hidden_size as f64).powf(-0.5))
}

fn ref_states_with_ple_scale(fx: &Fixture, ids: &[u32], ple_scale: f64) -> Vec<f64> {
    let cfg = &fx.cfg;
    let t = ids.len();
    let emb = ref_embeddings(fx, ids);
    let ple = ref_ple_scaled(fx, &emb, t, ple_scale);
    let mut h = emb;
    for l in 0..cfg.num_hidden_layers {
        let pli = layer_signal(fx, &ple, t, l);
        h = ref_layer(fx, l, &h, &pli);
    }
    let h = rms_rows(&h, cfg.hidden_size, Some(fx.w("norm.weight")));
    lin(
        &h,
        t,
        cfg.hidden_size,
        fx.w("embedding_projection.weight"),
        cfg.embedding_dim,
    )
}

/// Mean over tokens, the leading `dim` entries, then L2.
fn ref_embedding(states: &[f64], dim_full: usize, dim: usize) -> Vec<f64> {
    let t = states.len() / dim_full;
    let mut mean = vec![0.0; dim_full];
    for row in states.chunks(dim_full) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += v / t as f64;
        }
    }
    mean.truncate(dim);
    let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
    mean.iter().map(|v| v / norm).collect()
}

fn max_diff(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs())
        .fold(0.0, f64::max)
}

fn max_diff32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// ---------------------------------------------------------------------------
// Forward parity with the scalar reference
// ---------------------------------------------------------------------------

fn layer_parity(l: usize) {
    let fx = fixture();
    let t = 9;
    let hidden = fx.cfg.hidden_size;
    let h0 = rand_vec(t * hidden, 11 + l as u64, 2.0);
    let emb = rand_vec(t * hidden, 99, 2.0);
    let (cos, sin) = fx.model.rope_table(l, t);
    let mut h = h0.clone();
    fx.model.run_layer(l, &mut h, &emb, &cos, &sin);

    let ple = ref_ple(&fx, &to64(&emb), t);
    let want = ref_layer(&fx, l, &to64(&h0), &layer_signal(&fx, &ple, t, l));
    let diff = max_diff(&h, &want);
    assert!(
        diff < 1e-5,
        "layer {l} differs from the scalar reference by {diff}"
    );
}

#[test]
fn sliding_layer_matches_scalar_reference() {
    layer_parity(0);
    layer_parity(2);
}

#[test]
fn full_layer_matches_scalar_reference() {
    layer_parity(1);
    layer_parity(3);
}

#[test]
fn whole_model_matches_scalar_reference() {
    let fx = fixture();
    let states = fx.model.token_states(&IDS).expect("forward runs");
    let want = ref_states(&fx, &IDS);
    let diff = max_diff(&states, &want);
    assert!(
        diff < 1e-5,
        "token states differ from the scalar reference by {diff}"
    );

    for dim in [10, 8, 5] {
        let got = fx.model.encode_ids(&IDS, Some(dim)).expect("encode runs");
        let want = ref_embedding(&want, fx.cfg.embedding_dim, dim);
        let diff = max_diff(&got, &want);
        assert!(
            diff < 1e-5,
            "embedding at {dim} differs from the reference by {diff}"
        );
    }
}

#[test]
fn the_per_layer_projection_scale_is_applied() {
    let fx = fixture_with_small_per_layer_signal();
    let right = (fx.cfg.hidden_size as f64).powf(-0.5);
    let want = ref_states_with_ple_scale(&fx, &IDS, right);
    let without_scale = ref_states_with_ple_scale(&fx, &IDS, 1.0);

    // Control: in this fixture a missing scale changes the reference output by a wide margin.
    let sensitivity = want
        .iter()
        .zip(&without_scale)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        sensitivity > 1e-2,
        "fixture is not sensitive to the per-layer scale (reference moves by {sensitivity})"
    );

    let got = fx.model.token_states(&IDS).expect("forward runs");
    let diff = max_diff(&got, &want);
    assert!(
        diff < 1e-5,
        "token states differ from the scaled scalar reference by {diff}"
    );
}

// ---------------------------------------------------------------------------
// Attention mask
// ---------------------------------------------------------------------------

/// Largest change in output row `i` of layer `l` when input row `j` is shifted.
fn influence(fx: &Fixture, l: usize, i: usize, j: usize) -> f32 {
    let t = 9;
    let hidden = fx.cfg.hidden_size;
    let h0 = rand_vec(t * hidden, 5, 1.5);
    let emb = rand_vec(t * hidden, 6, 1.5);
    let (cos, sin) = fx.model.rope_table(l, t);
    let mut base = h0.clone();
    fx.model.run_layer(l, &mut base, &emb, &cos, &sin);
    let mut moved = h0;
    for v in &mut moved[j * hidden..(j + 1) * hidden] {
        *v += 0.5;
    }
    fx.model.run_layer(l, &mut moved, &emb, &cos, &sin);
    max_diff32(
        &base[i * hidden..(i + 1) * hidden],
        &moved[i * hidden..(i + 1) * hidden],
    )
}

#[test]
fn sliding_layer_attends_exactly_to_keys_within_the_window() {
    let fx = fixture();
    let w = fx.cfg.sliding_window;
    assert_eq!(w, 2);
    for l in [0, 2] {
        assert_eq!(fx.cfg.layer_types[l], EmbeddingGemma2LayerKind::Sliding);
        for i in 0..9usize {
            for j in 0..9usize {
                if i == j {
                    continue;
                }
                let dist = i.abs_diff(j);
                let diff = influence(&fx, l, i, j);
                if dist <= w {
                    assert!(
                        diff > 1e-4,
                        "layer {l}: key {j} at distance {dist} must influence query {i}, saw {diff}"
                    );
                } else {
                    assert!(
                        diff < 1e-6,
                        "layer {l}: key {j} at distance {dist} must not influence query {i}, saw {diff}"
                    );
                }
            }
        }
    }
}

#[test]
fn full_layer_attends_to_every_key() {
    let fx = fixture();
    for l in [1, 3] {
        assert_eq!(fx.cfg.layer_types[l], EmbeddingGemma2LayerKind::Full);
        for i in 0..9usize {
            for j in 0..9usize {
                if i == j {
                    continue;
                }
                let diff = influence(&fx, l, i, j);
                assert!(
                    diff > 1e-4,
                    "layer {l}: key {j} must influence query {i} at distance {}, saw {diff}",
                    i.abs_diff(j)
                );
            }
        }
    }
}

#[test]
fn attention_is_not_causal() {
    let fx = fixture();
    let a = fx.model.token_states(&IDS).expect("forward runs");
    let mut changed = IDS;
    changed[8] = 30;
    let b = fx.model.token_states(&changed).expect("forward runs");
    let dim = fx.cfg.embedding_dim;
    let first_row = max_diff32(&a[..dim], &b[..dim]);
    assert!(
        first_row > 1e-4,
        "changing the last token must change the first token's state, saw {first_row}"
    );
}

/// Naive masked attention in f64 over `[seq, heads, hd]` inputs.
#[allow(clippy::too_many_arguments)]
fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    kvh: usize,
    hd: usize,
    window: Option<usize>,
) -> Vec<f64> {
    let mut out = vec![0.0; seq * heads * hd];
    for hh in 0..heads {
        let kv = hh / (heads / kvh);
        for i in 0..seq {
            let allowed: Vec<usize> = (0..seq)
                .filter(|&j| window.is_none_or(|w| i.abs_diff(j) <= w))
                .collect();
            let scores: Vec<f64> = allowed
                .iter()
                .map(|&j| {
                    (0..hd)
                        .map(|d| {
                            f64::from(q[(i * heads + hh) * hd + d])
                                * f64::from(k[(j * kvh + kv) * hd + d])
                        })
                        .sum()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for (&j, e) in allowed.iter().zip(&exps) {
                for d in 0..hd {
                    out[(i * heads + hh) * hd + d] +=
                        e / sum * f64::from(v[(j * kvh + kv) * hd + d]);
                }
            }
        }
    }
    out
}

#[test]
fn attention_kernel_matches_naive_across_tile_edges_and_windows() {
    // 300 positions span three query tiles, so tile boundaries fall inside every window below.
    let (seq, heads, kvh, hd) = (300, 4, 2, 8);
    let q = rand_vec(seq * heads * hd, 1, 1.0);
    let k = rand_vec(seq * kvh * hd, 2, 1.0);
    let v = rand_vec(seq * kvh * hd, 3, 1.0);
    for window in [
        None,
        Some(0),
        Some(1),
        Some(50),
        Some(127),
        Some(128),
        Some(400),
    ] {
        let got = bidirectional_attention(&q, &k, &v, seq, heads, kvh, hd, window);
        let want = naive_attention(&q, &k, &v, seq, heads, kvh, hd, window);
        let diff = max_diff(&got, &want);
        assert!(
            diff < 1e-5,
            "window {window:?}: differs from naive by {diff}"
        );
    }
}

// ---------------------------------------------------------------------------
// Pooling, truncation, limits
// ---------------------------------------------------------------------------

#[test]
fn pooling_covers_every_token_including_the_prompt() {
    let fx = fixture();
    let dim = fx.cfg.embedding_dim;
    let states = to64(&fx.model.token_states(&IDS).expect("forward runs"));
    let all = ref_embedding(&states, dim, dim);
    let got = fx.model.encode_ids(&IDS, None).expect("encode runs");
    assert!(max_diff(&got, &all) < 1e-5);

    // Dropping the first three tokens (a stand-in for a task prompt) must change the result.
    let without_prompt = ref_embedding(&states[3 * dim..], dim, dim);
    assert!(
        max_diff(&got, &without_prompt) > 1e-3,
        "the leading tokens must contribute to the pooled embedding"
    );
}

#[test]
fn embeddings_are_unit_length_at_every_width() {
    let fx = fixture();
    for dim in [None, Some(10), Some(8), Some(5), Some(1)] {
        let e = fx.model.encode_ids(&IDS, dim).expect("encode runs");
        assert_eq!(e.len(), dim.unwrap_or(10));
        let norm = e.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "width {dim:?} has norm {norm}");
    }
}

#[test]
fn matryoshka_truncation_is_the_renormalized_prefix() {
    let fx = fixture();
    let full = to64(&fx.model.encode_ids(&IDS, None).expect("encode runs"));
    for dim in [8usize, 5, 3] {
        let got = fx.model.encode_ids(&IDS, Some(dim)).expect("encode runs");
        let prefix = &full[..dim];
        let norm = prefix.iter().map(|v| v * v).sum::<f64>().sqrt();
        let want: Vec<f64> = prefix.iter().map(|v| v / norm).collect();
        assert!(max_diff(&got, &want) < 1e-6, "width {dim}");
        // Without the re-normalization the prefix would be shorter than a unit vector.
        assert!(
            norm < 1.0 - 1e-3,
            "width {dim}: fixture prefix must be strictly shorter"
        );
    }
}

#[test]
fn the_forward_pass_has_no_length_limit_of_its_own() {
    let fx = fixture();
    let long: Vec<u32> = (0..EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS + 37)
        .map(|i| (i * 7 % 32) as u32)
        .collect();
    let all = fx
        .model
        .encode_ids(&long, None)
        .expect("a sequence past the default limit embeds whole");
    let cut = fx
        .model
        .encode_ids(&long[..EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS], None)
        .expect("a sequence at the default limit embeds");
    assert_ne!(all, cut, "tokens past the default limit must be pooled");
}

#[test]
fn invalid_inputs_are_rejected() {
    let fx = fixture();
    assert!(matches!(
        fx.model.encode_ids(&[], None),
        Err(InferenceError::InvalidInput(_))
    ));
    assert!(matches!(
        fx.model.encode_ids(&[1, 32], None),
        Err(InferenceError::InvalidInput(_))
    ));
    for dim in [0usize, 11] {
        assert!(matches!(
            fx.model.encode_ids(&IDS, Some(dim)),
            Err(InferenceError::InvalidInput(_))
        ));
    }
    assert!(matches!(
        fx.model.tokenize("x"),
        Err(InferenceError::Tokenizer(_))
    ));
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

#[test]
fn loader_finds_the_text_tower_under_any_known_prefix() {
    let cfg = tiny_cfg();
    for prefix in ["model.language_model.", "model.", ""] {
        let list = synth_tensors(&cfg, prefix, 7);
        let mut file = SafetensorsFile::from_bytes(safetensors_bytes(&list, &[])).expect("valid");
        assert!(load_weights(&mut file, &cfg).is_ok(), "prefix {prefix:?}");
    }
    let list = synth_tensors(&cfg, "elsewhere.", 7);
    let mut file = SafetensorsFile::from_bytes(safetensors_bytes(&list, &[])).expect("valid");
    assert!(matches!(
        load_weights(&mut file, &cfg),
        Err(InferenceError::MissingTensor(_))
    ));
}

#[test]
fn loader_rejects_a_missing_tensor_a_wrong_shape_and_a_wrong_dtype() {
    let cfg = tiny_cfg();

    let mut list = synth_tensors(&cfg, PREFIX, 7);
    list.retain(|(n, _, _)| !n.ends_with("layers.2.mlp.up_proj.weight"));
    let mut file = SafetensorsFile::from_bytes(safetensors_bytes(&list, &[])).expect("valid");
    assert!(matches!(
        load_weights(&mut file, &cfg),
        Err(InferenceError::MissingTensor(_))
    ));

    let mut list = synth_tensors(&cfg, PREFIX, 7);
    for (name, shape, values) in &mut list {
        if name.ends_with("layers.1.self_attn.k_proj.weight") {
            // The full layers use one key/value head of width 8; two would be a Gemma 4 shape.
            *shape = vec![16, 8];
            *values = vec![0.0; 16 * 8];
        }
    }
    let mut file = SafetensorsFile::from_bytes(safetensors_bytes(&list, &[])).expect("valid");
    assert!(matches!(
        load_weights(&mut file, &cfg),
        Err(InferenceError::ShapeMismatch { .. })
    ));

    let list = synth_tensors(&cfg, PREFIX, 7);
    let target = format!("{PREFIX}norm.weight");
    let mut file =
        SafetensorsFile::from_bytes(safetensors_bytes(&list, &[target.as_str()])).expect("valid");
    assert!(matches!(
        load_weights(&mut file, &cfg),
        Err(InferenceError::Inference(_))
    ));
}

// ---------------------------------------------------------------------------
// Task prompts
// ---------------------------------------------------------------------------

#[test]
fn task_prefixes_match_the_published_prompts() {
    use EmbeddingGemma2Task::*;
    let expected = [
        (Query, "task: search result | query: "),
        (Document, "title: none | text: "),
        (CodeRetrieval, "task: code retrieval | query: "),
        (SentenceSimilarity, "task: sentence similarity | query: "),
        (Classification, "task: classification | query: "),
        (Clustering, "task: clustering | query: "),
        (QuestionAnswering, "task: question answering | query: "),
        (FactChecking, "task: fact checking | query: "),
    ];
    for (task, prefix) in expected {
        assert_eq!(task.prefix(), prefix);
        assert_eq!(task.format("hello"), format!("{prefix}hello"));
    }
    assert_eq!(
        format_titled_document("Rust", "A language."),
        "title: Rust | text: A language."
    );
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

/// A tokenizer.json in the Gemma shape with eleven entries: the five specials, `a b c d`, the
/// space marker, and one merge (`a` + `b`).
const TINY_TOKENIZER: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 4, "content": "<mask>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"},
  "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [{"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
    "special_tokens": {}
  },
  "decoder": {"type": "Sequence", "decoders": [
    {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
    {"type": "ByteFallback"},
    {"type": "Fuse"}
  ]},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": true,
    "byte_fallback": true,
    "ignore_merges": false,
    "vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4,
              "a": 5, "b": 6, "c": 7, "d": 8, "\u2581": 9, "ab": 10},
    "merges": [["a", "b"]]
  }
}"#;

const A: u32 = 5;
const C: u32 = 7;
const AB: u32 = 10;

fn fixture_with_tokenizer() -> Fixture {
    let mut fx = fixture();
    fx.model.tokenizer = Some(
        TextTokenizer::from_tokenizer_json_str(
            TINY_TOKENIZER,
            fx.cfg.bos_token_id,
            fx.cfg.eos_token_id,
        )
        .expect("tiny tokenizer loads"),
    );
    fx
}

#[test]
fn every_input_is_wrapped_in_the_beginning_and_end_tokens() {
    let fx = fixture_with_tokenizer();
    assert_eq!((fx.cfg.bos_token_id, fx.cfg.eos_token_id), (2, 1));
    assert_eq!(fx.model.tokenize("abc").unwrap(), vec![2, AB, C, 1]);
    assert_eq!(fx.model.tokenize("a").unwrap(), vec![2, A, 1]);
    assert_eq!(fx.model.tokenize("").unwrap(), vec![2, 1]);
}

#[test]
fn pooling_covers_the_beginning_and_end_tokens() {
    let fx = fixture_with_tokenizer();
    let got = fx.model.encode("abc", None).expect("encode runs");
    let wrapped = fx
        .model
        .encode_ids(&[2, AB, C, 1], None)
        .expect("encode runs");
    assert_eq!(got, wrapped);
    let bare = fx.model.encode_ids(&[AB, C], None).expect("encode runs");
    assert!(
        max_diff32(&got, &bare) > 1e-3,
        "the wrapping tokens must contribute to the pooled embedding"
    );
}

#[test]
fn the_token_limit_keeps_both_wrapping_tokens_and_cuts_the_text() {
    let text = "a".repeat(20);
    let fx = fixture_with_tokenizer();

    let five = fixture_with_tokenizer()
        .model
        .with_max_tokens(Some(5))
        .expect("limit of 5 is valid");
    assert_eq!(five.tokenize(&text).unwrap(), vec![2, A, A, A, 1]);
    let two = fixture_with_tokenizer()
        .model
        .with_max_tokens(Some(2))
        .expect("limit of 2 is valid");
    assert_eq!(two.tokenize(&text).unwrap(), vec![2, 1]);
    // A text that already fits is untouched.
    assert_eq!(five.tokenize("ab").unwrap(), vec![2, AB, 1]);

    for n in [0usize, 1] {
        assert!(matches!(
            fixture_with_tokenizer().model.with_max_tokens(Some(n)),
            Err(InferenceError::InvalidInput(_))
        ));
    }

    // Default: the 8192 limit counts the wrapping tokens and truncates instead of failing.
    let long = "a".repeat(EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS + 100);
    let capped = fx
        .model
        .tokenize(&long)
        .expect("over-long text is truncated");
    assert_eq!(capped.len(), EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS);
    assert_eq!((capped.first(), capped.last()), (Some(&2), Some(&1)));

    // No limit: every token is kept.
    let unlimited = fx
        .model
        .with_max_tokens(None)
        .expect("no limit is valid")
        .tokenize(&long)
        .expect("tokenizes");
    assert_eq!(unlimited.len(), long.len() + 2);
}
