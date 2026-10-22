use super::*;
use lattice_inference::backward::tape::swiglu_forward;
use lattice_inference::model::qwen35_config::LayerType;

// The FFN block of `forward_full` runs as one sequence-level GEMM per linear map. The
// reference is the single-position `swiglu_forward`, applied to the normalised rows
// rebuilt from the `h_mid` the forward stored. `matmul_bt` reassociates the dot products,
// so parity is a relative tolerance (same constant and error formula as the parity tests
// in `lattice_inference::backward::ops`), not bit-exact.
const PARITY_TOL: f64 = 1e-4;

const HIDDEN: usize = 21;
const INTER: usize = 37;
const VOCAB: usize = 13;
const NUM_Q_HEADS: usize = 3;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 8;
const ROPE_DIM: usize = 4;
const RANK: usize = 3;
const SEQ: usize = 9;
const COMPLETION_START: usize = 5;

fn rel_err(reference: &[f32], got: &[f32]) -> f64 {
    assert_eq!(reference.len(), got.len(), "length mismatch");
    let diff_sq: f64 = reference
        .iter()
        .zip(got.iter())
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    let norm_sq: f64 = reference.iter().map(|&a| (a as f64).powi(2)).sum();
    (diff_sq / norm_sq.max(1e-30)).sqrt()
}

fn assert_parity(what: &str, reference: &[f32], got: &[f32]) {
    let err = rel_err(reference, got);
    assert!(
        err < PARITY_TOL,
        "{what}: sequence-level vs per-position rel_err {err:.2e} >= {PARITY_TOL:.2e}"
    );
}

struct TinyLayer {
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    pre_shift: Vec<f32>,
    post_shift: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

fn tiny_layer(rng: &mut u64) -> TinyLayer {
    let q_dim = NUM_Q_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    TinyLayer {
        w_q: rand_fill(rng, 2 * q_dim * HIDDEN, 0.3),
        w_k: rand_fill(rng, kv_dim * HIDDEN, 0.3),
        w_v: rand_fill(rng, kv_dim * HIDDEN, 0.3),
        w_o: rand_fill(rng, HIDDEN * q_dim, 0.3),
        q_norm: rand_fill(rng, HEAD_DIM, 0.1),
        k_norm: rand_fill(rng, HEAD_DIM, 0.1),
        pre_shift: shifted(&rand_fill(rng, HIDDEN, 0.1)),
        post_shift: shifted(&rand_fill(rng, HIDDEN, 0.1)),
        w_gate: rand_fill(rng, INTER * HIDDEN, 0.3),
        w_up: rand_fill(rng, INTER * HIDDEN, 0.3),
        w_down: rand_fill(rng, HIDDEN * INTER, 0.3),
    }
}

fn layer_w(w: &TinyLayer, lora_slot: Option<usize>) -> LayerW<'_> {
    LayerW {
        kind: MixerKind::Gqa,
        w_q: &w.w_q,
        w_k: &w.w_k,
        w_v: &w.w_v,
        w_o: &w.w_o,
        q_norm: &w.q_norm,
        k_norm: &w.k_norm,
        gdn: None,
        pre_shift: w.pre_shift.clone(),
        post_shift: w.post_shift.clone(),
        w_gate: &w.w_gate,
        w_up: &w.w_up,
        w_down: &w.w_down,
        lora_slot,
    }
}

// A two-layer all-GQA model small enough to run in a unit test. Layer 0 is frozen and
// non-terminal; layer 1 is terminal and carries the only LoRA slot.
fn tiny_cfg() -> Qwen35Config {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.hidden_size = HIDDEN;
    cfg.num_hidden_layers = 2;
    cfg.vocab_size = VOCAB;
    cfg.intermediate_size = INTER;
    cfg.num_attention_heads = NUM_Q_HEADS;
    cfg.num_key_value_heads = NUM_KV_HEADS;
    cfg.head_dim = HEAD_DIM;
    cfg.partial_rotary_factor = 0.5;
    cfg.full_attention_interval = 1;
    cfg.layer_types = vec![LayerType::FullAttention; 2];
    cfg.layer_mask = vec![true; 2];
    cfg
}

fn tiny_dims(cfg: &Qwen35Config) -> Dims {
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
    assert_eq!(dims.rope_dim, ROPE_DIM);
    dims
}

fn rope_tables() -> (Vec<f32>, Vec<f32>) {
    let half = ROPE_DIM / 2;
    let angle =
        |i: usize| (i / half) as f32 / 10000f32.powf(2.0 * (i % half) as f32 / ROPE_DIM as f32);
    let cos = (0..SEQ * half).map(|i| angle(i).cos()).collect();
    let sin = (0..SEQ * half).map(|i| angle(i).sin()).collect();
    (cos, sin)
}

#[test]
fn rows_by_position_leaves_uncomputed_positions_empty() {
    let width = 3;
    let range = 2..5;
    let flat: Vec<f32> = (0..range.len() * width).map(|i| i as f32 + 1.0).collect();
    let rows = rows_by_position(flat.clone(), &range, 7, width);
    assert_eq!(rows.len(), 7);
    for t in 0..7 {
        if range.contains(&t) {
            let i = t - range.start;
            assert_eq!(rows[t], flat[i * width..(i + 1) * width], "row {t}");
        } else {
            assert!(rows[t].is_empty(), "position {t} was not computed");
        }
    }
    let none = rows_by_position(Vec::new(), &(4..4), 7, width);
    assert!(none.iter().all(Vec::is_empty));
}

#[test]
fn forward_full_ffn_matches_per_position_reference() {
    let cfg = tiny_cfg();
    let dims = tiny_dims(&cfg);
    let gdn_dims = GdnDims::from_cfg(&cfg);
    let slot_layers = [1usize];
    let no_gdn_slots: [usize; 0] = [];
    let train = TrainCtx::try_new(
        TapeGeometry::new(&dims, &gdn_dims, &cfg),
        RANK,
        2.0 * RANK as f32,
        SlotLayout::new(&slot_layers, &no_gdn_slots),
        AdamConfig::new(1e-3, 0.9, 0.999, 1e-8),
    )
    .expect("tiny geometry is valid");

    let mut rng = 0x5EED_1234_u64;
    let w0 = tiny_layer(&mut rng);
    let w1 = tiny_layer(&mut rng);
    let layers = [layer_w(&w0, None), layer_w(&w1, Some(0))];
    let q_dim = dims.q_dim;
    let loras = [LoraParams {
        a_q: rand_fill(&mut rng, RANK * HIDDEN, 0.2),
        b_q: rand_fill(&mut rng, 2 * q_dim * RANK, 0.2),
        a_v: rand_fill(&mut rng, RANK * HIDDEN, 0.2),
        b_v: rand_fill(&mut rng, dims.kv_dim * RANK, 0.2),
    }];
    let (cos, sin) = rope_tables();
    let ctx = SeqCtx {
        h_in: rand_fill(&mut rng, SEQ * HIDDEN, 1.0),
        cos,
        sin,
        tokens: (0..SEQ as u32)
            .map(|t| (t * 5 + 1) % VOCAB as u32)
            .collect(),
        completion_start: COMPLETION_START,
        seq_len: SEQ,
    };
    let lm_head = rand_fill(&mut rng, VOCAB * HIDDEN, 0.3);
    let final_shift = shifted(&rand_fill(&mut rng, HIDDEN, 0.1));
    let head = Head {
        lm_head: &lm_head,
        final_shift: &final_shift,
    };

    let fwd = forward_full(&ctx, &layers, &loras, &[], &head, &train).expect("forward_full");

    // Positions the terminal layer computes: the completion positions only.
    let completion = (COMPLETION_START - 1)..(SEQ - 1);
    assert_eq!(fwd.positions.len(), completion.len());

    for (li, lw) in layers.iter().enumerate() {
        let lf = &fwd.layers[li];
        let is_last_layer = li + 1 == layers.len();
        let computed = ffn_compute_range(is_last_layer, SEQ, completion.clone());
        let (normed_ffn, _) = rmsnorm_seq(&lf.h_mid, &lw.post_shift, HIDDEN, SEQ, dims.eps);
        let h_next = if is_last_layer {
            &fwd.h_final
        } else {
            &fwd.layers[li + 1].h_layer_in
        };
        for t in 0..SEQ {
            let mid = &lf.h_mid[t * HIDDEN..(t + 1) * HIDDEN];
            let next = &h_next[t * HIDDEN..(t + 1) * HIDDEN];
            if !computed.contains(&t) {
                assert!(
                    lf.gate_pre[t].is_empty() && lf.up_pre[t].is_empty(),
                    "layer {li} position {t} is outside the computed range and must hold no rows"
                );
                assert_eq!(
                    next, mid,
                    "layer {li} position {t}: no FFN output may be added"
                );
                continue;
            }
            assert_eq!(
                lf.gate_pre[t].len(),
                INTER,
                "layer {li} position {t} gate_pre"
            );
            assert_eq!(lf.up_pre[t].len(), INTER, "layer {li} position {t} up_pre");
            let (out_ref, gate_ref, up_ref) = swiglu_forward(
                &normed_ffn[t * HIDDEN..(t + 1) * HIDDEN],
                lw.w_gate,
                lw.w_up,
                lw.w_down,
                HIDDEN,
                INTER,
            );
            assert_parity(
                &format!("layer {li} position {t} gate_pre"),
                &gate_ref,
                &lf.gate_pre[t],
            );
            assert_parity(
                &format!("layer {li} position {t} up_pre"),
                &up_ref,
                &lf.up_pre[t],
            );
            let next_ref: Vec<f32> = mid.iter().zip(out_ref.iter()).map(|(m, f)| m + f).collect();
            assert_parity(&format!("layer {li} position {t} output"), &next_ref, next);
        }
    }

    // The reverse pass reads the stored rows back; it must run on the new storage.
    let (nll, n, grads, gdn_grads) =
        nll_and_grads(&fwd, &layers, &loras, &head, &train).expect("nll_and_grads");
    assert_eq!(n, completion.len());
    assert!(nll.is_finite());
    assert!(gdn_grads.is_empty());
    assert!(grads[0].b_q.iter().all(|g| g.is_finite()));
    assert!(
        grads[0].b_q.iter().any(|g| g.abs() > 1e-8),
        "b_q gradient is identically zero"
    );
}

// Mean completion NLL accumulated in f64 from the stored logits, so the finite-difference
// quotient below is not limited by an f32 sum or an f32 cast of the result.
fn mean_nll_f64(fwd: &FullFwd) -> f64 {
    let mut total = 0.0f64;
    for p in &fwd.positions {
        let max = p.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum_exp: f64 = p.logits.iter().map(|&l| (l as f64 - max).exp()).sum();
        total += max + sum_exp.ln() - p.logits[p.target as usize] as f64;
    }
    total / fwd.positions.len() as f64
}

// Whole-model gradient check through the same `forward_full` + `nll_and_grads` path
// training uses. Both layers carry a LoRA slot. Layer 0 is non-terminal, so its FFN block
// covers every position (rows start at position 0); layer 1 is terminal, so its FFN block
// covers only the completion positions (rows start at position `COMPLETION_START - 1`).
// A wrong position offset when the FFN output is added back changes the loss the forward
// computes. A wrong `rows_by_position` mapping leaves that loss alone (it only places the
// stored gate/up rows the backward reads) and misaligns the backward instead. Either way
// the analytic gradient and the finite difference of the loss disagree.
#[test]
fn forward_full_lora_gradients_match_finite_differences() {
    // Central differences on an f32 forward: the loss itself carries roughly 1e-6 of f32
    // rounding noise, so eps = 1e-2 keeps that noise near 5e-5 in the quotient while the
    // O(eps^2) truncation term stays below 1e-4 for parameters of magnitude 0.2.
    const EPS: f32 = 1e-2;
    // Relative to the larger of the two gradients, plus a small absolute floor. Measured on
    // this fixture: worst relative error 2.25e-3 and worst absolute error 8.7e-6 over the
    // 32 sampled entries, so 2e-2 leaves about 9x of margin and 2e-5 about 2.3x. A gradient
    // wrong by 10% (a scale error of 0.9) is outside the relative bound for every entry.
    const REL_TOL: f64 = 2e-2;
    const ABS_TOL: f64 = 2e-5;
    // An entry is significant when its gradient is large enough that the relative bound,
    // not the absolute floor, decides it. Every (layer, tensor) group must hold at least
    // `MIN_SIGNIFICANT_PER_GROUP` of them, so no group can pass on near-zero samples.
    const SIGNIFICANT: f64 = 2e-3;
    const MIN_SIGNIFICANT_PER_GROUP: usize = 2;

    let cfg = tiny_cfg();
    let dims = tiny_dims(&cfg);
    let gdn_dims = GdnDims::from_cfg(&cfg);
    let slot_layers = [0usize, 1];
    let no_gdn_slots: [usize; 0] = [];
    let train = TrainCtx::try_new(
        TapeGeometry::new(&dims, &gdn_dims, &cfg),
        RANK,
        2.0 * RANK as f32,
        SlotLayout::new(&slot_layers, &no_gdn_slots),
        AdamConfig::new(1e-3, 0.9, 0.999, 1e-8),
    )
    .expect("tiny geometry is valid");

    let mut rng = 0xC0FF_EE11_u64;
    let w0 = tiny_layer(&mut rng);
    let w1 = tiny_layer(&mut rng);
    let layers = [layer_w(&w0, Some(0)), layer_w(&w1, Some(1))];
    let mut loras: Vec<LoraParams> = (0..2)
        .map(|_| LoraParams {
            a_q: rand_fill(&mut rng, RANK * HIDDEN, 0.2),
            b_q: rand_fill(&mut rng, 2 * dims.q_dim * RANK, 0.2),
            a_v: rand_fill(&mut rng, RANK * HIDDEN, 0.2),
            b_v: rand_fill(&mut rng, dims.kv_dim * RANK, 0.2),
        })
        .collect();
    let (cos, sin) = rope_tables();
    let ctx = SeqCtx {
        h_in: rand_fill(&mut rng, SEQ * HIDDEN, 1.0),
        cos,
        sin,
        tokens: (0..SEQ as u32)
            .map(|t| (t * 5 + 1) % VOCAB as u32)
            .collect(),
        completion_start: COMPLETION_START,
        seq_len: SEQ,
    };
    let lm_head = rand_fill(&mut rng, VOCAB * HIDDEN, 0.3);
    let final_shift = shifted(&rand_fill(&mut rng, HIDDEN, 0.1));
    let head = Head {
        lm_head: &lm_head,
        final_shift: &final_shift,
    };

    let fwd = forward_full(&ctx, &layers, &loras, &[], &head, &train).expect("forward_full");
    let (nll_sum, n, mut grads, _) =
        nll_and_grads(&fwd, &layers, &loras, &head, &train).expect("nll_and_grads");
    let loss = mean_nll_f64(&fwd);
    assert!(
        (nll_sum as f64 / n as f64 - loss).abs() < 1e-5,
        "loss mismatch: reverse pass {} vs f64 recompute {loss}",
        nll_sum as f64 / n as f64
    );

    type Select = fn(&mut LoraParams) -> &mut Vec<f32>;
    let tensors: [(&str, Select); 4] = [
        ("a_q", |l| &mut l.a_q),
        ("b_q", |l| &mut l.b_q),
        ("a_v", |l| &mut l.a_v),
        ("b_v", |l| &mut l.b_v),
    ];
    let mut checked = 0usize;
    let mut worst_rel = 0.0f64;
    for slot in 0..2 {
        for (name, select) in tensors {
            let g = select(&mut grads[slot]).clone();
            let len = g.len();
            let mut significant = 0usize;
            for idx in [1, len / 3, 2 * len / 3, len - 2] {
                let original = select(&mut loras[slot])[idx];
                select(&mut loras[slot])[idx] = original + EPS;
                let plus = mean_nll_f64(
                    &forward_full(&ctx, &layers, &loras, &[], &head, &train)
                        .expect("forward_full +eps"),
                );
                select(&mut loras[slot])[idx] = original - EPS;
                let minus = mean_nll_f64(
                    &forward_full(&ctx, &layers, &loras, &[], &head, &train)
                        .expect("forward_full -eps"),
                );
                select(&mut loras[slot])[idx] = original;

                let numeric = (plus - minus) / (2.0 * EPS as f64);
                let ana = g[idx] as f64;
                let tol = REL_TOL * ana.abs().max(numeric.abs()) + ABS_TOL;
                // `!(diff <= tol)` so a NaN quotient fails instead of passing.
                assert!(
                    (ana - numeric).abs() <= tol,
                    "layer {slot} {name}[{idx}]: analytic {ana:.6e} vs finite-difference {numeric:.6e}"
                );
                checked += 1;
                if ana.abs() > SIGNIFICANT {
                    significant += 1;
                    worst_rel = worst_rel.max((ana - numeric).abs() / ana.abs().max(numeric.abs()));
                }
            }
            assert!(
                significant >= MIN_SIGNIFICANT_PER_GROUP,
                "layer {slot} {name}: only {significant} sampled gradients exceed {SIGNIFICANT:e}"
            );
        }
    }
    eprintln!(
        "gradient check: {checked} entries, worst significant relative error {worst_rel:.2e}"
    );
    assert_eq!(checked, 32);
}
