//! Tests on a synthetic configuration with random weights. They check the decoder against its
//! own invariants (decoding against prefill, the pass structure, the cache shape) and the
//! building blocks against values worked out by hand. None of them compares against the
//! reference implementation.

use lattice_arch::inkling2_config::{AttentionSpan, Inkling2TextConfig, parse_config_json};
use lattice_inference_core::InferenceError;
use lattice_weights::nvfp4::dequantize_nvfp4;

use crate::attention::{bias_at, fill_relative_bias};
use crate::conv::{ConvHistory, dynamic_conv, static_conv};
use crate::mixture::{NormalizedStates, message_gate};
use crate::moe::{MoeParams, route, softcapped_swiglu};
use crate::ops::{l2_clip, log_scale, rms_normalize_into, sigmoid, softcap, softmax_with_sink};
use crate::{
    AttentionWeights, DenseMatrix, ExpertBank, Inkling2Model, Inkling2Weights, KvConvolution,
    LayerWeights, LoopGates, MessageGateWeights, MoeWeights,
};

pub(crate) const CONFIG: &str = r#"{
  "architectures": ["Inkling2LoopForConditionalGeneration"],
  "model_type": "inkling2_loop",
  "eos_token_id": 1,
  "image_token_id": 2,
  "text_config": {
    "max_position_embeddings": 64,
    "hidden_size": 32,
    "num_hidden_layers": 12,
    "vocab_size": 40,
    "unpadded_vocab_size": 37,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "swa_num_attention_heads": 6,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 4,
    "sliding_window_size": 4,
    "d_rel": 4,
    "rel_extent": 4,
    "log_scaling_n_floor": 2,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "qk_l2_clip_threshold": 2.0,
    "rel_l2_clip_threshold": 1.5,
    "sconv_kernel_size": 4,
    "final_logit_softcapping": 30.0,
    "n_layers_per_attn_res_block": 2,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "loop_config": {"core_start": 4, "core_end": 8, "num_passes": PASSES, "gate_rank": 8}
  }
}"#;

const PROMPT: [u32; 11] = [3, 17, 5, 36, 9, 0, 22, 14, 8, 30, 11];

pub(crate) fn config(passes: usize) -> Inkling2TextConfig {
    parse_config_json(&CONFIG.replace("PASSES", &passes.to_string()))
        .expect("synthetic config parses")
        .text
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next() as f32 / 2_147_483_648.0 * 2.0 - 1.0
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 8) as u8
    }

    fn values(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.unit() * scale).collect()
    }

    fn around_one(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| 1.0 + 0.2 * self.unit()).collect()
    }

    fn dense(&mut self, rows: usize, cols: usize, scale: f32) -> DenseMatrix {
        DenseMatrix::from_f32(rows, cols, self.values(rows * cols, scale)).expect("f32 matrix")
    }

    fn dense_bf16(&mut self, rows: usize, cols: usize, scale: f32) -> DenseMatrix {
        let bits = self
            .values(rows * cols, scale)
            .into_iter()
            .map(|value| (value.to_bits() >> 16) as u16)
            .collect();
        DenseMatrix::from_bf16_bits(rows, cols, bits).expect("bf16 matrix")
    }

    fn nvfp4_parts(
        &mut self,
        experts: usize,
        rows: usize,
        cols: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        let elements = experts * rows * cols;
        let codes = (0..elements / 2).map(|_| self.byte()).collect();
        let scales = (0..elements / 16)
            .map(|_| 0x28 + self.byte() % 16)
            .collect();
        let multipliers = (0..experts).map(|_| 0.03 + 0.01 * self.unit()).collect();
        (codes, scales, multipliers)
    }

    fn nvfp4(&mut self, experts: usize, rows: usize, cols: usize) -> ExpertBank {
        let (codes, scales, multipliers) = self.nvfp4_parts(experts, rows, cols);
        ExpertBank::nvfp4(experts, rows, cols, codes, scales, multipliers).expect("NVFP4 bank")
    }

    fn mxfp8(&mut self, experts: usize, rows: usize, cols: usize) -> ExpertBank {
        let elements = experts * rows * cols;
        let values = (0..elements)
            .map(|_| {
                let byte = self.byte();
                if byte & 0x7f == 0x7f { byte ^ 1 } else { byte }
            })
            .collect();
        let scales = (0..elements / 32).map(|_| 116 + self.byte() % 4).collect();
        ExpertBank::mxfp8(experts, rows, cols, values, scales).expect("MXFP8 bank")
    }
}

fn gate(rng: &mut Rng, hidden: usize, rank: usize, scale: f32) -> MessageGateWeights {
    MessageGateWeights {
        down: rng.dense(rank, 2 * hidden, scale),
        forget: rng.dense(hidden, rank, scale),
        gain: rng.dense(hidden, rank, scale),
    }
}

fn kv_convolution(
    rng: &mut Rng,
    global: bool,
    channels: usize,
    kv_heads: usize,
    kernel: usize,
    hidden: usize,
) -> KvConvolution {
    if global {
        KvConvolution::Dynamic {
            static_taps: rng.values(kernel * channels, 0.3),
            projection: rng.dense(kv_heads * kernel, hidden, 0.1),
        }
    } else {
        KvConvolution::Static {
            taps: rng.values(channels * kernel, 0.3),
        }
    }
}

fn layer_weights(
    rng: &mut Rng,
    c: &Inkling2TextConfig,
    layer: usize,
    gate_scale: f32,
) -> LayerWeights {
    let h = c.hidden_size;
    let kernel = c.conv_kernel_size;
    let span = c.layer_spans[layer];
    let global = span == AttentionSpan::Global;
    let heads = c.heads(span);
    let q = heads.num_attention_heads * heads.head_dim;
    let kv = heads.num_key_value_heads * heads.head_dim;
    let r = heads.num_attention_heads * c.d_rel;
    let queries = if layer == 0 {
        0
    } else if global {
        3
    } else {
        1
    };
    let norms = if global { 3 } else { 1 };
    let experts = c.n_routed_experts;
    let intermediate = c.moe_intermediate_size;

    let attention_input_queries = (0..queries).map(|_| rng.values(h, 1.0)).collect();
    let mlp_input_query = rng.values(h, 1.0);
    let attention_norms = (0..norms).map(|_| rng.around_one(h)).collect();
    let mlp_norm = rng.around_one(h);
    let attention = AttentionWeights {
        query: rng.dense_bf16(q, h, 0.3),
        query_scale: rng.around_one(q),
        key: rng.dense(kv, h, 0.3),
        value: rng.dense_bf16(kv, h, 0.3),
        key_convolution: kv_convolution(rng, global, kv, heads.num_key_value_heads, kernel, h),
        value_convolution: kv_convolution(rng, global, kv, heads.num_key_value_heads, kernel, h),
        output_gate: rng.dense(q, h, 0.3),
        output: rng.dense(h, q, 0.3),
        sink: rng.dense(heads.num_attention_heads, h, 0.3),
        relative: rng.dense(r, h, 0.3),
        relative_scale: rng.around_one(r),
        relative_profiles: rng.dense(c.d_rel, c.relative_extent(span), 0.5),
    };
    let post_attention_convolution = rng.values(h * kernel, 0.3);
    let post_mlp_convolution = rng.values(h * kernel, 0.3);
    let (gate_up, down) = if layer < 10 {
        (
            rng.nvfp4(experts, 2 * intermediate, h),
            rng.nvfp4(experts, h, intermediate),
        )
    } else {
        (
            rng.mxfp8(experts, 2 * intermediate, h),
            rng.mxfp8(experts, h, intermediate),
        )
    };
    let moe = MoeWeights {
        router: rng.dense(experts, h, 0.5),
        selection_bias: rng.values(experts, 0.1),
        gate_up,
        down,
    };
    let lc = c.loop_config;
    let gates = if (lc.core_start..lc.core_end).contains(&layer) {
        Some(LoopGates {
            query: gate(rng, h, lc.gate_rank, gate_scale),
            key: gate(rng, h, lc.gate_rank, gate_scale),
            value: gate(rng, h, lc.gate_rank, gate_scale),
            mlp: gate(rng, h, lc.gate_rank, gate_scale),
        })
    } else {
        None
    };
    LayerWeights {
        attention_input_queries,
        mlp_input_query,
        attention_norms,
        mlp_norm,
        attention,
        post_attention_convolution,
        post_mlp_convolution,
        moe,
        gates,
    }
}

fn weights(c: &Inkling2TextConfig, seed: u64, gate_scale: f32) -> Inkling2Weights {
    let mut rng = Rng(seed);
    let h = c.hidden_size;
    let core = c.loop_config.core_end - c.loop_config.core_start;
    let embedding = rng.dense_bf16(c.vocab_size, h, 1.0);
    let embedding_norm = rng.around_one(h);
    let layers = (0..c.num_hidden_layers)
        .map(|layer| layer_weights(&mut rng, c, layer, gate_scale))
        .collect();
    let message_queries = (0..4 * core).map(|_| rng.values(h, 1.0)).collect();
    Inkling2Weights {
        embedding,
        embedding_norm,
        layers,
        message_queries,
        output_query: rng.values(h, 1.0),
        final_norm: rng.around_one(h),
        unembedding: rng.dense(c.vocab_size, h, 0.5),
    }
}

fn model(passes: usize, gate_scale: f32) -> Inkling2Model {
    let c = config(passes);
    let w = weights(&c, 0x1234_5678, gate_scale);
    Inkling2Model::new(c, w).expect("synthetic model builds")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && e.is_finite(), "value {index}: {a} vs {e}");
        assert!(
            (a - e).abs() <= tolerance * e.abs().max(1.0),
            "value {index}: {a} vs {e}"
        );
    }
}

fn max_difference(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn refusal_names(result: Result<Inkling2Model, InferenceError>, part: &str) -> bool {
    matches!(result, Err(InferenceError::InvalidInput(message)) if message.contains(part))
}

#[test]
fn l2_clip_shortens_long_vectors_only() {
    let mut long = [3.0f32, 4.0];
    l2_clip(&mut long, 1.0);
    assert_close(&long, &[0.6, 0.8], 1e-6);
    let mut short = [0.3f32, 0.4];
    l2_clip(&mut short, 1.0);
    assert_eq!(short, [0.3, 0.4]);
}

#[test]
fn log_scale_is_one_up_to_the_floor() {
    assert_eq!(log_scale(0, 0.1, 2), 1.0);
    assert_eq!(log_scale(1, 0.1, 2), 1.0);
    assert_close(&[log_scale(7, 0.1, 2)], &[1.0 + 0.1 * 4.0f32.ln()], 1e-6);
}

#[test]
fn static_convolution_weights_the_oldest_input_first_and_keeps_its_history() {
    let mut history = ConvHistory::new(1, 3);
    let taps = [1.0f32, 10.0, 100.0];
    let mut out = [0.0f32; 3];
    static_conv(&mut history, &taps, 3, &[1.0, 2.0, 3.0], &mut out);
    assert_eq!(out, [50.5, 106.0, 162.0]);
    assert_eq!(history.values(), &[2.0, 3.0]);
    let mut next = [0.0f32];
    static_conv(&mut history, &taps, 3, &[4.0], &mut next);
    assert_eq!(next, [218.0]);
}

#[test]
fn dynamic_convolution_indexes_taps_by_backward_distance() {
    let mut history = ConvHistory::new(2, 2);
    let static_taps = [1.0f32, 0.0, 10.0, 0.0];
    let token_taps = [0.0f32, 0.0, 2.0, 20.0, 0.0, 0.0, 3.0, 30.0];
    let mut out = [0.0f32; 4];
    dynamic_conv(
        &mut history,
        &static_taps,
        &token_taps,
        2,
        2,
        &[1.0, 1.0, 2.0, 3.0],
        &mut out,
    );
    let divisor = 5f64.sqrt() as f32;
    let expected = [2.0 / divisor, 3.0 / divisor, 14.0 / divisor, 42.0 / divisor];
    assert_close(&out, &expected, 1e-6);
    assert_eq!(history.values(), &[2.0, 3.0]);
}

#[test]
fn the_sink_takes_probability_mass_without_a_value() {
    let mut logits = [0.0f32, 0.0];
    softmax_with_sink(&mut logits, Some(0.0));
    assert_close(&logits, &[1.0 / 3.0, 1.0 / 3.0], 1e-6);
    let mut plain = [0.0f32, 0.0];
    softmax_with_sink(&mut plain, None);
    assert_eq!(plain, [0.5, 0.5]);
}

#[test]
fn relative_bias_mixes_the_profiles_and_is_zero_past_the_extent() {
    let profiles =
        DenseMatrix::from_f32(2, 3, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).expect("profiles");
    let relative = [1.0f32, 0.0, 0.5, 1.0];
    let mut bias = [0.0f32; 6];
    fill_relative_bias(&relative, 2, &profiles, 3, &mut bias);
    assert_eq!(bias, [1.0, 2.0, 3.0, 10.5, 21.0, 31.5]);
    assert_eq!(bias_at(&bias[3..], 2), 31.5);
    assert_eq!(bias_at(&bias[3..], 3), 0.0);
}

#[test]
fn the_router_chooses_with_the_bias_and_weights_without_it() {
    let router = DenseMatrix::from_f32(4, 1, vec![0.0, 1.0, 2.0, 3.0]).expect("router");
    let params = MoeParams {
        top_k: 2,
        route_scale: 2.0,
        gate_cap: 4.0,
        up_cap: 25.0,
        intermediate: 1,
    };
    let chosen = route(&params, &router, &[10.0, 0.0, 0.0, -10.0], &[1.0]);
    let (s0, s2) = (sigmoid(0.0), sigmoid(2.0));
    assert_eq!(chosen.len(), 2);
    assert_eq!((chosen[0].0, chosen[1].0), (0, 2));
    assert_close(
        &[chosen[0].1, chosen[1].1],
        &[s0 / (s0 + s2) * 2.0, s2 / (s0 + s2) * 2.0],
        1e-6,
    );
}

#[test]
fn the_swiglu_sigmoid_sees_the_raw_gate() {
    let expected = softcap(10.0, 4.0) * sigmoid(10.0) * softcap(1.0, 25.0);
    assert_eq!(softcapped_swiglu(10.0, 1.0, 4.0, 25.0), expected);
    let capped = softcap(10.0, 4.0);
    assert!(expected > capped * sigmoid(capped) * softcap(1.0, 25.0));
}

#[test]
fn a_zero_query_mixture_is_the_mean_of_the_normalized_states() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [-1.0f32, 0.0, 1.0, 0.0];
    let states = NormalizedStates::new(4, &[&a[..], &b[..]]);
    let mut out = [0.0f32; 4];
    states.attend(&[0.0; 4], &mut out);
    let mut na = [0.0f32; 4];
    let mut nb = [0.0f32; 4];
    rms_normalize_into(&a, 1e-6, &mut na);
    rms_normalize_into(&b, 1e-6, &mut nb);
    let expected: Vec<f32> = na.iter().zip(&nb).map(|(x, y)| 0.5 * x + 0.5 * y).collect();
    assert_close(&out, &expected, 1e-6);
}

#[test]
fn a_closed_message_gate_passes_the_hidden_state_through() {
    let zero = |rows: usize, cols: usize| {
        DenseMatrix::from_f32(rows, cols, vec![0.0; rows * cols]).expect("zero matrix")
    };
    let gate = MessageGateWeights {
        down: zero(2, 8),
        forget: zero(4, 2),
        gain: zero(4, 2),
    };
    let hidden = [0.5f32, -1.0, 2.0, 0.25];
    let message = [3.0f32, 1.0, -2.0, 4.0];
    let mut out = [0.0f32; 4];
    message_gate(&gate, 1e-6, &hidden, &message, &mut out);
    assert_eq!(out, hidden);
}

#[test]
fn expert_banks_refuse_bad_layouts() {
    assert!(ExpertBank::nvfp4(1, 1, 8, vec![0; 4], vec![0x30], vec![1.0]).is_err());
    assert!(ExpertBank::nvfp4(1, 1, 16, vec![0; 8], vec![0x80], vec![1.0]).is_err());
    assert!(ExpertBank::nvfp4(1, 1, 16, vec![0; 8], vec![0x30], vec![-1.0]).is_err());
    assert!(ExpertBank::nvfp4(1, 1, 16, vec![0; 8], vec![0x30], vec![f32::NAN]).is_err());
    assert!(ExpertBank::nvfp4(1, 1, 16, vec![0; 7], vec![0x30], vec![1.0]).is_err());
    assert!(ExpertBank::nvfp4(1, 1, 16, vec![0; 8], vec![0x30], vec![1.0]).is_ok());
    assert!(ExpertBank::mxfp8(1, 1, 32, vec![0; 32], vec![255]).is_err());
    assert!(ExpertBank::mxfp8(1, 1, 32, vec![0; 31], vec![127]).is_err());
    assert!(ExpertBank::mxfp8(1, 1, 16, vec![0; 16], vec![127]).is_err());
    assert!(ExpertBank::mxfp8(1, 1, 32, vec![0; 32], vec![127]).is_ok());
}

#[test]
fn an_expert_matvec_matches_its_dequantized_slice() {
    let mut rng = Rng(7);
    let (codes, scales, multipliers) = rng.nvfp4_parts(3, 4, 32);
    let bank = ExpertBank::nvfp4(3, 4, 32, codes.clone(), scales.clone(), multipliers.clone())
        .expect("bank");
    let x = rng.values(32, 1.0);
    let mut dense = vec![0.0f32; 128];
    dequantize_nvfp4(
        4,
        32,
        &codes[64..128],
        &scales[8..16],
        multipliers[1],
        &mut dense,
    )
    .expect("dequantize");
    let expected: Vec<f32> = dense
        .chunks_exact(32)
        .map(|row| row.iter().zip(&x).map(|(w, v)| w * v).sum())
        .collect();
    let mut out = vec![0.0f32; 4];
    bank.matvec(1, &x, &mut out).expect("matvec");
    assert_eq!(out, expected);
}

#[test]
fn a_bf16_matrix_widens_on_read() {
    let matrix = DenseMatrix::from_bf16_bits(1, 2, vec![0x3f80, 0xc000]).expect("matrix");
    assert_eq!(matrix.get(0, 0), Some(1.0));
    assert_eq!(matrix.get(0, 1), Some(-2.0));
    assert_eq!(matrix.get(1, 0), None);
    assert!(DenseMatrix::from_f32(2, 2, vec![0.0; 3]).is_err());
    assert!(DenseMatrix::from_f32(0, 2, Vec::new()).is_err());
}

#[test]
fn the_cache_has_a_slot_for_every_layer_application() {
    for (passes, slots) in [(1, 12), (3, 20)] {
        let m = model(passes, 0.2);
        let cache = m.new_cache();
        assert_eq!(cache.slot_count(), slots);
        assert_eq!(cache.slot_count(), m.config().total_layer_applications());
        assert_eq!(cache.position(), 0);
    }
}

#[test]
fn sliding_slots_keep_one_token_less_than_the_window() {
    let m = model(3, 0.2);
    let c = m.config();
    let mut cache = m.new_cache();
    m.forward(&PROMPT, &mut cache).expect("forward");
    assert_eq!(cache.position(), PROMPT.len());
    let lc = c.loop_config;
    for slot in 0..cache.slot_count() {
        let layer = if slot < c.num_hidden_layers {
            slot
        } else {
            lc.core_start + (slot - c.num_hidden_layers) % (lc.core_end - lc.core_start)
        };
        let expected = match c.layer_spans[layer] {
            AttentionSpan::Global => PROMPT.len(),
            AttentionSpan::Sliding => c.sliding_window_size - 1,
        };
        assert_eq!(cache.cached_tokens(slot), Some(expected), "slot {slot}");
    }
    assert_eq!(cache.cached_tokens(cache.slot_count()), None);
}

#[test]
fn decoding_token_by_token_and_in_chunks_matches_prefill() {
    let m = model(3, 0.2);
    let vocab = m.config().unpadded_vocab_size;
    let prefill = m.forward_all(&PROMPT, &mut m.new_cache()).expect("prefill");
    assert_eq!(prefill.len(), PROMPT.len() * vocab);
    let last = m.forward(&PROMPT, &mut m.new_cache()).expect("last logits");
    assert_close(&last, &prefill[(PROMPT.len() - 1) * vocab..], 1e-5);
    let spread =
        last.iter().fold(f32::MIN, |a, &b| a.max(b)) - last.iter().fold(f32::MAX, |a, &b| a.min(b));
    assert!(spread > 1e-2, "logits barely vary: {spread}");

    let mut decode_cache = m.new_cache();
    for (t, &token) in PROMPT.iter().enumerate() {
        let logits = m.forward(&[token], &mut decode_cache).expect("decode");
        assert_close(&logits, &prefill[t * vocab..(t + 1) * vocab], 1e-5);
    }

    let mut chunked_cache = m.new_cache();
    let mut chunked = m
        .forward_all(&PROMPT[..4], &mut chunked_cache)
        .expect("first chunk");
    chunked.extend(
        m.forward_all(&PROMPT[4..], &mut chunked_cache)
            .expect("second chunk"),
    );
    assert_close(&chunked, &prefill, 1e-5);
}

#[test]
fn three_passes_with_closed_message_gates_match_one_pass() {
    let one = model(1, 0.0);
    let one_logits = one
        .forward_all(&PROMPT, &mut one.new_cache())
        .expect("one pass");
    let three = model(3, 0.0);
    let three_logits = three
        .forward_all(&PROMPT, &mut three.new_cache())
        .expect("three passes");
    assert_close(&three_logits, &one_logits, 1e-6);

    let open = model(3, 0.2);
    let open_logits = open
        .forward_all(&PROMPT, &mut open.new_cache())
        .expect("three passes with open gates");
    assert!(max_difference(&open_logits, &one_logits) > 1e-3);
}

#[test]
fn one_pass_runs_the_stored_layers_in_order() {
    let m = model(1, 0.2);
    let looped = m.forward_all(&PROMPT, &mut m.new_cache()).expect("loop");
    let straight = m
        .forward_straight(&PROMPT, &mut m.new_cache())
        .expect("straight");
    assert_eq!(looped, straight);
}

#[test]
fn misshapen_weights_are_refused_with_the_layer_named() {
    let c = config(3);

    let mut w = weights(&c, 1, 0.2);
    w.layers[3].mlp_norm.pop();
    assert!(refusal_names(
        Inkling2Model::new(c.clone(), w),
        "layer 3 MLP norm"
    ));

    let mut w = weights(&c, 1, 0.2);
    w.layers[5].attention.key_convolution = KvConvolution::Static {
        taps: vec![0.0; 64],
    };
    assert!(refusal_names(
        Inkling2Model::new(c.clone(), w),
        "layer 5 key convolution"
    ));

    let mut w = weights(&c, 1, 0.2);
    w.layers[2].gates = w.layers[4].gates.clone();
    assert!(refusal_names(
        Inkling2Model::new(c.clone(), w),
        "layer 2 message gates"
    ));

    let mut w = weights(&c, 1, 0.2);
    w.layers[6].gates = None;
    assert!(refusal_names(
        Inkling2Model::new(c.clone(), w),
        "layer 6 message gates"
    ));

    let mut w = weights(&c, 1, 0.2);
    w.layers.pop();
    assert!(refusal_names(Inkling2Model::new(c, w), "layers"));
}

#[test]
fn bad_tokens_and_a_foreign_cache_are_refused() {
    let m = model(3, 0.2);
    let mut cache = m.new_cache();
    assert!(m.forward(&[], &mut cache).is_err());
    assert!(m.forward(&[37], &mut cache).is_err());
    assert_eq!(cache.position(), 0);
    let mut foreign = model(1, 0.2).new_cache();
    assert!(m.forward(&[3], &mut foreign).is_err());
    assert!(m.forward(&[36], &mut cache).is_ok());
    assert_eq!(cache.position(), 1);
}
