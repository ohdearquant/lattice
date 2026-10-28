//! MoE routing on the device for Eager expert storage: the router and
//! shared-gate kernels read the completed current-layer norm, the published
//! trace comes from the route buffers, and the router kernel's ordering and
//! weights follow the documented rule for ties and non-finite logits.

use super::super::engine::{MOE_ROUTE_MAX_EXPERTS, MOE_ROUTE_MAX_TOP_K, moe_device_route_buffer};
use super::*;
use crate::model::qwen35::{MoeLayerWeights, MoeRouter, RoutedExperts, SharedExpert};

const HIDDEN: usize = 512;
const VOCAB: usize = 32;
const EXPERTS: usize = 4;
const INTER: usize = 16;
const SHARED_INTER: usize = 32;

/// Embedding row of `token`: one constant across the hidden dimension, so the
/// RMS-normalized row is `sign * (1 + gamma)` whatever the stored precision.
fn embedding_value(token: usize) -> f32 {
    [1.0, 2.0, 0.5][token % 3]
}

/// Per active layer: post-attention norm gamma (raw; the kernel scales by
/// `1 + gamma`), router rows, and whether the shared expert carries weights.
struct EagerLayer {
    gamma: f32,
    router_rows: [f32; EXPERTS],
    shared: bool,
}

/// Eager MoE model with zero attention and zero routed experts, so each
/// layer's residual stays at the embedding row and the MoE output is the
/// shared expert's alone.
fn eager_moe_fixture(mask: &[bool], layers: &[EagerLayer]) -> (Qwen35Config, ModelWeights) {
    assert_eq!(mask.iter().filter(|active| **active).count(), layers.len());
    let cfg = Qwen35Config {
        hidden_size: HIDDEN,
        num_hidden_layers: mask.len(),
        vocab_size: VOCAB,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        rope_parameters: None,
        linear_num_key_heads: 1,
        linear_num_value_heads: Some(1),
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        num_experts: Some(EXPERTS),
        num_experts_per_tok: Some(1),
        moe_intermediate_size: Some(INTER),
        shared_expert_intermediate_size: Some(SHARED_INTER),
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        full_attention_interval: 1,
        layer_types: vec![LayerType::FullAttention; mask.len()],
        layer_mask: mask.to_vec(),
        eos_token_id: (VOCAB - 1) as u32,
        max_position_embeddings: 64,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let layer_weights = layers
        .iter()
        .map(|layer| {
            let full = AttentionWeights::Full(FullAttentionLayerWeights {
                q_proj: vec![0.0; 2 * cfg.full_q_dim() * HIDDEN],
                k_proj: vec![0.0; cfg.full_kv_dim() * HIDDEN],
                v_proj: vec![0.0; cfg.full_kv_dim() * HIDDEN],
                o_proj: vec![0.0; HIDDEN * cfg.full_q_dim()],
                q_norm: vec![1.0; cfg.head_dim],
                k_norm: vec![1.0; cfg.head_dim],
            });
            let router_gate = layer
                .router_rows
                .iter()
                .flat_map(|value| std::iter::repeat_n(*value, HIDDEN))
                .collect();
            let router =
                MoeRouter::new(router_gate, EXPERTS, 1, HIDDEN).expect("router fixture shape");
            let experts = RoutedExperts::new(
                vec![0.0; EXPERTS * 2 * INTER * HIDDEN],
                vec![0.0; EXPERTS * HIDDEN * INTER],
                EXPERTS,
                HIDDEN,
                INTER,
            )
            .expect("routed experts fixture shape");
            let (in_value, down_value) = if layer.shared {
                (1.0 / HIDDEN as f32, 1.0 / SHARED_INTER as f32)
            } else {
                (0.0, 0.0)
            };
            let shared_expert = SharedExpert::new(
                vec![in_value; SHARED_INTER * HIDDEN],
                vec![in_value; SHARED_INTER * HIDDEN],
                vec![down_value; HIDDEN * SHARED_INTER],
                vec![in_value; HIDDEN],
                HIDDEN,
                SHARED_INTER,
            )
            .expect("shared expert fixture shape");
            let common = CommonLayerWeights {
                input_layernorm: vec![0.0; HIDDEN],
                post_attention_layernorm: vec![layer.gamma; HIDDEN],
                ffn: FeedForwardWeights::Moe(MoeLayerWeights {
                    router,
                    experts,
                    shared_expert,
                }),
            };
            (full, common)
        })
        .collect();

    let mut embed = vec![0.0f32; VOCAB * HIDDEN];
    for (token, row) in embed.chunks_exact_mut(HIDDEN).enumerate() {
        row.fill(embedding_value(token));
    }
    let weights = ModelWeights {
        embed_tokens: embed,
        lm_head: None,
        final_norm: vec![0.0; HIDDEN],
        layers: layer_weights,
    };
    (cfg, weights)
}

fn assert_default_route_conditions() {
    assert!(
        !lattice_model_io::env_switch_enabled("LATTICE_DECODE_PROFILE"),
        "test requires the default decode path"
    );
    assert!(
        !lattice_model_io::env_switch_enabled("LATTICE_MOE_HOST_ROUTER"),
        "test requires the device route"
    );
    assert!(
        FORCED_MOE_EXPERTS_FOR_TEST.with(|forced| forced.borrow().is_none()),
        "test requires real routing"
    );
}

fn assert_every_layer_has_a_device_route(state: &MetalQwen35State) {
    for (index, (_, common)) in state.engine.layer_weights.iter().enumerate() {
        let MetalFfnWeights::Moe(moe) = &common.ffn else {
            panic!("layer {index} must be MoE");
        };
        assert!(
            matches!(moe.routed, RoutedExpertStorage::Eager { .. }),
            "layer {index} must use Eager expert storage"
        );
        assert!(
            moe.device_route.is_some(),
            "layer {index} must carry a route buffer"
        );
    }
}

struct RoutingTraceGuard;

impl Drop for RoutingTraceGuard {
    fn drop(&mut self) {
        let _ = take_moe_routing_trace();
        let _ = take_moe_router_runs_for_test();
    }
}

/// Router choice from the current-layer norm: the constant embedding row
/// normalizes to `+1`, so the norm output is `1 + gamma` and the router rows
/// `[1, -1, 0, 0]` pick expert 0 for a positive output and expert 1 for a
/// negative one. The pre-norm hidden is positive and would pick expert 0.
fn expected_choice(gamma: f32) -> usize {
    if 1.0 + gamma > 0.0 { 0 } else { 1 }
}

fn run_routing_freshness(mask: &[bool], gammas: &[f32]) {
    let layers: Vec<_> = gammas
        .iter()
        .map(|gamma| EagerLayer {
            gamma: *gamma,
            router_rows: [1.0, -1.0, 0.0, 0.0],
            shared: false,
        })
        .collect();
    let original_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(index, active)| active.then_some(index))
        .collect();
    assert!(
        gammas.iter().any(|gamma| expected_choice(*gamma) == 1),
        "fixture must separate the pre-norm and current-layer choices"
    );
    let (cfg, weights) = eager_moe_fixture(mask, &layers);

    let _gpu = gpu_test_lock();
    let _device = Device::system_default().expect("device route witness requires a Metal device");
    assert_default_route_conditions();
    let _trace_guard = RoutingTraceGuard;
    let _ = take_moe_routing_trace();
    let _ = take_moe_router_runs_for_test();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16).expect("Eager MoE fixture");
    assert_eq!(state.engine.layer_weights.len(), gammas.len());
    assert_every_layer_has_a_device_route(&state);

    arm_moe_routing_trace();
    let tokens = [0u32, 1, 2];
    for (position, token) in tokens.into_iter().enumerate() {
        let logits = state.forward_step(token, position);
        assert_eq!(logits.len(), VOCAB);
        assert!(logits.iter().all(|value| value.is_finite()));
    }
    let trace = take_moe_routing_trace();
    let runs = take_moe_router_runs_for_test();
    assert_eq!(
        runs,
        (0, tokens.len() * gammas.len()),
        "every MoE layer of every token must take the device route"
    );
    assert_eq!(trace.len(), tokens.len() * gammas.len());
    for (index, record) in trace.iter().enumerate() {
        let active = index % gammas.len();
        assert_eq!(
            (record.layer_idx, record.token_idx),
            (original_indices[active], index / gammas.len()),
            "trace record {index} must name the original layer index and the token"
        );
        assert_eq!(
            record.selected_ids,
            vec![expected_choice(gammas[active])],
            "routing must use the current-layer norm at layer {} token {}",
            record.layer_idx,
            record.token_idx
        );
        assert_eq!(record.gate_weights, vec![1.0]);
    }
}

#[test]
fn device_route_reads_completed_current_layer_norm() {
    run_routing_freshness(&[true, true], &[0.0, -2.0]);
}

#[test]
fn device_route_trace_names_original_layer_indices_under_a_mask() {
    run_routing_freshness(&[true, false, true], &[0.0, -2.0]);
}

#[test]
fn device_route_trace_is_empty_when_disarmed() {
    let layers = [EagerLayer {
        gamma: -2.0,
        router_rows: [1.0, -1.0, 0.0, 0.0],
        shared: false,
    }];
    let (cfg, weights) = eager_moe_fixture(&[true], &layers);

    let _gpu = gpu_test_lock();
    let _device = Device::system_default().expect("device route witness requires a Metal device");
    assert_default_route_conditions();
    let _trace_guard = RoutingTraceGuard;
    let _ = take_moe_routing_trace();
    let _ = take_moe_router_runs_for_test();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16).expect("Eager MoE fixture");
    let logits = state.forward_step(0, 0);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(take_moe_router_runs_for_test(), (0, 1));
    MOE_DEVICE_ROUTE_PENDING.with(|pending| {
        assert!(
            pending.borrow().is_empty(),
            "a disarmed trace must not retain route buffers"
        );
    });
    assert!(take_moe_routing_trace().is_empty());
}

#[test]
fn device_route_shared_gate_reads_completed_current_layer_norm() {
    let gamma = -2.0f32;
    let layers = [EagerLayer {
        gamma,
        router_rows: [0.0; EXPERTS],
        shared: true,
    }];
    let (cfg, weights) = eager_moe_fixture(&[true], &layers);

    let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
    let gate_weight = 1.0 / HIDDEN as f32;
    // Norm output is the constant `1 + gamma`; shared gate and up rows are
    // `1 / HIDDEN`, down rows `1 / SHARED_INTER`.
    let current = 1.0 + gamma;
    let projected = current * HIDDEN as f32 * gate_weight;
    let shared_output = projected * sigmoid(projected) * projected;
    let fresh_gate = sigmoid(current * HIDDEN as f32 * gate_weight);

    let _gpu = gpu_test_lock();
    let _device = Device::system_default().expect("shared gate witness requires a Metal device");
    assert_default_route_conditions();
    let _trace_guard = RoutingTraceGuard;
    let _ = take_moe_router_runs_for_test();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16).expect("Eager MoE fixture");
    assert_every_layer_has_a_device_route(&state);
    for position in 0..3usize {
        let token = position as u32;
        let stale_gate = sigmoid(embedding_value(position) * HIDDEN as f32 * gate_weight);
        let fresh = shared_output * fresh_gate;
        let stale = shared_output * stale_gate;
        assert!(
            (fresh - stale).abs() > 0.09,
            "fixture must separate the pre-norm and current-layer gates"
        );
        let logits = state.forward_step(token, position);
        assert!(logits.iter().all(|value| value.is_finite()));
        // SAFETY: forward_step completed all GPU work; ffn_out is a live Shared
        // [hidden] f32 buffer.
        let actual = unsafe { read_buffer(&state.session.activations.ffn_out, HIDDEN) };
        for (row, actual) in actual.iter().enumerate() {
            assert!(
                actual.is_finite() && (*actual - fresh).abs() < 1e-3,
                "shared expert must use the current-layer norm at position {position} row {row}: \
                 actual={actual}, fresh={fresh}, stale={stale}"
            );
            assert!((*actual - stale).abs() > 0.08);
        }
    }
    assert_eq!(take_moe_router_runs_for_test(), (0, 3));
}

/// `MoeRouteParams` in `qwen35.metal`.
#[repr(C)]
struct RouteParams {
    hidden: u32,
    num_experts: u32,
    top_k: u32,
    inter: u32,
}

const ROUTE_INTER: usize = 3;

/// Dispatch `moe_router_topk` alone on `x` and the row-major router `wr`.
fn run_router_kernel(state: &MetalQwen35State, x: &[f32], wr: &[f32], top_k: usize) -> Vec<u32> {
    let hidden = x.len();
    let num_experts = wr.len() / hidden;
    assert_eq!(num_experts * hidden, wr.len());
    let device = &state.engine.device;
    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        std::mem::size_of_val(x) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let wr_buf = device.new_buffer_with_data(
        wr.as_ptr() as *const _,
        std::mem::size_of_val(wr) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let words = 5 * top_k + 1;
    let poison = vec![u32::MAX; words];
    let route = device.new_buffer_with_data(
        poison.as_ptr() as *const _,
        (words * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let params = RouteParams {
        hidden: hidden as u32,
        num_experts: num_experts as u32,
        top_k: top_k as u32,
        inter: ROUTE_INTER as u32,
    };
    let cmd = state.engine.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&state.engine.pipelines.moe_router_topk);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&wr_buf), 0);
    enc.set_buffer(2, Some(&route), 0);
    enc.set_bytes(
        3,
        std::mem::size_of::<RouteParams>() as u64,
        &params as *const RouteParams as *const _,
    );
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    // SAFETY: the command buffer completed; `route` is a live Shared buffer of
    // `words` u32 values.
    unsafe { std::slice::from_raw_parts(route.contents() as *const u32, words).to_vec() }
}

/// The documented order on `(logit, id)`: every non-NaN logit ranks above
/// every NaN, larger logits rank higher, ties go to the lower id.
fn ranks_above(logits: &[f32], a: usize, b: usize) -> bool {
    let (la, lb) = (logits[a], logits[b]);
    if la.is_nan() != lb.is_nan() {
        return lb.is_nan();
    }
    if !la.is_nan() && la != lb {
        return la > lb;
    }
    a < b
}

/// Checks a route buffer against the routing rule for `logits` (one per
/// expert) and `hidden`: distinct in-range ids in rank order with nothing
/// unselected ranking above the last, finite non-negative weights summing to
/// one, and element offsets from the expert ids.
fn check_route(words: &[u32], logits: &[f32], top_k: usize, hidden: usize) -> Result<(), String> {
    let num_experts = logits.len();
    if words.len() != 5 * top_k + 1 {
        return Err(format!("route has {} words", words.len()));
    }
    let ids: Vec<usize> = words[..top_k].iter().map(|id| *id as usize).collect();
    for (rank, id) in ids.iter().enumerate() {
        if *id >= num_experts {
            return Err(format!("rank {rank} id {id} out of range"));
        }
        if ids[..rank].contains(id) {
            return Err(format!("rank {rank} id {id} repeated"));
        }
        if rank > 0 && !ranks_above(logits, ids[rank - 1], *id) {
            return Err(format!("rank {rank} id {id} out of order"));
        }
    }
    let last = ids[top_k - 1];
    if let Some(above) =
        (0..num_experts).find(|e| !ids.contains(e) && ranks_above(logits, *e, last))
    {
        return Err(format!(
            "unselected expert {above} ranks above rank {}",
            top_k - 1
        ));
    }
    let weights: Vec<f32> = words[top_k..2 * top_k]
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect();
    if !weights.iter().all(|w| w.is_finite() && *w >= 0.0) {
        return Err(format!("weights {weights:?} not finite and non-negative"));
    }
    let sum: f32 = weights.iter().sum();
    if (sum - 1.0).abs() > 1e-5 {
        return Err(format!("weights {weights:?} sum to {sum}"));
    }
    let matrix = ROUTE_INTER * hidden;
    for (rank, id) in ids.iter().enumerate() {
        let expected = [id * 2 * matrix, id * 2 * matrix + matrix, id * matrix];
        let actual = &words[2 * top_k + 3 * rank..2 * top_k + 3 * rank + 3];
        if actual.iter().map(|w| *w as usize).ne(expected) {
            return Err(format!(
                "rank {rank} offsets {actual:?}, expected {expected:?}"
            ));
        }
    }
    Ok(())
}

fn route_ids(words: &[u32], top_k: usize) -> Vec<u32> {
    words[..top_k].to_vec()
}

fn route_weights(words: &[u32], top_k: usize) -> Vec<f32> {
    words[top_k..2 * top_k]
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect()
}

fn assert_weights(actual: &[f32], expected: &[f32], case: &str) {
    assert_eq!(actual.len(), expected.len(), "{case}");
    for (rank, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "{case}: rank {rank} weight {actual}, expected {expected}"
        );
    }
}

/// Builds `(x, wr)` with `hidden == 1` and `x == [1]`, so each router row is
/// the expert's logit.
fn single_column(logits: &[f32]) -> (Vec<f32>, Vec<f32>) {
    (vec![1.0], logits.to_vec())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[test]
fn device_router_kernel_follows_the_routing_rule() {
    let layers = [EagerLayer {
        gamma: 0.0,
        router_rows: [0.0; EXPERTS],
        shared: false,
    }];
    let (cfg, weights) = eager_moe_fixture(&[true], &layers);
    let _gpu = gpu_test_lock();
    let _device = Device::system_default().expect("router kernel test requires a Metal device");
    let state = MetalQwen35State::new(&weights, &cfg, 16).expect("Eager MoE fixture");

    struct Case {
        name: &'static str,
        logits: Vec<f32>,
        top_k: usize,
        ids: Vec<u32>,
        weights: Vec<f32>,
    }
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let cases = [
        Case {
            name: "all ties",
            logits: vec![0.0; 8],
            top_k: 3,
            ids: vec![0, 1, 2],
            weights: vec![1.0 / 3.0; 3],
        },
        Case {
            name: "ordered with a tie",
            logits: vec![3.0, 7.0, 1.0, 5.0, 7.0, 0.0, 2.0, 6.0],
            top_k: 4,
            ids: vec![1, 4, 7, 3],
            weights: softmax(&[7.0, 7.0, 6.0, 5.0]),
        },
        Case {
            name: "positive infinity first, NaN last",
            logits: vec![nan, f32::MAX, inf, -inf, -f32::MAX, 1.0],
            top_k: 6,
            ids: vec![2, 1, 5, 4, 3, 0],
            weights: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        Case {
            name: "NaN and negative infinity weigh zero",
            logits: vec![nan, -inf, 2.0, nan, 1.0],
            top_k: 4,
            ids: vec![2, 4, 1, 0],
            weights: {
                let finite = softmax(&[2.0, 1.0]);
                vec![finite[0], finite[1], 0.0, 0.0]
            },
        },
        Case {
            name: "all NaN",
            logits: vec![nan; 4],
            top_k: 2,
            ids: vec![0, 1],
            weights: vec![0.5, 0.5],
        },
        Case {
            name: "two positive infinities",
            logits: vec![inf, 1.0, inf],
            top_k: 3,
            ids: vec![0, 2, 1],
            weights: vec![0.5, 0.5, 0.0],
        },
    ];
    for case in &cases {
        let (x, wr) = single_column(&case.logits);
        let words = run_router_kernel(&state, &x, &wr, case.top_k);
        if let Err(err) = check_route(&words, &case.logits, case.top_k, 1) {
            panic!("{}: {err}", case.name);
        }
        assert_eq!(route_ids(&words, case.top_k), case.ids, "{}", case.name);
        assert_weights(&route_weights(&words, case.top_k), &case.weights, case.name);
    }

    // Widest supported shape, logits from a dot product over a hidden size
    // that is not a multiple of the SIMD width.
    let hidden = 37usize;
    let num_experts = MOE_ROUTE_MAX_EXPERTS;
    let top_k = MOE_ROUTE_MAX_TOP_K;
    let x: Vec<f32> = (0..hidden)
        .map(|i| if i == 5 { 1.0 } else { 0.0 })
        .collect();
    let mut wr = vec![0.0f32; num_experts * hidden];
    let logits: Vec<f32> = (0..num_experts)
        .map(|e| ((e * 389) % num_experts) as f32 / 64.0)
        .collect();
    for (row, logit) in wr.chunks_exact_mut(hidden).zip(&logits) {
        row[5] = *logit;
        row[0] = 1000.0;
    }
    let words = run_router_kernel(&state, &x, &wr, top_k);
    check_route(&words, &logits, top_k, hidden).unwrap_or_else(|err| panic!("widest: {err}"));
    let mut by_logit: Vec<usize> = (0..num_experts).collect();
    by_logit.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    let expected: Vec<u32> = by_logit[..top_k].iter().map(|e| *e as u32).collect();
    assert_eq!(route_ids(&words, top_k), expected, "widest");
    let selected: Vec<f32> = by_logit[..top_k].iter().map(|e| logits[*e]).collect();
    assert_weights(&route_weights(&words, top_k), &softmax(&selected), "widest");
}

#[test]
fn route_checker_rejects_malformed_routes() {
    let logits = [0.5f32, 2.0, 1.0, -1.0];
    let hidden = 4usize;
    let matrix = ROUTE_INTER * hidden;
    let weights = softmax(&[2.0, 1.0]);
    let valid: Vec<u32> = [1u32, 2]
        .into_iter()
        .chain(weights.iter().map(|w| w.to_bits()))
        .chain([
            (2 * matrix) as u32,
            (3 * matrix) as u32,
            matrix as u32,
            (4 * matrix) as u32,
            (5 * matrix) as u32,
            (2 * matrix) as u32,
        ])
        .chain([0])
        .collect();
    assert_eq!(check_route(&valid, &logits, 2, hidden), Ok(()));

    let mutate = |name: &str, change: &dyn Fn(&mut Vec<u32>)| {
        let mut words = valid.clone();
        change(&mut words);
        assert!(
            check_route(&words, &logits, 2, hidden).is_err(),
            "checker must reject {name}"
        );
    };
    mutate("an out-of-range id", &|w| w[0] = logits.len() as u32);
    mutate("a repeated id", &|w| w[1] = w[0]);
    mutate("ranks out of order", &|w| w.swap(0, 1));
    mutate("a skipped higher-ranked expert", &|w| w[1] = 0);
    mutate("weights not summing to one", &|w| w[2] = 1.0f32.to_bits());
    mutate("a NaN weight", &|w| w[3] = f32::NAN.to_bits());
    mutate("a wrong offset", &|w| w[5] += 1);
    mutate("a short route", &|w| {
        w.pop();
    });
}

#[test]
fn device_route_buffer_refuses_unsupported_shapes() {
    let _gpu = gpu_test_lock();
    let device = Device::system_default().expect("route buffer test requires a Metal device");
    let build =
        |experts: usize, top_k: usize, inter: usize, hidden: usize, gate_up: usize, down: usize| {
            moe_device_route_buffer(
                &device,
                experts,
                top_k,
                inter,
                hidden,
                gate_up,
                down,
                "test.route",
            )
        };
    let exact = |experts: usize, top_k: usize, inter: usize, hidden: usize| {
        build(
            experts,
            top_k,
            inter,
            hidden,
            experts * 2 * inter * hidden,
            experts * inter * hidden,
        )
    };

    let route = exact(4, 2, 16, 8).expect("supported shape builds a route buffer");
    assert_eq!(route.length(), (5 * 2 + 1) * 4);
    let widest = exact(MOE_ROUTE_MAX_EXPERTS, MOE_ROUTE_MAX_TOP_K, 1, 1)
        .expect("the widest supported shape builds a route buffer");
    assert_eq!(widest.length(), (5 * MOE_ROUTE_MAX_TOP_K as u64 + 1) * 4);

    assert!(exact(0, 1, 16, 8).is_none(), "no experts");
    assert!(exact(4, 0, 16, 8).is_none(), "top_k zero");
    assert!(exact(4, 5, 16, 8).is_none(), "top_k above the expert count");
    assert!(
        exact(MOE_ROUTE_MAX_EXPERTS + 1, 1, 1, 1).is_none(),
        "expert count above the kernel limit"
    );
    assert!(
        exact(64, MOE_ROUTE_MAX_TOP_K + 1, 1, 1).is_none(),
        "top_k above the kernel limit"
    );
    assert!(
        build(4, 2, 16, 8, 4 * 2 * 16 * 8 - 1, 4 * 16 * 8).is_none(),
        "gate/up buffer shorter than the expert extent"
    );
    assert!(
        build(4, 2, 16, 8, 4 * 2 * 16 * 8, 4 * 16 * 8 - 1).is_none(),
        "down buffer shorter than the expert extent"
    );
    // Expert matrix 2^31 elements: the second expert's up offset is 3 * 2^31.
    assert!(
        build(2, 1, 1 << 16, 1 << 15, usize::MAX, usize::MAX).is_none(),
        "offset above u32"
    );
    // One expert: its up offset is 2^31, inside u32.
    assert!(
        build(1, 1, 1 << 16, 1 << 15, usize::MAX, usize::MAX).is_some(),
        "single expert with offsets inside u32"
    );
}

// Synthetic repeated-layer scheduling: this is not a claim that production
// currently encodes multiple MoE tokens in a single command buffer (#2221).
fn encode_route_witness_token<'a>(
    state: &mut MetalQwen35State,
    enc: &'a ComputeCommandEncoderRef,
    input: &Buffer,
    zeros: &Buffer,
    position: usize,
    cfg: &Qwen35Config,
) -> &'a ComputeCommandEncoderRef {
    // The copies run on the GPU in encoder order; changing shared memory from
    // the CPU before submission would make both router calls see the last input.
    state.dispatch_copy(
        enc,
        input,
        &state.session.activations.residual,
        HIDDEN as u32,
    );
    state.dispatch_copy(
        enc,
        zeros,
        &state.session.activations.attn_out,
        HIDDEN as u32,
    );
    let mut profile = StepProfile::default();
    state.encode_mlp_block(enc, 0, 0, position, cfg, &mut profile, false, |_| {
        panic!("route witness must not fall back to the host router")
    })
}

fn run_route_publication_witness(two_tokens_in_one_command: bool) {
    let scale = 1.0 / HIDDEN as f32;
    let layers = [EagerLayer {
        gamma: 0.0,
        router_rows: [scale, 0.25 * scale, -0.5 * scale, -2.0 * scale],
        shared: false,
    }];
    let (mut cfg, mut weights) = eager_moe_fixture(&[true], &layers);
    cfg.num_experts_per_tok = Some(2);
    let FeedForwardWeights::Moe(moe) = &mut weights.layers[0].1.ffn else {
        panic!("route witness requires an MoE layer");
    };
    moe.router.num_experts_per_tok = 2;

    let norm = 1.0 / (1.0 + cfg.rms_norm_eps).sqrt();
    let expected_ids = [[0usize, 1], [3, 2]];
    let expected_weights = [
        softmax(&[norm, 0.25 * norm]),
        softmax(&[2.0 * norm, 0.5 * norm]),
    ];
    assert_ne!(expected_ids[0], expected_ids[1]);
    assert!((expected_weights[0][0] - expected_weights[1][0]).abs() > 0.1);

    let _gpu = gpu_test_lock();
    let _device = Device::system_default().expect("route witness requires a Metal device");
    assert_default_route_conditions();
    let _trace_guard = RoutingTraceGuard;
    let _ = take_moe_routing_trace();
    let _ = take_moe_router_runs_for_test();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16).expect("Eager MoE fixture");
    assert_every_layer_has_a_device_route(&state);
    let device = &state.engine.device;
    let shared = MTLResourceOptions::StorageModeShared;
    let positive = [1.0f32; HIDDEN];
    let negative = [-1.0f32; HIDDEN];
    let zero = [0.0f32; HIDDEN];
    let input_buffers = [&positive, &negative].map(|input| {
        device.new_buffer_with_data(
            input.as_ptr() as *const _,
            std::mem::size_of_val(input) as u64,
            shared,
        )
    });
    let zeros = device.new_buffer_with_data(
        zero.as_ptr() as *const _,
        std::mem::size_of_val(&zero) as u64,
        shared,
    );
    let queue = state.engine.queue.clone();
    arm_moe_routing_trace();

    if two_tokens_in_one_command {
        let cmd = queue.new_command_buffer();
        let mut enc = cmd.new_compute_command_encoder();
        for (position, input) in input_buffers.iter().enumerate() {
            enc = encode_route_witness_token(&mut state, enc, input, &zeros, position, &cfg);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::Completed);
        // SAFETY: the command buffer containing both route writes completed.
        unsafe { publish_moe_device_routes() };
    } else {
        for (position, input) in input_buffers.iter().enumerate() {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let enc = encode_route_witness_token(&mut state, enc, input, &zeros, position, &cfg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::Completed);
            // SAFETY: this token's route write completed before publication.
            unsafe { publish_moe_device_routes() };
        }
    }
    assert_eq!(
        take_moe_router_runs_for_test(),
        (0, 2),
        "both routes must run on the device"
    );
    let trace = take_moe_routing_trace();
    assert_eq!(trace.len(), 2);
    eprintln!(
        "route witness: actual={trace:?}; expected_ids={expected_ids:?}; expected_weights={expected_weights:?}"
    );
    for (position, record) in trace.iter().enumerate() {
        assert_eq!((record.layer_idx, record.token_idx), (0, position));
        assert_eq!(
            record.selected_ids, expected_ids[position],
            "wrong route for token {position}"
        );
        assert_weights(
            &record.gate_weights,
            &expected_weights[position],
            "route publication",
        );
    }
}

#[test]
fn device_route_publication_per_command_control() {
    run_route_publication_witness(false);
}

#[test]
#[ignore = "#2221: desired correctness for two device routes sharing one command buffer is not yet established"]
fn device_route_publication_two_tokens_one_command_witness() {
    run_route_publication_witness(true);
}
