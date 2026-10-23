use super::*;
use crate::forward::cpu_q8::forward_step_q8;
use crate::weights::q8_weights::Q8Matrix;

const PROMPT: [u32; 3] = [1, 3, 2];
const DECODE_STEPS: usize = 2;

fn plan() -> GenerationPlan {
    GenerationPlan {
        rng_state: 7,
        prompt_ids: PROMPT.to_vec(),
        prompt_len: PROMPT.len(),
        required_capacity: PROMPT.len() + DECODE_STEPS,
    }
}

fn projections(weights: &Q8ModelWeights) -> Vec<&Q8Matrix> {
    let mut matrices = Vec::new();
    for (attention, common) in &weights.layers {
        match attention {
            Q8AttentionWeights::Linear(layer) => matrices.extend([
                &layer.in_proj_qkv,
                &layer.in_proj_z,
                &layer.in_proj_b,
                &layer.in_proj_a,
                &layer.out_proj,
            ]),
            Q8AttentionWeights::Full(layer) => {
                matrices.extend([&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj]);
            }
        }
        matrices.extend([&common.gate_proj, &common.up_proj, &common.down_proj]);
    }
    matrices
}

fn projections_mut(weights: &mut Q8ModelWeights) -> Vec<&mut Q8Matrix> {
    let mut matrices = Vec::new();
    for (attention, common) in &mut weights.layers {
        match attention {
            Q8AttentionWeights::Linear(layer) => matrices.extend([
                &mut layer.in_proj_qkv,
                &mut layer.in_proj_z,
                &mut layer.in_proj_b,
                &mut layer.in_proj_a,
                &mut layer.out_proj,
            ]),
            Q8AttentionWeights::Full(layer) => matrices.extend([
                &mut layer.q_proj,
                &mut layer.k_proj,
                &mut layer.v_proj,
                &mut layer.o_proj,
            ]),
        }
        matrices.extend([
            &mut common.gate_proj,
            &mut common.up_proj,
            &mut common.down_proj,
        ]);
    }
    matrices
}

fn assert_finite_bits(actual: &[f32], expected: &[f32], label: &str) {
    assert!(!expected.is_empty(), "{label}: empty reference");
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    assert!(
        expected.iter().any(|value| *value != 0.0),
        "{label}: reference must carry a nonzero signal"
    );
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label}[{index}]: actual={actual}, expected={expected}"
        );
        assert_eq!(actual.to_bits(), expected.to_bits(), "{label}[{index}]");
    }
}

fn assert_state(
    session: &StandaloneCpuSession<'_>,
    reference_gdn: &[GatedDeltaNetState],
    reference_kv: &KvCache,
    reference_scratch: &ForwardScratch,
) {
    assert_finite_bits(
        &session.scratch.logits[..session.cfg.vocab_size],
        &reference_scratch.logits[..session.cfg.vocab_size],
        "logits",
    );
    assert_eq!(session.kv_cache.seq_len, reference_kv.seq_len);
    assert_eq!(session.kv_cache.k.len(), reference_kv.k.len());
    assert_eq!(session.kv_cache.v.len(), reference_kv.v.len());
    assert_eq!(reference_kv.k.len(), 1, "fixture must exercise a KV layer");
    assert_eq!(reference_kv.v.len(), 1, "fixture must exercise a KV layer");
    for (actual, expected) in session.kv_cache.k.iter().zip(&reference_kv.k) {
        assert_eq!(
            expected.len(),
            reference_kv.seq_len * session.cfg.full_kv_dim()
        );
        assert_finite_bits(actual, expected, "K cache");
    }
    for (actual, expected) in session.kv_cache.v.iter().zip(&reference_kv.v) {
        assert_eq!(
            expected.len(),
            reference_kv.seq_len * session.cfg.full_kv_dim()
        );
        assert_finite_bits(actual, expected, "V cache");
    }
    assert_eq!(session.gdn_states.len(), reference_gdn.len());
    assert_eq!(reference_gdn.len(), 1, "fixture must exercise a GDN layer");
    for (actual, expected) in session.gdn_states.iter().zip(reference_gdn) {
        assert_finite_bits(&actual.s_matrices, &expected.s_matrices, "GDN state");
        assert_finite_bits(
            &actual.conv_buffer,
            &expected.conv_buffer,
            "GDN convolution",
        );
    }
}

// The fresh-scratch reference shares the kernel: this checks session ownership, not arithmetic.
pub(crate) fn assert_q8_session_scratch_reuse(
    weights: &Q8ModelWeights,
    cfg: &Qwen35Config,
    rope: &RopeTable,
) {
    assert_eq!(weights.layers.len(), 2);
    assert_eq!(cfg.num_linear_attention_layers(), 1);
    assert_eq!(cfg.num_full_attention_layers(), 1);
    assert!(cfg.layer_mask.iter().all(|enabled| *enabled));
    assert!(
        PROMPT
            .iter()
            .all(|token| (*token as usize) < cfg.vocab_size)
    );
    assert!(rope.max_positions() >= PROMPT.len() + DECODE_STEPS);
    let calls_per_token: usize = weights
        .layers
        .iter()
        .map(|(attention, _)| {
            3 + match attention {
                Q8AttentionWeights::Linear(_) => 5,
                Q8AttentionWeights::Full(_) => 4,
            }
        })
        .sum();
    assert_eq!(calls_per_token, 15);
    let matrices = projections(weights);
    assert_eq!(matrices.len(), calls_per_token);
    for matrix in matrices {
        assert!(matrix.data.iter().any(|value| *value != 0));
        assert!(
            !matrix.scales.is_empty()
                && matrix
                    .scales
                    .iter()
                    .all(|scale| scale.is_finite() && *scale > 0.0)
        );
    }

    let mut session = StandaloneCpuSession::new(StandaloneWeights::Q8(weights), cfg, rope, plan());
    let storage = session.q8_scratch.storage_snapshot();
    assert_eq!(session.q8_scratch.call_count(), 0);
    #[cfg(target_os = "macos")]
    {
        assert!(storage[0][1] > 0 && storage[0][2] >= storage[0][1]);
        assert_eq!(storage[1][1], 64);
        assert!(storage[1][2] >= 64);
    }
    #[cfg(not(target_os = "macos"))]
    assert_eq!(storage, [[0; 3]; 2]);
    session.q8_scratch.poison();
    assert_eq!(session.q8_scratch.storage_snapshot(), storage);

    let mut reference_gdn: Vec<_> = (0..cfg.num_linear_attention_layers())
        .map(|_| GatedDeltaNetState::new(cfg))
        .collect();
    let mut reference_kv = KvCache::new(cfg.num_full_attention_layers());
    let mut reference_scratch = ForwardScratch::new();
    for (position, token) in PROMPT.into_iter().enumerate() {
        forward_step_q8(
            weights,
            cfg,
            rope,
            token,
            position,
            &mut reference_gdn,
            &mut reference_kv,
            &mut reference_scratch,
        );
        reference_kv.seq_len += 1;
    }
    let never = || false;
    let stamp = session.prefill(&never).expect("Q8 session prefill");
    assert_eq!(stamp.evaluated_len, PROMPT.len());
    assert_eq!(session.q8_scratch.storage_snapshot(), storage);
    assert_eq!(
        session.q8_scratch.call_count(),
        PROMPT.len() * calls_per_token
    );
    assert_state(&session, &reference_gdn, &reference_kv, &reference_scratch);

    let gen_cfg = GenerateConfig {
        max_new_tokens: DECODE_STEPS + 1,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    let mut history = PROMPT.to_vec();
    let mut previous_logits = session.scratch.logits[..cfg.vocab_size].to_vec();
    for step in 0..DECODE_STEPS {
        let candidate = match session
            .select(&SelectionRequest {
                config: &gen_cfg,
                history: &history,
                grammar_mask: None,
            })
            .expect("Q8 session select")
        {
            SelectOutcome::Candidate(candidate) => candidate,
            SelectOutcome::GrammarExhausted => panic!("no grammar was supplied"),
        };
        history.push(candidate.candidate_id);
        session.q8_scratch.poison();
        let stamp = session
            .decode(
                &AcceptedToken {
                    final_id: candidate.candidate_id,
                    prediction: candidate.prediction,
                },
                &never,
            )
            .expect("Q8 session decode");
        forward_step_q8(
            weights,
            cfg,
            rope,
            candidate.candidate_id,
            reference_kv.seq_len,
            &mut reference_gdn,
            &mut reference_kv,
            &mut reference_scratch,
        );
        reference_kv.seq_len += 1;
        assert_eq!(stamp.evaluated_len, PROMPT.len() + step + 1);
        assert_eq!(session.q8_scratch.storage_snapshot(), storage);
        assert_eq!(
            session.q8_scratch.call_count(),
            (PROMPT.len() + step + 1) * calls_per_token
        );
        assert_state(&session, &reference_gdn, &reference_kv, &reference_scratch);
        let current = &session.scratch.logits[..cfg.vocab_size];
        assert!(
            current
                .iter()
                .zip(&previous_logits)
                .any(|(current, previous)| current.to_bits() != previous.to_bits()),
            "decode step {step}: fixture logits must change"
        );
        previous_logits.copy_from_slice(current);
    }
}

pub(crate) fn assert_q8_projection_widths(
    weights: &Q8ModelWeights,
    cfg: &Qwen35Config,
    rope: &RopeTable,
) {
    let original = projections(weights);
    assert_eq!(original.len(), 15);
    let original_max = original.iter().map(|matrix| matrix.cols).max().unwrap();
    assert_eq!(original_max, 128);
    let baseline = constructor_storage(weights, cfg, rope, original_max);
    for index in 0..original.len() {
        for width in [original_max + 257 + index * 17, usize::MAX] {
            let mut widened = weights.clone();
            let mut matrices = projections_mut(&mut widened);
            assert_eq!(matrices.len(), 15);
            matrices[index].cols = width;
            drop(matrices);
            // Malformed metadata must not set allocation sizes; this model is never forwarded.
            let storage = constructor_storage(&widened, cfg, rope, original_max);
            for (actual, expected) in storage.iter().zip(&baseline) {
                assert_eq!(
                    actual[1..],
                    expected[1..],
                    "projection {index}, cols={width}"
                );
            }
        }
    }

    assert_eq!(cfg.hidden_size, 64);
    assert_eq!(cfg.intermediate_size, 128);
    assert_eq!(cfg.linear_num_value_heads(), 2);
    assert_eq!(cfg.linear_value_head_dim, 32);
    assert_eq!(cfg.num_attention_heads, 2);
    assert_eq!(cfg.head_dim, 32);
    let mut hidden = cfg.clone();
    hidden.hidden_size = 193;
    let mut intermediate = cfg.clone();
    intermediate.intermediate_size = 211;
    let mut value_heads = cfg.clone();
    value_heads.linear_num_value_heads = Some(8);
    let mut value_dim = cfg.clone();
    value_dim.linear_value_head_dim = 97;
    let mut query_heads = cfg.clone();
    query_heads.num_attention_heads = 7;
    let mut head_dim = cfg.clone();
    head_dim.head_dim = 97;
    for (variant, width) in [
        (hidden, 193),
        (intermediate, 211),
        (value_heads, 256),
        (value_dim, 194),
        (query_heads, 224),
        (head_dim, 194),
    ] {
        // Constructor-only variants isolate each bound; their original weights are not forwarded.
        constructor_storage(weights, &variant, rope, width);
    }

    assert!(matches!(
        &weights.layers[0].0,
        Q8AttentionWeights::Linear(_)
    ));
    let mut first_layer_only = cfg.clone();
    first_layer_only.num_hidden_layers = 1;
    first_layer_only.layer_types.truncate(1);
    first_layer_only.layer_mask.truncate(1);
    first_layer_only.num_attention_heads = 31;
    constructor_storage(weights, &first_layer_only, rope, original_max);
}

fn constructor_storage(
    weights: &Q8ModelWeights,
    cfg: &Qwen35Config,
    rope: &RopeTable,
    expected_width: usize,
) -> [[usize; 3]; 2] {
    let session = StandaloneCpuSession::new(StandaloneWeights::Q8(weights), cfg, rope, plan());
    let storage = session.q8_scratch.storage_snapshot();
    assert_eq!(session.q8_scratch.call_count(), 0);
    #[cfg(target_os = "macos")]
    {
        assert_eq!(storage[0][1], 64 * expected_width);
        assert!(storage[0][2] >= 64 * expected_width);
        assert_eq!(storage[1][1], 64);
        assert!(storage[1][2] >= 64);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = expected_width;
        assert_eq!(storage, [[0; 3]; 2]);
    }
    storage
}
