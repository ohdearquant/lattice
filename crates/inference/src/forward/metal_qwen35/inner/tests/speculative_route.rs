//! Output and end-state characterization of the two speculative decode routes
//! (default MTP with the sequential verifier, and GDN-first self-speculation).
//!
//! Every scenario prints one `SPEC` line with the generated ids, text, stop
//! disposition and the state the request leaves behind (KV cursor, draft-cache
//! cursor, the pre-final hidden state's bit sum and cursor, and the checkpoint
//! pool's rollback bookkeeping). The lines are compared across two builds, so a
//! change to the route is visible in what it generates and in the state it
//! leaves, not only in a pass or a fail.

use super::prefix_cache_route::{patterned_model, projection};
use super::*;

#[derive(Clone, Copy, Debug)]
enum Head {
    /// Embedding-half identity and zero layers: the draft repeats the pending token.
    Copy,
    /// Nonzero everywhere: the draft is an unrelated guess.
    Scrambled,
}

fn mtp_head(device: &Device, cfg: &Qwen35Config, head: Head) -> MetalMtpWeights {
    match head {
        Head::Copy => synthetic_mtp_weights_for_test(device, cfg),
        Head::Scrambled => {
            let hidden = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let mut seed = 9_000u64;
            let mut next = || {
                seed += 1;
                seed
            };
            MetalMtpWeights {
                fc: make_buffer_f16(
                    device,
                    &projection(hidden * 2 * hidden, next(), 2 * hidden, 1.0),
                    "spec.mtp.fc",
                ),
                pre_fc_norm_embedding: make_buffer(device, &vec![1.0; hidden], "spec.mtp.pre_e"),
                pre_fc_norm_hidden: make_buffer(device, &vec![1.0; hidden], "spec.mtp.pre_h"),
                layers: vec![MetalMtpLayerWeights {
                    input_layernorm: make_buffer(device, &vec![1.0; hidden], "spec.mtp.in_ln"),
                    post_attention_layernorm: make_buffer(
                        device,
                        &vec![1.0; hidden],
                        "spec.mtp.post_ln",
                    ),
                    q_proj: make_buffer_f16(
                        device,
                        &projection(2 * q_dim * hidden, next(), hidden, 1.0),
                        "spec.mtp.q",
                    ),
                    k_proj: make_buffer_f16(
                        device,
                        &projection(kv_dim * hidden, next(), hidden, 1.0),
                        "spec.mtp.k",
                    ),
                    v_proj: make_buffer_f16(
                        device,
                        &projection(kv_dim * hidden, next(), hidden, 1.0),
                        "spec.mtp.v",
                    ),
                    o_proj: make_buffer_f16(
                        device,
                        &projection(hidden * q_dim, next(), q_dim, 0.7),
                        "spec.mtp.o",
                    ),
                    q_norm: make_buffer(device, &vec![1.0; cfg.head_dim], "spec.mtp.qn"),
                    k_norm: make_buffer(device, &vec![1.0; cfg.head_dim], "spec.mtp.kn"),
                    mlp: MetalMtpDenseMlpWeights {
                        gate_proj: make_buffer_f16(
                            device,
                            &projection(inter * hidden, next(), hidden, 1.0),
                            "spec.mtp.gate",
                        ),
                        up_proj: make_buffer_f16(
                            device,
                            &projection(inter * hidden, next(), hidden, 1.0),
                            "spec.mtp.up",
                        ),
                        down_proj: make_buffer_f16(
                            device,
                            &projection(hidden * inter, next(), inter, 0.5),
                            "spec.mtp.down",
                        ),
                    },
                }],
                norm: make_buffer(device, &vec![1.0; hidden], "spec.mtp.norm"),
                hidden_tap: MtpHiddenTap::PostFinalNorm,
            }
        }
    }
}

/// The patterned hybrid model with an MTP head attached.
fn mtp_state(
    weights: &ModelWeights,
    cfg: &Qwen35Config,
    head: Head,
    max_cache: usize,
) -> MetalQwen35State {
    let mut cfg = cfg.clone();
    cfg.mtp_num_hidden_layers = 1;
    let mut engine = MetalQwen35Engine::new(weights, &cfg).expect("patterned engine with MTP head");
    engine.mtp_weights = Some(mtp_head(&engine.device, &cfg, head));
    let session = engine
        .new_session(max_cache)
        .expect("patterned MTP session constructs");
    MetalQwen35State {
        engine,
        session,
        lora: None,
        use_gdn_chunked: true,
        use_kv_f16: false,
        cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
        path_proof_enabled: false,
        path_proof: PathProofCounters::default(),
    }
}

fn greedy_cfg(max_new_tokens: usize, stop: &[u32], mtp: bool) -> GenerateConfig {
    GenerateConfig {
        min_p: 0.0,
        max_new_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: stop.to_vec(),
        enable_thinking: false,
        enable_mtp: Some(mtp),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    }
}

fn state_line(state: &MetalQwen35State) -> String {
    let hidden_bits = state
        .session
        .last_pre_final_hidden
        .iter()
        .fold(0u64, |acc, v| acc.wrapping_add(u64::from(v.to_bits())));
    let pool = state.session.gdn_checkpoints.as_ref().map(|p| {
        (
            p.active_base_seq_len,
            p.mtp_base_seq_len,
            p.batch_repair_token,
        )
    });
    format!(
        "kv={} mtp={:?} hid={:016x} cursor={:?} pool={:?}",
        state.session.kv_cache.seq_len,
        state.session.mtp.as_ref().map(|m| m.cache.seq_len),
        hidden_bits,
        state.session.last_hidden_cursor,
        pool,
    )
}

fn describe(label: &str, out: &GenerateOutput, state: &MetalQwen35State) -> String {
    format!(
        "SPEC {label} ids={:?} text={:?} stopped={} reason={:?} gen={} prompt={} | {}",
        out.token_ids,
        out.text,
        out.stopped,
        out.stop_reason,
        out.generated_tokens,
        out.prompt_tokens,
        state_line(state),
    )
}

fn generate_line(
    label: &str,
    state: &mut MetalQwen35State,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) -> (String, Vec<u32>) {
    eprintln!("SPEC-BEGIN {label}");
    match state.generate(prompt, tokenizer, gen_cfg) {
        Ok(out) => (describe(label, &out, state), out.token_ids),
        Err(error) => (format!("SPEC {label} ERR {error:?}"), Vec::new()),
    }
}

/// The patterned model with its residual-adding projections scaled up, so the
/// greedy continuation depends on the context instead of repeating the prompt's
/// last token.
fn diverse_model(boost: f32) -> (Qwen35Config, ModelWeights) {
    let (cfg, mut weights) = patterned_model();
    for (attention, common) in &mut weights.layers {
        match attention {
            AttentionWeights::Linear(gdn) => gdn.out_proj.iter_mut().for_each(|v| *v *= boost),
            AttentionWeights::Full(full) => full.o_proj.iter_mut().for_each(|v| *v *= boost),
        }
        if let FeedForwardWeights::Dense(ffn) = &mut common.ffn {
            ffn.down_proj.iter_mut().for_each(|v| *v *= boost);
        }
    }
    (cfg, weights)
}

#[test]
fn speculative_route_characterization() {
    let _gpu = gpu_test_lock();
    if Device::system_default().is_none() {
        eprintln!("SKIP speculative route characterization: no Metal device");
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let models = [
        ("flat", patterned_model()),
        ("boost6", diverse_model(6.0)),
        ("boost12", diverse_model(12.0)),
    ];

    for (model, (cfg, weights)) in &models {
        // Default MTP, both heads, two prompts: length sweep, stop sweep and the
        // ordinary reference beside every row.
        for head in [Head::Copy, Head::Scrambled] {
            for prompt in ["abc", "hello"] {
                let mut state = mtp_state(weights, cfg, head, 64);
                let (line, reference) = generate_line(
                    &format!("ord/{model}/{head:?}/{prompt}/n16"),
                    &mut state,
                    &tokenizer,
                    prompt,
                    &greedy_cfg(16, &[], false),
                );
                eprintln!("{line}");
                for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24] {
                    let (line, _) = generate_line(
                        &format!("mtp/{model}/{head:?}/{prompt}/n{n}"),
                        &mut state,
                        &tokenizer,
                        prompt,
                        &greedy_cfg(n, &[], true),
                    );
                    eprintln!("{line}");
                }
                for k in 0..10usize.min(reference.len()) {
                    let (line, _) = generate_line(
                        &format!("mtp/{model}/{head:?}/{prompt}/stop@{k}"),
                        &mut state,
                        &tokenizer,
                        prompt,
                        &greedy_cfg(16, &[reference[k]], true),
                    );
                    eprintln!("{line}");
                }
            }
        }

        // The cache boundary: prompt plus budget filling the whole context.
        for head in [Head::Copy, Head::Scrambled] {
            let mut state = mtp_state(weights, cfg, head, 16);
            for n in [13usize, 12, 11, 10, 8] {
                let (line, _) = generate_line(
                    &format!("mtp-edge/{model}/{head:?}/abc/n{n}"),
                    &mut state,
                    &tokenizer,
                    "abc",
                    &greedy_cfg(n, &[], true),
                );
                eprintln!("{line}");
                let (line, _) = generate_line(
                    &format!("ord-edge/{model}/{head:?}/abc/n{n}"),
                    &mut state,
                    &tokenizer,
                    "abc",
                    &greedy_cfg(n, &[], false),
                );
                eprintln!("{line}");
            }
        }

        // Self-speculation on the same model.
        for prompt in ["abc", "hello"] {
            let mut state = with_self_spec_env(|| {
                MetalQwen35State::new(weights, cfg, 64).expect("patterned hybrid state")
            });
            let (line, reference) = with_self_spec_env(|| {
                generate_line(
                    &format!("self-ord/{model}/{prompt}/n16"),
                    &mut state,
                    &tokenizer,
                    prompt,
                    &greedy_cfg(16, &[], false),
                )
            });
            eprintln!("{line}");
            for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24] {
                let (line, _) = with_self_spec_env(|| {
                    generate_line(
                        &format!("self/{model}/{prompt}/n{n}"),
                        &mut state,
                        &tokenizer,
                        prompt,
                        &greedy_cfg(n, &[], false),
                    )
                });
                eprintln!("{line}");
            }
            for k in 0..10usize.min(reference.len()) {
                let (line, _) = with_self_spec_env(|| {
                    generate_line(
                        &format!("self/{model}/{prompt}/stop@{k}"),
                        &mut state,
                        &tokenizer,
                        prompt,
                        &greedy_cfg(16, &[reference[k]], false),
                    )
                });
                eprintln!("{line}");
            }
        }
        let mut state = with_self_spec_env(|| {
            MetalQwen35State::new(weights, cfg, 16).expect("patterned hybrid state")
        });
        for n in [13usize, 10, 8, 6, 4] {
            let (line, _) = with_self_spec_env(|| {
                generate_line(
                    &format!("self-edge/{model}/abc/n{n}"),
                    &mut state,
                    &tokenizer,
                    "abc",
                    &greedy_cfg(n, &[], false),
                )
            });
            eprintln!("{line}");
        }
    }
}

/// Forced first candidate on the zero-weight fixtures, driven through the route
/// entries directly: a constant draft of token 0 against a target that predicts
/// the first token of the pending token's residue class mod 3.
#[test]
fn speculative_route_forced_candidates_characterization() {
    let _gpu = gpu_test_lock();
    if Device::system_default().is_none() {
        eprintln!("SKIP forced speculative characterization: no Metal device");
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    for pending in [0usize, 3, 5, 2] {
        for stop in [vec![], vec![0u32], vec![2u32]] {
            if stop.contains(&(pending as u32)) {
                continue;
            }
            for n in [1usize, 2, 3, 4, 5, 6] {
                let mut state = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
                let mut logits = vec![-1.0f32; cfg.vocab_size];
                logits[pending] = 100.0;
                let gen_cfg = greedy_cfg(n, &stop, true);
                let out = state
                    .generate_greedy_mtp(&logits, 0, &tokenizer, &gen_cfg)
                    .expect("mtp route");
                eprintln!(
                    "{}",
                    describe(
                        &format!("forced-mtp/p{pending}/stop{stop:?}/n{n}"),
                        &out,
                        &state
                    )
                );
            }
        }
    }

    let (hybrid_cfg, hybrid_weights) = tiny_hybrid_fixture();
    for pending in [3usize, 5] {
        for stop in [vec![], vec![1u32], vec![0u32]] {
            if stop.contains(&(pending as u32)) {
                continue;
            }
            for n in [1usize, 2, 3, 4, 5, 8] {
                let out_and_state = with_self_spec_env(|| {
                    let mut state = MetalQwen35State::new(&hybrid_weights, &hybrid_cfg, 32)
                        .expect("tiny hybrid fixture");
                    state.reset_state();
                    let real = state.forward_prefill(&[1u32]);
                    let mut logits = vec![0.0f32; real.len()];
                    logits[pending] = 100.0;
                    let gen_cfg = greedy_cfg(n, &stop, false);
                    let out = state
                        .generate_greedy_self_spec(&logits, 1, &tokenizer, &gen_cfg)
                        .expect("self-spec route");
                    describe(
                        &format!("forced-self/p{pending}/stop{stop:?}/n{n}"),
                        &out,
                        &state,
                    )
                });
                eprintln!("{out_and_state}");
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Asserting tests for the shared speculative route (ADR-090 D6).
// -----------------------------------------------------------------------------

/// The route the MTP selector picks for a request with no adapter loaded, read from the
/// process environment exactly as the entry reads it.
/// A forced-candidate scenario: pending token, stop ids, token budget, expected ids and
/// whether the request is expected to end on a stop.
type ForcedRow = (usize, &'static [u32], usize, &'static [u32], bool);

fn expected_mtp_route() -> SpeculativeRoute {
    if crate::forward::metal_qwen35::use_batch_gemm_verifier(
        crate::env_switch_enabled("LATTICE_MTP_BATCH"),
        false,
    ) {
        SpeculativeRoute::MtpBatchGemmLegacy
    } else {
        SpeculativeRoute::MtpShared
    }
}

fn assert_same_output(label: &str, ordinary: &GenerateOutput, speculative: &GenerateOutput) {
    assert_eq!(
        (
            &speculative.token_ids,
            &speculative.text,
            speculative.stopped,
            speculative.stop_reason
        ),
        (
            &ordinary.token_ids,
            &ordinary.text,
            ordinary.stopped,
            ordinary.stop_reason
        ),
        "{label}: the speculative route must commit what ordinary greedy commits"
    );
}

/// The default MTP route (selector off) commits exactly what ordinary greedy commits, for
/// length limits and for stop tokens that fall inside verified blocks, and records the
/// shared route with every offered token published. With `LATTICE_MTP_BATCH` on, the same
/// requests are recorded as the excluded legacy route and offer nothing to the policy.
#[test]
fn speculative_route_default_mtp_matches_ordinary_and_is_marked() {
    let _gpu = gpu_test_lock();
    if Device::system_default().is_none() {
        eprintln!("SKIP default MTP route assertions: no Metal device");
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let expected = expected_mtp_route();
    let mut shared_rows = 0usize;
    for (model, (cfg, weights)) in [("flat", patterned_model()), ("boost6", diverse_model(6.0))] {
        for head in [Head::Copy, Head::Scrambled] {
            for prompt in ["abc", "hello"] {
                let mut state = mtp_state(&weights, &cfg, head, 64);
                let reference = state
                    .generate(prompt, &tokenizer, &greedy_cfg(16, &[], false))
                    .expect("ordinary reference");
                assert!(
                    state.session.speculative_route.is_none(),
                    "an ordinary request must not carry a speculative route record"
                );
                let mut requests: Vec<Vec<u32>> = vec![vec![]];
                for k in [0usize, 3, 6] {
                    requests.push(vec![reference.token_ids[k]]);
                }
                for stop in requests {
                    for n in [1usize, 3, 5, 8, 16] {
                        let label = format!("{model}/{head:?}/{prompt}/stop{stop:?}/n{n}");
                        let ordinary = state
                            .generate(prompt, &tokenizer, &greedy_cfg(n, &stop, false))
                            .expect("ordinary");
                        let out = state
                            .generate(prompt, &tokenizer, &greedy_cfg(n, &stop, true))
                            .expect("mtp");
                        let record = state
                            .session
                            .speculative_route
                            .expect("an MTP request records its route");
                        assert_eq!(record.route, expected, "{label}");
                        if record.route.uses_shared_policy() {
                            shared_rows += 1;
                            assert_same_output(&label, &ordinary, &out);
                            assert_eq!(
                                record.trace.offered,
                                out.token_ids.len(),
                                "{label}: every token the policy was offered is published"
                            );
                            assert_eq!(
                                record.trace.rounds > 0,
                                !out.token_ids.is_empty(),
                                "{label}"
                            );
                        } else {
                            assert_eq!(
                                record.trace,
                                crate::decoder::SpeculativeTrace::default(),
                                "{label}: the legacy route offers nothing to the shared policy"
                            );
                        }
                    }
                }
            }
        }
    }
    eprintln!("SPEC-ASSERT default-mtp expected={expected:?} shared_rows={shared_rows}");
}

/// Self-speculation under its selector commits what ordinary greedy commits and records
/// its own shared route.
#[test]
fn speculative_route_self_spec_matches_ordinary_and_is_marked() {
    let _gpu = gpu_test_lock();
    if Device::system_default().is_none() {
        eprintln!("SKIP self-spec route assertions: no Metal device");
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    for (model, (cfg, weights)) in [("flat", patterned_model()), ("boost6", diverse_model(6.0))] {
        for prompt in ["abc", "hello"] {
            let mut state = with_self_spec_env(|| {
                MetalQwen35State::new(&weights, &cfg, 64).expect("patterned hybrid state")
            });
            let reference = with_self_spec_env(|| {
                state
                    .generate(prompt, &tokenizer, &greedy_cfg(16, &[], false))
                    .expect("self-spec request")
            });
            let mut requests: Vec<Vec<u32>> = vec![vec![]];
            for k in [0usize, 3, 6] {
                requests.push(vec![reference.token_ids[k]]);
            }
            for stop in requests {
                for n in [1usize, 3, 5, 8, 16] {
                    let label = format!("{model}/{prompt}/stop{stop:?}/n{n}");
                    let out = with_self_spec_env(|| {
                        state
                            .generate(prompt, &tokenizer, &greedy_cfg(n, &stop, false))
                            .expect("self-spec")
                    });
                    let record = state
                        .session
                        .speculative_route
                        .expect("a self-spec request records its route");
                    assert_eq!(record.route, SpeculativeRoute::SelfSpecShared, "{label}");
                    assert!(record.route.uses_shared_policy(), "{label}");
                    assert_eq!(record.trace.offered, out.token_ids.len(), "{label}");
                    // Ordinary greedy for the same request: the selector is read per
                    // request, so a plain state built outside the env scope is the reference.
                    let mut plain =
                        MetalQwen35State::new(&weights, &cfg, 64).expect("patterned hybrid state");
                    let ordinary = plain
                        .generate(prompt, &tokenizer, &greedy_cfg(n, &stop, false))
                        .expect("ordinary");
                    assert_same_output(&label, &ordinary, &out);
                }
            }
        }
    }
}

/// A rejected draft is rolled back exactly: after every round the KV cursor sits at the
/// prompt plus the committed tokens, and the tokens committed across rounds, including the
/// rounds after a rejection, equal ordinary greedy.
#[test]
fn speculative_route_rejection_restores_state_and_next_tokens_equal_ordinary() {
    let _gpu = gpu_test_lock();
    if Device::system_default().is_none() {
        eprintln!("SKIP rejection assertions: no Metal device");
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = diverse_model(6.0);
    let mut state = mtp_state(&weights, &cfg, Head::Scrambled, 64);
    let reference = state
        .generate("abc", &tokenizer, &greedy_cfg(24, &[], false))
        .expect("ordinary reference");

    let gen_cfg = greedy_cfg(24, &[], true);
    let GenerationPreparation::Ready(plan) = state
        .prepare_direct_generation("abc", &tokenizer, &gen_cfg)
        .expect("prepare")
    else {
        panic!("a three-character prompt prepares for generation");
    };
    state.reset_state();
    let prefill = state
        .try_forward_prefill(&plan.prompt_ids)
        .expect("prefill");
    state.session.mtp_active = true;
    state.mtp_prefill(&plan.prompt_ids);

    let eos = state.engine.config.eos_token_id;
    let is_stop = move |id: u32| id == eos;
    let mut metrics = SpeculativeMetrics::mtp();
    let mut pending = crate::sampling::argmax_f32_first_wins(&prefill);
    let mut committed: Vec<u32> = Vec::new();
    while committed.len() < 16 {
        let round = state.speculative_round(&mut metrics, pending, usize::MAX / 2, &is_stop);
        assert!(!round.cache_full);
        committed.extend_from_slice(&round.committed);
        assert_eq!(
            state.session.kv_cache.seq_len,
            plan.prompt_ids.len() + committed.len(),
            "the KV cursor must sit exactly past the committed tokens after every round"
        );
        pending = round.next.expect("a continuation");
    }
    assert_eq!(committed, reference.token_ids[..committed.len()]);
    let SpeculativeMetrics::Mtp(counters) = &metrics else {
        panic!("an MTP metrics value");
    };
    assert!(
        counters.verify_calls > counters.accepted_extra_tokens,
        "the scenario must contain rejected drafts: verify={} accepted={}",
        counters.verify_calls,
        counters.accepted_extra_tokens
    );
}

/// A stop token inside a verified block ends the output at the same token as before the
/// route moved to the shared policy. The expected rows are the pre-change results of the
/// same forced-candidate scenarios.
#[test]
fn speculative_route_stop_and_length_inside_a_verified_block() {
    let _gpu = gpu_test_lock();
    if Device::system_default().is_none() {
        eprintln!("SKIP forced stop assertions: no Metal device");
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let expected_route = expected_mtp_route();
    // (pending, stop ids, n, ids, stopped)
    let mtp_rows: [ForcedRow; 6] = [
        (3, &[0], 1, &[3], false),
        (3, &[0], 2, &[3], true),
        (3, &[0], 6, &[3], true),
        (5, &[2], 2, &[5], true),
        (5, &[], 3, &[5, 2, 2], false),
        (0, &[], 3, &[0, 0, 0], false),
    ];
    for (pending, stop, n, ids, stopped) in mtp_rows {
        let mut state = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
        let mut logits = vec![-1.0f32; cfg.vocab_size];
        logits[pending] = 100.0;
        let out = state
            .generate_greedy_mtp(&logits, 0, &tokenizer, &greedy_cfg(n, stop, true))
            .expect("mtp route");
        let label = format!("forced-mtp/p{pending}/stop{stop:?}/n{n}");
        assert_eq!(out.token_ids, ids, "{label}");
        assert_eq!(out.stopped, stopped, "{label}");
        let reason = if stopped {
            StopReason::Eos
        } else {
            StopReason::Length
        };
        assert_eq!(out.stop_reason, Some(reason), "{label}");
        let record = state.session.speculative_route.expect("route record");
        assert_eq!(record.route, expected_route, "{label}");
    }

    let (hybrid_cfg, hybrid_weights) = tiny_hybrid_fixture();
    let self_rows: [ForcedRow; 4] = [
        (3, &[1], 1, &[3], false),
        (3, &[1], 4, &[3], true),
        (5, &[1], 8, &[5], true),
        (3, &[0], 8, &[3, 1, 1, 1, 1, 1, 1, 1], false),
    ];
    for (pending, stop, n, ids, stopped) in self_rows {
        let out = with_self_spec_env(|| {
            let mut state = MetalQwen35State::new(&hybrid_weights, &hybrid_cfg, 32)
                .expect("tiny hybrid fixture");
            state.reset_state();
            let real = state.forward_prefill(&[1u32]);
            let mut logits = vec![0.0f32; real.len()];
            logits[pending] = 100.0;
            let out = state
                .generate_greedy_self_spec(&logits, 1, &tokenizer, &greedy_cfg(n, stop, false))
                .expect("self-spec route");
            let record = state.session.speculative_route.expect("route record");
            assert_eq!(record.route, SpeculativeRoute::SelfSpecShared);
            out
        });
        let label = format!("forced-self/p{pending}/stop{stop:?}/n{n}");
        assert_eq!(out.token_ids, ids, "{label}");
        assert_eq!(out.stopped, stopped, "{label}");
    }
}

/// The two MTP entries are told apart by their recorded route, whatever the selector says:
/// the legacy loop records the excluded route and never reaches the shared driver, and the
/// shared entry records the shared route.
#[test]
fn speculative_route_batch_gemm_loop_is_marked_excluded_and_never_shared() {
    let _gpu = gpu_test_lock();
    if Device::system_default().is_none() {
        eprintln!("SKIP batch exclusion assertions: no Metal device");
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let mut logits = vec![-1.0f32; cfg.vocab_size];
    logits[5] = 100.0;
    let gen_cfg = greedy_cfg(4, &[], true);

    let mut legacy = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
    let out = legacy.generate_greedy_mtp_batch_gemm_legacy(&logits, 0, &tokenizer, &gen_cfg);
    assert!(!out.token_ids.is_empty());
    let record = legacy.session.speculative_route.expect("legacy record");
    assert_eq!(record.route, SpeculativeRoute::MtpBatchGemmLegacy);
    assert!(!record.route.uses_shared_policy());
    assert_eq!(record.trace, crate::decoder::SpeculativeTrace::default());

    let mut shared = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
    let out = shared
        .generate_speculative_shared(SpeculativeMetrics::mtp(), &logits, 0, &tokenizer, &gen_cfg)
        .expect("shared route");
    let record = shared.session.speculative_route.expect("shared record");
    assert_eq!(record.route, SpeculativeRoute::MtpShared);
    assert!(record.route.uses_shared_policy());
    assert_eq!(record.trace.offered, out.token_ids.len());

    // The selector entry picks between the two exactly as the environment says.
    let mut entry = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
    entry
        .generate_greedy_mtp(&logits, 0, &tokenizer, &gen_cfg)
        .expect("selector entry");
    assert_eq!(
        entry.session.speculative_route.expect("entry record").route,
        expected_mtp_route()
    );
}

/// The real Q4 checkpoint with its MTP head, driven through the entry the selectors choose:
/// `LATTICE_MTP` unset must leave no speculative record, set must record the route the
/// batch selector implies. Prints the actual output token counts and the route; nothing is
/// timed. Skipped, never passed, when the checkpoint or tokenizer is absent.
#[test]
fn speculative_route_real_checkpoint_follows_the_declared_selectors() {
    let _gpu = gpu_test_lock();
    if Device::system_default().is_none() {
        eprintln!("SKIP real-checkpoint speculative route: no Metal device");
        return;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let dir = std::env::var("LATTICE_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(&home).join(".lattice/models/qwen3.5-0.8b-q4")
        });
    let tok_dir = std::env::var("LATTICE_TOKENIZER_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(&home).join(".lattice/models/qwen3.5-0.8b"));
    if !dir.join("config.json").exists()
        || !dir.join("mtp_fc_weight.q4").exists()
        || !tok_dir.join("tokenizer.json").exists()
    {
        eprintln!(
            "SKIP real-checkpoint speculative route: Q4 checkpoint with MTP weights or tokenizer \
             missing"
        );
        return;
    }
    let cfg = Qwen35Config::from_config_json(&dir.join("config.json")).expect("checkpoint config");
    let tok_path = tok_dir.join("tokenizer.json");
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tok_path).expect("checkpoint tokenizer");
    let mut state =
        MetalQwen35State::from_q4_dir(&dir, &tok_path, &cfg, 4096).expect("Q4 checkpoint loads");

    let prompt = "Explain the theory of general relativity in simple terms, covering spacetime curvature and";
    let mut gen_cfg = greedy_cfg(32, &[], false);
    gen_cfg.enable_mtp = Some(false);
    let baseline = state
        .generate(prompt, &tokenizer, &gen_cfg)
        .expect("ordinary baseline");
    assert!(state.session.speculative_route.is_none());

    // Selector-driven: `enable_mtp` unset defers to `LATTICE_MTP`.
    gen_cfg.enable_mtp = None;
    let mtp_selected = crate::env_switch_enabled("LATTICE_MTP");
    let out = state
        .generate(prompt, &tokenizer, &gen_cfg)
        .expect("selector-driven request");
    let record = state.session.speculative_route;
    let batch_selected = crate::env_switch_enabled("LATTICE_MTP_BATCH");
    match record {
        None => {
            assert!(
                !mtp_selected,
                "LATTICE_MTP is set but no MTP route was taken"
            );
            eprintln!(
                "REAL-ROUTE LATTICE_MTP=unset LATTICE_MTP_BATCH={batch_selected} route=ordinary \
                 tokens={} equal_to_baseline={}",
                out.token_ids.len(),
                out.token_ids == baseline.token_ids
            );
        }
        Some(record) => {
            assert!(mtp_selected, "an MTP route was taken without LATTICE_MTP");
            assert_eq!(record.route, expected_mtp_route());
            if record.route.uses_shared_policy() {
                assert_eq!(record.trace.offered, out.token_ids.len());
            }
            eprintln!(
                "REAL-ROUTE LATTICE_MTP=set LATTICE_MTP_BATCH={batch_selected} route={} \
                 shared_policy={} tokens={} offered={} driver_rounds={} equal_to_baseline={}",
                record.route.label(),
                record.route.uses_shared_policy(),
                out.token_ids.len(),
                record.trace.offered,
                record.trace.rounds,
                out.token_ids == baseline.token_ids
            );
        }
    }
    eprintln!("REAL-ROUTE baseline_tokens={}", baseline.token_ids.len());
}
