//! Cross-turn KV prefix cache benchmark (#462).
//!
//! Compares `chat_metal`'s pre-#462 behavior (`reset_state()` + a full
//! re-prefill of the entire ChatML-formatted conversation on every turn)
//! against the cache-aware path shipped in #462
//! (`chat_completion_streaming_with_prefix_cache`, which reuses the previous
//! turn's shared KV/GDN state and prefills only the new suffix), at a couple
//! of conversation depths.
//!
//! Besides those two, the cache path is timed on three more request shapes,
//! each asserting the `CrossTurnCacheStats` mode it is named for and a
//! nonempty output: `cache_aware_fresh` (nothing retained, so `FullRefill`),
//! `cache_aware_invalidated` (a warm entry the new history diverges from, so
//! `FullRefill` again) and `cache_aware_cancelled` (the incremental turn
//! cancelled after its first delta).
//!
//! `cross_turn_long_reply` times the turn after a long reply. One warm turn
//! asks for a long answer and is cut at exactly N tokens; the timed turn asks a
//! follow-up and decodes one token, so it is the prefill plus one step. Two arms
//! differ only in how the reply comes back as history: `template_replay` renders
//! it as the published template does, which drops the reply's empty reasoning
//! block, so the turn replays the whole reply from the latest turn start;
//! `keep_empty_append` keeps the block, so the turn appends to the cached reply.
//! Their difference is the cost of re-prefilling an N-token reply. The prompt,
//! reused and prefilled token counts of each arm are printed after the group.
//!
//! Requires a real Qwen3.5 model at `~/.lattice/models/qwen3.5-0.8b/`
//! (config.json + safetensors + tokenizer.json). A missing or unloadable model
//! panics: a run that measured nothing must not read as a pass.
//!
//! Run: `cargo bench -p lattice-inference --features metal-gpu,f16 -- cross_turn_prefix_cache`
#![allow(clippy::field_reassign_with_default)]

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::path::PathBuf;
use std::time::Duration;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::GenerateConfig;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::kv_cache::{CrossTurnSlotId, PrefixReuseMode};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::tokenizer::bpe::BpeTokenizer;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_model_qwen35::forward::metal_qwen35::ChatHistoryReasoning;

/// Nine short, realistic user turns. Long enough to cover both depths
/// benchmarked below (2 and 8 prior turns), with one extra turn left over as
/// the "next" (timed) turn at the deepest configuration.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const USER_TURNS: &[&str] = &[
    "What is the capital of France?",
    "Can you tell me a bit about its history?",
    "What are some famous landmarks there?",
    "How is the weather typically in spring?",
    "What local foods should I try?",
    "Is it easy to get around without a car?",
    "What is the best time of year to visit?",
    "Can you summarize everything we've discussed?",
    "One more question, any day trip suggestions?",
];

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const SYSTEM_PROMPT: &str = "You are a helpful, concise travel assistant.";

/// Shares only a short opening with `SYSTEM_PROMPT`, so a history that starts
/// with it diverges from a warm entry built on `SYSTEM_PROMPT` before any turn
/// boundary the entry retains a checkpoint at.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const DIVERGENT_SYSTEM_PROMPT: &str = "You are a terse assistant.";

/// Asks for an answer longer than any reply budget below, so the warm turn of
/// `cross_turn_long_reply` is cut by its budget rather than ending on its own.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const LONG_REPLY_REQUEST: &str =
    "Write a long, detailed essay on the history of Paris, from its founding to today.";

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const LONG_REPLY_FOLLOW_UP: &str = "Summarize that in one sentence.";

fn safetensors_model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b"));
    if dir.join("config.json").exists() {
        Some(dir)
    } else {
        None
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn load_state_and_tokenizer() -> Result<(MetalQwen35State, BpeTokenizer), String> {
    use lattice_inference::model::qwen35::Qwen35Model;
    let dir = safetensors_model_dir().ok_or_else(|| {
        "no config.json under ~/.lattice/models/qwen3.5-0.8b (or HOME is unset)".to_string()
    })?;
    let model = Qwen35Model::from_safetensors(&dir)
        .map_err(|error| format!("loading the safetensors weights failed: {error}"))?;
    let cfg = model.config().clone();
    let state = MetalQwen35State::new(model.weights(), &cfg, 4096)
        .map_err(|error| format!("building the Metal state failed: {error}"))?;
    let tokenizer = BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))
        .map_err(|error| format!("loading tokenizer.json failed: {error}"))?;
    Ok((state, tokenizer))
}

/// Replays `prior_turns` real exchanges through the prefix-cache path from an
/// empty cache, so the slot ends up holding the conversation so far, and
/// returns that history with the next user turn appended, together with the
/// latest turn start: the token length of the last replayed prompt, which
/// ends at its assistant opener.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn warm_cache_through_prior_turns(
    state: &mut MetalQwen35State,
    tokenizer: &BpeTokenizer,
    gen_cfg: &GenerateConfig,
    slot_id: CrossTurnSlotId,
    prior_turns: usize,
) -> (Vec<ChatMessage>, usize) {
    state.reset_state();
    state.clear_cross_turn_prefix_cache();
    let mut history = vec![ChatMessage::system(SYSTEM_PROMPT)];
    let mut latest_turn_start = 0;
    for user_text in &USER_TURNS[..prior_turns] {
        history.push(ChatMessage::user(*user_text));
        let out = state
            .chat_completion_streaming_with_prefix_cache(
                slot_id,
                &history,
                tokenizer,
                gen_cfg,
                |_delta, _id| true,
            )
            .expect("cache warm-up turn must succeed");
        latest_turn_start = out.cache.prompt_tokens;
        history.push(out.output.message);
    }
    history.push(ChatMessage::user(USER_TURNS[prior_turns]));
    (history, latest_turn_start)
}

/// An arm that did not take the path it is named for, or produced nothing,
/// measured something else: refuse instead of timing it.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn require_arm(
    arm: &str,
    mode: PrefixReuseMode,
    expected_mode: PrefixReuseMode,
    completion_tokens: usize,
) {
    assert_eq!(
        mode, expected_mode,
        "{arm} arm must plan {expected_mode:?}, got {mode:?}"
    );
    assert!(
        completion_tokens > 0,
        "{arm} arm produced no completion tokens"
    );
}

/// The incremental arms reuse everything up to the latest turn start: an
/// exact append when the previous reply renders unchanged, or a replay from
/// exactly that turn start when the template rewrites the reply (an earlier
/// assistant turn renders without its reasoning block). A replay from any
/// earlier boundary refills turns the cache already held, so it is refused.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn require_incremental_arm(
    arm: &str,
    mode: PrefixReuseMode,
    latest_turn_start: usize,
    completion_tokens: usize,
) {
    let expected_replay = PrefixReuseMode::ReplayFromCheckpoint {
        checkpoint_len: latest_turn_start,
    };
    assert!(
        mode == PrefixReuseMode::ExactAppend || mode == expected_replay,
        "{arm} arm must plan ExactAppend or {expected_replay:?}, got {mode:?}"
    );
    assert!(
        completion_tokens > 0,
        "{arm} arm produced no completion tokens"
    );
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn bench_gen_cfg(max_new_tokens: usize) -> GenerateConfig {
    let mut cfg = GenerateConfig::default();
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature = 0.0;
    cfg.top_k = 1;
    cfg.top_p = 1.0;
    cfg.repetition_penalty = 1.0;
    cfg.seed = Some(7);
    cfg.stop_token_ids = vec![];
    cfg.enable_thinking = false;
    cfg.enable_mtp = Some(false);
    cfg.grammar = None;
    cfg.stop_strings = vec![];
    cfg.reasoning_budget = None;
    cfg.logprobs = None;
    cfg
}

/// Benchmarks one conversation depth: `prior_turns` completed user/assistant
/// exchanges already in history, then one more ("next") user turn.
///
/// * `full_reprefill` times `chat_metal`'s pre-#462 behavior: reset the whole
///   session and re-run `chat_completion_streaming` over the entire
///   `prior_turns + 1` history from scratch, every call.
/// * `cache_aware_incremental` times the #462 path: `setup` (excluded from
///   timing) rebuilds the cache by replaying `prior_turns` through
///   `chat_completion_streaming_with_prefix_cache`, then the timed `routine`
///   is just the one incremental final-turn call. It reuses everything up to
///   the latest turn start and prefills only what follows: `ExactAppend` when
///   the previous reply renders unchanged, or `ReplayFromCheckpoint` at that
///   turn start when the template drops the reply's reasoning block. The plan
///   each depth took is printed after the group.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn bench_depth(
    c: &mut Criterion,
    state: &mut MetalQwen35State,
    tokenizer: &BpeTokenizer,
    prior_turns: usize,
    label: &str,
) {
    let gen_cfg = bench_gen_cfg(32);
    let slot_id = CrossTurnSlotId::DEFAULT;
    let next_user = USER_TURNS[prior_turns];

    // Build the fixed `prior_turns`-deep history once with real, greedy-decoded
    // assistant replies (not placeholder text), so `full_reprefill`'s workload
    // has realistic token counts. The cache-aware path is just as valid a way
    // to harvest this since it always produces token-identical output to the
    // plain path (see the `cross_turn_cache_*` correctness tests).
    state.reset_state();
    state.clear_cross_turn_prefix_cache();
    let mut old_path_history = vec![ChatMessage::system(SYSTEM_PROMPT)];
    for user_text in &USER_TURNS[..prior_turns] {
        old_path_history.push(ChatMessage::user(*user_text));
        let out = state
            .chat_completion_streaming_with_prefix_cache(
                slot_id,
                &old_path_history,
                tokenizer,
                &gen_cfg,
                |_delta, _id| true,
            )
            .expect("harvest turn must succeed");
        old_path_history.push(out.output.message);
    }
    old_path_history.push(ChatMessage::user(next_user));

    let mut group = c.benchmark_group("cross_turn_prefix_cache");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    // `full_reprefill`'s closure and `cache_aware_incremental`'s two closures
    // (setup + routine) each need their own turn at mutably borrowing `state`;
    // Criterion invokes them strictly sequentially (never concurrently), so a
    // RefCell lets every closure below take an independent `&mut` in turn
    // instead of fighting over one long-lived borrow.
    let state_cell = std::cell::RefCell::new(state);

    // Old path (pre-#462): reset + full re-prefill of the whole conversation
    // (including the new turn) from scratch, every single call.
    group.bench_function(BenchmarkId::new("full_reprefill", label), |b| {
        b.iter(|| {
            let mut state = state_cell.borrow_mut();
            state.reset_state();
            let _ = state.chat_completion_streaming(
                &old_path_history,
                tokenizer,
                &gen_cfg,
                |_delta, _id| true,
            );
        });
    });

    // New path (#462): `setup` rebuilds the cache up to `prior_turns` (real
    // model calls, excluded from timing — "the conversation so far"), then
    // the timed `routine` is only the incremental next-turn call.
    // The plan each incremental arm took, printed after the group so the
    // output records whether the depth was an exact append or a replay.
    let incremental_plan = std::cell::Cell::new(None::<PrefixReuseMode>);
    let cancelled_plan = std::cell::Cell::new(None::<PrefixReuseMode>);

    group.bench_function(BenchmarkId::new("cache_aware_incremental", label), |b| {
        b.iter_batched(
            || {
                let mut state = state_cell.borrow_mut();
                warm_cache_through_prior_turns(
                    &mut state,
                    tokenizer,
                    &gen_cfg,
                    slot_id,
                    prior_turns,
                )
            },
            |(history, latest_turn_start)| {
                let mut state = state_cell.borrow_mut();
                let out = state
                    .chat_completion_streaming_with_prefix_cache(
                        slot_id,
                        &history,
                        tokenizer,
                        &gen_cfg,
                        |_delta, _id| true,
                    )
                    .expect("incremental turn must succeed");
                require_incremental_arm(
                    "cache_aware_incremental",
                    out.cache.mode,
                    latest_turn_start,
                    out.output.completion_tokens,
                );
                incremental_plan.set(Some(out.cache.mode));
                out
            },
            BatchSize::SmallInput,
        );
    });

    // Fresh: the same history `full_reprefill` times, sent through the prefix
    // cache with nothing retained, so the entry cannot help and the whole
    // history is prefilled.
    group.bench_function(BenchmarkId::new("cache_aware_fresh", label), |b| {
        b.iter_batched(
            || {
                let mut state = state_cell.borrow_mut();
                state.reset_state();
                state.clear_cross_turn_prefix_cache();
            },
            |()| {
                let mut state = state_cell.borrow_mut();
                let out = state
                    .chat_completion_streaming_with_prefix_cache(
                        slot_id,
                        &old_path_history,
                        tokenizer,
                        &gen_cfg,
                        |_delta, _id| true,
                    )
                    .expect("fresh cache-path turn must succeed");
                require_arm(
                    "cache_aware_fresh",
                    out.cache.mode,
                    PrefixReuseMode::FullRefill,
                    out.output.completion_tokens,
                );
                out
            },
            BatchSize::SmallInput,
        );
    });

    // Invalidated: a warm entry exists, but the history sent opens with a
    // different system prompt, so it mismatches the entry near its start and
    // the whole history is prefilled after the entry is discarded.
    group.bench_function(BenchmarkId::new("cache_aware_invalidated", label), |b| {
        b.iter_batched(
            || {
                let mut state = state_cell.borrow_mut();
                let (mut history, _) = warm_cache_through_prior_turns(
                    &mut state,
                    tokenizer,
                    &gen_cfg,
                    slot_id,
                    prior_turns,
                );
                history[0] = ChatMessage::system(DIVERGENT_SYSTEM_PROMPT);
                history
            },
            |history| {
                let mut state = state_cell.borrow_mut();
                let out = state
                    .chat_completion_streaming_with_prefix_cache(
                        slot_id,
                        &history,
                        tokenizer,
                        &gen_cfg,
                        |_delta, _id| true,
                    )
                    .expect("invalidated turn must succeed");
                require_arm(
                    "cache_aware_invalidated",
                    out.cache.mode,
                    PrefixReuseMode::FullRefill,
                    out.output.completion_tokens,
                );
                out
            },
            BatchSize::SmallInput,
        );
    });

    // Cancelled: the incremental turn of `cache_aware_incremental`, cancelled
    // as soon as the first delta has been delivered.
    group.bench_function(BenchmarkId::new("cache_aware_cancelled", label), |b| {
        b.iter_batched(
            || {
                let mut state = state_cell.borrow_mut();
                warm_cache_through_prior_turns(
                    &mut state,
                    tokenizer,
                    &gen_cfg,
                    slot_id,
                    prior_turns,
                )
            },
            |(history, latest_turn_start)| {
                let mut state = state_cell.borrow_mut();
                let first_delta_delivered = std::cell::Cell::new(false);
                let out = state
                    .chat_completion_streaming_with_prefix_cache_and_cancel(
                        slot_id,
                        &history,
                        tokenizer,
                        &gen_cfg,
                        |_delta, _id| {
                            first_delta_delivered.set(true);
                            true
                        },
                        || first_delta_delivered.get(),
                    )
                    .expect("cancelled turn must succeed");
                require_incremental_arm(
                    "cache_aware_cancelled",
                    out.cache.mode,
                    latest_turn_start,
                    out.output.completion_tokens,
                );
                assert!(
                    !out.output.stopped && out.output.completion_tokens < gen_cfg.max_new_tokens,
                    "cache_aware_cancelled arm must be cut short by the cancel, not run to its end"
                );
                cancelled_plan.set(Some(out.cache.mode));
                out
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
    for (arm, plan) in [
        ("cache_aware_incremental", incremental_plan.get()),
        ("cache_aware_cancelled", cancelled_plan.get()),
    ] {
        if let Some(plan) = plan {
            println!("PLAN cross_turn_prefix_cache/{arm}/{label} {plan:?}");
        }
    }
}

/// Warms the slot with one long-answer turn under `history_reasoning` and
/// returns the follow-up history and that turn's start. The comparison rests on
/// the reply being exactly `reply_tokens` long and opening with the empty
/// reasoning block the two arms render differently, so anything else refuses.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn warm_long_reply(
    state: &mut MetalQwen35State,
    tokenizer: &BpeTokenizer,
    warm_cfg: &GenerateConfig,
    slot_id: CrossTurnSlotId,
    history_reasoning: ChatHistoryReasoning,
    arm: &str,
) -> (Vec<ChatMessage>, usize) {
    state.reset_state();
    state.clear_cross_turn_prefix_cache();
    state.set_chat_history_reasoning(history_reasoning);
    let mut history = vec![
        ChatMessage::system(SYSTEM_PROMPT),
        ChatMessage::user(LONG_REPLY_REQUEST),
    ];
    let out = state
        .chat_completion_streaming_with_prefix_cache(
            slot_id,
            &history,
            tokenizer,
            warm_cfg,
            |_delta, _id| true,
        )
        .expect("long-reply warm-up turn must succeed");
    assert_eq!(
        out.output.completion_tokens, warm_cfg.max_new_tokens,
        "{arm} arm: the warm reply must run its whole budget, or two reply lengths time the \
         same reply"
    );
    assert!(
        out.output
            .message
            .content
            .starts_with("<think>\n\n</think>"),
        "{arm} arm: the warm reply must open with an empty reasoning block, the part the two \
         arms render differently; got {:?}",
        out.output.message.content
    );
    let latest_turn_start = out.cache.prompt_tokens;
    history.push(out.output.message);
    history.push(ChatMessage::user(LONG_REPLY_FOLLOW_UP));
    (history, latest_turn_start)
}

/// Times the turn after a `reply_tokens`-long reply, replayed under the
/// published rendering and appended to under `KeepEmpty`.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn bench_long_reply(
    c: &mut Criterion,
    state: &mut MetalQwen35State,
    tokenizer: &BpeTokenizer,
    reply_tokens: usize,
) {
    let warm_cfg = bench_gen_cfg(reply_tokens);
    let timed_cfg = bench_gen_cfg(1);
    let slot_id = CrossTurnSlotId::DEFAULT;
    let label = format!("{reply_tokens}_reply_tokens");

    let mut group = c.benchmark_group("cross_turn_long_reply");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    let state_cell = std::cell::RefCell::new(state);
    let mut recorded = Vec::new();
    for (arm, history_reasoning) in [
        ("template_replay", ChatHistoryReasoning::Template),
        ("keep_empty_append", ChatHistoryReasoning::KeepEmpty),
    ] {
        let tokens = std::cell::Cell::new(None::<(PrefixReuseMode, usize, usize, usize)>);
        group.bench_function(BenchmarkId::new(arm, &label), |b| {
            b.iter_batched(
                || {
                    let mut state = state_cell.borrow_mut();
                    warm_long_reply(
                        &mut state,
                        tokenizer,
                        &warm_cfg,
                        slot_id,
                        history_reasoning,
                        arm,
                    )
                },
                |(history, latest_turn_start)| {
                    let mut state = state_cell.borrow_mut();
                    let out = state
                        .chat_completion_streaming_with_prefix_cache(
                            slot_id,
                            &history,
                            tokenizer,
                            &timed_cfg,
                            |_delta, _id| true,
                        )
                        .expect("turn after the long reply must succeed");
                    let expected_mode = match history_reasoning {
                        ChatHistoryReasoning::Template => PrefixReuseMode::ReplayFromCheckpoint {
                            checkpoint_len: latest_turn_start,
                        },
                        ChatHistoryReasoning::KeepEmpty | ChatHistoryReasoning::RestoreEmpty => {
                            PrefixReuseMode::ExactAppend
                        }
                    };
                    require_arm(
                        arm,
                        out.cache.mode,
                        expected_mode,
                        out.output.completion_tokens,
                    );
                    tokens.set(Some((
                        out.cache.mode,
                        out.cache.prompt_tokens,
                        out.cache.reused_tokens,
                        out.cache.prefetched_tokens,
                    )));
                    out
                },
                BatchSize::SmallInput,
            );
        });
        recorded.push((arm, tokens.get()));
    }
    group.finish();
    state_cell
        .into_inner()
        .set_chat_history_reasoning(ChatHistoryReasoning::Template);

    for (arm, tokens) in recorded {
        if let Some((mode, prompt, reused, prefilled)) = tokens {
            println!(
                "TOKENS cross_turn_long_reply/{arm}/{label} {mode:?} prompt={prompt} \
                 reused={reused} prefilled={prefilled}"
            );
        }
    }
}

fn bench_cross_turn_prefix_cache(c: &mut Criterion) {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("SKIP: cross_turn_prefix_cache bench requires macOS + metal-gpu feature");
        let _ = c;
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        let (mut state, tokenizer) = match load_state_and_tokenizer() {
            Ok(loaded) => loaded,
            Err(reason) => panic!(
                "cross_turn_prefix_cache bench requires a real model at \
                 ~/.lattice/models/qwen3.5-0.8b (config.json + safetensors + tokenizer.json): \
                 {reason}"
            ),
        };

        bench_depth(c, &mut state, &tokenizer, 2, "2_prior_turns");
        bench_depth(c, &mut state, &tokenizer, 8, "8_prior_turns");
        bench_long_reply(c, &mut state, &tokenizer, 64);
        bench_long_reply(c, &mut state, &tokenizer, 256);
    }
}

fn bench_locked_cross_turn_prefix_cache(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    bench_cross_turn_prefix_cache(c);
}

criterion_group!(benches, bench_locked_cross_turn_prefix_cache);
criterion_main!(benches);
