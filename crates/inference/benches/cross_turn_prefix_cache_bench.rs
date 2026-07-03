//! Cross-turn KV prefix cache benchmark (#462).
//!
//! Compares `chat_metal`'s pre-#462 behavior (`reset_state()` + a full
//! re-prefill of the entire ChatML-formatted conversation on every turn)
//! against the cache-aware path shipped in #462
//! (`chat_completion_streaming_with_prefix_cache`, which reuses the previous
//! turn's shared KV/GDN state and prefills only the new suffix), at a couple
//! of conversation depths.
//!
//! Requires a real Qwen3.5 model at `~/.lattice/models/qwen3.5-0.8b/`
//! (config.json + safetensors + tokenizer.json).
//!
//! Run: `cargo bench -p lattice-inference --features metal-gpu,f16 -- cross_turn_prefix_cache`

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::path::PathBuf;
use std::time::Duration;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::kv_cache::CrossTurnSlotId;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::model::qwen35_config::GenerateConfig;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_inference::tokenizer::bpe::BpeTokenizer;

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
fn load_state_and_tokenizer() -> Option<(MetalQwen35State, BpeTokenizer)> {
    use lattice_inference::model::qwen35::Qwen35Model;
    let dir = safetensors_model_dir()?;
    let model = Qwen35Model::from_safetensors(&dir).ok()?;
    let cfg = model.config().clone();
    let state = MetalQwen35State::new(model.weights(), &cfg, 4096).ok()?;
    let tokenizer = BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json")).ok()?;
    Some((state, tokenizer))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn bench_gen_cfg(max_new_tokens: usize) -> GenerateConfig {
    GenerateConfig {
        max_new_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(7),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
    }
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
///   is just the one incremental final-turn call, which should land on
///   `ExactAppend` and prefill only the new suffix instead of the whole
///   history.
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
    group.bench_function(BenchmarkId::new("cache_aware_incremental", label), |b| {
        b.iter_batched(
            || {
                let mut state = state_cell.borrow_mut();
                state.reset_state();
                state.clear_cross_turn_prefix_cache();
                let mut history = vec![ChatMessage::system(SYSTEM_PROMPT)];
                for user_text in &USER_TURNS[..prior_turns] {
                    history.push(ChatMessage::user(*user_text));
                    let out = state
                        .chat_completion_streaming_with_prefix_cache(
                            slot_id,
                            &history,
                            tokenizer,
                            &gen_cfg,
                            |_delta, _id| true,
                        )
                        .expect("cache warm-up turn must succeed");
                    history.push(out.output.message);
                }
                history.push(ChatMessage::user(next_user));
                history
            },
            |history| {
                let mut state = state_cell.borrow_mut();
                state.chat_completion_streaming_with_prefix_cache(
                    slot_id,
                    &history,
                    tokenizer,
                    &gen_cfg,
                    |_delta, _id| true,
                )
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_cross_turn_prefix_cache(c: &mut Criterion) {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("SKIP: cross_turn_prefix_cache bench requires macOS + metal-gpu feature");
        let _ = c;
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        let Some((mut state, tokenizer)) = load_state_and_tokenizer() else {
            eprintln!(
                "SKIP: cross_turn_prefix_cache bench requires a real model at \
                 ~/.lattice/models/qwen3.5-0.8b (config.json + safetensors + tokenizer.json)"
            );
            return;
        };

        bench_depth(c, &mut state, &tokenizer, 2, "2_prior_turns");
        bench_depth(c, &mut state, &tokenizer, 8, "8_prior_turns");
    }
}

criterion_group!(benches, bench_cross_turn_prefix_cache);
criterion_main!(benches);
