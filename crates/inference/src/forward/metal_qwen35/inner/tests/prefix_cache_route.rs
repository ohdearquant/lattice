//! The prefix-cache entry's route through the shared decoder driver, against
//! the loop it replaced, which `generate_streaming_with_prefix_cache_and_cancel_legacy`
//! keeps as an oracle. A scenario is a conversation: the routed entry runs it
//! on one fresh state and the oracle on another, turn by turn, and every turn
//! must observe the same output, cache statistics, delivered `(text, id)`
//! pairs, cancellation polls, post-request slot and KV cursor. A request that
//! left a boundary is followed by one more turn through the oracle on both
//! states, so a difference in what the slot holds shows up as a difference in
//! the next turn's output.
//!
//! The tiny fixture's route plan follows the environment (`LATTICE_COMPACT_TOPK`
//! unset means the dense readback); run the module again with it set to cover the
//! compact route.
//!
//! `prefix_cache_route_matches_the_legacy_loop_on_a_real_checkpoint` runs the same
//! comparison over real tokenized chat prompts on a Qwen3.5 Q4 checkpoint. It needs
//! `LATTICE_PREFIX_ROUTE_MODEL_DIR` (and optionally `LATTICE_PREFIX_ROUTE_TOKENIZER_DIR`,
//! defaulting to the model directory). With the variable unset it prints one `SKIP`
//! line and passes; with it set, any load, tokenizer or device failure panics.

use super::prefix_cache_disposition::metal_device_present;
use super::*;
use crate::generation::GenerateOutput;
use crate::kv_cache::{CrossTurnSlotId, PrefixReuseMode};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;

const SLOT: CrossTurnSlotId = CrossTurnSlotId::DEFAULT;

#[derive(Clone, Copy, Debug)]
enum Entry {
    Route,
    Oracle,
}

/// How a request is cut short, if it is.
#[derive(Clone, Copy, Debug)]
enum Cut {
    None,
    /// The cancellation poll returns true from its n-th call on.
    CancelFromPoll(u32),
    /// `on_token` refuses from its n-th delivery on.
    RefuseFromDelivery(u32),
    /// The poll returns true once the first delta was delivered.
    CancelAfterFirstDelta,
}

#[derive(Debug, PartialEq)]
struct Output {
    token_ids: Vec<u32>,
    text: String,
    stop_reason: Option<StopReason>,
    stopped: bool,
    prompt_tokens: usize,
    generated_tokens: usize,
}

#[derive(Debug, PartialEq)]
struct Slot {
    token_ids: Vec<u32>,
    represented_len: usize,
    gdn_snapshot_len: usize,
    checkpoints: Vec<usize>,
}

#[derive(Debug, PartialEq)]
struct Observed {
    result: Result<Output, String>,
    cache: Option<(PrefixReuseMode, usize, usize, usize)>,
    delivered: Vec<(String, u32)>,
    polls: u32,
    slot: Option<Slot>,
    kv_len: usize,
    route_engaged: bool,
}

fn run_turn(
    state: &mut MetalQwen35State,
    entry: Entry,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    gen_cfg: &GenerateConfig,
    cut: Cut,
) -> Observed {
    let delivered: RefCell<Vec<(String, u32)>> = RefCell::new(Vec::new());
    let polls = Cell::new(0u32);
    let cancel_now = Cell::new(false);
    let on_token = |text: &str, id: u32| {
        let mut delivered = delivered.borrow_mut();
        delivered.push((text.to_string(), id));
        if matches!(cut, Cut::CancelAfterFirstDelta) && delivered.len() == 1 {
            cancel_now.set(true);
        }
        !matches!(cut, Cut::RefuseFromDelivery(n) if delivered.len() as u32 >= n)
    };
    let should_cancel = || {
        polls.set(polls.get() + 1);
        cancel_now.get() || matches!(cut, Cut::CancelFromPoll(n) if polls.get() >= n)
    };
    let outcome = match entry {
        Entry::Route => state.generate_streaming_with_prefix_cache_and_cancel(
            SLOT,
            prompt,
            tokenizer,
            gen_cfg,
            on_token,
            should_cancel,
        ),
        Entry::Oracle => state.generate_streaming_with_prefix_cache_and_cancel_legacy(
            SLOT,
            prompt,
            tokenizer,
            gen_cfg,
            on_token,
            should_cancel,
        ),
    };
    let (result, cache) = match outcome {
        Ok(turn) => {
            let GenerateOutput {
                text,
                token_ids,
                prompt_tokens,
                generated_tokens,
                stopped,
                stop_reason,
                token_logprobs,
            } = turn.output;
            assert!(token_logprobs.is_empty());
            (
                Ok(Output {
                    token_ids,
                    text,
                    stop_reason,
                    stopped,
                    prompt_tokens,
                    generated_tokens,
                }),
                Some((
                    turn.cache.mode,
                    turn.cache.prompt_tokens,
                    turn.cache.reused_tokens,
                    turn.cache.prefetched_tokens,
                )),
            )
        }
        Err(error) => (Err(format!("{error:?}")), None),
    };
    Observed {
        result,
        cache,
        delivered: delivered.into_inner(),
        polls: polls.get(),
        slot: state.cross_turn_prefix_cache.get(SLOT).map(|entry| Slot {
            token_ids: entry.generic.token_ids.clone(),
            represented_len: entry.generic.represented_len,
            gdn_snapshot_len: entry.generic.gdn_snapshot_len,
            checkpoints: entry.checkpoints.iter().map(|c| c.len).collect(),
        }),
        kv_len: state.session.kv_cache.seq_len,
        route_engaged: state.session.compact_topk != 0,
    }
}

struct Fixture {
    cfg: Qwen35Config,
    weights: ModelWeights,
    tokenizer: BpeTokenizer,
    /// What the grammar engine reads each id as.
    grammar_bytes: Vec<Vec<u8>>,
}

/// The single-character vocabulary: ids 0..26 are `a`..`z`, 26..32 are `A`..`F`.
fn plain_fixture() -> Fixture {
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    Fixture {
        cfg,
        weights,
        tokenizer: single_char_vocab_tokenizer(),
        grammar_bytes: single_char_vocab_bytes(),
    }
}

/// The single-character vocabulary with id 1 decoding to the lone byte 0xE4,
/// which the detokenizer holds until the final flush renders it as U+FFFD,
/// though a grammar reads it as `b`; and id 30 spelled `</think>`.
fn lone_byte_fixture() -> Fixture {
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut grammar_bytes = single_char_vocab_bytes();
    grammar_bytes[30] = b"</think>".to_vec();
    let mut vocab: HashMap<String, u32> = grammar_bytes
        .iter()
        .zip(0u32..)
        .map(|(bytes, id)| (String::from_utf8_lossy(bytes).into_owned(), id))
        .collect();
    vocab.remove("b");
    vocab.insert("\u{e4}".to_string(), 1);
    let tokenizer =
        BpeTokenizer::from_vocab_and_merges(vocab, Vec::new()).expect("lone-byte tokenizer");
    Fixture {
        cfg,
        weights,
        tokenizer,
        grammar_bytes,
    }
}

fn greedy(seed: u64, max_new_tokens: usize) -> GenerateConfig {
    cross_turn_test_gen_cfg(seed, max_new_tokens)
}

fn sampled(seed: u64, max_new_tokens: usize, repetition_penalty: f32) -> GenerateConfig {
    GenerateConfig {
        temperature: 0.9,
        top_k: 8,
        top_p: 0.95,
        repetition_penalty,
        ..cross_turn_test_gen_cfg(seed, max_new_tokens)
    }
}

/// A greedy request whose grammar fixes the generated ids.
fn forced(fixture: &Fixture, gbnf: &str, max_new_tokens: usize) -> GenerateConfig {
    use crate::grammar::{GrammarEngine, GrammarSpec};
    let engine = GrammarEngine::new(
        &GrammarSpec::Gbnf(gbnf.to_string()),
        fixture.grammar_bytes.clone(),
    )
    .expect("grammar engine builds over the fixture vocabulary");
    GenerateConfig {
        grammar: Some(std::sync::Arc::new(engine)),
        ..greedy(1, max_new_tokens)
    }
}

fn with_stop_tokens(cfg: GenerateConfig, ids: &[u32]) -> GenerateConfig {
    GenerateConfig {
        stop_token_ids: ids.to_vec(),
        ..cfg
    }
}

fn with_stop_strings(cfg: GenerateConfig, strings: &[&str]) -> GenerateConfig {
    GenerateConfig {
        stop_strings: strings.iter().map(ToString::to_string).collect(),
        ..cfg
    }
}

fn thinking(cfg: GenerateConfig, reasoning_budget: usize) -> GenerateConfig {
    GenerateConfig {
        enable_thinking: true,
        reasoning_budget: Some(reasoning_budget),
        ..cfg
    }
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    /// No warm entry.
    Fresh,
    /// A warm entry the request extends.
    ExactAppend,
    /// An earlier turn edited, at a retained checkpoint boundary.
    Replay,
    /// A warm entry the request does not extend.
    Invalidated,
}

/// The two states advance in lockstep; `turn` asserts they observed the same
/// thing and returns it.
struct Pair<'t> {
    tokenizer: &'t BpeTokenizer,
    routed: MetalQwen35State,
    oracle: MetalQwen35State,
}

impl<'t> Pair<'t> {
    fn new(fixture: &'t Fixture) -> Self {
        let state = || {
            MetalQwen35State::new(&fixture.weights, &fixture.cfg, 64).expect("tiny hybrid fixture")
        };
        Self::with_states(&fixture.tokenizer, state(), state())
    }

    fn with_states(
        tokenizer: &'t BpeTokenizer,
        routed: MetalQwen35State,
        oracle: MetalQwen35State,
    ) -> Self {
        Self {
            tokenizer,
            routed,
            oracle,
        }
    }

    fn turn(&mut self, label: &str, prompt: &str, gen_cfg: &GenerateConfig, cut: Cut) -> Observed {
        let tokenizer = self.tokenizer;
        let routed = run_turn(
            &mut self.routed,
            Entry::Route,
            tokenizer,
            prompt,
            gen_cfg,
            cut,
        );
        let oracle = run_turn(
            &mut self.oracle,
            Entry::Oracle,
            tokenizer,
            prompt,
            gen_cfg,
            cut,
        );
        assert_eq!(routed, oracle, "{label}: prompt {prompt:?}, cut {cut:?}");
        assert!(
            !routed.route_engaged,
            "{label}: the sampling route must be torn down after the request"
        );
        routed
    }

    /// One more turn through the oracle on both states, extending `prompt` by the
    /// request's own text, when the request left an entry to extend.
    fn follow_up(&mut self, label: &str, prompt: &str, observed: &Observed) {
        let (Ok(output), Some(_)) = (&observed.result, &observed.slot) else {
            return;
        };
        let prompt = format!("{prompt}{}c", output.text);
        let gen_cfg = greedy(9, 2);
        let tokenizer = self.tokenizer;
        let first = run_turn(
            &mut self.routed,
            Entry::Oracle,
            tokenizer,
            &prompt,
            &gen_cfg,
            Cut::None,
        );
        let second = run_turn(
            &mut self.oracle,
            Entry::Oracle,
            tokenizer,
            &prompt,
            &gen_cfg,
            Cut::None,
        );
        assert_eq!(first, second, "{label}: the turn after the request");
    }
}

/// Runs `scenario` with `measured` as the final turn, in lockstep.
fn play(fixture: &Fixture, scenario: Scenario, name: &str, measured: &GenerateConfig, cut: Cut) {
    let label = format!("{scenario:?}/{name}");
    let mut pair = Pair::new(fixture);
    let warm = |pair: &mut Pair<'_>, prompt: &str, gen_cfg: &GenerateConfig| {
        let warm = pair.turn("warm-up", prompt, gen_cfg, Cut::None);
        match warm.result {
            Ok(output) => output.text,
            Err(error) => panic!("{label}: a warm-up turn must not fail: {error}"),
        }
    };
    let lone_byte = fixture.grammar_bytes[30] == b"</think>";
    let warm_cfg = |seed: u64| {
        if lone_byte {
            forced(fixture, "root ::= \"a\" \"c\" \"a\"\n", 3)
        } else {
            greedy(seed, 3)
        }
    };
    let first = if lone_byte { "ac" } else { "ab" };
    let prompt = match scenario {
        Scenario::Fresh => first.to_string(),
        Scenario::ExactAppend => {
            let text = warm(&mut pair, first, &warm_cfg(7));
            format!("{first}{text}q")
        }
        Scenario::Invalidated => {
            warm(&mut pair, first, &warm_cfg(7));
            "xyz".to_string()
        }
        Scenario::Replay => {
            assert!(!lone_byte, "the replay scenario runs on the plain fixture");
            let t1 = warm(&mut pair, "abc", &greedy(42, 3));
            let history = format!("abc{t1}");
            let t2_prompt = format!("{history}de");
            let t2 = warm(&mut pair, &t2_prompt, &greedy(42, 3));
            let t3_prompt = format!("{t2_prompt}{t2}gh");
            warm(&mut pair, &t3_prompt, &greedy(42, 3));
            format!("{history}wxyz")
        }
    };
    let observed = pair.turn(&label, &prompt, measured, cut);
    pair.follow_up(&label, &prompt, &observed);
}

const CUTS: [Cut; 9] = [
    Cut::None,
    Cut::CancelFromPoll(1),
    Cut::CancelFromPoll(2),
    Cut::CancelFromPoll(3),
    Cut::CancelFromPoll(4),
    Cut::RefuseFromDelivery(1),
    Cut::RefuseFromDelivery(2),
    Cut::RefuseFromDelivery(3),
    Cut::CancelAfterFirstDelta,
];

#[test]
fn prefix_cache_entry_runs_through_driver() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let fixture = plain_fixture();
    let mut state =
        MetalQwen35State::new(&fixture.weights, &fixture.cfg, 64).expect("tiny hybrid fixture");

    // A length stop: every generated token was a prediction opened by the
    // driver, and every one but the last was consumed by a decode step.
    let (turn, trace) = state
        .generate_streaming_with_prefix_cache_with_trace(
            SLOT,
            "ab",
            &fixture.tokenizer,
            &greedy(7, 3),
            |_, _| true,
            || false,
        )
        .expect("a fresh request");
    assert_eq!(turn.output.generated_tokens, 3);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Length));
    assert_eq!(trace.opened, 3, "one select per generated token");
    assert_eq!(trace.consumed + 1, trace.opened, "the driver's invariant");

    // A stop token ends the request on an opened prediction that is never
    // pushed: the exception the driver documents.
    let stop = with_stop_tokens(forced(&fixture, "root ::= \"a\" \"b\" \"c\"\n", 5), &[2]);
    let (turn, trace) = state
        .generate_streaming_with_prefix_cache_with_trace(
            SLOT,
            "a",
            &fixture.tokenizer,
            &stop,
            |_, _| true,
            || false,
        )
        .expect("a stop token inside the loop");
    assert_eq!(turn.output.token_ids, vec![0, 1]);
    assert_eq!(trace.opened, turn.output.generated_tokens + 1);
    assert_eq!(trace.consumed + 1, trace.opened);

    // A warm entry extended by the next turn runs through the driver too, and
    // the worker-visible statistics report the reuse.
    let (first, _) = state
        .generate_streaming_with_prefix_cache_with_trace(
            SLOT,
            "ab",
            &fixture.tokenizer,
            &greedy(7, 3),
            |_, _| true,
            || false,
        )
        .expect("a warm-up turn");
    let prompt = format!("ab{}q", first.output.text);
    let (second, trace) = state
        .generate_streaming_with_prefix_cache_with_trace(
            SLOT,
            &prompt,
            &fixture.tokenizer,
            &greedy(8, 2),
            |_, _| true,
            || false,
        )
        .expect("an exact-append turn");
    assert_eq!(second.cache.mode, PrefixReuseMode::ExactAppend);
    assert_eq!(second.cache.prefetched_tokens, 1);
    assert_eq!(trace.opened, 2);

    // A request that returns before the driver runs leaves the trace at zero.
    let (_, trace) = state
        .generate_streaming_with_prefix_cache_with_trace(
            SLOT,
            "ab",
            &fixture.tokenizer,
            &greedy(7, 0),
            |_, _| true,
            || false,
        )
        .expect("a zero-budget request");
    assert_eq!((trace.opened, trace.consumed), (0, 0));
}

#[test]
fn prefix_cache_route_matches_the_legacy_loop_on_the_plain_vocabulary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let fixture = plain_fixture();
    let abc = "root ::= \"a\" \"b\" \"c\"\n";
    let configs: Vec<(&str, GenerateConfig)> = vec![
        ("greedy-length-3", greedy(7, 3)),
        ("greedy-length-1", greedy(7, 1)),
        ("sampled", sampled(5, 4, 1.0)),
        ("sampled-penalty", sampled(5, 4, 1.2)),
        (
            "grammar-complete-first",
            forced(&fixture, "root ::= \"a\"\n", 4),
        ),
        (
            "grammar-complete-loop",
            forced(&fixture, "root ::= \"a\" \"a\"\n", 4),
        ),
        (
            "grammar-blocked-step-0",
            forced(&fixture, "root ::= \"1\"\n", 4),
        ),
        (
            "stop-token-first",
            with_stop_tokens(forced(&fixture, "root ::= \"a\" \"a\"\n", 4), &[0]),
        ),
        (
            "stop-token-loop",
            with_stop_tokens(forced(&fixture, abc, 5), &[2]),
        ),
        (
            "stop-string-first",
            with_stop_strings(forced(&fixture, abc, 5), &["a"]),
        ),
        (
            "stop-string-loop",
            with_stop_strings(forced(&fixture, abc, 5), &["b"]),
        ),
        (
            "stop-string-span",
            with_stop_strings(forced(&fixture, abc, 5), &["bc"]),
        ),
        (
            "stop-string-held-never-matched",
            with_stop_strings(
                forced(&fixture, "root ::= \"a\" \"b\" \"c\" \"d\" \"e\"\n", 3),
                &["cde"],
            ),
        ),
    ];
    let cut_configs: Vec<(&str, GenerateConfig)> = vec![
        ("greedy-length-4", greedy(7, 4)),
        ("sampled", sampled(5, 4, 1.0)),
        (
            "stop-string-held-never-matched",
            with_stop_strings(
                forced(&fixture, "root ::= \"a\" \"b\" \"c\" \"d\" \"e\"\n", 4),
                &["cde"],
            ),
        ),
    ];
    for scenario in [
        Scenario::Fresh,
        Scenario::ExactAppend,
        Scenario::Replay,
        Scenario::Invalidated,
    ] {
        for (name, gen_cfg) in &configs {
            play(&fixture, scenario, name, gen_cfg, Cut::None);
        }
        for (name, gen_cfg) in &cut_configs {
            for cut in CUTS {
                play(&fixture, scenario, name, gen_cfg, cut);
            }
        }
    }
}

#[test]
fn prefix_cache_route_matches_the_legacy_loop_on_a_flushed_lone_byte() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let fixture = lone_byte_fixture();
    let flush = &["\u{fffd}"];
    let answer_budget = format!("root ::= \"</think>\"{}\n", " \"a\"".repeat(6));
    let configs: Vec<(&str, GenerateConfig)> = vec![
        (
            "lone-byte-length",
            forced(&fixture, "root ::= \"a\" \"b\" \"b\" \"b\"\n", 2),
        ),
        (
            "lone-byte-length-then-stop-string",
            with_stop_strings(
                forced(&fixture, "root ::= \"a\" \"b\" \"b\" \"b\"\n", 2),
                flush,
            ),
        ),
        (
            "lone-byte-stop-token",
            with_stop_tokens(forced(&fixture, "root ::= \"a\" \"b\" \"c\"\n", 5), &[2]),
        ),
        (
            "lone-byte-stop-token-then-stop-string",
            with_stop_strings(
                with_stop_tokens(forced(&fixture, "root ::= \"a\" \"b\" \"c\"\n", 5), &[2]),
                flush,
            ),
        ),
        (
            "lone-byte-grammar-complete-first",
            forced(&fixture, "root ::= \"b\"\n", 4),
        ),
        (
            "budget-rejection",
            thinking(forced(&fixture, "root ::= \"a\" \"a\"\n", 4), 1),
        ),
        (
            "budget-rejection-with-held-byte",
            thinking(forced(&fixture, "root ::= \"b\" \"b\"\n", 4), 1),
        ),
        (
            "budget-rejection-with-held-byte-then-stop-string",
            with_stop_strings(
                thinking(forced(&fixture, "root ::= \"b\" \"b\"\n", 4), 1),
                flush,
            ),
        ),
        (
            "answer-budget",
            thinking(forced(&fixture, &answer_budget, 2), 3),
        ),
    ];
    let cut_configs = [
        ("lone-byte-length", &configs[0].1),
        ("lone-byte-length-then-stop-string", &configs[1].1),
    ];
    for scenario in [Scenario::Fresh, Scenario::ExactAppend] {
        for (name, gen_cfg) in &configs {
            play(&fixture, scenario, name, gen_cfg, Cut::None);
        }
        for (name, gen_cfg) in cut_configs {
            for cut in CUTS {
                play(&fixture, scenario, name, gen_cfg, cut);
            }
        }
    }
}

/// One stderr line per real-checkpoint case, so a reader of the log can see that
/// real tokens flowed.
fn report_real_case(case: &str, observed: &Observed) {
    let (generated, stop) = match &observed.result {
        Ok(output) => (output.generated_tokens, format!("{:?}", output.stop_reason)),
        Err(error) => (0, format!("error {error}")),
    };
    eprintln!(
        "prefix_cache_route real case={case} generated={generated} stop={stop} \
         cache={:?} deliveries={} slot_len={:?} kv_len={}",
        observed.cache,
        observed.delivered.len(),
        observed.slot.as_ref().map(|slot| slot.represented_len),
        observed.kv_len,
    );
}

/// The text a successful turn produced; a failed turn is a failed run.
fn real_turn_text(case: &str, observed: &Observed) -> String {
    match &observed.result {
        Ok(output) if output.generated_tokens > 0 => output.text.clone(),
        Ok(_) => panic!("{case}: the request generated no tokens, so it proves nothing"),
        Err(error) => panic!("{case}: the request failed: {error}"),
    }
}

/// Route and legacy oracle in lockstep over real tokenized chat prompts: a fresh
/// turn that stops on length, an exact-append turn under greedy decoding, an
/// exact-append turn under seeded sampling, a replay from a retained checkpoint,
/// and a stop-string turn. Each turn is compared in full by `Pair::turn`; the mode
/// and stop assertions below only check that a turn reached the path its case
/// names, so a prompt that tokenizes differently from the generated ids fails
/// loudly instead of silently testing another path.
#[test]
fn prefix_cache_route_matches_the_legacy_loop_on_a_real_checkpoint() {
    let Some(model_dir) = std::env::var_os("LATTICE_PREFIX_ROUTE_MODEL_DIR").map(PathBuf::from)
    else {
        eprintln!("SKIP prefix_cache_route real checkpoint: LATTICE_PREFIX_ROUTE_MODEL_DIR unset");
        return;
    };
    let tokenizer_dir = std::env::var_os("LATTICE_PREFIX_ROUTE_TOKENIZER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.clone());
    let _gpu_guard = gpu_test_lock();

    let tokenizer_path = tokenizer_dir.join("tokenizer.json");
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)
        .unwrap_or_else(|error| panic!("tokenizer {}: {error}", tokenizer_path.display()));
    let im_end = tokenizer
        .special_token_id("<|im_end|>")
        .expect("the checkpoint's tokenizer has no <|im_end|> token");
    let cfg = Qwen35Config::from_model_dir(&model_dir)
        .unwrap_or_else(|error| panic!("config.json in {}: {error}", model_dir.display()));
    let load = || {
        MetalQwen35State::from_q4_dir(&model_dir, &tokenizer_path, &cfg, 1024)
            .unwrap_or_else(|error| panic!("from_q4_dir {}: {error}", model_dir.display()))
    };
    let mut pair = Pair::with_states(&tokenizer, load(), load());

    let chat =
        |messages: &[ChatMessage]| crate::forward::metal_qwen35::format_chat_template(messages);
    let q1 = "Write a short poem about the sea.";
    let q2 = "Now one about the sky.";
    let q_edit = "Now one about the mountains.";
    let q3 = "Now one about the desert.";
    let stop = |cfg: GenerateConfig| with_stop_tokens(cfg, &[im_end]);

    // Fresh, greedy, stopped by length.
    let prompt1 = chat(&[ChatMessage::user(q1)]);
    let turn1 = pair.turn(
        "fresh/greedy/length",
        &prompt1,
        &stop(greedy(7, 6)),
        Cut::None,
    );
    report_real_case("fresh/greedy/length", &turn1);
    let text1 = real_turn_text("fresh/greedy/length", &turn1);
    assert!(
        matches!(turn1.cache, Some((PrefixReuseMode::FullRefill, ..))),
        "fresh/greedy/length: precondition: the first turn must be a full refill"
    );
    assert!(
        matches!(&turn1.result, Ok(output) if output.stop_reason == Some(StopReason::Length)),
        "fresh/greedy/length: precondition: a six-token budget must end on length"
    );

    // Exact append onto the boundary the first turn saved, greedy.
    let history2 = [
        ChatMessage::user(q1),
        ChatMessage::assistant(text1.clone()),
        ChatMessage::user(q2),
    ];
    let turn2 = pair.turn(
        "exact-append/greedy",
        &chat(&history2),
        &stop(greedy(8, 10)),
        Cut::None,
    );
    report_real_case("exact-append/greedy", &turn2);
    let text2 = real_turn_text("exact-append/greedy", &turn2);
    assert!(
        matches!(turn2.cache, Some((PrefixReuseMode::ExactAppend, ..))),
        "exact-append/greedy: precondition: the prompt must extend the saved boundary token for \
         token; got {:?}",
        turn2.cache
    );

    // Exact append under seeded sampling.
    let history3 = [
        ChatMessage::user(q1),
        ChatMessage::assistant(text1.clone()),
        ChatMessage::user(q2),
        ChatMessage::assistant(text2),
        ChatMessage::user(q3),
    ];
    let turn3 = pair.turn(
        "exact-append/seeded",
        &chat(&history3),
        &stop(sampled(5, 12, 1.0)),
        Cut::None,
    );
    report_real_case("exact-append/seeded", &turn3);
    real_turn_text("exact-append/seeded", &turn3);
    assert!(
        matches!(turn3.cache, Some((PrefixReuseMode::ExactAppend, ..))),
        "exact-append/seeded: precondition: the prompt must extend the saved boundary token for \
         token; got {:?}",
        turn3.cache
    );

    // The second user message edited: the history diverges after the first turn's
    // boundary, which the save after the second turn retained as a checkpoint.
    let history4 = [
        ChatMessage::user(q1),
        ChatMessage::assistant(text1.clone()),
        ChatMessage::user(q_edit),
    ];
    let turn4 = pair.turn(
        "replay-from-checkpoint/greedy",
        &chat(&history4),
        &stop(greedy(8, 10)),
        Cut::None,
    );
    report_real_case("replay-from-checkpoint/greedy", &turn4);
    let text4 = real_turn_text("replay-from-checkpoint/greedy", &turn4);
    assert!(
        matches!(
            turn4.cache,
            Some((PrefixReuseMode::ReplayFromCheckpoint { .. }, ..))
        ),
        "replay-from-checkpoint/greedy: precondition: the edited history must replay from a \
         retained checkpoint; got {:?}",
        turn4.cache
    );

    // A stop string a few tokens in: nothing behind the match may be saved.
    let history5 = [
        ChatMessage::user(q1),
        ChatMessage::assistant(text1),
        ChatMessage::user(q_edit),
        ChatMessage::assistant(text4),
        ChatMessage::user(q3),
    ];
    let turn5 = pair.turn(
        "stop-string",
        &chat(&history5),
        &with_stop_strings(stop(greedy(9, 24)), &[" "]),
        Cut::None,
    );
    report_real_case("stop-string", &turn5);
    real_turn_text("stop-string", &turn5);
    assert!(
        matches!(&turn5.result, Ok(output) if output.stopped) && turn5.slot.is_none(),
        "stop-string: precondition: the stop string must have fired and left the slot empty"
    );
}
