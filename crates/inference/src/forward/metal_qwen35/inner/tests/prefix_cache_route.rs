//! What the prefix-cache entry leaves behind after each request it routes
//! through the shared decoder driver, over a sweep of scenario x config x cut.
//! A scenario is a conversation on one fresh state; every request in it must
//! tear the sampling route down afterwards, and when it leaves a boundary in the
//! slot that boundary must represent exactly the live KV cursor and be a prefix
//! of the request's prompt ids followed by the ids it generated. The slot's
//! recurrent snapshot and the live key/value rows it represents must also equal
//! those of a fresh state prefilled with the slot's ids, within a tolerance set
//! from the clean run, and that reference must be nonzero: the model the sweeps
//! run on has nonzero weights everywhere, so a corrupted state shows up in the
//! comparison instead of hiding behind zeros. A request that left a boundary is
//! then followed by one more turn, run on the warm state and on a fresh one, and
//! the two must generate the same ids, text and stop reason: reusing the slot
//! must not change what the next turn produces.
//!
//! The tiny fixture's route plan follows the environment (`LATTICE_COMPACT_TOPK`
//! unset means the dense readback); run the module again with it set to cover the
//! compact route.
//!
//! `prefix_cache_route_slot_follows_the_request_on_a_real_checkpoint` runs the
//! first two checks over real tokenized chat prompts on a Qwen3.5 Q4 checkpoint,
//! and requires every turn after the first to succeed. It does not compare a warm
//! turn with a cold one: a real 4-bit checkpoint's cached and cold numerics need
//! not agree token for token. It needs `LATTICE_PREFIX_ROUTE_MODEL_DIR` (and
//! optionally `LATTICE_PREFIX_ROUTE_TOKENIZER_DIR`, defaulting to the model
//! directory). With the variable unset it prints one `SKIP` line and passes; with
//! it set, any load, tokenizer or device failure panics.

use super::prefix_cache_disposition::{ids_of, lone_byte_vocabulary, metal_device_present};
use super::*;
use crate::generation::GenerateOutput;
use crate::kv_cache::{CrossTurnSlotId, PrefixReuseMode};
use std::cell::{Cell, RefCell};
use std::path::PathBuf;

const SLOT: CrossTurnSlotId = CrossTurnSlotId::DEFAULT;

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

#[derive(Debug)]
struct Output {
    token_ids: Vec<u32>,
    text: String,
    stop_reason: Option<StopReason>,
    stopped: bool,
    generated_tokens: usize,
}

#[derive(Debug)]
struct Slot {
    token_ids: Vec<u32>,
    represented_len: usize,
}

#[derive(Debug)]
struct Observed {
    result: Result<Output, String>,
    cache: Option<(PrefixReuseMode, usize, usize, usize)>,
    delivered: Vec<(String, u32)>,
    slot: Option<Slot>,
    kv_len: usize,
    route_engaged: bool,
}

fn run_turn(
    state: &mut MetalQwen35State,
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
    let outcome = state.generate_streaming_with_prefix_cache_and_cancel(
        SLOT,
        prompt,
        tokenizer,
        gen_cfg,
        on_token,
        should_cancel,
    );
    let (result, cache) = match outcome {
        Ok(turn) => {
            let GenerateOutput {
                text,
                token_ids,
                prompt_tokens: _,
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
        slot: state.cross_turn_prefix_cache.get(SLOT).map(|entry| Slot {
            token_ids: entry.generic.token_ids.clone(),
            represented_len: entry.generic.represented_len,
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

/// Deterministic, roughly uniform values with the given root mean square and no
/// structure shared between seeds or positions.
fn scrambled(len: usize, seed: u64, rms: f32) -> Vec<f32> {
    (0..len as u64)
        .map(|i| {
            let mut x = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(i.wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            let unit = (x >> 40) as f32 / (1u64 << 24) as f32;
            (unit * 2.0 - 1.0) * 3f32.sqrt() * rms
        })
        .collect()
}

/// A projection whose outputs have about `gain` times the root mean square of an
/// input of `fan_in` unit-scale values.
fn projection(len: usize, seed: u64, fan_in: usize, gain: f32) -> Vec<f32> {
    scrambled(len, seed, gain / (fan_in as f32).sqrt())
}

/// The tiny hybrid model's shape (vocabulary 32, head dimension 256, three
/// recurrent layers and one full-attention layer) with weights that are nonzero
/// everywhere, so the recurrent state and the key/value rows a request leaves
/// behind are nonzero and reach the logits. The shape comes from
/// `tiny_hybrid_fixture` unchanged; only the weights are replaced.
fn patterned_model() -> (Qwen35Config, ModelWeights) {
    let (mut cfg, mut weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let hidden = cfg.hidden_size;
    let mut seed = 100u64;
    let mut next_seed = || {
        seed += 1;
        seed
    };
    weights.embed_tokens = scrambled(weights.embed_tokens.len(), next_seed(), 1.0);
    for (attention, common) in &mut weights.layers {
        match attention {
            AttentionWeights::Linear(gdn) => {
                gdn.in_proj_qkv = projection(gdn.in_proj_qkv.len(), next_seed(), hidden, 1.0);
                gdn.in_proj_z = projection(gdn.in_proj_z.len(), next_seed(), hidden, 1.0);
                gdn.in_proj_b = projection(gdn.in_proj_b.len(), next_seed(), hidden, 0.5);
                gdn.in_proj_a = projection(gdn.in_proj_a.len(), next_seed(), hidden, 0.5);
                // A slow decay, so the recurrent state carries the whole history.
                gdn.a_log = vec![-2.5; gdn.a_log.len()];
                gdn.conv1d_weight =
                    projection(gdn.conv1d_weight.len(), next_seed(), gdn.kernel_size, 1.0);
                gdn.out_proj = projection(gdn.out_proj.len(), next_seed(), gdn.out_proj_cols, 0.7);
            }
            AttentionWeights::Full(full) => {
                let q_dim = cfg.full_q_dim();
                full.q_proj = projection(full.q_proj.len(), next_seed(), hidden, 1.0);
                full.k_proj = projection(full.k_proj.len(), next_seed(), hidden, 1.0);
                full.v_proj = projection(full.v_proj.len(), next_seed(), hidden, 1.0);
                full.o_proj = projection(full.o_proj.len(), next_seed(), q_dim, 0.7);
            }
        }
        if let FeedForwardWeights::Dense(ffn) = &mut common.ffn {
            let intermediate = cfg.intermediate_size;
            ffn.gate_proj = projection(ffn.gate_proj.len(), next_seed(), hidden, 1.0);
            ffn.up_proj = projection(ffn.up_proj.len(), next_seed(), hidden, 1.0);
            ffn.down_proj = projection(ffn.down_proj.len(), next_seed(), intermediate, 0.5);
        }
    }
    (cfg, weights)
}

/// The single-character vocabulary: ids 0..26 are `a`..`z`, 26..32 are `A`..`F`.
fn plain_fixture() -> Fixture {
    let (cfg, weights) = patterned_model();
    Fixture {
        cfg,
        weights,
        tokenizer: single_char_vocab_tokenizer(),
        grammar_bytes: single_char_vocab_bytes(),
    }
}

/// The lone-byte vocabulary (see `lone_byte_vocabulary`).
fn lone_byte_fixture() -> Fixture {
    let (cfg, weights) = patterned_model();
    let (tokenizer, grammar_bytes) = lone_byte_vocabulary();
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

/// The ids a tokenizer reads a prompt as, for a checkpoint whose vocabulary the
/// test cannot write out by hand.
fn tokenized_prompt_ids(tokenizer: &BpeTokenizer, prompt: &str) -> Vec<u32> {
    let input = tokenizer.tokenize(prompt);
    input.input_ids[..input.real_length].to_vec()
}

fn fixture_prompt_ids(_: &BpeTokenizer, prompt: &str) -> Vec<u32> {
    ids_of(prompt)
}

fn fresh_state(fixture: &Fixture) -> MetalQwen35State {
    MetalQwen35State::new(&fixture.weights, &fixture.cfg, 64).expect("tiny hybrid fixture")
}

/// What a request generated, without the cache statistics that differ by design
/// between a warm and a cold run.
fn generated(observed: &Observed) -> Result<(&[u32], &str, Option<StopReason>), &str> {
    match &observed.result {
        Ok(output) => Ok((&output.token_ids, &output.text, output.stop_reason)),
        Err(error) => Err(error),
    }
}

/// A saved slot represents exactly the live KV cursor, and its token ids are a
/// prefix of the prompt ids followed by the ids the request generated. An empty
/// slot asserts nothing.
fn assert_slot_follows_the_request(label: &str, observed: &Observed, prompt_ids: &[u32]) {
    let Some(slot) = &observed.slot else {
        return;
    };
    assert_eq!(
        slot.represented_len, observed.kv_len,
        "{label}: a saved slot must represent exactly the live KV cursor"
    );
    assert_eq!(
        slot.token_ids.len(),
        slot.represented_len,
        "{label}: a saved slot's token ids must span its represented length"
    );
    let mut request_ids = prompt_ids.to_vec();
    if let Ok(output) = &observed.result {
        request_ids.extend_from_slice(&output.token_ids);
    }
    assert!(
        request_ids.starts_with(&slot.token_ids),
        "{label}: a saved slot must be a prefix of the prompt ids followed by the generated \
         ids; slot {:?}, prompt and generated {:?}",
        slot.token_ids,
        request_ids
    );
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "compared state sizes must match");
    left.iter()
        .zip(right)
        .fold(0.0f32, |acc, (l, r)| acc.max((l - r).abs()))
}

/// Every recurrent matrix and convolution buffer of a snapshot, in layer order.
fn gdn_values(snapshot: &crate::attention::gdn::GdnSnapshot) -> Vec<f32> {
    snapshot
        .iter()
        .flat_map(|(matrices, conv)| matrices.iter().chain(conv))
        .copied()
        .collect()
}

/// The first `rows` key and value rows of every full-attention layer, as f32.
fn live_kv_rows(state: &MetalQwen35State, rows: usize) -> Vec<f32> {
    let cache = &state.session.kv_cache;
    let elems = rows * cache.kv_dim;
    let mut values = Vec::new();
    for buffer in cache.k_bufs.iter().chain(&cache.v_bufs) {
        // SAFETY: the buffers are StorageModeShared and hold max_cache_len rows of
        // kv_dim elements in the state's KV precision; `rows` is at most the live
        // cursor, so every element read was written, and no command buffer is in
        // flight once a request has returned.
        unsafe {
            if state.use_kv_f16 {
                let bits = std::slice::from_raw_parts(buffer.contents() as *const u16, elems);
                values.extend(bits.iter().map(|&b| f16_to_f32(b)));
            } else {
                let floats = std::slice::from_raw_parts(buffer.contents() as *const f32, elems);
                values.extend_from_slice(floats);
            }
        }
    }
    values
}

/// Tolerances and floors of the saved-state check. A tolerance is relative to the
/// largest magnitude in the reference state. The reference is one batched prefill
/// and the saved state was built by a prefill and single-token steps, whose half
/// precision activations round differently, so a clean run is not exact: over the
/// whole clean suite (359 saved slots) the largest ratio `max_abs(diff) /
/// max_abs(reference)` was 8.27e-3 for the recurrent snapshot and 4.85e-3 for the
/// key/value rows, and each tolerance is at least ten times its clean maximum. A
/// floor is about a tenth of the smallest clean reference magnitude (4.87 for the
/// snapshot, 6.74 for the rows), far above the exact zero of a model whose weights
/// are zero.
const GDN_TOL: f32 = 0.09;
const KV_TOL: f32 = 0.05;
const GDN_FLOOR: f32 = 0.5;
const KV_FLOOR: f32 = 0.5;

/// A saved slot's recurrent snapshot, and the live key/value rows it represents,
/// equal those of a fresh state of the same model prefilled with the slot's token
/// ids, within a relative tolerance. The reference state must be nonzero, so a
/// model whose state is zero fails the check instead of passing it.
fn assert_saved_state_follows_a_prefill(label: &str, state: &MetalQwen35State, fixture: &Fixture) {
    use crate::speculative::MtpTargetVerifier as _;

    let Some(entry) = state.cross_turn_prefix_cache.get(SLOT) else {
        return;
    };
    let ids = entry.generic.token_ids.clone();
    assert_eq!(
        entry.generic.gdn_snapshot_len,
        ids.len(),
        "{label}: a saved slot's recurrent snapshot must sit at its represented length"
    );
    let mut reference = fresh_state(fixture);
    reference.use_gdn_chunked = state.use_gdn_chunked;
    assert_eq!(reference.use_kv_f16, state.use_kv_f16);
    reference
        .try_forward_prefill(&ids)
        .expect("the reference prefill of a saved slot's ids must succeed");

    let saved_gdn = gdn_values(&entry.gdn_snapshot);
    let reference_gdn = gdn_values(&reference.snapshot_gdn_states());
    let gdn_scale = max_abs(&reference_gdn);
    let gdn_diff = max_abs_diff(&saved_gdn, &reference_gdn);
    let saved_kv = live_kv_rows(state, ids.len());
    let reference_kv = live_kv_rows(&reference, ids.len());
    let kv_scale = max_abs(&reference_kv);
    let kv_diff = max_abs_diff(&saved_kv, &reference_kv);
    assert!(
        gdn_scale > GDN_FLOOR,
        "{label}: the reference recurrent state must be nonzero (max {gdn_scale:e}, floor \
         {GDN_FLOOR:e}); a model with zero state cannot show a corrupted one"
    );
    assert!(
        kv_scale > KV_FLOOR,
        "{label}: the reference key/value rows must be nonzero (max {kv_scale:e}, floor \
         {KV_FLOOR:e}); a model with zero rows cannot show corrupted ones"
    );
    assert!(
        gdn_diff <= GDN_TOL * gdn_scale,
        "{label}: a saved slot's recurrent snapshot must equal a fresh prefill of its ids \
         (max difference {gdn_diff:e}, reference max {gdn_scale:e}, tolerance {GDN_TOL:e})"
    );
    assert!(
        kv_diff <= KV_TOL * kv_scale,
        "{label}: the live key/value rows of a saved slot must equal a fresh prefill of its \
         ids (max difference {kv_diff:e}, reference max {kv_scale:e}, tolerance {KV_TOL:e})"
    );
}

/// One conversation on one state, every request checked as it runs.
struct Conversation<'t> {
    tokenizer: &'t BpeTokenizer,
    prompt_ids: fn(&BpeTokenizer, &str) -> Vec<u32>,
    /// The model the state was built from, when it is small enough to prefill
    /// again for the saved-state check.
    fixture: Option<&'t Fixture>,
    state: MetalQwen35State,
}

impl<'t> Conversation<'t> {
    fn new(fixture: &'t Fixture) -> Self {
        Self {
            tokenizer: &fixture.tokenizer,
            prompt_ids: fixture_prompt_ids,
            fixture: Some(fixture),
            state: fresh_state(fixture),
        }
    }

    fn with_state(
        tokenizer: &'t BpeTokenizer,
        prompt_ids: fn(&BpeTokenizer, &str) -> Vec<u32>,
        state: MetalQwen35State,
    ) -> Self {
        Self {
            tokenizer,
            prompt_ids,
            fixture: None,
            state,
        }
    }

    /// Runs one request and asserts that the sampling route was torn down after
    /// it, that the slot it left follows the request, and, when the model is small
    /// enough to prefill again, that the saved state matches a fresh prefill.
    fn turn(&mut self, label: &str, prompt: &str, gen_cfg: &GenerateConfig, cut: Cut) -> Observed {
        let observed = run_turn(&mut self.state, self.tokenizer, prompt, gen_cfg, cut);
        assert!(
            !observed.route_engaged,
            "{label}: the sampling route must be torn down after the request"
        );
        let prompt_ids = (self.prompt_ids)(self.tokenizer, prompt);
        assert_slot_follows_the_request(label, &observed, &prompt_ids);
        if let Some(fixture) = self.fixture {
            assert_saved_state_follows_a_prefill(label, &self.state, fixture);
        }
        observed
    }

    /// When the request left a boundary, one more turn extending `prompt` by the
    /// request's own text runs on this warm state and on a fresh one, and the
    /// two must generate the same ids, text and stop reason.
    fn assert_follow_up_matches_a_cold_run(
        &mut self,
        fixture: &Fixture,
        label: &str,
        prompt: &str,
        observed: &Observed,
    ) {
        let (Ok(output), Some(_)) = (&observed.result, &observed.slot) else {
            return;
        };
        let prompt = format!("{prompt}{}c", output.text);
        let gen_cfg = greedy(9, 2);
        let warm = run_turn(
            &mut self.state,
            self.tokenizer,
            &prompt,
            &gen_cfg,
            Cut::None,
        );
        assert!(
            !warm.route_engaged,
            "{label}: the sampling route must be torn down after the follow-up"
        );
        let cold = run_turn(
            &mut fresh_state(fixture),
            self.tokenizer,
            &prompt,
            &gen_cfg,
            Cut::None,
        );
        assert_eq!(
            generated(&warm),
            generated(&cold),
            "{label}: the follow-up {prompt:?} must generate what a fresh state generates"
        );
    }
}

/// Plays the warm-up turns of `scenario` through `turn` and returns the prompt
/// of the measured turn.
fn scenario_prompt(
    fixture: &Fixture,
    scenario: Scenario,
    label: &str,
    turn: &mut dyn FnMut(&str, &GenerateConfig) -> Observed,
) -> String {
    let mut warm = |prompt: &str, gen_cfg: &GenerateConfig| match turn(prompt, gen_cfg).result {
        Ok(output) => output.text,
        Err(error) => panic!("{label}: a warm-up turn must not fail: {error}"),
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
    match scenario {
        Scenario::Fresh => first.to_string(),
        Scenario::ExactAppend => {
            let text = warm(first, &warm_cfg(7));
            format!("{first}{text}q")
        }
        Scenario::Invalidated => {
            warm(first, &warm_cfg(7));
            "xyz".to_string()
        }
        Scenario::Replay => {
            assert!(!lone_byte, "the replay scenario runs on the plain fixture");
            let t1 = warm("abc", &greedy(42, 3));
            let history = format!("abc{t1}");
            let t2_prompt = format!("{history}de");
            let t2 = warm(&t2_prompt, &greedy(42, 3));
            let t3_prompt = format!("{t2_prompt}{t2}gh");
            warm(&t3_prompt, &greedy(42, 3));
            format!("{history}wxyz")
        }
    }
}

/// Runs `scenario` with `measured` as the final turn.
fn play(fixture: &Fixture, scenario: Scenario, name: &str, measured: &GenerateConfig, cut: Cut) {
    let label = format!("{scenario:?}/{name}");
    let mut conversation = Conversation::new(fixture);
    let prompt = scenario_prompt(fixture, scenario, &label, &mut |prompt, gen_cfg| {
        conversation.turn(&format!("{label}: warm-up"), prompt, gen_cfg, Cut::None)
    });
    let observed = conversation.turn(
        &format!("{label}: prompt {prompt:?}, cut {cut:?}"),
        &prompt,
        measured,
        cut,
    );
    conversation.assert_follow_up_matches_a_cold_run(fixture, &label, &prompt, &observed);
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
fn prefix_cache_route_slot_follows_the_request_on_the_plain_vocabulary() {
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
fn prefix_cache_route_slot_follows_the_request_on_a_flushed_lone_byte() {
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

/// The route over real tokenized chat prompts: a fresh turn that stops on length,
/// an exact-append turn under greedy decoding, an exact-append turn under seeded
/// sampling, a replay from a retained checkpoint, and a stop-string turn. Each
/// turn runs through `Conversation::turn`, which asserts that the sampling route
/// was torn down and that the slot follows the request; each turn after the first
/// extends the one before, and `real_turn_text` fails the run unless it succeeded.
/// The mode and stop assertions below only check that a turn reached the path its
/// case names, so a prompt that tokenizes differently from the generated ids fails
/// loudly instead of silently testing another path.
#[test]
fn prefix_cache_route_slot_follows_the_request_on_a_real_checkpoint() {
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
    let mut conversation = Conversation::with_state(&tokenizer, tokenized_prompt_ids, load());

    let chat =
        |messages: &[ChatMessage]| crate::forward::metal_qwen35::format_chat_template(messages);
    let q1 = "Write a short poem about the sea.";
    let q2 = "Now one about the sky.";
    let q_edit = "Now one about the mountains.";
    let q3 = "Now one about the desert.";
    let stop = |cfg: GenerateConfig| with_stop_tokens(cfg, &[im_end]);

    // Fresh, greedy, stopped by length.
    let prompt1 = chat(&[ChatMessage::user(q1)]);
    let turn1 = conversation.turn(
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
    let turn2 = conversation.turn(
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
    let turn3 = conversation.turn(
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
    let turn4 = conversation.turn(
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
    let turn5 = conversation.turn(
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
