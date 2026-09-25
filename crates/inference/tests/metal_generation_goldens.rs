//! Metal ordinary-generation characterization goldens.
//!
//! **Why this exists**: the two ordinary Metal generation entries,
//! `MetalQwen35State::generate` and `MetalQwen35State::generate_streaming_with_cancel`,
//! each run their own decode loop. Before either loop is moved anywhere, the
//! token streams they produce today have to be frozen, so a later change can
//! show it reproduced them rather than asserting it. Greedy output alone cannot
//! see a change to the sampler's draw schedule, so the table below also pins
//! seeded sampled streams, both readback routes (dense full-vocab logits and
//! the compact candidate readback), grammar, stop strings, stop tokens, the
//! token budget, streaming logprobs, the streaming reasoning budget, the two
//! refusals the direct entry makes, and both streaming cancellation points.
//!
//! **What is frozen per case**: generated token ids, returned text, stop
//! reason, stopped flag, prompt and generated token counts, the ids and text
//! handed to the streaming callback, logprob token ids, and the Metal logit
//! readback path-proof counters (full-vocab vs compact-candidate, counts and
//! bytes, prefill and decode). The counters are what lets a later change prove
//! the compact route stayed engaged: a token match alone would also pass on a
//! route that silently fell back to full-vocab readback.
//!
//! **The fixture is generated, never hand-written.** `write_metal_generation_goldens`
//! produces it from the current code on a machine with Metal and a checkpoint.
//! It runs every case twice and refuses to write if the two runs differ, and it
//! refuses to write if any case failed to exercise the condition its name
//! claims (see `check_intent`). Regenerate deliberately:
//!   ```bash
//!   LATTICE_METAL_GENERATION_MODEL_DIR=/abs/path/to/qwen3.5-0.8b \
//!   LATTICE_METAL_GENERATION_GOLDEN_WRITE=1 \
//!   cargo test --release -p lattice-inference --test metal_generation_goldens \
//!       --features metal-gpu,f16 -- --ignored write_metal_generation_goldens --nocapture
//!   ```
//!
//! **Replay** (`metal_generation_goldens_replay`):
//!   ```bash
//!   LATTICE_METAL_GENERATION_MODEL_DIR=/abs/path/to/qwen3.5-0.8b \
//!   cargo test --release -p lattice-inference --test metal_generation_goldens \
//!       --features metal-gpu,f16 -- metal_generation_goldens_replay --nocapture
//!   ```
//!   The directory may hold a Q4 artifact or safetensors weights; the fixture
//!   records which, and a replay against the other format refuses. The path must
//!   be absolute: `cargo test` runs test binaries with the crate directory as CWD.
//!
//! **Skip and refusal contract.** Without a checkpoint, or on a build without
//! macOS + `metal-gpu`, the replay prints a `LATTICE_METAL_GENERATION_GOLDEN_SKIPPED`
//! line and returns, unless `LATTICE_METAL_GENERATION_GOLDEN_ENFORCE=1` is set,
//! in which case it panics. A missing fixture file is a skip line too, unless
//! `LATTICE_REQUIRE_FIXTURES` is enabled, in which case every test that reads
//! the fixture refuses: a committed artifact that is absent is a broken
//! checkout, not a machine without a GPU. The `controls` module drives each of
//! those refusal paths without Metal or a checkpoint.
#![allow(clippy::field_reassign_with_default)]

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: u32 = 1;
const ARTIFACT_KIND: &str = "lattice-metal-ordinary-generation-golden";
const FIXTURE_RELATIVE_PATH: &str =
    "tests/fixtures/metal_generation_goldens_v1/qwen35_metal_generation.json";
const MODEL_DIR_VAR: &str = "LATTICE_METAL_GENERATION_MODEL_DIR";
const ENFORCE_VAR: &str = "LATTICE_METAL_GENERATION_GOLDEN_ENFORCE";
const WRITE_VAR: &str = "LATTICE_METAL_GENERATION_GOLDEN_WRITE";
const REQUIRE_FIXTURES_VAR: &str = "LATTICE_REQUIRE_FIXTURES";
const SKIP_MARKER: &str = "LATTICE_METAL_GENERATION_GOLDEN_SKIPPED";

const GREEDY_PROMPT: &str = "The capital of France is";
const GRAMMAR_SCHEMA: &str =
    r#"{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"]}"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Entry {
    Direct,
    Streaming,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Prompt {
    Plain(&'static str),
    /// One user turn in the chat template, followed by an empty reasoning block
    /// so the reply starts with the answer.
    ChatNoThink(&'static str),
}

impl Prompt {
    fn render(&self) -> String {
        match self {
            Self::Plain(text) => (*text).to_string(),
            Self::ChatNoThink(user) => format!(
                "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Sampler {
    Greedy,
    Seeded {
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cancel {
    Never,
    /// `on_token` returns `false` on its n-th invocation.
    OnTokenCall(usize),
    /// `should_cancel` returns `true` on its second poll, which the streaming
    /// entry makes immediately after prefill and before the first sample.
    AfterPrefill,
}

/// The condition a case exists to exercise. Checked on every live run and on
/// the committed fixture, so a case whose name claims a stop string but whose
/// capture ran to the token budget cannot pass as coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    Length,
    StopString,
    StopToken,
    GrammarJson,
    LogprobsCaptured,
    ReasoningBudgetForcesClose,
    CancelledByCallback,
    CancelledAfterPrefill,
    Refused,
}

/// The decode-step logit readback route the case must take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readback {
    /// Direct greedy zero-copy argmax: no per-step logit readback at all.
    Argmax,
    /// Full-vocab f32 logits read back every decode step.
    Dense,
    /// Compact candidate readback every step, including prefill.
    Compact,
    /// No decode step is expected to run (refusal, cancel after prefill).
    NoDecode,
}

#[derive(Debug)]
struct CaseSpec {
    name: &'static str,
    entry: Entry,
    prompt: Prompt,
    /// Value of `LATTICE_COMPACT_TOPK` while the case runs.
    compact_env: bool,
    sampler: Sampler,
    max_new_tokens: usize,
    stop_strings: &'static [&'static str],
    stop_on_im_end: bool,
    grammar: bool,
    logprobs: Option<usize>,
    enable_thinking: bool,
    reasoning_budget: Option<usize>,
    cancel: Cancel,
    expect: Expect,
    readback: Readback,
}

impl CaseSpec {
    /// Canonical description stored beside each case in the fixture. An edit to
    /// the table that is not followed by a regeneration changes this string and
    /// fails the fixture check instead of replaying a stale capture against a
    /// different request.
    fn description(&self) -> String {
        format!("{self:?}")
    }
}

const BASE: CaseSpec = CaseSpec {
    name: "",
    entry: Entry::Direct,
    prompt: Prompt::Plain(GREEDY_PROMPT),
    compact_env: false,
    sampler: Sampler::Greedy,
    max_new_tokens: 16,
    stop_strings: &[],
    stop_on_im_end: false,
    grammar: false,
    logprobs: None,
    enable_thinking: false,
    reasoning_budget: None,
    cancel: Cancel::Never,
    expect: Expect::Length,
    readback: Readback::Dense,
};

const SAMPLED_DENSE: Sampler = Sampler::Seeded {
    temperature: 0.8,
    top_k: 40,
    top_p: 0.9,
    seed: 0x5EED_0001,
};

/// `top_p` must be exactly 1.0 for the block top-k route; `top_k` must be one of
/// the precompiled local-k variants.
const SAMPLED_BLOCK_TOPK: Sampler = Sampler::Seeded {
    temperature: 0.8,
    top_k: 40,
    top_p: 1.0,
    seed: 0x5EED_0002,
};

const CASES: &[CaseSpec] = &[
    CaseSpec {
        name: "direct_greedy_argmax",
        readback: Readback::Argmax,
        ..BASE
    },
    CaseSpec {
        name: "direct_greedy_compact",
        compact_env: true,
        readback: Readback::Compact,
        ..BASE
    },
    CaseSpec {
        name: "direct_sampled_dense",
        sampler: SAMPLED_DENSE,
        ..BASE
    },
    CaseSpec {
        name: "direct_sampled_block_topk",
        compact_env: true,
        sampler: SAMPLED_BLOCK_TOPK,
        readback: Readback::Compact,
        ..BASE
    },
    CaseSpec {
        name: "direct_grammar_json",
        prompt: Prompt::ChatNoThink(
            "Reply with a JSON object whose key \"answer\" holds the number 42.",
        ),
        compact_env: true,
        max_new_tokens: 48,
        grammar: true,
        expect: Expect::GrammarJson,
        ..BASE
    },
    CaseSpec {
        name: "direct_stop_string",
        max_new_tokens: 48,
        stop_strings: &[".\n"],
        expect: Expect::StopString,
        readback: Readback::Argmax,
        ..BASE
    },
    CaseSpec {
        name: "direct_stop_token",
        prompt: Prompt::ChatNoThink("Reply with the single word: yes"),
        max_new_tokens: 48,
        stop_on_im_end: true,
        expect: Expect::StopToken,
        readback: Readback::Argmax,
        ..BASE
    },
    CaseSpec {
        name: "direct_logprobs_refused",
        logprobs: Some(2),
        expect: Expect::Refused,
        readback: Readback::NoDecode,
        ..BASE
    },
    CaseSpec {
        name: "direct_reasoning_budget_refused",
        enable_thinking: true,
        reasoning_budget: Some(4),
        expect: Expect::Refused,
        readback: Readback::NoDecode,
        ..BASE
    },
    CaseSpec {
        name: "streaming_greedy_dense",
        entry: Entry::Streaming,
        ..BASE
    },
    CaseSpec {
        name: "streaming_greedy_compact",
        entry: Entry::Streaming,
        compact_env: true,
        readback: Readback::Compact,
        ..BASE
    },
    CaseSpec {
        name: "streaming_sampled_dense",
        entry: Entry::Streaming,
        sampler: SAMPLED_DENSE,
        ..BASE
    },
    CaseSpec {
        name: "streaming_sampled_block_topk",
        entry: Entry::Streaming,
        compact_env: true,
        sampler: SAMPLED_BLOCK_TOPK,
        readback: Readback::Compact,
        ..BASE
    },
    CaseSpec {
        name: "streaming_grammar_json",
        entry: Entry::Streaming,
        prompt: Prompt::ChatNoThink(
            "Reply with a JSON object whose key \"answer\" holds the number 42.",
        ),
        compact_env: true,
        max_new_tokens: 48,
        grammar: true,
        expect: Expect::GrammarJson,
        ..BASE
    },
    CaseSpec {
        name: "streaming_stop_string",
        entry: Entry::Streaming,
        max_new_tokens: 48,
        stop_strings: &[".\n"],
        expect: Expect::StopString,
        ..BASE
    },
    CaseSpec {
        name: "streaming_stop_token",
        entry: Entry::Streaming,
        prompt: Prompt::ChatNoThink("Reply with the single word: yes"),
        max_new_tokens: 48,
        stop_on_im_end: true,
        expect: Expect::StopToken,
        ..BASE
    },
    CaseSpec {
        name: "streaming_logprobs",
        entry: Entry::Streaming,
        compact_env: true,
        max_new_tokens: 8,
        logprobs: Some(2),
        expect: Expect::LogprobsCaptured,
        ..BASE
    },
    CaseSpec {
        name: "streaming_reasoning_budget",
        entry: Entry::Streaming,
        max_new_tokens: 8,
        enable_thinking: true,
        reasoning_budget: Some(4),
        expect: Expect::ReasoningBudgetForcesClose,
        ..BASE
    },
    CaseSpec {
        name: "streaming_cancel_by_callback",
        entry: Entry::Streaming,
        cancel: Cancel::OnTokenCall(3),
        expect: Expect::CancelledByCallback,
        ..BASE
    },
    CaseSpec {
        name: "streaming_cancel_after_prefill",
        entry: Entry::Streaming,
        cancel: Cancel::AfterPrefill,
        expect: Expect::CancelledAfterPrefill,
        readback: Readback::NoDecode,
        ..BASE
    },
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadbackCounters {
    decode_full_vocab: u64,
    decode_compact_candidate: u64,
    prefill_full_vocab: u64,
    prefill_compact_candidate: u64,
    decode_full_vocab_bytes: u64,
    decode_compact_candidate_bytes: u64,
    prefill_full_vocab_bytes: u64,
    prefill_compact_candidate_bytes: u64,
}

impl ReadbackCounters {
    fn is_all_zero(&self) -> bool {
        *self == Self::zero()
    }

    fn zero() -> Self {
        Self {
            decode_full_vocab: 0,
            decode_compact_candidate: 0,
            prefill_full_vocab: 0,
            prefill_compact_candidate: 0,
            decode_full_vocab_bytes: 0,
            decode_compact_candidate_bytes: 0,
            prefill_full_vocab_bytes: 0,
            prefill_compact_candidate_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Generated {
    token_ids: Vec<u32>,
    text: String,
    stop_reason: String,
    stopped: bool,
    prompt_tokens: usize,
    generated_tokens: usize,
    callback_ids: Vec<u32>,
    callback_text: String,
    logprob_token_ids: Vec<u32>,
    top_logprob_ids: Vec<Vec<u32>>,
    readback: ReadbackCounters,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Refused {
    error: String,
    readback: ReadbackCounters,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum Outcome {
    Generated(Generated),
    Refused(Refused),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseRecord {
    name: String,
    entry: Entry,
    config: String,
    outcome: Outcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointDescriptor {
    model_format: String,
    vocab_size: usize,
    eos_token_id: u32,
    im_end_token_id: u32,
    think_close_token_id: u32,
    kv_f16: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: u32,
    artifact_kind: String,
    checkpoint: CheckpointDescriptor,
    cases: Vec<CaseRecord>,
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_RELATIVE_PATH)
}

fn env_lookup(var: &str) -> Option<String> {
    std::env::var(var).ok()
}

/// Same value semantics as the crate's own environment switches: unset, empty,
/// `0`, `false`, `no` and `off` are off; anything else is on.
fn switch_on(lookup: &impl Fn(&str) -> Option<String>, var: &str) -> bool {
    match lookup(var) {
        None => false,
        Some(raw) => {
            let text = raw.trim();
            !(text.is_empty()
                || ["0", "false", "no", "off"]
                    .iter()
                    .any(|off| text.eq_ignore_ascii_case(off)))
        }
    }
}

/// Reads and validates the committed fixture.
///
/// `Ok(None)` means the file is absent and fixtures are not required; the
/// caller prints the skip line. An absent file under `LATTICE_REQUIRE_FIXTURES`
/// and any file that fails to parse are errors.
fn load_fixture(
    path: &Path,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<Option<Fixture>, String> {
    if !path.exists() {
        if switch_on(lookup, REQUIRE_FIXTURES_VAR) {
            return Err(format!(
                "{} is absent and {REQUIRE_FIXTURES_VAR} is enabled; the Metal generation \
                 golden is a committed artifact. Generate it with write_metal_generation_goldens \
                 and commit it.",
                path.display()
            ));
        }
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("reading {}: {error}", path.display()))?;
    parse_fixture(&text).map(Some)
}

fn parse_fixture(text: &str) -> Result<Fixture, String> {
    let fixture: Fixture =
        serde_json::from_str(text).map_err(|error| format!("fixture does not parse: {error}"))?;
    if fixture.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "fixture schema_version {} is not {SCHEMA_VERSION}",
            fixture.schema_version
        ));
    }
    if fixture.artifact_kind != ARTIFACT_KIND {
        return Err(format!(
            "fixture artifact_kind {:?} is not {ARTIFACT_KIND:?}",
            fixture.artifact_kind
        ));
    }
    Ok(fixture)
}

/// The fixture's cases must be exactly the table's cases, in table order, each
/// captured under the request the table describes today.
fn check_fixture_matches_table(fixture: &Fixture) -> Result<(), String> {
    let recorded: Vec<&str> = fixture.cases.iter().map(|c| c.name.as_str()).collect();
    let declared: Vec<&str> = CASES.iter().map(|c| c.name).collect();
    if recorded != declared {
        return Err(format!(
            "fixture cases {recorded:?} are not the declared cases {declared:?}; regenerate"
        ));
    }
    let mut failures = Vec::new();
    for (record, spec) in fixture.cases.iter().zip(CASES) {
        if record.entry != spec.entry {
            failures.push(format!(
                "case {}: fixture entry {:?}, table entry {:?}",
                spec.name, record.entry, spec.entry
            ));
        }
        if record.config != spec.description() {
            failures.push(format!(
                "case {}: fixture was captured under a different request\n  fixture: {}\n  table:   {}",
                spec.name,
                record.config,
                spec.description()
            ));
        }
        if let Err(reason) = check_intent(spec, &record.outcome, &fixture.checkpoint) {
            failures.push(format!("case {}: {reason}", spec.name));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

/// Field-by-field comparison. Every mismatch is listed; a token-list mismatch
/// names the first diverging index.
fn compare_records(expected: &CaseRecord, actual: &CaseRecord) -> Vec<String> {
    let mut diffs = Vec::new();
    let name = &expected.name;
    if expected.name != actual.name {
        diffs.push(format!(
            "case name {:?} vs {:?}",
            expected.name, actual.name
        ));
    }
    if expected.entry != actual.entry {
        diffs.push(format!(
            "case {name}: entry {:?} vs {:?}",
            expected.entry, actual.entry
        ));
    }
    if expected.config != actual.config {
        diffs.push(format!(
            "case {name}: config\n  expected: {}\n  actual:   {}",
            expected.config, actual.config
        ));
    }
    match (&expected.outcome, &actual.outcome) {
        (Outcome::Generated(e), Outcome::Generated(a)) => {
            if e.token_ids != a.token_ids {
                let first = e
                    .token_ids
                    .iter()
                    .zip(&a.token_ids)
                    .position(|(x, y)| x != y)
                    .unwrap_or_else(|| e.token_ids.len().min(a.token_ids.len()));
                diffs.push(format!(
                    "case {name}: token ids diverge at index {first}\n  expected: {:?}\n  actual:   {:?}",
                    e.token_ids, a.token_ids
                ));
            }
            let fields: [(&str, String, String); 10] = [
                ("text", format!("{:?}", e.text), format!("{:?}", a.text)),
                ("stop_reason", e.stop_reason.clone(), a.stop_reason.clone()),
                ("stopped", e.stopped.to_string(), a.stopped.to_string()),
                (
                    "prompt_tokens",
                    e.prompt_tokens.to_string(),
                    a.prompt_tokens.to_string(),
                ),
                (
                    "generated_tokens",
                    e.generated_tokens.to_string(),
                    a.generated_tokens.to_string(),
                ),
                (
                    "callback_ids",
                    format!("{:?}", e.callback_ids),
                    format!("{:?}", a.callback_ids),
                ),
                (
                    "callback_text",
                    format!("{:?}", e.callback_text),
                    format!("{:?}", a.callback_text),
                ),
                (
                    "logprob_token_ids",
                    format!("{:?}", e.logprob_token_ids),
                    format!("{:?}", a.logprob_token_ids),
                ),
                (
                    "top_logprob_ids",
                    format!("{:?}", e.top_logprob_ids),
                    format!("{:?}", a.top_logprob_ids),
                ),
                (
                    "readback",
                    format!("{:?}", e.readback),
                    format!("{:?}", a.readback),
                ),
            ];
            for (field, want, got) in fields {
                if want != got {
                    diffs.push(format!(
                        "case {name}: {field}\n  expected: {want}\n  actual:   {got}"
                    ));
                }
            }
        }
        (Outcome::Refused(e), Outcome::Refused(a)) => {
            if e != a {
                diffs.push(format!(
                    "case {name}: refusal\n  expected: {e:?}\n  actual:   {a:?}"
                ));
            }
        }
        (e, a) => diffs.push(format!(
            "case {name}: outcome kind\n  expected: {e:?}\n  actual:   {a:?}"
        )),
    }
    diffs
}

/// Checks that an outcome exercised the condition and the readback route its
/// case declares.
fn check_intent(
    spec: &CaseSpec,
    outcome: &Outcome,
    checkpoint: &CheckpointDescriptor,
) -> Result<(), String> {
    let generated = match (spec.expect, outcome) {
        (Expect::Refused, Outcome::Refused(refused)) => {
            if refused.error.trim().is_empty() {
                return Err("refusal recorded with an empty error".into());
            }
            if !refused.readback.is_all_zero() {
                return Err(format!(
                    "refused request still read logits back: {:?}",
                    refused.readback
                ));
            }
            return Ok(());
        }
        (Expect::Refused, Outcome::Generated(_)) => {
            return Err("expected the entry to refuse this request, but it generated".into());
        }
        (_, Outcome::Refused(refused)) => {
            return Err(format!(
                "expected generation, but the entry refused: {}",
                refused.error
            ));
        }
        (_, Outcome::Generated(generated)) => generated,
    };
    check_readback(spec, generated, checkpoint)?;

    let len = generated.token_ids.len();
    let eos = "Some(Eos)";
    match spec.expect {
        Expect::Refused => unreachable!("handled above"),
        Expect::Length => {
            if generated.stop_reason != "Some(Length)"
                || generated.stopped
                || len != spec.max_new_tokens
            {
                return Err(format!(
                    "expected a token-budget stop at {} tokens, got {} tokens, stop_reason {}, \
                     stopped {}",
                    spec.max_new_tokens, len, generated.stop_reason, generated.stopped
                ));
            }
        }
        Expect::StopString => {
            let hit = spec
                .stop_strings
                .iter()
                .all(|stop| !generated.text.contains(stop));
            if generated.stop_reason != eos
                || !generated.stopped
                || len == 0
                || len >= spec.max_new_tokens
                || !hit
            {
                return Err(format!(
                    "expected a stop-string stop inside the budget with the match excluded from \
                     the text, got {len} tokens, stop_reason {}, stopped {}, text {:?}",
                    generated.stop_reason, generated.stopped, generated.text
                ));
            }
        }
        Expect::StopToken => {
            if generated.stop_reason != eos
                || !generated.stopped
                || len >= spec.max_new_tokens
                || generated.token_ids.contains(&checkpoint.im_end_token_id)
            {
                return Err(format!(
                    "expected a stop-token stop inside the budget with the stop token excluded, \
                     got {len} tokens, stop_reason {}, stopped {}",
                    generated.stop_reason, generated.stopped
                ));
            }
        }
        Expect::GrammarJson => {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(generated.text.trim());
            let answer_is_integer = parsed
                .as_ref()
                .ok()
                .and_then(|value| value.get("answer"))
                .is_some_and(|answer| answer.is_i64() || answer.is_u64());
            if !answer_is_integer {
                return Err(format!(
                    "expected grammar-constrained text to be a JSON object with an integer \
                     \"answer\", got {:?} (stop_reason {})",
                    generated.text, generated.stop_reason
                ));
            }
        }
        Expect::LogprobsCaptured => {
            let want_top = spec.logprobs.unwrap_or(0);
            if len == 0
                || generated.logprob_token_ids != generated.token_ids
                || generated.top_logprob_ids.len() != len
                || generated
                    .top_logprob_ids
                    .iter()
                    .any(|top| top.len() != want_top)
            {
                return Err(format!(
                    "expected one logprob entry per generated token with {want_top} \
                     alternatives each, got token_ids {:?}, logprob ids {:?}, top {:?}",
                    generated.token_ids, generated.logprob_token_ids, generated.top_logprob_ids
                ));
            }
        }
        Expect::ReasoningBudgetForcesClose => {
            let budget = spec.reasoning_budget.unwrap_or(0);
            let close_at = generated
                .token_ids
                .iter()
                .position(|&id| id == checkpoint.think_close_token_id);
            if budget == 0 || close_at.is_none_or(|index| index > budget) {
                return Err(format!(
                    "expected the reasoning close token {} at or before index {budget}, got \
                     token_ids {:?}",
                    checkpoint.think_close_token_id, generated.token_ids
                ));
            }
        }
        Expect::CancelledByCallback => {
            let Cancel::OnTokenCall(call) = spec.cancel else {
                return Err("a callback-cancel case must declare Cancel::OnTokenCall".into());
            };
            if generated.stop_reason != "Some(Interrupt)"
                || generated.stopped
                || generated.callback_ids.len() != call
            {
                return Err(format!(
                    "expected an interrupt on callback {call}, got {} callbacks, stop_reason {}, \
                     stopped {}",
                    generated.callback_ids.len(),
                    generated.stop_reason,
                    generated.stopped
                ));
            }
        }
        Expect::CancelledAfterPrefill => {
            let prefill = generated.readback.prefill_full_vocab
                + generated.readback.prefill_compact_candidate;
            if generated.stop_reason != "Some(Interrupt)"
                || generated.stopped
                || len != 0
                || !generated.callback_ids.is_empty()
                || prefill == 0
            {
                return Err(format!(
                    "expected an interrupt after prefill and before the first token, got {len} \
                     tokens, {} callbacks, {prefill} prefill readbacks, stop_reason {}",
                    generated.callback_ids.len(),
                    generated.stop_reason
                ));
            }
        }
    }
    Ok(())
}

fn check_readback(
    spec: &CaseSpec,
    generated: &Generated,
    checkpoint: &CheckpointDescriptor,
) -> Result<(), String> {
    let r = &generated.readback;
    if r.is_all_zero() {
        return Err(
            "every logit readback counter is zero: the path-proof counters were not live \
             (LATTICE_METAL_PATH_PROOF must be enabled when the state is constructed)"
                .into(),
        );
    }
    let full_bytes = (checkpoint.vocab_size * std::mem::size_of::<f32>()) as u64;
    if r.decode_full_vocab_bytes != r.decode_full_vocab * full_bytes
        || r.prefill_full_vocab_bytes != r.prefill_full_vocab * full_bytes
    {
        return Err(format!(
            "full-vocab readback bytes do not match count x vocab_size x 4: {r:?}"
        ));
    }
    let ok = match spec.readback {
        Readback::Argmax => {
            r.decode_full_vocab == 0
                && r.decode_compact_candidate == 0
                && r.prefill_full_vocab >= 1
                && r.prefill_compact_candidate == 0
                && generated.token_ids.len() >= 2
        }
        Readback::Dense => {
            r.decode_full_vocab >= 1
                && r.decode_compact_candidate == 0
                && r.prefill_compact_candidate == 0
        }
        Readback::Compact => {
            r.decode_compact_candidate >= 1
                && r.decode_full_vocab == 0
                && r.prefill_compact_candidate >= 1
                && r.prefill_full_vocab == 0
        }
        Readback::NoDecode => r.decode_full_vocab == 0 && r.decode_compact_candidate == 0,
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "path-proof readback counters do not show the {:?} route: {r:?}",
            spec.readback
        ))
    }
}

enum Checkpoint {
    Run(PathBuf),
    Skip(String),
}

/// Resolves the checkpoint directory. A missing or unset directory is a skip
/// unless the enforce variable is on, in which case it is an error.
fn resolve_checkpoint(
    lookup: &impl Fn(&str) -> Option<String>,
    exists: impl Fn(&Path) -> bool,
) -> Result<Checkpoint, String> {
    let enforce = switch_on(lookup, ENFORCE_VAR);
    let refuse_or_skip = |reason: String| {
        if enforce {
            Err(format!("{reason}, and {ENFORCE_VAR} is enabled"))
        } else {
            Ok(Checkpoint::Skip(reason))
        }
    };
    let Some(raw) = lookup(MODEL_DIR_VAR) else {
        return refuse_or_skip(format!("{MODEL_DIR_VAR} is unset"));
    };
    let path = PathBuf::from(&raw);
    if !path.is_absolute() {
        return Err(format!(
            "{MODEL_DIR_VAR}={raw:?} is relative; cargo test runs test binaries with the crate \
             directory as CWD. Pass an absolute path."
        ));
    }
    if !exists(&path) {
        return refuse_or_skip(format!("{MODEL_DIR_VAR}={raw:?} does not exist"));
    }
    Ok(Checkpoint::Run(path))
}

fn skip(test: &str, reason: &str) {
    eprintln!("{SKIP_MARKER} test={test} reason={reason}");
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod metal {
    use super::*;
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::grammar::{GrammarEngine, GrammarSpec};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::model_format::{ModelFormat, detect_format};
    use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};
    use lattice_inference::{GenerateConfig, GenerateOutput};
    use std::sync::{Arc, Mutex, MutexGuard};

    /// Environment switches that select a decode route. Every case runs with
    /// each of them pinned, so an ambient value in the caller's shell cannot
    /// move a case onto a different route than the one it was captured on.
    const ROUTE_SWITCHES: &[(&str, &str)] = &[
        ("LATTICE_METAL_PATH_PROOF", "1"),
        ("LATTICE_COMPACT_TOPK", "0"),
        ("LATTICE_COMPACT_TOPK_SELECT", "0"),
        ("LATTICE_COMPACT_TOPP_APPROX", "0"),
        ("LATTICE_SELF_SPEC", "0"),
        ("LATTICE_MTP", "0"),
    ];

    const MAX_CACHE_LEN: usize = 2048;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Pins [`ROUTE_SWITCHES`] for its lifetime and restores the prior values on
    /// drop. `set_var` is `unsafe` because a concurrent read on another thread is
    /// a data race; the mutex serializes every writer in this binary, which is
    /// the same convention the in-crate compact-route tests use.
    struct RouteEnvironment {
        prior: Vec<(&'static str, Option<std::ffi::OsString>)>,
        _lock: MutexGuard<'static, ()>,
    }

    impl RouteEnvironment {
        fn pin() -> Self {
            let lock = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let prior = ROUTE_SWITCHES
                .iter()
                .map(|(name, _)| (*name, std::env::var_os(name)))
                .collect();
            for (name, value) in ROUTE_SWITCHES {
                // SAFETY: writers are serialized by ENV_LOCK, held for this guard's life.
                unsafe { std::env::set_var(name, value) };
            }
            Self { prior, _lock: lock }
        }

        fn set_compact(&self, on: bool) {
            // SAFETY: writers are serialized by ENV_LOCK, held by `self`.
            unsafe { std::env::set_var("LATTICE_COMPACT_TOPK", if on { "1" } else { "0" }) };
        }
    }

    impl Drop for RouteEnvironment {
        fn drop(&mut self) {
            for (name, value) in &self.prior {
                // SAFETY: writers are serialized by ENV_LOCK, still held here.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    struct Resolved {
        checkpoint: CheckpointDescriptor,
        grammar: Arc<GrammarEngine>,
    }

    fn build_config(spec: &CaseSpec, resolved: &Resolved) -> GenerateConfig {
        let mut cfg = GenerateConfig::default();
        cfg.max_new_tokens = spec.max_new_tokens;
        match spec.sampler {
            Sampler::Greedy => {
                cfg.temperature = 0.0;
                cfg.top_k = 1;
                cfg.top_p = 1.0;
                cfg.seed = Some(1);
            }
            Sampler::Seeded {
                temperature,
                top_k,
                top_p,
                seed,
            } => {
                cfg.temperature = temperature;
                cfg.top_k = top_k;
                cfg.top_p = top_p;
                cfg.seed = Some(seed);
            }
        }
        cfg.min_p = 0.0;
        cfg.repetition_penalty = 1.0;
        cfg.stop_token_ids = if spec.stop_on_im_end {
            vec![resolved.checkpoint.im_end_token_id]
        } else {
            vec![]
        };
        cfg.enable_thinking = spec.enable_thinking;
        cfg.enable_mtp = Some(false);
        cfg.grammar = spec.grammar.then(|| Arc::clone(&resolved.grammar));
        cfg.stop_strings = spec.stop_strings.iter().map(|s| (*s).to_string()).collect();
        cfg.reasoning_budget = spec.reasoning_budget;
        cfg.logprobs = spec.logprobs;
        cfg
    }

    fn counters(state: &MetalQwen35State) -> ReadbackCounters {
        let s = state.logit_readback_path_proof_snapshot();
        ReadbackCounters {
            decode_full_vocab: s.decode_full_vocab,
            decode_compact_candidate: s.decode_compact_candidate,
            prefill_full_vocab: s.prefill_full_vocab,
            prefill_compact_candidate: s.prefill_compact_candidate,
            decode_full_vocab_bytes: s.decode_full_vocab_bytes,
            decode_compact_candidate_bytes: s.decode_compact_candidate_bytes,
            prefill_full_vocab_bytes: s.prefill_full_vocab_bytes,
            prefill_compact_candidate_bytes: s.prefill_compact_candidate_bytes,
        }
    }

    fn generated(
        output: GenerateOutput,
        callback_ids: Vec<u32>,
        callback_text: String,
        readback: ReadbackCounters,
    ) -> Outcome {
        Outcome::Generated(Generated {
            logprob_token_ids: output.token_logprobs.iter().map(|t| t.token_id).collect(),
            top_logprob_ids: output
                .token_logprobs
                .iter()
                .map(|t| t.top.iter().map(|alt| alt.token_id).collect())
                .collect(),
            token_ids: output.token_ids,
            text: output.text,
            stop_reason: format!("{:?}", output.stop_reason),
            stopped: output.stopped,
            prompt_tokens: output.prompt_tokens,
            generated_tokens: output.generated_tokens,
            callback_ids,
            callback_text,
            readback,
        })
    }

    fn run_case(
        state: &mut MetalQwen35State,
        tokenizer: &BpeTokenizer,
        env: &RouteEnvironment,
        spec: &CaseSpec,
        resolved: &Resolved,
    ) -> CaseRecord {
        env.set_compact(spec.compact_env);
        let cfg = build_config(spec, resolved);
        let prompt = spec.prompt.render();
        state.reset_path_proof_counters();
        let outcome = match spec.entry {
            Entry::Direct => {
                let result = state.generate(&prompt, tokenizer, &cfg);
                let readback = counters(state);
                match result {
                    Ok(output) => generated(output, Vec::new(), String::new(), readback),
                    Err(error) => Outcome::Refused(Refused {
                        error: error.to_string(),
                        readback,
                    }),
                }
            }
            Entry::Streaming => {
                let mut callback_ids = Vec::new();
                let mut callback_text = String::new();
                let mut polls = 0usize;
                let result = state.generate_streaming_with_cancel(
                    &prompt,
                    tokenizer,
                    &cfg,
                    |text, id| {
                        callback_ids.push(id);
                        callback_text.push_str(text);
                        !matches!(spec.cancel, Cancel::OnTokenCall(n) if callback_ids.len() >= n)
                    },
                    || {
                        polls += 1;
                        spec.cancel == Cancel::AfterPrefill && polls >= 2
                    },
                );
                let readback = counters(state);
                match result {
                    Ok(output) => generated(output, callback_ids, callback_text, readback),
                    Err(error) => Outcome::Refused(Refused {
                        error: error.to_string(),
                        readback,
                    }),
                }
            }
        };
        CaseRecord {
            name: spec.name.to_string(),
            entry: spec.entry,
            config: spec.description(),
            outcome,
        }
    }

    /// Loads the checkpoint and runs every case `rounds` times. Returns the
    /// descriptor and, per round, one record per case in table order.
    pub(super) fn capture(
        model_dir: &Path,
        rounds: usize,
    ) -> Result<(CheckpointDescriptor, Vec<Vec<CaseRecord>>), String> {
        let _gpu_guard = lattice_inference::measurement::gpu_test_lock();
        let env = RouteEnvironment::pin();

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)
            .map_err(|error| format!("tokenizer {}: {error}", tokenizer_path.display()))?;
        let (mut state, cfg, model_format) = match detect_format(model_dir) {
            ModelFormat::Q4 => {
                let cfg = Qwen35Config::from_model_dir(model_dir)
                    .map_err(|error| format!("config.json: {error}"))?;
                let state =
                    MetalQwen35State::from_q4_dir(model_dir, &tokenizer_path, &cfg, MAX_CACHE_LEN)
                        .map_err(|error| format!("from_q4_dir: {error}"))?;
                (state, cfg, "q4")
            }
            ModelFormat::Safetensors => {
                let model = Qwen35Model::from_safetensors(model_dir)
                    .map_err(|error| format!("from_safetensors: {error}"))?;
                let cfg = model.config().clone();
                let state = MetalQwen35State::new(model.weights(), &cfg, MAX_CACHE_LEN)
                    .map_err(|error| format!("MetalQwen35State::new: {error}"))?;
                (state, cfg, "safetensors")
            }
            _ => {
                return Err(format!(
                    "{} holds neither safetensors weights nor a Q4 artifact",
                    model_dir.display()
                ));
            }
        };

        let im_end_token_id = tokenizer
            .special_token_id("<|im_end|>")
            .ok_or("tokenizer has no <|im_end|> token")?;
        // `</think>` is an added token that is not marked special, so
        // `special_token_id` does not see it; tokenizing the marker must yield
        // exactly one id that renders back as the marker.
        let think_close_token_id = {
            let encoded = tokenizer.tokenize("</think>");
            match &encoded.input_ids[..encoded.real_length] {
                [id] if tokenizer.token_for_id(*id) == Some("</think>") => *id,
                other => {
                    return Err(format!(
                        "</think> does not tokenize to a single marker id: {other:?}"
                    ));
                }
            }
        };
        let spec = GrammarSpec::json_schema_str(GRAMMAR_SCHEMA)
            .map_err(|error| format!("grammar schema: {error}"))?;
        let vocab_bytes = tokenizer
            .vocab_bytes(cfg.vocab_size)
            .map_err(|error| format!("vocab bytes: {error}"))?;
        let grammar = GrammarEngine::new(&spec, vocab_bytes)
            .map_err(|error| format!("grammar engine: {error}"))?;
        let resolved = Resolved {
            checkpoint: CheckpointDescriptor {
                model_format: model_format.to_string(),
                vocab_size: cfg.vocab_size,
                eos_token_id: cfg.eos_token_id,
                im_end_token_id,
                think_close_token_id,
                kv_f16: state.path_proof_snapshot().kv_f16,
            },
            grammar: Arc::new(grammar),
        };

        let mut all_rounds = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let records = CASES
                .iter()
                .map(|spec| run_case(&mut state, &tokenizer, &env, spec, &resolved))
                .collect();
            all_rounds.push(records);
        }
        Ok((resolved.checkpoint, all_rounds))
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
mod metal {
    use super::*;

    pub(super) fn capture(
        _model_dir: &Path,
        _rounds: usize,
    ) -> Result<(CheckpointDescriptor, Vec<Vec<CaseRecord>>), String> {
        Err("this build lacks macOS + the metal-gpu feature".into())
    }
}

fn metal_available() -> bool {
    cfg!(all(target_os = "macos", feature = "metal-gpu"))
}

/// Checkpoint-free: the committed fixture parses, names exactly the declared
/// cases under the requests the table describes, and every case exercised what
/// its name claims, readback route included.
#[test]
fn metal_generation_golden_fixture_is_valid() {
    let fixture = match load_fixture(&fixture_path(), &env_lookup) {
        Ok(Some(fixture)) => fixture,
        Ok(None) => {
            skip(
                "metal_generation_golden_fixture_is_valid",
                &format!("fixture_absent path={}", fixture_path().display()),
            );
            return;
        }
        Err(reason) => panic!("{reason}"),
    };
    if let Err(reason) = check_fixture_matches_table(&fixture) {
        panic!("{reason}");
    }
}

/// Replays every case on the real checkpoint and compares against the fixture
/// exactly. The live run's own readback route is checked first, so a case that
/// left the compact route fails on the path-proof counters even before the
/// token comparison.
#[test]
fn metal_generation_goldens_replay() {
    const TEST: &str = "metal_generation_goldens_replay";
    let fixture = match load_fixture(&fixture_path(), &env_lookup) {
        Ok(Some(fixture)) => fixture,
        Ok(None) => {
            skip(
                TEST,
                &format!("fixture_absent path={}", fixture_path().display()),
            );
            return;
        }
        Err(reason) => panic!("{reason}"),
    };
    if !metal_available() {
        if switch_on(&env_lookup, ENFORCE_VAR) {
            panic!(
                "{ENFORCE_VAR} is enabled but this build lacks macOS + metal-gpu; run with \
                 --features metal-gpu,f16 on macOS"
            );
        }
        skip(TEST, "metal_gpu_unavailable");
        return;
    }
    let model_dir = match resolve_checkpoint(&env_lookup, Path::exists) {
        Ok(Checkpoint::Run(dir)) => dir,
        Ok(Checkpoint::Skip(reason)) => {
            skip(TEST, &reason);
            return;
        }
        Err(reason) => panic!("{reason}"),
    };

    let (checkpoint, mut rounds) =
        metal::capture(&model_dir, 1).unwrap_or_else(|reason| panic!("{reason}"));
    let live = rounds.pop().unwrap_or_default();
    assert_eq!(
        live.len(),
        CASES.len(),
        "the live run did not cover every case"
    );
    assert_eq!(
        checkpoint, fixture.checkpoint,
        "the checkpoint does not match the one the fixture was captured from"
    );

    let mut failures = Vec::new();
    for (spec, record) in CASES.iter().zip(&live) {
        if let Err(reason) = check_intent(spec, &record.outcome, &checkpoint) {
            failures.push(format!("case {} (live run): {reason}", spec.name));
        }
    }
    if let Err(reason) = check_fixture_matches_table(&fixture) {
        failures.push(reason);
    }
    for (expected, actual) in fixture.cases.iter().zip(&live) {
        failures.extend(compare_records(expected, actual));
    }
    assert!(
        failures.is_empty(),
        "Metal generation diverged from {}:\n{}",
        fixture_path().display(),
        failures.join("\n")
    );
}

/// Writes the fixture from the current code. Ignored, and additionally gated on
/// an explicit variable, so `--include-ignored` alone cannot overwrite a
/// committed golden.
#[test]
#[ignore = "writes the committed fixture; needs Metal, a checkpoint and LATTICE_METAL_GENERATION_GOLDEN_WRITE=1"]
fn write_metal_generation_goldens() {
    assert!(
        switch_on(&env_lookup, WRITE_VAR),
        "{WRITE_VAR} must be enabled to overwrite {}",
        fixture_path().display()
    );
    let model_dir = match resolve_checkpoint(
        &|var: &str| {
            if var == ENFORCE_VAR {
                Some("1".to_string())
            } else {
                env_lookup(var)
            }
        },
        Path::exists,
    ) {
        Ok(Checkpoint::Run(dir)) => dir,
        Ok(Checkpoint::Skip(reason)) => panic!("{reason}"),
        Err(reason) => panic!("{reason}"),
    };
    let (checkpoint, rounds) =
        metal::capture(&model_dir, 2).unwrap_or_else(|reason| panic!("{reason}"));
    let [first, second] = <[Vec<CaseRecord>; 2]>::try_from(rounds)
        .unwrap_or_else(|rounds| panic!("expected two capture rounds, got {}", rounds.len()));
    assert!(
        first.len() == CASES.len() && second.len() == CASES.len(),
        "a capture round did not cover every case"
    );

    let mut failures = Vec::new();
    for (a, b) in first.iter().zip(&second) {
        for diff in compare_records(a, b) {
            failures.push(format!("not reproducible across two runs: {diff}"));
        }
    }
    for (spec, record) in CASES.iter().zip(&first) {
        if let Err(reason) = check_intent(spec, &record.outcome, &checkpoint) {
            failures.push(format!("case {}: {reason}", spec.name));
        }
    }
    assert!(
        failures.is_empty(),
        "refusing to write the fixture:\n{}",
        failures.join("\n")
    );

    let fixture = Fixture {
        schema_version: SCHEMA_VERSION,
        artifact_kind: ARTIFACT_KIND.to_string(),
        checkpoint,
        cases: first,
    };
    check_fixture_matches_table(&fixture).unwrap_or_else(|reason| panic!("{reason}"));
    let mut text = serde_json::to_string_pretty(&fixture).expect("fixture serializes");
    text.push('\n');
    let path = fixture_path();
    let parent = path.parent().expect("fixture path has a parent");
    std::fs::create_dir_all(parent)
        .unwrap_or_else(|error| panic!("creating {}: {error}", parent.display()));
    let staging = path.with_extension("json.tmp");
    std::fs::write(&staging, &text)
        .unwrap_or_else(|error| panic!("writing {}: {error}", staging.display()));
    std::fs::rename(&staging, &path)
        .unwrap_or_else(|error| panic!("renaming onto {}: {error}", path.display()));
    eprintln!("wrote {} ({} cases)", path.display(), fixture.cases.len());
}

/// Checkpoint-free controls: each drives a refusal path directly, with a
/// must-pass arm beside it so an unconditional refusal cannot pass as a
/// working one.
mod controls {
    use super::*;

    fn lookup_from(
        pairs: &'static [(&'static str, &'static str)],
    ) -> impl Fn(&str) -> Option<String> {
        move |var| {
            pairs
                .iter()
                .find(|(k, _)| *k == var)
                .map(|(_, v)| (*v).to_string())
        }
    }

    fn checkpoint() -> CheckpointDescriptor {
        CheckpointDescriptor {
            model_format: "q4".into(),
            vocab_size: 1000,
            eos_token_id: 7,
            im_end_token_id: 8,
            think_close_token_id: 9,
            kv_f16: false,
        }
    }

    fn spec(name: &str) -> &'static CaseSpec {
        CASES
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no case {name}"))
    }

    fn dense_counters(decode_steps: u64) -> ReadbackCounters {
        ReadbackCounters {
            decode_full_vocab: decode_steps,
            decode_full_vocab_bytes: decode_steps * 4000,
            prefill_full_vocab: 1,
            prefill_full_vocab_bytes: 4000,
            ..ReadbackCounters::zero()
        }
    }

    fn compact_counters(decode_steps: u64) -> ReadbackCounters {
        ReadbackCounters {
            decode_compact_candidate: decode_steps,
            decode_compact_candidate_bytes: decode_steps * 8,
            prefill_compact_candidate: 1,
            prefill_compact_candidate_bytes: 8,
            ..ReadbackCounters::zero()
        }
    }

    fn length_record(name: &str, readback: ReadbackCounters) -> CaseRecord {
        let spec = spec(name);
        let token_ids: Vec<u32> = (100..100 + spec.max_new_tokens as u32).collect();
        CaseRecord {
            name: spec.name.into(),
            entry: spec.entry,
            config: spec.description(),
            outcome: Outcome::Generated(Generated {
                generated_tokens: token_ids.len(),
                token_ids,
                text: "text".into(),
                stop_reason: "Some(Length)".into(),
                stopped: false,
                prompt_tokens: 5,
                callback_ids: vec![],
                callback_text: String::new(),
                logprob_token_ids: vec![],
                top_logprob_ids: vec![],
                readback,
            }),
        }
    }

    fn generated_mut(record: &mut CaseRecord) -> &mut Generated {
        match &mut record.outcome {
            Outcome::Generated(generated) => generated,
            Outcome::Refused(_) => panic!("record is a refusal"),
        }
    }

    #[test]
    fn missing_fixture_refuses_under_require_fixtures() {
        let err = load_fixture(
            Path::new("/nonexistent/metal_generation.json"),
            &lookup_from(&[(REQUIRE_FIXTURES_VAR, "1")]),
        )
        .unwrap_err();
        assert!(err.contains("is absent"), "unexpected message: {err}");
    }

    #[test]
    fn missing_fixture_without_require_fixtures_is_a_skip() {
        // Must-pass arm for the refusal above, and the `0` spelling that must not
        // count as enabled.
        for pairs in [
            &[][..],
            &[(REQUIRE_FIXTURES_VAR, "0")][..],
            &[(REQUIRE_FIXTURES_VAR, "off")][..],
        ] {
            let lookup = |var: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == var)
                    .map(|(_, v)| (*v).to_string())
            };
            let loaded = load_fixture(Path::new("/nonexistent/metal_generation.json"), &lookup)
                .expect("absent fixture without the require flag is a skip, not an error");
            assert!(loaded.is_none());
        }
    }

    #[test]
    fn token_mismatch_is_detected() {
        let expected = length_record("direct_sampled_dense", dense_counters(15));
        let mut actual = expected.clone();
        generated_mut(&mut actual).token_ids[7] += 1;
        let diffs = compare_records(&expected, &actual);
        assert_eq!(diffs.len(), 1, "unexpected diffs: {diffs:?}");
        assert!(
            diffs[0].contains("token ids diverge at index 7"),
            "unexpected diff: {}",
            diffs[0]
        );
    }

    #[test]
    fn identical_records_compare_clean() {
        let expected = length_record("direct_sampled_dense", dense_counters(15));
        assert!(compare_records(&expected, &expected.clone()).is_empty());
    }

    #[test]
    fn counter_mismatch_is_detected() {
        let expected = length_record("streaming_greedy_compact", compact_counters(15));
        let mut actual = expected.clone();
        generated_mut(&mut actual).readback = dense_counters(15);
        let diffs = compare_records(&expected, &actual);
        assert!(
            diffs.iter().any(|d| d.contains("readback")),
            "readback change not reported: {diffs:?}"
        );
    }

    fn fixture_text() -> String {
        let fixture = Fixture {
            schema_version: SCHEMA_VERSION,
            artifact_kind: ARTIFACT_KIND.into(),
            checkpoint: checkpoint(),
            cases: vec![length_record("direct_sampled_dense", dense_counters(15))],
        };
        serde_json::to_string_pretty(&fixture).expect("serializes")
    }

    #[test]
    fn complete_fixture_parses() {
        parse_fixture(&fixture_text()).expect("a complete fixture parses");
    }

    #[test]
    fn fixture_missing_a_path_proof_counter_refuses() {
        let text = fixture_text();
        let line = text
            .lines()
            .find(|line| line.contains("\"decode_compact_candidate_bytes\""))
            .expect("serialized fixture carries the counter");
        let without = text.replacen(&format!("{line}\n"), "", 1);
        assert_ne!(without, text, "the counter line was not removed");
        let err = parse_fixture(&without).unwrap_err();
        assert!(
            err.contains("missing field `decode_compact_candidate_bytes`"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn fixture_with_an_unknown_field_refuses() {
        let text = fixture_text().replacen(
            "\"decode_full_vocab\":",
            "\"decode_full_vocab_legacy\": 0,\n\"decode_full_vocab\":",
            1,
        );
        assert!(parse_fixture(&text).is_err());
    }

    #[test]
    fn all_zero_counters_refuse() {
        let record = length_record("direct_sampled_dense", ReadbackCounters::zero());
        let err =
            check_intent(spec("direct_sampled_dense"), &record.outcome, &checkpoint()).unwrap_err();
        assert!(err.contains("not live"), "unexpected message: {err}");
    }

    #[test]
    fn compact_case_on_the_dense_route_refuses() {
        let record = length_record("direct_sampled_block_topk", dense_counters(15));
        let err = check_intent(
            spec("direct_sampled_block_topk"),
            &record.outcome,
            &checkpoint(),
        )
        .unwrap_err();
        assert!(err.contains("Compact route"), "unexpected message: {err}");
    }

    #[test]
    fn compact_case_on_the_compact_route_passes() {
        let record = length_record("direct_sampled_block_topk", compact_counters(15));
        check_intent(
            spec("direct_sampled_block_topk"),
            &record.outcome,
            &checkpoint(),
        )
        .expect("compact counters satisfy a compact case");
    }

    #[test]
    fn dense_case_on_the_compact_route_refuses() {
        let record = length_record("streaming_greedy_dense", compact_counters(15));
        assert!(
            check_intent(
                spec("streaming_greedy_dense"),
                &record.outcome,
                &checkpoint()
            )
            .is_err()
        );
    }

    #[test]
    fn full_vocab_bytes_must_match_the_vocabulary() {
        let mut counters = dense_counters(15);
        counters.decode_full_vocab_bytes = 15 * 8;
        let record = length_record("direct_sampled_dense", counters);
        let err =
            check_intent(spec("direct_sampled_dense"), &record.outcome, &checkpoint()).unwrap_err();
        assert!(err.contains("bytes"), "unexpected message: {err}");
    }

    #[test]
    fn length_case_that_stopped_early_refuses() {
        let mut record = length_record("direct_sampled_dense", dense_counters(15));
        generated_mut(&mut record).token_ids.truncate(3);
        assert!(
            check_intent(spec("direct_sampled_dense"), &record.outcome, &checkpoint()).is_err()
        );
    }

    #[test]
    fn refusal_case_that_generated_refuses() {
        let record = length_record("direct_logprobs_refused", ReadbackCounters::zero());
        let err = check_intent(
            spec("direct_logprobs_refused"),
            &record.outcome,
            &checkpoint(),
        )
        .unwrap_err();
        assert!(err.contains("refuse"), "unexpected message: {err}");
    }

    #[test]
    fn unset_checkpoint_is_a_skip_and_refuses_under_enforce() {
        assert!(matches!(
            resolve_checkpoint(&lookup_from(&[]), |_| true),
            Ok(Checkpoint::Skip(_))
        ));
        let err = resolve_checkpoint(&lookup_from(&[(ENFORCE_VAR, "1")]), |_| true)
            .err()
            .expect("enforced unset checkpoint refuses");
        assert!(err.contains(ENFORCE_VAR), "unexpected message: {err}");
    }

    #[test]
    fn missing_checkpoint_is_a_skip_and_refuses_under_enforce() {
        assert!(matches!(
            resolve_checkpoint(&lookup_from(&[(MODEL_DIR_VAR, "/abs/missing")]), |_| false),
            Ok(Checkpoint::Skip(_))
        ));
        assert!(
            resolve_checkpoint(
                &lookup_from(&[(MODEL_DIR_VAR, "/abs/missing"), (ENFORCE_VAR, "1")]),
                |_| false
            )
            .is_err()
        );
    }

    #[test]
    fn relative_checkpoint_refuses() {
        assert!(
            resolve_checkpoint(&lookup_from(&[(MODEL_DIR_VAR, "models/qwen")]), |_| true).is_err()
        );
    }

    #[test]
    fn present_absolute_checkpoint_runs() {
        let resolved =
            resolve_checkpoint(&lookup_from(&[(MODEL_DIR_VAR, "/abs/checkpoint")]), |_| {
                true
            })
            .expect("resolves");
        assert!(matches!(resolved, Checkpoint::Run(dir) if dir == Path::new("/abs/checkpoint")));
    }

    #[test]
    fn stale_table_description_refuses() {
        let mut record = length_record("direct_sampled_dense", dense_counters(15));
        record.config = record.config.replace("0.9", "0.95");
        let fixture = Fixture {
            schema_version: SCHEMA_VERSION,
            artifact_kind: ARTIFACT_KIND.into(),
            checkpoint: checkpoint(),
            cases: CASES
                .iter()
                .map(|spec| {
                    if spec.name == "direct_sampled_dense" {
                        record.clone()
                    } else {
                        CaseRecord {
                            name: spec.name.into(),
                            entry: spec.entry,
                            config: spec.description(),
                            outcome: Outcome::Refused(Refused {
                                error: "placeholder".into(),
                                readback: ReadbackCounters::zero(),
                            }),
                        }
                    }
                })
                .collect(),
        };
        let err = check_fixture_matches_table(&fixture).unwrap_err();
        assert!(
            err.contains(
                "case direct_sampled_dense: fixture was captured under a different request"
            ),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn grammar_and_chat_prompts_are_what_the_cases_assume() {
        let schema: serde_json::Value =
            serde_json::from_str(GRAMMAR_SCHEMA).expect("grammar schema is JSON");
        assert_eq!(schema["properties"]["answer"]["type"], "integer");
        assert_eq!(schema["required"][0], "answer");
        for spec in CASES {
            let rendered = spec.prompt.render();
            match spec.prompt {
                Prompt::Plain(text) => assert_eq!(rendered, text),
                Prompt::ChatNoThink(user) => {
                    assert!(rendered.starts_with("<|im_start|>user\n"));
                    assert!(rendered.contains(user));
                    assert!(
                        rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                        "{}: the reply must open after an empty reasoning block",
                        spec.name
                    );
                }
            }
        }
    }

    #[test]
    fn case_table_is_well_formed() {
        let mut names: Vec<&str> = CASES.iter().map(|c| c.name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "case names must be unique");
        for spec in CASES {
            assert!(!spec.name.is_empty());
            assert_eq!(
                spec.stop_on_im_end,
                spec.expect == Expect::StopToken,
                "{}: only the stop-token cases stop on <|im_end|>",
                spec.name
            );
            if spec.reasoning_budget.is_some() {
                // With thinking off the budget is inert, so the case would
                // capture the unbudgeted path under a budgeted name.
                assert!(spec.enable_thinking, "{}", spec.name);
            }
            let prefix = match spec.entry {
                Entry::Direct => "direct_",
                Entry::Streaming => "streaming_",
            };
            assert!(
                spec.name.starts_with(prefix),
                "{} must be prefixed {prefix}",
                spec.name
            );
            if spec.readback == Readback::Compact {
                // The planner routes to the fused compact head only for these
                // widths and exact top-p, with no grammar, logprobs or penalty.
                let (top_k, top_p) = match spec.sampler {
                    Sampler::Greedy => (1, 1.0),
                    Sampler::Seeded { top_k, top_p, .. } => (top_k, top_p),
                };
                assert!(
                    spec.compact_env,
                    "{}: compact route needs the env",
                    spec.name
                );
                assert!([1, 8, 16, 40, 64].contains(&top_k), "{}", spec.name);
                assert_eq!(top_p, 1.0, "{}", spec.name);
                assert!(!spec.grammar && spec.logprobs.is_none(), "{}", spec.name);
            }
        }
        // Coverage the table promises: each entry pins both readback routes for
        // greedy and seeded sampled streams. Direct greedy without the compact
        // route decodes through the zero-copy argmax path, not dense readback.
        for entry in [Entry::Direct, Entry::Streaming] {
            for readback in [Readback::Dense, Readback::Compact] {
                for sampled in [false, true] {
                    let want = if entry == Entry::Direct && readback == Readback::Dense && !sampled
                    {
                        Readback::Argmax
                    } else {
                        readback
                    };
                    assert!(
                        CASES.iter().any(|c| c.entry == entry
                            && c.readback == want
                            && matches!(c.sampler, Sampler::Seeded { .. }) == sampled
                            && c.expect == Expect::Length),
                        "{entry:?} lacks a {want:?} {} case",
                        if sampled { "sampled" } else { "greedy" }
                    );
                }
            }
        }
    }
}
