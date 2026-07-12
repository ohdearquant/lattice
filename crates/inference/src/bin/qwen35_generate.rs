//! Qwen3.5 text generation demo.
//!
//! Usage: cargo run --release --bin qwen35_generate -- [--model-dir PATH] [--prompt "Hello"] [--max-tokens 64] [--seed 42] [--repetition-penalty 1.0]
//!
//! `--emit-phase-events` switches this binary into the flagship CPU
//! load->prefill->decode measurement mode (benchmark-overhaul program, PR 2:
//! "Real CPU flagship smoke", DESIGN.md section 2 "Measurement boundary").
//! It reuses this same production entry point --
//! `Qwen35Model::from_safetensors` + `Qwen35Model::generate_streaming_with_cancel`,
//! the identical CPU generation path `chat_metal`/`lattice_serve`/`lattice.rs`
//! drive in production -- rather than a parallel forward implementation, and
//! prints machine-readable phase-event lines a Python supervisor
//! (`scripts/bench_cpu_flagship_supervisor.py`) parses to compute the five
//! mandated metrics. See that script's module docstring for the sampler and
//! run-record side of the contract.

use lattice_inference::tokenizer::common::Tokenizer as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Shared `--model-dir` / `--model` resolution, used by both the plain demo
/// path and `--emit-phase-events` mode.
fn resolve_model_dir(args: &[String]) -> PathBuf {
    if let Some(dir) = parse_arg(args, "--model-dir") {
        PathBuf::from(dir)
    } else {
        let model_name = parse_arg(args, "--model").unwrap_or_else(|| "qwen3.5-0.8b".to_string());
        std::env::var("LATTICE_MODEL_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").expect("HOME not set");
                PathBuf::from(home).join(".lattice").join("models")
            })
            .join(model_name)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if has_flag(&args, "--emit-phase-events") {
        std::process::exit(run_emit_phase_events(&args));
    }

    let prompt =
        parse_arg(&args, "--prompt").unwrap_or_else(|| "What is the meaning of life?".to_string());

    let max_tokens: usize = parse_arg(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let seed: Option<u64> = parse_arg(&args, "--seed").and_then(|s| s.parse().ok());

    let temperature: Option<f32> = parse_arg(&args, "--temperature").and_then(|s| s.parse().ok());

    let model_dir = resolve_model_dir(&args);

    println!("Loading model from {model_dir:?}...");
    let t0 = Instant::now();

    let model = match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(&model_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    let load_ms = t0.elapsed().as_millis();
    println!("Model loaded in {load_ms}ms\n");

    let mut gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: max_tokens,
        seed,
        ..Default::default()
    };
    if let Some(t) = temperature {
        gen_cfg.temperature = t;
    }
    // GenerateConfig::default() carries a production-serving repetition_penalty
    // of 1.1 (matches chat_metal.rs's own default). A caller doing a strict
    // greedy comparison against a reference implementation that applies no
    // repetition penalty (e.g. HF `model.generate(do_sample=False)`) needs to
    // override this explicitly, or "greedy" silently means two different
    // sampling distributions. See scripts/e2e_parity_check.py, which passes
    // --repetition-penalty 1.0 for exactly this reason.
    if let Some(rp) = parse_arg(&args, "--repetition-penalty").and_then(|s| s.parse().ok()) {
        gen_cfg.repetition_penalty = rp;
    }

    println!("Prompt: {prompt}");
    println!(
        "Config: temp={}, top_k={}, top_p={}, rep_penalty={}, seed={:?}",
        gen_cfg.temperature, gen_cfg.top_k, gen_cfg.top_p, gen_cfg.repetition_penalty, gen_cfg.seed
    );
    println!("Generating up to {max_tokens} tokens...\n");

    let t1 = Instant::now();
    match model.generate(&prompt, &gen_cfg) {
        Ok(output) => {
            let gen_ms = t1.elapsed().as_millis();
            let tokens_per_sec = if gen_ms > 0 {
                output.generated_tokens as f64 / (gen_ms as f64 / 1000.0)
            } else {
                0.0
            };

            println!("--- Generated Text ---");
            println!("{}", output.text);
            println!("--- Stats ---");
            println!("Token IDs: {:?}", output.token_ids);
            println!("Prompt tokens:    {}", output.prompt_tokens);
            println!("Generated tokens: {}", output.generated_tokens);
            println!("Time:             {gen_ms}ms");
            println!("Speed:            {tokens_per_sec:.1} tok/s");
        }
        Err(e) => {
            eprintln!("Generation failed: {e}");
            std::process::exit(1);
        }
    }
}

// ----------------------------------------------------------------------------
// --emit-phase-events: the flagship CPU load -> prefill -> decode adapter
// ----------------------------------------------------------------------------

/// One monotonic marker, printed as a `@@bench `-prefixed JSON line matching
/// `bench_decode_harness.PhaseEvent`'s field shape exactly (`name`,
/// `monotonic_ns`, optional `token_index`) so the Python supervisor can feed
/// captured lines straight into `bench_decode_harness.parse_phase_event`
/// without a second ad hoc parser. `monotonic_ns` is nanoseconds elapsed
/// since this process's own `t0` (the first instruction of
/// `run_emit_phase_events`) -- self-consistent within one child process,
/// which is all `PhaseEvent.monotonic_ns` needs to be (DESIGN.md section 2:
/// "a fresh child" per trial session).
fn emit_phase(t0: Instant, name: &str) {
    println!(
        "@@bench {{\"ev\":\"phase\",\"name\":\"{name}\",\"monotonic_ns\":{}}}",
        t0.elapsed().as_nanos()
    );
    let _ = std::io::stdout().flush();
}

fn emit_token_available(t0: Instant, token_index: usize) {
    println!(
        "@@bench {{\"ev\":\"phase\",\"name\":\"token_available\",\"monotonic_ns\":{},\"token_index\":{token_index}}}",
        t0.elapsed().as_nanos()
    );
    let _ = std::io::stdout().flush();
}

/// Filler text used to build an exact-token-length prompt (DESIGN.md section
/// 2: "Context means exact tokenizer output length, not an approximate
/// character-padded prompt; actual length mismatch invalidates the cell").
/// Ordinary English prose so the byte-level BPE tokenizer produces the
/// same small set of common multi-character tokens the model would see in
/// real serving traffic, not a pathological single-byte-per-token string.
const FILLER_TEXT: &str = "The quick brown fox jumps over the lazy dog while the \
autumn wind carries fallen leaves across the quiet stone courtyard and a \
distant bell rings twice before the evening market finally closes its \
wooden stalls for the night and the old lighthouse keeper climbs the \
spiral stairs to trim the lamp before the fog rolls in from the cold \
northern sea and every ship still out on the water turns slowly toward \
the safety of the sheltered harbor lights";

/// Builds a prompt string that tokenizes to *exactly* `target` real tokens
/// (DESIGN.md section 2). Two-phase search: grow word-by-word until the
/// token count reaches or passes `target` (coarse, few tokenize calls per
/// word), then back off the last word and backfill character-by-character
/// (each byte-level BPE token typically absorbs a small, non-decreasing
/// span of trailing characters, so single-character growth reliably lands
/// on an exact hit). Fails closed (`Err`) rather than returning an
/// approximately-sized prompt if neither phase converges -- an exact
/// context length is part of the profile contract, not a best-effort.
fn build_prompt_of_exact_length(
    model: &lattice_inference::model::qwen35::Qwen35Model,
    target: usize,
) -> Result<String, String> {
    if target == 0 {
        return Err("--context must be a positive token count".to_string());
    }
    let words: Vec<&str> = FILLER_TEXT.split_whitespace().collect();
    let mut text = String::new();
    let max_word_steps = target * 2 + 64;
    for word_idx in 0..max_word_steps {
        let real_len = model.tokenizer().tokenize(&text).real_length;
        if real_len == target {
            return Ok(text);
        }
        if real_len > target {
            break;
        }
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(words[word_idx % words.len()]);
    }

    // Overshot (or never reached target) via whole-word growth: back off the
    // last word and backfill one character at a time.
    if let Some(last_space) = text.rfind(' ') {
        text.truncate(last_space);
    } else {
        text.clear();
    }
    let fill_chars: Vec<char> = FILLER_TEXT.chars().filter(|c| !c.is_whitespace()).collect();
    if fill_chars.is_empty() {
        return Err("internal error: empty filler alphabet".to_string());
    }
    let max_char_steps = target * 4 + 256;
    for char_idx in 0..max_char_steps {
        let real_len = model.tokenizer().tokenize(&text).real_length;
        if real_len == target {
            return Ok(text);
        }
        if real_len > target {
            return Err(format!(
                "overshot exact token target {target} during character-level backfill \
                 (reached {real_len} tokens) -- tokenizer merge behavior at this boundary \
                 is not monotonic for the current filler text; this context point cannot \
                 be built exactly and the cell must be marked unsupported, not approximated"
            ));
        }
        text.push(' ');
        text.push(fill_chars[char_idx % fill_chars.len()]);
    }
    Err(format!(
        "could not reach exact token target {target} after word- and character-level \
         growth (stuck below target) -- this context point cannot be built exactly"
    ))
}

fn run_emit_phase_events(args: &[String]) -> i32 {
    let t0 = Instant::now();
    emit_phase(t0, "load_start");

    if parse_arg(args, "--prompt").is_some() {
        eprintln!(
            "FAIL: --emit-phase-events builds an exact-token-length prompt internally \
             from --context; pass --context <N>, not --prompt"
        );
        return 1;
    }
    let context: usize = match parse_arg(args, "--context").and_then(|s| s.parse().ok()) {
        Some(v) if v >= 1 => v,
        _ => {
            eprintln!("FAIL: --emit-phase-events requires --context <positive integer>");
            return 1;
        }
    };
    let max_tokens: usize = parse_arg(args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let seed: u64 = parse_arg(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let warmup_tokens: usize = parse_arg(args, "--warmup-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let model_dir = resolve_model_dir(args);
    let mut model =
        match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(&model_dir) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("FAIL: could not load model from {model_dir:?}: {e}");
                return 1;
            }
        };
    emit_phase(t0, "backend_ready");

    let prompt = match build_prompt_of_exact_length(&model, context) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL: {e}");
            return 1;
        }
    };
    let actual_prompt_tokens = model.tokenizer().tokenize(&prompt).real_length;
    if actual_prompt_tokens != context {
        eprintln!(
            "FAIL: internal error -- built prompt tokenizes to {actual_prompt_tokens} \
             tokens, expected exactly {context}"
        );
        return 1;
    }

    // Force continuation past EOS / configured stop-token ids so this trial
    // always decodes the exact requested token count. `GenerateConfig` is a
    // plain public-literal struct with a `Default` impl -- a dedicated
    // request-scoped field there would be `constructible_struct_adds_field`
    // under `cargo-semver-checks` (a semver-major break); overriding the
    // model's own post-load `eos_token_id` instead (via `config_mut`, the
    // idiom this crate's own test suite already uses) plus
    // `stop_token_ids: vec![]` below achieves the identical effect through
    // `should_stop_token` (the single shared stop predicate every CPU/Metal
    // decode loop calls) without adding any new public surface to
    // `GenerateConfig`.
    model.config_mut().eos_token_id = u32::MAX;

    // The deterministic profile (DESIGN.md section 2 "initial inference
    // profile"): greedy, EOS disabled (forced fixed-length decode),
    // temperature 0 / top-k 1 / top-p 1 / repetition_penalty 1, thinking and
    // MTP off, no grammar, no string-level stops, no reasoning budget.
    let base_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: max_tokens,
        seed: Some(seed),
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        ..Default::default()
    };

    // One untimed warmup pass after load (DESIGN.md section 2), same prompt
    // and config, `warmup_tokens` generated tokens, discarded, no phase
    // events emitted for it -- distinct from the measured trial below.
    if warmup_tokens > 0 {
        let warmup_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
            max_new_tokens: warmup_tokens,
            ..base_cfg.clone()
        };
        if let Err(e) =
            model.generate_streaming_with_cancel(&prompt, &warmup_cfg, |_delta| true, || false)
        {
            eprintln!("FAIL: untimed warmup failed: {e}");
            return 1;
        }
    }

    // The measured trial: fresh timing region, phase events from here on are
    // the ones the supervisor aggregates into the five mandated metrics.
    //
    // `prefill_end` and `token_available` are emitted from
    // `generate_streaming_with_observer`'s raw-event observer (codex
    // round-1 blocker #1), not from the text-delta callback below: the
    // incremental UTF-8 detokenizer deliberately buffers incomplete
    // multi-byte codepoints, so one text delta is not guaranteed to equal
    // one raw sampled token, and marking `prefill_end` off the first delta
    // would fire it *after* the first sample rather than before it. The
    // text callback (`on_token`) is kept purely for `delta_call_count`
    // disclosure below -- it no longer emits any phase event itself.
    emit_phase(t0, "prefill_start");
    let mut raw_token_count = 0usize;
    let mut delta_call_count = 0usize;
    let result = model.generate_streaming_with_observer(
        &prompt,
        &base_cfg,
        |_delta: &str| {
            delta_call_count += 1;
            true
        },
        || false,
        |evt| match evt {
            lattice_inference::model::qwen35::RawGenEvent::PrefillEnd => {
                emit_phase(t0, "prefill_end");
            }
            lattice_inference::model::qwen35::RawGenEvent::RawToken { index } => {
                raw_token_count += 1;
                emit_token_available(t0, index);
            }
        },
    );

    let output = match result {
        Ok(o) => o,
        Err(e) => {
            eprintln!("FAIL: measured generation failed: {e}");
            return 1;
        }
    };

    // The raw-event observer fires exactly once per token that becomes part
    // of `GenerateOutput` (see `RawGenEvent::RawToken`'s doc comment), so
    // this count is expected to always equal `generated_tokens` by
    // construction; fail loud rather than silently trust it, since the
    // supervisor's phase-sequence validator depends on this invariant
    // holding (codex round-1 blocker #1's adapter-side requirement).
    if raw_token_count != output.generated_tokens {
        eprintln!(
            "FAIL: internal error -- raw phase-event token count ({raw_token_count}) \
             does not match GenerateOutput.generated_tokens ({}); the phase-event \
             trace for this trial cannot be trusted",
            output.generated_tokens
        );
        return 1;
    }

    // Honest disclosure (DESIGN.md's own "possible sub-interval undercount
    // ... is part of the method metadata" standard applied here): the
    // streaming callback fires once per *confirmed UTF-8 text delta*, not
    // once per raw sampled token id -- for byte-level BPE this coincides
    // 1:1 with generated tokens for ordinary text, but is not a hard
    // guarantee when a multi-byte codepoint spans more than one token.
    // Reported here rather than silently assumed. (Distinct from
    // `raw_token_count` above, which is always exact by construction.)
    let delta_matches_generated_tokens = delta_call_count == output.generated_tokens;
    let model_dir_json = json_escape(&model_dir.display().to_string());

    println!(
        "@@bench {{\"ev\":\"summary\",\"prompt_tokens\":{},\"generated_tokens\":{},\
\"requested_max_tokens\":{max_tokens},\"delta_call_count\":{delta_call_count},\
\"delta_matches_generated_tokens\":{delta_matches_generated_tokens},\
\"stopped\":{},\"model_dir\":\"{model_dir_json}\",\"seed\":{seed},\"disable_eos\":true}}",
        output.prompt_tokens, output.generated_tokens, output.stopped,
    );
    let _ = std::io::stdout().flush();
    0
}

/// Minimal JSON string escaping for the one caller-controlled string
/// (`model_dir`, a filesystem path) embedded in the hand-built `@@bench`
/// summary line -- avoids depending on `serde_json` in this binary for a
/// single field.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
