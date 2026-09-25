//! Streaming-profile decode measurement consumer.
//!
//! Drives `MetalQwen35State::generate_streaming_with_cancel` end to end, the
//! entry the serving loops and `generate_streaming` share, and reports load,
//! first-token and decode time separately. It calls only public API, so the
//! same source builds and measures identically on both sides of a change to
//! the streaming decode loop.
//!
//! Arms, each with one untimed warmup and `BENCH_RUNS` measured repetitions:
//!   sampler=greedy   temperature 0, top_k 1
//!   sampler=sampled  temperature 0.8, top_k 40, top_p 1.0, fixed seed
//! each at prompt=short (one sentence) and prompt=long (padded to
//! `BENCH_LONG_PROMPT_TOKENS`). Both samplers are eligible for the compact
//! candidate readback, so `LATTICE_COMPACT_TOPK=1` versus unset selects the
//! compact or the dense arm of every run.
//!
//! Env:
//!   LATTICE_MODEL_DIR         model dir, Q4 or safetensors
//!                             (default ~/.lattice/models/qwen3.5-0.8b)
//!   LATTICE_TOKENIZER_DIR     tokenizer dir (default: LATTICE_MODEL_DIR)
//!   BENCH_N                   completion tokens requested per run (default 128)
//!   BENCH_RUNS                measured repetitions per arm (default 5)
//!   BENCH_LONG_PROMPT_TOKENS  long-prompt length in tokens (default 1024)
//!   BENCH_EXPECT_COMPACT      when set to 1 or 0, requires
//!                             LATTICE_METAL_PATH_PROOF=1 and refuses any run
//!                             whose decode readbacks contradict it
//!
//! Output, one line per measured run:
//!   RESULT profile=streaming sampler=<s> prompt=<p> prompt_tokens=<n> n_req=<n>
//!     completion=<n> ttft_ms=<f> decode_ms=<f> total_ms=<f> readback=<...>
//! plus one `LOAD load_ms=<f>` line. `ttft_ms` is the time to the first
//! `on_token` callback, which includes prefill; `decode_ms` is the remainder.
//!
//! Takes the machine-wide GPU lock itself, so it cannot run under
//! `scripts/bench-command.sh` without `--gpu-handoff`; hold the bench window and
//! let this binary acquire the GPU lock, as for the other self-locking targets.
#![allow(clippy::field_reassign_with_default)]

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_decode_streaming requires macOS + the metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = run() {
            eprintln!("bench_decode_streaming failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse()
            .map_err(|_| format!("{name}={raw:?} is not a non-negative integer")),
        Err(_) => Ok(default),
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::GenerateConfig;
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::model_format::{ModelFormat, detect_format};
    use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};
    use std::time::Instant;

    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    let home = std::env::var("HOME")?;
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir_str);
    let tokenizer_dir_str =
        std::env::var("LATTICE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir_str.clone());
    let tokenizer_path = std::path::Path::new(&tokenizer_dir_str).join("tokenizer.json");

    let n = env_usize("BENCH_N", 128)?;
    let runs = env_usize("BENCH_RUNS", 5)?;
    let long_prompt_tokens = env_usize("BENCH_LONG_PROMPT_TOKENS", 1024)?;
    let expect_compact = match std::env::var("BENCH_EXPECT_COMPACT").as_deref() {
        Err(_) => None,
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        Ok(other) => return Err(format!("BENCH_EXPECT_COMPACT={other:?} must be 1 or 0").into()),
    };
    let path_proof = matches!(
        std::env::var("LATTICE_METAL_PATH_PROOF").as_deref(),
        Ok("1") | Ok("true")
    );
    if expect_compact.is_some() && !path_proof {
        return Err(
            "BENCH_EXPECT_COMPACT needs LATTICE_METAL_PATH_PROOF=1, or the readback counters \
             it checks stay zero"
                .into(),
        );
    }

    let format = detect_format(dir);
    eprintln!("[bench] loading {model_dir_str} ({format:?})");
    let load_start = Instant::now();
    let mut metal = match format {
        ModelFormat::Q4 => {
            let cfg = Qwen35Config::from_model_dir(dir)?;
            MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 4096)
                .map_err(|e| format!("Metal Q4 init: {e}"))?
        }
        ModelFormat::Safetensors => {
            let model = Qwen35Model::from_safetensors(dir)?;
            let cfg = model.config().clone();
            MetalQwen35State::new(model.weights(), &cfg, 4096)
                .map_err(|e| format!("Metal init: {e}"))?
        }
        _ => {
            return Err(format!("{model_dir_str} holds neither safetensors nor Q4 files").into());
        }
    };
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)?;
    println!(
        "LOAD load_ms={:.3}",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    let base = "The quick brown fox jumps over the lazy dog. \
                Once upon a time in a land far away, there lived a wise old owl \
                who knew many secrets. Every morning the sun rose over the \
                mountains and cast long shadows across the quiet valley. ";
    let mut long_prompt = String::new();
    while tokenizer.tokenize(&long_prompt).real_length < long_prompt_tokens {
        long_prompt.push_str(base);
    }
    let prompts = [("short", base.to_string()), ("long", long_prompt)];

    let mut greedy = GenerateConfig::default();
    greedy.max_new_tokens = n;
    greedy.temperature = 0.0;
    greedy.top_k = 1;
    greedy.top_p = 1.0;
    greedy.min_p = 0.0;
    greedy.repetition_penalty = 1.0;
    greedy.seed = Some(42);
    greedy.stop_token_ids = vec![];
    greedy.enable_thinking = false;
    greedy.enable_mtp = Some(false);
    greedy.grammar = None;
    greedy.stop_strings = vec![];
    greedy.reasoning_budget = None;
    greedy.logprobs = None;
    let mut sampled = greedy.clone();
    sampled.temperature = 0.8;
    sampled.top_k = 40;
    sampled.top_p = 1.0;
    sampled.seed = Some(0x5EED);
    let samplers = [("greedy", greedy), ("sampled", sampled)];

    for (sampler_name, gen_cfg) in &samplers {
        for (prompt_name, prompt) in &prompts {
            let prompt_tokens = tokenizer.tokenize(prompt).real_length;
            for run in 0..=runs {
                metal.reset_path_proof_counters();
                let start = Instant::now();
                let mut first_token: Option<f64> = None;
                let output = metal.generate_streaming_with_cancel(
                    prompt,
                    &tokenizer,
                    gen_cfg,
                    |_, _| {
                        if first_token.is_none() {
                            first_token = Some(start.elapsed().as_secs_f64() * 1000.0);
                        }
                        true
                    },
                    || false,
                )?;
                let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                if run == 0 {
                    continue;
                }
                let readback = metal.logit_readback_path_proof_snapshot();
                if let Some(want_compact) = expect_compact {
                    let compact =
                        readback.decode_compact_candidate > 0 && readback.decode_full_vocab == 0;
                    let dense =
                        readback.decode_full_vocab > 0 && readback.decode_compact_candidate == 0;
                    if (want_compact && !compact) || (!want_compact && !dense) {
                        return Err(format!(
                            "sampler={sampler_name} prompt={prompt_name}: decode readbacks \
                             {readback:?} contradict BENCH_EXPECT_COMPACT={}",
                            u8::from(want_compact)
                        )
                        .into());
                    }
                }
                let ttft_ms = first_token.unwrap_or(total_ms);
                let readback_field = if path_proof {
                    format!(
                        "decode_full_vocab:{},decode_compact_candidate:{}",
                        readback.decode_full_vocab, readback.decode_compact_candidate
                    )
                } else {
                    "off".to_string()
                };
                println!(
                    "RESULT profile=streaming sampler={sampler_name} prompt={prompt_name} \
                     prompt_tokens={prompt_tokens} n_req={n} completion={} ttft_ms={ttft_ms:.3} \
                     decode_ms={:.3} total_ms={total_ms:.3} readback={readback_field}",
                    output.generated_tokens,
                    total_ms - ttft_ms,
                );
            }
        }
    }
    Ok(())
}
