//! Issue #1597 measurement cell: "PAIR on comparable output semantics, setup/prefill/decode
//! separately." Replays the stage-5 golden prompt through Gemma 4's fixed-count greedy path
//! ([`Gemma4Model::generate_greedy`]) and its shared-driver path
//! ([`Gemma4Model::generate_streaming_with_cancel`]), asserts the two produce identical token
//! ids, and reports load/prefill/decode time for each. CPU-only: no Metal dispatch, so it does
//! not take `/tmp/lion-metal-gpu-test.lock`.
//!
//! **Both paths run in one process, interleaved ABBA per round** (mirrors
//! `scripts/bench-compare.sh`'s own arm ordering), rather than behind a `--path` flag: the
//! identical-token-id assertion this binary must make can only be done in-process, so there is
//! no scenario where the two paths need separate invocations. ABBA alternates which path goes
//! first each round so a linear drift across the run (thermal ramp, cache effects) lands on both
//! paths rather than favoring one.
//!
//! Timing boundaries -- neither path exposes a real prefill/decode split on the public API, so
//! each is honestly approximated a different way:
//!   - **greedy**: [`Gemma4Model::generate_greedy`] is one blocking call with no per-token hook.
//!     `prefill` times a `max_new_tokens=0` call (prompt forward passes only, same cache size as
//!     the timed run); `decode_total` is a separate `max_new_tokens=n` call's elapsed time minus
//!     `prefill`. This runs the prompt forward pass twice per round and assumes it costs the same
//!     both times (deterministic CPU greedy decode, no RNG). `decode_total / n` includes token 0.
//!   - **driver**: `generate_streaming_with_cancel`'s `on_token` callback fires once per emitted
//!     token, including the first (`decoder::driver::run`'s `check_initial_stop`/
//!     `transition_with_metadata` path feeds the same confirmed-output sink for token 0 and every
//!     later token). Time-to-first-callback is charged as `prefill`; it also includes token 0's
//!     own selection, which has no separate boundary on this API. `decode_total` is the last
//!     callback minus the first, so `decode_total / (n - 1)` excludes token 0.
//!
//! The two paths' per-token decode averages are therefore not directly comparable (greedy
//! includes token 0, driver excludes it) -- both are reported plainly rather than forced to
//! align.
//!
//! **Semantic gap this binary is built to surface, not to paper over**: `generate_greedy` never
//! checks EOS and always emits exactly `n` tokens; the driver path stops early on
//! `config.eos_token_id`. If EOS falls within the requested `n` on the real checkpoint, the two
//! token-id vectors legitimately differ in length and the identical-output assertion below fails
//! -- that is the "comparable output semantics" question the issue asks.
//!
//! Env:
//!   LATTICE_GEMMA4_MODEL_DIR  checkpoint dir (default ~/.lattice/models/gemma-4-e2b-it)
//! Args:
//!   --tokens N     greedy tokens per path per round (default 16)
//!   --repeats N    timed rounds, each round times both paths once (default 5)
//!
//! Output: one `sample` line per (round, path) to stdout; a `setup` line once, before any
//! sample; a `MISMATCH` line to stderr for any round whose two paths disagree.
//!
//! Run it on a quiet machine with the checkpoint present:
//!   scripts/bench-command.sh --label gemma4-driver-pair --durable -- \
//!     cargo run --release -p lattice-inference --features f16 \
//!     --bin bench_gemma4_driver_pair -- --tokens 16 --repeats 5

use lattice_inference::GenerateConfig;
use lattice_inference::model::gemma4_model::Gemma4Model;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(serde::Deserialize)]
struct Golden {
    input_ids: Vec<u32>,
}

const GOLDEN_FIXTURE: &str = include_str!("../../tests/fixtures/gemma4/stage5/e2e_golden.json");

fn model_dir() -> PathBuf {
    let raw = std::env::var("LATTICE_GEMMA4_MODEL_DIR")
        .unwrap_or_else(|_| "~/.lattice/models/gemma-4-e2b-it".to_string());
    match raw.strip_prefix("~/") {
        Some(rest) => {
            let home = std::env::var("HOME")
                .expect("HOME must be set to resolve a ~/ LATTICE_GEMMA4_MODEL_DIR");
            PathBuf::from(home).join(rest)
        }
        None => PathBuf::from(&raw),
    }
}

fn arg_after(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn tokens_arg() -> usize {
    arg_after("--tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16)
}

fn repeats_arg() -> usize {
    arg_after("--repeats")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
}

struct PathTiming {
    prefill: Duration,
    decode_total: Duration,
    /// Tokens the `decode_total` average divides by -- `n` for greedy (includes token 0),
    /// `n.saturating_sub(1)` for driver (excludes it; see module doc).
    decode_token_count: usize,
    token_ids: Vec<u32>,
}

/// Prefill timed by a separate `max_new_tokens=0` call at the SAME `max_seq_len` as the timed
/// run, so both calls allocate an identically sized cache -- see module doc for the assumption
/// this rests on.
fn run_greedy(model: &Gemma4Model, prompt_ids: &[u32], n: usize, max_seq_len: usize) -> PathTiming {
    let t0 = Instant::now();
    model
        .generate_greedy(prompt_ids, 0, max_seq_len)
        .expect("prefill-only greedy call must succeed");
    let prefill = t0.elapsed();

    let t1 = Instant::now();
    let token_ids = model
        .generate_greedy(prompt_ids, n, max_seq_len)
        .expect("greedy generation must succeed");
    let total = t1.elapsed();

    PathTiming {
        prefill,
        decode_total: total.saturating_sub(prefill),
        decode_token_count: n,
        token_ids,
    }
}

fn driver_config(n: usize) -> GenerateConfig {
    let mut cfg = GenerateConfig::default();
    cfg.max_new_tokens = n;
    cfg.temperature = 0.0;
    cfg.repetition_penalty = 1.0;
    cfg.stop_token_ids = vec![];
    cfg
}

fn run_driver(model: &Gemma4Model, prompt_ids: &[u32], n: usize) -> PathTiming {
    let gen_cfg = driver_config(n);
    let mut first: Option<Instant> = None;
    let mut last: Option<Instant> = None;
    let mut calls = 0usize;

    let start = Instant::now();
    let output = model
        .generate_streaming_with_cancel(
            prompt_ids,
            &gen_cfg,
            |_delta| {
                let now = Instant::now();
                first.get_or_insert(now);
                last = Some(now);
                calls += 1;
                true
            },
            || false,
        )
        .expect("driver generation must succeed");

    let first = first.expect("max_new_tokens > 0 must emit at least one token");
    let last = last.unwrap_or(first);
    PathTiming {
        prefill: first.duration_since(start),
        decode_total: last.duration_since(first),
        decode_token_count: calls.saturating_sub(1),
        token_ids: output.token_ids,
    }
}

fn print_sample(round: usize, path: &str, timing: &PathTiming) {
    let per_token = if timing.decode_token_count > 0 {
        timing.decode_total.as_secs_f64() / timing.decode_token_count as f64
    } else {
        0.0
    };
    println!(
        "sample round={round} path={path} prefill_secs={:.6} decode_total_secs={:.6} \
         decode_per_token_secs={per_token:.6} decode_token_count={}",
        timing.prefill.as_secs_f64(),
        timing.decode_total.as_secs_f64(),
        timing.decode_token_count
    );
}

fn main() {
    // Debug-build CPU timing is meaningless. Refused at run time rather than
    // compile time so `cargo clippy --all-targets --features f16` still builds it.
    if cfg!(debug_assertions) {
        eprintln!(
            "bench_gemma4_driver_pair: build with --release; debug CPU timing is meaningless"
        );
        std::process::exit(2);
    }
    let n = tokens_arg();
    let repeats = repeats_arg().max(1);
    let dir = model_dir();

    let load_start = Instant::now();
    let model = Gemma4Model::from_safetensors(&dir)
        .unwrap_or_else(|e| panic!("failed to load checkpoint at {}: {e}", dir.display()));
    let load_secs = load_start.elapsed().as_secs_f64();
    println!("setup path=load dir={} secs={load_secs:.6}", dir.display());

    let golden: Golden = serde_json::from_str(GOLDEN_FIXTURE).expect("golden fixture parses");
    let prompt_ids = golden.input_ids;
    let max_seq_len = prompt_ids.len() + n;

    // Untimed warm-up, one call per path, discarded -- first-call allocation and page-in costs
    // must not land in the timed samples below.
    let _ = model.generate_greedy(&prompt_ids, n, max_seq_len);
    let warm_cfg = driver_config(n);
    let _ = model.generate_streaming_with_cancel(&prompt_ids, &warm_cfg, |_| true, || false);

    let mut mismatches = 0usize;
    for round in 0..repeats {
        let (greedy, driver) = if round % 2 == 0 {
            let g = run_greedy(&model, &prompt_ids, n, max_seq_len);
            let d = run_driver(&model, &prompt_ids, n);
            (g, d)
        } else {
            let d = run_driver(&model, &prompt_ids, n);
            let g = run_greedy(&model, &prompt_ids, n, max_seq_len);
            (g, d)
        };

        if greedy.token_ids != driver.token_ids {
            mismatches += 1;
            eprintln!(
                "MISMATCH round={round} greedy={:?} driver={:?}",
                greedy.token_ids, driver.token_ids
            );
        }

        print_sample(round, "greedy", &greedy);
        print_sample(round, "driver", &driver);
    }

    if mismatches > 0 {
        eprintln!(
            "FAILED: {mismatches}/{repeats} round(s) produced different token ids between the \
             greedy and driver paths (see module doc: generate_greedy ignores EOS, the driver \
             path honors it)"
        );
        std::process::exit(1);
    }
}
