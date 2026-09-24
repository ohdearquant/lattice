//! Issue #1597 measurement cell: "PAIR on comparable output semantics, setup/prefill/decode
//! separately." Replays the stage-5 golden prompt through Gemma 4's fixed-count greedy path
//! ([`Gemma4Model::generate_greedy`]) and its shared-driver path
//! ([`Gemma4Model::generate_streaming_with_cancel`]) at the SAME token count and asserts the two
//! produce identical token ids. CPU-only: no Metal dispatch, so it does not take
//! `/tmp/lion-metal-gpu-test.lock`.
//!
//! **The driver stops itself at the model's end-of-turn id; the greedy path does not check EOS
//! at all and always emits exactly the count it is given.** So the token count comparable output
//! semantics can be judged over is prompt-determined, not a number this binary picks. For the
//! stage-5 golden prompt ("The capital of France is"), the model's stop set unions
//! `text_config.eos_token_id` with the checkpoint's `generation_config.json` (or, if that file is
//! absent, top-level `config.json`) `eos_token_id` -- see
//! `model::gemma4_config::resolve_stop_token_ids` -- so the driver stops after 2 tokens
//! (`[9079, 236761]`), before the end-of-turn id 106.
//!
//! Each repeat therefore runs an UNTIMED driver call first (capped at `--tokens`, default 16) to
//! learn that repeat's `N = driver.token_ids.len()`, then a greedy call with
//! `max_new_tokens = N` at `max_seq_len = prompt_len + N` -- equal work, so the id comparison is
//! exact rather than an artifact of one path stopping earlier than the other. This costs one
//! extra, untimed generation per repeat.
//!
//! **Both paths run in one process; the TIMED calls alternate ABBA per repeat** (mirrors
//! `scripts/bench-compare.sh`'s own arm ordering) so a linear drift across the run (thermal
//! ramp, cache effects) lands on both paths rather than favoring one. The untimed N-discovery
//! call always runs first and is not part of that alternation -- it has to, since neither timed
//! call can start before `N` is known.
//!
//! **Timing**: each path's number is the WHOLE call's wall time at that repeat's `N` --
//! `generate_streaming_with_cancel` for the driver, `generate_greedy` for greedy. Neither path
//! exposes a real prefill/decode split on the public API. `generate_greedy`'s own separate
//! `max_new_tokens=0` call (prompt forward passes only, same cache size as the timed call) is
//! reported alongside as an informational reference figure, not subtracted from anything. The
//! driver's `on_token` callback count is also reported as information only: it does not fire
//! once per emitted token (special tokens can decode to an empty text delta and the callback is
//! not guaranteed to fire on every one), so it is not a decode-per-token instrument.
//!
//! Env:
//!   LATTICE_GEMMA4_MODEL_DIR  checkpoint dir (default ~/.lattice/models/gemma-4-e2b-it)
//! Args:
//!   --tokens N     cap for the untimed N-discovery driver call (default 16)
//!   --repeats N    timed repeats, each fixing its own N and timing both paths once (default 5)
//!
//! Output: one `sample` line per (repeat, path) to stdout; a `setup` line once, before any
//! sample; a `MISMATCH` line to stderr for any repeat whose two paths disagree at equal N.
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

fn driver_config(n: usize) -> GenerateConfig {
    let mut cfg = GenerateConfig::default();
    cfg.max_new_tokens = n;
    cfg.temperature = 0.0;
    cfg.repetition_penalty = 1.0;
    cfg.stop_token_ids = vec![];
    cfg
}

/// Runs the driver path once, untimed, capped at `token_cap`, and returns however many ids it
/// actually produced -- the driver stops itself at the model's end-of-turn id, so this is
/// usually well under `token_cap` (see module doc).
fn discover_n(model: &Gemma4Model, prompt_ids: &[u32], token_cap: usize) -> usize {
    let gen_cfg = driver_config(token_cap);
    let output = model
        .generate_streaming_with_cancel(prompt_ids, &gen_cfg, |_delta| true, || false)
        .expect("driver generation must succeed");
    output.token_ids.len()
}

struct GreedyTiming {
    total: Duration,
    token_ids: Vec<u32>,
}

fn time_greedy(
    model: &Gemma4Model,
    prompt_ids: &[u32],
    n: usize,
    max_seq_len: usize,
) -> GreedyTiming {
    let start = Instant::now();
    let token_ids = model
        .generate_greedy(prompt_ids, n, max_seq_len)
        .expect("greedy generation must succeed");
    GreedyTiming {
        total: start.elapsed(),
        token_ids,
    }
}

/// Prompt-forward-passes-only timing at the same cache size as the timed call; informational,
/// not subtracted from `time_greedy`'s total (see module doc).
fn time_greedy_prefill_only(
    model: &Gemma4Model,
    prompt_ids: &[u32],
    max_seq_len: usize,
) -> Duration {
    let start = Instant::now();
    model
        .generate_greedy(prompt_ids, 0, max_seq_len)
        .expect("prefill-only greedy call must succeed");
    start.elapsed()
}

struct DriverTiming {
    total: Duration,
    /// Informational only -- not a per-token decode instrument (see module doc).
    on_token_calls: usize,
    token_ids: Vec<u32>,
}

fn time_driver(model: &Gemma4Model, prompt_ids: &[u32], n: usize) -> DriverTiming {
    let gen_cfg = driver_config(n);
    let mut on_token_calls = 0usize;
    let start = Instant::now();
    let output = model
        .generate_streaming_with_cancel(
            prompt_ids,
            &gen_cfg,
            |_delta| {
                on_token_calls += 1;
                true
            },
            || false,
        )
        .expect("driver generation must succeed");
    DriverTiming {
        total: start.elapsed(),
        on_token_calls,
        token_ids: output.token_ids,
    }
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
    let token_cap = tokens_arg();
    let repeats = repeats_arg().max(1);
    let dir = model_dir();

    let load_start = Instant::now();
    let model = Gemma4Model::from_safetensors(&dir)
        .unwrap_or_else(|e| panic!("failed to load checkpoint at {}: {e}", dir.display()));
    let load_secs = load_start.elapsed().as_secs_f64();
    println!("setup path=load dir={} secs={load_secs:.6}", dir.display());

    let golden: Golden = serde_json::from_str(GOLDEN_FIXTURE).expect("golden fixture parses");
    let prompt_ids = golden.input_ids;

    // Untimed warm-up, one call per path at `token_cap`, discarded -- first-call allocation and
    // page-in costs must not land in the timed samples below.
    let warmup_max_seq_len = prompt_ids.len() + token_cap;
    let _ = model.generate_greedy(&prompt_ids, token_cap, warmup_max_seq_len);
    let warm_cfg = driver_config(token_cap);
    let _ = model.generate_streaming_with_cancel(&prompt_ids, &warm_cfg, |_| true, || false);

    let mut mismatches = 0usize;
    for repeat in 0..repeats {
        // Untimed: fixes this repeat's N. Always first -- neither timed call below can start
        // before N is known, so it cannot take part in the ABBA alternation.
        let repeat_n = discover_n(&model, &prompt_ids, token_cap);
        let repeat_max_seq_len = prompt_ids.len() + repeat_n;

        let (greedy, driver) = if repeat % 2 == 0 {
            let g = time_greedy(&model, &prompt_ids, repeat_n, repeat_max_seq_len);
            let d = time_driver(&model, &prompt_ids, repeat_n);
            (g, d)
        } else {
            let d = time_driver(&model, &prompt_ids, repeat_n);
            let g = time_greedy(&model, &prompt_ids, repeat_n, repeat_max_seq_len);
            (g, d)
        };

        if greedy.token_ids != driver.token_ids {
            mismatches += 1;
            eprintln!(
                "MISMATCH repeat={repeat} n={repeat_n} greedy={:?} driver={:?}",
                greedy.token_ids, driver.token_ids
            );
        }

        let prefill_only = time_greedy_prefill_only(&model, &prompt_ids, repeat_max_seq_len);

        println!(
            "sample repeat={repeat} path=greedy n={repeat_n} total_secs={:.6} \
             prefill_only_informational_secs={:.6}",
            greedy.total.as_secs_f64(),
            prefill_only.as_secs_f64()
        );
        println!(
            "sample repeat={repeat} path=driver n={repeat_n} total_secs={:.6} \
             on_token_calls_informational={}",
            driver.total.as_secs_f64(),
            driver.on_token_calls
        );
    }

    if mismatches > 0 {
        eprintln!(
            "FAILED: {mismatches}/{repeats} repeat(s) produced different token ids between the \
             greedy and driver paths at equal N -- both ran the same token count, so this is a \
             real decode divergence, not a stopping-condition mismatch"
        );
        std::process::exit(1);
    }
}
