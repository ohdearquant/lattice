//! D_1 measurement harness (MTP K=1 draft-cost anchor).
//!
//! Implements the frozen pre-registration methodology: measures, on the real
//! `enable_mtp` decode path, T_target (plain greedy decode step, MTP off),
//! T_draft (one MTP draft-token forward), and T_verify (one K=1 verification
//! step), n=5 independent runs per quantity, arms interleaved A/B/A/B/... in
//! one process to cancel thermal drift. Prints per-run individual values
//! (never aggregates only) plus the Gate 0 / Gate 1 outcome.
//!
//! Real decode path: T_target is measured via the public, unmodified
//! `MetalQwen35State::forward_step` (identical call the plain greedy loop in
//! `generate()` makes per step). T_draft/T_verify are measured via
//! `d1_mtp_draft_step`/`d1_mtp_verify_step`, two thin bench-internals-gated
//! pass-throughs added to `forward/metal_qwen35.rs` that call the exact same
//! private `mtp_forward_one`/`verify_tokens_batched` methods
//! `generate_greedy_mtp` calls in its draft/verify phases — no new timing
//! code inside the library, only bench-only visibility (same pattern as the
//! pre-existing `bench_support` module).
//!
//! Env:
//!   D1_MODEL_DIR       Q4 model dir (default ~/.lattice/models/_fresh616_q4)
//!   D1_TOKENIZER_DIR   tokenizer dir (default ~/.lattice/models/qwen3.5-0.8b)
//!   D1_RUNS            independent runs per quantity (default 5; GATES.md n=5 is frozen —
//!                       only override for a dry run, never for the registered measurement)
//!   D1_STEPS           measured steps per run (default 200; must be >= 100 per GATES.md
//!                       T_target definition — the binary refuses to run below that floor)
//!   D1_WARMUP_STEPS    warmup steps discarded before measurement starts (default 20; this
//!                       is the "exact warmup count pinned in the run script before first
//!                       run" GATES.md requires — the seat script pins it explicitly via
//!                       this env var rather than leaving a library default implicit)
//!   D1_SEED            recorded for provenance only (default 42; decode here is greedy —
//!                       deterministic given a fixed prompt — so this does not change any
//!                       measured value, only the CONFIG line)
//!   D1_PROMPT_TOKENS   optional: pad the fixed prompt to at least this many tokens
//!
//! Output (stdout, one line per run per metric; stable prefixes for parsing):
//!   CONFIG ...
//!   RUN metric=T_target arm=A seq=<n> run=<r> mean_ns=<f> steps=<n> warmup=<n>
//!   RUN metric=T_draft  arm=B seq=<n> run=<r> mean_ns=<f> steps=<n> warmup=<n>
//!   RUN metric=T_verify arm=B seq=<n> run=<r> mean_ns=<f> steps=<n> warmup=<n>
//!   GATE0 spread=<f> median_target_ns=<f> pass=<bool>
//!   GATE1 T_target_ns=<f> T_draft_ns=<f> T_verify_ns=<f> D1=<f> C1=<f> Smax=<f> branch=<A|B>
//!   STOP reason=<...>   (in place of GATE0/GATE1 on any STOP condition)

/// Zero-copy argmax over a full-vocab logits slice. First-tied-wins, matching
/// `crate::sampling::argmax_f32_first_wins`'s contract (pure, testable without a model).
fn argmax_f32(logits: &[f32]) -> u32 {
    let mut best_id = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i as u32;
        }
    }
    best_id
}

fn mean_ns(samples: &[u64]) -> f64 {
    samples.iter().copied().sum::<u64>() as f64 / samples.len() as f64
}

/// Median of a small f64 slice. Does not mutate the caller's slice.
fn median_f64(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite timing values"));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Gate 0: relative spread (max-min)/median of the 5 T_target run means.
fn relative_spread(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    (max - min) / median_f64(values)
}

/// D_1 := T_draft / T_target (token-equivalents).
fn compute_d1(t_draft: f64, t_target: f64) -> f64 {
    t_draft / t_target
}

/// C_1 := (T_draft + T_verify) / T_target.
fn compute_c1(t_draft: f64, t_verify: f64, t_target: f64) -> f64 {
    (t_draft + t_verify) / t_target
}

/// S_max = 2 / C_1_measured (maximum attainable K=1 speedup at alpha = 1).
fn compute_smax(c1: f64) -> f64 {
    2.0 / c1
}

#[derive(Debug, PartialEq, Eq)]
enum Gate1Branch {
    /// S_max < 1.10: K=1 MTP promotion lane is DEAD with a measured anchor.
    ADead,
    /// S_max >= 1.10: measure alpha on the registered workload (out of scope
    /// for this harness; a separate follow-on measurement).
    BMeasureAlpha,
}

fn gate1_branch(smax: f64) -> Gate1Branch {
    if smax < 1.10 {
        Gate1Branch::ADead
    } else {
        Gate1Branch::BMeasureAlpha
    }
}

fn main() {
    #[cfg(not(all(
        target_os = "macos",
        feature = "metal-gpu",
        feature = "bench-internals"
    )))]
    {
        eprintln!("Requires macOS + metal-gpu + bench-internals features.");
        std::process::exit(1);
    }

    #[cfg(all(
        target_os = "macos",
        feature = "metal-gpu",
        feature = "bench-internals"
    ))]
    {
        if let Err(e) = run() {
            eprintln!("d1_profile failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};

    let home = std::env::var("HOME")?;
    let model_dir_str = std::env::var("D1_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/_fresh616_q4"));
    let tokenizer_dir_str = std::env::var("D1_TOKENIZER_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir_str);
    let tokenizer_dir = std::path::Path::new(&tokenizer_dir_str);

    let runs: usize = std::env::var("D1_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let steps: usize = std::env::var("D1_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    if steps < 100 {
        return Err(format!(
            "D1_STEPS={steps} below the GATES.md T_target floor of >= 100 steps per run; refusing to run"
        )
        .into());
    }
    let warmup: usize = std::env::var("D1_WARMUP_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let seed: u64 = std::env::var("D1_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let cfg = Qwen35Config::from_model_dir(dir).map_err(|e| format!("config.json load: {e}"))?;
    let mut metal =
        MetalQwen35State::from_q4_dir(dir, &tokenizer_dir.join("tokenizer.json"), &cfg, 4096)
            .map_err(|e| format!("Metal Q4 init: {e}"))?;
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_dir.join("tokenizer.json"))?;

    let has_mtp = metal.has_mtp();

    let base = "The quick brown fox jumps over the lazy dog. \
                Once upon a time in a land far away, there lived a wise old owl \
                who knew many secrets. Every morning the sun rose over the \
                mountains and cast long shadows across the quiet valley. ";
    let prompt: String = match std::env::var("D1_PROMPT_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(target) if target > 0 => {
            let mut p = String::new();
            while tokenizer.tokenize(&p).real_length < target {
                p.push_str(base);
            }
            p
        }
        _ => base.to_string(),
    };
    let tokenized = tokenizer.tokenize(&prompt);
    let prompt_ids: Vec<u32> = tokenized.input_ids[..tokenized.real_length].to_vec();

    println!(
        "CONFIG model_dir={model_dir_str} tokenizer_dir={tokenizer_dir_str} runs={runs} \
         steps={steps} warmup_steps={warmup} seed={seed} has_mtp={has_mtp} \
         prompt_tokens={}",
        prompt_ids.len()
    );

    if !has_mtp {
        println!(
            "STOP reason=checkpoint_has_no_mtp_weights; Gate 0/Gate 1 require T_draft/T_verify \
             from the live enable_mtp path, which this checkpoint does not carry"
        );
        return Ok(());
    }

    let mut target_run_means_ns: Vec<f64> = Vec::with_capacity(runs);
    let mut draft_run_means_ns: Vec<f64> = Vec::with_capacity(runs);
    let mut verify_run_means_ns: Vec<f64> = Vec::with_capacity(runs);

    let mut seq = 0usize;

    for run in 0..runs {
        // --- Arm A: target-only decode (MTP off) ---
        metal.reset_state();
        let prefill_logits = metal.forward_prefill(&prompt_ids);
        let mut pos = prompt_ids.len();
        let mut token = argmax_f32(&prefill_logits);

        for _ in 0..warmup {
            let logits = metal.forward_step(token, pos);
            token = argmax_f32(&logits);
            pos += 1;
        }

        let mut target_ns = Vec::with_capacity(steps);
        for _ in 0..steps {
            let t = std::time::Instant::now();
            let logits = metal.forward_step(token, pos);
            target_ns.push(t.elapsed().as_nanos() as u64);
            token = argmax_f32(&logits);
            pos += 1;
        }
        let target_mean = mean_ns(&target_ns);
        target_run_means_ns.push(target_mean);
        seq += 1;
        println!(
            "RUN metric=T_target arm=A seq={seq} run={run} mean_ns={target_mean:.1} steps={steps} warmup={warmup}"
        );

        // --- Arm B: MTP draft + K=1 verify (interleaved immediately after arm A) ---
        metal.reset_state();
        let prefill_logits = metal.forward_prefill(&prompt_ids);
        let mut pos = prompt_ids.len();
        let mut pending = argmax_f32(&prefill_logits);

        for _ in 0..warmup {
            let draft = metal.d1_mtp_draft_step(pending, pos);
            pending = metal
                .d1_mtp_verify_step(pending, draft, pos)
                .map_err(|e| format!("verify (warmup) failed: {e}"))?;
            pos += 2;
        }

        let mut draft_ns = Vec::with_capacity(steps);
        let mut verify_ns = Vec::with_capacity(steps);
        for _ in 0..steps {
            let t0 = std::time::Instant::now();
            let draft = metal.d1_mtp_draft_step(pending, pos);
            draft_ns.push(t0.elapsed().as_nanos() as u64);

            let t1 = std::time::Instant::now();
            pending = metal
                .d1_mtp_verify_step(pending, draft, pos)
                .map_err(|e| format!("verify failed: {e}"))?;
            verify_ns.push(t1.elapsed().as_nanos() as u64);

            pos += 2;
        }
        let draft_mean = mean_ns(&draft_ns);
        let verify_mean = mean_ns(&verify_ns);
        draft_run_means_ns.push(draft_mean);
        verify_run_means_ns.push(verify_mean);
        seq += 1;
        println!(
            "RUN metric=T_draft arm=B seq={seq} run={run} mean_ns={draft_mean:.1} steps={steps} warmup={warmup}"
        );
        println!(
            "RUN metric=T_verify arm=B seq={seq} run={run} mean_ns={verify_mean:.1} steps={steps} warmup={warmup}"
        );
    }

    let spread = relative_spread(&target_run_means_ns);
    let gate0_pass = spread < 0.05;
    let median_target_ns = median_f64(&target_run_means_ns);
    println!("GATE0 spread={spread:.4} median_target_ns={median_target_ns:.1} pass={gate0_pass}");

    if !gate0_pass {
        println!(
            "STOP reason=gate0_spread_ge_5pct; measurement invalid per GATES.md; \
             no Gate 1 decision may be taken"
        );
        return Ok(());
    }

    let t_target = median_target_ns;
    let t_draft = median_f64(&draft_run_means_ns);
    let t_verify = median_f64(&verify_run_means_ns);
    let d1 = compute_d1(t_draft, t_target);
    let c1 = compute_c1(t_draft, t_verify, t_target);
    let smax = compute_smax(c1);
    let branch = match gate1_branch(smax) {
        Gate1Branch::ADead => "A",
        Gate1Branch::BMeasureAlpha => "B",
    };
    println!(
        "GATE1 T_target_ns={t_target:.1} T_draft_ns={t_draft:.1} T_verify_ns={t_verify:.1} \
         D1={d1:.4} C1={c1:.4} Smax={smax:.4} branch={branch}"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_f32_picks_maximum() {
        assert_eq!(argmax_f32(&[1.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn argmax_f32_first_tied_wins() {
        assert_eq!(argmax_f32(&[2.0, 5.0, 5.0, 1.0]), 1);
    }

    #[test]
    fn argmax_f32_empty_returns_zero() {
        assert_eq!(argmax_f32(&[]), 0);
    }

    #[test]
    fn mean_ns_basic() {
        assert_eq!(mean_ns(&[100, 200, 300]), 200.0);
    }

    #[test]
    fn median_f64_odd_count() {
        assert_eq!(median_f64(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn median_f64_even_count_averages_middle_two() {
        assert_eq!(median_f64(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn median_f64_does_not_mutate_input_order() {
        let values = [3.0, 1.0, 2.0];
        let _ = median_f64(&values);
        assert_eq!(values, [3.0, 1.0, 2.0]);
    }

    #[test]
    fn relative_spread_zero_when_all_equal() {
        assert_eq!(relative_spread(&[10.0, 10.0, 10.0, 10.0, 10.0]), 0.0);
    }

    #[test]
    fn relative_spread_matches_max_minus_min_over_median() {
        // values: 90,95,100,105,120 -> median 100, spread (120-90)/100 = 0.30
        let s = relative_spread(&[100.0, 90.0, 120.0, 95.0, 105.0]);
        assert!((s - 0.30).abs() < 1e-9, "got {s}");
    }

    #[test]
    fn gate0_boundary_just_under_5pct_passes() {
        // median 100, spread must be < 5 -> max-min < 5.0
        let values = [100.0, 100.0, 100.0, 100.0, 104.9];
        let s = relative_spread(&values);
        assert!(s < 0.05, "spread {s} should be < 0.05");
    }

    #[test]
    fn gate0_boundary_at_5pct_does_not_pass() {
        // spread exactly 0.05 must NOT pass: GATES.md gate is "< 5%", so
        // "== 5%" falls into the ">= 5%" STOP branch.
        let values = [100.0, 100.0, 100.0, 100.0, 105.0];
        let s = relative_spread(&values);
        assert!((s - 0.05).abs() < 1e-9);
        assert!(!(s < 0.05));
    }

    #[test]
    fn compute_d1_matches_definition() {
        assert_eq!(compute_d1(55.0, 100.0), 0.55);
    }

    #[test]
    fn compute_c1_matches_definition() {
        // packet 0298's inferred anchor: D_1=0.55, C_1=1.90 given T_verify/T_target=1.35
        let c1 = compute_c1(55.0, 135.0, 100.0);
        assert!((c1 - 1.90).abs() < 1e-9, "got {c1}");
    }

    #[test]
    fn compute_smax_matches_definition() {
        let smax = compute_smax(1.90);
        assert!((smax - (2.0 / 1.90)).abs() < 1e-9, "got {smax}");
    }

    #[test]
    fn gate1_branch_a_when_smax_below_threshold() {
        assert_eq!(gate1_branch(1.05), Gate1Branch::ADead);
    }

    #[test]
    fn gate1_branch_b_at_exact_threshold() {
        // GATES.md: "Branch B: S_max >= 1.10" -- the boundary itself is Branch B,
        // not Branch A ("< 1.10").
        assert_eq!(gate1_branch(1.10), Gate1Branch::BMeasureAlpha);
    }

    #[test]
    fn gate1_branch_b_above_threshold() {
        assert_eq!(gate1_branch(1.50), Gate1Branch::BMeasureAlpha);
    }

    #[test]
    fn packet_0298_inferred_anchor_would_have_been_branch_a() {
        // Sanity cross-check against the packet's own inferred numbers
        // (D_1=0.55, C_1=1.90 -> S_max ~= 1.0526), confirming this harness's
        // Gate 1 arithmetic reproduces the packet's own DEAD verdict when fed
        // the packet's inferred (not measured) anchor.
        let c1 = compute_c1(55.0, 135.0, 100.0);
        let smax = compute_smax(c1);
        assert_eq!(gate1_branch(smax), Gate1Branch::ADead);
    }
}
