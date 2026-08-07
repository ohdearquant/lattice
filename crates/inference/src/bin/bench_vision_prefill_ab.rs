//! Issue #1336 multimodal-prefill A/B measurement harness.
//!
//! Measures Metal M-RoPE multimodal prefill wall time on a real checkpoint
//! with two arms that differ ONLY in one runtime flag, never in source or
//! binary:
//!   - baseline (`always_emit_head=true`): terminal RMSNorm + lm_head
//!     dispatched at every prefill position — the pre-#1336 shape.
//!   - optimized (`always_emit_head=false`): terminal head dispatched only
//!     at the final prefill position — the shipped
//!     `generate_multimodal_vision_impl` behavior (`emit_head = pos ==
//!     last_prefill_pos`).
//!
//! Both arms call `bench_support::run_multimodal_prefill`
//! (`forward/metal_qwen35.rs`), which drives the exact same
//! `forward_step_mrope`/`forward_step_injected_mrope` functions the
//! production path calls. There is no recompile between arms: one process,
//! one binary, the flag is a function argument selected per repeat.
//!
//! What this harness does NOT measure: ViT/merger (vision-encoder) cost.
//! `post_merger_rows` are synthetic, checkpoint-shape-correct values, not a
//! real decoded image — the emit_head optimization is entirely inside the
//! decoder's per-position forward step, downstream of and insensitive to
//! merger output content, only to its shape (hidden_size, row count). A
//! number from this harness is "decoder multimodal-prefill time", not
//! "end-to-end multimodal request latency"; do not quote it as the latter.
//!
//! Declared confounds this harness does NOT control for:
//!   - Shared-machine contention: other processes competing for the GPU,
//!     CPU, or memory bandwidth bias wall-clock time in either direction.
//!     Run under the fleet's exclusive heavy-lane + GPU lock discipline.
//!   - Thermal state: sustained runs on a laptop/mini can throttle Metal
//!     clocks mid-run; the first repeat after a cold start and the last
//!     repeat of a long run are not directly comparable.
//!   - Command-buffer / driver warm-up: the single untimed warmup repeat
//!     below reduces but does not eliminate first-dispatch overhead
//!     (pipeline-state compilation, allocator warm-up).
//!   - OS scheduler noise on the CPU side of each `wait_until_completed`.
//!   - This is single-request prefill only — no batching, no concurrent
//!     requests, no KV-cache reuse across requests.
//!
//! Env:
//!   LATTICE_MODEL_DIR    model dir (default ~/.lattice/models/qwen3.5-0.8b),
//!                        must be a real vision-capable checkpoint
//!                        (safetensors format; Q4 vision loading is out of
//!                        scope for this harness)
//!   BENCH_VISUAL_ROWS    target post-merger visual row count (default 1024;
//!                        actual count is rounded up to the next square grid
//!                        and printed)
//!   BENCH_WARMUP         untimed warmup repeats per arm (default 1)
//!   BENCH_REPEATS        timed repeats per arm (default 5)
//!
//! Output: one `RESULT` line per (arm, repeat) to stdout, plus a
//! `SUMMARY` line with median baseline/optimized/delta, all parseable
//! without scraping prose.
//!
//! Operator command (run elsewhere, on real hardware — this environment is
//! prohibited from executing it):
//!   cargo run --release -p lattice-inference --features f16,metal-gpu,bench-internals \
//!     --bin bench_vision_prefill_ab
//! with LATTICE_MODEL_DIR pointed at the checkpoint if it is not at the
//! default path.

#[cfg(not(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
)))]
fn main() {
    eprintln!("Requires macOS + metal-gpu + bench-internals features.");
    std::process::exit(1);
}

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
fn main() {
    if let Err(e) = run() {
        eprintln!("bench_vision_prefill_ab failed: {e}");
        std::process::exit(1);
    }
}

// Debug-build Metal timing is meaningless (unoptimized dispatch encoding,
// debug assertions on the hot path) — fail to build rather than produce a
// number nobody should trust. Compile-time check because
// `debug_assertions` is a compile-time constant.
#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals",
    debug_assertions
))]
compile_error!("bench_vision_prefill_ab: MUST build --release (debug Metal timing is meaningless)");

/// Rejects `repeats == 0` up front. Left unchecked, it leaves both timing
/// vectors empty and [`median`]'s `v[v.len() / 2]` indexing panics deep
/// inside the measurement loop — after the (slow) model load has already
/// paid its cost. Fail with an operator-facing message before that point
/// instead.
fn validate_repeats(repeats: usize) -> Result<(), String> {
    if repeats == 0 {
        return Err(
            "BENCH_REPEATS=0 would leave both timing vectors empty; set BENCH_REPEATS to a \
             positive integer (default 5)"
                .to_string(),
        );
    }
    Ok(())
}

/// Sorts `values` in place and returns the median. Returns `Err` instead of
/// panicking on an empty slice — [`validate_repeats`] is what keeps `run()`
/// from ever reaching this on an empty vector, but the helper does not rely
/// on that caller discipline to stay panic-free.
fn median(values: &mut [f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("median: cannot compute a median of an empty timing vector".to_string());
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in timing data"));
    Ok(values[values.len() / 2])
}

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::forward::metal_qwen35::bench_support;
    use lattice_inference::model::qwen35::Qwen35Model;

    eprintln!("[bench] acquiring shared Metal GPU lock ...");
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    let home = std::env::var("HOME")?;
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir_str);

    let visual_rows_target: usize = std::env::var("BENCH_VISUAL_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let repeats: usize = std::env::var("BENCH_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    validate_repeats(repeats)?;

    eprintln!("[bench] loading {model_dir_str} (safetensors)");
    let model = Qwen35Model::from_safetensors(dir).map_err(|e| format!("load model: {e}"))?;
    let cfg = model.config().clone();

    let request = bench_support::vision_prefill_fixture(&cfg, visual_rows_target, 0.02)?;
    let prompt_len = request.input_ids.len();
    let visual_rows_actual = request.image_pad_count();
    // Generous cache headroom above the prefill length; no decode steps run.
    let cache_len = (prompt_len + 256).max(4096);

    eprintln!(
        "[bench] visual_rows_target={visual_rows_target} visual_rows_actual={visual_rows_actual} \
         prompt_tokens={prompt_len} cache_len={cache_len}"
    );

    let mut state = MetalQwen35State::new(model.weights(), &cfg, cache_len)
        .map_err(|e| format!("init metal: {e}"))?;

    let run_arm = |state: &mut MetalQwen35State, always_emit_head: bool| -> Result<f64, String> {
        state.reset_state();
        let start = std::time::Instant::now();
        let logits =
            bench_support::run_multimodal_prefill(state, &request, &cfg, always_emit_head)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if logits.is_empty() {
            return Err(
                "final-position prefill must return non-empty logits in both arms \
                         (final position always has emit_head=true)"
                    .to_string(),
            );
        }
        Ok(elapsed_ms)
    };

    for _ in 0..warmup {
        let _ = run_arm(&mut state, true)?;
        let _ = run_arm(&mut state, false)?;
    }

    let mut baseline_ms = Vec::with_capacity(repeats);
    let mut optimized_ms = Vec::with_capacity(repeats);
    for i in 0..repeats {
        let b = run_arm(&mut state, true)?;
        let o = run_arm(&mut state, false)?;
        let delta_ms = b - o;
        let delta_pct = if b > 0.0 { 100.0 * delta_ms / b } else { 0.0 };
        println!(
            "RESULT repeat={i} visual_rows={visual_rows_actual} prompt_tokens={prompt_len} \
             baseline_always_emit_ms={b:.3} optimized_emit_skip_ms={o:.3} delta_ms={delta_ms:.3} \
             delta_pct={delta_pct:.2}"
        );
        baseline_ms.push(b);
        optimized_ms.push(o);
    }

    let baseline_median = median(&mut baseline_ms)?;
    let optimized_median = median(&mut optimized_ms)?;
    let delta_median_ms = baseline_median - optimized_median;
    let delta_median_pct = if baseline_median > 0.0 {
        100.0 * delta_median_ms / baseline_median
    } else {
        0.0
    };
    println!(
        "SUMMARY visual_rows={visual_rows_actual} prompt_tokens={prompt_len} repeats={repeats} \
         baseline_always_emit_median_ms={baseline_median:.3} \
         optimized_emit_skip_median_ms={optimized_median:.3} delta_median_ms={delta_median_ms:.3} \
         delta_median_pct={delta_median_pct:.2}"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{median, validate_repeats};

    #[test]
    fn validate_repeats_rejects_zero() {
        let err = validate_repeats(0).expect_err("BENCH_REPEATS=0 must be rejected");
        assert!(
            err.contains("BENCH_REPEATS=0"),
            "error message must name the offending env var; got: {err}"
        );
    }

    #[test]
    fn validate_repeats_accepts_positive_values() {
        assert!(validate_repeats(1).is_ok());
        assert!(validate_repeats(5).is_ok());
    }

    #[test]
    fn median_rejects_empty_slice_instead_of_panicking() {
        let mut empty: Vec<f64> = Vec::new();
        assert!(median(&mut empty).is_err());
    }

    #[test]
    fn median_of_single_repeat_is_that_repeat() {
        let mut v = vec![42.0];
        assert_eq!(median(&mut v).unwrap(), 42.0);
    }

    #[test]
    fn median_sorts_before_indexing() {
        let mut v = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&mut v).unwrap(), 2.0);
    }
}
