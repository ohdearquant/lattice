/// Concurrent session throughput benchmark for Qwen3.6-27B Q4 inference.
///
/// Tests the 50+ tok/s hypothesis: can interleaving two independent decode streams
/// improve effective throughput over a single-stream baseline?
///
/// IMPORTANT: The ideal design (two sessions sharing one MetalQwen35Engine) is blocked
/// by the current public API. MetalQwen35Engine and InferenceSession are not exported
/// from examples. This benchmark therefore loads two independent MetalQwen35State values,
/// each owning its own engine copy, and interleaves their forward_step calls on a single
/// thread. Memory cost is approximately doubled vs. the shared-engine design.
///
/// Usage:
///   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.6-27b-q4 \
///   LATTICE_TOKENIZER_DIR=~/.lattice/models/qwen3.6-27b \
///   LATTICE_CONCURRENT_STEPS=256 \
///   cargo run --release --example bench_concurrent -p lattice-inference --features "f16,metal-gpu"
///
/// Go/no-go threshold: if concurrent effective tok/s >= 1.7x single-stream,
/// recommend Phase A (public concurrent API). Otherwise recommend Phase B (LNDL token reduction).
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_concurrent requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(err) = run() {
            eprintln!("[bench_concurrent] ERROR: {err}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const DEFAULT_STEPS: usize = 256;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const WARMUP_STEPS: usize = 8;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const MAX_CACHE_LEN: usize = 512;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const WORKING_SET_SAFETY_FACTOR: f64 = 0.90;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const TOKEN_START_A: u32 = 200;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const TOKEN_START_B: u32 = 400;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const TOKEN_CYCLE: u32 = 64;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Debug, Clone)]
struct TimingStats {
    label: &'static str,
    tokens: usize,
    elapsed: std::time::Duration,
    tokens_per_second: f64,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Debug, Clone)]
struct MemorySnapshot {
    rss_bytes: u64,
    metal_allocated_bytes: u64,
    metal_recommended_working_set_bytes: u64,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[derive(Debug, Clone)]
struct MemoryPreflight {
    sessions: usize,
    loaded_states: usize,
    per_state_estimated_bytes: u64,
    projected_metal_allocated_bytes: u64,
    recommended_limit_bytes: u64,
    safety_limit_bytes: u64,
    headroom_bytes: i128,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn snapshot_memory(device: &metal::Device) -> Result<MemorySnapshot, String> {
    let rss_bytes = get_process_rss()?;
    let metal_allocated_bytes = device.current_allocated_size() as u64;
    let metal_recommended_working_set_bytes = device.recommended_max_working_set_size();
    Ok(MemorySnapshot {
        rss_bytes,
        metal_allocated_bytes,
        metal_recommended_working_set_bytes,
    })
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn preflight_memory_check(
    device: &metal::Device,
    sessions: usize,
    loaded_states: usize,
    baseline_metal_allocated_bytes: u64,
    current_metal_allocated_bytes: u64,
    safety_factor: f64,
) -> Result<MemoryPreflight, String> {
    let recommended_limit_bytes = device.recommended_max_working_set_size();
    let safety_limit_bytes = (recommended_limit_bytes as f64 * safety_factor) as u64;

    let per_state_estimated_bytes = if loaded_states > 0 {
        current_metal_allocated_bytes.saturating_sub(baseline_metal_allocated_bytes)
            / loaded_states as u64
    } else {
        0
    };

    let projected_metal_allocated_bytes =
        baseline_metal_allocated_bytes + per_state_estimated_bytes * sessions as u64;

    let headroom_bytes = safety_limit_bytes as i128 - projected_metal_allocated_bytes as i128;

    if headroom_bytes < 0 {
        return Err(format!(
            "insufficient Metal working set for {sessions} public states: \
             projected={projected_metal_allocated_bytes} safety_limit={safety_limit_bytes} \
             recommended={recommended_limit_bytes}; shared-engine API is unavailable, \
             so this examples-only benchmark would likely trigger memory pressure"
        ));
    }

    Ok(MemoryPreflight {
        sessions,
        loaded_states,
        per_state_estimated_bytes,
        projected_metal_allocated_bytes,
        recommended_limit_bytes,
        safety_limit_bytes,
        headroom_bytes,
    })
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn measure_single_session(
    state: &mut lattice_inference::forward::metal_qwen35::MetalQwen35State,
    steps: usize,
    token_start: u32,
    start_position: usize,
) -> Result<TimingStats, String> {
    state.reset_state();

    for i in 0..WARMUP_STEPS {
        let _ = state.forward_step(token_start + (i as u32 % TOKEN_CYCLE), start_position + i);
    }

    let t = std::time::Instant::now();
    for i in 0..steps {
        let pos = start_position + WARMUP_STEPS + i;
        let _ = state.forward_step(token_start + (i as u32 % TOKEN_CYCLE), pos);
    }
    let elapsed = t.elapsed();

    Ok(TimingStats {
        label: "single",
        tokens: steps,
        elapsed,
        tokens_per_second: steps as f64 / elapsed.as_secs_f64(),
    })
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn measure_concurrent_interleaved(
    state_a: &mut lattice_inference::forward::metal_qwen35::MetalQwen35State,
    state_b: &mut lattice_inference::forward::metal_qwen35::MetalQwen35State,
    steps_per_session: usize,
    token_start_a: u32,
    token_start_b: u32,
    start_position: usize,
) -> Result<TimingStats, String> {
    state_a.reset_state();
    state_b.reset_state();

    for i in 0..WARMUP_STEPS {
        let pos = start_position + i;
        let _ = state_a.forward_step(token_start_a + (i as u32 % TOKEN_CYCLE), pos);
        let _ = state_b.forward_step(token_start_b + (i as u32 % TOKEN_CYCLE), pos);
    }

    let t = std::time::Instant::now();
    for step in 0..steps_per_session {
        let pos = start_position + WARMUP_STEPS + step;
        let _ = state_a.forward_step(token_start_a + (step as u32 % TOKEN_CYCLE), pos);
        let _ = state_b.forward_step(token_start_b + (step as u32 % TOKEN_CYCLE), pos);
    }
    let elapsed = t.elapsed();
    let total_tokens = 2 * steps_per_session;

    Ok(TimingStats {
        label: "concurrent_interleaved",
        tokens: total_tokens,
        elapsed,
        tokens_per_second: total_tokens as f64 / elapsed.as_secs_f64(),
    })
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn report_results(
    device_name: &str,
    total_steps: usize,
    concurrent_steps_per_session: usize,
    single: &TimingStats,
    concurrent: &TimingStats,
    memory_before: &MemorySnapshot,
    memory_after_state1: &MemorySnapshot,
    memory_after_state2: &MemorySnapshot,
    memory_after_concurrent: &MemorySnapshot,
    preflight: &MemoryPreflight,
) {
    let speedup = concurrent.tokens_per_second / single.tokens_per_second;
    let go = speedup >= 1.7;

    let metal_delta_during_run = memory_after_concurrent
        .metal_allocated_bytes
        .saturating_sub(memory_after_state2.metal_allocated_bytes);
    let recommendation = if go {
        "GO_PHASE_A_CONCURRENT_API"
    } else if metal_delta_during_run > preflight.per_state_estimated_bytes / 4 {
        "NO_GO_MEMORY_OR_QUEUE_PRESSURE"
    } else {
        "NO_GO_QUEUE_SERIALIZATION_OR_DISPATCH_OVERHEAD"
    };

    // Human-readable summary on stderr
    eprintln!();
    eprintln!("=== bench_concurrent results ===");
    eprintln!("Device: {device_name}");
    eprintln!(
        "Single-session:     {:.2} tok/s  ({} tokens, {:.1}ms)",
        single.tokens_per_second,
        single.tokens,
        single.elapsed.as_secs_f64() * 1000.0
    );
    eprintln!(
        "Interleaved (2×{}): {:.2} effective tok/s  ({} tokens, {:.1}ms)",
        concurrent_steps_per_session,
        concurrent.tokens_per_second,
        concurrent.tokens,
        concurrent.elapsed.as_secs_f64() * 1000.0
    );
    eprintln!("Speedup: {speedup:.2}x  (threshold 1.70x)");
    eprintln!("Recommendation: {recommendation}");
    if go {
        eprintln!("  → Concurrent throughput >= 1.7x single-stream.");
        eprintln!(
            "    Recommend Phase A: expose public concurrent API (shared MetalQwen35Engine +"
        );
        eprintln!(
            "    multiple InferenceSession). NOTE: this benchmark used two independent states,"
        );
        eprintln!(
            "    so it measures interleaving headroom — not completed shared-weight validation."
        );
    } else {
        eprintln!("  → Concurrent throughput < 1.7x single-stream.");
        eprintln!(
            "    Recommend Phase B: prioritize LNDL token reduction as primary throughput lever."
        );
    }
    eprintln!();
    eprintln!("Memory summary:");
    eprintln!(
        "  Before load:    RSS={:.1}MB  Metal={:.1}MB",
        memory_before.rss_bytes as f64 / 1_048_576.0,
        memory_before.metal_allocated_bytes as f64 / 1_048_576.0
    );
    eprintln!(
        "  After state1:   RSS={:.1}MB  Metal={:.1}MB",
        memory_after_state1.rss_bytes as f64 / 1_048_576.0,
        memory_after_state1.metal_allocated_bytes as f64 / 1_048_576.0
    );
    eprintln!(
        "  After state2:   RSS={:.1}MB  Metal={:.1}MB",
        memory_after_state2.rss_bytes as f64 / 1_048_576.0,
        memory_after_state2.metal_allocated_bytes as f64 / 1_048_576.0
    );
    eprintln!(
        "  After run:      RSS={:.1}MB  Metal={:.1}MB",
        memory_after_concurrent.rss_bytes as f64 / 1_048_576.0,
        memory_after_concurrent.metal_allocated_bytes as f64 / 1_048_576.0
    );
    eprintln!(
        "  Recommended working set: {:.1}MB  Safety limit ({:.0}%): {:.1}MB",
        memory_before.metal_recommended_working_set_bytes as f64 / 1_048_576.0,
        WORKING_SET_SAFETY_FACTOR * 100.0,
        preflight.safety_limit_bytes as f64 / 1_048_576.0
    );
    eprintln!(
        "  Preflight headroom: {:.1}MB",
        preflight.headroom_bytes as f64 / 1_048_576.0
    );
    eprintln!();

    // Stable machine-readable output on stdout
    println!("BENCH_CONCURRENT_REPORT");
    println!(
        "BENCH_CONCURRENT_CONFIG public_shared_engine=false mode=two_independent_states_interleaved \
         total_steps={total_steps} concurrent_steps_per_session={concurrent_steps_per_session} \
         warmup_steps={WARMUP_STEPS} max_cache_len={MAX_CACHE_LEN}"
    );
    println!("BENCH_CONCURRENT_DEVICE name=\"{device_name}\"");
    println!(
        "BENCH_CONCURRENT_SINGLE tokens={} elapsed_ms={:.2} tok_s={:.2}",
        single.tokens,
        single.elapsed.as_secs_f64() * 1000.0,
        single.tokens_per_second
    );
    println!(
        "BENCH_CONCURRENT_INTERLEAVED tokens={} elapsed_ms={:.2} effective_tok_s={:.2} speedup_vs_single={:.2}",
        concurrent.tokens,
        concurrent.elapsed.as_secs_f64() * 1000.0,
        concurrent.tokens_per_second,
        speedup
    );
    println!(
        "BENCH_CONCURRENT_MEMORY_BEFORE rss_bytes={} metal_allocated_bytes={} metal_recommended_working_set_bytes={}",
        memory_before.rss_bytes,
        memory_before.metal_allocated_bytes,
        memory_before.metal_recommended_working_set_bytes
    );
    println!(
        "BENCH_CONCURRENT_MEMORY_AFTER_STATE1 rss_bytes={} metal_allocated_bytes={}",
        memory_after_state1.rss_bytes, memory_after_state1.metal_allocated_bytes
    );
    println!(
        "BENCH_CONCURRENT_MEMORY_AFTER_STATE2 rss_bytes={} metal_allocated_bytes={}",
        memory_after_state2.rss_bytes, memory_after_state2.metal_allocated_bytes
    );
    println!(
        "BENCH_CONCURRENT_MEMORY_AFTER_RUN rss_bytes={} metal_allocated_bytes={}",
        memory_after_concurrent.rss_bytes, memory_after_concurrent.metal_allocated_bytes
    );
    println!(
        "BENCH_CONCURRENT_PREFLIGHT sessions={} loaded_states={} \
         per_state_estimated_bytes={} projected_metal_allocated_bytes={} \
         safety_limit_bytes={} headroom_bytes={}",
        preflight.sessions,
        preflight.loaded_states,
        preflight.per_state_estimated_bytes,
        preflight.projected_metal_allocated_bytes,
        preflight.safety_limit_bytes,
        preflight.headroom_bytes
    );
    println!(
        "BENCH_CONCURRENT_RECOMMENDATION recommendation={recommendation} threshold_speedup=1.70 observed_speedup={speedup:.2}"
    );
    println!(
        "BENCH_CONCURRENT_CAVEAT shared_engine_api_blocked=true reason=\"MetalQwen35Engine and \
         InferenceSession are not public from examples; benchmark uses two independent MetalQwen35State values\""
    );
    println!(
        "BENCH_CONCURRENT_CAVEAT_SYNC forward_step_synchronous=true reason=\"forward_step calls cmd.commit()+wait_until_completed() per call — no GPU-level overlap between interleaved calls; true pipelining requires restructuring forward_step for async command buffer submission\""
    );
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn get_process_rss() -> Result<u64, String> {
    let pid = std::process::id();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .map_err(|e| format!("ps command failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let kib: u64 = stdout
        .trim()
        .parse()
        .map_err(|e| format!("failed to parse ps rss output {:?}: {e}", stdout.trim()))?;
    Ok(kib * 1024)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), String> {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use metal::Device;

    // --- Parse env vars ---
    let home = std::env::var("HOME").map_err(|_| "HOME env var not set".to_string())?;
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b-q4"));
    let tokenizer_dir_str = std::env::var("LATTICE_TOKENIZER_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b"));
    let total_steps: usize = std::env::var("LATTICE_CONCURRENT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_STEPS);
    if total_steps < 2 {
        return Err(format!(
            "LATTICE_CONCURRENT_STEPS must be >= 2, got {total_steps}"
        ));
    }

    let model_dir = std::path::Path::new(&model_dir_str).to_path_buf();
    let tokenizer_path = std::path::Path::new(&tokenizer_dir_str).join("tokenizer.json");

    if !model_dir.is_dir() {
        return Err(format!(
            "LATTICE_MODEL_DIR does not exist or is not a directory: {}",
            model_dir.display()
        ));
    }
    if !tokenizer_path.exists() {
        return Err(format!(
            "LATTICE_TOKENIZER_DIR must contain tokenizer.json: {}",
            tokenizer_path.display()
        ));
    }

    // --- Metal device ---
    let device = Device::system_default().ok_or_else(|| {
        "Metal device unavailable: Device::system_default() returned None".to_string()
    })?;
    let device_name = device.name().to_string();

    // --- Load config ---
    let cfg = if model_dir.join("config.json").exists() {
        Qwen35Config::from_config_json(&model_dir.join("config.json"))
            .map_err(|err| format!("failed to parse config.json: {err}"))?
    } else {
        Qwen35Config::qwen36_27b()
    };

    // Validate tokenizer path by loading it (path check is not enough for corrupted files)
    let _tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path).map_err(|err| {
        format!(
            "failed to load tokenizer from {}: {err}",
            tokenizer_path.display()
        )
    })?;

    // --- Memory snapshot before any model loading ---
    let memory_before = snapshot_memory(&device)?;

    // --- Load state 1 ---
    eprintln!(
        "[bench_concurrent] Loading state1 from {} ...",
        model_dir.display()
    );
    let t_load1 = std::time::Instant::now();
    let mut state1 =
        MetalQwen35State::from_q4_dir(&model_dir, &tokenizer_path, &cfg, MAX_CACHE_LEN)
            .map_err(|err| format!("failed to load state1 from {}: {err}", model_dir.display()))?;
    eprintln!(
        "[bench_concurrent] state1 loaded in {:.1}s",
        t_load1.elapsed().as_secs_f64()
    );

    let memory_after_state1 = snapshot_memory(&device)?;

    // --- Preflight: check whether 2 states can fit within the working set ---
    let preflight = preflight_memory_check(
        &device,
        2,
        1,
        memory_before.metal_allocated_bytes,
        memory_after_state1.metal_allocated_bytes,
        WORKING_SET_SAFETY_FACTOR,
    )?;

    // --- Single-session baseline ---
    eprintln!("[bench_concurrent] Measuring single-session baseline: {total_steps} steps");
    let single = measure_single_session(&mut state1, total_steps, TOKEN_START_A, 0)?;

    // --- Load state 2 ---
    eprintln!(
        "[bench_concurrent] Loading state2 from {} ...",
        model_dir.display()
    );
    let t_load2 = std::time::Instant::now();
    let mut state2 =
        MetalQwen35State::from_q4_dir(&model_dir, &tokenizer_path, &cfg, MAX_CACHE_LEN)
            .map_err(|err| format!("failed to load state2 from {}: {err}", model_dir.display()))?;
    eprintln!(
        "[bench_concurrent] state2 loaded in {:.1}s",
        t_load2.elapsed().as_secs_f64()
    );

    let memory_after_state2 = snapshot_memory(&device)?;

    // --- Interleaved concurrent measurement ---
    let concurrent_steps_per_session = total_steps / 2;
    eprintln!(
        "[bench_concurrent] Measuring interleaved run: {concurrent_steps_per_session} steps/session"
    );
    let concurrent = measure_concurrent_interleaved(
        &mut state1,
        &mut state2,
        concurrent_steps_per_session,
        TOKEN_START_A,
        TOKEN_START_B,
        0,
    )?;

    let memory_after_concurrent = snapshot_memory(&device)?;

    // --- Print report with go/no-go recommendation ---
    report_results(
        &device_name,
        total_steps,
        concurrent_steps_per_session,
        &single,
        &concurrent,
        &memory_before,
        &memory_after_state1,
        &memory_after_state2,
        &memory_after_concurrent,
        &preflight,
    );

    Ok(())
}
