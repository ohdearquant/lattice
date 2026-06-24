/// Per-kernel GPU time attribution for a single Metal decode step on Qwen3.5-0.8B.
///
/// Sets `LATTICE_DECODE_PROFILE=1` and runs `n_steps` decode steps. For each step
/// the library prints a `[DECODE_PROFILE]` table to stderr with true GPU execution
/// time (commit+wait) per kernel group: gdn_layer, gqa_layer, lm_head.
///
/// Model format is auto-detected from the directory: `.q4` files → Q4 weights,
/// `model.safetensors` → f16 weights.
///
/// Usage (f16):
///   LATTICE_MODEL_DIR=/Users/lion/.lattice/models/qwen3.5-0.8b \
///   cargo run --release --example decode_profile -p lattice-inference --features "f16,metal-gpu"
///
/// Usage (Q4):
///   LATTICE_MODEL_DIR=/Users/lion/.lattice/models/qwen3.5-0.8b-q4 \
///   cargo run --release --example decode_profile -p lattice-inference --features "f16,metal-gpu"
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("decode_profile requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use std::time::Instant;

    // Activate per-group GPU timing in forward_step_inner_impl.
    // SAFETY: single-threaded example binary; no races.
    unsafe { std::env::set_var("LATTICE_DECODE_PROFILE", "1") };

    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .expect("set LATTICE_MODEL_DIR to the model directory (f16 or Q4)");
    let dir = std::path::Path::new(&model_dir_str);

    let is_q4 = dir
        .read_dir()
        .map(|mut entries| {
            entries.any(|e| {
                e.map(|e| e.path().extension().map(|x| x == "q4").unwrap_or(false))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    eprintln!(
        "[decode_profile] Loading {} model from {} ...",
        if is_q4 { "Q4" } else { "f16" },
        dir.display()
    );

    let t_load = Instant::now();
    let mut state: MetalQwen35State = if is_q4 {
        let cfg = if dir.join("config.json").exists() {
            Qwen35Config::from_config_json(&dir.join("config.json")).expect("parse config.json")
        } else {
            Qwen35Config::qwen35_0_8b()
        };
        let tokenizer_path = dir.join("tokenizer.json");
        MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 512)
            .expect("from_q4_dir failed — MSL compile or weight load error")
    } else {
        let model = Qwen35Model::from_safetensors(dir).expect("from_safetensors failed");
        MetalQwen35State::new(model.weights(), model.config(), 512)
            .expect("MetalQwen35State::new failed")
    };
    eprintln!(
        "[decode_profile] Model loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );

    // Prefill with a ~256-token prompt using cycling IDs 100..356.
    let prefill_len = 256usize;
    let prompt_ids: Vec<u32> = (0u32..prefill_len as u32)
        .map(|i| 100 + (i % 256))
        .collect();

    eprintln!("[decode_profile] Warmup prefill ({prefill_len} tokens)...");
    state.reset_state();
    let prefill_logits = state.forward_prefill(&prompt_ids);
    let first_argmax = argmax_f32(&prefill_logits);
    eprintln!("[decode_profile] Prefill complete. greedy first token = {first_argmax}");

    // Warmup decode: 3 steps (not timed, LATTICE_DECODE_PROFILE already set).
    for i in 0..3usize {
        let _ = state.forward_step(200 + i as u32, prefill_len + i);
    }

    let n_steps = 10usize;
    eprintln!();
    eprintln!("[decode_profile] Running {n_steps} profiled decode steps.");
    eprintln!("[decode_profile] Attribution tables follow (one per step):");
    eprintln!();

    let mut step_wall_us: Vec<u128> = Vec::with_capacity(n_steps);

    for i in 0..n_steps {
        let pos = prefill_len + 3 + i;
        let t = Instant::now();
        let logits = state.forward_step(200 + i as u32, pos);
        let wall_us = t.elapsed().as_micros();
        step_wall_us.push(wall_us);
        // Suppress "value unused" without touching logits contents.
        let _ = logits.len();
        eprintln!();
    }

    let total_wall_us: u128 = step_wall_us.iter().sum();
    let mean_wall_us = total_wall_us as f64 / n_steps as f64;
    let min_wall_us = *step_wall_us.iter().min().unwrap_or(&0) as f64;
    let max_wall_us = *step_wall_us.iter().max().unwrap_or(&0) as f64;

    eprintln!("[decode_profile] === Wall-time summary ({n_steps} steps) ===");
    eprintln!(
        "  mean: {mean_wall_us:.0} µs  ({:.1} tok/s under profiling)",
        1_000_000.0 / mean_wall_us
    );
    eprintln!("  min:  {min_wall_us:.0} µs");
    eprintln!("  max:  {max_wall_us:.0} µs");
    eprintln!();
    eprintln!("  NOTE: throughput above is lower than fused-path throughput because");
    eprintln!("  LATTICE_DECODE_PROFILE serializes GPU pipelining via per-group commit/wait.");
    eprintln!("  Refer to the [DECODE_PROFILE] tables above for per-group breakdown.");
    eprintln!();
    eprintln!("[decode_profile] OFF-path sanity: greedy first token from prefill = {first_argmax}");
    eprintln!(
        "  (re-run without LATTICE_DECODE_PROFILE to confirm the same argmax via the fused path)"
    );
}

fn argmax_f32(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
