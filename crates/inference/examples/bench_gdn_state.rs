/// GDN recurrent-state byte-traffic counters for a single Metal decode step on Qwen3.5-0.8B.
///
/// Sets `LATTICE_GDN_STATE_COUNTERS=1` and runs `n_steps` direct decode steps. After each
/// step, reads and resets the per-session GDN state-traffic report via
/// `take_gdn_state_traffic_report()` and prints a `GDN_STATE_STEP` line with logical
/// read/write bytes and snapshot/restore counts for the decode and MTP-verify scopes,
/// plus a cross-check against `active_gdn_layers * per_layer_state_bytes`.
///
/// Each `GDN_STATE_STEP` line also reports `state_bytes_moved` (decode read + write for
/// that step) — the MEASURED per-step recurrent-state byte traffic used by the #491
/// bandwidth-share decision gate. The state buffers (conv cache + S matrix) are a fixed
/// function of model config, not sequence position, so this figure is expected to be
/// CONTEXT-INVARIANT (does not grow with context length, unlike KV-cache bytes/token).
/// `GDN_STATE_BYTES_PER_STEP` reports the average across the measured steps plus the
/// min/max spread, so context-invariance can be checked directly from the bench output.
///
/// Model format is auto-detected from the directory: `.q4` files → Q4 weights,
/// `model.safetensors` → f16 weights.
///
/// Usage (f16):
///   LATTICE_MODEL_DIR=/Users/lion/.lattice/models/qwen3.5-0.8b \
///   cargo run --release --example bench_gdn_state -p lattice-inference \
///     --features "f16,metal-gpu,gdn-state-counters"
///
/// Usage (Q4):
///   LATTICE_MODEL_DIR=/Users/lion/.lattice/models/qwen3.5-0.8b-q4 \
///   cargo run --release --example bench_gdn_state -p lattice-inference \
///     --features "f16,metal-gpu,gdn-state-counters"
///
/// Env:
///   LATTICE_GDN_STATE_STEPS  number of measured decode steps (default 10)
fn main() {
    #[cfg(not(all(
        target_os = "macos",
        feature = "metal-gpu",
        feature = "gdn-state-counters"
    )))]
    {
        eprintln!("bench_gdn_state requires macOS + metal-gpu + gdn-state-counters features.");
        std::process::exit(1);
    }

    #[cfg(all(
        target_os = "macos",
        feature = "metal-gpu",
        feature = "gdn-state-counters"
    ))]
    run();
}

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "gdn-state-counters"
))]
fn run() {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::Qwen35Config;

    // Activate GDN state-traffic accounting in InferenceSession construction.
    // SAFETY: single-threaded example binary; no races.
    unsafe { std::env::set_var("LATTICE_GDN_STATE_COUNTERS", "1") };

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
        "[bench_gdn_state] Loading {} model from {} ...",
        if is_q4 { "Q4" } else { "f16" },
        dir.display()
    );

    let mut state: MetalQwen35State = if is_q4 {
        let cfg = Qwen35Config::from_model_dir(dir).expect("load model config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 512)
            .expect("from_q4_dir failed — MSL compile or weight load error")
    } else {
        let model = Qwen35Model::from_safetensors(dir).expect("from_safetensors failed");
        MetalQwen35State::new(model.weights(), model.config(), 512)
            .expect("MetalQwen35State::new failed")
    };

    let shape = state
        .gdn_state_traffic_report()
        .expect("LATTICE_GDN_STATE_COUNTERS=1 must activate the counters")
        .shape;
    println!(
        "GDN_STATE_SHAPE active_layers={} allocated_layers={} conv_bytes_per_layer={} s_bytes_per_layer={} per_layer_bytes={} active_state_bytes={} allocated_state_bytes={}",
        shape.active_gdn_layers,
        shape.allocated_gdn_layers,
        shape.conv_bytes_per_layer,
        shape.s_bytes_per_layer,
        shape.per_layer_state_bytes,
        shape.active_state_bytes,
        shape.allocated_state_bytes,
    );

    // Prefill with a small deterministic prompt using cycling IDs 100..116.
    let prefill_len = 16usize;
    let prompt_ids: Vec<u32> = (0u32..prefill_len as u32)
        .map(|i| 100 + (i % 256))
        .collect();

    state.reset_state();
    let _ = state.forward_prefill(&prompt_ids);

    // Warmup decode: 3 steps, not measured. Discard their counter deltas.
    for i in 0..3usize {
        let _ = state.forward_step(200 + i as u32, prefill_len + i);
    }
    let _ = state.take_gdn_state_traffic_report();

    let n_steps: usize = std::env::var("LATTICE_GDN_STATE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let expected_decode_bytes = shape.active_state_bytes;

    let mut total_decode_read = 0u64;
    let mut total_decode_write = 0u64;
    let mut total_decode_copy = 0u64;
    let mut total_mtp_read = 0u64;
    let mut total_mtp_write = 0u64;
    let mut total_mtp_copy = 0u64;
    let mut min_state_bytes_moved = u64::MAX;
    let mut max_state_bytes_moved = 0u64;

    for i in 0..n_steps {
        let pos = prefill_len + 3 + i;
        let _ = state.forward_step(200 + i as u32, pos);
        let report = state
            .take_gdn_state_traffic_report()
            .expect("counters active for the whole run");

        let decode_read_bytes = report.decode.read_bytes;
        let decode_write_bytes = report.decode.write_bytes;
        let decode_copy_count = report.decode.copy_count();
        let mtp_read_bytes = report.mtp_verify.read_bytes;
        let mtp_write_bytes = report.mtp_verify.write_bytes;
        let mtp_copy_count = report.mtp_verify.copy_count();
        let ok = decode_read_bytes == expected_decode_bytes
            && decode_write_bytes == expected_decode_bytes;
        let state_bytes_moved = decode_read_bytes + decode_write_bytes;

        println!(
            "GDN_STATE_STEP step={} pos={} decode_read_bytes={} decode_write_bytes={} state_bytes_moved={} decode_copy_count={} mtp_read_bytes={} mtp_write_bytes={} mtp_copy_count={} expected_decode_bytes={} ok={}",
            i + 1,
            pos,
            decode_read_bytes,
            decode_write_bytes,
            state_bytes_moved,
            decode_copy_count,
            mtp_read_bytes,
            mtp_write_bytes,
            mtp_copy_count,
            expected_decode_bytes,
            ok,
        );

        total_decode_read += decode_read_bytes;
        total_decode_write += decode_write_bytes;
        total_decode_copy += decode_copy_count;
        total_mtp_read += mtp_read_bytes;
        total_mtp_write += mtp_write_bytes;
        total_mtp_copy += mtp_copy_count;
        min_state_bytes_moved = min_state_bytes_moved.min(state_bytes_moved);
        max_state_bytes_moved = max_state_bytes_moved.max(state_bytes_moved);
    }

    println!(
        "GDN_STATE_TOTAL steps={n_steps} decode_read_bytes={total_decode_read} decode_write_bytes={total_decode_write} decode_copy_count={total_decode_copy} mtp_read_bytes={total_mtp_read} mtp_write_bytes={total_mtp_write} mtp_copy_count={total_mtp_copy}"
    );

    println!(
        "GDN_STATE_CROSSCHECK expected_per_step={expected_decode_bytes} profiler_group=gdn_mixer formula=active_gdn_layers*per_layer_state_bytes"
    );

    // MEASURED per-step state-byte traffic for the #491 bandwidth-share decision gate.
    // avg == min == max is expected: state buffer size is fixed by config, not context
    // length, so this figure should be context-invariant (contrast with KV bytes/token,
    // which grows linearly with context length via `kv_bytes_per_token`).
    let avg_state_bytes_moved = (total_decode_read + total_decode_write) / n_steps as u64;
    println!(
        "GDN_STATE_BYTES_PER_STEP avg={avg_state_bytes_moved} min={min_state_bytes_moved} max={max_state_bytes_moved} context_invariant={}",
        min_state_bytes_moved == max_state_bytes_moved
    );
}
