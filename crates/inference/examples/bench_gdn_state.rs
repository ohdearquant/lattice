/// GDN recurrent-state byte-traffic counters for a single Metal decode step on Qwen3.5-0.8B.
///
/// Sets `LATTICE_GDN_STATE_COUNTERS=1` and runs `n_steps` direct decode steps. After each
/// step, reads and resets the per-session GDN state-traffic report via
/// `take_gdn_state_traffic_report()` and prints a `GDN_STATE_STEP` line with logical
/// read/write bytes and snapshot/restore counts for the decode and MTP-verify scopes,
/// plus a cross-check against `active_gdn_layers * per_layer_state_bytes`.
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

        println!(
            "GDN_STATE_STEP step={} pos={} decode_read_bytes={} decode_write_bytes={} decode_copy_count={} mtp_read_bytes={} mtp_write_bytes={} mtp_copy_count={} expected_decode_bytes={} ok={}",
            i + 1,
            pos,
            decode_read_bytes,
            decode_write_bytes,
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
    }

    println!(
        "GDN_STATE_TOTAL steps={n_steps} decode_read_bytes={total_decode_read} decode_write_bytes={total_decode_write} decode_copy_count={total_decode_copy} mtp_read_bytes={total_mtp_read} mtp_write_bytes={total_mtp_write} mtp_copy_count={total_mtp_copy}"
    );

    println!(
        "GDN_STATE_CROSSCHECK expected_per_step={expected_decode_bytes} profiler_group=gdn_mixer formula=active_gdn_layers*per_layer_state_bytes"
    );
}
