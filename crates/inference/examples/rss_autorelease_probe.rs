/// Paired RSS probe for lattice#1584: does the Qwen3.5 Metal decode loop accumulate
/// autoreleased command buffers when no autorelease pool encloses the dispatch?
///
/// Two arms, selected by `LATTICE_RSS_ARM`, run as separate processes so neither
/// inherits the other's high-water mark:
///   nopool  — `state.forward_step(..)` exactly as the library is called today
///   pool    — the same call inside `objc::rc::autoreleasepool`, which drains at the
///             same granularity a per-iteration pool inside the loop would
///
/// Usage:
///   LATTICE_RSS_ARM=nopool LATTICE_MODEL_DIR="$HOME/.lattice/models/qwen3.5-0.8b" \
///     cargo run --release --example rss_autorelease_probe -p lattice-inference \
///     --features "f16,metal-gpu"
///
/// Env: LATTICE_RSS_STEPS (default 3000) · LATTICE_RSS_SAMPLE_EVERY (default 100)
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("rss_autorelease_probe requires macOS + metal-gpu.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn rss_bytes() -> u64 {
    let pid = std::process::id();
    let out = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("ps failed");
    let s = String::from_utf8_lossy(&out.stdout);
    let kib: u64 = s
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("unparseable ps rss {:?}: {e}", s.trim()));
    kib * 1024
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use std::time::Instant;

    let arm = std::env::var("LATTICE_RSS_ARM").unwrap_or_else(|_| "nopool".to_string());
    assert!(
        arm == "nopool" || arm == "pool",
        "LATTICE_RSS_ARM must be nopool or pool, got {arm:?}"
    );
    let steps: usize = std::env::var("LATTICE_RSS_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let sample_every: usize = std::env::var("LATTICE_RSS_SAMPLE_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let home = std::env::var("HOME").expect("HOME");
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir_str);

    let rss_pre_load = rss_bytes();

    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    let is_q4 = dir
        .read_dir()
        .map(|mut e| {
            e.any(|x| {
                x.map(|x| x.path().extension().map(|q| q == "q4").unwrap_or(false))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    let prefill_len = 256usize;
    let cache_len = prefill_len + steps + 64;

    let t_load = Instant::now();
    let mut state: MetalQwen35State = if is_q4 {
        let cfg = Qwen35Config::from_model_dir(dir).expect("config.json");
        let tok = dir.join("tokenizer.json");
        MetalQwen35State::from_q4_dir(dir, &tok, &cfg, cache_len).expect("from_q4_dir")
    } else {
        let model = Qwen35Model::from_safetensors(dir).expect("from_safetensors");
        MetalQwen35State::new(model.weights(), model.config(), cache_len).expect("new")
    };
    let rss_post_load = rss_bytes();
    eprintln!(
        "[rss_probe] arm={arm} q4={is_q4} loaded in {:.1}s cache_len={cache_len}",
        t_load.elapsed().as_secs_f64()
    );
    // Positive control for the sampler: loading weights must move RSS by model scale.
    println!(
        "RSS_CONTROL arm={arm} pre_load_bytes={rss_pre_load} post_load_bytes={rss_post_load} \
         delta_bytes={}",
        rss_post_load as i64 - rss_pre_load as i64
    );

    state.reset_state();
    let prompt_ids: Vec<u32> = (0u32..prefill_len as u32)
        .map(|i| 100 + (i % 256))
        .collect();
    let _ = state.forward_prefill(&prompt_ids);

    // Warmup: first steps allocate scratch and caches that are not the subject.
    for i in 0..16usize {
        let pos = prefill_len + i;
        if arm == "pool" {
            objc::rc::autoreleasepool(|| {
                let _ = state.forward_step(200 + (i as u32 % 900), pos);
            });
        } else {
            let _ = state.forward_step(200 + (i as u32 % 900), pos);
        }
    }

    let base_pos = prefill_len + 16;
    let mut samples: Vec<(usize, u64)> = Vec::new();
    let rss_start = rss_bytes();
    samples.push((0, rss_start));
    println!("RSS_SAMPLE arm={arm} step=0 rss_bytes={rss_start}");

    let t0 = Instant::now();
    for i in 0..steps {
        let pos = base_pos + i;
        let tok = 200 + (i as u32 % 900);
        if arm == "pool" {
            objc::rc::autoreleasepool(|| {
                let _ = state.forward_step(tok, pos);
            });
        } else {
            let _ = state.forward_step(tok, pos);
        }
        if (i + 1) % sample_every == 0 {
            let r = rss_bytes();
            samples.push((i + 1, r));
            println!("RSS_SAMPLE arm={arm} step={} rss_bytes={r}", i + 1);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let (first_step, first_rss) = samples[0];
    let (last_step, last_rss) = *samples.last().expect("samples");
    let span = (last_step - first_step) as f64;
    let slope = if span > 0.0 {
        (last_rss as f64 - first_rss as f64) / span
    } else {
        0.0
    };
    println!(
        "RSS_SUMMARY arm={arm} steps={steps} samples={} first_rss_bytes={first_rss} \
         last_rss_bytes={last_rss} delta_bytes={} bytes_per_step={slope:.1} \
         tok_per_s={:.1}",
        samples.len(),
        last_rss as i64 - first_rss as i64,
        steps as f64 / elapsed
    );
}
