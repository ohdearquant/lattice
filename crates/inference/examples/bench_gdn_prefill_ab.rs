/// Benchmark: Qwen3.5-0.8B Metal `forward_prefill` latency for serial vs chunked GDN prefill.
///
/// Chunked GDN prefill is the default; set LATTICE_GDN_CHUNKED=0 to force serial.
///
/// Usage (single mode — uses whatever is default or set by env):
///   BENCH_PREFILL_LENS=145,289,529,1009 BENCH_RUNS=5 \
///   cargo run --release --example bench_gdn_prefill_ab -p lattice-inference --features "f16,metal-gpu"
///
/// Usage (interleaved A/B — measures both paths back-to-back):
///   BENCH_INTERLEAVED_AB=1 BENCH_PREFILL_LENS=145,289,529,1009 BENCH_RUNS=5 \
///   cargo run --release --example bench_gdn_prefill_ab -p lattice-inference --features "f16,metal-gpu"
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_gdn_prefill_ab requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use std::path::PathBuf;
    use std::time::Instant;

    let home = std::env::var("HOME").expect("HOME not set");
    let model_dir = std::env::var("LATTICE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(home).join(".lattice/models/qwen3.5-0.8b"));
    let lens = parse_lens();
    let runs = parse_env_usize("BENCH_RUNS", 5);
    let warmups = parse_env_usize("BENCH_WARMUPS", 1);
    let max_len = lens.iter().copied().max().unwrap_or(1009);
    let max_cache_len = parse_env_usize("BENCH_MAX_CACHE_LEN", max_len + 64).max(max_len + 1);
    if std::env::var_os("BENCH_INTERLEAVED_AB").is_some() {
        run_interleaved(&model_dir, &lens, runs, warmups, max_cache_len);
        return;
    }

    let mode = match std::env::var("LATTICE_GDN_CHUNKED").as_deref() {
        Ok("0") | Ok("false") => "serial",
        _ => "chunked",
    };

    eprintln!("[bench_gdn_prefill_ab] mode={mode}");
    eprintln!("[bench_gdn_prefill_ab] model_dir={}", model_dir.display());
    eprintln!("[bench_gdn_prefill_ab] lens={lens:?} warmups={warmups} runs={runs}");

    let t_load = Instant::now();
    let model = Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    let mut state =
        MetalQwen35State::new(model.weights(), model.config(), max_cache_len).expect("Metal state");
    eprintln!(
        "[bench_gdn_prefill_ab] loaded in {:.2}s active_gdn_layers={}",
        t_load.elapsed().as_secs_f64(),
        model.config().num_active_linear_attention_layers()
    );

    for seq_len in lens {
        let prompt_ids: Vec<u32> = (0..seq_len).map(|i| 100u32 + (i as u32 % 1000)).collect();

        for _ in 0..warmups {
            state.reset_state();
            let _ = state.forward_prefill(&prompt_ids);
        }

        let mut ms = Vec::with_capacity(runs);
        for run in 0..runs {
            state.reset_state();
            let t = Instant::now();
            let logits = state.forward_prefill(&prompt_ids);
            let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(logits.len(), model.config().vocab_size);
            eprintln!(
                "  mode={mode} seq_len={seq_len} run={} ms={elapsed_ms:.3} tok_s={:.3}",
                run + 1,
                seq_len as f64 / (elapsed_ms / 1000.0)
            );
            ms.push(elapsed_ms);
        }

        ms.sort_by(f64::total_cmp);
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let median = if ms.len() % 2 == 0 {
            (ms[ms.len() / 2 - 1] + ms[ms.len() / 2]) / 2.0
        } else {
            ms[ms.len() / 2]
        };
        let variance = ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ms.len() as f64;
        let std_dev = variance.sqrt();
        let cv_pct = if mean > 0.0 {
            std_dev / mean * 100.0
        } else {
            0.0
        };
        let mean_tps = seq_len as f64 / (mean / 1000.0);
        let median_tps = seq_len as f64 / (median / 1000.0);

        println!(
            "PREFILL_BENCH_RESULT mode={mode} seq_len={seq_len} runs={} \
             mean_ms={mean:.3} median_ms={median:.3} std_dev_ms={std_dev:.3} \
             cv_pct={cv_pct:.3} mean_toks_per_sec={mean_tps:.3} \
             median_toks_per_sec={median_tps:.3}",
            ms.len()
        );
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_interleaved(
    model_dir: &std::path::Path,
    lens: &[usize],
    runs: usize,
    warmups: usize,
    max_cache_len: usize,
) {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use std::time::Instant;

    eprintln!("[bench_gdn_prefill_ab] mode=interleaved_ab");
    eprintln!("[bench_gdn_prefill_ab] model_dir={}", model_dir.display());
    eprintln!("[bench_gdn_prefill_ab] lens={lens:?} warmups={warmups} runs={runs}");

    let t_load = Instant::now();
    let model = Qwen35Model::from_safetensors(model_dir).expect("load qwen3.5-0.8b");
    let mut state =
        MetalQwen35State::new(model.weights(), model.config(), max_cache_len).expect("Metal state");
    eprintln!(
        "[bench_gdn_prefill_ab] loaded in {:.2}s active_gdn_layers={}",
        t_load.elapsed().as_secs_f64(),
        model.config().num_active_linear_attention_layers()
    );

    for &seq_len in lens {
        let prompt_ids: Vec<u32> = (0..seq_len).map(|i| 100u32 + (i as u32 % 1000)).collect();

        for _ in 0..warmups {
            let _ = run_one(&mut state, &prompt_ids, false);
            let _ = run_one(&mut state, &prompt_ids, true);
        }

        let mut serial_ms = Vec::with_capacity(runs);
        let mut chunked_ms = Vec::with_capacity(runs);
        for pair in 0..runs {
            let serial_first = pair % 2 == 0;
            if serial_first {
                serial_ms.push(run_one(&mut state, &prompt_ids, false));
                chunked_ms.push(run_one(&mut state, &prompt_ids, true));
            } else {
                chunked_ms.push(run_one(&mut state, &prompt_ids, true));
                serial_ms.push(run_one(&mut state, &prompt_ids, false));
            }
            eprintln!(
                "  pair={} seq_len={seq_len} serial_ms={:.3} chunked_ms={:.3} order={}",
                pair + 1,
                serial_ms[pair],
                chunked_ms[pair],
                if serial_first {
                    "serial_first"
                } else {
                    "chunked_first"
                }
            );
        }

        let serial = summarize(seq_len, &mut serial_ms);
        let chunked = summarize(seq_len, &mut chunked_ms);
        let delta_mean_pct = (chunked.mean_toks_per_sec - serial.mean_toks_per_sec)
            / serial.mean_toks_per_sec
            * 100.0;
        let delta_median_pct = (chunked.median_toks_per_sec - serial.median_toks_per_sec)
            / serial.median_toks_per_sec
            * 100.0;

        println!(
            "PREFILL_AB_RESULT seq_len={seq_len} runs={runs} \
             serial_mean_ms={:.3} serial_median_ms={:.3} serial_cv_pct={:.3} \
             serial_mean_toks_per_sec={:.3} serial_median_toks_per_sec={:.3} \
             chunked_mean_ms={:.3} chunked_median_ms={:.3} chunked_cv_pct={:.3} \
             chunked_mean_toks_per_sec={:.3} chunked_median_toks_per_sec={:.3} \
             delta_mean_pct={delta_mean_pct:.3} delta_median_pct={delta_median_pct:.3}",
            serial.mean_ms,
            serial.median_ms,
            serial.cv_pct,
            serial.mean_toks_per_sec,
            serial.median_toks_per_sec,
            chunked.mean_ms,
            chunked.median_ms,
            chunked.cv_pct,
            chunked.mean_toks_per_sec,
            chunked.median_toks_per_sec,
        );
    }
    state.set_gdn_chunked(false);
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_one(
    state: &mut lattice_inference::forward::metal_qwen35::MetalQwen35State,
    prompt_ids: &[u32],
    chunked: bool,
) -> f64 {
    // `use_gdn_chunked` is latched from the env at construction, so a mid-run env
    // toggle is a no-op (the bug this replaces compared chunked-against-itself).
    // Flip the live state's path directly via the runtime setter.
    state.set_gdn_chunked(chunked);
    state.reset_state();
    let t = std::time::Instant::now();
    let _ = state.forward_prefill(prompt_ids);
    t.elapsed().as_secs_f64() * 1000.0
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
struct Summary {
    mean_ms: f64,
    median_ms: f64,
    cv_pct: f64,
    mean_toks_per_sec: f64,
    median_toks_per_sec: f64,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn summarize(seq_len: usize, ms: &mut [f64]) -> Summary {
    ms.sort_by(f64::total_cmp);
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    let median = if ms.len().is_multiple_of(2) {
        (ms[ms.len() / 2 - 1] + ms[ms.len() / 2]) / 2.0
    } else {
        ms[ms.len() / 2]
    };
    let variance = ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ms.len() as f64;
    let std_dev = variance.sqrt();
    let cv_pct = if mean > 0.0 {
        std_dev / mean * 100.0
    } else {
        0.0
    };
    Summary {
        mean_ms: mean,
        median_ms: median,
        cv_pct,
        mean_toks_per_sec: seq_len as f64 / (mean / 1000.0),
        median_toks_per_sec: seq_len as f64 / (median / 1000.0),
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn parse_lens() -> Vec<usize> {
    let raw = std::env::var("BENCH_PREFILL_LENS").unwrap_or_else(|_| "145,289,529,1009".into());
    raw.split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid BENCH_PREFILL_LENS entry: {s}"))
        })
        .collect()
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be usize, got {v}"))
        })
        .unwrap_or(default)
}
