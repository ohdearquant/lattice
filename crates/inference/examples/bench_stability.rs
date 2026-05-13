/// Numerical stability benchmark for long-sequence generation.
///
/// Generates 1K-4K tokens and monitors for quality degradation:
/// - N-gram repetition rate (4-gram overlap in sliding window)
/// - Token diversity (unique tokens / total in window)
/// - Throughput drift (tok/s should be stable for GDN, may degrade for GQA)
/// - EOS rate (premature stopping signals distribution collapse)
///
/// Usage:
///   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.6-27b-q4 \
///   LATTICE_TOKENIZER_DIR=~/.lattice/models/qwen3.6-27b \
///   cargo run --release --example bench_stability -p lattice-inference --features "f16,metal-gpu"
///
/// Optional env vars:
///   LATTICE_STABILITY_TOKENS=4096    max tokens to generate (default: 2048)
///   LATTICE_STABILITY_WINDOW=256     monitoring window size (default: 256)
///   LATTICE_PROFILE=1                enable per-step GPU profiling
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_stability requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};
    use std::collections::HashMap;
    use std::time::Instant;

    let home = std::env::var("HOME").expect("HOME not set");
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b-q4"));
    let tokenizer_dir_str = std::env::var("LATTICE_TOKENIZER_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b"));

    let max_tokens: usize = std::env::var("LATTICE_STABILITY_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let window_size: usize = std::env::var("LATTICE_STABILITY_WINDOW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    let dir = std::path::Path::new(&model_dir_str);
    let tokenizer_path = std::path::Path::new(&tokenizer_dir_str).join("tokenizer.json");

    let cfg = if dir.join("config.json").exists() {
        Qwen35Config::from_config_json(&dir.join("config.json")).expect("parse config.json")
    } else {
        Qwen35Config::qwen36_27b()
    };

    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path).expect("load tokenizer");

    // Long-form prompts that should produce extended coherent output
    let prompts: &[(&str, &str)] = &[
        (
            "reasoning_en",
            "<|im_start|>user\nExplain in detail how a compiler transforms source code into \
             machine code. Cover lexical analysis, parsing, semantic analysis, optimization, \
             and code generation. For each phase, give a concrete example using a simple \
             expression like `x = a * b + c`.<|im_end|>\n<|im_start|>assistant\n",
        ),
        (
            "reasoning_zh",
            "<|im_start|>user\n请详细解释量子计算的基本原理。从量子比特开始，\
             讨论叠加态、纠缠、量子门操作，以及量子算法（如Shor算法和Grover算法）\
             的工作原理。每个概念都请举具体例子说明。<|im_end|>\n<|im_start|>assistant\n",
        ),
        (
            "code_gen",
            "<|im_start|>user\nWrite a complete implementation of a B-tree in Rust. Include \
             insert, search, delete, and iteration. Add doc comments and unit tests. \
             Use generic keys with Ord bound.<|im_end|>\n<|im_start|>assistant\n",
        ),
    ];

    // Cache length must accommodate prompt + generated tokens
    let cache_len = max_tokens + 512;

    eprintln!("[bench_stability] Loading model...");
    let t_load = Instant::now();
    let mut state = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, cache_len)
        .expect("model load failed");
    eprintln!(
        "  Loaded in {:.1}s (layers={}, cache={})",
        t_load.elapsed().as_secs_f64(),
        cfg.num_hidden_layers,
        cache_len
    );

    let gen_cfg = GenerateConfig {
        max_new_tokens: max_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        enable_thinking: false,
        ..Default::default()
    };

    eprintln!(
        "[bench_stability] Generating {} tokens per prompt (greedy), window={}",
        max_tokens, window_size
    );
    eprintln!();

    for (name, prompt) in prompts {
        eprintln!("=== Prompt: {name} ===");

        let mut token_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut window_starts: Vec<Instant> = Vec::new();
        let decode_start = Instant::now();
        let mut prefill_done = false;
        let mut prefill_time_ms = 0.0;

        state.generate_streaming(prompt, &tokenizer, &gen_cfg, |_text, tok_id| {
            if !prefill_done {
                prefill_time_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
                prefill_done = true;
            }
            token_ids.push(tok_id);
            window_starts.push(Instant::now());
            true
        });

        let total_time = decode_start.elapsed().as_secs_f64();
        let n = token_ids.len();

        if n == 0 {
            eprintln!("  WARNING: zero tokens generated (EOS immediately)");
            eprintln!();
            continue;
        }

        // Compute per-window metrics
        eprintln!(
            "  Generated {} tokens in {:.1}s ({:.1} tok/s overall, prefill {:.0}ms)",
            n,
            total_time,
            n as f64 / total_time,
            prefill_time_ms
        );
        eprintln!();
        eprintln!(
            "  {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
            "window", "tokens", "4gram_rep", "diversity", "eos_count", "tok/s"
        );

        let mut windows: Vec<WindowMetrics> = Vec::new();
        let num_windows = (n + window_size - 1) / window_size;

        for w in 0..num_windows {
            let start = w * window_size;
            let end = (start + window_size).min(n);
            let window_tokens = &token_ids[start..end];
            let wlen = window_tokens.len();

            // 4-gram repetition: fraction of 4-grams that appeared before in this window
            let rep_rate = ngram_repetition_rate(window_tokens, 4);

            // Token diversity: unique tokens / window size
            let unique: std::collections::HashSet<u32> = window_tokens.iter().cloned().collect();
            let diversity = unique.len() as f64 / wlen as f64;

            // EOS count in window
            let eos_count = window_tokens
                .iter()
                .filter(|&&t| t == cfg.eos_token_id)
                .count();

            // Throughput for this window (approximate from wall clock)
            let tok_per_sec = if w + 1 < num_windows && start + window_size <= window_starts.len() {
                let w_elapsed = window_starts[end.min(window_starts.len()) - 1]
                    .duration_since(window_starts[start])
                    .as_secs_f64();
                if w_elapsed > 0.0 {
                    wlen as f64 / w_elapsed
                } else {
                    0.0
                }
            } else if window_starts.len() >= 2 && start < window_starts.len() {
                let w_elapsed = window_starts[window_starts.len() - 1]
                    .duration_since(window_starts[start])
                    .as_secs_f64();
                if w_elapsed > 0.0 {
                    (window_starts.len() - start) as f64 / w_elapsed
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let m = WindowMetrics {
                window_idx: w,
                token_count: wlen,
                ngram_rep_rate: rep_rate,
                diversity,
                eos_count,
                tok_per_sec,
            };

            eprintln!(
                "  {:>8} {:>8} {:>10.4} {:>10.4} {:>10} {:>10.1}",
                format!("[{}-{}]", start, end),
                wlen,
                m.ngram_rep_rate,
                m.diversity,
                m.eos_count,
                m.tok_per_sec
            );

            windows.push(m);
        }

        // Summary statistics
        eprintln!();
        let (rep_mean, rep_max) = summary_stats(&windows, |m| m.ngram_rep_rate);
        let (div_mean, div_min) = summary_stats_min(&windows, |m| m.diversity);
        let (tps_mean, tps_min) = summary_stats_min(&windows, |m| m.tok_per_sec);

        // Token frequency analysis (top-20 most common)
        let mut freq: HashMap<u32, usize> = HashMap::new();
        for &t in &token_ids {
            *freq.entry(t).or_insert(0) += 1;
        }
        let mut freq_sorted: Vec<(u32, usize)> = freq.into_iter().collect();
        freq_sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let top_token_frac = freq_sorted[0].1 as f64 / n as f64;

        // Detect degeneration patterns
        let mut flags: Vec<&str> = Vec::new();
        if rep_max > 0.5 {
            flags.push("HIGH_REPETITION");
        }
        if div_min < 0.1 {
            flags.push("COLLAPSED_DIVERSITY");
        }
        if top_token_frac > 0.15 {
            flags.push("DOMINANT_TOKEN");
        }
        // Check if throughput dropped >20% between first and last full window
        if windows.len() >= 3 {
            let first_tps = windows[1].tok_per_sec; // skip window 0 (includes prefill tail)
            let last_tps = windows[windows.len() - 1].tok_per_sec;
            if last_tps > 0.0 && first_tps > 0.0 && (last_tps / first_tps) < 0.8 {
                flags.push("THROUGHPUT_DEGRADATION");
            }
        }

        eprintln!("  Summary:");
        eprintln!("    4gram repetition: mean={rep_mean:.4}  max={rep_max:.4}");
        eprintln!("    Token diversity:  mean={div_mean:.4}  min={div_min:.4}");
        eprintln!("    Throughput:       mean={tps_mean:.1}  min={tps_min:.1} tok/s");
        eprintln!(
            "    Top token: id={} freq={}/{} ({:.1}%)",
            freq_sorted[0].0,
            freq_sorted[0].1,
            n,
            top_token_frac * 100.0
        );

        if flags.is_empty() {
            eprintln!("    VERDICT: STABLE");
        } else {
            eprintln!("    VERDICT: UNSTABLE — {}", flags.join(", "));
        }

        // Machine-readable output
        println!(
            "BENCH_STABILITY {name} tokens={n} rep_mean={rep_mean:.4} rep_max={rep_max:.4} \
             div_mean={div_mean:.4} div_min={div_min:.4} tps_mean={tps_mean:.1} \
             tps_min={tps_min:.1} top_tok_frac={top_token_frac:.4} flags={}",
            if flags.is_empty() {
                "none".to_string()
            } else {
                flags.join(",")
            }
        );

        eprintln!();
    }

    eprintln!("[bench_stability] Done.");
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[allow(dead_code)]
struct WindowMetrics {
    window_idx: usize,
    token_count: usize,
    ngram_rep_rate: f64,
    diversity: f64,
    eos_count: usize,
    tok_per_sec: f64,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn ngram_repetition_rate(tokens: &[u32], n: usize) -> f64 {
    if tokens.len() < n {
        return 0.0;
    }
    let mut seen: std::collections::HashSet<&[u32]> = std::collections::HashSet::new();
    let mut repeated = 0usize;
    let total = tokens.len() - n + 1;
    for i in 0..total {
        let gram = &tokens[i..i + n];
        if !seen.insert(gram) {
            repeated += 1;
        }
    }
    repeated as f64 / total as f64
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn summary_stats(windows: &[WindowMetrics], f: impl Fn(&WindowMetrics) -> f64) -> (f64, f64) {
    let vals: Vec<f64> = windows.iter().map(&f).collect();
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, max)
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn summary_stats_min(windows: &[WindowMetrics], f: impl Fn(&WindowMetrics) -> f64) -> (f64, f64) {
    let vals: Vec<f64> = windows.iter().map(&f).filter(|v| *v > 0.0).collect();
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    (mean, min)
}
