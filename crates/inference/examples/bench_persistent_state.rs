/// Baseline benchmark: multi-turn chat latency under the current full-replay-per-turn path.
///
/// Each chat turn resets state and re-prefills the entire conversation history from scratch.
/// This establishes BEFORE numbers for the persistent KV/GDN state optimization.
/// Run this before any implementation changes to capture the baseline.
///
/// After the optimization lands, re-run to compare. The same iteration/prompt/config
/// parameters must be kept identical for a valid before/after comparison.
///
/// Usage:
///   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.6-27b-q4 \
///   LATTICE_TOKENIZER_DIR=~/.lattice/models/qwen3.6-27b \
///   cargo run --release --example bench_persistent_state -p lattice-inference --features "f16,metal-gpu"
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_persistent_state requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    use lattice_inference::forward::metal_qwen35::{
        ChatMessage, MetalQwen35State, format_chat_template,
    };
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::bpe::BpeTokenizer;
    use std::time::Instant;

    let home = std::env::var("HOME").expect("HOME not set");
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b-q4"));
    let tokenizer_dir_str = std::env::var("LATTICE_TOKENIZER_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b"));

    let dir = std::path::Path::new(&model_dir_str);
    let tokenizer_path = std::path::Path::new(&tokenizer_dir_str).join("tokenizer.json");

    let cfg = if dir.join("config.json").exists() {
        Qwen35Config::from_config_json(&dir.join("config.json")).expect("parse config.json")
    } else {
        Qwen35Config::qwen36_27b()
    };

    eprintln!("[bench] Loading Q4 model from {} ...", dir.display());
    let t_load = Instant::now();
    let mut state = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, 4096)
        .expect("from_q4_dir failed — MSL compile or weight load error");
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path).expect("load tokenizer");
    eprintln!(
        "[bench] Model loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );

    // Fixed prompts from spec §6 — do not change for valid before/after comparison.
    let user_prompts = [
        "Answer with one short sentence: what is a cache?",
        "Now give one benefit.",
        "Now give one risk.",
    ];
    let n_turns = user_prompts.len();

    let im_end_id = tokenizer.special_token_id("<|im_end|>").unwrap_or(151645);
    // Deterministic config from spec §6 — do not change for valid before/after comparison.
    let gen_cfg = GenerateConfig {
        max_new_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![im_end_id],
        enable_thinking: false,
    };
    let system_msg = ChatMessage::system("You are a helpful assistant. Be concise.");

    // Warmup: one full 3-turn conversation to prime Metal GPU kernels and cache.
    eprintln!("[bench] Warmup (1 iteration)...");
    {
        let mut history: Vec<ChatMessage> = vec![system_msg.clone()];
        for &user_msg in &user_prompts {
            history.push(ChatMessage::user(user_msg));
            let full_prompt = format_chat_template(&history);
            let result = state.generate(&full_prompt, &tokenizer, &gen_cfg);
            history.push(ChatMessage::assistant(result.text.trim().to_string()));
        }
    }

    let n_iters = 3usize;
    // [iter][turn] = ms
    let mut turn_times_ms: Vec<Vec<f64>> = Vec::with_capacity(n_iters);
    let mut turn_prompt_tokens: Vec<Vec<usize>> = Vec::with_capacity(n_iters);
    let mut turn_gen_tokens: Vec<Vec<usize>> = Vec::with_capacity(n_iters);
    // Capture first-iter outputs for reproducibility note in baseline.md.
    let mut first_iter_outputs: Vec<String> = Vec::with_capacity(n_turns);

    eprintln!("[bench] Measuring ({n_iters} iterations)...");
    for iter in 0..n_iters {
        eprintln!("[bench] Iteration {}/{}", iter + 1, n_iters);
        let mut history: Vec<ChatMessage> = vec![system_msg.clone()];
        let mut iter_times: Vec<f64> = Vec::with_capacity(n_turns);
        let mut iter_prompt: Vec<usize> = Vec::with_capacity(n_turns);
        let mut iter_gen: Vec<usize> = Vec::with_capacity(n_turns);

        for (turn_idx, &user_msg) in user_prompts.iter().enumerate() {
            history.push(ChatMessage::user(user_msg));

            // Full chat history formatted into a single prompt string.
            let t_fmt = Instant::now();
            let full_prompt = format_chat_template(&history);
            let fmt_ms = t_fmt.elapsed().as_secs_f64() * 1000.0;

            // Current behavior: generate() tokenizes, resets state, prefills from pos 0, decodes.
            // This is the full-replay cost we're optimizing away.
            let t_gen = Instant::now();
            let result = state.generate(&full_prompt, &tokenizer, &gen_cfg);
            let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
            let total_ms = fmt_ms + gen_ms;

            iter_times.push(total_ms);
            iter_prompt.push(result.prompt_tokens);
            iter_gen.push(result.generated_tokens);

            if iter == 0 {
                first_iter_outputs.push(result.text.trim().to_string());
            }

            eprintln!(
                "  iter={} turn={} fmt={:.1}ms gen={:.2}ms total={:.0}ms | \
                 prompt={} gen={} tokens",
                iter + 1,
                turn_idx + 1,
                fmt_ms,
                gen_ms,
                total_ms,
                result.prompt_tokens,
                result.generated_tokens,
            );

            // Print per-iteration per-turn structured line for scripted capture.
            println!(
                "BENCH_REPLAY_RAW iter={} turn={} fmt_ms={fmt_ms:.2} gen_ms={gen_ms:.2} \
                 total_ms={total_ms:.2} prompt_tokens={} gen_tokens={}",
                iter + 1,
                turn_idx + 1,
                result.prompt_tokens,
                result.generated_tokens,
            );

            history.push(ChatMessage::assistant(result.text.trim().to_string()));
        }

        turn_times_ms.push(iter_times);
        turn_prompt_tokens.push(iter_prompt);
        turn_gen_tokens.push(iter_gen);
    }

    // Aggregate stats per turn.
    eprintln!("\n[bench] Per-turn summary ({n_iters} iters, mode=full_replay):");
    let mut total_per_iter: Vec<f64> = vec![0.0; n_iters];

    for turn in 0..n_turns {
        let times: Vec<f64> = (0..n_iters).map(|i| turn_times_ms[i][turn]).collect();
        let mean = times.iter().sum::<f64>() / n_iters as f64;
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[n_iters / 2];
        let variance = times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_iters as f64;
        let std_dev = variance.sqrt();
        let cv = if mean > 0.0 {
            std_dev / mean * 100.0
        } else {
            0.0
        };
        let p_tok = turn_prompt_tokens[0][turn];
        let g_tok = turn_gen_tokens[0][turn];

        eprintln!(
            "  turn {}: median={median:.0}ms  mean={mean:.0}±{std_dev:.0}ms  CV={cv:.1}%  \
             prompt={p_tok}tok gen={g_tok}tok",
            turn + 1,
        );
        println!(
            "BENCH_REPLAY_TURN mode=full_replay turn={} \
             median_ms={median:.1} mean_ms={mean:.1} std_dev_ms={std_dev:.1} cv_pct={cv:.1} \
             prompt_tokens={p_tok} gen_tokens={g_tok}",
            turn + 1,
        );

        if cv > 20.0 {
            eprintln!(
                "  ⚠️  turn {} CV={cv:.1}% > 20% — high noise, results may be unreliable",
                turn + 1
            );
        }

        for i in 0..n_iters {
            total_per_iter[i] += turn_times_ms[i][turn];
        }
    }

    // Total conversation stats.
    let t_mean = total_per_iter.iter().sum::<f64>() / n_iters as f64;
    let mut t_sorted = total_per_iter.clone();
    t_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t_median = t_sorted[n_iters / 2];
    let t_var = total_per_iter
        .iter()
        .map(|x| (x - t_mean).powi(2))
        .sum::<f64>()
        / n_iters as f64;
    let t_std = t_var.sqrt();

    eprintln!(
        "\n  total conversation ({n_turns} turns): median={t_median:.0}ms  mean={t_mean:.0}±{t_std:.0}ms"
    );
    println!(
        "BENCH_REPLAY_TOTAL mode=full_replay turns={n_turns} iters={n_iters} \
         median_ms={t_median:.1} mean_ms={t_mean:.1} std_dev_ms={t_std:.1}"
    );

    // First-iteration sample outputs for reproducibility verification.
    eprintln!("\n[bench] First-iteration sample outputs (deterministic, temperature=0.0):");
    for (i, out) in first_iter_outputs.iter().enumerate() {
        eprintln!("  turn {}: {:?}", i + 1, out);
    }
}
