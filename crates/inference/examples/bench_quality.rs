/// Benchmark: Hidden-state divergence and output quality for Pruned-8 and Pruned-12 masks.
///
/// Measures logit-space cosine similarity (proxy for hidden-state divergence) and greedy
/// next-token match rate against the unmodified baseline model.
///
/// Pruned-8  mask: GDN layers 12,16,20,24,28,32,36,40 removed (every 4th GDN in middle).
/// Pruned-12 mask: GDN layers 8,12,16,20,24,28,32,36,40,44,48,52 removed (evenly spread).
///
/// Gate (from critic spec): mean_cos >= 0.95 AND min_cos >= 0.85.
///
/// Usage:
///   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.6-27b-q4 \
///   LATTICE_TOKENIZER_DIR=~/.lattice/models/qwen3.6-27b \
///   cargo run --release --example bench_quality -p lattice-inference --features "f16,metal-gpu"
///
/// Optional: set LATTICE_QUALITY_SCORE=1 to also run score_layer_importance (~30 min extra).
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_quality requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    use lattice_inference::forward::metal_qwen35::{LayerPruningPlan, MetalQwen35State};
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};
    use std::time::Instant;

    let home = std::env::var("HOME").expect("HOME not set");
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b-q4"));
    let tokenizer_dir_str = std::env::var("LATTICE_TOKENIZER_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.6-27b"));
    let run_scorer = std::env::var("LATTICE_QUALITY_SCORE").is_ok();

    let dir = std::path::Path::new(&model_dir_str);
    let tokenizer_path = std::path::Path::new(&tokenizer_dir_str).join("tokenizer.json");

    let base_cfg = if dir.join("config.json").exists() {
        Qwen35Config::from_config_json(&dir.join("config.json")).expect("parse config.json")
    } else {
        Qwen35Config::qwen36_27b()
    };
    let num_layers = base_cfg.num_hidden_layers;

    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path).expect("load tokenizer");

    // 12 calibration prompts: diverse domains, ≤32 tokens
    let prompts: &[&str] = &[
        "The quick brown fox jumps over the lazy dog near the river.",
        "Apple announced new silicon chips with improved neural engine performance.",
        "Machine learning models require large amounts of training data and compute.",
        "The Federal Reserve raised interest rates to combat rising inflation.",
        "Scientists discovered a new species of deep-sea fish near the Mariana Trench.",
        "Rust provides memory safety without garbage collection at the systems level.",
        "The ancient Roman Empire fell in 476 CE after centuries of gradual decline.",
        "Climate change is causing unprecedented shifts in global weather patterns.",
        "Large language models demonstrate emergent capabilities at unprecedented scale.",
        "The human brain contains approximately 86 billion neurons and trillions of synapses.",
        "Paris is the capital city of France and a center of art and cultural heritage.",
        "Quantum computers use qubits to perform calculations exponentially faster than classical.",
    ];

    // Tokenize all prompts; keep only real (non-padded) token IDs
    let tokenized: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            let t = tokenizer.tokenize(p);
            t.input_ids[..t.real_length].to_vec()
        })
        .collect();

    // Short token sequences for score_layer_importance (≤8 tokens)
    let scoring_prompts: Vec<Vec<u32>> = tokenized[..4]
        .iter()
        .map(|ids| ids[..ids.len().min(8)].to_vec())
        .collect();

    // Pruning masks — GDN layers only, type-balanced is not needed since both configs
    // are pure-GDN and the critic requested these specific indices
    let pruned_8_layers: Vec<usize> = vec![12, 16, 20, 24, 28, 32, 36, 40];
    let pruned_12_layers: Vec<usize> = vec![8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52];

    let make_mask = |prune: &[usize]| -> Vec<bool> {
        let mut m = vec![true; num_layers];
        for &i in prune {
            m[i] = false;
        }
        m
    };

    // Numerically stable cosine similarity using f64 accumulators
    let cosine_sim = |a: &[f32], b: &[f32]| -> f32 {
        let (mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            aa += x as f64 * x as f64;
            bb += y as f64 * y as f64;
        }
        let denom = aa.sqrt() * bb.sqrt();
        if denom < 1e-9 {
            0.0f32
        } else {
            (dot / denom) as f32
        }
    };

    let argmax = |v: &[f32]| -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    };

    // ---- BASELINE ----
    eprintln!("\n[bench_quality] Loading baseline model ({num_layers} active layers)...");
    let t0 = Instant::now();
    let mut state = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &base_cfg, 4096)
        .expect("baseline load failed");
    eprintln!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    eprintln!(
        "  Running baseline forward passes ({} prompts)...",
        prompts.len()
    );
    let mut baseline_logits: Vec<Vec<f32>> = Vec::with_capacity(prompts.len());
    let mut baseline_next_tok: Vec<u32> = Vec::with_capacity(prompts.len());

    for (i, ids) in tokenized.iter().enumerate() {
        state.reset_state();
        let logits = state.forward_prefill(ids);
        let tok = argmax(&logits);
        baseline_next_tok.push(tok);
        baseline_logits.push(logits);
        eprintln!("    prompt {:2}: len={:3}  argmax_tok={tok}", i, ids.len());
    }

    // Optional: data-driven layer importance
    let scoring_result: Option<LayerPruningPlan> = if run_scorer {
        eprintln!(
            "\n[bench_quality] Running score_layer_importance ({} prompts, prune_layers=12).",
            scoring_prompts.len()
        );
        eprintln!(
            "  NOTE: This runs {} forward passes. Estimated time: 20-40 min.",
            num_layers * scoring_prompts.len()
        );
        let t_score = Instant::now();
        match state.score_layer_importance(&scoring_prompts, 12) {
            Ok(plan) => {
                eprintln!(
                    "  Done in {:.0}s. Top candidates (highest cosine = least important):",
                    t_score.elapsed().as_secs_f64()
                );
                for score in plan.scores.iter().take(16) {
                    eprintln!(
                        "    layer {:2}  type={:?}  mean_cos={:.6}  importance={:.6}",
                        score.layer_idx, score.layer_type, score.mean_cosine, score.importance
                    );
                }
                if let Some(w) = &plan.warning {
                    eprintln!("  WARN: {w}");
                }
                Some(plan)
            }
            Err(e) => {
                eprintln!("  score_layer_importance error: {e}");
                None
            }
        }
    } else {
        eprintln!(
            "\n[bench_quality] score_layer_importance skipped (set LATTICE_QUALITY_SCORE=1 to enable, ~30 min)."
        );
        None
    };

    drop(state);

    // ---- PRUNED-8 ----
    eprintln!(
        "\n[bench_quality] Loading pruned-8 model (layers {:?} removed)...",
        pruned_8_layers
    );
    let mut cfg8 = base_cfg.clone();
    cfg8.apply_layer_mask(make_mask(&pruned_8_layers));
    let active8 = cfg8.num_active_layers();
    let t0 = Instant::now();
    let mut state8 = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg8, 4096)
        .expect("pruned-8 load failed");
    eprintln!(
        "  Loaded in {:.1}s  (active={active8})",
        t0.elapsed().as_secs_f64()
    );

    let mut cos_8: Vec<f32> = Vec::with_capacity(prompts.len());
    let mut match_8: Vec<bool> = Vec::with_capacity(prompts.len());

    for (i, ids) in tokenized.iter().enumerate() {
        state8.reset_state();
        let logits = state8.forward_prefill(ids);
        let tok = argmax(&logits);
        let cos = cosine_sim(&logits, &baseline_logits[i]);
        let matched = tok == baseline_next_tok[i];
        cos_8.push(cos);
        match_8.push(matched);
        eprintln!(
            "    prompt {:2}: cos={:.6}  tok_match={}  ({} vs {})",
            i, cos, matched, tok, baseline_next_tok[i]
        );
    }
    drop(state8);

    // ---- PRUNED-12 ----
    eprintln!(
        "\n[bench_quality] Loading pruned-12 model (layers {:?} removed)...",
        pruned_12_layers
    );
    let mut cfg12 = base_cfg.clone();
    cfg12.apply_layer_mask(make_mask(&pruned_12_layers));
    let active12 = cfg12.num_active_layers();
    let t0 = Instant::now();
    let mut state12 = MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg12, 4096)
        .expect("pruned-12 load failed");
    eprintln!(
        "  Loaded in {:.1}s  (active={active12})",
        t0.elapsed().as_secs_f64()
    );

    let mut cos_12: Vec<f32> = Vec::with_capacity(prompts.len());
    let mut match_12: Vec<bool> = Vec::with_capacity(prompts.len());

    for (i, ids) in tokenized.iter().enumerate() {
        state12.reset_state();
        let logits = state12.forward_prefill(ids);
        let tok = argmax(&logits);
        let cos = cosine_sim(&logits, &baseline_logits[i]);
        let matched = tok == baseline_next_tok[i];
        cos_12.push(cos);
        match_12.push(matched);
        eprintln!(
            "    prompt {:2}: cos={:.6}  tok_match={}  ({} vs {})",
            i, cos, matched, tok, baseline_next_tok[i]
        );
    }
    drop(state12);

    // ---- SUMMARISE ----
    let n = prompts.len() as f32;
    let mean8 = cos_8.iter().sum::<f32>() / n;
    let min8 = cos_8.iter().cloned().fold(f32::INFINITY, f32::min);
    let max8 = cos_8.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let match_rate8 = match_8.iter().filter(|&&x| x).count() as f32 / n;

    let mean12 = cos_12.iter().sum::<f32>() / n;
    let min12 = cos_12.iter().cloned().fold(f32::INFINITY, f32::min);
    let max12 = cos_12.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let match_rate12 = match_12.iter().filter(|&&x| x).count() as f32 / n;

    let gate8 = mean8 >= 0.95 && min8 >= 0.85;
    let gate12 = mean12 >= 0.95 && min12 >= 0.85;

    println!(
        "BENCH_QUALITY_HEADER config,active_layers,mean_cos,min_cos,max_cos,tok_match_rate,gate_pass"
    );
    println!(
        "BENCH_QUALITY_RESULT pruned-8,{active8},{mean8:.6},{min8:.6},{max8:.6},{match_rate8:.3},{}",
        gate8
    );
    println!(
        "BENCH_QUALITY_RESULT pruned-12,{active12},{mean12:.6},{min12:.6},{max12:.6},{match_rate12:.3},{}",
        gate12
    );

    eprintln!("\n=== QUALITY SUMMARY ===");
    eprintln!(
        "Pruned-8 ({active8} active):   mean_cos={mean8:.4}  min={min8:.4}  max={max8:.4}  tok_match={:.0}%  GATE={}",
        match_rate8 * 100.0,
        if gate8 { "PASS" } else { "FAIL" }
    );
    eprintln!(
        "Pruned-12 ({active12} active):  mean_cos={mean12:.4}  min={min12:.4}  max={max12:.4}  tok_match={:.0}%  GATE={}",
        match_rate12 * 100.0,
        if gate12 { "PASS" } else { "FAIL" }
    );

    // Per-prompt table
    eprintln!("\n=== PER-PROMPT TABLE ===");
    eprintln!(
        "{:>4}  {:>8}  {:>8}  {:>7}  {:>7}",
        "idx", "cos_p8", "cos_p12", "match_p8", "match_p12"
    );
    for i in 0..prompts.len() {
        eprintln!(
            "{:>4}  {:>8.6}  {:>8.6}  {:>8}  {:>9}",
            i,
            cos_8[i],
            cos_12[i],
            if match_8[i] { "yes" } else { "no" },
            if match_12[i] { "yes" } else { "no" }
        );
    }

    // Scorer comparison
    if let Some(plan) = scoring_result {
        let scored_prune_idxs: Vec<usize> = (0..num_layers)
            .filter(|&i| !plan.recommended_mask[i])
            .collect();
        let h8: std::collections::HashSet<usize> = pruned_8_layers.iter().cloned().collect();
        let h12: std::collections::HashSet<usize> = pruned_12_layers.iter().cloned().collect();
        let scored_set: std::collections::HashSet<usize> =
            scored_prune_idxs.iter().cloned().collect();
        let overlap8 = h8.intersection(&scored_set).count();
        let overlap12 = h12.intersection(&scored_set).count();

        eprintln!("\n=== SCORED vs HEURISTIC ===");
        eprintln!("  Scored top-12 prune indices: {scored_prune_idxs:?}");
        eprintln!("  Heuristic pruned-8 indices:  {:?}", pruned_8_layers);
        eprintln!("  Heuristic pruned-12 indices: {:?}", pruned_12_layers);
        eprintln!("  Overlap (heuristic-8 ∩ scored-12):  {overlap8}/8");
        eprintln!("  Overlap (heuristic-12 ∩ scored-12): {overlap12}/12");
    }

    eprintln!("\n[bench_quality] Done.");
}
