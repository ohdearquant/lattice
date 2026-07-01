//! ADR-064 Phase-0: decode slope/intercept fit harness.
//!
//! Sweeps a context grid, runs greedy decode at each point, and emits
//! per-(ctx, tokens, ms) lines that the post-processor fits with a
//! robust linear model (Theil-Sen + bootstrap CI).
//!
//! # Determinism
//!
//! `MetalQwen35State::generate()` (not `chat_completion`) is called with a raw
//! continuation prompt — no chat template — so the model is never in a completed
//! conversation state and won't emit `<|im_end|>` naturally.  We additionally
//! set `stop_token_ids: vec![]` and accept that `eos_token_id` (248044) remains
//! a stop condition built into `generate()`.  In practice, at all grid points
//! {64, 256, 512} the greedy continuation of a padded filler text reaches
//! `max_new_tokens` before EOS (verified in smoke run).  If EOS fires early at
//! a specific context, the actual token count is reported and the post-processor
//! uses it, so measurements remain comparable on a per-token basis.
//!
//! Forcing decode past EOS would require modifying the forward hot path, which
//! is forbidden by issue #168 hard constraints.  Honest-nil applies: CI will be
//! wider where EOS fires, and the fit notes the actual token counts.
//!
//! # Dispatch / command-buffer counters
//!
//! The engine does not expose dispatch or command-buffer counts through the
//! public `MetalQwen35State` API.  Those fields are emitted as `null` in the
//! final JSON per ADR-064 §Honest-nil policy.
//!
//! # Env vars
//!
//!   LATTICE_MODEL_DIR     model dir  (default ~/.lattice/models/qwen3.5-0.8b)
//!   LATTICE_TOKENIZER_DIR tokenizer  (default LATTICE_MODEL_DIR)
//!   SLOPEFIT_CONTEXTS     space-separated ctx values (default "64 256 512")
//!   SLOPEFIT_WARMUP       warmup tokens per point    (default 32)
//!   SLOPEFIT_MEASURE      measured tokens per point  (default 256)
//!   SLOPEFIT_REPEATS      in-process repeats         (default 7)
//!   SLOPEFIT_FULL         set to "1" to add `[1024,2048,4096,8192,16384]` to grid
//!
//! # Output (stdout)
//!
//!   One tagged line per measured repeat:
//!     SLOPEFIT `ctx=<N> tokens=<actual> warmup_ms=0.0 measure_ms=<f> rep=<i>`
//!
//!   Plus `SLOPEFIT_META` lines carrying data the post-processor needs for its
//!   ADR-064 TBV self-checks (KV-cache-cap corruption, token-count sanity):
//!     `SLOPEFIT_META kv_cache_len=<N> warmup=<N> measure=<N> repeats=<N>` (once)
//!     `SLOPEFIT_META ctx=<N> actual_prompt_tokens=<N>` (once per context)
//!
//!   The post-processor (`scripts/bench_decode_slopefit.py`) reads these lines
//!   and produces the final JSON.

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("bench_decode_slopefit requires macOS + --features metal-gpu.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = run() {
            eprintln!("bench_decode_slopefit failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};

    let home = std::env::var("HOME")?;
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir_str);

    let warmup_tokens: usize = std::env::var("SLOPEFIT_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let measure_tokens: usize = std::env::var("SLOPEFIT_MEASURE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let repeats: usize = std::env::var("SLOPEFIT_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    // Build the context grid.  Default is the smoke grid {64, 256, 512}.
    // SLOPEFIT_FULL=1 appends the production tail {1024,2048,4096,8192,16384}.
    let mut grid: Vec<usize> = if let Ok(v) = std::env::var("SLOPEFIT_CONTEXTS") {
        v.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect()
    } else {
        vec![64, 256, 512]
    };
    if std::env::var("SLOPEFIT_FULL").as_deref() == Ok("1") {
        for &c in &[1024usize, 2048, 4096, 8192, 16384] {
            if !grid.contains(&c) {
                grid.push(c);
            }
        }
    }
    grid.sort_unstable();
    grid.dedup();

    let max_ctx = grid.iter().copied().max().unwrap_or(512);

    // Detect Q4 dir (same heuristic as bench_decode_ab).
    let is_q4_dir = !dir.join("model.safetensors").exists()
        && std::fs::read_dir(dir)
            .ok()
            .and_then(|mut entries| {
                entries.find(|e| {
                    e.as_ref()
                        .ok()
                        .and_then(|e| e.file_name().to_str().map(|n| n.ends_with(".q4")))
                        .unwrap_or(false)
                })
            })
            .is_some();

    let tokenizer_dir_str =
        std::env::var("LATTICE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir_str.clone());
    let tokenizer_dir = std::path::Path::new(&tokenizer_dir_str);

    eprintln!(
        "[slopefit] loading {model_dir_str} ({})",
        if is_q4_dir { "Q4" } else { "safetensors" }
    );

    // Tokenizer first — the KV cache must hold the *padded* prompt, and the pad
    // loop below overshoots `ctx` by up to one `base` block. Sizing from `max_ctx`
    // alone leaves the deepest grid point short by that overshoot and silently
    // truncates its decode (a milder echo of the tokens=1 corruption the previous
    // hard-coded 4096 cache produced at every point >= 4096). Build the deepest
    // prompt exactly as the measurement loop does and size from its real length.
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_dir.join("tokenizer.json"))?;

    // Raw continuation prompt — no chat template — so the model sees a partial
    // sentence and produces a continuation rather than a completed conversation.
    // This avoids the early EOS that chat_completion triggers via im_end.
    // The base text is intentionally left incomplete (no closing punctuation)
    // so greedy continuation runs as a prose generation, not a Q&A exchange.
    let base = "The quick brown fox jumps over the lazy dog and then continues \
                running through the meadow past the old stone wall while the sun \
                sets slowly over the distant mountains painting the sky in shades \
                of orange and gold as the evening breeze stirs the tall grass and \
                the river flows gently southward toward the ancient city where ";

    let mut deepest_prompt = String::new();
    while tokenizer.tokenize(&deepest_prompt).real_length < max_ctx {
        deepest_prompt.push_str(base);
    }
    let deepest_prompt_tokens = tokenizer.tokenize(&deepest_prompt).real_length;
    // Peak occupancy is prompt + the larger decode phase: warmup and measure each
    // run after their own reset_state(), so the cache never holds both at once.
    let cache_len = deepest_prompt_tokens + warmup_tokens.max(measure_tokens) + 16;

    // Load model — same dual ownership pattern as bench_decode_ab.
    let mut metal: MetalQwen35State;

    if is_q4_dir {
        let cfg = if dir.join("config.json").exists() {
            Qwen35Config::from_config_json(&dir.join("config.json"))
                .map_err(|e| format!("config.json parse: {e}"))?
        } else {
            Qwen35Config::qwen35_0_8b()
        };
        metal = MetalQwen35State::from_q4_dir(
            dir,
            &tokenizer_dir.join("tokenizer.json"),
            &cfg,
            cache_len,
        )
        .map_err(|e| format!("Metal Q4 init: {e}"))?;
    } else {
        let model = Qwen35Model::from_safetensors(dir).map_err(|e| format!("load model: {e}"))?;
        let cfg = model.config().clone();
        metal = MetalQwen35State::new(model.weights(), &cfg, cache_len)
            .map_err(|e| format!("init metal: {e}"))?;
    }

    // Greedy, stop_token_ids empty — EOS token (248044) still stops generation
    // via the hard-coded path in generate().  The prompt design above avoids
    // natural EOS in the first 256 tokens on the 0.8B model (verified empirically).
    let make_cfg = |n_tokens: usize| GenerateConfig {
        max_new_tokens: n_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: None,
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
    };

    eprintln!(
        "[slopefit] grid={grid:?} warmup={warmup_tokens} measure={measure_tokens} repeats={repeats}"
    );
    eprintln!(
        "[slopefit] kv_cache_len={cache_len} \
         (deepest_prompt={deepest_prompt_tokens} for max_ctx={max_ctx} + decode_horizon {})",
        warmup_tokens.max(measure_tokens)
    );

    // Emitted on stdout (not stderr) so the Python post-processor can read it
    // through the same pipe and run the KV-cache-cap TBV self-check.
    println!(
        "SLOPEFIT_META kv_cache_len={cache_len} warmup={warmup_tokens} measure={measure_tokens} repeats={repeats}"
    );

    for &ctx in &grid {
        // Pad prompt to approximately `ctx` tokens by repeating the base text.
        let mut prompt = String::new();
        while tokenizer.tokenize(&prompt).real_length < ctx {
            prompt.push_str(base);
        }
        let actual_prompt_tokens = tokenizer.tokenize(&prompt).real_length;

        eprintln!("[slopefit] ctx={ctx} actual_prompt_tokens={actual_prompt_tokens}");
        println!("SLOPEFIT_META ctx={ctx} actual_prompt_tokens={actual_prompt_tokens}");

        for rep in 0..repeats {
            // Warmup phase: unrecorded decode primes Metal pipeline caches.
            metal.reset_state();
            let _ = metal.generate(&prompt, &tokenizer, &make_cfg(warmup_tokens));

            // Measurement phase: fresh reset → identical KV state each repeat.
            metal.reset_state();
            let t = std::time::Instant::now();
            let result = metal.generate(&prompt, &tokenizer, &make_cfg(measure_tokens));
            let measure_ms = t.elapsed().as_secs_f64() * 1000.0;
            let actual_tokens = result.generated_tokens;

            println!(
                "SLOPEFIT ctx={ctx} tokens={actual_tokens} warmup_ms=0.0 measure_ms={measure_ms:.3} rep={rep}"
            );
        }
    }

    Ok(())
}
