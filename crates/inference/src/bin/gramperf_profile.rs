//! Profiling harness for issue #734 (grammar-constrained decode ~4.5x
//! slower per token when a schema exceeds `MAX_GRAMMAR_STATES`).
//!
//! PROFILING ONLY — no optimization landed here. This binary measures four
//! things over a real greedy generation on Qwen3.5-0.8B-Q4 (Metal):
//!
//!   a. per-decode-step wall time split: forward pass (residual) vs grammar
//!      masking (precomputed-hit / `mask_by_simulation` fallback / the
//!      context-dependent recheck loop) vs PDA `advance`.
//!   b. state revisit rate: distinct grammar states visited vs total decode
//!      steps, and the hit rate a lazy memoized mask cache would have had.
//!   c. compile-time breakdown: BFS state enumeration vs eager
//!      `VocabPartition::build`, and reachable states found vs the 256 cap.
//!   d. mask density: how many of the ~250K vocab tokens are masked per step.
//!
//! Env:
//!   LATTICE_MODEL_DIR      Q4 model dir (default ~/.lattice/models/_fresh616_q4)
//!   LATTICE_TOKENIZER_DIR  tokenizer dir (default ~/.lattice/models/qwen3.5-0.8b)
//!   GRAMPERF_RUNS          grammar-constrained generation runs (default 3)
//!   GRAMPERF_MAX_TOKENS    max_new_tokens per run (default 150)
//!   GRAMPERF_PROBE_CAP     uncapped BFS probe ceiling (default 1024; bounded
//!                          because probe cost scales ~linearly with states)
//!
//! Output: machine-readable `RESULT key=value ...` lines to stdout, one
//! block per measurement, consumed by hand for the profiling report (this is
//! a diagnostic tool, not a bench-gate harness).

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("Requires macOS + metal-gpu feature.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        if let Err(e) = run() {
            eprintln!("gramperf_profile failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::grammar::engine::{
        enable_mask_profiling, last_build_profile, probe_reachable_states, take_mask_profile,
    };
    use lattice_inference::grammar::pda::StackFrame;
    use lattice_inference::grammar::{GrammarEngine, GrammarSpec};
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::BpeTokenizer;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    let home = std::env::var("HOME")?;
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/_fresh616_q4"));
    let dir = std::path::Path::new(&model_dir_str);
    let tokenizer_dir_str = std::env::var("LATTICE_TOKENIZER_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let tokenizer_dir = std::path::Path::new(&tokenizer_dir_str);

    let runs: usize = std::env::var("GRAMPERF_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let max_tokens: usize = std::env::var("GRAMPERF_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let probe_cap: usize = std::env::var("GRAMPERF_PROBE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    eprintln!("[gramperf] loading {model_dir_str} (Q4) / tokenizer {tokenizer_dir_str}");
    let cfg = Qwen35Config::from_model_dir(dir).map_err(|e| format!("config.json load: {e}"))?;
    let mut metal =
        MetalQwen35State::from_q4_dir(dir, &tokenizer_dir.join("tokenizer.json"), &cfg, 4096)
            .map_err(|e| format!("Metal Q4 init: {e}"))?;
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_dir.join("tokenizer.json"))?;

    // #734-shape schema, reconstructed (the exact schema is not in the repo
    // or the issue text): 4 levels of nested objects (level1..level4), 3
    // array fields (tags, items, flags), 6 string-enum fields (6 members
    // each: status, category, priority, region, mode, role).
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "level3": {
                                "type": "object",
                                "properties": {
                                    "level4": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "enum": ["active", "inactive", "pending", "archived", "deleted", "draft"]},
                                            "value": {"type": "integer"}
                                        },
                                        "required": ["status", "value"]
                                    }
                                },
                                "required": ["level4"]
                            },
                            "category": {"type": "string", "enum": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]}
                        },
                        "required": ["level3", "category"]
                    },
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["level2", "tags"]
            },
            "items": {"type": "array", "items": {"type": "integer"}},
            "flags": {"type": "array", "items": {"type": "boolean"}},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent", "critical", "none"]},
            "region": {"type": "string", "enum": ["us", "eu", "apac", "latam", "mea", "other"]},
            "mode": {"type": "string", "enum": ["sync", "async", "batch", "stream", "manual", "auto"]},
            "role": {"type": "string", "enum": ["admin", "user", "guest", "owner", "viewer", "editor"]}
        },
        "required": ["level1", "items", "flags", "priority", "region", "mode", "role"]
    });
    let spec = GrammarSpec::JsonSchema(schema);

    eprintln!("[gramperf] tokenizer vocab_size={}", cfg.vocab_size);
    let vocab_bytes = tokenizer.vocab_bytes(cfg.vocab_size)?;

    // (c) compile-time breakdown: capped (production) build.
    let compile_t0 = Instant::now();
    let engine = GrammarEngine::new(&spec, vocab_bytes.clone())?;
    let compile_total_ns = compile_t0.elapsed().as_nanos() as u64;
    let build_profile = last_build_profile();
    println!(
        "RESULT kind=compile compile_total_ns={} bfs_ns={} partition_build_ns={} \
         reachable_states_capped={} state_cap={} exceeds_state_budget={}",
        compile_total_ns,
        build_profile.bfs_ns,
        build_profile.partition_build_ns,
        build_profile.reachable_states,
        build_profile.capped_states,
        engine.exceeds_state_budget(),
    );

    // (c) uncapped-ish BFS probe: how many states does the schema actually
    // reach past the 256 cap? Bounded at `probe_cap` — probe cost scales
    // ~linearly with states explored, so this is a lower bound if the true
    // count exceeds `probe_cap`.
    let probe_t0 = Instant::now();
    let probed_states = probe_reachable_states(&spec, &vocab_bytes, probe_cap)?;
    let probe_ns = probe_t0.elapsed().as_nanos() as u64;
    let probe_truncated = probed_states >= probe_cap;
    println!(
        "RESULT kind=probe probe_cap={probe_cap} probed_states={probed_states} \
         probe_truncated={probe_truncated} probe_ns={probe_ns}"
    );

    let engine = Arc::new(engine);

    let prompt = "Generate a JSON object describing a record with a deeply \
                  nested configuration, some tags and numeric items, and a \
                  handful of categorical flags.";

    // (a) grammar-constrained runs: real forward pass + real mask_logits +
    // real advance, with mask profiling enabled to attribute decode-step
    // cost to precomputed-hit / recheck / fallback / advance / residual
    // (forward pass + sampling).
    let mut grammar_token_ids: Vec<u32> = Vec::new();
    for i in 0..runs {
        metal.reset_state();
        let gen_cfg = GenerateConfig {
            max_new_tokens: max_tokens,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
            stop_token_ids: vec![],
            enable_thinking: false,
            enable_mtp: None,
            grammar: Some(Arc::clone(&engine)),
            stop_strings: vec![],
            reasoning_budget: None,
            logprobs: None,
        };
        enable_mask_profiling();
        let t0 = Instant::now();
        let output = metal.generate(prompt, &tokenizer, &gen_cfg)?;
        let wall_ns = t0.elapsed().as_nanos() as u64;
        let mp = take_mask_profile();
        let steps = output.generated_tokens.max(1) as u64;
        let mask_ns = mp.precomputed_ns + mp.context_recheck_ns + mp.fallback_ns;
        let residual_ns = wall_ns.saturating_sub(mask_ns + mp.advance_ns);
        println!(
            "RESULT kind=grammar_run run={i} generated_tokens={} wall_ns={wall_ns} \
             wall_ns_per_step={} precomputed_calls={} precomputed_ns={} \
             precomputed_ns_per_step={} context_recheck_calls={} context_recheck_ns={} \
             context_recheck_ns_per_step={} fallback_calls={} fallback_ns={} \
             fallback_ns_per_step={} advance_calls={} advance_ns={} advance_ns_per_step={} \
             residual_ns_per_step={} stop_reason={:?} trie_build_ns={}",
            output.generated_tokens,
            wall_ns / steps,
            mp.precomputed_calls,
            mp.precomputed_ns,
            mp.precomputed_ns
                .checked_div(mp.precomputed_calls.max(1))
                .unwrap_or(0),
            mp.context_recheck_calls,
            mp.context_recheck_ns,
            mp.context_recheck_ns
                .checked_div(mp.context_recheck_calls.max(1))
                .unwrap_or(0),
            mp.fallback_calls,
            mp.fallback_ns,
            mp.fallback_ns
                .checked_div(mp.fallback_calls.max(1))
                .unwrap_or(0),
            mp.advance_calls,
            mp.advance_ns,
            mp.advance_ns / steps,
            residual_ns / steps,
            output.stop_reason,
            engine.trie_build_ns(),
        );
        if i == 0 {
            grammar_token_ids = output.token_ids.clone();
        }
    }

    // Unconstrained baseline (context, not the primary measurement): same
    // prompt, no grammar, greedy.
    {
        metal.reset_state();
        let gen_cfg = GenerateConfig {
            max_new_tokens: max_tokens,
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
            logprobs: None,
        };
        let t0 = Instant::now();
        let output = metal.generate(prompt, &tokenizer, &gen_cfg)?;
        let wall_ns = t0.elapsed().as_nanos() as u64;
        let steps = output.generated_tokens.max(1) as u64;
        println!(
            "RESULT kind=baseline_unconstrained generated_tokens={} wall_ns={wall_ns} \
             wall_ns_per_step={}",
            output.generated_tokens,
            wall_ns / steps,
        );
    }

    // (e) Multi-request cross-request state-revisit measurement. Gates the
    // GrammarEngine-lifetime memoized-mask decision from the profiling
    // report's ranked fix #2 (measurement only — no memo is implemented
    // here). M=5 sequential generations share the SAME `engine` Arc (the
    // production caching unit: one compiled schema reused across many
    // requests), each with different prompt wording so content-dependent
    // states (which enum member, which digits) can plausibly diverge
    // across requests while schema-literal scaffold states (JSON
    // punctuation, property-name keys) should still recur. A
    // `(stack, complete)` visit map persists across all M runs — a hit
    // means a later request landed on a state some earlier request already
    // visited, which a per-engine (not per-request) memoized mask cache
    // could have served from cache.
    {
        let multi_prompts = [
            "Generate a JSON object describing a record with a deeply \
             nested configuration, some tags and numeric items, and a \
             handful of categorical flags.",
            "Produce a JSON object for a different record with its own \
             nested configuration, different tags, different numeric \
             items, and different categorical flags.",
            "Emit a JSON object representing yet another record: nested \
             configuration, a fresh set of tags, fresh numeric items, and \
             fresh categorical flags.",
            "Return a JSON object for a new record instance with a \
             distinct nested configuration, distinct tags, distinct \
             numeric items, and distinct categorical flags.",
            "Write a JSON object capturing one more record: its own \
             nested configuration, its own tags, its own numeric items, \
             and its own categorical flags.",
        ];
        let m = multi_prompts.len();
        let mut cross_request_visits: HashMap<(Vec<StackFrame>, bool), u32> = HashMap::new();
        let mut total_steps = 0u64;
        let mut cross_request_hits = 0u64;

        for (run_idx, p) in multi_prompts.iter().enumerate() {
            metal.reset_state();
            let gen_cfg = GenerateConfig {
                max_new_tokens: max_tokens,
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                repetition_penalty: 1.0,
                seed: Some(42),
                stop_token_ids: vec![],
                enable_thinking: false,
                enable_mtp: None,
                grammar: Some(Arc::clone(&engine)),
                stop_strings: vec![],
                reasoning_budget: None,
                logprobs: None,
            };
            let output = metal.generate(p, &tokenizer, &gen_cfg)?;

            // Replay this run's token sequence through the SAME engine
            // (fresh GrammarState per run — one request = one decode
            // sequence — but `cross_request_visits` persists across runs)
            // to count revisits against states seen in EARLIER runs.
            let mut state = engine.initial_state();
            let mut new_states_this_run = 0u64;
            for &token_id in &output.token_ids {
                let key: (Vec<StackFrame>, bool) = (state.stack.clone(), state.complete);
                if cross_request_visits.contains_key(&key) {
                    cross_request_hits += 1;
                } else {
                    new_states_this_run += 1;
                }
                *cross_request_visits.entry(key).or_insert(0) += 1;
                total_steps += 1;
                if !engine.advance(&mut state, token_id) {
                    break;
                }
            }
            println!(
                "RESULT kind=multi_request_run run={run_idx} generated_tokens={} \
                 new_states={new_states_this_run}",
                output.generated_tokens,
            );
        }

        let hit_rate = if total_steps > 0 {
            cross_request_hits as f64 / total_steps as f64
        } else {
            0.0
        };
        println!(
            "RESULT kind=multi_request_summary requests={m} total_steps={total_steps} \
             cross_request_hits={cross_request_hits} cross_request_hit_rate={hit_rate:.4} \
             distinct_states_total={} trie_build_ns={}",
            cross_request_visits.len(),
            engine.trie_build_ns(),
        );
    }

    // (b) + (d): replay the exact token sequence from grammar run 0 through
    // the same engine (no GPU needed) to get state-revisit / memoization-hit
    // and mask-density stats faithfully, without perturbing the timed runs
    // above with density-counting overhead.
    {
        let mut state = engine.initial_state();
        let mut visits: HashMap<(Vec<StackFrame>, bool), u32> = HashMap::new();
        let mut steps = 0u64;
        let mut cache_hits = 0u64;
        let mut density_sum = 0f64;
        let mut density_min = f64::MAX;
        let mut density_max = f64::MIN;
        let vocab_size = cfg.vocab_size;
        let mut dummy_logits = vec![0.0f32; vocab_size];

        for &token_id in &grammar_token_ids {
            let key: (Vec<StackFrame>, bool) = (state.stack.clone(), state.complete);
            let seen_before = visits.contains_key(&key);
            *visits.entry(key).or_insert(0) += 1;
            if seen_before {
                cache_hits += 1;
            }

            for l in dummy_logits.iter_mut() {
                *l = 0.0;
            }
            engine.mask_logits(&mut state, &mut dummy_logits);
            let masked = dummy_logits
                .iter()
                .filter(|&&l| l == f32::NEG_INFINITY)
                .count();
            let density = masked as f64 / vocab_size as f64;
            density_sum += density;
            density_min = density_min.min(density);
            density_max = density_max.max(density);
            steps += 1;

            if !engine.advance(&mut state, token_id) {
                break;
            }
        }

        let distinct_states = visits.len() as u64;
        let hit_rate = if steps > 0 {
            cache_hits as f64 / steps as f64
        } else {
            0.0
        };
        let density_mean = if steps > 0 {
            density_sum / steps as f64
        } else {
            0.0
        };
        println!(
            "RESULT kind=replay steps={steps} distinct_states={distinct_states} \
             cache_hits={cache_hits} would_be_hit_rate={hit_rate:.4} \
             mask_density_mean={density_mean:.4} mask_density_min={density_min:.4} \
             mask_density_max={density_max:.4} vocab_size={vocab_size}"
        );
    }

    Ok(())
}
