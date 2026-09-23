//! GDN-recurrence-isolating prefill A/B bench harness (issue #175, Phase B).
//!
//! Measures GDN recurrence dispatch time ONLY (conv1d + per-token recurrence on the
//! serial path; the `gdn_chunk_*` C32 dispatch family on the chunked path) for the
//! 0.8B Qwen3.5 model, isolated from the surrounding O(n^2) full-attention layers and
//! from the GDN layers' own non-recurrence GEMMs (QKV/Z projections, out_proj, MLP).
//! This is the INSTRUMENT for the ADR-064 #175 acceptance gate (recurrence speedup vs
//! sequential scan >=5x@4K, >=10x@16K) — see the bench methodology notes in the PR /
//! bench docs for the full spec this implements. Every number this bin prints is
//! PROVISIONAL until a maintainer re-verifies the harness fresh on an idle machine.
//!
//! Isolation method: dedicated command-buffer CPU wall-clock timing (spec's preferred
//! method 1). `MetalQwen35State::forward_prefill_chunk_gdn_isolated` (metal_qwen35.rs,
//! `bench-internals`-gated) is a structural copy of the production
//! `forward_prefill_batched_chunk` with command-buffer boundaries inserted immediately
//! before/after each GDN layer's recurrence dispatch — same kernels, same dispatch
//! args, same buffers, only the encoder/command-buffer split points differ. See that
//! method's doc comment for the full argument that this cannot alter GDN numerics.
//!
//! Scope guard: 0.8B path ONLY. Does not touch or measure the 27B path.
//!
//! Env:
//!   LATTICE_MODEL_DIR   model dir (default ~/.lattice/models/qwen3.5-0.8b)
//!   BENCH_LENGTHS       comma-separated token counts (default "1024,4096,16384")
//!   BENCH_WARMUP        warmup prefills per (length, path), discarded (default 2)
//!   BENCH_REPEATS       timed repeats per (length, path) (default 5)
//!
//! Output: one TSV row per (length, path) to stdout, plus a serial/chunked speedup
//! summary and any self-validation FLAGS to stderr and stdout.
//!
//! Calibration: in addition to the isolated (per-layer-split) sweep above, this
//! bin also runs a production-path calibration sweep per length —
//! `bench_support::forward_prefill_production_chunk`, the real unmodified
//! `forward_prefill_batched_chunk` dispatch (single command buffer, no per-layer
//! splits). This exists because the isolated method's own command-buffer splits
//! add CPU<->GPU sync overhead that biases its serial/chunked ratio, so it cannot
//! be checked against a tight historical prior directly. The production sweep's
//! serial_total/chunked_total ratio is the TIGHT anchor check instead (see
//! `FLAG[production_total_anchor]` below); the isolated sweep's own ratio check
//! is a wide sanity bound only (`FLAG[reality_anchor]`), not a calibration.

#[cfg(not(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
)))]
fn main() {
    eprintln!("Requires macOS + metal-gpu + bench-internals features.");
    std::process::exit(1);
}

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
fn main() {
    if let Err(e) = run() {
        eprintln!("bench_gdn_prefill_ab failed: {e}");
        std::process::exit(1);
    }
}

/// Canonical model identity used to key `prod_anchor_band`, derived the same
/// way the scope-guard check in `run()` already recognizes "the 0.8B
/// checkpoint": a substring match on `LATTICE_MODEL_DIR`. This harness has
/// only ever loaded, and its production-anchor bands have only ever been
/// calibrated against, that one checkpoint (see the module doc's "Scope
/// guard: 0.8B path ONLY" and `LATTICE_MODEL_DIR`'s own default above).
/// `None` covers every model directory that substring match does not
/// recognize, including a differently-sized or differently-quantized
/// checkpoint reached by overriding `LATTICE_MODEL_DIR`.
#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
fn model_identity(model_dir_str: &str) -> Option<&'static str> {
    if model_dir_str.contains("0.8b") || model_dir_str.contains("0_8b") {
        Some("qwen3.5-0.8b")
    } else {
        None
    }
}

/// Production-total anchor bands, versioned to a measured baseline and keyed
/// on `(model, length)`.
///
/// Baseline: re-measured at d96c26fccce846db9174e1c1ad15ebb0f2bf6856 on the
/// qwen3.5-0.8b checkpoint, from four ABBA arms (base1/head1/head2/base2) on
/// an idle machine, unmodified production dispatch, warmup>=2 repeats>=5:
/// 1.227x @1024, 1.154x @4096. Bands are baseline -20%/+20%, floored at 1.0
/// (a ratio below 1.0 would mean chunking made the production path slower,
/// which is a different alarm than "this anchor is stale"). Full derivation
/// and the underlying per-arm data: issue #1654.
///
/// The previous baseline recorded here (2.415x @1024, 2.072x @4096, 1.476x
/// @16384) is removed rather than kept under a guessed identity: those
/// ratios run roughly double this checkpoint's measured behavior and no
/// model they were actually calibrated against could be established (see
/// #1654) — carrying them forward keyed to a name would repeat the defect
/// this `(model, length)` key exists to close.
///
/// 16384 has no re-measured baseline under this method and is left with no
/// calibrated band for any model until one exists; it takes the `None` arm
/// below unconditionally.
///
/// Anchors go stale as unrelated optimizations land: when this flag fires on
/// an otherwise-clean idle run, re-measure the baseline at current HEAD and
/// update these constants (with the new SHA) rather than widening the band.
/// A `(model, length)` pair with no calibrated band returns `None`, and the
/// caller skips the tight check instead of comparing against a band
/// calibrated for a different checkpoint or a different length.
#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
fn prod_anchor_band(model: Option<&str>, length: usize) -> Option<(f64, f64)> {
    match (model, length) {
        (Some("qwen3.5-0.8b"), 1024) => Some((1.00, 1.47)),
        (Some("qwen3.5-0.8b"), 4096) => Some((1.00, 1.38)),
        _ => None,
    }
}

#[cfg(all(
    test,
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
mod prod_anchor_band_tests {
    use super::{model_identity, prod_anchor_band};

    #[test]
    fn calibrated_pair_returns_its_band() {
        assert_eq!(
            prod_anchor_band(Some("qwen3.5-0.8b"), 1024),
            Some((1.00, 1.47))
        );
        assert_eq!(
            prod_anchor_band(Some("qwen3.5-0.8b"), 4096),
            Some((1.00, 1.38))
        );
    }

    /// 16384 has no re-measured baseline (issue #1654): it must stay
    /// uncalibrated for the calibrated model too, not just for an unknown
    /// one, and it must never fall back to the old length-only constant.
    #[test]
    fn calibrated_model_at_16384_is_skipped() {
        assert_eq!(prod_anchor_band(Some("qwen3.5-0.8b"), 16384), None);
    }

    /// Length 1024 has a calibrated band, but only for qwen3.5-0.8b. An
    /// unidentified (or differently identified) model must not reuse it —
    /// this is the defect #1654 reports: a length-only key would return
    /// `Some((1.00, 1.47))` here regardless of which model produced the
    /// ratio.
    #[test]
    fn uncalibrated_model_at_a_calibrated_length_is_skipped() {
        assert_eq!(prod_anchor_band(None, 1024), None);
    }

    #[test]
    fn unknown_length_is_skipped_even_for_the_calibrated_model() {
        assert_eq!(prod_anchor_band(Some("qwen3.5-0.8b"), 2048), None);
    }

    #[test]
    fn model_identity_recognizes_only_the_calibrated_checkpoint() {
        assert_eq!(model_identity("models/qwen3.5-0.8b"), Some("qwen3.5-0.8b"));
        assert_eq!(model_identity("models/qwen3.5-0_8b"), Some("qwen3.5-0.8b"));
        assert_eq!(model_identity("models/qwen3.5-27b"), None);
    }
}

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::forward::metal_qwen35::bench_support::{self, GdnIsolatedChunkTiming};
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::tokenizer::{BpeTokenizer, Tokenizer};

    // --- INVALIDATION GUARD: debug builds produce meaningless Metal timing. This is
    // a compile-time check (not a runtime assert!) precisely because
    // `debug_assertions` is a compile-time constant — clippy correctly flags
    // `assert!(!cfg!(debug_assertions))` as a pointless constant assertion, and
    // failing to even build in debug mode is a strictly better guard anyway. ---
    #[cfg(debug_assertions)]
    compile_error!(
        "bench_gdn_prefill_ab: MUST build --release (debug Metal timing is meaningless — see \
         the bench methodology notes in the PR for 'What INVALIDATES a run')"
    );

    // --- INVALIDATION GUARD: hold the fleet-wide Metal GPU flock BEFORE any Metal
    // work (model load below doesn't touch Metal, but MetalQwen35State::new does). ---
    eprintln!("[bench] acquiring shared Metal GPU lock ...");
    let flock_acquired_at = std::time::Instant::now();
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();
    eprintln!(
        "[bench] flock held ({:.1}s wait)",
        flock_acquired_at.elapsed().as_secs_f64()
    );
    let flock_held = true; // unreachable past acquire_for_process() otherwise (it panics)

    let home = std::env::var("HOME")?;
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir_str);

    let lengths: Vec<usize> = std::env::var("BENCH_LENGTHS")
        .unwrap_or_else(|_| "1024,4096,16384".to_string())
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .expect("BENCH_LENGTHS: comma-separated integers")
        })
        .collect();
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let repeats: usize = std::env::var("BENCH_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    assert!(
        warmup >= 2,
        "bench methodology requires >=2 warmup prefills"
    );
    assert!(repeats >= 5, "bench methodology requires >=5 timed repeats");
    // --interleaved: toggle the two arms (chunked/serial) per repeat rather than
    // running all repeats of one arm before the other, so thermal/clock drift over
    // the run lands symmetrically on both arms instead of biasing whichever arm
    // runs second. Off by default to keep today's acquisition order unchanged.
    let interleaved: bool = std::env::args().any(|a| a == "--interleaved");
    if interleaved {
        eprintln!("[bench] mode=interleaved (per-prefill arm toggling)");
    }

    // --- SCOPE GUARD: 0.8B path only (name check is advisory; the hard check is
    // the config-shape assert after Metal init below). Uses the same identity
    // check the production-anchor band lookup below keys its calibration on. ---
    if model_identity(&model_dir_str).is_none() {
        eprintln!(
            "[bench] WARNING: LATTICE_MODEL_DIR={model_dir_str} does not look like the 0.8B \
             checkpoint. This harness only measures configs supporting the chunked GDN \
             prefill path; the shape check below hard-fails otherwise."
        );
    }

    eprintln!("[bench] loading {model_dir_str}");
    let model = Qwen35Model::from_safetensors(dir).map_err(|e| format!("load model: {e}"))?;
    let cfg = model.config().clone();
    let max_len_needed = *lengths.iter().max().expect("BENCH_LENGTHS non-empty");
    // A little headroom above the longest sweep length.
    let max_cache_len = max_len_needed + 512;
    let mut state = MetalQwen35State::new(model.weights(), &cfg, max_cache_len)
        .map_err(|e| format!("Metal init: {e}"))?;
    let tokenizer_dir_str =
        std::env::var("LATTICE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir_str.clone());
    let tokenizer = BpeTokenizer::from_tokenizer_json(
        &std::path::Path::new(&tokenizer_dir_str).join("tokenizer.json"),
    )?;

    // HARD scope check: the chunked arm must actually be the chunked path. Without
    // this, an unsupported config would fall back to the serial recurrence while the
    // harness labels the arm "chunked" — the isolated forward also asserts this, but
    // failing here is earlier and names the cause.
    if !bench_support::gdn_chunked_prefill_supported(&state) {
        return Err(format!(
            "model at {model_dir_str} does not support the chunked GDN prefill path — \
             this harness would mislabel a serial measurement as 'chunked'; refusing to run"
        )
        .into());
    }

    let mp = bench_support::max_prefill(&state);
    eprintln!("[bench] max_prefill (session chunk cap) = {mp}");

    // --- Fixed prompt, deterministically repeated + truncated to EXACT token counts. ---
    // Real prose (not a numeric filler pattern) so tokenization exercises normal BPE
    // merge behavior, matching the pattern already used by this file's own
    // `long_real_text_tokens` test helper (metal_qwen35.rs) for the same reason.
    let paragraphs = [
        "During a late engineering review, the team walked through the inference trace \
         one layer at a time. They checked where state changed, which buffers were reused, \
         and how a long request should preserve every earlier token.",
        "The prompt continued with ordinary prose about debugging, benchmarks, release \
         notes, and careful handoffs. It used full sentences, punctuation, and varied \
         vocabulary so tokenization looked like real input rather than a repeated numeric \
         pattern.",
        "A second reviewer asked for evidence at the boundary between chunks. The answer \
         described rotary positions, cache rows, recurrent memory, and causal attention \
         in concrete terms before any optimization was accepted.",
        "Verification across chunk boundaries requires that each token's absolute \
         position index matches the RoPE table row, the KV cache write offset, and the \
         attention causal mask, all three using the same absolute coordinate, never a \
         chunk-local one.",
    ];
    // Tokenize per-paragraph and accumulate ids, NOT the whole accumulated text each
    // iteration: re-tokenizing the growing string is O(iterations x text_len) — at
    // max_len_needed=16384 that burned >38 min of CPU before the first measurement
    // and blew the run timeout (TBV finding 2026-07-07; the 33-min "silent" smoke run
    // was the same defect, not GPU contention). Per-paragraph token boundaries differ
    // slightly from whole-text tokenization at the join points, which is irrelevant
    // here: the prompt is a deterministic fixed input held identical across both
    // measured paths.
    let base_tokens: Vec<u32> = {
        let mut ids: Vec<u32> = Vec::with_capacity(max_len_needed + 1024);
        let mut i = 0usize;
        while ids.len() < max_len_needed && i <= max_len_needed {
            let text = if i == 0 {
                paragraphs[0].to_string()
            } else {
                format!("\n\n{}", paragraphs[i % paragraphs.len()])
            };
            i += 1;
            let input = tokenizer.tokenize(&text);
            ids.extend_from_slice(&input.input_ids[..input.real_length]);
            // i > max_len_needed guard: real prose alone did not reach the target
            // length quickly enough (pathological BPE ratio) — fall back to padding below.
        }
        ids
    };
    // Validate every token id against the model vocab BEFORE any forward call: the
    // bench-support entry points bypass the public path's token-id validation, and
    // the embedding copy is an unsafe read indexed by token id. A mismatched
    // LATTICE_TOKENIZER_DIR must fail loudly here, not corrupt the first warmup.
    // (`tokens_for`'s padding id 1 is trivially in-vocab.)
    let vocab = bench_support::vocab_size(&state);
    if let Some((i, id)) = bench_support::first_out_of_vocab(&base_tokens, vocab) {
        return Err(format!(
            "token id {id} at base_tokens[{i}] is >= vocab_size {vocab} — tokenizer/model \
             mismatch (check LATTICE_TOKENIZER_DIR vs LATTICE_MODEL_DIR)"
        )
        .into());
    }
    let tokens_for = |n: usize| -> Vec<u32> {
        if n <= base_tokens.len() {
            base_tokens[..n].to_vec()
        } else {
            // Pad with token id 1, matching the existing boundary-sweep test's filler
            // convention (metal_qwen35.rs
            // gdn_chunked_prefill_vs_serial_prefill_logit_parity).
            let mut v = base_tokens.clone();
            v.resize(n, 1u32);
            v
        }
    };

    // Run `n` tokens through the isolated-timing chunk loop once; returns accumulated
    // GDN/non-GDN timing across all `max_prefill`-sized chunks. Caller must have
    // already called `state.set_gdn_chunked(..)` and `state.reset_state()`.
    let run_once = |state: &mut MetalQwen35State,
                    tokens: &[u32]|
     -> Result<GdnIsolatedChunkTiming, lattice_inference::InferenceError> {
        let mut total = GdnIsolatedChunkTiming::default();
        let mut start = 0usize;
        for chunk in tokens.chunks(mp) {
            let t = bench_support::forward_prefill_gdn_isolated_chunk(state, chunk, start)?;
            total.accumulate(&t);
            start += chunk.len();
        }
        Ok(total)
    };

    #[derive(Clone, Copy)]
    struct Stats {
        median: f64,
        min: f64,
        max: f64,
        iqr_pct_of_median: f64,
        suspect: bool,
    }
    fn stats(mut xs: Vec<f64>) -> Stats {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = xs.len();
        let percentile = |p: f64| -> f64 {
            if n == 1 {
                return xs[0];
            }
            let idx = p * (n as f64 - 1.0);
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            if lo == hi {
                xs[lo]
            } else {
                xs[lo] + (xs[hi] - xs[lo]) * (idx - lo as f64)
            }
        };
        let median = percentile(0.5);
        let q1 = percentile(0.25);
        let q3 = percentile(0.75);
        let iqr = q3 - q1;
        let iqr_pct_of_median = if median > 0.0 {
            iqr / median * 100.0
        } else {
            0.0
        };
        Stats {
            median,
            min: xs[0],
            max: xs[n - 1],
            iqr_pct_of_median,
            suspect: iqr_pct_of_median > 15.0,
        }
    }

    struct RowResult {
        length: usize,
        path: &'static str,
        gdn: Stats,
        total: Stats,
        non_gdn: Stats,
    }
    let mut rows: Vec<RowResult> = Vec::new();

    // Run `n` tokens through the UNMODIFIED production dispatch path once (single
    // command buffer per max_prefill-sized chunk, no per-layer splits) and return
    // total CPU wall-clock time in ms. Caller must have already called
    // `state.set_gdn_chunked(..)` and `state.reset_state()`. Unlike `run_once`
    // above, no internal command-buffer-boundary timing is needed: production
    // dispatch already commits + waits synchronously per chunk, so a plain
    // `Instant` wrapped around the whole chunking loop is the real measurement.
    let run_once_production = |state: &mut MetalQwen35State,
                               tokens: &[u32]|
     -> Result<f64, lattice_inference::InferenceError> {
        let start = std::time::Instant::now();
        let mut pos = 0usize;
        for chunk in tokens.chunks(mp) {
            bench_support::forward_prefill_production_chunk(state, chunk, pos)?;
            pos += chunk.len();
        }
        Ok(start.elapsed().as_secs_f64() * 1000.0)
    };

    struct ProdRowResult {
        length: usize,
        path: &'static str,
        total: Stats,
    }
    let mut prod_rows: Vec<ProdRowResult> = Vec::new();

    // Prompt-identity guard: every arm and sweep claiming to measure the same input
    // must receive a byte-identical token stream — ASSERTED via hash, not assumed
    // (both sweeps call tokens_for independently; this pins them to each other, and
    // the printed hash makes cross-run prompt identity checkable from logs).
    fn fnv1a_tokens(tokens: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &t in tokens {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }
    let mut prompt_hashes: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();

    // --- Production-path calibration sweep (runs BEFORE the isolated sweep). ---
    for &length in &lengths {
        let tokens = tokens_for(length);
        assert_eq!(tokens.len(), length, "tokens_for produced wrong length");
        let h = fnv1a_tokens(&tokens);
        prompt_hashes.insert(length, h);
        eprintln!("[bench] len={length:6} prompt_fnv1a={h:016x}");

        let prod_arms: [(bool, &'static str); 2] = [(true, "chunked_prod"), (false, "serial_prod")];
        if interleaved {
            // Warm each arm once, in arm order, before the interleaved timed loop
            // below — warmup order does not matter, only the timed samples need to
            // toggle arms so thermal/clock drift lands symmetrically on both.
            for &(chunked, _) in &prod_arms {
                state.set_gdn_chunked(chunked);
                for _ in 0..warmup {
                    state.reset_state();
                    let _ = run_once_production(&mut state, &tokens)?;
                }
            }
            let mut total_samples: [Vec<f64>; 2] =
                [Vec::with_capacity(repeats), Vec::with_capacity(repeats)];
            for _ in 0..repeats {
                for (i, &(chunked, _)) in prod_arms.iter().enumerate() {
                    state.set_gdn_chunked(chunked);
                    state.reset_state();
                    total_samples[i].push(run_once_production(&mut state, &tokens)?);
                }
            }
            for (i, &(_, path_name)) in prod_arms.iter().enumerate() {
                let total = stats(std::mem::take(&mut total_samples[i]));
                eprintln!(
                    "[bench] len={length:6} path={path_name:12} total_ms median={:.2} \
                     [{:.2},{:.2}] IQR%={:.1}{}  (production calibration, unmodified dispatch)",
                    total.median,
                    total.min,
                    total.max,
                    total.iqr_pct_of_median,
                    if total.suspect { " SUSPECT" } else { "" },
                );
                prod_rows.push(ProdRowResult {
                    length,
                    path: path_name,
                    total,
                });
            }
        } else {
            for (chunked, path_name) in prod_arms {
                state.set_gdn_chunked(chunked);

                for _ in 0..warmup {
                    state.reset_state();
                    let _ = run_once_production(&mut state, &tokens)?;
                }

                let mut total_samples = Vec::with_capacity(repeats);
                for _ in 0..repeats {
                    state.reset_state();
                    total_samples.push(run_once_production(&mut state, &tokens)?);
                }
                let total = stats(total_samples);
                eprintln!(
                    "[bench] len={length:6} path={path_name:12} total_ms median={:.2} \
                     [{:.2},{:.2}] IQR%={:.1}{}  (production calibration, unmodified dispatch)",
                    total.median,
                    total.min,
                    total.max,
                    total.iqr_pct_of_median,
                    if total.suspect { " SUSPECT" } else { "" },
                );
                prod_rows.push(ProdRowResult {
                    length,
                    path: path_name,
                    total,
                });
            }
        }
    }

    for &length in &lengths {
        let tokens = tokens_for(length);
        assert_eq!(tokens.len(), length, "tokens_for produced wrong length");
        let h = fnv1a_tokens(&tokens);
        assert_eq!(
            Some(&h),
            prompt_hashes.get(&length),
            "prompt-identity violation at len={length}: isolated-sweep tokens hash \
             {h:016x} != production-sweep hash — arms are not measuring the same input"
        );

        let isolated_arms: [(bool, &'static str); 2] = [(true, "chunked"), (false, "serial")];
        if interleaved {
            // Warm each arm once, in arm order, before the interleaved timed loop
            // below — warmup order does not matter, only the timed samples need to
            // toggle arms so thermal/clock drift lands symmetrically on both.
            for &(chunked, _) in &isolated_arms {
                state.set_gdn_chunked(chunked);
                for _ in 0..warmup {
                    state.reset_state();
                    let _ = run_once(&mut state, &tokens)?;
                }
            }
            let mut gdn_samples: [Vec<f64>; 2] =
                [Vec::with_capacity(repeats), Vec::with_capacity(repeats)];
            let mut total_samples: [Vec<f64>; 2] =
                [Vec::with_capacity(repeats), Vec::with_capacity(repeats)];
            let mut non_gdn_samples: [Vec<f64>; 2] =
                [Vec::with_capacity(repeats), Vec::with_capacity(repeats)];
            for _ in 0..repeats {
                for (i, &(chunked, _)) in isolated_arms.iter().enumerate() {
                    state.set_gdn_chunked(chunked);
                    state.reset_state();
                    let t = run_once(&mut state, &tokens)?;
                    gdn_samples[i].push(t.gdn_ms);
                    total_samples[i].push(t.total_ms());
                    non_gdn_samples[i].push(t.non_gdn_ms);
                }
            }
            for (i, &(_, path_name)) in isolated_arms.iter().enumerate() {
                let gdn = stats(std::mem::take(&mut gdn_samples[i]));
                let total = stats(std::mem::take(&mut total_samples[i]));
                let non_gdn = stats(std::mem::take(&mut non_gdn_samples[i]));
                eprintln!(
                    "[bench] len={length:6} path={path_name:8} gdn_ms median={:.2} \
                     [{:.2},{:.2}] IQR%={:.1}{}  total_ms median={:.2}  non_gdn_ms median={:.2}",
                    gdn.median,
                    gdn.min,
                    gdn.max,
                    gdn.iqr_pct_of_median,
                    if gdn.suspect { " SUSPECT" } else { "" },
                    total.median,
                    non_gdn.median,
                );
                rows.push(RowResult {
                    length,
                    path: path_name,
                    gdn,
                    total,
                    non_gdn,
                });
            }
        } else {
            for (chunked, path_name) in isolated_arms {
                state.set_gdn_chunked(chunked);

                for _ in 0..warmup {
                    state.reset_state();
                    let _ = run_once(&mut state, &tokens)?;
                }

                let mut gdn_samples = Vec::with_capacity(repeats);
                let mut total_samples = Vec::with_capacity(repeats);
                let mut non_gdn_samples = Vec::with_capacity(repeats);
                for _ in 0..repeats {
                    state.reset_state();
                    let t = run_once(&mut state, &tokens)?;
                    gdn_samples.push(t.gdn_ms);
                    total_samples.push(t.total_ms());
                    non_gdn_samples.push(t.non_gdn_ms);
                }

                let gdn = stats(gdn_samples);
                let total = stats(total_samples);
                let non_gdn = stats(non_gdn_samples);
                eprintln!(
                    "[bench] len={length:6} path={path_name:8} gdn_ms median={:.2} \
                     [{:.2},{:.2}] IQR%={:.1}{}  total_ms median={:.2}  non_gdn_ms median={:.2}",
                    gdn.median,
                    gdn.min,
                    gdn.max,
                    gdn.iqr_pct_of_median,
                    if gdn.suspect { " SUSPECT" } else { "" },
                    total.median,
                    non_gdn.median,
                );
                rows.push(RowResult {
                    length,
                    path: path_name,
                    gdn,
                    total,
                    non_gdn,
                });
            }
        }
    }

    // --- Self-validation + TSV emission, per length. ---
    println!(
        "length\tpath\tgdn_ms_median\tgdn_ms_min\tgdn_ms_max\ttotal_ms_median\t\
         non_gdn_ms_median\tn_repeats\tflock_held\tvalidation_flags"
    );
    if interleaved {
        println!("# mode=interleaved");
    }

    let mut any_flags = false;
    for &length in &lengths {
        let serial = rows
            .iter()
            .find(|r| r.length == length && r.path == "serial")
            .expect("serial row present");
        let chunked = rows
            .iter()
            .find(|r| r.length == length && r.path == "chunked")
            .expect("chunked row present");
        let prod_serial = prod_rows
            .iter()
            .find(|r| r.length == length && r.path == "serial_prod")
            .expect("serial_prod row present");
        let prod_chunked = prod_rows
            .iter()
            .find(|r| r.length == length && r.path == "chunked_prod")
            .expect("chunked_prod row present");

        // Production-calibration TSV rows: same column shape as the isolated rows
        // so the file stays parseable, gdn_ms_*/non_gdn_ms_median columns are N/A
        // for these paths (0.000, not "-" — those columns are numeric everywhere
        // else in this file) since production dispatch has no isolated GDN segment.
        for r in [prod_serial, prod_chunked] {
            let mut flags: Vec<&str> = Vec::new();
            if r.total.suspect {
                flags.push("SUSPECT_TOTAL_IQR");
            }
            let flags_str = if flags.is_empty() {
                "-".to_string()
            } else {
                any_flags = true;
                flags.join(",")
            };
            println!(
                "{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}",
                r.length,
                r.path,
                0.000,
                0.000,
                0.000,
                r.total.median,
                0.000,
                repeats,
                flock_held,
                flags_str,
            );
        }

        for r in [serial, chunked] {
            let mut flags: Vec<&str> = Vec::new();
            if r.gdn.suspect {
                flags.push("SUSPECT_GDN_IQR");
            }
            if r.total.suspect {
                flags.push("SUSPECT_TOTAL_IQR");
            }
            let flags_str = if flags.is_empty() {
                "-".to_string()
            } else {
                any_flags = true;
                flags.join(",")
            };
            println!(
                "{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}",
                r.length,
                r.path,
                r.gdn.median,
                r.gdn.min,
                r.gdn.max,
                r.total.median,
                r.non_gdn.median,
                repeats,
                flock_held,
                flags_str,
            );
        }

        // Self-validation check 1: non-GDN consistency.
        let non_gdn_diff_pct = if serial.non_gdn.median > 0.0 {
            (serial.non_gdn.median - chunked.non_gdn.median).abs() / serial.non_gdn.median * 100.0
        } else {
            0.0
        };
        let non_gdn_flag = non_gdn_diff_pct > 15.0;

        // Self-validation check 2: cross-check (serial_total - chunked_total) ≈
        // (serial_GDN - chunked_GDN). A near-zero GDN delta does NOT make the check
        // pass: if the isolated measurement breaks in a way that zeroes both arms'
        // GDN time while the totals still differ, that is exactly a broken
        // instrument, so the undefined-ratio case flags unless the total delta is
        // also negligible (1ms absolute). Non-finite inputs always flag.
        let total_diff = serial.total.median - chunked.total.median;
        let gdn_diff = serial.gdn.median - chunked.gdn.median;
        let cross_check_undefined = gdn_diff.abs() <= 1e-9;
        let cross_check_diff_pct = if cross_check_undefined {
            0.0
        } else {
            (total_diff - gdn_diff).abs() / gdn_diff.abs() * 100.0
        };
        let cross_check_flag = if !total_diff.is_finite() || !gdn_diff.is_finite() {
            true
        } else if cross_check_undefined {
            total_diff.abs() > 1.0
        } else {
            cross_check_diff_pct > 20.0
        };

        // Self-validation check 3: isolated-ratio sanity bound. This is NOT a
        // calibration against any historical prior — no isolated-recurrence-only
        // prior exists by construction (the isolated method's own command-buffer
        // splits add CPU<->GPU sync overhead unique to this harness, so nothing
        // measured outside it is comparable). The tight calibration against a
        // prior interleaved A/B measurement (2026-06) lives in the production-total
        // check below instead. This check only rejects results that cannot be a
        // real recurrence speedup
        // under any honest accounting: a reversal (chunked slower than serial,
        // ratio < 1.0) or an implausibly large ratio (>20x) for the same kernels
        // running the same recurrence work.
        let ratio = if chunked.gdn.median > 0.0 {
            serial.gdn.median / chunked.gdn.median
        } else {
            f64::NAN
        };
        let anchor_flag = !(1.0..=20.0).contains(&ratio) || ratio.is_nan();

        println!(
            "# len={length}: serial_GDN/chunked_GDN speedup = {:.3}x  \
             [serial {:.2},{:.2},{:.2}] / [chunked {:.2},{:.2},{:.2}] (median,min,max ms)",
            ratio,
            serial.gdn.min,
            serial.gdn.median,
            serial.gdn.max,
            chunked.gdn.min,
            chunked.gdn.median,
            chunked.gdn.max,
        );

        // Self-validation check 4: production-total anchor. Unlike the isolated
        // sweep, the production sweep uses the UNMODIFIED dispatch path (single
        // command buffer per chunk), so it is the correct place to check against a
        // measured baseline. The baseline is versioned per-length via
        // `prod_anchor_band` rather than a single fixed band, because a single
        // band drifts stale as unrelated optimizations land elsewhere in the
        // dispatch path (a June-4-era 1.35-1.75x band false-fired after a month of
        // non-GDN changes moved the true ratio outside it). Lengths without a
        // calibrated band skip the tight check and print an informational line
        // instead of firing FLAG[production_total_anchor].
        let prod_ratio = if prod_chunked.total.median > 0.0 {
            prod_serial.total.median / prod_chunked.total.median
        } else {
            f64::NAN
        };
        let model = model_identity(&model_dir_str);
        let band = prod_anchor_band(model, length);
        let prod_anchor_flag = match band {
            Some((lo, hi)) => !(lo..=hi).contains(&prod_ratio) || prod_ratio.is_nan(),
            None => false,
        };
        if band.is_none() {
            println!(
                "# len={length}: no production-anchor band calibrated for model={} at this \
                 length (bands: qwen3.5-0.8b @ 1024/4096); tight anchor skipped",
                model.unwrap_or("unrecognized"),
            );
        }

        println!(
            "# len={length}: serial_prod_total/chunked_prod_total speedup = {:.3}x  \
             [serial_prod {:.2},{:.2},{:.2}] / [chunked_prod {:.2},{:.2},{:.2}] \
             (median,min,max ms; unmodified production dispatch)",
            prod_ratio,
            prod_serial.total.min,
            prod_serial.total.median,
            prod_serial.total.max,
            prod_chunked.total.min,
            prod_chunked.total.median,
            prod_chunked.total.max,
        );

        let mut length_flags: Vec<String> = Vec::new();
        if non_gdn_flag {
            length_flags.push(format!(
                "FLAG[non_gdn_consistency]: non-GDN work differs {non_gdn_diff_pct:.1}% between \
                 serial/chunked at len={length} (serial={:.2}ms chunked={:.2}ms) — GDN isolation \
                 may be leaking non-recurrence work into the timed segment.",
                serial.non_gdn.median, chunked.non_gdn.median
            ));
        }
        if cross_check_flag {
            if cross_check_undefined {
                length_flags.push(format!(
                    "FLAG[cross_check]: undefined cross-check ratio at len={length} — \
                     (serial_GDN-chunked_GDN)={gdn_diff:.4}ms is ~zero (or non-finite) while \
                     (serial_total-chunked_total)={total_diff:.2}ms is not: the isolated GDN \
                     measurement is not accounting for the arms' total-time difference."
                ));
            } else {
                length_flags.push(format!(
                    "FLAG[cross_check]: (serial_total-chunked_total)={total_diff:.2}ms vs \
                     (serial_GDN-chunked_GDN)={gdn_diff:.2}ms differ by \
                     {cross_check_diff_pct:.1}% at len={length}."
                ));
            }
        }
        if anchor_flag {
            length_flags.push(format!(
                "FLAG[reality_anchor]: serial_GDN/chunked_GDN={ratio:.3}x at len={length} is \
                 outside the wide isolated-sweep sanity bound 1.0-20.0x (no reversal, no \
                 implausible blowup) — the harness's isolation is suspect, do NOT trust this \
                 length's numbers. This is a sanity bound only, not a calibration; see \
                 FLAG[production_total_anchor] for the tight anchor check."
            ));
        }
        if prod_anchor_flag {
            let (lo, hi) = band.expect("prod_anchor_flag only set when band is Some");
            // `band` is only `Some` when `model_identity` matched, so this unwrap is safe.
            let model_name = model.unwrap_or("?");
            length_flags.push(format!(
                "FLAG[production_total_anchor]: serial_prod_total/chunked_prod_total=\
                 {prod_ratio:.3}x at len={length} is outside the expected {lo:.2}-{hi:.2}x band \
                 (baseline @ d96c26fccce846db9174e1c1ad15ebb0f2bf6856 on {model_name}, four ABBA \
                 arms, ±20%/floor-1.0 rule — see issue #1654) — measured under the unmodified \
                 production dispatch path (single command buffer per chunk, same regime the \
                 baseline was measured under). Do NOT trust this length's speedup claim."
            ));
        }
        if length_flags.is_empty() {
            println!("# len={length}: self-validation flags: none (clean)");
        } else {
            any_flags = true;
            for f in &length_flags {
                println!("# len={length}: {f}");
                eprintln!("{f}");
            }
        }
    }

    if any_flags {
        eprintln!(
            "[bench] *** ONE OR MORE SELF-VALIDATION FLAGS FIRED — see FLAG lines above. \
             This run's numbers are NOT trustworthy as-is. ***"
        );
    } else {
        eprintln!("[bench] self-validation: all flags clean on this run.");
    }
    eprintln!(
        "[bench] REMINDER: all numbers from this harness are PROVISIONAL until a \
         maintainer re-verifies the harness fresh on an idle machine (see the bench \
         methodology notes in the PR)."
    );

    Ok(())
}
