use super::*;

fn base_config() -> FullDriverConfig {
    FullDriverConfig {
        model_dir: PathBuf::from("/nonexistent/model-dir"),
        data_dir: PathBuf::from("/nonexistent/data-dir"),
        first_layer: 19,
        steps: 10,
        lr: 1e-3,
        rank: 4,
        alpha: 8.0,
        seq_len_cap: 64,
        max_train: 3,
        max_valid: 0,
        seed: 0xFEED_FACEu64,
        log_every: 5,
        gradcheck: false,
        gradcheck_strided_probes: false,
        probe: 0,
        fd_eps: 1e-3,
        save_path: None,
        load_path: None,
        a_init_amp: None,
        gdn_modules: GdnModuleSelection::default(),
    }
}

#[test]
fn training_seed_controls_initial_a_q_bytes() {
    let dims = Dims {
        hidden: 4,
        q_dim: 2,
        kv_dim: 2,
        ..tape_test_dims()
    };
    // Goes through the same function `run` calls, so re-hardcoding the seed on
    // that path fails here. Calling the per-slot helper instead would not: that
    // version passed with `run`'s wiring reverted.
    let init = |config: &FullDriverConfig| {
        let (loras, next_rng) = initial_training_loras(config.seed, 2, &dims, config.rank, 0.5);
        let bytes: Vec<u8> = loras
            .into_iter()
            .flat_map(|l| l.a_q)
            .flat_map(f32::to_ne_bytes)
            .collect();
        (bytes, next_rng)
    };
    let config = base_config();
    let mut other = base_config();
    other.seed = 42;
    assert_eq!(
        init(&config),
        init(&base_config()),
        "same seed must reproduce bytes"
    );
    assert_ne!(
        init(&config).0,
        init(&other).0,
        "different seeds must differ"
    );
    // The GDN slots continue this stream, so the advanced state must move with
    // the seed as well; otherwise only half the adapter would be reproducible.
    assert_ne!(
        init(&config).1,
        init(&other).1,
        "advanced RNG state must follow the seed"
    );
}

/// `FullDriverOutcome` has no `Debug` impl, so `Result::unwrap_err` isn't
/// available — extract the error message by hand.
fn expect_err(result: Result<FullDriverOutcome, Box<dyn std::error::Error>>) -> String {
    match result {
        Ok(_) => panic!("expected run() to reject this config"),
        Err(e) => e.to_string(),
    }
}

/// Without the rank cap, an oversized rank reaches
/// model loading and allocation instead of being rejected up front.
#[test]
fn run_rejects_rank_over_max() {
    let mut cfg = base_config();
    cfg.rank = MAX_RANK + 1;
    let err = expect_err(run(cfg));
    assert!(
        err.contains("--rank") && err.contains(&MAX_RANK.to_string()),
        "expected rank-over-cap rejection; got: {err}"
    );
}

/// Without the rank>0 guard, rank=0 reaches the tape
/// and divides by zero deriving `scale = alpha / rank`.
#[test]
fn run_rejects_zero_rank() {
    let mut cfg = base_config();
    cfg.rank = 0;
    let err = expect_err(run(cfg));
    assert!(
        err.contains("--rank"),
        "expected zero-rank rejection; got: {err}"
    );
}

/// Without the step cap, `steps = usize::MAX` would
/// monopolize a worker indefinitely instead of being rejected up front.
#[test]
fn run_rejects_steps_over_max() {
    let mut cfg = base_config();
    cfg.steps = MAX_STEPS + 1;
    let err = expect_err(run(cfg));
    assert!(
        err.contains("--steps") && err.contains(&MAX_STEPS.to_string()),
        "expected steps-over-cap rejection; got: {err}"
    );
}

/// Without `validate_loaded_depth`, a model shallower
/// than `TOP_LAYER + 1` reaches `first_layer..=TOP_LAYER` layer access and
/// panics indexing `weights.layers[layer_idx]` out of bounds instead of
/// returning an error.
#[test]
fn validate_loaded_depth_rejects_shallow_model() {
    let err = validate_loaded_depth(TOP_LAYER).unwrap_err();
    assert!(
        err.contains("loaded layers") && err.contains("out of range"),
        "expected shallow-model rejection; got: {err}"
    );
}

#[test]
fn validate_loaded_depth_accepts_sufficient_model() {
    validate_loaded_depth(TOP_LAYER + 1).unwrap();
    validate_loaded_depth(40).unwrap();
}

/// Without the `log_every == 0` guard, a caller
/// passing `--log-every 0` with `steps > 0` reaches `step % log_every`
/// in the training loop and panics on divide-by-zero.
#[test]
fn run_rejects_zero_log_every() {
    let mut cfg = base_config();
    cfg.log_every = 0;
    let err = expect_err(run(cfg));
    assert!(
        err.contains("--log-every"),
        "expected log-every rejection; got: {err}"
    );
}

/// Without the `--max-train` cap, an oversized value
/// reaches JSONL tokenization and per-sample activation-cache work
/// instead of being rejected up front.
#[test]
fn run_rejects_max_train_over_max() {
    let mut cfg = base_config();
    cfg.max_train = MAX_SAMPLES + 1;
    let err = expect_err(run(cfg));
    assert!(
        err.contains("--max-train") && err.contains(&MAX_SAMPLES.to_string()),
        "expected max-train-over-cap rejection; got: {err}"
    );
}

/// Without the `--max-valid` cap, an oversized value
/// reaches JSONL tokenization and per-sample activation-cache work
/// instead of being rejected up front.
#[test]
fn run_rejects_max_valid_over_max() {
    let mut cfg = base_config();
    cfg.max_valid = MAX_SAMPLES + 1;
    let err = expect_err(run(cfg));
    assert!(
        err.contains("--max-valid") && err.contains(&MAX_SAMPLES.to_string()),
        "expected max-valid-over-cap rejection; got: {err}"
    );
}

/// Without the `--seq-len-cap` cap, an oversized
/// value reaches tokenization and per-sample activation-cache allocation
/// (`seq_len * hidden` per sample) unbounded.
#[test]
fn run_rejects_seq_len_cap_over_max() {
    let mut cfg = base_config();
    cfg.seq_len_cap = MAX_SEQ_LEN_CAP + 1;
    let err = expect_err(run(cfg));
    assert!(
        err.contains("--seq-len-cap") && err.contains(&MAX_SEQ_LEN_CAP.to_string()),
        "expected seq-len-cap-over-cap rejection; got: {err}"
    );
}

#[test]
fn run_rejects_zero_seq_len_cap() {
    let mut cfg = base_config();
    cfg.seq_len_cap = 0;
    let err = expect_err(run(cfg));
    assert!(
        err.contains("--seq-len-cap"),
        "expected zero-seq-len-cap rejection; got: {err}"
    );
}

/// `MAX_SAMPLES` and `MAX_SEQ_LEN_CAP` bound sample
/// count and per-sample length independently; neither bounds their
/// product. This sample set passes both per-dimension caps individually
/// (far under `MAX_SAMPLES` samples, far under `MAX_SEQ_LEN_CAP` tokens
/// each) but its aggregate `build_caches` allocation exceeds
/// `MAX_CACHE_BYTES`. Without `check_cache_budget`, nothing in `run()`
/// would have rejected this combination before `build_caches` started
/// allocating.
#[test]
fn check_cache_budget_rejects_oversized_aggregate() {
    let hidden = 4096;
    let rope_dim = 128;
    let samples: Vec<Sample> = (0..2_000)
        .map(|_| Sample {
            tokens: vec![0u32; 100],
            completion_start: 0,
        })
        .collect();
    assert!(
        samples.len() < MAX_SAMPLES,
        "test setup must pass the sample-count cap"
    );
    assert!(
        samples.iter().all(|s| s.tokens.len() < MAX_SEQ_LEN_CAP),
        "test setup must pass the per-sample length cap"
    );
    let total_tokens: u64 = samples.iter().map(|s| s.tokens.len() as u64).sum();
    assert!(
        total_tokens * ((hidden + rope_dim) as u64) * 4 > MAX_CACHE_BYTES,
        "test setup must actually exceed the aggregate cache budget"
    );

    let err = check_cache_budget(&samples, hidden, rope_dim, "train").unwrap_err();
    assert!(
        err.contains("frozen-prefix cache") && err.contains("--max-train"),
        "expected aggregate cache-budget rejection; got: {err}"
    );
}

#[test]
fn check_cache_budget_accepts_small_aggregate() {
    let samples: Vec<Sample> = (0..4)
        .map(|_| Sample {
            tokens: vec![0u32; 32],
            completion_start: 0,
        })
        .collect();
    check_cache_budget(&samples, 4096, 128, "train").unwrap();
}

#[test]
fn check_live_logits_budget_is_per_sample_not_aggregate() {
    let vocab = 2_097_152;
    let completion_positions = 192;
    let samples: Vec<Sample> = (0..2)
        .map(|_| Sample {
            tokens: vec![0u32; completion_positions + 1],
            completion_start: 1,
        })
        .collect();
    let aggregate_bytes = samples.len() as u128
        * completion_positions as u128
        * vocab as u128
        * std::mem::size_of::<f32>() as u128;
    assert!(aggregate_bytes > MAX_LOGITS_BYTES as u128);

    check_live_logits_budget(&samples, vocab, BACKWARD_LOGITS_WORKSPACE_ROWS, "training")
        .expect("only one sample's logits are live at a time");
}

#[test]
fn check_training_logits_budget_respects_execution_mode() {
    let vocab = 248_320;
    let completion_positions = 2_162;
    let samples = vec![
        Sample {
            tokens: vec![0u32; 2],
            completion_start: 1,
        },
        Sample {
            tokens: vec![0u32; completion_positions + 1],
            completion_start: 1,
        },
    ];
    let retained_bytes = completion_positions * vocab * std::mem::size_of::<f32>();
    let backward_bytes = (completion_positions + 1) * vocab * std::mem::size_of::<f32>();
    assert!(retained_bytes <= MAX_LOGITS_BYTES);
    assert!(backward_bytes > MAX_LOGITS_BYTES);

    check_training_logits_budget(&samples, vocab, 1, true)
        .expect("gradcheck only forwards the first training sample");
    check_training_logits_budget(&samples, vocab, 1, false)
        .expect("one step only runs backward on the safe first sample");
}

#[test]
fn check_training_logits_budget_rejects_sample_reached_by_update() {
    let vocab = 248_320;
    let samples = vec![
        Sample {
            tokens: vec![0u32; 2],
            completion_start: 1,
        },
        Sample {
            tokens: vec![0u32; 2_163],
            completion_start: 1,
        },
    ];

    let err = check_training_logits_budget(&samples, vocab, 2, false).unwrap_err();
    assert!(
        err.contains("training sample 2") && err.contains("2162 supervised positions"),
        "expected two steps to reject the second backward sample; got: {err}"
    );
}

#[test]
fn check_training_logits_budget_with_zero_steps_charges_evaluation_only() {
    let vocab = 248_320;
    let completion_positions = 2_162;
    let sample = Sample {
        tokens: vec![0u32; completion_positions + 1],
        completion_start: 1,
    };
    let retained_bytes = completion_positions * vocab * std::mem::size_of::<f32>();
    let backward_bytes = (completion_positions + 1) * vocab * std::mem::size_of::<f32>();
    assert!(retained_bytes <= MAX_LOGITS_BYTES);
    assert!(backward_bytes > MAX_LOGITS_BYTES);

    check_training_logits_budget(&[sample], vocab, 0, false)
        .expect("zero steps evaluate the sample without running backward");
}

#[test]
fn check_valid_logits_budget_skips_held_out_in_gradcheck() {
    let vocab = 248_320;
    let train_sample = Sample {
        tokens: vec![0u32; 2],
        completion_start: 1,
    };
    let valid_sample = Sample {
        tokens: vec![0u32; 2_164],
        completion_start: 1,
    };
    let valid_bytes = 2_163 * vocab * std::mem::size_of::<f32>();
    assert!(valid_bytes > MAX_LOGITS_BYTES);

    check_training_logits_budget(&[train_sample], vocab, 1, true)
        .expect("gradcheck's first training sample is within the live cap");
    check_valid_logits_budget(std::slice::from_ref(&valid_sample), vocab, true)
        .expect("gradcheck never forwards held-out samples");
}

#[test]
fn check_valid_logits_budget_charges_held_out_during_training() {
    let vocab = 248_320;
    let valid_sample = Sample {
        tokens: vec![0u32; 2_164],
        completion_start: 1,
    };

    let err = check_valid_logits_budget(&[valid_sample], vocab, false).unwrap_err();
    assert!(
        err.contains("held-out sample 1") && err.contains("2163 supervised positions"),
        "expected normal held-out evaluation to reject the oversized sample; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_rejects_one_oversized_completion() {
    let vocab = 2_097_152;
    let samples = vec![
        Sample {
            tokens: vec![0u32; 2],
            completion_start: 1,
        },
        Sample {
            tokens: vec![0u32; 258],
            completion_start: 1,
        },
        Sample {
            tokens: vec![0u32; 2],
            completion_start: 1,
        },
    ];

    let err = check_live_logits_budget(&samples, vocab, BACKWARD_LOGITS_WORKSPACE_ROWS, "training")
        .unwrap_err();
    assert!(
        err.contains("sample 2")
            && err.contains("257 supervised positions")
            && err.contains("exceeds"),
        "expected per-sample live-logits rejection; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_accepts_training_exact_live_cap() {
    let vocab = 2_097_152;
    let sample = Sample {
        tokens: vec![0u32; 1_024],
        completion_start: 769,
    };
    let raw_token_projection =
        sample.tokens.len() as u128 * vocab as u128 * std::mem::size_of::<f32>() as u128;
    assert!(raw_token_projection > MAX_LOGITS_BYTES as u128);

    check_live_logits_budget(&[sample], vocab, BACKWARD_LOGITS_WORKSPACE_ROWS, "training")
        .expect("255 retained rows plus one backward workspace row must reach the exact cap");
}

#[test]
fn check_live_logits_budget_rejects_training_one_row_over_live_cap() {
    let sample = Sample {
        tokens: vec![0u32; 257],
        completion_start: 1,
    };

    let err = check_live_logits_budget(
        &[sample],
        2_097_152,
        BACKWARD_LOGITS_WORKSPACE_ROWS,
        "training",
    )
    .unwrap_err();
    assert!(
        err.contains("256 supervised positions") && err.contains("exceeds"),
        "expected training workspace row to exceed the live cap; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_accepts_held_out_exact_retained_cap() {
    let sample = Sample {
        tokens: vec![0u32; 257],
        completion_start: 1,
    };

    check_live_logits_budget(&[sample], 2_097_152, EVAL_LOGITS_WORKSPACE_ROWS, "held-out")
        .expect("256 retained rows and no backward workspace must reach the exact cap");
}

#[test]
fn check_live_logits_budget_rejects_held_out_one_row_over_retained_cap() {
    let sample = Sample {
        tokens: vec![0u32; 258],
        completion_start: 1,
    };

    let err =
        check_live_logits_budget(&[sample], 2_097_152, EVAL_LOGITS_WORKSPACE_ROWS, "held-out")
            .unwrap_err();
    assert!(
        err.contains("257 supervised positions") && err.contains("exceeds"),
        "expected retained logits to exceed the held-out cap; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_rejects_size_overflow() {
    let sample = Sample {
        tokens: vec![0u32; 3],
        completion_start: 1,
    };
    let err = check_live_logits_budget(
        &[sample],
        usize::MAX,
        EVAL_LOGITS_WORKSPACE_ROWS,
        "held-out",
    )
    .unwrap_err();
    assert!(
        err.contains("held-out") && err.contains("overflows"),
        "expected checked-arithmetic rejection; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_rejects_byte_size_overflow() {
    let sample = Sample {
        tokens: vec![0u32; 2],
        completion_start: 1,
    };
    assert!(usize::MAX.checked_mul(1).is_some());
    assert!(usize::MAX.checked_mul(std::mem::size_of::<f32>()).is_none());

    let err = check_live_logits_budget(
        &[sample],
        usize::MAX,
        EVAL_LOGITS_WORKSPACE_ROWS,
        "held-out",
    )
    .unwrap_err();
    assert!(
        err.contains("held-out") && err.contains("overflows"),
        "expected checked byte-size rejection; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_rejects_workspace_row_overflow() {
    let sample = Sample {
        tokens: vec![0u32; 2],
        completion_start: 1,
    };
    let err = check_live_logits_budget(&[sample], 32, usize::MAX, "training").unwrap_err();
    assert!(
        err.contains("row count") && err.contains("overflows"),
        "expected checked workspace-row rejection; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_rejects_zero_completion_start() {
    let sample = Sample {
        tokens: vec![0u32; 2],
        completion_start: 0,
    };
    let err = check_live_logits_budget(&[sample], 32, BACKWARD_LOGITS_WORKSPACE_ROWS, "training")
        .unwrap_err();
    assert!(
        err.contains("completion_start") && err.contains("1 <= completion_start < 2"),
        "expected zero completion boundary rejection; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_rejects_completion_start_at_token_count() {
    let sample = Sample {
        tokens: vec![0u32; 2],
        completion_start: 2,
    };
    let err = check_live_logits_budget(&[sample], 32, EVAL_LOGITS_WORKSPACE_ROWS, "held-out")
        .unwrap_err();
    assert!(
        err.contains("completion_start") && err.contains("1 <= completion_start < 2"),
        "expected empty completion boundary rejection; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_rejects_completion_start_past_token_count() {
    let sample = Sample {
        tokens: vec![0u32; 1],
        completion_start: 2,
    };
    let err = check_live_logits_budget(&[sample], 32, BACKWARD_LOGITS_WORKSPACE_ROWS, "training")
        .unwrap_err();
    assert!(
        err.contains("completion_start"),
        "expected past-end completion boundary rejection; got: {err}"
    );
}

#[test]
fn check_live_logits_budget_rejects_empty_tokens() {
    let sample = Sample {
        tokens: Vec::new(),
        completion_start: 1,
    };
    let err = check_live_logits_budget(&[sample], 32, BACKWARD_LOGITS_WORKSPACE_ROWS, "training")
        .unwrap_err();
    assert!(
        err.contains("completion_start") && err.contains("1 <= completion_start < 0"),
        "expected empty-token completion boundary rejection; got: {err}"
    );
}

fn tape_test_dims() -> Dims {
    Dims {
        hidden: 2048,
        vocab: 32_000,
        num_q_heads: 8,
        num_kv_heads: 2,
        head_dim: 128,
        rope_dim: 128,
        inter: 4096,
        q_dim: 1024,
        kv_dim: 256,
        eps: 1e-6,
    }
}

/// Smaller than [`tape_test_dims`] so its own linear tape term stays well
/// under [`MAX_TAPE_BYTES`] at `MAX_SEQ_LEN_CAP` — isolating the
/// quadratic `softmax_probs` term as the sole reason the full budget
/// check rejects it.
fn quadratic_dominant_dims() -> Dims {
    Dims {
        hidden: 1024,
        vocab: 32_000,
        num_q_heads: 8,
        num_kv_heads: 1,
        head_dim: 64,
        rope_dim: 64,
        inter: 2048,
        q_dim: 512,
        kv_dim: 128,
        eps: 1e-6,
    }
}

fn tape_test_gdn_dims() -> GdnDims {
    GdnDims {
        num_kh: 2,
        value_heads: 8,
        key_dim: 64,
        value_dim: 64,
        qkv_dim: 2 * 64 * 2 + 8 * 64,
        output_dim: 8 * 64,
        kernel_size: 4,
        scale: 1.0 / (64f32).sqrt(),
    }
}

/// A single sample at `seq_len_cap` clears [`check_cache_budget`] (linear
/// in seq-len — only `h_in`/`cos`/`sin`), yet its `forward_full` tape
/// across the same materialized GQA layers is quadratic in seq-len via
/// `softmax_probs`. Without `check_tape_budget`, this combination would
/// pass every existing guard and exhaust memory once training starts.
#[test]
fn check_tape_budget_rejects_quadratic_overflow_that_cache_budget_misses() {
    let dims = quadratic_dominant_dims();
    let gdn_dims = tape_test_gdn_dims();
    let seq_len_cap = MAX_SEQ_LEN_CAP; // 8_192
    let rank = 8;
    let num_gqa_layers = 2;

    let single_sample = vec![Sample {
        tokens: vec![0u32; seq_len_cap],
        completion_start: 0,
    }];
    check_cache_budget(&single_sample, dims.hidden, dims.rope_dim, "train")
        .expect("test setup: a single max-length sample must clear the linear cache budget");

    let seq = seq_len_cap as u64;
    let linear_only_bytes =
        num_gqa_layers as u64 * gqa_layer_linear_elems_per_token(&dims, rank) * seq * 4;
    assert!(
        linear_only_bytes <= MAX_TAPE_BYTES,
        "test setup must keep the linear tape term alone under budget, so the rejection \
             below is attributable to the quadratic softmax_probs term specifically; got \
             {linear_only_bytes} bytes"
    );

    let err = check_tape_budget(seq_len_cap, &dims, &gdn_dims, rank, num_gqa_layers, 0)
        .expect_err("expected quadratic tape-budget rejection");
    assert!(
        err.contains("--seq-len-cap") && err.contains("softmax_probs"),
        "expected tape-budget rejection naming the quadratic term; got: {err}"
    );
}

#[test]
fn check_tape_budget_accepts_modest_config() {
    let dims = tape_test_dims();
    let gdn_dims = tape_test_gdn_dims();
    check_tape_budget(512, &dims, &gdn_dims, 16, 4, 4)
        .expect("a modest seq-len/layer-count combination must stay under the tape budget");
}
