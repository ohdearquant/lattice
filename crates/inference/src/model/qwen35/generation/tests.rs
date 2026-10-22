use super::*;
// row R03 deleted this module's own free `has_finite_logit` (folded into
// `QwenCpuSession::select`); these two tests still probe the guard directly, so
// they now import the crate-private copy the driver's session actually calls.
use crate::decoder::qwen_cpu::has_finite_logit;

// -----------------------------------------------------------------------
// ADR-090 row C: driver trace vs the pre-migration golden
// -----------------------------------------------------------------------

/// Mirrors `tests/cpu_pre_migration_greedy_golden.rs` exactly (same fixture, same
/// checkpoint, same replayed config), but through the crate-private
/// `generate_with_trace` that external integration test cannot see -- and asserts
/// the one thing that test cannot: that those ids came through the driver, in the
/// same test that asserts them, and `DriverTrace` does not exist outside this
/// crate. This does not modify, replace, or duplicate that
/// gate's own enforcement; it stays untouched and still runs on its own. This is
/// an additional, narrower check the golden's own crate boundary cannot perform.
#[test]
#[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_CPU_GREEDY_MODEL_DIR"]
fn driver_trace_matches_ids_len_on_the_pre_migration_golden() {
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        prompt: String,
        expected_generated_ids: Vec<u32>,
        #[serde(default)]
        reasoning_budget: Option<usize>,
    }
    #[derive(serde::Deserialize)]
    struct GoldenGeneration {
        temperature: f32,
        repetition_penalty: f32,
        seed: Option<u64>,
        enable_thinking: bool,
    }
    #[derive(serde::Deserialize)]
    struct Golden {
        max_new_tokens: usize,
        generation: GoldenGeneration,
        cases: Vec<Case>,
    }

    const FIXTURE: &str = include_str!(
        "../../../../tests/fixtures/cpu_pre_migration_greedy_v1/qwen35_0_8b_cpu_greedy_tokens.json"
    );
    let golden: Golden = serde_json::from_str(FIXTURE).expect("golden fixture parses");

    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_CPU_GREEDY_MODEL_DIR");
    let model = Qwen35Model::from_safetensors(&model_dir)
        .unwrap_or_else(|e| panic!("loading {model_dir:?} failed: {e}"));

    for case in &golden.cases {
        let cfg = GenerateConfig {
            max_new_tokens: golden.max_new_tokens,
            temperature: golden.generation.temperature,
            repetition_penalty: golden.generation.repetition_penalty,
            seed: golden.generation.seed,
            enable_thinking: golden.generation.enable_thinking,
            reasoning_budget: case.reasoning_budget,
            ..Default::default()
        };

        let (output, trace) = model
            .generate_with_trace(&case.prompt, &cfg)
            .unwrap_or_else(|e| panic!("case {}: generation failed: {e}", case.name));

        assert_eq!(
            output.token_ids, case.expected_generated_ids,
            "case {}: driver-routed ids diverged from the pre-migration golden",
            case.name
        );
        assert_eq!(
            trace.opened,
            output.token_ids.len(),
            "case {}: one select() per emitted token on a natural \
                 (non-EOS-at-step-0) finish",
            case.name
        );
        // The row's acceptance originally asked for `consumed == ids.len()`.
        // It is the wrong equality and the driver's doc comment says why:
        // squaring it would mean issuing one extra forward pass per request
        // whose logits nothing reads. `consumed == opened - 1` is the
        // invariant the loop actually maintains, and it detects a bypassed
        // step exactly as well, because a bypass shortens `opened` too.
        assert_eq!(
            trace.consumed + 1,
            trace.opened,
            "case {}: exactly one prediction stays open at finish",
            case.name
        );

        // The SAME case replayed with `logprobs` set. Row R03 deleted the
        // `generate_inline`/`decode_loop`/`decode_loop_with_stops` fallback this
        // comment used to describe: a logprobs request now runs through the SAME
        // `generate_via_driver` path as the plain request above, so this is no
        // longer a paired negative arm against a default trace -- it is the same
        // positive assertion the un-suffixed case above makes, replayed with
        // logprobs on to prove that turning logprobs on does not change routing
        // or the sampled ids. Capturing log probabilities does not change which
        // token greedy decode picks, so the ids must be the same ids, and the
        // trace must show the same one-open-per-token / one-still-open-at-finish
        // shape driver::run maintains regardless of the logprobs flag.
        let inline_cfg = GenerateConfig {
            min_p: 0.0,
            logprobs: Some(0),
            ..cfg.clone()
        };
        let (inline_output, inline_trace) = model
            .generate_with_trace(&case.prompt, &inline_cfg)
            .unwrap_or_else(|e| panic!("case {}: inline generation failed: {e}", case.name));
        assert_eq!(
            inline_output.token_ids, case.expected_generated_ids,
            "case {}: the logprobs-requesting route diverged from the same golden",
            case.name
        );
        assert_eq!(
            inline_trace.opened,
            inline_output.token_ids.len(),
            "case {}: logprobs must route through the driver too (row R03) -- one \
                 select() per emitted token on a natural finish",
            case.name
        );
        assert_eq!(
            inline_trace.consumed + 1,
            inline_trace.opened,
            "case {}: logprobs must route through the driver too (row R03) -- \
                 exactly one prediction stays open at finish",
            case.name
        );
        assert_eq!(
            inline_output.token_logprobs.len(),
            inline_output.token_ids.len(),
            "case {}: every emitted token must carry a recorded logprob",
            case.name
        );
    }
}

// -----------------------------------------------------------------------
// ADR-092 arm 1: the CPU streaming entry, driver trace vs the same
// pre-migration golden. Mirrors the non-streaming test directly above --
// same fixture, same checkpoint, same replayed config -- but through
// `generate_streaming_with_trace`, which the non-streaming test's own
// `generate_with_trace` cannot exercise (it never reaches
// `generate_streaming_via_driver`).
// -----------------------------------------------------------------------

#[test]
#[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_CPU_GREEDY_MODEL_DIR"]
fn streaming_driver_trace_matches_ids_len_on_the_pre_migration_golden() {
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        prompt: String,
        expected_generated_ids: Vec<u32>,
        #[serde(default)]
        reasoning_budget: Option<usize>,
    }
    #[derive(serde::Deserialize)]
    struct GoldenGeneration {
        temperature: f32,
        repetition_penalty: f32,
        seed: Option<u64>,
        enable_thinking: bool,
    }
    #[derive(serde::Deserialize)]
    struct Golden {
        max_new_tokens: usize,
        generation: GoldenGeneration,
        cases: Vec<Case>,
    }

    const FIXTURE: &str = include_str!(
        "../../../../tests/fixtures/cpu_pre_migration_greedy_v1/qwen35_0_8b_cpu_greedy_tokens.json"
    );
    let golden: Golden = serde_json::from_str(FIXTURE).expect("golden fixture parses");

    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_CPU_GREEDY_MODEL_DIR");
    let model = Qwen35Model::from_safetensors(&model_dir)
        .unwrap_or_else(|e| panic!("loading {model_dir:?} failed: {e}"));

    for case in &golden.cases {
        let cfg = GenerateConfig {
            max_new_tokens: golden.max_new_tokens,
            temperature: golden.generation.temperature,
            repetition_penalty: golden.generation.repetition_penalty,
            seed: golden.generation.seed,
            enable_thinking: golden.generation.enable_thinking,
            reasoning_budget: case.reasoning_budget,
            ..Default::default()
        };

        // Acceptance 1: real consumer, real tokens -- pinned by the SAME
        // pre-migration golden the non-streaming route is pinned by.
        let mut streamed_text = String::new();
        let (output, trace) = model
            .generate_streaming_with_trace(
                &case.prompt,
                &cfg,
                |delta| {
                    streamed_text.push_str(delta);
                    true
                },
                || false,
                |_evt| {},
            )
            .unwrap_or_else(|e| panic!("case {}: streaming generation failed: {e}", case.name));

        assert_eq!(
            output.token_ids, case.expected_generated_ids,
            "case {}: streaming driver-routed ids diverged from the pre-migration golden",
            case.name
        );
        assert_eq!(
            output.text, streamed_text,
            "case {}: the returned text must equal the concatenation of every \
                 on_token delta",
            case.name
        );

        // Acceptance 2: the marker, for the named request shape (no grammar, no
        // logprobs) -- same pair the non-streaming test above uses, and the same
        // pair `driver::run`'s own doc comment names as the invariant a bypassed
        // step cannot satisfy.
        assert_eq!(
            trace.opened,
            output.token_ids.len(),
            "case {}: one select() per emitted token on a natural \
                 (non-EOS-at-step-0) finish",
            case.name
        );
        assert_eq!(
            trace.consumed + 1,
            trace.opened,
            "case {}: exactly one prediction stays open at finish",
            case.name
        );

        // Acceptance 3: row R03 deleted `generate_streaming_inline` -- a logprobs
        // streaming request now runs through the SAME `generate_streaming_via_driver`
        // path as acceptance 1 above, so this replays the same request with
        // `logprobs` set and asserts the same driver-routed shape rather than a
        // default (never-reached-the-driver) trace.
        let inline_cfg = GenerateConfig {
            min_p: 0.0,
            logprobs: Some(0),
            ..cfg.clone()
        };
        let (inline_output, inline_trace) = model
            .generate_streaming_with_trace(
                &case.prompt,
                &inline_cfg,
                |_delta| true,
                || false,
                |_evt| {},
            )
            .unwrap_or_else(|e| {
                panic!(
                    "case {}: streaming logprobs generation failed: {e}",
                    case.name
                )
            });
        assert_eq!(
            inline_output.token_ids, case.expected_generated_ids,
            "case {}: the streaming logprobs-requesting route diverged from the same \
                 golden",
            case.name
        );
        assert_eq!(
            inline_trace.opened,
            inline_output.token_ids.len(),
            "case {}: streaming logprobs must route through the driver too (row \
                 R03) -- one select() per emitted token on a natural finish",
            case.name
        );
        assert_eq!(
            inline_trace.consumed + 1,
            inline_trace.opened,
            "case {}: streaming logprobs must route through the driver too (row \
                 R03) -- exactly one prediction stays open at finish",
            case.name
        );
        assert_eq!(
            inline_output.token_logprobs.len(),
            inline_output.token_ids.len(),
            "case {}: every emitted token must carry a recorded logprob",
            case.name
        );

        // The bypass control for the marker is deliberately NOT an assert here.
        // Mutating a local copy of `trace` and asserting the mutated copy breaks
        // `consumed + 1 == opened` is arithmetic, true of any two numbers in that
        // relation, and would hold with the driver deleted. The control that
        // carries information is a source mutation -- skipping one
        // `session.decode()`/`session.select()` pair inside `decoder::driver::run`
        // -- run by hand against this test, which must redden the two marker
        // assertions above while the golden id assertion stays green. It is run
        // and recorded at review time, not committed.
    }
}

// -----------------------------------------------------------------------
// Grammar wiring — mutation-sensitive unit tests (#397)
// -----------------------------------------------------------------------

/// Grammar masking must block the argmax (highest-logit) token when the grammar
/// forbids it, causing the sampler to select a lower-logit allowed token instead.
///
/// Constructs a three-token grammar that allows "t" (index 0) and "f" (index 1)
/// but forbids "x" (index 2). Logits are set so index 2 would win greedy argmax
/// WITHOUT masking. With masking, index 2 must become NEG_INFINITY and greedy
/// sampling must return index 1 (next highest allowed logit).
///
/// Mutation sensitivity:
///   Remove the `engine.mask_logits(gs, ...)` call from the wiring site →
///   logits[2] stays 1000.0, greedy sampling returns 2, `assert_ne!(sampled, 2)`
///   fails. This proves the mask_logits call is load-bearing.
#[test]
fn grammar_masking_blocks_argmax_token() {
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    // Grammar: root ::= "t" | "f" — index 0 and 1 are valid, index 2 ("x") is not.
    let spec = GrammarSpec::Gbnf("root ::= \"t\" | \"f\"\n".to_string());
    let vocab = vec![b"t".to_vec(), b"f".to_vec(), b"x".to_vec()];
    let engine = Arc::new(GrammarEngine::new(&spec, vocab).expect("trivial grammar must compile"));

    let mut state = engine.initial_state();

    // Set logits so the forbidden token (index 2) has the highest value.
    // Without masking, greedy sampling would return 2.
    let mut logits = vec![1.0_f32, 2.0_f32, 1000.0_f32];
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("matching vocab length");

    // The forbidden token must be blocked.
    assert_eq!(
        logits[2],
        f32::NEG_INFINITY,
        "mask_logits must set the forbidden token to NEG_INFINITY"
    );

    // At least one allowed token must remain finite.
    assert!(
        has_finite_logit(&logits),
        "at least one allowed logit must survive the mask"
    );

    // Greedy sampling (temperature=0) must choose the highest ALLOWED logit,
    // which is index 1 ("f", logit 2.0), not the blocked index 2.
    let gen_cfg = GenerateConfig {
        temperature: 0.0, // greedy
        ..Default::default()
    };
    let mut rng = 1u64;
    let sampled = sample_token(&logits, &gen_cfg, &[], &mut rng);
    assert_ne!(
        sampled, 2,
        "blocked token must not be selected by the sampler"
    );
    assert_eq!(
        sampled, 1,
        "greedy must select the highest remaining allowed logit (index 1)"
    );
}

/// An all-blocking grammar mask must be detected by `has_finite_logit` so the
/// caller can return a typed error rather than silently emitting token 0.
///
/// When every logit is NEG_INFINITY, the sampler's non-finite-max short-circuit
/// returns token 0 (the lowest-id token after sorting an all-NEG_INFINITY
/// candidate set). The `has_finite_logit` guard catches this before the sampler
/// is invoked and allows the generation loop to return `InvalidInput` instead.
///
/// Mutation sensitivity:
///   Change `has_finite_logit` to always return `true` →
///   `assert!(!has_finite_logit(...))` fails. This proves the guard is load-bearing
///   for the empty-mask fail-closed path.
#[test]
fn grammar_all_blocked_mask_detected_by_has_finite_logit() {
    // Simulate a grammar engine that blocked every token.
    let all_blocked = vec![f32::NEG_INFINITY; 8];

    assert!(
        !has_finite_logit(&all_blocked),
        "all-NEG_INFINITY logit buffer must NOT pass the has_finite_logit guard; \
             the caller must return a typed error, not silently emit token 0"
    );

    // A buffer with a single finite logit must pass the guard.
    let mut one_allowed = vec![f32::NEG_INFINITY; 8];
    one_allowed[3] = 1.0_f32;
    assert!(
        has_finite_logit(&one_allowed),
        "a single finite logit must pass the guard (grammar still has valid tokens)"
    );
}

// -----------------------------------------------------------------------
// Grammar wiring — end-to-end production-seam test (#397)
// -----------------------------------------------------------------------

/// Proves that `generate()` reaches real grammar masking at its first
/// (step-0) `select` call — i.e., the production wiring is real, not just
/// the primitive tested by `grammar_masking_blocks_argmax_token`.
///
/// Strategy: build a minimal synthetic model (4 layers, 64-dim hidden,
/// 97-token vocab), then construct a grammar engine whose vocabulary table
/// is 97 empty byte sequences. `VocabPartition::build` automatically rejects
/// empty entries (they can never advance the PDA), so the precomputed bitmask
/// for the initial state is all-zeros: `mask_logits` sets every one of the 97
/// logit positions to `NEG_INFINITY`. `has_finite_logit` then fires the
/// fail-closed guard inside `QwenCpuSession::select` (`decoder::qwen_cpu`),
/// which reports `SelectOutcome::GrammarExhausted`; since no token was
/// ever sampled, `decoder::driver::run` resolves that as
/// `Err(InferenceError::GrammarConstraintBlocked)`, not a completed grammar.
///
/// Coverage: `select`'s masking call for the step-0 candidate. Every later
/// decode-loop token reaches the exact same `QwenCpuSession::select` call
/// site — there is no longer a separate "post-prefill" vs "decode-loop"
/// masking site to distinguish, since one driver loop calls `select` for
/// every token, step 0 included — so those later calls are not separately
/// covered here; they would require additional forward-step iterations
/// that cannot be isolated without a controllable-output model.
///
/// Mutation sensitivity: removing the `request.grammar_mask` call from
/// `QwenCpuSession::select` leaves logits at their raw (finite) model
/// values. `has_finite_logit` then returns `true`, `select` reports a real
/// candidate instead of `GrammarExhausted`, and this test's assertion
/// against `Err(InferenceError::GrammarConstraintBlocked(_))` fails —
/// proving the call is load-bearing.
///
/// The model-building helpers below mirror `lora_serving::build_model` in
/// tests.rs; they are duplicated here to keep generation.rs self-contained
/// without a cross-module test-support coupling.
#[test]
fn grammar_wiring_mask_logits_called_in_generate() {
    use crate::attention::gdn::GatedDeltaNetWeights;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use crate::lora_hook::NoopLoraHook;
    use crate::model::qwen35::{
        AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
        FullAttentionLayerWeights, ModelWeights,
    };
    use crate::model::qwen35_config::{LayerType, compute_layer_types};
    use crate::rope::RopeTable;
    use crate::tokenizer::bpe::BpeTokenizer;
    use std::sync::Arc;

    // Tiny 4-layer hybrid config (3 linear + 1 full), vocab_size must match
    // the grammar engine entry count so mask_logits covers all 97 logits.
    const H: usize = 64;
    const VOCAB: usize = 97;
    const I: usize = 128;
    const NUM_LAYERS: usize = 4;
    const FULL_INTERVAL: usize = 4;
    const HEAD_DIM: usize = 16;
    const LINEAR_KH: usize = 4;
    const KERNEL: usize = 4;

    let cfg = Qwen35Config {
        hidden_size: H,
        num_hidden_layers: NUM_LAYERS,
        vocab_size: VOCAB,
        intermediate_size: I,
        rms_norm_eps: 1e-6,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: HEAD_DIM,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        rope_parameters: None,
        linear_num_key_heads: LINEAR_KH,
        linear_num_value_heads: Some(LINEAR_KH),
        linear_key_head_dim: HEAD_DIM,
        linear_value_head_dim: HEAD_DIM,
        linear_conv_kernel_dim: KERNEL,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: FULL_INTERVAL,
        layer_types: compute_layer_types(NUM_LAYERS, FULL_INTERVAL),
        layer_mask: vec![true; NUM_LAYERS],
        eos_token_id: (VOCAB - 1) as u32,
        max_position_embeddings: 1024,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    // Deterministic xorshift for reproducible synthetic weights.
    fn rand_vec(rng: &mut u64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                ((*rng >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.02
            })
            .collect()
    }
    let mut rng = 0xA55E_u64 | 1;

    let qkv_dim = cfg.linear_qkv_dim();
    let out_dim = cfg.linear_output_dim();
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();

    // layer_types = [Linear, Linear, Linear, Full] for interval=4.
    let mut layers = Vec::with_capacity(NUM_LAYERS);
    for lt in &cfg.layer_types {
        let common = CommonLayerWeights {
            input_layernorm: rand_vec(&mut rng, H),
            post_attention_layernorm: rand_vec(&mut rng, H),
            ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                gate_proj: rand_vec(&mut rng, I * H),
                up_proj: rand_vec(&mut rng, I * H),
                down_proj: rand_vec(&mut rng, H * I),
            }),
        };
        let attn = match lt {
            LayerType::LinearAttention => AttentionWeights::Linear(GatedDeltaNetWeights {
                in_proj_qkv: rand_vec(&mut rng, qkv_dim * H),
                in_proj_qkv_rows: qkv_dim,
                in_proj_qkv_cols: H,
                in_proj_z: rand_vec(&mut rng, out_dim * H),
                in_proj_z_rows: out_dim,
                in_proj_z_cols: H,
                in_proj_b: rand_vec(&mut rng, LINEAR_KH * H),
                in_proj_b_rows: LINEAR_KH,
                in_proj_b_cols: H,
                in_proj_a: rand_vec(&mut rng, LINEAR_KH * H),
                in_proj_a_rows: LINEAR_KH,
                in_proj_a_cols: H,
                a_log: rand_vec(&mut rng, LINEAR_KH),
                dt_bias: rand_vec(&mut rng, LINEAR_KH),
                conv1d_weight: rand_vec(&mut rng, qkv_dim * KERNEL),
                conv_dim: qkv_dim,
                kernel_size: KERNEL,
                norm_weight: rand_vec(&mut rng, out_dim),
                out_proj: rand_vec(&mut rng, H * out_dim),
                out_proj_rows: H,
                out_proj_cols: out_dim,
            }),
            LayerType::FullAttention => AttentionWeights::Full(FullAttentionLayerWeights {
                q_proj: rand_vec(&mut rng, 2 * q_dim * H),
                k_proj: rand_vec(&mut rng, kv_dim * H),
                v_proj: rand_vec(&mut rng, kv_dim * H),
                o_proj: rand_vec(&mut rng, H * q_dim),
                q_norm: rand_vec(&mut rng, HEAD_DIM),
                k_norm: rand_vec(&mut rng, HEAD_DIM),
            }),
        };
        layers.push((attn, common));
    }

    // Minimal 7-token BPE tokenizer: "a" (id 1) serves as the one-token
    // prompt. The grammar fires before EOS (id 96) is ever needed.
    let tok_json = r#"{
  "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5," ":6},"merges":[]}
}"#;
    let tokenizer = BpeTokenizer::from_tokenizer_json_str(tok_json).expect("test tokenizer parses");

    let rope = RopeTable::new(
        cfg.rope_dim(),
        cfg.max_position_embeddings.min(8192),
        cfg.rope_theta,
    );

    let model = Qwen35Model {
        config: cfg.clone(),
        weights: ModelWeights {
            embed_tokens: rand_vec(&mut rng, VOCAB * H),
            lm_head: None,
            final_norm: rand_vec(&mut rng, H),
            layers,
        },
        tokenizer,
        rope,
        lora: Box::new(NoopLoraHook),
    };

    // Grammar engine with VOCAB empty byte sequences. VocabPartition::build
    // skips empty entries, so the precomputed bitmask for the initial state is
    // all-zeros: mask_logits sets all VOCAB logits to NEG_INFINITY.
    let vocab_bytes: Vec<Vec<u8>> = vec![vec![]; VOCAB];
    let spec = GrammarSpec::Gbnf("root ::= \"ok\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, vocab_bytes).expect("grammar engine builds with empty vocab"),
    );

    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        grammar: Some(engine),
        ..Default::default()
    };

    // The all-blocking grammar must trigger has_finite_logit inside generate()
    // and return Err(GrammarConstraintBlocked). Any other outcome is a wiring failure.
    let result = model.generate("a", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::GrammarConstraintBlocked(_))),
        "grammar blocking every token must return Err(GrammarConstraintBlocked); got {result:?}"
    );
}

#[test]
fn generate_config_default_stop_strings_empty() {
    let cfg = GenerateConfig::default();
    assert!(cfg.stop_strings.is_empty());
}

#[test]
fn generate_config_stop_strings_field_explicit() {
    let cfg = GenerateConfig {
        stop_strings: vec!["</s>".to_string(), "\nUser:".to_string()],
        ..Default::default()
    };
    assert_eq!(cfg.stop_strings.len(), 2);
    assert_eq!(cfg.stop_strings[0], "</s>");
}

// -----------------------------------------------------------------------
// check_stop_strings_not_set / check_reasoning_budget_not_set
// (ADR-080 C3, #783) — mutation-sensitive unit tests for the shared guard
// primitives every alternate CPU/Metal decode loop calls.
// -----------------------------------------------------------------------

/// Mutation sensitivity: change `check_stop_strings_not_set` to always
/// return `Ok(())` → this assertion fails, catching a regression that
/// would let a non-empty `stop_strings` silently pass through an
/// unwired decode loop.
#[test]
fn check_stop_strings_not_set_rejects_nonempty() {
    let cfg = GenerateConfig {
        stop_strings: vec!["</s>".to_string()],
        ..Default::default()
    };
    let result = check_stop_strings_not_set(&cfg);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "non-empty stop_strings must be rejected with InvalidInput; got {result:?}"
    );
}

/// Mutation sensitivity: change the guard to always return `Err(..)` →
/// this assertion fails, catching a regression that would reject every
/// caller including the default (no stop strings requested) config.
#[test]
fn check_stop_strings_not_set_allows_empty() {
    assert!(
        check_stop_strings_not_set(&GenerateConfig::default()).is_ok(),
        "empty stop_strings (the default) must be allowed"
    );
}

/// Mutation sensitivity: change `check_reasoning_budget_not_set` to
/// always return `Ok(())` → this assertion fails, catching a regression
/// that would let a set `reasoning_budget` silently pass through an
/// unwired decode loop.
#[test]
fn check_reasoning_budget_not_set_rejects_some() {
    let cfg = GenerateConfig {
        reasoning_budget: Some(128),
        ..Default::default()
    };
    let result = check_reasoning_budget_not_set(&cfg);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "Some(reasoning_budget) must be rejected with InvalidInput; got {result:?}"
    );
}

/// Mutation sensitivity: change the guard to always return `Err(..)` →
/// this assertion fails, catching a regression that would reject every
/// caller including the default (no reasoning budget requested) config.
#[test]
fn check_reasoning_budget_not_set_allows_none() {
    assert!(
        check_reasoning_budget_not_set(&GenerateConfig::default()).is_ok(),
        "reasoning_budget: None (the default) must be allowed"
    );
}

// -----------------------------------------------------------------------
// check_prompt_not_empty (#856) — the shared empty-prompt preflight every
// one of the seven CPU/Metal generation entry points now calls instead of
// its own inline `if prompt_len == 0` copy. Pure-function unit tests
// here; per-entry-point production-seam tests live alongside each real
// call site (cpu_f16.rs, cpu_q8.rs, neon_forward.rs,
// forward/metal_qwen35.rs) and `qwen35_model_generate_rejects_empty_prompt`
// below, which exercises this exact function through `Qwen35Model::generate`.
// -----------------------------------------------------------------------

/// Mutation sensitivity: change `check_prompt_not_empty` to always return
/// `Ok(())` → this assertion fails, catching a regression that would let
/// an empty prompt silently pass through every entry point that calls it
/// (the CPU-vs-Metal split #856 fixes).
#[test]
fn check_prompt_not_empty_rejects_zero() {
    let result = check_prompt_not_empty(0);
    assert!(
        matches!(result, Err(InferenceError::Inference(ref msg)) if msg.contains("empty prompt")),
        "prompt_len == 0 must be rejected with Err(Inference(\"empty prompt\")); got {result:?}"
    );
}

/// Mutation sensitivity: change the guard to always return `Err(..)` →
/// this assertion fails, catching a regression that would reject every
/// non-empty-prompt caller too.
#[test]
fn check_prompt_not_empty_allows_nonzero() {
    assert!(
        check_prompt_not_empty(1).is_ok(),
        "prompt_len == 1 (a real prompt) must be allowed"
    );
}

// -----------------------------------------------------------------------
// check_context_budget (#922) — the shared total-context admission bound
// (prompt_len + decode budget <= max_context) every CPU `generate` /
// `generate_streaming` call site (this module) and every Metal generation
// entry point (`forward::metal_qwen35`: `generate`,
// `generate_streaming_with_cancel`, `generate_multimodal`,
// `generate_streaming_with_prefix_cache_and_cancel`) now calls, instead of
// the Metal-only bug of bounding `prompt_len` alone. Pure-function unit
// tests here (no GPU/Metal device needed); per-entry-point production-seam
// tests for the Metal call sites live in `forward/metal_qwen35.rs`.
// -----------------------------------------------------------------------

/// A prompt that fits the window by itself, but whose decode budget pushes
/// `prompt_len + max_new_tokens` past `max_context`, must be rejected.
///
/// This is exactly the gap #922 reports: before the fix, the three Metal
/// entry points only checked `prompt_len > max_context` (8 <= 10 here), so
/// a request needing 8 + 5 = 13 tokens of window was wrongly admitted.
///
/// Mutation sensitivity: change `check_context_budget` to check
/// `prompt_len > max_context` instead of the summed bound → this
/// assertion fails (8 > 10 is false), catching exactly the #922 gap this
/// function exists to close.
#[test]
fn check_context_budget_rejects_prompt_plus_budget_overflow() {
    let result = check_context_budget(8, None, 5, 10);
    assert!(
        matches!(result, Err(InferenceError::Inference(ref msg))
                if msg.contains("8 tokens") && msg.contains("5") && msg.contains("10")),
        "prompt (8) + max_new_tokens (5) = 13 > max_context (10) must be \
             rejected, naming all three lengths in the message; got {result:?}"
    );
}

/// Boundary: `prompt_len + max_new_tokens == max_context` exactly must be
/// admitted (the CPU bound is `<=`, not `<`).
///
/// Mutation sensitivity: change the comparison from `>` to `>=` → this
/// assertion fails, catching a regression that would reject the exact
/// boundary request the CPU path (and the HTTP server) accepts.
#[test]
fn check_context_budget_admits_exact_boundary() {
    assert!(
        check_context_budget(8, None, 2, 10).is_ok(),
        "prompt (8) + max_new_tokens (2) == max_context (10) must be admitted"
    );
}

/// A prompt that fits alone AND fits with its budget must be admitted.
#[test]
fn check_context_budget_admits_well_within_bound() {
    assert!(
        check_context_budget(4, None, 2, 10).is_ok(),
        "prompt (4) + max_new_tokens (2) = 6 <= max_context (10) must be admitted"
    );
}

/// A reasoning budget extends the effective decode cap by
/// `reasoning_budget + 1` (the forced `</think>` delimiter, see
/// `decode_cap`), so a request that fits on `max_new_tokens` alone must
/// still be rejected once its reasoning budget is accounted for.
///
/// Mutation sensitivity: change `check_context_budget` to ignore
/// `reasoning_budget` (pass `max_new_tokens` directly to the bound
/// instead of `decode_cap(reasoning_budget, max_new_tokens)`) → this
/// assertion fails (8 + 2 = 10 <= 10 would wrongly admit), catching a
/// regression that silently drops budgeted-reasoning requests from the
/// bound.
#[test]
fn check_context_budget_accounts_for_reasoning_budget() {
    // decode_cap(Some(3), 2) = 3 + 2 + 1 = 6; prompt 8 + 6 = 14 > 10.
    let result = check_context_budget(8, Some(3), 2, 10);
    assert!(
        matches!(
            result,
            Err(InferenceError::Inference(ref message))
                if message == "prompt (8 tokens) plus effective decode cap (6 tokens; \
                               max_new_tokens=2, reasoning_budget=3) exceeds model context \
                               window (10)"
        ),
        "reasoning-budget admission errors must report the effective cap and its inputs; \
             got {result:?}"
    );
}

/// `reasoning_budget` is inert when `enable_thinking` is false: the decode
/// loop's actual iteration cap (`DecodePolicy::cap()`, what every CPU and
/// Metal generation loop bounds `for _ in 1..cap` against) must equal
/// `max_new_tokens`, not `decode_cap`'s budgeted `rb + max_new_tokens + 1`
/// -- there is no reasoning block to close, so reserving room for a forced
/// `</think>` that `force_close_think` can never emit is dead capacity at
/// best and an unenforceable overrun at worst.
///
/// Mutation sensitivity: change `DecodePolicy::init` back to
/// `reasoning_budget: gen_cfg.reasoning_budget` (the raw field, no
/// `enable_thinking` gate) → `cap()` returns
/// `decode_cap(Some(1), 1) == 3` instead of `1`, and this assertion fails.
#[test]
fn decode_policy_cap_ignores_reasoning_budget_when_thinking_disabled() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(1),
        enable_thinking: false,
        max_new_tokens: 1,
        ..GenerateConfig::default()
    };
    let mut token_logprobs = Vec::new();
    let policy = DecodePolicy::init(
        &gen_cfg,
        /* think_close_id */ None,
        /* reasoning_newline_boundary */ None,
        &mut token_logprobs,
        /* first_emitted_id */ 0,
        /* first_logits */ &[],
        /* temperature */ 0.0,
        /* first_generated_len */ 1,
        /* streaming */ false,
    );
    assert_eq!(
        policy.cap(),
        1,
        "decode cap must equal max_new_tokens (1) when enable_thinking is false, \
             not decode_cap(Some(1), 1) == 3"
    );
}

/// Sibling outcome to [`decode_policy_cap_ignores_reasoning_budget_when_thinking_disabled`]:
/// total-context admission (`check_context_budget`, called with
/// `GenerateConfig::effective_reasoning_budget`) must also ignore a
/// positive `reasoning_budget` once thinking is disabled. A request whose
/// prompt plus `max_new_tokens` fits the window, but whose prompt plus the
/// *budgeted* cap would not, must be admitted rather than rejected for
/// exceeding a decode reach the loop will never attempt.
///
/// Mutation sensitivity: change `effective_reasoning_budget` to return
/// `self.reasoning_budget` unconditionally (no `enable_thinking` filter)
/// → `check_context_budget(8, Some(3), 2, 10)` rejects (14 > 10) and this
/// assertion fails.
#[test]
fn context_admission_ignores_reasoning_budget_when_thinking_disabled() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(3),
        enable_thinking: false,
        max_new_tokens: 2,
        ..GenerateConfig::default()
    };
    // Honouring the budget would compute decode_cap(Some(3), 2) = 6, and
    // 8 + 6 = 14 > 10 would be rejected (see
    // check_context_budget_accounts_for_reasoning_budget above, same
    // inputs with enable_thinking's effect factored out). Ignoring it
    // computes decode_cap(None, 2) = 2, and 8 + 2 = 10 <= 10 is admitted.
    let result = check_context_budget(
        8,
        gen_cfg.effective_reasoning_budget(),
        gen_cfg.max_new_tokens,
        10,
    );
    assert!(
        result.is_ok(),
        "prompt (8) + max_new_tokens (2) == max_context (10) must be admitted when \
             thinking is disabled, even with reasoning_budget=Some(3) set; got {result:?}"
    );
}

// -----------------------------------------------------------------------
// StopReason mutation-sensitive tests (#456)
// -----------------------------------------------------------------------
//
// Each test exercises one specific code path that sets stop_reason and
// asserts the exact variant. If the wrong variant is assigned at that
// site, the assertion fails — proving the assignment is load-bearing.
//
// Build helper: all-zero weights → all logits == 0.0 after any forward
// pass → greedy sampling always picks token 0 (first-wins on equal logits).
// This gives deterministic sampling without relying on random-weight outputs.
//
// The zero-weight fixture itself lives in `qwen35::test_support`
// (reachable here via its `cfg(any(test, feature = "test-utils"))`
// gate), so this crate's own library tests and `bin/lattice.rs`'s
// separate `--features test-utils` compilation unit build the identical
// fixture from one definition instead of two copies (#816).
use super::super::test_support::{
    tiny_zero_model as build_tiny_zero_model,
    tiny_zero_model_with_tokenizer as build_tiny_zero_model_tok,
};

/// `qwen35::test_support` must be reachable from this crate's own
/// library tests via plain `cfg(test)`, without enabling the
/// `test-utils` feature — otherwise `generation.rs`'s `StopReason`
/// tests would have no zero-weight fixture to build against in a
/// default `cargo test -p lattice-inference` run (#816).
///
/// Mutation sensitivity: revert `test_support`'s gate on
/// `qwen35/mod.rs` from `cfg(any(test, feature = "test-utils"))` back
/// to `cfg(feature = "test-utils")` alone → this module fails to
/// compile under a default (no `--features test-utils`) `cargo test`,
/// so this test (and every other test in this module using
/// `build_tiny_zero_model`/`build_tiny_zero_model_tok`) fails closed
/// with a compile error rather than silently reintroducing a private
/// duplicate fixture.
#[test]
fn test_support_zero_model_reachable_without_test_utils_feature() {
    let via_alias = build_tiny_zero_model();
    let via_direct_path = super::super::test_support::tiny_zero_model();
    assert_eq!(
        via_alias.config.hidden_size,
        via_direct_path.config.hidden_size
    );
    assert_eq!(
        via_alias.weights.embed_tokens,
        via_direct_path.weights.embed_tokens
    );
}

const A_FIRST_TINY_TOK_JSON: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"a":0,"<unk>":1,"b":2,"c":3,"d":4,"e":5," ":6},"merges":[]}
}"#;

fn build_a_first_zero_model() -> Qwen35Model {
    build_tiny_zero_model_tok(A_FIRST_TINY_TOK_JSON)
}

fn a_token_grammar(gbnf: &str) -> std::sync::Arc<crate::grammar::GrammarEngine> {
    use crate::grammar::{GrammarEngine, GrammarSpec};

    let mut vocab_bytes = vec![Vec::new(); 97];
    vocab_bytes[0] = b"a".to_vec();
    std::sync::Arc::new(
        GrammarEngine::new(&GrammarSpec::Gbnf(gbnf.to_string()), vocab_bytes)
            .expect("test grammar compiles"),
    )
}

fn grammar_config(gbnf: &str) -> GenerateConfig {
    GenerateConfig {
        max_new_tokens: 4,
        temperature: 0.0,
        grammar: Some(a_token_grammar(gbnf)),
        stop_token_ids: vec![],
        ..Default::default()
    }
}

#[test]
fn completed_grammar_stops_before_remaining_budget_errors() {
    let model = build_a_first_zero_model();
    let result = model
        .generate("b", &grammar_config("root ::= \"a\"\n"))
        .expect("a completed grammar must stop successfully");

    assert_eq!(result.text, "a");
    assert_eq!(result.token_ids, vec![0]);
    assert_eq!(result.generated_tokens, 1);
    assert!(result.stopped);
    assert_eq!(result.stop_reason, Some(StopReason::Grammar));
}

#[test]
fn shared_prefix_accepting_state_continues_while_a_longer_alternative_is_live() {
    let model = build_a_first_zero_model();
    // After "a" the grammar is complete, but the alternative `"aa"` is
    // still live, so the accepting state has a continuation and generation
    // goes on. After "aa" nothing can follow, which is where it stops. This
    // pins a no-continuation stop, not a general
    // stop-at-first-accepting-state policy.
    let result = model
        .generate("b", &grammar_config("root ::= \"a\" | \"aa\"\n"))
        .expect("the accepting state must terminate without an all-masked error");

    assert_eq!(result.text, "aa");
    assert_eq!(result.token_ids, vec![0, 0]);
    assert!(result.stopped);
    assert_eq!(result.stop_reason, Some(StopReason::Grammar));
}

#[test]
fn completed_grammar_streaming_matches_nonstreaming() {
    let model = build_a_first_zero_model();
    let gen_cfg = grammar_config("root ::= \"a\"\n");
    let nonstreaming = model
        .generate("b", &gen_cfg)
        .expect("non-streaming completion succeeds");
    let mut streamed_text = String::new();
    let streaming = model
        .generate_streaming("b", &gen_cfg, |delta| streamed_text.push_str(delta))
        .expect("streaming completion succeeds");

    assert_eq!(streaming.text, streamed_text);
    assert_eq!(streaming.text, nonstreaming.text);
    assert_eq!(streaming.token_ids, nonstreaming.token_ids);
    assert_eq!(streaming.generated_tokens, nonstreaming.generated_tokens);
    assert_eq!(streaming.stopped, nonstreaming.stopped);
    assert_eq!(streaming.stop_reason, nonstreaming.stop_reason);
    assert_eq!(streaming.stop_reason, Some(StopReason::Grammar));
}

#[test]
fn grammar_completion_after_decode_step_covers_all_loop_siblings() {
    let model = build_a_first_zero_model();

    for stop_strings in [vec![], vec!["never".to_string()]] {
        let mut gen_cfg = grammar_config("root ::= \"aa\"\n");
        gen_cfg.stop_strings = stop_strings;

        let nonstreaming = model
            .generate("b", &gen_cfg)
            .expect("non-streaming loop must stop on grammar completion");
        let mut streamed_text = String::new();
        let streaming = model
            .generate_streaming("b", &gen_cfg, |delta| streamed_text.push_str(delta))
            .expect("streaming loop must stop on grammar completion");

        for result in [&nonstreaming, &streaming] {
            assert_eq!(result.text, "aa");
            assert_eq!(result.token_ids, vec![0, 0]);
            assert_eq!(result.generated_tokens, 2);
            assert!(result.stopped);
            assert_eq!(result.stop_reason, Some(StopReason::Grammar));
        }
        assert_eq!(streaming.text, streamed_text);
    }
}

/// ADR-090 row R03 equivalence pin, written and run against the
/// pre-migration `generate_inline`/`decode_loop`/`decode_loop_with_stops`
/// implementation BEFORE those functions are deleted and grammar +
/// logprobs are routed through `decoder::driver::run`: grammar-constrained
/// decoding AND per-token logprob capture, combined, covering both the
/// step-0 sample (`DecodePolicy::init`) and a mid-loop sample that
/// completes the grammar (`DecodePolicy::transition`), across both the
/// fast (no `stop_strings`) and stop-strings decode paths, and across
/// both `generate` and `generate_streaming`. This is the "same
/// seed/config, identical grammar-constrained tokens, identical
/// logprobs, identical streaming event sequence" equivalence check the
/// row's acceptance criteria ask for -- expressed as a pinned expected
/// value (this crate has no feature flag to run the same request through
/// two loops side by side post-cutover, so the legacy run's own output,
/// captured here before the routing change, IS the pin).
///
/// Every recorded logprob is deterministically 0.0: `a_token_grammar`'s
/// `root ::= "aa"` grammar has exactly one legal byte sequence ("a",
/// vocab index 0) in its 97-entry vocab table, so `mask_logits` always
/// leaves exactly one finite logit -- softmax over a single candidate
/// assigns it probability 1.0, and `ln(1.0) == 0.0` -- making the exact
/// value reproducible on the zero-weight fixture without a real
/// checkpoint.
///
/// Mutation sensitivity (recorded at review time per the row's
/// acceptance criteria, not committed): routing `record_metadata`
/// (through the session) / the grammar-advance step (through the
/// driver's own owned state) at the wrong point (e.g. scoring against
/// pre-mask logits, or skipping the mid-loop grammar-advance call)
/// changes either `token_logprobs` or `token_ids` here without
/// necessarily changing the existing grammar-only / logprobs-only tests
/// above, which is exactly the combined-path gap this test closes.
#[test]
fn grammar_and_logprobs_combined_equivalence_across_stop_strings_and_streaming() {
    let model = build_a_first_zero_model();

    for stop_strings in [vec![], vec!["never".to_string()]] {
        let mut gen_cfg = grammar_config("root ::= \"aa\"\n");
        gen_cfg.stop_strings = stop_strings.clone();
        gen_cfg.logprobs = Some(1);

        let nonstreaming = model
            .generate("b", &gen_cfg)
            .expect("combined grammar+logprobs non-streaming generate must succeed");
        let mut streamed_text = String::new();
        let streaming = model
            .generate_streaming("b", &gen_cfg, |delta| streamed_text.push_str(delta))
            .expect("combined grammar+logprobs streaming generate must succeed");

        for (label, result) in [("non-streaming", &nonstreaming), ("streaming", &streaming)] {
            assert_eq!(result.text, "aa", "{label}: stop_strings={stop_strings:?}");
            assert_eq!(
                result.token_ids,
                vec![0, 0],
                "{label}: stop_strings={stop_strings:?}"
            );
            assert_eq!(result.generated_tokens, 2, "{label}");
            assert!(result.stopped, "{label}");
            assert_eq!(result.stop_reason, Some(StopReason::Grammar), "{label}");

            assert_eq!(
                result.token_logprobs.len(),
                2,
                "{label}: one recorded logprob per emitted token -- the first from \
                     DecodePolicy::init (step 0), the second from a mid-loop transition"
            );
            for (i, entry) in result.token_logprobs.iter().enumerate() {
                assert_eq!(entry.token_id, 0, "{label}: entry {i}");
                assert!(
                    (entry.logprob - 0.0).abs() < 1e-6,
                    "{label}: entry {i} logprob={} (grammar leaves exactly one legal \
                         token at every step, so probability is exactly 1.0)",
                    entry.logprob
                );
                assert_eq!(
                    entry.top.len(),
                    1,
                    "{label}: entry {i} requested top_logprobs=Some(1) (top_n=1)"
                );
                assert_eq!(entry.top[0].token_id, 0, "{label}: entry {i}");
            }
        }

        assert_eq!(streaming.text, streamed_text);
    }
}

/// `max_new_tokens == 0` must set `stop_reason = Some(StopReason::Length)` on the
/// early-return path that fires before any forward pass or token sampling.
///
/// Mutation sensitivity: changing `StopReason::Length` in the `max_new_tokens == 0`
/// guard to any other variant causes `assert_eq!(stop_reason, ...)` to fail.
#[test]
fn stop_reason_length_on_zero_max_tokens() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 0,
        temperature: 0.0,
        ..Default::default()
    };
    let result = model
        .generate("a", &gen_cfg)
        .expect("zero-token generate must succeed");
    assert_eq!(
        result.stop_reason,
        Some(StopReason::Length),
        "max_new_tokens == 0 must return StopReason::Length; got {:?}",
        result.stop_reason
    );
    assert_eq!(
        result.generated_tokens, 0,
        "zero max_new_tokens must produce no tokens"
    );
}

/// `Qwen35Model::generate` must reject an empty prompt with a typed
/// `Err` via the shared `check_prompt_not_empty` guard (#856) -- this is
/// a bonus dedup alongside the seven audited entry points: this
/// function already had this exact behavior before #856 (it is not one
/// of the CPU-vs-Metal divergent paths), but its inline check is now
/// routed through the same shared preflight rather than carrying its
/// own duplicate copy, so this test also proves that wiring.
///
/// Mutation sensitivity: reverting this call site back to a no-op (or
/// removing it) lets an empty prompt reach `all_ids.last()` in the
/// decode loop with a zero-length prompt, which either panics or
/// silently generates from no context -- `result.is_err()` fails either
/// way.
#[test]
fn qwen35_model_generate_rejects_empty_prompt() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 4,
        temperature: 0.0,
        ..Default::default()
    };
    let result = model.generate("", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::Inference(ref msg)) if msg.contains("empty prompt")),
        "Qwen35Model::generate must reject an empty prompt with \
             Err(Inference(\"empty prompt\")); got {result:?}"
    );
}

/// First sampled token matching a `stop_token_ids` entry must set
/// `stop_reason = Some(StopReason::Eos)` on the early-return path.
///
/// Zero-weight model → all logits == 0.0 → greedy picks token 0 (first-wins on ties).
/// `stop_token_ids = [0]` causes `should_stop_token` to fire on the first decode step.
///
/// Mutation sensitivity: changing `StopReason::Eos` at the `should_stop_token` return
/// site to any other variant causes `assert_eq!(stop_reason, Some(StopReason::Eos))` to fail.
#[test]
fn stop_reason_eos_on_first_stop_token() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 5,
        temperature: 0.0,
        stop_token_ids: vec![0], // token 0 is greedy-sampled with all-zero logits
        ..Default::default()
    };
    let result = model
        .generate("a", &gen_cfg)
        .expect("eos-on-first-token generate must succeed");
    assert_eq!(
        result.stop_reason,
        Some(StopReason::Eos),
        "stop_token_ids match on first token must return StopReason::Eos; got {:?}",
        result.stop_reason
    );
}

/// `Qwen35Model::set_eos_token_id().eos_token_id = u32::MAX` combined with
/// `GenerateConfig::stop_token_ids: vec![]` (the flagship CPU/Metal
/// benchmark determinism knob -- `qwen35_generate --emit-phase-events`,
/// PR #882) must force continuation past a token that would otherwise
/// stop generation on the very first step, so a benchmark trial always
/// decodes the exact requested `max_new_tokens` count.
///
/// This is the established idiom (`cfg.eos_token_id = u32::MAX`) used
/// throughout this crate's own test suite to push EOS out of the
/// reachable vocab range, rather than a dedicated `GenerateConfig`
/// field: `GenerateConfig` is a plain public-literal struct with a
/// `Default` impl, so a new field there is
/// `constructible_struct_adds_field` under `cargo-semver-checks`, a
/// semver-major break `Qwen35Model.config` (private, `config_mut`
/// setter) does not incur.
///
/// Two-phase design (baseline, then override), both driven through
/// `set_eos_token_id()`: `build_tiny_zero_model` always greedy-samples token 0
/// from this all-zero-logit fixture (see the sibling comment on
/// `stop_reason_eos_on_first_stop_token`), so phase 1 first points the
/// model's own `eos_token_id` AT 0 via `set_eos_token_id()` and confirms that
/// alone stops generation immediately (`StopReason::Eos`, 0 tokens
/// emitted) -- this is itself mutation-sensitive to `set_eos_token_id()`
/// returning a real reference into
/// `Qwen35Model`'s private `config` field rather than, say, a detached
/// clone: a detached clone would leave the real `eos_token_id` at its
/// original `VOCAB - 1 = 96`, which the greedy-sampled token 0 never
/// matches, so phase 1's own assertions would fail first. Phase 2 then
/// re-points `eos_token_id` at the unreachable `u32::MAX` sentinel and
/// confirms the SAME config now runs to `max_new_tokens` instead.
#[test]
fn eos_token_id_override_forces_continuation_past_matching_stop_token() {
    let mut model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 5,
        temperature: 0.0,
        stop_token_ids: vec![], // benchmark profile: no configured stop tokens either
        ..Default::default()
    };

    // Phase 1 (baseline): point eos_token_id at the always-sampled
    // token 0 -- must stop after exactly 1 token.
    model.set_eos_token_id(0);
    let baseline = model
        .generate("a", &gen_cfg)
        .expect("baseline generate must succeed");
    assert_eq!(
        baseline.stop_reason,
        Some(StopReason::Eos),
        "sanity: eos_token_id = 0 must stop generation on the first greedy-sampled \
             token (this fixture always samples token 0); got {:?} -- if this fails, \
             set_eos_token_id() is not reaching the real config should_stop_token reads",
        baseline.stop_reason
    );
    assert_eq!(
        baseline.generated_tokens, 0,
        "sanity: eos_token_id = 0 must stop before any token is emitted into the \
             output (the matching token itself is excluded, per should_stop_token's \
             stop-before-append semantics -- see stop_reason_eos_on_first_stop_token \
             above), got {}",
        baseline.generated_tokens
    );

    // Phase 2 (the benchmark override): push eos_token_id out of the
    // reachable vocab range -- the same config now runs to max_new_tokens.
    model.set_eos_token_id(u32::MAX);
    let result = model
        .generate("a", &gen_cfg)
        .expect("eos-override generate must succeed");
    assert_eq!(
        result.stop_reason,
        Some(StopReason::Length),
        "overriding eos_token_id out of range must force the loop to run to \
             max_new_tokens (StopReason::Length), not stop early; got {:?}",
        result.stop_reason
    );
    assert_eq!(
        result.generated_tokens, 5,
        "eos_token_id override must decode exactly max_new_tokens (5), got {}",
        result.generated_tokens
    );
}

/// Sibling of the test above, covering `should_stop_token`'s
/// `cfg.eos_token_id` branch directly (the pure predicate `generate()`'s
/// decode loop calls every step) rather than threading a real decode
/// run through to confirm the override takes effect.
#[test]
fn eos_token_id_override_suppresses_match_in_should_stop_token() {
    let mut model = build_tiny_zero_model();
    let base_cfg = GenerateConfig {
        stop_token_ids: vec![],
        ..Default::default()
    };
    let original_eos_token_id = model.config.eos_token_id;
    assert!(
        should_stop_token(&model.config, &base_cfg, original_eos_token_id),
        "sanity: without the override, the model's own eos_token_id must stop generation"
    );
    model.set_eos_token_id(u32::MAX);
    assert!(
        !should_stop_token(&model.config, &base_cfg, original_eos_token_id),
        "eos_token_id override must suppress a match against the model's original \
             (now-superseded) eos_token_id -- the sentinel u32::MAX itself trivially \
             matches u32::MAX, so this must check the ORIGINAL id stays unmatched"
    );
}

/// Grammar `advance` returning `false` on the first sampled token must set
/// `stop_reason = Some(StopReason::Grammar)`.
///
/// Mechanism: the grammar engine has vocab_size = 1 (["t"]). `mask_logits` blocks
/// token 0 ("t"); tokens 1..96 (beyond grammar vocab_size) stay at 0.0 and are
/// finite. Greedy picks token 1. `advance(1)`: 1 >= grammar.vocab_size (1) → `false`.
///
/// Mutation sensitivity: changing `StopReason::Grammar` at the `advance`-returns-false
/// return site to any other variant causes the assertion to fail.
#[test]
fn stop_reason_grammar_on_advance_false() {
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let model = build_tiny_zero_model();

    // vocab = ["t"] (size 1). Grammar root ::= "x" blocks token 0 via mask;
    // tokens 1..96 remain finite. Greedy picks token 1. advance(1): 1 >= 1 → false.
    let spec = GrammarSpec::Gbnf("root ::= \"x\"\n".to_string());
    let vocab = vec![b"t".to_vec()];
    let engine = Arc::new(GrammarEngine::new(&spec, vocab).expect("single-token grammar compiles"));

    let gen_cfg = GenerateConfig {
        max_new_tokens: 5,
        temperature: 0.0,
        grammar: Some(engine),
        stop_token_ids: vec![],
        ..Default::default()
    };
    let result = model
        .generate("a", &gen_cfg)
        .expect("grammar-advance-false generate must succeed");
    assert_eq!(
        result.stop_reason,
        Some(StopReason::Grammar),
        "grammar advance returning false must return StopReason::Grammar; got {:?}",
        result.stop_reason
    );
}

// Same tiny zero-weight model, but the tokenizer carries `</think>` as added
// token id 7 so reasoning-budget forcing can fire. The marker is deliberately
// `special:false` to cover content-based added-token resolution.
fn build_tiny_thinking_model() -> Qwen35Model {
    build_tiny_zero_model_tok(
        r#"{
  "version":"1.0","truncation":null,"padding":null,
  "added_tokens":[{"id":7,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5," ":6},"merges":[]}
}"#,
    )
}

fn build_tiny_newline_thinking_model() -> Qwen35Model {
    build_tiny_zero_model_tok(
        r#"{
  "version":"1.0","truncation":null,"padding":null,
  "added_tokens":[{"id":7,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"Ċ":0,"<unk>":1,"a":2,"b":3,"c":4,"d":5,"e":6},"merges":[]}
}"#,
    )
}

fn boundary_at(id: usize) -> std::sync::Arc<[bool]> {
    let mut table = vec![false; id.saturating_add(1).max(10)];
    table[id] = true;
    table.into()
}

fn reasoning_budget_parity(model: &Qwen35Model, gen_cfg: &GenerateConfig) {
    let nonstreaming = model
        .generate("a", gen_cfg)
        .expect("non-streaming budgeted generation must succeed");
    let mut streamed_text = String::new();
    let streaming = model
        .generate_streaming("a", gen_cfg, |delta| streamed_text.push_str(delta))
        .expect("streaming budgeted generation must succeed");

    assert_eq!(
        streaming.text, nonstreaming.text,
        "streamed text must equal non-streamed text"
    );
    assert_eq!(
        streamed_text, streaming.text,
        "the streamed deltas must concatenate to the generation output"
    );
    assert_eq!(streaming.token_ids, nonstreaming.token_ids);
    let (streamed_reasoning, streamed_answer) = streaming
        .text
        .split_once("</think>")
        .expect("the budgeted request must emit a close marker");
    let (nonstreamed_reasoning, nonstreamed_answer) = nonstreaming
        .text
        .split_once("</think>")
        .expect("the non-streamed request must emit a close marker");
    assert_eq!(
        streamed_reasoning, nonstreamed_reasoning,
        "streamed reasoning text must equal non-streamed reasoning text"
    );
    assert_eq!(
        streamed_answer, nonstreamed_answer,
        "streamed answer text must equal non-streamed answer text"
    );
}

#[test]
fn reasoning_budget_streaming_matches_nonstreaming_on_tiny_zero_model() {
    let gen_cfg = GenerateConfig {
        max_new_tokens: 40,
        temperature: 0.0,
        reasoning_budget: Some(4),
        enable_thinking: true,
        stop_token_ids: vec![],
        ..Default::default()
    };
    reasoning_budget_parity(&build_tiny_thinking_model(), &gen_cfg);
    reasoning_budget_parity(&build_tiny_newline_thinking_model(), &gen_cfg);
}

#[test]
fn reasoning_budget_cap_can_end_generation_before_the_close() {
    let model = build_tiny_thinking_model();
    let close_id = model
        .tokenizer
        .special_token_id("</think>")
        .expect("thinking model tokenizer resolves </think>");
    let gen_cfg = GenerateConfig {
        max_new_tokens: 2,
        temperature: 0.0,
        reasoning_budget: Some(4),
        enable_thinking: true,
        stop_token_ids: vec![],
        ..Default::default()
    };

    let output = model
        .generate("a", &gen_cfg)
        .expect("a cap-limited budgeted request must succeed");

    assert_eq!(
        output.stop_reason,
        Some(StopReason::Length),
        "the decode cap, not the deferred close, must end this request"
    );
    assert_eq!(
        output.generated_tokens, 7,
        "decode_cap(Some(4), 2) must remain 7"
    );
    assert!(
        !output.token_ids.contains(&close_id),
        "the close must not be emitted when the cap ends first"
    );
    assert!(
        !output.text.contains("</think>"),
        "the rendered output must leave the thinking block open"
    );
}

/// `PrefillEnd` must fire on the post-prefill cancellation early return
/// too: prefill genuinely completed before that check runs, so an
/// observer that pairs prefill start/end never sees an unterminated
/// prefill just because the client disconnected during it.
///
/// Mechanism: `should_cancel` returns false at the pre-prefill checkpoint
/// and true at the post-prefill one, so this takes the second early
/// return specifically. With the `on_raw_event(PrefillEnd)` call sited
/// after that check (as it was), no event is observed at all here.
#[test]
fn prefill_end_fires_before_post_prefill_cancel_return() {
    let model = build_tiny_zero_model();
    let calls = std::cell::Cell::new(0usize);
    let events = std::cell::RefCell::new(Vec::<RawGenEvent>::new());

    let result = model
        .generate_streaming_with_observer(
            "a",
            &GenerateConfig {
                max_new_tokens: 3,
                temperature: 0.0,
                stop_token_ids: vec![],
                ..Default::default()
            },
            |_delta| true,
            || {
                let n = calls.get();
                calls.set(n + 1);
                n > 0
            },
            |evt| events.borrow_mut().push(evt),
        )
        .expect("a post-prefill cancellation returns Ok(Interrupt)");

    assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
    let events = events.into_inner();
    assert_eq!(
        events.first(),
        Some(&RawGenEvent::PrefillEnd),
        "prefill completed before the cancellation check, so PrefillEnd must \
             still have fired; got {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RawGenEvent::RawToken { .. })),
        "no token was committed on the cancellation path: {events:?}"
    );
}

/// Resolving `</think>` only through the tokenizer's special-token table
/// leaves this base-vocabulary marker undiscovered and makes the
/// resolver return an error instead of token 7.
#[test]
fn reasoning_budget_resolves_close_marker_by_token_content() {
    use std::collections::HashMap;

    let tokenizer = BpeTokenizer::from_vocab_and_merges(
        HashMap::from([("</think>".to_string(), 7)]),
        Vec::new(),
    )
    .expect("base-vocabulary tokenizer must load");
    assert_eq!(tokenizer.special_token_id("</think>"), None);
    assert_eq!(
        resolve_reasoning_close_token(&tokenizer, Some(1), true, 97)
            .expect("base-vocabulary marker must resolve"),
        Some(7)
    );
}

/// `reasoning_budget` is inert when `enable_thinking` is false, in both
/// tokenizer states: a tokenizer with a resolvable `</think>` marker
/// (which would otherwise resolve to `Some(id)`) and one with none at all
/// (which would otherwise reject with `InvalidInput`). Before this fix
/// the two functions disagreed about what `enable_thinking: false` means
/// for `reasoning_budget` -- `force_close_think` already treated it as a
/// no-op, but `resolve_reasoning_close_token` had no `enable_thinking`
/// parameter at all, so a marker-absent tokenizer under this combination
/// was rejected for a budget that could never have been enforced or
/// violated either way.
#[test]
fn reasoning_budget_inert_when_thinking_disabled_in_both_tokenizer_states() {
    use std::collections::HashMap;

    let with_marker = BpeTokenizer::from_vocab_and_merges(
        HashMap::from([("</think>".to_string(), 7)]),
        Vec::new(),
    )
    .expect("base-vocabulary tokenizer must load");
    assert_eq!(
        resolve_reasoning_close_token(&with_marker, Some(1), false, 97)
            .expect("enable_thinking: false must never error, marker present or not"),
        None,
        "a marker-resolvable tokenizer must still resolve to None when thinking is \
             disabled -- force_close_think never reads a close id in that state, so \
             validating and returning one here would be dead work at best and a false \
             rejection (e.g. on a stricter vocab_size bound) at worst"
    );

    let without_marker =
        BpeTokenizer::from_vocab_and_merges(HashMap::from([("a".to_string(), 0)]), Vec::new())
            .expect("marker-free tokenizer must load");
    assert!(
        lattice_tokenizer::__internal::token_ids_for_content(&without_marker, "</think>")
            .is_empty()
    );
    assert_eq!(
        resolve_reasoning_close_token(&without_marker, Some(1), false, 97)
            .expect("enable_thinking: false must never error, marker present or not"),
        None,
        "a budget that could not have been enforced or violated either way (no \
             reasoning block is produced when thinking is disabled) must not be rejected \
             just because the tokenizer happens to lack a </think> marker"
    );

    // Control: the identical marker-absent tokenizer and budget WITH
    // thinking enabled must still reject, exactly as
    // `reasoning_budget_rejects_tokenizer_without_close_marker` pins end
    // to end. Without this, a resolver that always returned `Ok(None)`
    // regardless of `enable_thinking` would pass both assertions above.
    let result = resolve_reasoning_close_token(&without_marker, Some(1), true, 97);
    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("</think>") && message.contains("reasoning_budget")
        ),
        "enable_thinking: true on the same marker-absent tokenizer and budget must \
             still fail closed; got {result:?}"
    );
}

/// The disabled path must stay exactly as it was: no budget (or a zero
/// budget) resolves to `None` and never errors, even for a tokenizer with
/// no `</think>` marker at all -- byte-identical to pre-fix behaviour,
/// which the e2e-parity gate pins.
#[test]
fn reasoning_budget_absent_resolves_to_none_without_error() {
    use std::collections::HashMap;

    let tokenizer =
        BpeTokenizer::from_vocab_and_merges(HashMap::from([("a".to_string(), 0)]), Vec::new())
            .expect("marker-free tokenizer must load");
    assert!(
        lattice_tokenizer::__internal::token_ids_for_content(&tokenizer, "</think>").is_empty()
    );
    assert_eq!(
        resolve_reasoning_close_token(&tokenizer, None, true, 97)
            .expect("None budget never errors"),
        None
    );
    assert_eq!(
        resolve_reasoning_close_token(&tokenizer, Some(0), true, 97)
            .expect("zero budget never errors"),
        None
    );

    // End to end: a full generation with no budget set succeeds on a
    // tokenizer that could not resolve the marker.
    let model = build_tiny_zero_model();
    let output = model
        .generate(
            "a",
            &GenerateConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                stop_token_ids: vec![],
                ..Default::default()
            },
        )
        .expect("no-budget generation must be unaffected by close-marker resolution");
    assert_eq!(output.generated_tokens, 2);
}

/// Accepting an unresolved closing marker returns an unbudgeted `Ok`
/// output, so this typed-error assertion fails.
#[test]
fn reasoning_budget_rejects_tokenizer_without_close_marker() {
    let model = build_tiny_zero_model();
    let result = model.generate(
        "a",
        &GenerateConfig {
            max_new_tokens: 1,
            temperature: 0.0,
            enable_thinking: true,
            reasoning_budget: Some(1),
            stop_token_ids: vec![],
            ..Default::default()
        },
    );

    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("</think>") && message.contains("reasoning_budget")
        ),
        "an unenforceable reasoning budget must fail with a descriptive InvalidInput; got {result:?}"
    );
}

/// Out-of-range `</think>` id: a mismatched tokenizer can declare
/// `</think>` in its base vocabulary at an id the loaded model has no
/// embedding/logit row for. `forward_step` only `assert!`s on an
/// out-of-range `token_id` (a release panic), so the resolver -- the one
/// place upstream of every forced-decode call site -- must reject it with
/// a typed error instead of ever returning it.
///
/// The `vocab_size` bound check must reject id 7 with `Err`, not resolve
/// it as `Ok(Some(7))`.
#[test]
fn reasoning_budget_rejects_close_marker_id_at_or_beyond_vocab_size() {
    use std::collections::HashMap;

    let tokenizer = BpeTokenizer::from_vocab_and_merges(
        HashMap::from([("</think>".to_string(), 7)]),
        Vec::new(),
    )
    .expect("base-vocabulary tokenizer must load");

    // vocab_size = 7 means valid ids are 0..=6; the resolved id (7) is
    // out of range.
    let result = resolve_reasoning_close_token(&tokenizer, Some(1), true, 7);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(ref message)) if message.contains("vocab")),
        "a </think> id >= vocab_size must be rejected as InvalidInput before \
             it can be forced into decoding, got {result:?}"
    );

    // Sanity: the identical id resolves once vocab_size actually covers it.
    assert_eq!(
        resolve_reasoning_close_token(&tokenizer, Some(1), true, 8)
            .expect("in-range marker must resolve"),
        Some(7)
    );
}

/// Two ids, one spelling: the resolver must refuse rather than pick one.
///
/// `DecodePolicy` recognises a single close-marker id. If `</think>` exists
/// as a base-vocabulary entry AND as a `special: false` added token, the
/// tiers give it two different ids, and resolving to the first means the
/// model can emit the other, render `</think>` into the output, and leave
/// `thinking_closed` false -- the budget then keeps counting reasoning
/// tokens and eventually forces a second marker. That is a silently
/// unenforced budget, which is the failure this resolver exists to refuse,
/// so ambiguity has to fail the same way absence does.
///
/// Driven through the real public loader rather than a hand-built struct,
/// and both ids sit well inside `vocab_size` so the range check cannot be
/// what fires. The control at the end is the discriminator: drop the added
/// token and the identical vocabulary resolves cleanly, so the rejection is
/// attributable to the collision and not to anything else in the fixture.
#[test]
fn reasoning_budget_rejects_close_marker_spelled_by_more_than_one_id() {
    let ambiguous = BpeTokenizer::from_tokenizer_json_str(
        r#"{
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "</think>": 7},
                    "merges": []
                },
                "added_tokens": [
                    {"id": 100, "content": "</think>", "special": false}
                ]
            }"#,
    )
    .expect("colliding-spelling tokenizer must load");

    // Assert the SET, not the order. Which tier the real loader files an
    // added token under is its business and not this test's subject, and
    // measured here it files this one AHEAD of the base-vocabulary entry --
    // so through the real loader the added alias wins precedence and id 7 is
    // the one discarded, the opposite direction from the hand-built fixture
    // in `bpe.rs`. Both directions are the same defect, which is the reason
    // this resolver refuses ambiguity outright rather than trying to pick
    // the right one.
    let mut ids = lattice_tokenizer::__internal::token_ids_for_content(&ambiguous, "</think>");
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![7, 100],
        "fixture precondition: the loader must actually produce two ids"
    );

    let result = resolve_reasoning_close_token(&ambiguous, Some(1), true, 200);
    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("</think>")
                    && message.contains("more than one")
                    && message.contains("100")
        ),
        "a </think> spelled by two ids must be rejected as InvalidInput naming both, \
             got {result:?}"
    );

    // Control: the same base vocabulary with no colliding added token
    // resolves. Without this, a resolver that rejected every tokenizer would
    // pass the assertion above.
    let unambiguous = BpeTokenizer::from_tokenizer_json_str(
        r#"{
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "</think>": 7},
                    "merges": []
                },
                "added_tokens": []
            }"#,
    )
    .expect("single-spelling tokenizer must load");
    assert_eq!(
        resolve_reasoning_close_token(&unambiguous, Some(1), true, 200)
            .expect("an unambiguous marker must resolve"),
        Some(7)
    );
}

/// The collision entirely INSIDE the added-token table, with no base
/// vocabulary entry involved.
///
/// This is a separate test rather than another case in the one above because
/// it was invisible for a different reason, at a different layer. The parse
/// step used to key renderable added tokens by CONTENT, so a second
/// `</think>` entry overwrote the first before any lookup ran: the ambiguity
/// check could not have caught it no matter how it was written, because only
/// one id survived loading. Keying that map by id is what makes the two
/// entries reach the resolver at all, and that is the thing this test pins.
#[test]
fn reasoning_budget_rejects_duplicate_added_token_aliases_of_the_close_marker() {
    let tokenizer = BpeTokenizer::from_tokenizer_json_str(
        r#"{
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1},
                    "merges": []
                },
                "added_tokens": [
                    {"id": 100, "content": "</think>", "special": false},
                    {"id": 101, "content": "</think>", "special": false}
                ]
            }"#,
    )
    .expect("duplicate-alias tokenizer must load");

    let mut ids = lattice_tokenizer::__internal::token_ids_for_content(&tokenizer, "</think>");
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![100, 101],
        "both added-token aliases must survive loading; if only one does, the \
             collision was discarded at parse time and no resolver check can see it"
    );

    let result = resolve_reasoning_close_token(&tokenizer, Some(1), true, 200);
    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("more than one") && message.contains("101")
        ),
        "duplicate added-token aliases must be rejected like any other ambiguity, \
             got {result:?}"
    );

    // Control: one alias, same shape, resolves. This is what distinguishes
    // "the resolver rejects ambiguity" from "the resolver rejects added
    // tokens", which would pass the assertion above just as well.
    let single = BpeTokenizer::from_tokenizer_json_str(
        r#"{
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1},
                    "merges": []
                },
                "added_tokens": [
                    {"id": 100, "content": "</think>", "special": false}
                ]
            }"#,
    )
    .expect("single-alias tokenizer must load");
    assert_eq!(
        resolve_reasoning_close_token(&single, Some(1), true, 200)
            .expect("a single added-token alias must resolve"),
        Some(100)
    );
}

/// Early-success bypass, CPU `generate`: an active reasoning budget with
/// no resolvable `</think>` marker must fail closed even when the very
/// first sampled token is itself a stop token -- `should_stop_token`'s
/// early `Ok` return must not be reached before budget validation runs.
///
/// Mechanism: `build_tiny_zero_model` always greedy-samples token 0 (see
/// `eos_token_id_override_forces_continuation_past_matching_stop_token`);
/// pointing `eos_token_id` at 0 makes `should_stop_token` true on that
/// very first sampled token, so this exercises the first-token-EOS
/// early-return path specifically, not the decode loop.
///
/// `think_close_id` must resolve before the `should_stop_token` check
/// runs, so this returns `Err`, not `Ok(GenerateOutput { stopped: true,
/// stop_reason: Eos, .. })`.
#[test]
fn reasoning_budget_bypass_blocked_on_first_token_eos() {
    let mut model = build_tiny_zero_model();
    model.set_eos_token_id(0);
    let result = model.generate(
        "a",
        &GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            enable_thinking: true,
            reasoning_budget: Some(1),
            stop_token_ids: vec![],
            ..Default::default()
        },
    );

    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("</think>") && message.contains("reasoning_budget")
        ),
        "an active reasoning budget with no resolvable </think> marker must fail \
             closed even when the very first sampled token is a stop token (the \
             first-token-EOS early-success path must not bypass budget validation); \
             got {result:?}"
    );
}

/// Sibling of the test above for the grammar early-return path -- an
/// active reasoning budget with no resolvable `</think>` marker must fail
/// closed even when the grammar's `advance()` rejects the first sampled
/// token (the `StopReason::Grammar` early `Ok` return in
/// `stop_reason_grammar_on_advance_false`).
///
/// `think_close_id` must resolve before the grammar early-return, so
/// this returns `Err`, not `Ok(GenerateOutput { stop_reason:
/// Some(StopReason::Grammar), .. })`.
#[test]
fn reasoning_budget_bypass_blocked_on_grammar_advance_false() {
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let model = build_tiny_zero_model();

    // Same vocab-size-1 setup as stop_reason_grammar_on_advance_false:
    // mask_logits blocks token 0, greedy picks token 1, advance(1)
    // returns false (1 >= grammar vocab_size 1).
    let spec = GrammarSpec::Gbnf("root ::= \"x\"\n".to_string());
    let vocab = vec![b"t".to_vec()];
    let engine = Arc::new(GrammarEngine::new(&spec, vocab).expect("single-token grammar compiles"));

    let result = model.generate(
        "a",
        &GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            enable_thinking: true,
            reasoning_budget: Some(1),
            grammar: Some(engine),
            stop_token_ids: vec![],
            ..Default::default()
        },
    );

    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("</think>") && message.contains("reasoning_budget")
        ),
        "an active reasoning budget with no resolvable </think> marker must fail \
             closed even when the grammar's advance() rejects the first sampled token \
             (the grammar early-success path must not bypass budget validation); \
             got {result:?}"
    );
}

/// Streaming sibling of `reasoning_budget_bypass_blocked_on_first_token_eos`:
/// the same bypass, exercised through `generate_streaming_with_observer`
/// rather than `generate`, since the two functions duplicate this control
/// flow independently. This test isolates the streaming copy: it must
/// return `Err` on its own ordering, not by relying on `generate`'s
/// ordering to cover it.
#[test]
fn reasoning_budget_bypass_blocked_on_first_token_eos_streaming() {
    let mut model = build_tiny_zero_model();
    model.set_eos_token_id(0);
    let result = model.generate_streaming_with_observer(
        "a",
        &GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            enable_thinking: true,
            reasoning_budget: Some(1),
            stop_token_ids: vec![],
            ..Default::default()
        },
        |_delta| true,
        || false,
        |_evt| {},
    );

    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("</think>") && message.contains("reasoning_budget")
        ),
        "generate_streaming_with_observer must also fail closed on first-token-EOS \
             with an active, unenforceable reasoning budget; got {result:?}"
    );
}

/// Streaming sibling for the cancellation early-return, which only the
/// streaming path has: a request cancelled before prefill must still be
/// rejected when its reasoning budget is unenforceable, rather than
/// returning `Ok(StopReason::Interrupt)` with the budget never checked.
#[test]
fn reasoning_budget_bypass_blocked_on_pre_prefill_cancel_streaming() {
    let model = build_tiny_zero_model();
    let result = model.generate_streaming_with_observer(
        "a",
        &GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            enable_thinking: true,
            reasoning_budget: Some(1),
            stop_token_ids: vec![],
            ..Default::default()
        },
        |_delta| true,
        || true,
        |_evt| {},
    );

    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("</think>") && message.contains("reasoning_budget")
        ),
        "an unenforceable reasoning budget must be rejected before the pre-prefill \
             cancellation early return; got {result:?}"
    );
}

/// Streaming observer contract: a generation that fails because it cannot
/// resolve the reasoning-close marker must not have already emitted a
/// `RawToken` event for the prefill-derived first token. The observer
/// contract promises `RawToken` events only for tokens that end up in the
/// returned `Ok(GenerateOutput)`; a caller counting `RawToken` events as
/// committed generated tokens must never see one for a generation that
/// returns `Err`.
#[test]
fn raw_observer_emits_no_raw_token_when_close_marker_resolution_fails() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        enable_thinking: true,
        reasoning_budget: Some(1),
        ..Default::default()
    };
    let events = std::cell::RefCell::new(Vec::<RawGenEvent>::new());
    let result = model.generate_streaming_with_observer(
        "a",
        &gen_cfg,
        |_delta| true,
        || false,
        |evt| events.borrow_mut().push(evt),
    );

    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "generation must fail closed when the tokenizer has no </think> marker, got {result:?}"
    );
    let events = events.into_inner();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RawGenEvent::RawToken { .. })),
        "no RawToken may be emitted on a path that then returns Err: {events:?}"
    );
}

/// COMBINED grammar × reasoning-budget path: when the s1 budget forces `</think>`
/// but the active grammar forbids that token, decoding must **fail closed** — stop
/// with `StopReason::Grammar` and NOT emit the forbidden `</think>`.
///
/// This pins the load-bearing weave in `DecodePolicy::transition_inner` (driven,
/// in production, via `decoder::driver::run`): grammar `advance` runs on the
/// budget-FORCED token (`next_id`), not the pre-force `sampled_id`. Setup: grammar
/// `root ::= "a"+` with a 7-entry grammar vocab (ids 0..=6); the tokenizer carries
/// `</think>` at id 7 (outside the grammar vocab). All-zero weights → greedy always
/// picks token 0's argmax after masking. Post-prefill emits one `'a'` (id 1), and the
/// repeating grammar accepts one at every decode step. The fallback arrives at
/// `generated_len == budget + 32`, so `force_close_think` overrides the sampled `'a'`
/// with `</think>` (id 7). `advance(7)`: `7 >= grammar.vocab_size (7)` → `false` →
/// `StopReason::Grammar`, before `</think>` is pushed.
///
/// Mutation sensitivity: if `advance` were called on `sampled_id` (1, grammar-legal)
/// instead of the forced `next_id` (7), `advance(1)` would succeed, `</think>` would
/// be emitted, and `token_ids` would contain a final 7, failing the assertion below.
#[test]
fn grammar_budget_forced_close_fails_closed() {
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let model = build_tiny_thinking_model();
    let close_id = model
        .tokenizer
        .special_token_id("</think>")
        .expect("thinking model tokenizer resolves </think>");
    assert!(
        close_id >= 7,
        "test assumes </think> id ({close_id}) is outside the 7-token grammar vocab"
    );

    // root ::= "a"+: grammar vocab ids 0..=6 (size 7), never terminal, so decoding
    // reaches the budget fallback. </think> (id 7) is out of the grammar vocab, so
    // advance(7) fail-closes.
    let spec = GrammarSpec::Gbnf("root ::= \"a\"+\n".to_string());
    let vocab: Vec<Vec<u8>> = vec![
        b"<unk>".to_vec(),
        b"a".to_vec(),
        b"b".to_vec(),
        b"c".to_vec(),
        b"d".to_vec(),
        b"e".to_vec(),
        b" ".to_vec(),
    ];
    let engine = Arc::new(GrammarEngine::new(&spec, vocab).expect("aa grammar compiles"));

    let gen_cfg = GenerateConfig {
        max_new_tokens: 40,
        temperature: 0.0,
        enable_thinking: true,
        reasoning_budget: Some(1),
        grammar: Some(engine),
        stop_token_ids: vec![],
        ..Default::default()
    };
    let result = model
        .generate("a", &gen_cfg)
        .expect("combined grammar+budget generate must succeed");

    assert_eq!(
        result.stop_reason,
        Some(StopReason::Grammar),
        "budget-forced </think> forbidden by grammar must stop with Grammar; got {:?}",
        result.stop_reason
    );
    assert_eq!(
        result.token_ids,
        vec![1; 33],
        "fail-closed: the budget-forced </think> must NOT be emitted; only the \
             sampled 'a' tokens (id 1) survive. A final 7 means advance ran on the \
             sampled token, not the forced token"
    );
    assert_eq!(result.generated_tokens, 33);
}

// -----------------------------------------------------------------------
// DecodePolicy::transition token-finalization tests
// -----------------------------------------------------------------------
//
// The zero-weight test fixture's greedy sampler always selects vocab
// index 0 (strict `v > best_val` tie-break), which makes it impossible to
// drive `Qwen35Model::generate` through more than one distinguishable
// decode position. The tests below construct `DecodePolicy` directly
// instead, mirroring the exact call shape every decode loop above uses
// (`transition`'s closures wired the same way `decoder::driver::run` wires
// `transition_with_metadata`'s), to isolate the finalization ordering
// itself from sampling.

/// A budget-forced `</think>` that clears the grammar-advance and EOS
/// checks must be emitted and have its own logprob recorded under the
/// forced id, not the sample it replaced.
///
/// Mutation sensitivity: recording the logprob under `sampled_id` instead
/// of the post-override `next_id` makes `token_logprobs[1].token_id`
/// read back as the stale sample (3) instead of the forced close id (7).
#[test]
fn forced_close_with_logprobs_records_the_post_override_token() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(1),
        enable_thinking: true,
        logprobs: Some(0),
        ..GenerateConfig::default()
    };
    let close_id = 7u32;
    let stale_sample = 3u32;
    let logits = [0.0f32; 10];

    let mut token_logprobs = Vec::new();
    let mut policy = DecodePolicy::init(
        &gen_cfg,
        Some(close_id),
        Some(boundary_at(1)),
        &mut token_logprobs,
        /* first_emitted_id */ 1,
        &logits,
        0.0,
        /* first_generated_len */ 1,
        false,
    );

    let mut text = String::new();
    let mut offsets = Vec::new();
    let outcome = policy
        .transition(
            &mut token_logprobs,
            stale_sample,
            &logits,
            0.0,
            /* generated_len_before */ 1,
            |_next_id| Ok(true),
            |_next_id| false,
            |_next_id| {},
            |_next_id| String::new(),
            &mut text,
            &mut offsets,
            |_delta, _next_id| true,
        )
        .expect("these callbacks cannot fail");

    match outcome {
        StepOutcome::Emitted { token_id, .. } => assert_eq!(
            token_id, close_id,
            "an accepted forced close must emit the close id, not the sample it replaced"
        ),
        _ => panic!("a forced close that clears grammar-advance and EOS must be Emitted"),
    }
    assert_eq!(
        token_logprobs.len(),
        2,
        "one entry from init's first token, one from this transition"
    );
    assert_eq!(
        token_logprobs[1].token_id, close_id,
        "the recorded logprob entry must describe the actually-emitted forced token, \
             not the stale pre-override sample"
    );
}

/// A budget-forced close must be checked for EOS against the token it
/// actually emits, never against the raw sample it replaced. Without this
/// ordering, a raw sample that happens to be an EOS/stop id could end the
/// generation one step early even though the override already replaced
/// it with a different, legal token.
///
/// Mutation sensitivity: checking `is_eos(sampled_id)` instead of
/// `is_eos(next_id)` makes the closure below see the stale sample (3),
/// which it treats as EOS, and the outcome becomes `Eos` instead of
/// `Emitted`.
#[test]
fn transition_refuses_eos_on_the_stale_pre_override_sample() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(1),
        enable_thinking: true,
        ..GenerateConfig::default()
    };
    let close_id = 7u32;
    let stale_sample = 3u32;

    let mut token_logprobs = Vec::new();
    let mut policy = DecodePolicy::init(
        &gen_cfg,
        Some(close_id),
        Some(boundary_at(1)),
        &mut token_logprobs,
        /* first_emitted_id */ 1,
        &[],
        0.0,
        /* first_generated_len */ 1,
        false,
    );

    let mut text = String::new();
    let mut offsets = Vec::new();
    let outcome = policy
        .transition(
            &mut token_logprobs,
            stale_sample,
            &[],
            0.0,
            /* generated_len_before */ 1,
            |_next_id| Ok(true),
            // Only the stale, pre-override sample would trigger EOS here.
            |next_id| next_id == stale_sample,
            |_next_id| {},
            |_next_id| String::new(),
            &mut text,
            &mut offsets,
            |_delta, _next_id| true,
        )
        .expect("these callbacks cannot fail");

    match outcome {
        StepOutcome::Emitted { token_id, .. } => assert_eq!(
            token_id, close_id,
            "the emitted token must be the forced close id, not the stale sample"
        ),
        StepOutcome::Eos => panic!(
            "EOS must not fire on the stale pre-override sample ({stale_sample}); \
                 the forced close id ({close_id}) is not itself EOS-eligible here"
        ),
        _ => panic!("unexpected outcome for an accepted forced close"),
    }
}

/// Once the reasoning block has already closed, a later step whose raw
/// budget condition (`generated_so_far >= budget`) is still trivially
/// true must not force a second close — the close id is consumed at most
/// once per generation.
///
/// Mutation sensitivity: dropping the already-closed guard from the
/// forced-close condition makes it force `close_id` again here; the
/// emitted token becomes 7 instead of the untouched sample, and this
/// test fails.
#[test]
fn transition_refuses_to_force_close_a_second_time_once_already_closed() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(1),
        enable_thinking: true,
        ..GenerateConfig::default()
    };
    let close_id = 7u32;
    let sampled_id = 3u32;

    let mut token_logprobs = Vec::new();
    // The prefill-derived first token IS the close marker, so `init`
    // seeds `thinking_closed = true` immediately.
    let mut policy = DecodePolicy::init(
        &gen_cfg,
        Some(close_id),
        None,
        &mut token_logprobs,
        /* first_emitted_id */ close_id,
        &[],
        0.0,
        /* first_generated_len */ 1,
        false,
    );

    let mut text = String::new();
    let mut offsets = Vec::new();
    // generated_len_before is far past the budget (1), so the raw
    // budget condition is still trivially satisfied on its own.
    let outcome = policy
        .transition(
            &mut token_logprobs,
            sampled_id,
            &[],
            0.0,
            /* generated_len_before */ 5,
            |_next_id| Ok(true),
            |_next_id| false,
            |_next_id| {},
            |_next_id| String::new(),
            &mut text,
            &mut offsets,
            |_delta, _next_id| true,
        )
        .expect("these callbacks cannot fail");

    match outcome {
        StepOutcome::Emitted { token_id, .. } => assert_eq!(
            token_id, sampled_id,
            "an already-closed block must not force the close id again; \
                 the untouched sample must pass through"
        ),
        _ => panic!("unexpected outcome for an untouched, already-closed step"),
    }
}

/// The prefill-derived first token can itself be the close marker (a
/// prompt that pushes the model straight into it). `DecodePolicy::init`
/// must still record that token's own logprob when requested — the same
/// contract `init_records_the_prefill_tokens_logprob_before_any_transition_call`
/// pins for the ordinary case, isolated here for the close-marker-as-
/// first-token edge specifically.
#[test]
fn init_records_the_first_tokens_logprob_when_it_is_already_the_close_marker() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(1),
        enable_thinking: true,
        logprobs: Some(0),
        ..GenerateConfig::default()
    };
    let close_id = 7u32;
    let logits = [0.0f32; 10];

    let mut token_logprobs = Vec::new();
    let _policy = DecodePolicy::init(
        &gen_cfg,
        Some(close_id),
        None,
        &mut token_logprobs,
        /* first_emitted_id */ close_id,
        &logits,
        0.0,
        /* first_generated_len */ 1,
        false,
    );

    assert_eq!(
        token_logprobs.len(),
        1,
        "init must record exactly one logprob entry for the first token"
    );
    assert_eq!(
        token_logprobs[0].token_id, close_id,
        "the recorded entry must describe the actual first token (the close marker)"
    );
}

/// When the reasoning block is already closed as of the first token, the
/// answer-budget window opens immediately from that token's position.
/// `max_new_tokens: 1` then means exactly one more (answer) token closes
/// the window — `answer_budget_exhausted` must report `true` on that
/// step, so the decode loop breaks instead of attempting a next decode.
///
/// Mutation sensitivity: seeding the reasoning-end position from anything
/// other than the first token's own generated length (e.g. leaving it
/// unset when the first token already closed the block) makes
/// `answer_budget_exhausted` stay `false` here, and the assertion fails.
#[test]
fn answer_budget_exhausts_on_the_terminal_token_with_no_next_decode_expected() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(1),
        enable_thinking: true,
        max_new_tokens: 1,
        ..GenerateConfig::default()
    };
    let close_id = 7u32;
    let sampled_id = 3u32;

    let mut token_logprobs = Vec::new();
    let mut policy = DecodePolicy::init(
        &gen_cfg,
        Some(close_id),
        None,
        &mut token_logprobs,
        /* first_emitted_id */ close_id,
        &[],
        0.0,
        /* first_generated_len */ 1,
        false,
    );

    let mut text = String::new();
    let mut offsets = Vec::new();
    let outcome = policy
        .transition(
            &mut token_logprobs,
            sampled_id,
            &[],
            0.0,
            /* generated_len_before */ 1,
            |_next_id| Ok(true),
            |_next_id| false,
            |_next_id| {},
            |_next_id| String::new(),
            &mut text,
            &mut offsets,
            |_delta, _next_id| true,
        )
        .expect("these callbacks cannot fail");

    match outcome {
        StepOutcome::Emitted {
            answer_budget_exhausted,
            ..
        } => assert!(
            answer_budget_exhausted,
            "one answer token after an already-closed block must exhaust a \
                 max_new_tokens: 1 answer budget on this same step"
        ),
        _ => panic!("unexpected outcome for the terminal answer token"),
    }
}

/// The ordinary case: a reasoning budget nowhere near exhaustion must
/// leave every step's outcome and logprob untouched by the override
/// machinery — the positive baseline the refusal tests above are
/// contrasted against.
#[test]
fn transition_leaves_an_unspent_reasoning_budget_inert() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(1_000),
        enable_thinking: true,
        logprobs: Some(0),
        ..GenerateConfig::default()
    };
    let close_id = 7u32;
    let sampled_id = 3u32;
    let logits = [0.0f32; 10];

    let mut token_logprobs = Vec::new();
    let mut policy = DecodePolicy::init(
        &gen_cfg,
        Some(close_id),
        None,
        &mut token_logprobs,
        /* first_emitted_id */ 1,
        &logits,
        0.0,
        /* first_generated_len */ 1,
        false,
    );

    let mut text = String::new();
    let mut offsets = Vec::new();
    let outcome = policy
        .transition(
            &mut token_logprobs,
            sampled_id,
            &logits,
            0.0,
            /* generated_len_before */ 1,
            |_next_id| Ok(true),
            |_next_id| false,
            |_next_id| {},
            |_next_id| String::new(),
            &mut text,
            &mut offsets,
            |_delta, _next_id| true,
        )
        .expect("these callbacks cannot fail");

    match outcome {
        StepOutcome::Emitted { token_id, .. } => assert_eq!(
            token_id, sampled_id,
            "far from budget exhaustion, the sampled token must pass through unmodified"
        ),
        _ => panic!("unexpected outcome for an ordinary, budget-unspent step"),
    }
    assert_eq!(
        token_logprobs.len(),
        2,
        "one entry from init's first token, one from this transition"
    );
    assert_eq!(token_logprobs[1].token_id, sampled_id);
}

mod streaming;
