fn enforce() -> bool {
    std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some()
        || std::env::var("LATTICE_POCR_GATE_ENFORCE").as_deref() == Ok("1")
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn config(head_dim: usize) -> crate::model::ernie45::Ernie45Config {
    use crate::model::ernie45::{Ernie45Config, Ernie45RopeScaling};
    Ernie45Config {
        hidden_size: 1024,
        intermediate_size: 3072,
        num_hidden_layers: 18,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        head_dim,
        vocab_size: 103424,
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        rope_scaling: Ernie45RopeScaling {
            mrope_section: vec![head_dim / 8, head_dim * 3 / 16, head_dim * 3 / 16],
        },
        tie_word_embeddings: false,
        use_bias: false,
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn metal_ernie45_rejects_unsupported_head_dim() {
    use super::state::validate_shape;
    use crate::InferenceError;
    validate_shape(&config(128), 17).expect("supported HD128 configuration");
    let unsupported = config(64);
    unsupported
        .validate()
        .expect("valid CPU HD64 configuration");
    match validate_shape(&unsupported, 17) {
        Err(InferenceError::InvalidInput(reason)) => assert!(
            reason.contains("head_dim") && reason.contains("128"),
            "wrong rejection: {reason}"
        ),
        other => panic!("expected the Metal head_dim=128 restriction, got {other:?}"),
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn metal_ernie45_rejects_empty_layers_and_invalid_capacity() {
    use super::state::validate_shape;
    use crate::InferenceError;
    use crate::model::ernie45::MAX_SEQ_LEN;
    let mut empty = config(128);
    empty.num_hidden_layers = 0;
    assert!(matches!(
        validate_shape(&empty, 17),
        Err(InferenceError::InvalidInput(reason)) if reason.contains("num_hidden_layers")
    ));
    for capacity in [0, MAX_SEQ_LEN + 1] {
        assert!(matches!(
            validate_shape(&config(128), capacity),
            Err(InferenceError::InvalidInput(reason)) if reason.contains("max_seq_len")
        ));
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu", feature = "f16"))]
mod cache_guards;

#[cfg(all(target_os = "macos", feature = "metal-gpu", feature = "f16"))]
mod real {
    use super::super::MetalErnie45State;
    use crate::InferenceError;
    use crate::measurement::gpu_test_lock;
    use crate::model::ernie45::{Ernie45Config, Ernie45Model, Ernie45Trace, Ernie45Weights};
    use crate::weights::SafetensorsFile;
    use serde::Deserialize;
    use std::path::PathBuf;

    const ATOL: f32 = 2e-3;
    const RTOL: f32 = 2e-3;
    const DISPATCHES: (u32, u32, u32, u32, u32, u32, u32) = (127, 37, 36, 18, 36, 1, 18);

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Golden {
        revision: String,
        dtype: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        id: String,
        ids: Vec<u32>,
        checkpoints: Vec<Checkpoint>,
        logits: LogitsGolden,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Checkpoint {
        name: String,
        last_tok_first8: Vec<f32>,
        mean_abs: f32,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LogitsGolden {
        argmax_per_pos: Vec<usize>,
        last_tok_first8: Vec<f32>,
        last_tok_top5: Vec<(usize, f32)>,
        last_tok_mean_abs: f32,
    }

    struct Oracle {
        name: String,
        ids: Vec<u32>,
        logits: Vec<f32>,
        golden: Option<LogitsGolden>,
    }

    #[derive(Debug)]
    struct Comparison {
        mismatches: usize,
        worst_absolute: f32,
        worst_normalized: f32,
    }

    fn bound(expected: f32) -> f32 {
        ATOL + RTOL * expected.abs()
    }

    fn assert_close(actual: f32, expected: f32, context: &str) {
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= bound(expected),
            "{context}: {actual} vs {expected}"
        );
    }

    fn compare(actual: &[f32], expected: &[f32]) -> Comparison {
        assert_eq!(actual.len(), expected.len(), "output length mismatch");
        assert!(!actual.is_empty(), "empty output is not parity evidence");
        let mut result = Comparison {
            mismatches: 0,
            worst_absolute: 0.0,
            worst_normalized: 0.0,
        };
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(a.is_finite() && e.is_finite(), "nonfinite output at {i}");
            let difference = (a - e).abs();
            result.mismatches += usize::from(difference > bound(e));
            result.worst_absolute = result.worst_absolute.max(difference);
            result.worst_normalized = result.worst_normalized.max(difference / bound(e));
        }
        result
    }

    fn argmax(row: &[f32]) -> usize {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("nonempty vocabulary")
    }

    fn assert_same_kv_rows(actual: (&[f32], &[f32]), expected: (&[f32], &[f32]), context: &str) {
        for (name, actual, expected) in [("K", actual.0, expected.0), ("V", actual.1, expected.1)] {
            assert_eq!(actual.len(), expected.len(), "{context}: {name} row shape");
            assert!(!actual.is_empty(), "{context}: empty {name} rows");
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{context}: {name}[{index}] {actual} vs {expected}"
                );
            }
        }
    }

    fn assert_cached_parity(actual: &[f32], expected: &[f32], context: &str) -> Comparison {
        let comparison = compare(actual, expected);
        assert_eq!(comparison.mismatches, 0, "{context}: {comparison:?}");
        assert_eq!(argmax(actual), argmax(expected), "{context}: argmax");
        comparison
    }

    fn cache_rows(cache: &super::super::MetalErnie45KvCache) -> Vec<(Vec<f32>, Vec<f32>)> {
        (0..cache.layers())
            .map(|layer| cache.layer_rows_for_test(layer))
            .collect()
    }

    fn assert_cache_rows(
        cache: &super::super::MetalErnie45KvCache,
        expected: &[(Vec<f32>, Vec<f32>)],
        context: &str,
    ) {
        assert_eq!(cache.layers(), expected.len(), "{context}: layer count");
        assert!(!expected.is_empty(), "{context}: no layer evidence");
        for (layer, expected) in expected.iter().enumerate() {
            let actual = cache.layer_rows_for_test(layer);
            assert_same_kv_rows(
                (&actual.0, &actual.1),
                (&expected.0, &expected.1),
                &format!("{context}, layer {layer}"),
            );
        }
    }

    fn assert_hf_logits(logits: &[f32], golden: &LogitsGolden, seq: usize, vocab: usize) {
        assert_eq!(logits.len(), seq * vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(golden.argmax_per_pos.len(), seq);
        for (row, &expected) in logits.chunks_exact(vocab).zip(&golden.argmax_per_pos) {
            assert_eq!(argmax(row), expected, "per-position HF argmax");
        }
        let last = &logits[(seq - 1) * vocab..];
        assert_eq!(golden.last_tok_first8.len(), 8);
        for (&actual, &expected) in last[..8].iter().zip(&golden.last_tok_first8) {
            assert_close(actual, expected, "HF last-token first-eight logits");
        }
        assert_eq!(golden.last_tok_top5.len(), 5);
        for &(id, expected) in &golden.last_tok_top5 {
            assert!(id < vocab);
            assert_close(last[id], expected, "HF recorded top-five logit value");
        }
        assert_close(
            last.iter().map(|v| v.abs()).sum::<f32>() / vocab as f32,
            golden.last_tok_mean_abs,
            "HF last-token mean absolute logit",
        );
    }

    fn assert_cpu_checkpoints(trace: &Ernie45Trace, case: &Case) {
        assert_eq!(case.checkpoints.len(), 20);
        assert_eq!(trace.layer_outputs.len(), 18);
        for (index, checkpoint) in case.checkpoints.iter().enumerate() {
            let (expected_name, values) = match index {
                0 => ("embed".to_owned(), &trace.embed),
                1..=18 => (
                    format!("layer_{}", index - 1),
                    &trace.layer_outputs[index - 1],
                ),
                _ => ("final_norm".to_owned(), &trace.final_norm),
            };
            assert_eq!(checkpoint.name, expected_name);
            assert_eq!(values.len(), case.ids.len() * 1024);
            assert_eq!(checkpoint.last_tok_first8.len(), 8);
            for (&actual, &expected) in values[(case.ids.len() - 1) * 1024..][..8]
                .iter()
                .zip(&checkpoint.last_tok_first8)
            {
                assert_close(actual, expected, &expected_name);
            }
            assert_close(
                values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32,
                checkpoint.mean_abs,
                &format!("{expected_name} mean absolute activation"),
            );
        }
    }

    fn model_dir() -> Option<PathBuf> {
        let dir = match std::env::var_os("LATTICE_POCR_MODEL_DIR") {
            Some(path) => PathBuf::from(path),
            None => {
                PathBuf::from(std::env::var_os("HOME")?).join(".lattice/models/paddleocr-vl-1.6")
            }
        };
        (dir.join("config.json").is_file() && dir.join("model.safetensors").is_file())
            .then_some(dir)
    }

    #[test]
    fn metal_ernie45_full_prefill_real_weights_match_cpu() {
        let _gpu_guard = gpu_test_lock();
        if metal::Device::system_default().is_none() {
            assert!(!super::enforce(), "Metal device required under enforcement");
            eprintln!("SKIP metal_ernie45: Metal device missing");
            return;
        }
        let Some(dir) = model_dir() else {
            assert!(
                !super::enforce(),
                "checkpoint required; set LATTICE_POCR_MODEL_DIR"
            );
            eprintln!("SKIP metal_ernie45: checkpoint missing");
            return;
        };
        let golden: Golden = serde_json::from_str(include_str!(
            "../../../tests/fixtures/paddleocr_vl/decoder/decoder_goldens.json"
        ))
        .expect("valid committed decoder fixture");
        assert_eq!(golden.revision, "c5630abae1d940eafe0697512a0325494b02ab42");
        assert_eq!(
            golden.dtype,
            "weights bf16 upcast to f32, eager attention, no cache"
        );
        let expected_cases = [
            ("ascii", 4),
            ("cjk_medical", 16),
            ("chat_prompt", 14),
            ("table_row", 11),
        ];
        assert_eq!(golden.cases.len(), expected_cases.len());
        for (case, &(name, seq)) in golden.cases.iter().zip(&expected_cases) {
            assert_eq!(case.id, name);
            assert_eq!(case.ids.len(), seq);
        }
        let table_ids = golden.cases[3].ids.clone();
        assert!(!table_ids.len().is_multiple_of(4));
        assert!(!table_ids.len().is_multiple_of(16));
        let crossing_ids: Vec<u32> = table_ids.iter().copied().cycle().take(17).collect();
        let cfg = Ernie45Config::from_config_json(&dir.join("config.json"))
            .expect("checkpoint configuration loads");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 18);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 103424);
        assert_eq!(cfg.rope_scaling.mrope_section, [16, 24, 24]);
        let vocab = cfg.vocab_size;
        let mut source =
            SafetensorsFile::open(&dir.join("model.safetensors")).expect("real checkpoint opens");
        let weights = Ernie45Weights::load(&mut source, &cfg).expect("real weights load");
        drop(source);
        let original_down = weights.layers[17].down_proj.clone();
        assert!(original_down.iter().all(|value| value.is_finite()));
        assert!(original_down.iter().any(|&value| value != 0.0));
        let negated_down: Vec<f32> = original_down.iter().map(|value| -*value).collect();
        let mut state = MetalErnie45State::new(&cfg, &weights, 17).expect("Metal model constructs");
        let model = Ernie45Model::new(cfg, weights).expect("CPU model validates");
        let mut oracles = Vec::new();
        for case in golden.cases {
            let trace = model
                .forward_trace(&case.ids)
                .expect("CPU reference forward");
            assert_cpu_checkpoints(&trace, &case);
            assert_hf_logits(&trace.logits, &case.logits, case.ids.len(), vocab);
            oracles.push(Oracle {
                name: case.id,
                ids: case.ids,
                logits: trace.logits,
                golden: Some(case.logits),
            });
        }
        let crossing_trace = model
            .forward_trace(&crossing_ids)
            .expect("CPU cross-tile forward");
        oracles.push(Oracle {
            name: "table_row_repeat_17".into(),
            ids: crossing_ids,
            logits: crossing_trace.logits,
            golden: None,
        });
        drop(model);

        for (ids, output_len) in [
            (Vec::new(), 0),
            (vec![vocab as u32], vocab),
            (vec![table_ids[0]; 18], 18 * vocab),
            (vec![table_ids[0]], vocab - 1),
        ] {
            let mut output = vec![1234567.0; output_len];
            assert!(matches!(
                state.prefill(&ids, &mut output),
                Err(InferenceError::InvalidInput(_))
            ));
            assert!(output.iter().all(|&value| value == 1234567.0));
            assert_eq!(state.last_dispatch_counts(), (0, 0, 0, 0, 0, 0, 0));
        }
        for oracle in &oracles {
            let mut logits = vec![f32::NAN; oracle.ids.len() * vocab];
            state
                .prefill(&oracle.ids, &mut logits)
                .expect("full Metal prefill");
            assert_eq!(state.last_dispatch_counts(), DISPATCHES);
            let comparison = compare(&logits, &oracle.logits);
            assert_eq!(comparison.mismatches, 0, "{}: {comparison:?}", oracle.name);
            for (actual, expected) in logits
                .chunks_exact(vocab)
                .zip(oracle.logits.chunks_exact(vocab))
            {
                assert_eq!(
                    argmax(actual),
                    argmax(expected),
                    "{}: CPU argmax",
                    oracle.name
                );
            }
            if let Some(golden) = &oracle.golden {
                assert_hf_logits(&logits, golden, oracle.ids.len(), vocab);
            }
            eprintln!(
                "[METAL_ERNIE45_FULL_PREFILL] executed=true layers=18 hd=128 seq_len={} case={} comparison={comparison:?}",
                oracle.ids.len(),
                oracle.name
            );
        }
        state
            .replace_last_down_projection_for_test(&negated_down)
            .expect("perturb layer 17");
        for oracle in &oracles {
            let mut logits = vec![f32::NAN; oracle.ids.len() * vocab];
            state
                .prefill(&oracle.ids, &mut logits)
                .expect("perturbed Metal prefill");
            assert_eq!(state.last_dispatch_counts(), DISPATCHES);
            let comparison = compare(&logits, &oracle.logits);
            assert!(
                comparison.mismatches > 0,
                "{}: mutation escaped {comparison:?}",
                oracle.name
            );
            eprintln!(
                "[METAL_ERNIE45_FULL_MUST_DIFFER] layer=17 seq_len={} case={} rejected=true comparison={comparison:?}",
                oracle.ids.len(),
                oracle.name
            );
        }
        state
            .replace_last_down_projection_for_test(&original_down)
            .expect("restore layer 17");
        assert!(state.last_down_projection_matches_for_test(&original_down));
        let table = &oracles[3];
        let mut restored = vec![f32::NAN; table.ids.len() * vocab];
        state
            .prefill(&table.ids, &mut restored)
            .expect("restored Metal prefill");
        assert_eq!(state.last_dispatch_counts(), DISPATCHES);
        let comparison = compare(&restored, &table.logits);
        assert_eq!(comparison.mismatches, 0, "restored parity: {comparison:?}");
        assert_hf_logits(
            &restored,
            table.golden.as_ref().expect("table golden"),
            table.ids.len(),
            vocab,
        );
        eprintln!(
            "[METAL_ERNIE45_FULL_RESTORE] seq_len=11 bit_exact_weights=true comparison={comparison:?}"
        );
        eprintln!("[METAL_ERNIE45_FULL_GATE] cases=5 controls=5 restored=true");
    }

    #[test]
    fn metal_ernie45_kv_one_layer_real_weights_match_cpu() {
        let _gpu_guard = gpu_test_lock();
        if metal::Device::system_default().is_none() {
            assert!(!super::enforce(), "Metal device required under enforcement");
            eprintln!("SKIP metal_ernie45 cached decode: Metal device missing");
            return;
        }
        let Some(dir) = model_dir() else {
            assert!(
                !super::enforce(),
                "checkpoint required; set LATTICE_POCR_MODEL_DIR"
            );
            eprintln!("SKIP metal_ernie45 cached decode: checkpoint missing");
            return;
        };
        const CAPACITY: usize = 12;
        const PROMPT_LEN: usize = 11;
        const LAYER_DISPATCHES: (u32, u32, u32, u32, u32, u32, u32) = (8, 3, 2, 1, 2, 0, 1);
        const SENTINEL: f32 = 1234567.0;
        let golden: Golden = serde_json::from_str(include_str!(
            "../../../tests/fixtures/paddleocr_vl/decoder/decoder_goldens.json"
        ))
        .expect("valid committed decoder fixture");
        assert_eq!(golden.revision, "c5630abae1d940eafe0697512a0325494b02ab42");
        let case = golden
            .cases
            .iter()
            .find(|case| case.id == "table_row")
            .expect("table-row fixture exists");
        assert_eq!(case.ids.len(), PROMPT_LEN);
        assert!(!PROMPT_LEN.is_multiple_of(4));
        assert!(!PROMPT_LEN.is_multiple_of(16));
        let mut cfg = Ernie45Config::from_config_json(&dir.join("config.json"))
            .expect("checkpoint configuration loads");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 18);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 103424);
        assert_eq!(cfg.rope_scaling.mrope_section, [16, 24, 24]);
        cfg.num_hidden_layers = 1;
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let mut source =
            SafetensorsFile::open(&dir.join("model.safetensors")).expect("real checkpoint opens");
        let weights =
            Ernie45Weights::load(&mut source, &cfg).expect("real layer-zero weights load");
        drop(source);
        assert_eq!(weights.layers.len(), 1);
        assert!(
            weights.layers[0]
                .down_proj
                .iter()
                .any(|&value| value != 0.0)
        );
        let mut embeds = Vec::with_capacity(CAPACITY * h);
        for &id in &case.ids {
            assert!((id as usize) < vocab);
            embeds.extend_from_slice(&weights.embed_tokens[id as usize * h..][..h]);
        }
        // Unequal gaps expose an ignored position input even when a uniform shift would cancel.
        let mut positions: Vec<[u32; 3]> = (0..PROMPT_LEN as u32)
            .map(|i| [7 + i * (i + 3) / 2; 3])
            .collect();
        let mut state =
            MetalErnie45State::new(&cfg, &weights, CAPACITY).expect("one-layer Metal constructs");
        let model = Ernie45Model::new(cfg, weights).expect("one-layer CPU model validates");
        let mut cache = state.new_kv_cache(CAPACITY).expect("Metal cache allocates");
        let mut cpu_cache = model.new_kv_cache(CAPACITY).expect("CPU cache allocates");
        assert_eq!(cache.capacity(), CAPACITY);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        let mut rejected = vec![SENTINEL; vocab];
        assert!(
            state
                .kv_decode_step(&embeds[..h], positions[0], &mut cache, &mut rejected)
                .is_err(),
            "decode must refuse an empty cache"
        );
        assert!(rejected.iter().all(|&value| value == SENTINEL));
        assert!(cache.is_empty());
        assert!(
            state
                .kv_prefill(
                    &embeds[..embeds.len() - 1],
                    &positions,
                    &mut cache,
                    &mut rejected
                )
                .is_err(),
            "prefill must reject a partial embedding row"
        );
        assert!(rejected.iter().all(|&value| value == SENTINEL));
        assert!(cache.is_empty());
        let mut short_output = vec![SENTINEL; vocab - 1];
        assert!(
            state
                .kv_prefill(&embeds, &positions, &mut cache, &mut short_output)
                .is_err(),
            "prefill must reject a short output slice"
        );
        assert!(short_output.iter().all(|&value| value == SENTINEL));
        assert!(cache.is_empty());

        let cpu_prefill = model
            .kv_prefill(&embeds, &positions, &mut cpu_cache)
            .expect("CPU cached prefill");
        let mut metal_prefill = vec![f32::NAN; vocab];
        state
            .kv_prefill(&embeds, &positions, &mut cache, &mut metal_prefill)
            .expect("Metal cached prefill");
        let prefill_dispatches = state.last_dispatch_counts();
        assert_eq!(prefill_dispatches, LAYER_DISPATCHES);
        assert_eq!(cache.len(), PROMPT_LEN);
        assert_eq!(cpu_cache.len(), PROMPT_LEN);
        let prefill_comparison = compare(&metal_prefill, &cpu_prefill);
        assert_eq!(prefill_comparison.mismatches, 0, "{prefill_comparison:?}");
        assert_eq!(argmax(&metal_prefill), argmax(&cpu_prefill));
        let prompt_rows = cache.layer_rows_for_test(0);
        assert_eq!(prompt_rows.0.len(), PROMPT_LEN * kv_dim);
        assert_eq!(prompt_rows.1.len(), PROMPT_LEN * kv_dim);

        let mut full_logits = vec![f32::NAN; vocab];
        state
            .prefill_embeds_for_test(&embeds, &positions, &mut full_logits)
            .expect("uncached Metal prompt forward");
        assert_eq!(state.last_dispatch_counts(), LAYER_DISPATCHES);
        let full_prompt_rows = state.last_layer_rows_for_test(PROMPT_LEN);
        assert_same_kv_rows(
            (&prompt_rows.0, &prompt_rows.1),
            (&full_prompt_rows.0, &full_prompt_rows.1),
            "cached prefill versus independent Metal prefill",
        );
        let prefill_self_comparison = compare(&metal_prefill, &full_logits);
        assert_eq!(
            prefill_self_comparison.mismatches, 0,
            "{prefill_self_comparison:?}"
        );
        assert_eq!(argmax(&metal_prefill), argmax(&full_logits));
        assert!(
            state
                .kv_prefill(&embeds, &positions, &mut cache, &mut rejected)
                .is_err(),
            "prefill must refuse a nonempty cache"
        );
        assert!(rejected.iter().all(|&value| value == SENTINEL));
        assert_eq!(cache.len(), PROMPT_LEN);
        let after_rejected_prefill = cache.layer_rows_for_test(0);
        assert_same_kv_rows(
            (&after_rejected_prefill.0, &after_rejected_prefill.1),
            (&prompt_rows.0, &prompt_rows.1),
            "rejected prefill preserves the cache",
        );

        let next_id = argmax(&cpu_prefill);
        let next_embed = &model.embed_tokens()[next_id * h..][..h];
        let next_index = PROMPT_LEN as u32;
        let next_position = [7 + next_index * (next_index + 3) / 2; 3];
        assert!(
            state
                .kv_decode_step(
                    &next_embed[..h - 1],
                    next_position,
                    &mut cache,
                    &mut rejected
                )
                .is_err(),
            "decode must reject a partial embedding row"
        );
        assert!(rejected.iter().all(|&value| value == SENTINEL));
        assert_eq!(cache.len(), PROMPT_LEN);
        let after_rejected_decode = cache.layer_rows_for_test(0);
        assert_same_kv_rows(
            (&after_rejected_decode.0, &after_rejected_decode.1),
            (&prompt_rows.0, &prompt_rows.1),
            "rejected decode preserves the cache",
        );

        let cpu_decode = model
            .kv_decode_step(next_embed, next_position, &mut cpu_cache)
            .expect("one CPU cached decode step");
        let mut metal_decode = vec![f32::NAN; vocab];
        state
            .kv_decode_step(next_embed, next_position, &mut cache, &mut metal_decode)
            .expect("one Metal cached decode step");
        let decode_dispatches = state.last_dispatch_counts();
        assert_eq!(decode_dispatches, LAYER_DISPATCHES);
        assert_eq!(cache.len(), CAPACITY);
        assert_eq!(cpu_cache.len(), CAPACITY);
        let decode_comparison = compare(&metal_decode, &cpu_decode);
        assert_eq!(decode_comparison.mismatches, 0, "{decode_comparison:?}");
        assert_eq!(argmax(&metal_decode), argmax(&cpu_decode));
        let decode_rows = cache.layer_rows_for_test(0);
        assert_eq!(decode_rows.0.len(), CAPACITY * kv_dim);
        assert_eq!(decode_rows.1.len(), CAPACITY * kv_dim);
        assert_same_kv_rows(
            (
                &decode_rows.0[..PROMPT_LEN * kv_dim],
                &decode_rows.1[..PROMPT_LEN * kv_dim],
            ),
            (&prompt_rows.0, &prompt_rows.1),
            "one decode step preserves all prompt rows",
        );

        embeds.extend_from_slice(next_embed);
        positions.push(next_position);
        state
            .prefill_embeds_for_test(&embeds, &positions, &mut full_logits)
            .expect("uncached Metal growing-prefix forward");
        assert_eq!(state.last_dispatch_counts(), LAYER_DISPATCHES);
        let full_decode_rows = state.last_layer_rows_for_test(CAPACITY);
        assert_same_kv_rows(
            (&decode_rows.0, &decode_rows.1),
            (&full_decode_rows.0, &full_decode_rows.1),
            "cached decode versus independent Metal prefill",
        );
        let decode_self_comparison = compare(&metal_decode, &full_logits);
        assert_eq!(
            decode_self_comparison.mismatches, 0,
            "{decode_self_comparison:?}"
        );
        assert_eq!(argmax(&metal_decode), argmax(&full_logits));

        assert!(
            state
                .kv_decode_step(next_embed, next_position, &mut cache, &mut rejected)
                .is_err(),
            "decode must refuse a cache at capacity"
        );
        assert!(rejected.iter().all(|&value| value == SENTINEL));
        assert_eq!(cache.len(), CAPACITY);
        let after_capacity_refusal = cache.layer_rows_for_test(0);
        assert_same_kv_rows(
            (&after_capacity_refusal.0, &after_capacity_refusal.1),
            (&decode_rows.0, &decode_rows.1),
            "capacity refusal preserves the cache",
        );
        eprintln!(
            "[METAL_ERNIE45_KV_ONE_LAYER] executed=true layers=1 prompt_len=11 decode_steps=1 cache_len=12 bit_exact_prefill_rows=true bit_exact_decode_rows=true prompt_rows_preserved=true prefill_dispatches={prefill_dispatches:?} decode_dispatches={decode_dispatches:?} prefill={prefill_comparison:?} decode={decode_comparison:?}"
        );
    }
    #[test]
    fn metal_ernie45_kv_full_model_reuses_every_layer_across_tiles() {
        let _gpu_guard = gpu_test_lock();
        if metal::Device::system_default().is_none() {
            assert!(!super::enforce(), "Metal device required under enforcement");
            eprintln!("SKIP metal_ernie45 full cached decode: Metal device missing");
            return;
        }
        let Some(dir) = model_dir() else {
            assert!(
                !super::enforce(),
                "checkpoint required; set LATTICE_POCR_MODEL_DIR"
            );
            eprintln!("SKIP metal_ernie45 full cached decode: checkpoint missing");
            return;
        };
        const PROMPT: usize = 11;
        const STEPS: usize = 8;
        const CAPACITY: usize = PROMPT + STEPS;
        const COUNTS: (u32, u32, u32, u32, u32, u32, u32) = (127, 37, 36, 18, 36, 0, 18);
        const SENTINEL: f32 = 1234567.0;
        let golden: Golden = serde_json::from_str(include_str!(
            "../../../tests/fixtures/paddleocr_vl/decoder/decoder_goldens.json"
        ))
        .expect("committed decoder fixture");
        assert_eq!(golden.revision, "c5630abae1d940eafe0697512a0325494b02ab42");
        let case = golden
            .cases
            .iter()
            .find(|case| case.id == "table_row")
            .expect("table-row fixture");
        assert_eq!(case.ids.len(), PROMPT);
        assert!(!PROMPT.is_multiple_of(4) && !CAPACITY.is_multiple_of(4));
        assert!(!PROMPT.is_multiple_of(16) && !CAPACITY.is_multiple_of(16));
        let cfg = Ernie45Config::from_config_json(&dir.join("config.json"))
            .expect("checkpoint configuration");
        assert_eq!(cfg.num_hidden_layers, 18);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 103424);
        assert_eq!(cfg.rope_scaling.mrope_section, [16, 24, 24]);
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let mut source =
            SafetensorsFile::open(&dir.join("model.safetensors")).expect("real checkpoint opens");
        let weights = Ernie45Weights::load(&mut source, &cfg).expect("all real layers load");
        drop(source);
        assert_eq!(weights.layers.len(), 18);
        let mut embeds = Vec::with_capacity(CAPACITY * h);
        for &id in &case.ids {
            assert!((id as usize) < vocab);
            embeds.extend_from_slice(&weights.embed_tokens[id as usize * h..][..h]);
        }
        let mut positions: Vec<[u32; 3]> = (0..PROMPT as u32)
            .map(|i| [7 + i * (i + 3) / 2; 3])
            .collect();
        let mut state = MetalErnie45State::new(&cfg, &weights, CAPACITY)
            .expect("full Metal decoder constructs");
        let model = Ernie45Model::new(cfg, weights).expect("full CPU decoder validates");
        let mut cache = state.new_kv_cache(CAPACITY).expect("full Metal cache");
        let mut cpu_cache = model.new_kv_cache(CAPACITY).expect("full CPU cache");
        let mut cpu_logits = model
            .kv_prefill(&embeds, &positions, &mut cpu_cache)
            .expect("full CPU cached prefill");
        let mut metal_logits = vec![f32::NAN; vocab];
        let mut full_logits = vec![f32::NAN; vocab];
        state
            .kv_prefill(&embeds, &positions, &mut cache, &mut metal_logits)
            .expect("full Metal cached prefill");
        assert_eq!(state.last_dispatch_counts(), COUNTS);
        assert_eq!(cache.len(), PROMPT);
        let prefill = assert_cached_parity(&metal_logits, &cpu_logits, "full cached prefill CPU");
        let trace = state
            .prefill_embeds_trace_for_test(&embeds, &positions, &mut full_logits)
            .expect("independent full Metal prefill trace");
        assert_eq!(state.last_dispatch_counts(), COUNTS);
        assert_cached_parity(&metal_logits, &full_logits, "full cached prefill Metal");
        assert_cache_rows(&cache, &trace, "prefill cache versus independent trace");
        eprintln!(
            "[METAL_ERNIE45_KV_FULL_STEP] step=0 cache_len=11 bit_exact_all_layers=true cpu={prefill:?}"
        );

        let mut crossed_tile = false;
        let mut corruption_checked = false;
        for step in 1..=STEPS {
            let previous_len = cache.len();
            let previous_rows = cache_rows(&cache);
            let next_id = argmax(&cpu_logits);
            let next_embed = &model.embed_tokens()[next_id * h..][..h];
            let i = previous_len as u32;
            let position = [7 + i * (i + 3) / 2; 3];
            let mut control = if previous_len == 16 {
                let mut other = state.new_kv_cache(CAPACITY).expect("control cache");
                let mut other_logits = vec![f32::NAN; vocab];
                state
                    .kv_prefill(&embeds, &positions, &mut other, &mut other_logits)
                    .expect("independent control prefill");
                assert_cached_parity(
                    &other_logits,
                    &metal_logits,
                    "control starts from clean prefix",
                );
                assert_cache_rows(
                    &other,
                    &previous_rows,
                    "control prefix matches all cached layers",
                );
                let rows = other.layer_rows_for_test(17);
                let offset = 15 * kv_dim;
                let saved = rows.1[offset..offset + kv_dim].to_vec();
                let changed: Vec<f32> = saved
                    .iter()
                    .enumerate()
                    .map(|(lane, &value)| value + if lane % 2 == 0 { 4096.0 } else { -4096.0 })
                    .collect();
                assert!(changed.iter().all(|value| value.is_finite()));
                other
                    .replace_value_row_for_test(17, 15, &changed)
                    .expect("perturb one live V row");
                let changed_rows = other.layer_rows_for_test(17);
                assert!(
                    changed_rows.1[offset..offset + kv_dim]
                        .iter()
                        .zip(&saved)
                        .any(|(a, b)| a.to_bits() != b.to_bits()),
                    "V-row control did not change any bits"
                );
                assert!(
                    rows.0
                        .iter()
                        .zip(&changed_rows.0)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "V perturbation must preserve K bits"
                );
                assert!(
                    rows.1[..offset]
                        .iter()
                        .zip(&changed_rows.1[..offset])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "earlier V row bits changed"
                );
                Some((other, saved))
            } else {
                None
            };

            cpu_logits = model
                .kv_decode_step(next_embed, position, &mut cpu_cache)
                .expect("full CPU cached step");
            state
                .kv_decode_step(next_embed, position, &mut cache, &mut metal_logits)
                .expect("full Metal cached step");
            assert_eq!(state.last_dispatch_counts(), COUNTS);
            assert_eq!(cache.len(), previous_len + 1);
            assert_eq!(cpu_cache.len(), cache.len());
            let cpu = assert_cached_parity(&metal_logits, &cpu_logits, "full cached decode CPU");
            for (layer, previous) in previous_rows.iter().enumerate() {
                let current = cache.layer_rows_for_test(layer);
                assert_same_kv_rows(
                    (
                        &current.0[..previous_len * kv_dim],
                        &current.1[..previous_len * kv_dim],
                    ),
                    (&previous.0, &previous.1),
                    &format!("step {step} layer {layer} preserves live prefix"),
                );
            }

            if let Some((other, saved)) = control.as_mut() {
                let mut changed_logits = vec![f32::NAN; vocab];
                state
                    .kv_decode_step(next_embed, position, other, &mut changed_logits)
                    .expect("finite corrupted-cache decode");
                let changed = compare(&changed_logits, &metal_logits);
                assert!(
                    changed.mismatches > 0,
                    "V-row corruption did not exceed the declared bound: {changed:?}"
                );
                let failure = std::panic::catch_unwind(|| {
                    assert_cached_parity(&changed_logits, &metal_logits, "corrupted cache parity");
                })
                .expect_err("the passing parity assertion must reject the corrupted cache");
                let reason = failure
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| failure.downcast_ref::<&str>().copied())
                    .unwrap_or("");
                assert!(
                    reason.contains("corrupted cache parity"),
                    "unexpected control panic: {reason}"
                );
                eprintln!(
                    "[METAL_ERNIE45_KV_MUST_DIFFER] layer=17 token=15 cache_len=16 mismatches={} worst_absolute={} worst_normalized={}",
                    changed.mismatches, changed.worst_absolute, changed.worst_normalized
                );
                other
                    .replace_value_row_for_test(17, 15, saved)
                    .expect("restore saved V row");
                let restored_row = other.layer_rows_for_test(17);
                assert!(
                    restored_row.1[15 * kv_dim..16 * kv_dim]
                        .iter()
                        .zip(saved.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "restored V row differs by bits"
                );
                other.clear();
                let mut restored = vec![f32::NAN; vocab];
                state
                    .kv_prefill(&embeds, &positions, other, &mut restored)
                    .expect("rebuild restored prefix");
                assert_cache_rows(other, &previous_rows, "restored prefix cache");
                state
                    .kv_decode_step(next_embed, position, other, &mut restored)
                    .expect("restored control step");
                assert_cached_parity(&restored, &metal_logits, "restored cache parity");
                assert_cache_rows(
                    other,
                    &cache_rows(&cache),
                    "restored control matches all clean layers",
                );
                corruption_checked = true;
            }

            embeds.extend_from_slice(next_embed);
            positions.push(position);
            let trace = state
                .prefill_embeds_trace_for_test(&embeds, &positions, &mut full_logits)
                .expect("independent growing-prefix Metal trace");
            assert_eq!(state.last_dispatch_counts(), COUNTS);
            let full = assert_cached_parity(
                &metal_logits,
                &full_logits,
                "cached versus uncached Metal decode",
            );
            assert_cache_rows(
                &cache,
                &trace,
                &format!("step {step} independent all-layer trace"),
            );
            crossed_tile |= previous_len == 16 && cache.len() == 17;
            eprintln!(
                "[METAL_ERNIE45_KV_FULL_STEP] step={step} cache_len={} bit_exact_all_layers=true prompt_rows_preserved=true dispatches={:?} cpu={cpu:?} metal={full:?}",
                cache.len(),
                state.last_dispatch_counts()
            );
        }
        assert!(crossed_tile && corruption_checked);
        assert_eq!(cache.len(), CAPACITY);
        let before_refusal = cache_rows(&cache);
        let mut rejected = vec![SENTINEL; vocab];
        let result = state.kv_decode_step(&embeds[..h], [999; 3], &mut cache, &mut rejected);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(reason)) if reason == "ernie45 Metal: kv decode cache is full")
        );
        assert!(rejected.iter().all(|&value| value == SENTINEL));
        assert_eq!(cache.len(), CAPACITY);
        assert_cache_rows(
            &cache,
            &before_refusal,
            "capacity refusal preserves every layer",
        );
        eprintln!(
            "[METAL_ERNIE45_KV_FULL_GATE] executed=true layers=18 prompt_len=11 decode_steps=8 cache_len=19 tile_crossed=true bit_exact_all_layers=true prompt_rows_preserved=true must_differ=true restored=true capacity_refused=true"
        );
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu", feature = "f16")))]
mod real {
    #[test]
    fn metal_ernie45_full_prefill_real_weights_match_cpu() {
        assert!(
            !super::enforce(),
            "Metal ERNIE parity requires macOS and metal-gpu,f16"
        );
        eprintln!("SKIP metal_ernie45: requires macOS and metal-gpu,f16 features");
    }

    #[test]
    fn metal_ernie45_kv_one_layer_real_weights_match_cpu() {
        assert!(
            !super::enforce(),
            "Metal ERNIE cached decode requires macOS and metal-gpu,f16"
        );
        eprintln!("SKIP metal_ernie45 cached decode: requires macOS and metal-gpu,f16 features");
    }
    #[test]
    fn metal_ernie45_kv_full_model_reuses_every_layer_across_tiles() {
        assert!(
            !super::enforce(),
            "Metal ERNIE full cached decode requires macOS and metal-gpu,f16"
        );
        eprintln!(
            "SKIP metal_ernie45 full cached decode: requires macOS and metal-gpu,f16 features"
        );
    }
}
