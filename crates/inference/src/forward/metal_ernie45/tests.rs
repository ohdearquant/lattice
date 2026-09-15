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
}
