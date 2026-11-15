//! Explicitly selected exact-bit output for CPU embedding comparisons.
// The dump requires the CPU model path; a Metal build would construct GPU state.
#![cfg(not(all(target_os = "macos", feature = "metal-gpu")))]

use lattice_model_qwen3_embedding::{QwenEmbeddingBackend, QwenModel};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;

#[path = "support/embedding_checkpoint.rs"]
mod support;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn selected_widths(hidden: usize) -> Vec<Option<usize>> {
    let specification = std::env::var("LATTICE_QWEN_FINAL_ROW_WIDTHS")
        .expect("real-checkpoint output requires an explicit selected width list");
    let mut widths = Vec::new();
    for token in specification.split(',') {
        let width = match token.trim() {
            "none" => None,
            "hidden" => Some(hidden),
            "hidden+1" => Some(hidden.checked_add(1).unwrap()),
            "max" => Some(usize::MAX),
            value => Some(value.parse::<usize>().expect("invalid selected width")),
        };
        if !widths.contains(&width) {
            widths.push(width);
        }
    }
    assert!(!widths.is_empty());
    widths
}

#[test]
#[ignore = "requires a new LATTICE_QWEN_FINAL_ROW_DUMP path; optionally an explicit real checkpoint"]
fn dump_cpu_embedding_bits() {
    let destination = PathBuf::from(
        std::env::var_os("LATTICE_QWEN_FINAL_ROW_DUMP")
            .expect("LATTICE_QWEN_FINAL_ROW_DUMP is required"),
    );
    assert!(
        destination.is_absolute(),
        "dump destination must be absolute"
    );
    assert!(!destination.exists(), "dump destination must be new");
    let real = std::env::var_os("LATTICE_QWEN_FINAL_ROW_MODEL_DIR").map(PathBuf::from);
    let fixture = real.is_none().then(support::checkpoint);
    let directory = real
        .as_deref()
        .unwrap_or_else(|| fixture.as_ref().unwrap().path());
    assert!(directory.is_absolute(), "model directory must be absolute");
    let mut model = QwenModel::from_directory(directory).unwrap();
    assert!(
        !model.has_gpu(),
        "dump must execute the explicit CPU model path"
    );
    let hidden = model.config().hidden_size;
    let (coverage, widths) = if real.is_some() {
        assert_eq!(hidden, 1024, "real fixture must be Qwen3-Embedding-0.6B");
        assert_eq!(model.config().num_hidden_layers, 28);
        ("selected-real-widths", selected_widths(hidden))
    } else {
        support::assert_fixture(&model);
        assert!(
            std::env::var_os("LATTICE_QWEN_FINAL_ROW_WIDTHS").is_none(),
            "synthetic widths are always exhaustive"
        );
        let mut widths = vec![None];
        widths.extend((0..=hidden).map(Some));
        widths.extend([Some(hidden + 1), Some(usize::MAX)]);
        assert_eq!(hidden, support::HIDDEN);
        assert_eq!(widths.len(), 36);
        ("exhaustive-synthetic-widths", widths)
    };
    let lengths: Vec<usize> = support::TEXTS
        .iter()
        .map(|text| model.embedding_token_counts(text).embedded)
        .collect();
    assert!(lengths[0] > lengths[1] && lengths[2] > lengths[1]);
    if real.is_none() {
        assert_eq!(lengths, support::LENGTHS);
    }

    let mut cases = Vec::new();
    let mut row_count = 0usize;
    let mut value_count = 0usize;
    for &width in &widths {
        model.set_output_dim(width);
        let expected_width = width.unwrap_or(hidden).min(hidden);
        for route in ["serial", "identified-uncached", "batch-one", "batch-mixed"] {
            model.cache_clear();
            assert_eq!(model.cache_size(), 0);
            let (outputs, expected_cache) = match route {
                "serial" => {
                    let output = model.encode(support::TEXTS[0]).unwrap();
                    let cached = model.encode(support::TEXTS[0]).unwrap();
                    assert_eq!(bits(&output), bits(&cached));
                    (vec![output], 1)
                }
                "identified-uncached" => {
                    let (output, backend) =
                        model.encode_identified_uncached(support::TEXTS[0]).unwrap();
                    assert_eq!(backend, QwenEmbeddingBackend::Cpu);
                    (vec![output], 0)
                }
                "batch-one" => (model.encode_batch(&support::TEXTS[..1]).unwrap(), 1),
                "batch-mixed" => (model.encode_batch(&support::TEXTS).unwrap(), 3),
                _ => unreachable!(),
            };
            assert_eq!(model.cache_size(), expected_cache);
            assert_eq!(outputs.len(), if route == "batch-mixed" { 3 } else { 1 });
            for output in &outputs {
                assert_eq!(output.len(), expected_width, "{route}: requested={width:?}");
                assert!(output.iter().all(|value| value.is_finite()));
                if expected_width > 0 {
                    assert!(output.iter().any(|value| *value != 0.0));
                }
            }
            if width.is_none() && route == "batch-mixed" {
                assert!(
                    outputs[1..]
                        .iter()
                        .any(|output| bits(output) != bits(&outputs[0])),
                    "mixed full-width outputs must be nonconstant"
                );
            }
            row_count += outputs.len();
            value_count += outputs.iter().map(Vec::len).sum::<usize>();
            cases.push(json!({
                "requested_width": width,
                "effective_width": expected_width,
                "route": route,
                "token_lengths": if route == "batch-mixed" { &lengths[..] } else { &lengths[..1] },
                "rows": outputs.len(),
                "bits": outputs.iter().map(|output| bits(output)).collect::<Vec<_>>()
            }));
        }
    }
    assert_eq!(cases.len(), widths.len() * 4);
    assert_eq!(row_count, widths.len() * 6);
    if real.is_none() {
        assert_eq!(cases.len(), 144);
        assert_eq!(row_count, 216);
        assert_eq!(value_count, 3744);
    }
    assert!(value_count > 0);
    let artifact = json!({
        "schema": "qwen-cpu-final-row-bits-v1",
        "coverage": coverage,
        "hidden_size": hidden,
        "layers": model.config().num_hidden_layers,
        "heads": model.config().num_attention_heads,
        "kv_heads": model.config().num_key_value_heads,
        "requested_widths": widths,
        "case_count": cases.len(),
        "row_count": row_count,
        "value_count": value_count,
        "cases": cases
    });
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&destination)
        .unwrap();
    serde_json::to_writer_pretty(&mut file, &artifact).unwrap();
    file.write_all(b"\n").unwrap();
    file.sync_all().unwrap();
    println!(
        "wrote {} cases, {row_count} rows, {value_count} exact f32 bit patterns; coverage={coverage}",
        widths.len() * 4
    );
}
