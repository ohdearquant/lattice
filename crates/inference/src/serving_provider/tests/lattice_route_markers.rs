use super::*;
use std::fs;

const QWEN_CPU: &str = "[route] selected family=qwen35 backend=cpu format=safetensors";
const QWEN_METAL: &str = "[route] selected family=qwen35 backend=metal format=q4";
const GEMMA_CPU: &str = "[route] selected family=gemma4 backend=cpu format=safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expected {
    Marker(&'static str),
    GemmaQ4Refusal,
    UnloadableRefusal,
}

const CONFIGS: [(&str, Option<&[u8]>); 6] = [
    ("qwen", Some(br#"{"model_type":"qwen3_5"}"#)),
    ("qwen_moe", Some(br#"{"model_type":"qwen3_5_moe"}"#)),
    ("gemma", Some(br#"{"model_type":"gemma4"}"#)),
    ("other", Some(br#"{"model_type":"bert"}"#)),
    ("malformed", Some(b"{")),
    ("missing", None),
];

fn expected(format: FormatFixture, config: &str) -> Expected {
    match (format.expected(), config) {
        (ModelFormat::Safetensors, "gemma") => Expected::Marker(GEMMA_CPU),
        (ModelFormat::Safetensors, _) => Expected::Marker(QWEN_CPU),
        (ModelFormat::Q4, "gemma") => Expected::GemmaQ4Refusal,
        (ModelFormat::Q4, _) => Expected::Marker(QWEN_METAL),
        _ => Expected::UnloadableRefusal,
    }
}

fn gemma_q4_refusal(directory: &Path) -> String {
    format!(
        "gemma_metal_unsupported: '{}' is a Gemma 4 checkpoint in the native Q4 format, which \
         only the Metal backend reads, and Gemma 4 is served on the CPU backend only. Point \
         --model at the Gemma 4 safetensors directory instead.",
        directory.display()
    )
}

fn checkpoint(format: FormatFixture, config: Option<&[u8]>) -> tempfile::TempDir {
    let temp = tempfile::tempdir().expect("create checkpoint tempdir");
    format.write(temp.path());
    if let Some(config) = config {
        fs::write(temp.path().join("config.json"), config).expect("write model config");
    }
    temp
}

#[test]
fn lattice_entry_marker_and_refusal_text_are_unchanged_for_every_fixture() {
    let mut checked = 0;
    for format in FORMATS {
        for (name, config) in CONFIGS {
            let temp = checkpoint(format, config);
            let selected = select(inspect_checkpoint(temp.path()), ServingEntry::Lattice);
            let actual = match (&selected, expected(format, name)) {
                (Ok(selected), Expected::Marker(marker)) => {
                    assert_eq!(lattice_marker(selected), marker, "{format:?} {name}");
                    true
                }
                (Err(error), Expected::GemmaQ4Refusal) => {
                    assert_eq!(
                        error.message(),
                        gemma_q4_refusal(temp.path()),
                        "{format:?} {name}"
                    );
                    true
                }
                (Err(error), Expected::UnloadableRefusal) => {
                    assert_eq!(
                        error.message(),
                        crate::model_format::unloadable_format_message(temp.path()),
                        "{format:?} {name}"
                    );
                    true
                }
                (other, want) => panic!(
                    "{format:?} {name}: expected {want:?}, got {:?}",
                    other
                        .as_ref()
                        .map(lattice_marker)
                        .map_err(SelectionError::message)
                ),
            };
            assert!(actual);
            checked += 1;
        }
    }
    assert_eq!(checked, FORMATS.len() * CONFIGS.len());
}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[test]
fn q4_qwen_open_without_the_metal_build_reports_the_rebuild_message() {
    let temp = checkpoint(FormatFixture::Q4Only, Some(br#"{"model_type":"qwen3_5"}"#));
    let selected = select(inspect_checkpoint(temp.path()), ServingEntry::Lattice)
        .expect("Q4 Qwen is admitted on the lattice entry");
    let Err(error) = selected.open_lattice(
        LatticeOpenOptions {
            tokenizer_dir: None,
            max_pending: 1,
            max_resident_adapters: 1,
            max_resident_adapter_bytes: 1,
            preload_vision: false,
            context_window: 0,
        },
        Arc::new(Semaphore::new(1)),
    ) else {
        panic!("a Q4 checkpoint cannot open without the Metal build");
    };
    assert_eq!(
        error,
        crate::model_format::metal_gpu_required_message(temp.path())
    );
}

fn lattice_marker(selected: &SelectedProvider<'_>) -> String {
    selected
        .lattice_selection_marker()
        .unwrap_or_else(|| "<no lattice route plan>".to_owned())
}
