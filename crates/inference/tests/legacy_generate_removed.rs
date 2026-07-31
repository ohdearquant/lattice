use std::fs;
use std::path::{Path, PathBuf};

const TEXT_EXTENSIONS: &[&str] = &["md", "py", "rs", "sh", "toml", "yaml", "yml"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("inference crate is nested under workspace/crates")
        .to_path_buf()
}

fn collect_text_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).expect("read workspace directory") {
        let entry = entry.expect("read workspace entry");
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|name| name.to_str());
            if !matches!(name, Some(".git" | "target")) {
                collect_text_files(&path, files);
            }
        } else if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| TEXT_EXTENSIONS.contains(&extension))
        {
            files.push(path);
        }
    }
}

#[test]
fn legacy_decode_artifacts_are_absent() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let retired_module = ["src/", "generate", ".rs"].concat();
    assert!(!manifest_dir.join(retired_module).exists());
    let retired_bench = ["compute", "_attention", "_bench.rs"].concat();
    assert!(!manifest_dir.join("benches").join(retired_bench).exists());

    let lib = fs::read_to_string(manifest_dir.join("src/lib.rs")).expect("read crate root");
    let module_declaration = ["pub mod ", "generate", ";"].concat();
    assert!(!lib.contains(&module_declaration));

    let manifest =
        fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("read inference manifest");
    let retired_target = ["compute", "_attention", "_bench"].concat();
    assert!(!manifest.contains(&retired_target));
}

#[test]
fn workspace_has_no_deleted_api_references() {
    let deleted_references = [
        ["crate", "::", "generate"].concat(),
        ["lattice_inference", "::", "generate"].concat(),
        ["generate", "::", "compute_attention"].concat(),
        ["src/", "generate", ".rs"].concat(),
        ["compute", "_attention", "_bench"].concat(),
    ];
    let mut files = Vec::new();
    collect_text_files(&workspace_root(), &mut files);

    let mut offenders = Vec::new();
    for path in files {
        let content = fs::read_to_string(&path).expect("read text workspace file");
        for deleted in &deleted_references {
            if content.contains(deleted) {
                offenders.push(format!("{} contains {deleted}", path.display()));
            }
        }
    }
    assert!(offenders.is_empty(), "{}", offenders.join("\n"));
}

#[test]
fn canonical_generation_contract_remains() {
    let config = lattice_inference::model::GenerateConfig::default();
    assert_eq!(config.max_new_tokens, 256);
    let _ = std::mem::size_of::<lattice_inference::model::qwen35_config::GenerateOutput>();
    let _ = std::mem::size_of::<lattice_inference::QwenModel>();
}
