use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const TEXT_EXTENSIONS: &[&str] = &["md", "py", "rs", "sh", "toml", "yaml", "yml"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("inference crate is nested under workspace/crates")
        .to_path_buf()
}

fn tracked_text_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["ls-files", "--cached", "--full-name", "-z"])
        .output()
        .map_err(|error| format!("discover tracked files in {}: {error}", root.display()))?;
    if !output.status.success() {
        return Err(format!(
            "git ls-files failed in {} ({}): {}",
            root.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let listing = std::str::from_utf8(&output.stdout)
        .map_err(|error| format!("tracked paths are not UTF-8: {error}"))?;
    let mut files: Vec<_> = listing
        .split_terminator('\0')
        .map(PathBuf::from)
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| TEXT_EXTENSIONS.contains(&extension))
        })
        .collect();
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(format!(
            "no tracked text files discovered in {}",
            root.display()
        ));
    }
    Ok(files)
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

fn deleted_references() -> [String; 5] {
    [
        ["crate", "::", "generate"].concat(),
        ["lattice_inference", "::", "generate"].concat(),
        ["generate", "::", "compute_attention"].concat(),
        ["src/", "generate", ".rs"].concat(),
        ["compute", "_attention", "_bench"].concat(),
    ]
}

fn find_deleted_references(root: &Path, files: &[PathBuf]) -> Result<Vec<String>, String> {
    if files.is_empty() {
        return Err("cannot scan an empty text population".to_owned());
    }
    let deleted_references = deleted_references();
    let mut offenders = Vec::new();
    for path in files {
        let content = fs::read_to_string(root.join(path))
            .map_err(|error| format!("read tracked file {}: {error}", path.display()))?;
        for deleted in &deleted_references {
            if content.contains(deleted) {
                offenders.push(format!("{} contains {deleted}", path.display()));
            }
        }
    }
    Ok(offenders)
}

#[test]
fn workspace_has_no_deleted_api_references() {
    let root = workspace_root();
    let files = tracked_text_files(&root).expect("discover tracked text population");
    let offenders = find_deleted_references(&root, &files).expect("scan tracked text population");
    println!(
        "tracked text files: {}; offending references: {}",
        files.len(),
        offenders.len()
    );
    assert!(offenders.is_empty(), "{}", offenders.join("\n"));
}

#[test]
fn canonical_generation_contract_remains() {
    let config = lattice_inference::GenerateConfig::default();
    assert_eq!(config.max_new_tokens, 256);
    let _ = std::mem::size_of::<lattice_inference::GenerateOutput>();
    let _ = std::mem::size_of::<lattice_inference::TokenLogprob>();
    let _ = std::mem::size_of::<lattice_inference::TopLogprob>();

    // ADR-092 keeps the pre-move paths resolving. Asserting that is the point of this
    // test, so the deprecation is allowed here deliberately rather than silenced by
    // rewriting these to the canonical path with the rest of the tree.
    #[allow(deprecated)]
    {
        let legacy_family = lattice_inference::model::qwen35_config::GenerateConfig::default();
        assert_eq!(legacy_family.max_new_tokens, config.max_new_tokens);
        let legacy_model = lattice_inference::model::GenerateConfig::default();
        assert_eq!(legacy_model.max_new_tokens, config.max_new_tokens);
        let _ = std::mem::size_of::<lattice_inference::model::qwen35_config::GenerateOutput>();
    }
    let _ = std::mem::size_of::<lattice_inference::QwenModel>();

    let _ = lattice_inference::model::Qwen35Model::generate;
    // `generate_streaming` takes `impl FnMut(&str)`, so it has no explicit generic
    // parameter to turbofish. Naming it inside a closure body type-checks the call
    // without an instance, and still fails the build if the method is renamed.
    let _streaming_entry_point =
        |model: &lattice_inference::model::Qwen35Model,
         prompt: &str,
         gen_cfg: &lattice_inference::GenerateConfig| {
            model.generate_streaming(prompt, gen_cfg, |_token: &str| {})
        };
}

fn fixture_git(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn tracked_scan_reports_reintroductions_and_excludes_local_artifacts() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path();
    fixture_git(root, &["init", "--quiet"]);
    fs::write(root.join(".gitignore"), "scratch/\n").unwrap();
    fs::create_dir(root.join("scratch")).unwrap();
    let deleted = deleted_references();
    let planted = deleted.join("\n");
    fs::write(root.join("scratch/ignored.md"), &planted).unwrap();
    fs::write(root.join("untracked.md"), &planted).unwrap();
    fs::write(root.join("tracked.bin"), &planted).unwrap();
    fs::write(root.join("clean.md"), "supported generation").unwrap();
    fixture_git(root, &["add", ".gitignore", "tracked.bin", "clean.md"]);

    let mut expected = Vec::new();
    for extension in TEXT_EXTENSIONS {
        let name = format!("reintroduced reference.{extension}");
        fs::write(root.join(&name), &planted).unwrap();
        fixture_git(root, &["add", "--", &name]);
        for reference in &deleted {
            expected.push(format!("{name} contains {reference}"));
        }
    }
    let files = tracked_text_files(root).unwrap();
    assert_eq!(files.len(), TEXT_EXTENSIONS.len() + 1);
    let mut offenders = find_deleted_references(root, &files).unwrap();
    offenders.sort();
    expected.sort();
    assert_eq!(offenders, expected);
    println!(
        "synthetic tracked text files: {}; offending references: {}",
        files.len(),
        offenders.len()
    );

    fs::remove_file(root.join("clean.md")).unwrap();
    assert!(
        find_deleted_references(root, &files)
            .unwrap_err()
            .contains("clean.md")
    );
}

#[test]
fn empty_text_populations_are_errors() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path();
    fixture_git(root, &["init", "--quiet"]);
    assert!(
        tracked_text_files(root)
            .unwrap_err()
            .contains("no tracked text files")
    );
    fs::write(root.join("tracked.bin"), "binary-only population").unwrap();
    fixture_git(root, &["add", "tracked.bin"]);
    assert!(
        tracked_text_files(root)
            .unwrap_err()
            .contains("no tracked text files")
    );
    assert!(
        find_deleted_references(root, &[])
            .unwrap_err()
            .contains("empty text population")
    );
}

#[test]
fn failed_discovery_is_an_error() {
    let fixture = tempfile::tempdir().unwrap();
    assert!(
        tracked_text_files(fixture.path())
            .unwrap_err()
            .contains("git ls-files failed")
    );
}
