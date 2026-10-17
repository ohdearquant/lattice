#[path = "support/metal_qwen35_population.rs"]
#[allow(dead_code)]
mod metal_qwen35_population;
#[path = "support/source_graph.rs"]
#[allow(dead_code)]
mod source_graph;

use metal_qwen35_population::{FileKind, PopulationManifest, ProductionItem, derive_population};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const ROOT: &str = "src/forward/metal_qwen35.rs";
const SUBTREE: &str = "src/forward/metal_qwen35";

struct Fixture {
    directory: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = loop {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir().join(format!(
                "lattice-metal-population-{}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create fixture directory {}: {error}", candidate.display()),
            }
        };
        fs::create_dir_all(directory.join("src/forward/metal_qwen35"))
            .expect("create fixture source tree");
        Self { directory }
    }

    fn write(&self, path: &str, content: &str) {
        let path = self.directory.join(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create fixture parent");
        }
        fs::write(path, content).expect("write fixture source");
    }

    fn root(&self, content: &str) {
        self.write(ROOT, content);
    }

    fn derive(&self) -> Result<PopulationManifest, String> {
        derive_population(&self.directory, Path::new(ROOT), Path::new(SUBTREE))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn omitting_a_discovered_production_file_fails_item_coverage() {
    let fixture = Fixture::new();
    fixture.root("mod production;\n");
    fixture.write(
        "src/forward/metal_qwen35/production.rs",
        "pub fn production_item() {}\n",
    );
    let manifest = fixture.derive().expect("derive fixture population");
    let read_items = manifest
        .production_items
        .iter()
        .filter(|item| item.file != "src/forward/metal_qwen35/production.rs")
        .cloned()
        .collect::<Vec<ProductionItem>>();

    let error = manifest
        .require_production_items_read(&read_items)
        .expect_err("omitted production item must fail coverage");
    assert!(error.contains("src/forward/metal_qwen35/production.rs"));
}

#[test]
fn nested_production_files_are_covered_without_a_file_list() {
    let fixture = Fixture::new();
    fixture.root("mod inner;\n");
    fixture.write("src/forward/metal_qwen35/inner.rs", "mod nested;\n");
    fixture.write(
        "src/forward/metal_qwen35/inner/nested.rs",
        "pub fn nested_production_item() {}\n",
    );

    let manifest = fixture.derive().expect("derive fixture population");
    assert!(
        manifest
            .discovered_files
            .iter()
            .any(|file| file == "src/forward/metal_qwen35/inner/nested.rs")
    );
    assert!(manifest.production_items.iter().any(|item| {
        item.file == "src/forward/metal_qwen35/inner/nested.rs"
            && item.qualified_name.ends_with("nested_production_item")
    }));
}

#[test]
fn shared_source_reachability_reclassifies_descendant_production_items() {
    let fixture = Fixture::new();
    fixture.root(
        "#[cfg(test)]\n#[path = \"metal_qwen35/shared.rs\"]\nmod test_route;\n\
         #[path = \"metal_qwen35/shared.rs\"]\nmod production_route;\n",
    );
    fixture.write("src/forward/metal_qwen35/shared.rs", "mod leaf;\n");
    fixture.write(
        "src/forward/metal_qwen35/leaf.rs",
        "pub fn shared_leaf_item() {}\n",
    );

    let manifest = fixture.derive().expect("derive shared source population");
    let classification = manifest
        .classifications
        .iter()
        .find(|classification| classification.file == "src/forward/metal_qwen35/leaf.rs")
        .expect("shared descendant file classification");
    assert_eq!(classification.kind, FileKind::Production);
    assert!(manifest.production_items.iter().any(|item| {
        item.file == "src/forward/metal_qwen35/leaf.rs"
            && item.qualified_name.ends_with("shared_leaf_item")
    }));

    let read_items = manifest
        .production_items
        .iter()
        .filter(|item| item.file != "src/forward/metal_qwen35/leaf.rs")
        .cloned()
        .collect::<Vec<ProductionItem>>();
    let error = manifest
        .require_production_items_read(&read_items)
        .expect_err("omitted shared descendant must fail coverage");
    assert!(error.contains("src/forward/metal_qwen35/leaf.rs"));
}

#[test]
fn included_rust_files_join_the_population() {
    let fixture = Fixture::new();
    fixture.root("include!(\"metal_qwen35/included.rs\");\n");
    fixture.write(
        "src/forward/metal_qwen35/included.rs",
        "pub fn included_production_item() {}\n",
    );

    let manifest = fixture.derive().expect("derive fixture population");
    assert!(
        manifest
            .discovered_files
            .iter()
            .any(|file| file == "src/forward/metal_qwen35/included.rs")
    );
    assert!(manifest.production_items.iter().any(|item| {
        item.file == "src/forward/metal_qwen35/included.rs"
            && item.qualified_name.ends_with("included_production_item")
    }));
}

#[test]
fn an_unreachable_file_under_the_subtree_fails_reconciliation() {
    let fixture = Fixture::new();
    fixture.root("pub fn root_item() {}\n");
    fixture.write(
        "src/forward/metal_qwen35/unlisted.rs",
        "pub fn unlisted_item() {}\n",
    );

    let error = fixture
        .derive()
        .expect_err("unreachable subtree file must fail reconciliation");
    assert!(error.contains("src/forward/metal_qwen35/unlisted.rs"));
}

#[test]
fn raw_strings_and_test_modules_do_not_truncate_production_items() {
    let fixture = Fixture::new();
    fixture.root(
        r##"
const RAW_SOURCE: &str = r#"mod tests {"#;

#[cfg(test)]
    mod tests {
        fn test_only_item() {}
    }

fn production_after_tests() {}
"##,
    );

    let manifest = fixture.derive().expect("derive fixture population");
    assert!(
        manifest
            .production_items
            .iter()
            .any(|item| { item.qualified_name == "production_after_tests" && item.file == ROOT })
    );
    assert!(
        !manifest
            .production_items
            .iter()
            .any(|item| { item.qualified_name.ends_with("test_only_item") })
    );
}

#[test]
fn test_or_metal_cfg_is_production_reachable() {
    let fixture = Fixture::new();
    fixture.root(
        "#[cfg(any(test, all(target_os = \"macos\", feature = \"metal-gpu\")))]\nmod route;\n",
    );
    fixture.write(
        "src/forward/metal_qwen35/route.rs",
        "pub fn metal_route_item() {}\n",
    );

    let manifest = fixture.derive().expect("derive fixture population");
    let classification = manifest
        .classifications
        .iter()
        .find(|classification| classification.file == "src/forward/metal_qwen35/route.rs")
        .expect("route file classification");
    assert_eq!(classification.kind, FileKind::Production);
    assert!(manifest.production_items.iter().any(|item| {
        item.file == "src/forward/metal_qwen35/route.rs"
            && item.qualified_name.ends_with("metal_route_item")
    }));
}

#[test]
fn metal_and_portable_cfg_items_remain_production_reachable() {
    let fixture = Fixture::new();
    fixture.root(
        r#"
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn metal_item() {}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub fn portable_item() {}

#[cfg(test)]
fn test_only_item() {}
"#,
    );

    let manifest = fixture.derive().expect("derive configuration union");
    for name in ["metal_item", "portable_item"] {
        assert!(
            manifest
                .production_items
                .iter()
                .any(|item| item.file == ROOT && item.qualified_name == name),
            "production item {name} is missing from the configuration union"
        );
    }
    assert!(
        !manifest
            .production_items
            .iter()
            .any(|item| item.file == ROOT && item.qualified_name == "test_only_item")
    );
}

#[test]
fn unresolved_include_and_path_destinations_fail_with_the_path() {
    let include_fixture = Fixture::new();
    include_fixture.root("include!(\"missing_include.rs\");\n");
    let include_error = include_fixture
        .derive()
        .expect_err("unresolved include must fail");
    assert!(include_error.contains("missing_include.rs"));

    let path_fixture = Fixture::new();
    path_fixture.root("#[path = \"metal_qwen35/missing_path.rs\"]\nmod missing;\n");
    let path_error = path_fixture
        .derive()
        .expect_err("unresolved path destination must fail");
    assert!(path_error.contains("missing_path.rs"));
}

#[test]
fn ambiguous_cfg_and_unparsable_files_fail_closed() {
    let cfg_fixture = Fixture::new();
    cfg_fixture.root("#[cfg(custom(value))]\npub fn ambiguous_item() {}\n");
    let cfg_error = cfg_fixture
        .derive()
        .expect_err("unsupported cfg form must fail classification");
    assert!(cfg_error.contains(ROOT));
    assert!(cfg_error.contains("unsupported cfg predicate"));

    let syntax_fixture = Fixture::new();
    syntax_fixture.root("pub fn incomplete_item(\n");
    let syntax_error = syntax_fixture
        .derive()
        .expect_err("unparsable Rust file must fail classification");
    assert!(syntax_error.contains(ROOT));
    assert!(syntax_error.contains("Rust syntax could not be classified"));
}

#[test]
fn real_metal_source_population_reconciles() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest = derive_population(&crate_dir, Path::new(ROOT), Path::new(SUBTREE))
        .expect("derive real source population");

    assert!(!manifest.discovered_files.is_empty());
    assert!(manifest.discovered_files.iter().any(|file| file == ROOT));
    assert!(manifest.reconciliation_failures.is_empty());
}
