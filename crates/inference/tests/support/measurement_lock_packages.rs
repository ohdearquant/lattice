use super::*;
use serde_json::{Value, json};
use std::path::PathBuf;

struct PackageFixture {
    _directory: tempfile::TempDir,
    inference: PathBuf,
    packages: Vec<ScannedPackage>,
    metadata: Value,
}

impl PackageFixture {
    fn new(first: &str, second: &str) -> Self {
        let directory = tempfile::tempdir().expect("create package coverage fixture");
        let names = [
            ("inference", "lattice-inference"),
            ("serve", "lattice-serve"),
            ("core", "lattice-inference-core"),
            ("tokenizer", "lattice-tokenizer"),
            ("grammar", "lattice-grammar"),
        ];
        let mut records = Vec::new();
        let mut members = Vec::new();
        for (relative, name) in names {
            let root = directory.path().join(relative);
            std::fs::create_dir_all(root.join("src")).expect("create fixture source directory");
            std::fs::write(root.join("src/lib.rs"), "").expect("write fixture library");
            let manifest = root.join("Cargo.toml");
            std::fs::write(
                &manifest,
                format!("[package]\nname = {name:?}\nversion = \"0.0.0\"\nedition = \"2024\"\n"),
            )
            .expect("write fixture package manifest");
            let dependencies = if relative == "serve" {
                json!([{"name": "lattice-inference", "rename": "renamed_runtime", "features": ["serving-support"]}])
            } else {
                json!([])
            };
            let id = format!("fixture:{name}");
            members.push(json!(id));
            records.push(json!({
                "id": id,
                "name": name,
                "manifest_path": manifest,
                "dependencies": dependencies,
            }));
        }
        std::fs::write(
            directory.path().join("Cargo.toml"),
            "[workspace]\nresolver = \"2\"\nmembers = [\"inference\", \"serve\", \"core\", \"tokenizer\", \"grammar\"]\n",
        ).expect("write fixture workspace");
        for (relative, source) in [("inference", first), ("serve", second)] {
            let examples = directory.path().join(relative).join("examples");
            std::fs::create_dir_all(&examples).expect("create fixture examples");
            std::fs::write(examples.join("metal.rs"), source).expect("write fixture Metal target");
        }
        let inference = directory.path().join("inference");
        let packages = scanned_packages_at(&inference, &[".", "../serve"])
            .expect("resolve both fixture packages");
        Self {
            _directory: directory,
            inference,
            packages,
            metadata: json!({"version": 1, "packages": records, "workspace_members": members}),
        }
    }
}

const UNGUARDED_CONSTRUCTION: &str = "fn main() { MetalQwen35State::new(); }\n";
const GUARDED_CONSTRUCTION: &str = "fn main() { let _guard = lattice_inference::measurement::gpu_test_lock(); MetalQwen35State::new(); }\n";

#[test]
fn scanned_package_list_matches_serving_support_workspace_members() {
    validate_current_scanned_package_membership()
        .expect("scanned packages must cover inference, EmbeddingGemma 2, Qwen3 and every serving-support consumer");
}

#[test]
fn lock_inventory_rows_name_scanned_packages() {
    let packages = scanned_packages();
    let owners = RAW_HARNESS_ENTRYPOINTS
        .iter()
        .chain(ADDITIONAL_GUARDED_ENTRYPOINTS)
        .chain(STATE_WORK_HELPER_ENTRYPOINTS)
        .map(|(package, _, _)| *package)
        .chain(
            TARGETS_WITH_STATE_DRIVEN_WORK
                .iter()
                .map(|(package, _)| *package),
        )
        .chain(
            TARGETS_WITH_RECOGNIZED_METAL_MARKERS
                .iter()
                .map(|(package, _)| *package),
        )
        .chain(
            TARGETS_WITHOUT_RECOGNIZED_METAL_MARKERS
                .iter()
                .map(|(package, _)| *package),
        )
        .chain(
            IN_CRATE_COMMAND_BUFFER_TESTS
                .iter()
                .map(|(package, _)| *package),
        )
        .chain(GENERATE_WORK_TARGETS.iter().map(|target| target.package))
        .chain(BENCH_HARNESSES.iter().map(|target| target.package))
        .chain(
            CONSTRUCTION_EXEMPTIONS
                .iter()
                .map(|exemption| exemption.package),
        );
    for owner in owners {
        measurement_lock_helpers::require_package(&packages, owner);
    }
}

#[test]
fn scanned_package_membership_tracks_renamed_serving_support_dependencies() {
    let fixture = PackageFixture::new("fn main() {}", "fn main() {}");
    validate_scanned_package_membership(&fixture.inference, &[".", "../serve"], &fixture.metadata)
        .expect("actual dependency name must survive a Cargo rename");
    let error = validate_scanned_package_membership(&fixture.inference, &["."], &fixture.metadata)
        .expect_err("omitted serving-support consumer must fail");
    assert!(error.contains("membership mismatch"), "{error}");
    let mut unrelated = fixture.metadata.clone();
    unrelated["packages"][1]["dependencies"][0]["name"] = json!("renamed_runtime");
    validate_scanned_package_membership(&fixture.inference, &["."], &unrelated)
        .expect("an unrelated dependency's rename is not lattice-inference");
    assert!(
        validate_scanned_package_membership(&fixture.inference, &[".", "../serve"], &unrelated,)
            .is_err(),
        "an unlisted package identity was accepted"
    );
}

#[test]
fn scanned_package_membership_rejects_malformed_and_duplicate_identities() {
    let fixture = PackageFixture::new("fn main() {}", "fn main() {}");
    for identities in [
        vec![".", "."],
        vec!["./"],
        vec!["../serve", "../serve"],
        vec![],
    ] {
        assert!(validate_scanned_package_membership(
            &fixture.inference, &identities, &fixture.metadata,
        ).is_err(), "accepted package identities {identities:?}");
    }
    let mut malformed = Vec::new();
    let mut missing_features = fixture.metadata.clone();
    missing_features["packages"][1]["dependencies"][0]
        .as_object_mut()
        .unwrap()
        .remove("features");
    malformed.push(missing_features);
    let mut missing_dependencies = fixture.metadata.clone();
    missing_dependencies["packages"][1]
        .as_object_mut()
        .unwrap()
        .remove("dependencies");
    malformed.push(missing_dependencies);
    let mut invalid_feature = fixture.metadata.clone();
    invalid_feature["packages"][1]["dependencies"][0]["features"] = json!([7]);
    malformed.push(invalid_feature);
    let mut duplicate_features = fixture.metadata.clone();
    duplicate_features["packages"][1]["dependencies"][0]["features"] =
        json!(["serving-support", "serving-support"]);
    malformed.push(duplicate_features);
    let mut missing_name = fixture.metadata.clone();
    missing_name["packages"][1]["dependencies"][0]["name"] = Value::Null;
    malformed.push(missing_name);
    let mut duplicate_manifest = fixture.metadata.clone();
    duplicate_manifest["packages"][1]["manifest_path"] =
        duplicate_manifest["packages"][0]["manifest_path"].clone();
    malformed.push(duplicate_manifest);
    let mut duplicate_member = fixture.metadata.clone();
    duplicate_member["workspace_members"]
        .as_array_mut()
        .unwrap()
        .push(json!("fixture:lattice-serve"));
    malformed.push(duplicate_member);
    let mut unknown_member = fixture.metadata.clone();
    unknown_member["workspace_members"]
        .as_array_mut()
        .unwrap()
        .push(json!("fixture:missing"));
    malformed.push(unknown_member);
    for metadata in malformed {
        assert!(
            validate_scanned_package_membership(&fixture.inference, &[".", "../serve"], &metadata,)
                .is_err(),
            "accepted malformed workspace metadata: {metadata}"
        );
    }
}

#[test]
fn package_construction_scan_reports_unclassified_second_package() {
    let fixture = PackageFixture::new(GUARDED_CONSTRUCTION, UNGUARDED_CONSTRUCTION);
    let inventory = measurement_lock_construction::scan_constructions(&fixture.packages, &[]);
    assert_eq!(
        inventory.discovered.len(),
        2,
        "both packages must enter the production collector"
    );
    assert_eq!(
        inventory.classified.len(),
        1,
        "the first package retains its live guard"
    );
    assert_eq!(
        inventory.violations.len(),
        1,
        "the second package must be reported"
    );
    assert!(inventory.violations[0].starts_with("../serve:example:metal:examples/metal.rs=>../serve:examples/metal.rs::main::MetalQwen35State::new()#1:"), "{:?}", inventory.violations);
    let unclassified = inventory
        .discovered
        .difference(&inventory.classified)
        .collect::<Vec<_>>();
    assert_eq!(unclassified.len(), 1);
    assert!(unclassified[0].starts_with("../serve:"));
    assert!(
        std::panic::catch_unwind(|| assert_construction_inventory_classified(
            &inventory.discovered,
            &inventory.classified,
        ))
        .is_err(),
        "the production inventory assertion must reject the reported site"
    );
}

#[test]
fn package_construction_keys_keep_same_relative_sites_distinct() {
    let fixture = PackageFixture::new(UNGUARDED_CONSTRUCTION, UNGUARDED_CONSTRUCTION);
    let inventory = measurement_lock_construction::scan_constructions(&fixture.packages, &[]);
    let expected = BTreeSet::from([
        ".:example:metal:examples/metal.rs=>.:examples/metal.rs::main::MetalQwen35State::new()#1".to_owned(),
        "../serve:example:metal:examples/metal.rs=>../serve:examples/metal.rs::main::MetalQwen35State::new()#1".to_owned(),
    ]);
    assert_eq!(
        inventory.discovered, expected,
        "same relative path and target name need distinct package keys"
    );
    assert_eq!(inventory.violations.len(), 2);
    assert!(inventory.classified.is_empty());
}

#[test]
fn scanned_package_membership_requires_embeddinggemma2_and_rejects_extra_packages() {
    let mut fixture = PackageFixture::new("fn main() {}", "fn main() {}");
    let family = fixture._directory.path().join("models/embeddinggemma2");
    std::fs::create_dir_all(family.join("src")).expect("create family fixture source directory");
    std::fs::write(family.join("src/lib.rs"), "").expect("write family fixture source");
    let manifest = family.join("Cargo.toml");
    std::fs::write(
        &manifest,
        "[package]\nname = \"lattice-model-embeddinggemma2\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
    )
    .expect("write family fixture manifest");
    fixture.metadata["packages"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "id": "fixture:lattice-model-embeddinggemma2",
            "name": "lattice-model-embeddinggemma2",
            "manifest_path": manifest,
            "dependencies": [],
        }));
    fixture.metadata["workspace_members"]
        .as_array_mut()
        .unwrap()
        .push(json!("fixture:lattice-model-embeddinggemma2"));
    let complete = [".", "../serve", "../models/embeddinggemma2"];
    validate_scanned_package_membership(&fixture.inference, &complete, &fixture.metadata)
        .expect("accept the family and serving consumers together");
    let omitted = validate_scanned_package_membership(
        &fixture.inference,
        &[".", "../serve"],
        &fixture.metadata,
    )
    .expect_err("omitting the family must fail membership validation");
    assert!(omitted.contains("membership mismatch"), "{omitted}");
    let extra = validate_scanned_package_membership(
        &fixture.inference,
        &[".", "../serve", "../models/embeddinggemma2", "../core"],
        &fixture.metadata,
    )
    .expect_err("an extra unrelated package must fail membership validation");
    assert!(extra.contains("membership mismatch"), "{extra}");
}

#[test]
fn qwen_family_membership_and_construction_guards_are_required() {
    let mut fixture = PackageFixture::new("fn main() {}", "fn main() {}");
    let family = fixture.inference.join("../models/qwen3");
    std::fs::create_dir_all(family.join("src")).unwrap();
    std::fs::write(
        family.join("Cargo.toml"),
        "[package]\nname = \"lattice-model-qwen3\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    let guarded = "#[cfg(test)] mod tests { #[test] fn checkpoint() { let _guard = gpu_test_lock(); QwenModel::from_directory(); } }";
    std::fs::write(family.join("src/lib.rs"), guarded).unwrap();
    let workspace = fixture.inference.join("../Cargo.toml");
    let members = std::fs::read_to_string(&workspace).unwrap();
    std::fs::write(
        &workspace,
        members.replace("\"grammar\"]", "\"grammar\", \"models/qwen3\"]"),
    )
    .unwrap();
    fixture.metadata["packages"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "id": "fixture:lattice-model-qwen3",
            "name": "lattice-model-qwen3",
            "manifest_path": family.join("Cargo.toml"),
            "dependencies": [],
        }));
    fixture.metadata["workspace_members"]
        .as_array_mut()
        .unwrap()
        .push(json!("fixture:lattice-model-qwen3"));
    assert!(
        validate_scanned_package_membership(
            &fixture.inference,
            &[".", "../serve"],
            &fixture.metadata
        )
        .is_err()
    );
    validate_scanned_package_membership(
        &fixture.inference,
        &[".", "../serve", "../models/qwen3"],
        &fixture.metadata,
    )
    .unwrap();
    fixture.packages =
        scanned_packages_at(&fixture.inference, &[".", "../serve", "../models/qwen3"]).unwrap();
    let inventory = measurement_lock_construction::scan_constructions(&fixture.packages, &[]);
    assert_eq!(inventory.discovered.len(), 1);
    assert!(
        inventory
            .discovered
            .iter()
            .all(|site| site.starts_with("../models/qwen3:")
                && site.contains("src/lib.rs::tests::checkpoint::QwenModel::from_directory()#1"))
    );
    assert_eq!(inventory.discovered, inventory.classified);
    assert!(inventory.violations.is_empty());
    std::fs::write(
        family.join("src/lib.rs"),
        guarded.replace("let _guard = gpu_test_lock(); ", ""),
    )
    .unwrap();
    let unguarded = measurement_lock_construction::scan_constructions(&fixture.packages, &[]);
    assert_eq!(unguarded.discovered, inventory.discovered);
    assert!(unguarded.classified.is_empty());
    assert_eq!(unguarded.violations.len(), 1);
}

#[test]
fn scanned_package_membership_requires_ernie45_and_rejects_extra_packages() {
    let mut fixture = PackageFixture::new("fn main() {}", "fn main() {}");
    let family = fixture._directory.path().join("models/ernie45");
    std::fs::create_dir_all(family.join("src")).expect("create family fixture source directory");
    std::fs::write(family.join("src/lib.rs"), "").expect("write family fixture source");
    let manifest = family.join("Cargo.toml");
    std::fs::write(
        &manifest,
        "[package]\nname = \"lattice-model-ernie45\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
    )
    .expect("write family fixture manifest");
    fixture.metadata["packages"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "id": "fixture:lattice-model-ernie45",
            "name": "lattice-model-ernie45",
            "manifest_path": manifest,
            "dependencies": [],
        }));
    fixture.metadata["workspace_members"]
        .as_array_mut()
        .unwrap()
        .push(json!("fixture:lattice-model-ernie45"));
    let complete = [".", "../serve", "../models/ernie45"];
    validate_scanned_package_membership(&fixture.inference, &complete, &fixture.metadata)
        .expect("accept the family and serving consumers together");
    let omitted = validate_scanned_package_membership(
        &fixture.inference,
        &[".", "../serve"],
        &fixture.metadata,
    )
    .expect_err("omitting the family must fail membership validation");
    assert!(omitted.contains("membership mismatch"), "{omitted}");
    let extra = validate_scanned_package_membership(
        &fixture.inference,
        &[".", "../serve", "../models/ernie45", "../core"],
        &fixture.metadata,
    )
    .expect_err("an extra unrelated package must fail membership validation");
    assert!(extra.contains("membership mismatch"), "{extra}");
}

#[test]
fn scanned_package_membership_requires_qwen35_and_rejects_extra_packages() {
    let mut fixture = PackageFixture::new("fn main() {}", "fn main() {}");
    let family = fixture._directory.path().join("models/qwen35");
    std::fs::create_dir_all(family.join("src")).expect("create family fixture source directory");
    std::fs::write(family.join("src/lib.rs"), "").expect("write family fixture source");
    let manifest = family.join("Cargo.toml");
    std::fs::write(
        &manifest,
        "[package]\nname = \"lattice-model-qwen35\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
    )
    .expect("write family fixture manifest");
    fixture.metadata["packages"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "id": "fixture:lattice-model-qwen35",
            "name": "lattice-model-qwen35",
            "manifest_path": manifest,
            "dependencies": [],
        }));
    fixture.metadata["workspace_members"]
        .as_array_mut()
        .unwrap()
        .push(json!("fixture:lattice-model-qwen35"));
    let complete = [".", "../serve", "../models/qwen35"];
    validate_scanned_package_membership(&fixture.inference, &complete, &fixture.metadata)
        .expect("accept the family and serving consumers together");
    let omitted = validate_scanned_package_membership(
        &fixture.inference,
        &[".", "../serve"],
        &fixture.metadata,
    )
    .expect_err("omitting the family must fail membership validation");
    assert!(omitted.contains("membership mismatch"), "{omitted}");
    let extra = validate_scanned_package_membership(
        &fixture.inference,
        &[".", "../serve", "../models/qwen35", "../core"],
        &fixture.metadata,
    )
    .expect_err("an extra unrelated package must fail membership validation");
    assert!(extra.contains("membership mismatch"), "{extra}");
}
