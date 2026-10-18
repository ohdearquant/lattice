#[path = "support/metal_qwen35_population.rs"]
#[allow(dead_code)]
mod metal_qwen35_population;
#[path = "support/metal_source_guards.rs"]
#[allow(dead_code)]
mod metal_source_guards;
#[path = "support/source_graph.rs"]
#[allow(dead_code)]
mod source_graph;

use metal_qwen35_population::{FileKind, derive_population};
use metal_source_guards::{CallSpec, GuardPopulation, OwnerKind, OwnerSpec, Scope};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const ROOT: &str = "src/forward/metal_qwen35.rs";
const SUBTREE: &str = "src/forward/metal_qwen35";
const SHARED: &str = "src/forward/shared.rs";
const RETIRED: &str = "gdn_cpu_recurrence";
const OWNERS: &[OwnerSpec] = &[
    OwnerSpec {
        id: "sampler",
        path: "crate::forward::metal_qwen35::sample",
        kind: OwnerKind::Function,
        scope: Scope::Production,
    },
    OwnerSpec {
        id: "caller",
        path: "crate::forward::metal_qwen35::generate",
        kind: OwnerKind::Function,
        scope: Scope::Production,
    },
];
const CALLS: &[CallSpec] = &[CallSpec {
    caller: "caller",
    callee: "sampler",
    count: 1,
}];
const LOCAL: &str = "pub fn sample() {}\n\
    use self::sample as selected_sample;\n\
    pub fn generate() { selected_sample(); }\n";
const RELOCATED: &str = "pub use super::shared::sample;\n\
    use self::sample as selected_sample;\n\
    pub fn generate() { selected_sample(); }\n";

struct Fixture {
    directory: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let fixture = Self {
            directory: tempfile::tempdir().expect("create source population fixture"),
        };
        fs::create_dir_all(fixture.path().join(SUBTREE)).expect("create population subtree");
        fixture.write("src/lib.rs", "pub mod forward; pub mod decoder;\n");
        fixture.write("src/forward/mod.rs", "pub mod metal_qwen35;\n");
        fixture.write("src/decoder.rs", "pub mod qwen_metal;\n");
        fixture.write("src/decoder/qwen_metal.rs", "pub fn decoder_ready() {}\n");
        fixture.write(ROOT, LOCAL);
        fixture
    }

    fn path(&self) -> &Path {
        self.directory.path()
    }

    fn write(&self, path: &str, source: &str) {
        let path = self.path().join(path);
        fs::create_dir_all(path.parent().expect("fixture path has a parent"))
            .expect("create fixture source parent");
        fs::write(path, source).expect("write fixture source");
    }

    fn remove(&self, path: &str) {
        fs::remove_file(self.path().join(path)).expect("remove fixture source");
    }

    fn load(&self) -> GuardPopulation {
        GuardPopulation::load(self.path(), OWNERS).expect("resolve fixture owners and population")
    }

    fn relocate(&self) {
        self.write("src/forward/mod.rs", "pub mod metal_qwen35; mod shared;\n");
        self.write(ROOT, RELOCATED);
        self.write(SHARED, "pub fn sample() {}\n");
    }

    fn legacy_modules(&self) {
        self.write(
            ROOT,
            &format!(
                "{LOCAL}\n\
                 // mod comment_decoy;\n\
                 const TEXT: &str = \"mod string_decoy;\";\n\
                 #[cfg(test)] mod tests {{ mod test_decoy; }}\n\
                 #[cfg(test)] mod external_test_decoy;\n\
                 #[cfg(any(test, feature = \"metal-gpu\"))] mod weights;\n\
                 mod inner {{ pub(super) mod layers; }}\n"
            ),
        );
        self.write("src/forward/metal_qwen35/weights.rs", "");
        self.write("src/forward/metal_qwen35/inner/layers/mod.rs", "mod gqa;\n");
        self.write("src/forward/metal_qwen35/inner/layers/gqa.rs", "");
        self.write(
            "src/forward/metal_qwen35/tests/test_decoy.rs",
            "fn gdn_cpu_recurrence() {}\n",
        );
        self.write(
            "src/forward/metal_qwen35/external_test_decoy.rs",
            "fn gdn_cpu_recurrence() {}\n",
        );
    }
}

fn assert_retired_refusal(population: &GuardPopulation, file: &str) {
    let error = population
        .reject_declared_functions(&[RETIRED])
        .expect_err("retired production declaration must fail the guard");
    assert!(
        error.contains(RETIRED),
        "wrong declaration refusal: {error}"
    );
    assert!(error.contains(file), "missing offending file: {error}");
}

#[test]
fn external_sibling_relocation_preserves_frozen_owners_and_callers() {
    let fixture = Fixture::new();
    let before = fixture.load();
    before.require_calls(CALLS).expect("original caller census");
    before
        .reject_declared_functions(&[RETIRED])
        .expect("original declarations are live");
    let sampler_body = before.owner_body("sampler").unwrap().to_owned();
    let caller_body = before.owner_body("caller").unwrap().to_owned();

    fixture.relocate();
    assert!(!RELOCATED.contains("mod shared"));
    assert!(!RELOCATED.contains("#[path"));
    assert!(!RELOCATED.contains("include!"));
    let after = fixture.load();
    assert!(after.production_sources().any(|(path, _)| path == SHARED));
    assert_eq!(after.owner_body("sampler").unwrap(), sampler_body);
    assert_eq!(after.owner_body("caller").unwrap(), caller_body);
    after.require_calls(CALLS).expect("relocated caller census");
    after
        .reject_declared_functions(&[RETIRED])
        .expect("relocated declarations are live");
}

#[test]
fn qualified_reexport_calls_resolve_the_external_owner() {
    let fixture = Fixture::new();
    fixture.relocate();
    fixture.write(
        ROOT,
        "pub use super::shared::sample;\n\
         pub fn generate() { crate::forward::metal_qwen35::sample(); }\n",
    );
    let population = fixture.load();
    assert!(
        population
            .production_sources()
            .any(|(path, _)| path == SHARED)
    );
    population
        .require_calls(CALLS)
        .expect("qualified facade call must resolve the relocated owner");
}

#[test]
fn relocated_caller_bypass_is_rejected() {
    let fixture = Fixture::new();
    fixture.relocate();
    fixture.write(
        ROOT,
        "pub use super::shared::sample;\n\
         fn unrelated() {}\n\
         pub fn generate() { unrelated(); }\n",
    );
    let population = fixture.load();
    assert!(
        population
            .production_sources()
            .any(|(path, _)| path == SHARED)
    );
    population
        .reject_declared_functions(&[RETIRED])
        .expect("the bypass fixture contains no retired declaration");
    let error = population
        .require_calls(CALLS)
        .expect_err("relocated caller bypass must fail the declared call census");
    assert!(error.contains("sampler"), "wrong caller refusal: {error}");
}

#[test]
fn retired_declaration_in_external_owner_file_is_rejected() {
    let fixture = Fixture::new();
    fixture.relocate();
    fixture.write(SHARED, "pub fn sample() {}\nfn gdn_cpu_recurrence() {}\n");
    let population = fixture.load();
    population
        .require_calls(CALLS)
        .expect("caller remains intact");
    assert_retired_refusal(&population, SHARED);

    fixture.write(
        SHARED,
        "pub fn sample() {}\n\
         stringify! { const TEXT: &str = \"fn gdn_cpu_recurrence() {}\";\n\
         /* fn gdn_cpu_recurrence() {} */ }\n",
    );
    fixture
        .load()
        .reject_declared_functions(&[RETIRED])
        .expect("macro string and comment decoys are not declarations");

    fixture.write(
        SHARED,
        "pub fn sample() {}\nstringify! { fn gdn_cpu_recurrence() {} }\n",
    );
    let population = fixture.load();
    population
        .require_calls(CALLS)
        .expect("macro fixture caller remains intact");
    assert_retired_refusal(&population, SHARED);
}

#[test]
fn production_masking_keeps_attributed_call_arguments_parseable() {
    let fixture = Fixture::new();
    fixture.write(
        ROOT,
        "pub fn sample() {}\n\
         use self::sample as selected_sample;\n\
         pub fn generate() { selected_sample(); configure(#[cfg(test)] None, 1); }\n\
         pub fn configure(value: Option<u8>, scope: u8) { let _ = (value, scope); }\n",
    );
    let population = fixture.load();
    population
        .require_calls(CALLS)
        .expect("the caller census must survive argument masking");
}

#[test]
fn omitted_external_destination_refuses_before_negative_scan() {
    let fixture = Fixture::new();
    fixture.relocate();
    let scan_entered = Cell::new(false);
    let result =
        GuardPopulation::load_with_discovery_filter(fixture.path(), OWNERS, |path| path != SHARED)
            .and_then(|population| {
                scan_entered.set(true);
                population.reject_declared_functions(&[RETIRED])
            });
    assert!(
        !scan_entered.get(),
        "negative declaration scan ran before omitted destination refused"
    );
    let error = result.expect_err("omitted external destination must fail completeness");
    assert!(
        error.contains(SHARED),
        "missing destination diagnostic: {error}"
    );
}

#[test]
fn duplicate_discovery_candidates_are_refused() {
    let fixture = Fixture::new();
    let candidates = vec![ROOT.to_owned(), ROOT.to_owned()];
    let error = GuardPopulation::load_with_candidates(fixture.path(), OWNERS, &candidates)
        .err()
        .expect("duplicate discovery candidates must refuse");
    assert!(
        error.contains("duplicate"),
        "wrong duplicate refusal: {error}"
    );
    assert!(error.contains(ROOT), "missing duplicate path: {error}");
}

#[test]
fn unresolved_external_owner_is_refused() {
    let fixture = Fixture::new();
    fixture.relocate();
    fixture.write(SHARED, "pub fn replacement() {}\n");
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("unresolved external owner must refuse");
    assert!(
        error.contains("sample"),
        "missing unresolved symbol: {error}"
    );

    fixture.write(SHARED, "pub fn sample() {}\n");
    fixture.write(
        ROOT,
        "pub use super::shared::*;\n\
         pub use super::missing::*;\n\
         pub fn generate() { sample(); }\n",
    );
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("one resolved glob must not hide another unresolved destination");
    assert!(
        error.contains("missing"),
        "missing glob diagnostic: {error}"
    );
}

#[test]
fn ambiguous_external_owner_is_refused() {
    let fixture = Fixture::new();
    fixture.write(
        "src/forward/mod.rs",
        "pub mod metal_qwen35; mod left; mod right;\n",
    );
    fixture.write("src/forward/left.rs", "pub fn sample() {}\n");
    fixture.write("src/forward/right.rs", "pub fn sample() {}\n");
    fixture.write(
        ROOT,
        "pub use super::left::sample;\n\
         pub use super::right::sample;\n\
         pub fn generate() { sample(); }\n",
    );
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("ambiguous external owner must refuse");
    assert!(
        error.contains("ambiguous"),
        "wrong ownership refusal: {error}"
    );
    assert!(
        error.contains("sample"),
        "missing ambiguous symbol: {error}"
    );

    fixture.write(
        ROOT,
        "pub use super::left::*;\n\
         pub use super::right::*;\n\
         pub fn generate() { sample(); }\n",
    );
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("competing glob owners must refuse");
    assert!(error.contains("ambiguous"), "wrong glob refusal: {error}");

    fixture.write(
        ROOT,
        "pub fn sample() {}\n\
         pub use super::left::*;\n\
         pub use super::right::*;\n\
         pub fn generate() { sample(); }\n",
    );
    fixture
        .load()
        .require_calls(CALLS)
        .expect("a direct local binding takes precedence over module globs");
}

#[test]
fn cyclic_glob_ownership_is_refused() {
    let fixture = Fixture::new();
    fixture.write(
        "src/forward/mod.rs",
        "pub mod metal_qwen35; mod left; mod right;\n",
    );
    fixture.write("src/forward/left.rs", "pub use super::right::*;\n");
    fixture.write("src/forward/right.rs", "pub use super::left::*;\n");
    fixture.write(
        ROOT,
        "pub use super::left::*;\n\
         pub fn generate() { sample(); }\n",
    );
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("a cycle while resolving a guarded owner must refuse");
    assert!(error.contains("cycle"), "wrong glob cycle refusal: {error}");
    assert!(error.contains("sample"), "missing guarded symbol: {error}");
}

#[test]
fn formerly_unlisted_nested_file_cannot_hide_a_retired_declaration() {
    let fixture = Fixture::new();
    fixture.write(ROOT, &format!("{LOCAL}\nmod formerly_unlisted;\n"));
    let nested = "src/forward/metal_qwen35/formerly_unlisted.rs";
    fixture.write(nested, "fn gdn_cpu_recurrence() {}\n");
    let population = fixture.load();
    population
        .require_calls(CALLS)
        .expect("caller remains intact");
    assert!(
        population
            .production_sources()
            .any(|(path, _)| path == nested)
    );
    assert_retired_refusal(&population, nested);
}

#[test]
fn nested_test_only_declarations_do_not_trigger_retirement_guard() {
    let fixture = Fixture::new();
    fixture.write(
        ROOT,
        &format!(
            "{LOCAL}\n\
             #[cfg(test)] mod first {{ mod nested {{ fn gdn_cpu_recurrence() {{}} }} }}\n\
             mod mixed {{ #[cfg(test)] fn gdn_cpu_recurrence() {{}} pub fn live() {{}} }}\n\
             pub fn local_scope() {{ #[cfg(test)] fn gdn_cpu_recurrence() {{}} }}\n\
             #[cfg(test)] test_helpers! {{ fn gdn_cpu_recurrence() {{}} }}\n"
        ),
    );
    let population = fixture.load();
    population
        .require_calls(CALLS)
        .expect("caller remains intact");
    population
        .reject_declared_functions(&[RETIRED])
        .expect("test-only declarations must not trigger the production guard");
}

#[test]
fn test_or_metal_declaration_remains_in_the_production_union() {
    let fixture = Fixture::new();
    fixture.write(
        ROOT,
        &format!(
            "{LOCAL}\n\
             #[cfg(any(test, all(target_os = \"macos\", feature = \"metal-gpu\")))]\n\
             mod reachable {{ fn gdn_cpu_recurrence() {{}} }}\n"
        ),
    );
    let population = fixture.load();
    population
        .require_calls(CALLS)
        .expect("caller remains intact");
    assert_retired_refusal(&population, ROOT);
}

#[test]
fn moving_tests_out_of_line_preserves_production_facts() {
    let fixture = Fixture::new();
    let decoy = "const RAW: &str = r#\"mod tests { #[cfg(test)]\"#;\n";
    fixture.write(
        ROOT,
        &format!("{decoy}#[cfg(test)] mod tests {{ fn gdn_cpu_recurrence() {{}} }}\n{LOCAL}"),
    );
    let before = fixture.load();
    let sampler_body = before.owner_body("sampler").unwrap().to_owned();
    let caller_body = before.owner_body("caller").unwrap().to_owned();
    before
        .require_calls(CALLS)
        .expect("inline-test caller census");
    before
        .reject_declared_functions(&[RETIRED])
        .expect("inline test-only declaration is excluded");

    fixture.write(ROOT, &format!("{decoy}#[cfg(test)] mod tests;\n{LOCAL}"));
    fixture.write(
        "src/forward/metal_qwen35/tests.rs",
        "fn gdn_cpu_recurrence() {}\n",
    );
    let after = fixture.load();
    assert_eq!(after.owner_body("sampler").unwrap(), sampler_body);
    assert_eq!(after.owner_body("caller").unwrap(), caller_body);
    after
        .require_calls(CALLS)
        .expect("out-of-line caller census");
    after
        .reject_declared_functions(&[RETIRED])
        .expect("out-of-line test-only declaration is excluded");
}

#[test]
fn derived_population_preserves_nested_and_cfg_module_coverage() {
    let fixture = Fixture::new();
    fixture.legacy_modules();
    let manifest = derive_population(fixture.path(), Path::new(ROOT), Path::new(SUBTREE))
        .expect("derive the nested module fixture");
    let production: BTreeSet<_> = manifest
        .classifications
        .iter()
        .filter(|file| file.kind == FileKind::Production)
        .map(|file| file.file.as_str())
        .collect();
    assert_eq!(
        production,
        BTreeSet::from([
            ROOT,
            "src/forward/metal_qwen35/weights.rs",
            "src/forward/metal_qwen35/inner/layers/mod.rs",
            "src/forward/metal_qwen35/inner/layers/gqa.rs",
        ])
    );
    for file in [
        "src/forward/metal_qwen35/tests/test_decoy.rs",
        "src/forward/metal_qwen35/external_test_decoy.rs",
    ] {
        assert!(
            manifest
                .classifications
                .iter()
                .any(|entry| { entry.file == file && entry.kind == FileKind::TestOnly })
        );
    }
    let population = fixture.load();
    population
        .require_calls(CALLS)
        .expect("nested fixture caller census");
    population
        .reject_declared_functions(&[RETIRED])
        .expect("the nested test decoys are excluded");
}

#[test]
fn missing_declared_module_destinations_are_refused() {
    let missing_weights = Fixture::new();
    missing_weights.legacy_modules();
    missing_weights.remove("src/forward/metal_qwen35/weights.rs");
    let error = GuardPopulation::load(missing_weights.path(), OWNERS)
        .err()
        .expect("missing weights module must refuse");
    assert!(
        error.contains("weights"),
        "missing weights diagnostic: {error}"
    );

    let missing_child = Fixture::new();
    missing_child.legacy_modules();
    missing_child.write(
        "src/forward/metal_qwen35/inner/layers/mod.rs",
        "mod gqa; mod extracted;\n",
    );
    let error = GuardPopulation::load(missing_child.path(), OWNERS)
        .err()
        .expect("newly declared but absent child must refuse");
    assert!(
        error.contains("extracted"),
        "missing child diagnostic: {error}"
    );
}

#[test]
fn unreachable_physical_source_is_refused() {
    let fixture = Fixture::new();
    let path = "src/forward/metal_qwen35/tests/test_decoy.rs";
    fixture.write(path, "fn gdn_cpu_recurrence() {}\n");
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("unreachable physical source must refuse before searching declarations");
    assert!(error.contains(path), "missing orphan diagnostic: {error}");
}

#[test]
fn ambiguous_flat_and_nested_module_files_are_refused() {
    let fixture = Fixture::new();
    fixture.legacy_modules();
    fixture.write("src/forward/metal_qwen35/weights/mod.rs", "");
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("both module destinations must refuse");
    assert!(error.contains("ambiguous"), "wrong module refusal: {error}");
    assert!(error.contains("weights"), "missing module name: {error}");
}

#[test]
fn absent_root_and_malformed_source_are_refused() {
    let fixture = Fixture::new();
    let error = GuardPopulation::load_with_candidates(fixture.path(), OWNERS, &[])
        .err()
        .expect("an empty supplied source population must refuse");
    assert!(
        error.contains("coverage is incomplete") && error.contains(ROOT),
        "wrong empty population refusal: {error}"
    );

    fixture.remove(ROOT);
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("an absent source root must refuse");
    assert!(
        error.contains("metal_qwen35"),
        "missing root diagnostic: {error}"
    );

    fixture.write(ROOT, "mod");
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("malformed source must refuse");
    assert!(
        error.contains(ROOT),
        "missing malformed source path: {error}"
    );
}

#[test]
fn literal_path_modules_are_resolved_and_missing_paths_refuse() {
    let fixture = Fixture::new();
    fixture.write(
        ROOT,
        &format!("{LOCAL}\n#[path = \"moved.rs\"] mod moved;\n"),
    );
    let destination = "src/forward/moved.rs";
    fixture.write(destination, "pub fn supported_path_item() {}\n");
    let population = fixture.load();
    assert!(
        population
            .production_sources()
            .any(|(path, _)| path == destination)
    );
    population
        .require_calls(CALLS)
        .expect("literal-path caller census");
    population
        .reject_declared_functions(&[RETIRED])
        .expect("literal-path production declarations are live");

    fixture.remove(destination);
    let error = GuardPopulation::load(fixture.path(), OWNERS)
        .err()
        .expect("unresolved literal module path must refuse");
    assert!(
        error.contains("moved.rs"),
        "missing literal path diagnostic: {error}"
    );
}
