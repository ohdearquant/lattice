use super::*;

/// Records the pre-expansion boundary of the checked binding convention.
///
/// Macro metavariables can expand into protected Metal work after this scanner
/// has run, so this source-level check intentionally does not classify them.
#[test]
fn macro_metavariable_expansion_is_outside_the_checked_binding_convention() {
    let source = r#"
macro_rules! submit {
    ($metal:ident, $device:ident, $system:ident, $queue:ident, $buffer:ident) => {{
        let device = $metal::$device::$system().unwrap();
        let queue = device.$queue();
        let command_buffer = queue.$buffer();
        command_buffer.commit();
    }};
}

fn main() {
    submit!(metal, Device, system_default, new_command_queue, new_command_buffer);
}
"#;
    let parsed = StructuredSource::parse("fixtures/macro_metavariable_limit.rs", source, false)
        .expect("parse macro-metavariable limit fixture");
    assert!(
        !parsed
            .has_top_level_metal_marker()
            .expect("classify the documented macro-metavariable limit"),
        "the pre-expansion convention unexpectedly claimed macro-expanded work"
    );
}

#[test]
fn malformed_rust_fails_closed_in_direct_source_parsing() {
    let result = StructuredSource::parse("fixtures/malformed_direct.rs", "fn broken( {", false);
    let Err(message) = result else {
        panic!("direct parser accepted malformed Rust");
    };
    assert!(
        message.contains("Rust syntax could not be classified"),
        "direct parser reported the wrong failure: {message}"
    );
}

#[test]
fn malformed_rust_fails_closed_in_module_closure_parsing() {
    let fixture = tempfile::tempdir().expect("create malformed module fixture");
    let root = fixture.path().join("main.rs");
    std::fs::write(&root, "mod broken;").expect("write fixture root");
    std::fs::write(fixture.path().join("broken.rs"), "fn broken( {")
        .expect("write malformed module");

    let result = module_source_closure(fixture.path(), &root, false);
    let Err(message) = result else {
        panic!("module closure accepted malformed Rust");
    };
    assert!(
        message.contains("broken.rs: Rust syntax could not be classified"),
        "module closure reported the wrong failure: {message}"
    );
}

#[test]
fn malformed_cfg_attribute_fails_closed_in_module_graph_parsing() {
    let fixture = tempfile::tempdir().expect("create malformed cfg fixture");
    let root = fixture.path().join("lib.rs");
    std::fs::write(&root, "#[cfg]\nmod child;\n").expect("write malformed cfg root");
    std::fs::write(fixture.path().join("child.rs"), "fn child() {}\n").expect("write cfg child");

    let result = module_source_closure(fixture.path(), &root, false);
    let Err(message) = result else {
        panic!("module graph accepted malformed cfg attribute");
    };
    assert!(
        message.contains("malformed `cfg` attribute"),
        "module graph reported the wrong cfg failure: {message}"
    );
}

#[test]
fn empty_cfg_predicate_segment_fails_closed() {
    let source = r#"
#[cfg(all(, unix))]
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    let result = StructuredSource::parse("fixtures/empty_cfg_segment.rs", source, false).and_then(
        |parsed| {
            parsed.validate_selector(
                0..parsed.tokens.len(),
                DEVICE_SYSTEM_DEFAULT,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )
        },
    );
    let Err(message) = result else {
        panic!("source contract accepted an empty cfg predicate segment");
    };
    assert!(
        message.contains("empty cfg predicate"),
        "empty cfg segment reported the wrong failure: {message}"
    );
}

#[test]
fn unretained_command_buffer_api_requires_a_function_lifetime_lock() {
    let source = r#"
#[test]
fn raw_dispatch() {
    let _command_buffer = queue.new_command_buffer_with_unretained_references();
}
"#;
    let parsed = StructuredSource::parse("fixtures/unretained_unguarded.rs", source, true)
        .expect("parse unretained command-buffer fixture");
    let function = parsed
        .test_functions()
        .expect("classify unretained command-buffer test")
        .into_iter()
        .next()
        .expect("find unretained command-buffer test");
    let work = parsed
        .call_sites(function.body.clone(), NEW_COMMAND_BUFFER_UNRETAINED, true)
        .expect("classify unretained command-buffer call");
    let error = parsed
        .validate_work_sites(
            &work,
            LOCAL_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        )
        .expect_err("unguarded unretained command buffer must fail");
    assert!(
        error.contains("no live shared-lock binding encloses this work"),
        "unretained command buffer reported the wrong failure: {error}"
    );
}

#[test]
fn unretained_command_buffer_api_accepts_a_function_lifetime_lock() {
    let source = r#"
#[test]
fn raw_dispatch() {
    let _gpu_guard = gpu_test_lock();
    let _command_buffer = queue.new_command_buffer_with_unretained_references();
}
"#;
    let parsed = StructuredSource::parse("fixtures/unretained_guarded.rs", source, true)
        .expect("parse guarded unretained command-buffer fixture");
    let function = parsed
        .test_functions()
        .expect("classify guarded unretained command-buffer test")
        .into_iter()
        .next()
        .expect("find guarded unretained command-buffer test");
    let work = parsed
        .call_sites(function.body.clone(), NEW_COMMAND_BUFFER_UNRETAINED, true)
        .expect("classify guarded unretained command-buffer call");
    parsed
        .validate_work_sites(
            &work,
            LOCAL_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        )
        .expect("guarded unretained command buffer satisfies the convention");
}

#[test]
fn cargo_metadata_discovers_nested_and_explicit_target_roots() {
    let fixture = tempfile::tempdir().expect("create Cargo target fixture");
    std::fs::create_dir_all(fixture.path().join("src/bin/nested"))
        .expect("create nested binary directory");
    std::fs::create_dir_all(fixture.path().join("tools"))
        .expect("create explicit target directory");
    std::fs::write(
        fixture.path().join("Cargo.toml"),
        r#"
[package]
name = "target-graph-fixture"
version = "0.0.0"
edition = "2024"

[[example]]
name = "explicit-example"
path = "tools/explicit_example.rs"
"#,
    )
    .expect("write fixture manifest");
    let nested = fixture.path().join("src/bin/nested/main.rs");
    let explicit = fixture.path().join("tools/explicit_example.rs");
    std::fs::write(&nested, "fn main() {}\n").expect("write nested binary");
    std::fs::write(&explicit, "fn main() {}\n").expect("write explicit example");

    let actual = cargo_target_roots(fixture.path(), &["bin", "example"])
        .expect("derive roots from Cargo metadata")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let expected = BTreeSet::from([
        std::fs::canonicalize(nested).expect("canonicalize nested target"),
        std::fs::canonicalize(explicit).expect("canonicalize explicit target"),
    ]);
    assert_eq!(
        actual, expected,
        "Cargo target roots did not match metadata"
    );
}

#[test]
fn cfg_attr_and_tokio_test_functions_are_included_in_raw_dispatch_inventory() {
    let source = r#"
#[cfg_attr(test, test)]
fn cfg_attr_registered_test() {
    let command_buffer = queue.new_command_buffer();
}

#[tokio::test]
async fn tokio_registered_test() {
    let command_buffer = queue.new_command_buffer();
}
"#;
    let parsed = StructuredSource::parse("fixtures/registered_tests.rs", source, true)
        .expect("parse registered test fixture");
    let discovered = parsed
        .test_functions()
        .expect("classify registered test functions")
        .into_iter()
        .filter(|function| {
            !parsed
                .call_sites(function.body.clone(), NEW_COMMAND_BUFFER, true)
                .expect("classify cfg_attr test work")
                .is_empty()
        })
        .map(|function| function.name.as_str())
        .collect::<BTreeSet<_>>();

    assert_eq!(
        discovered,
        BTreeSet::from(["cfg_attr_registered_test", "tokio_registered_test"])
    );
}

#[test]
fn const_expression_in_where_clause_does_not_hide_unguarded_dispatch() {
    let source = r#"
struct Queue;
impl Queue { fn new_command_buffer(&self) {} }
trait Marker<const N: usize> {}
impl Marker<1> for () {}

#[test]
fn raw_dispatch() where (): Marker<{ 1 }> {
    let queue = Queue;
    let _command_buffer = queue.new_command_buffer();
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/where_const_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn const_expression_in_return_type_does_not_hide_unguarded_dispatch() {
    let source = r#"
#[test]
fn raw_dispatch() -> Result<(), Marker<{ 1 }>> {
    let _command_buffer = queue.new_command_buffer();
    Ok(())
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "tests/return_const_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn labeled_loop_does_not_hide_unguarded_dispatch() {
    let source = r#"
#[test]
fn raw_dispatch() {
    'gpu: loop {
        let _command_buffer = queue.new_command_buffer();
        break 'gpu;
    }
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "tests/labeled_loop_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn closure_does_not_hide_unguarded_dispatch() {
    let source = r#"
#[test]
fn raw_dispatch() {
    let drive = || queue.new_command_buffer();
    drive();
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "tests/closure_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn unknown_test_registration_attribute_fails_closed() {
    let source = r#"
#[custom_runtime::test]
async fn raw_dispatch() {
    let queue = Queue;
    let _command_buffer = queue.new_command_buffer();
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "tests/custom_runtime_dispatch.rs",
        "unclassifiable test function attribute `custom_runtime::test`",
    );
}

#[test]
fn lint_level_attributes_are_not_test_registrations() {
    let source = r#"
#[expect(clippy::unwrap_used, reason = "the caller established the invariant")]
fn helper() -> u32 {
    Some(7u32).unwrap()
}

#[test]
#[deny(unused_must_use)]
#[expect(clippy::unwrap_used, reason = "the fixture owns the queue")]
fn raw_dispatch() {
    let _gpu_guard = gpu_test_lock();
    let _command_buffer = queue.new_command_buffer();
}
"#;
    let parsed = StructuredSource::parse("fixtures/lint_level_attributes.rs", source, true)
        .expect("parse lint-level attribute fixture");
    let functions = parsed
        .test_functions()
        .expect("classify lint-level attributes as non-registrations");
    let names: Vec<&str> = functions
        .iter()
        .map(|function| function.name.as_str())
        .collect();
    assert_eq!(
        names,
        ["raw_dispatch"],
        "only the #[test] function is a registration"
    );
    command_buffer_fixture_result(source, "fixtures/lint_level_attributes.rs")
        .expect("guarded work under lint-level attributes satisfies the convention");
}

#[test]
fn included_non_rust_source_in_test_scope_fails_closed() {
    let fixture = tempfile::tempdir().expect("create include fixture");
    let source_path = fixture.path().join("dispatch_test.rs");
    let included_path = fixture.path().join("included_dispatch.inc");
    let source = r#"
#[test]
fn raw_dispatch() {
    include!("included_dispatch.inc");
}
"#;
    std::fs::write(&source_path, source).expect("write include fixture source");
    std::fs::write(
        &included_path,
        "let _command_buffer = queue.new_command_buffer();\n",
    )
    .expect("write included non-Rust source");

    assert_command_buffer_fixture_rejected_with(
        &std::fs::read_to_string(&source_path).expect("read include fixture source"),
        "tests/dispatch_test.rs",
        "unclassifiable include! in test-bearing scope",
    );
}

#[test]
fn compiler_selected_non_rust_module_is_inside_the_test_boundary() {
    let fixture = tempfile::tempdir().expect("create non-Rust module fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create fixture source directory");
    std::fs::write(
        src.join("lib.rs"),
        "#[path = \"dispatch.inc\"]\nmod dispatch;\n",
    )
    .expect("write fixture crate root");
    std::fs::write(
        src.join("dispatch.inc"),
        r#"
#[test]
fn raw_dispatch() {
    let _command_buffer = queue.new_command_buffer();
}
"#,
    )
    .expect("write compiler-selected non-Rust module");

    let sources = parsed_module_closure(fixture.path(), &src.join("lib.rs"), true)
        .expect("parse compiler-selected module closure");
    let child = sources
        .iter()
        .find(|source| source.relative.ends_with("src/dispatch.inc"))
        .expect("non-Rust module belongs to the compiler-selected closure");
    let function = child
        .parsed
        .test_functions()
        .expect("classify non-Rust module tests")
        .into_iter()
        .find(|function| function.name == "raw_dispatch")
        .expect("find raw dispatch test");
    let work = child
        .parsed
        .call_sites(function.body.clone(), NEW_COMMAND_BUFFER, true)
        .expect("classify non-Rust module work");
    let error = child
        .parsed
        .validate_work_sites(
            &work,
            LOCAL_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        )
        .expect_err("unguarded non-Rust module work must fail");
    assert!(
        error.contains("no live shared-lock binding encloses this work"),
        "non-Rust module reported the wrong failure: {error}"
    );
}

fn assert_module_path_fixture(files: &[(&str, &str)]) {
    let fixture = tempfile::tempdir().expect("create module path fixture");
    for (relative, source) in files {
        let path = fixture.path().join(relative);
        std::fs::create_dir_all(path.parent().expect("fixture source parent"))
            .expect("create fixture source directory");
        std::fs::write(path, source).expect("write fixture source");
    }
    let sources = module_source_closure(fixture.path(), &fixture.path().join("src/lib.rs"), true)
        .expect("resolve compiler-selected module closure");
    let actual = sources
        .into_iter()
        .map(|source| source.path)
        .collect::<BTreeSet<_>>();
    let expected = files
        .iter()
        .map(|(relative, _)| {
            std::fs::canonicalize(fixture.path().join(relative))
                .expect("canonicalize fixture source")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn module_path_attribute_in_non_mod_rs_uses_source_directory() {
    assert_module_path_fixture(&[
        ("src/lib.rs", "mod a;"),
        ("src/a.rs", "#[path = \"a/t.rs\"] mod t;"),
        ("src/a/t.rs", ""),
    ]);
}

#[test]
fn module_path_attribute_in_crate_root_uses_source_directory() {
    assert_module_path_fixture(&[
        ("src/lib.rs", "#[path = \"nested/t.rs\"] mod t;"),
        ("src/nested/t.rs", ""),
    ]);
}

#[test]
fn module_path_attribute_in_mod_rs_uses_source_directory() {
    assert_module_path_fixture(&[
        ("src/lib.rs", "mod a;"),
        ("src/a/mod.rs", "#[path = \"t.rs\"] mod t;"),
        ("src/a/t.rs", ""),
    ]);
}

#[test]
fn module_path_attribute_file_children_use_containing_directory() {
    assert_module_path_fixture(&[
        (
            "src/lib.rs",
            "#[path = \"nested/alternate.rs\"] mod renamed;",
        ),
        ("src/nested/alternate.rs", "mod child;"),
        ("src/nested/child.rs", "mod grandchild;"),
        ("src/nested/child/grandchild.rs", ""),
    ]);
}

#[test]
fn module_path_attribute_inside_non_mod_rs_inline_uses_module_directory() {
    assert_module_path_fixture(&[
        ("src/lib.rs", "mod a;"),
        (
            "src/a.rs",
            "mod inline { #[path = \"other.rs\"] mod inner; }",
        ),
        ("src/a/inline/other.rs", ""),
    ]);
}

#[test]
fn module_path_attribute_inside_root_inline_uses_module_directory() {
    assert_module_path_fixture(&[
        (
            "src/lib.rs",
            "mod inline { #[path = \"other.rs\"] mod inner; }",
        ),
        ("src/inline/other.rs", ""),
    ]);
}

#[test]
fn module_path_attribute_file_inline_children_use_containing_directory() {
    assert_module_path_fixture(&[
        (
            "src/lib.rs",
            "#[path = \"nested/alternate.rs\"] mod renamed;",
        ),
        (
            "src/nested/alternate.rs",
            "mod inline { #[path = \"other.rs\"] mod inner; }",
        ),
        ("src/nested/inline/other.rs", ""),
    ]);
}

#[test]
fn cfg_attr_generated_module_path_fails_closed() {
    let fixture = tempfile::tempdir().expect("create cfg_attr path fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create fixture source directory");
    std::fs::write(
        src.join("lib.rs"),
        "#[cfg_attr(all(), path = \"dispatch.inc\")]\nmod dispatch;\n",
    )
    .expect("write fixture crate root");
    std::fs::write(src.join("dispatch.inc"), "").expect("write cfg_attr-selected module source");

    let error = module_source_closure(fixture.path(), &src.join("lib.rs"), true)
        .err()
        .expect("cfg_attr-generated path must fail closed");
    assert!(
        error.contains("unclassifiable cfg_attr-generated module path"),
        "cfg_attr module path reported the wrong failure: {error}"
    );
}

#[test]
fn forwarded_receiver_and_identifier_macro_fails_closed() {
    let source = r#"
macro_rules! raw_dispatch {
    ($receiver:expr, $method:ident) => {
        $receiver.$method()
    };
}

#[test]
fn forwarded_dispatch() {
    let _command_buffer = raw_dispatch!(queue, new_command_buffer);
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/forwarded_macro_dispatch.rs",
        "unclassifiable macro invocation `raw_dispatch!` in test-bearing scope",
    );
}

#[test]
fn guard_lifetime_rejects_an_immediate_drop_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    drop(gpu_guard);
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a guard dropped before Metal work");
}

#[test]
fn guard_lifetime_rejects_a_wildcard_that_drops_the_guard_immediately() {
    let source = r#"
fn measurement() {
    let _ = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a wildcard guard binding");
}

#[test]
fn guard_lifetime_rejects_a_cfg_elided_acquisition() {
    let source = r#"
fn measurement() {
    #[cfg(any())]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a cfg-elided guard acquisition");
}

#[test]
fn guard_lifetime_rejects_a_stacked_cfg_elided_acquisition() {
    let source = r#"
fn measurement() {
    #[cfg(any())]
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a stacked cfg-elided guard acquisition");
}

#[test]
fn guard_lifetime_rejects_an_active_cfg_attr_elided_acquisition() {
    let source = r#"
fn measurement() {
    #[cfg_attr(all(), cfg(any()))]
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "an active cfg_attr-elided guard acquisition");
}

#[test]
fn guard_lifetime_rejects_a_move_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let moved_guard = gpu_guard;
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a moved guard binding");
}

#[test]
fn guard_lifetime_rejects_shadowing_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let gpu_guard = ();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a shadowed guard binding");
}

#[test]
fn guard_lifetime_rejects_a_guard_in_a_nested_scope() {
    let source = r#"
fn measurement() {
    {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    }
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a guard dropped with a nested scope");
}

#[test]
fn guard_lifetime_rejects_an_enclosing_cfg_elided_function() {
    let source = r#"
#[cfg(any())]
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/enclosing_false_cfg.rs");
}

#[test]
fn guard_lifetime_rejects_an_enclosing_cfg_elided_module() {
    let source = r#"
#[cfg(any())]
mod hidden {
    fn measurement() {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
        let _device = Device::system_default();
    }
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/enclosing_false_module.rs");
}

#[test]
fn module_closure_excludes_an_out_of_line_cfg_elided_module() {
    let fixture = tempfile::tempdir().expect("create module-context fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create fixture source directory");
    std::fs::write(src.join("lib.rs"), "#[cfg(any())]\npub mod child;\n")
        .expect("write fixture module parent");
    let child = src.join("child.rs");
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    std::fs::write(&child, source).expect("write fixture module child");
    let sources = module_source_closure(fixture.path(), &src.join("lib.rs"), false)
        .expect("classify cfg-elided module closure");
    assert!(
        sources.iter().all(|source| source.path != child),
        "cfg-elided child entered the compiler-selected closure"
    );
}

#[test]
fn module_closure_excludes_an_out_of_line_module_in_a_cfg_elided_parent() {
    let fixture = tempfile::tempdir().expect("create nested module-context fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create nested fixture source directory");
    std::fs::write(
        src.join("lib.rs"),
        "#[cfg(any())]\nmod hidden {\n    #[path = \"child.rs\"]\n    pub mod child;\n}\n",
    )
    .expect("write nested fixture module parent");
    let child = src.join("child.rs");
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    std::fs::write(&child, source).expect("write nested fixture module child");
    let sources = module_source_closure(fixture.path(), &src.join("lib.rs"), false)
        .expect("classify nested cfg-elided module closure");
    assert!(
        sources.iter().all(|source| source.path != child),
        "child below a cfg-elided inline parent entered the compiler-selected closure"
    );
}

#[test]
fn guard_lifetime_rejects_an_enclosing_cfg_elided_block() {
    let source = r#"
fn measurement() {
    #[cfg(any())]
    unsafe {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
        let _device = Device::system_default();
    }
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/enclosing_false_block.rs");
}

#[test]
fn guard_lifetime_rejects_a_recursively_cfg_elided_function() {
    let source = r#"
#[cfg_attr(all(), cfg_attr(all(), cfg(any())))]
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/nested_false_cfg_attr.rs");
}

#[test]
fn guard_lifetime_checks_every_work_occurrence_after_a_false_gated_decoy() {
    let source = r#"
#[cfg(any())]
fn guarded_decoy() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}

fn live_work() {
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/false_gated_decoy.rs");
}

#[test]
fn guard_lifetime_ignores_a_comment_decoy_before_live_work() {
    let source = r#"
fn measurement() {
    {
        // Device::system_default()
    }
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_accepted(source, "fixtures/comment_decoy.rs");
}

#[test]
fn raw_dispatch_rejects_source_trivia_before_call_parentheses() {
    let source = r#"
#[test]
fn trivia_dispatch() {
    let command_buffer = queue.new_command_buffer /* comment */ ();
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/trivia_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn guard_lifetime_uses_braces_for_an_irregularly_indented_nested_scope() {
    let source = r#"
fn measurement() {
    {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
   }
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/irregular_nested_guard.rs");
}

#[test]
fn raw_dispatch_rejects_unclassifiable_macro_work() {
    let source = r#"
#[test]
fn macro_dispatch() {
    let gpu_guard = gpu_test_lock();
    dispatch!(queue.new_command_buffer());
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/macro_dispatch.rs",
        "unclassifiable macro invocation `dispatch!` in test-bearing scope",
    );
}

#[test]
fn raw_dispatch_rejects_macro_generated_test_work() {
    let source = r#"
macro_rules! raw_test {
    () => {
        #[test]
        fn generated() {
            let command_buffer = queue.new_command_buffer();
        }
    };
}
raw_test!();
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/macro_generated_test.rs",
        "unclassifiable macro invocation `raw_test!` in test-bearing scope",
    );
}

#[test]
fn guard_lifetime_rejects_stacked_cfg_separated_by_docs_and_blank_lines() {
    let source = r#"
fn measurement() {
    #[cfg(any())]

    /// The active Metal acquisition.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "stacked cfg with intervening documentation");
}

#[test]
fn guard_lifetime_rejects_a_raw_identifier_binding() {
    let source = r#"
fn measurement() {
    let r#gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a raw-identifier guard binding");
}

#[test]
fn guard_lifetime_rejects_a_let_else_derived_binding() {
    let source = r#"
fn measurement() {
    let Some(gpu_guard) = Some(lattice_inference::measurement::gpu_test_lock()) else {
        return;
    };
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a let-else-derived guard binding");
}

#[test]
fn guard_lifetime_rejects_an_if_let_derived_binding() {
    let source = r#"
fn measurement() {
    if let Some(gpu_guard) = Some(lattice_inference::measurement::gpu_test_lock()) {
        let _device = Device::system_default();
    }
}
"#;
    assert_guard_fixture_rejected(source, "an if-let-derived guard binding");
}

#[test]
fn guard_lifetime_rejects_a_match_derived_binding() {
    let source = r#"
fn measurement() {
    let gpu_guard = match () {
        () => lattice_inference::measurement::gpu_test_lock(),
    };
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a match-derived guard binding");
}
