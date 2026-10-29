use super::*;

#[test]
pub(super) fn raw_metal_measurement_harnesses_use_live_lock_bindings_across_the_entrypoint() {
    let packages = scanned_packages();
    let mut actual = BTreeSet::new();
    let reviewed = TARGETS_WITH_RECOGNIZED_METAL_MARKERS
        .iter()
        .filter(|(_, path)| path.starts_with("benches/"))
        .map(|(package, path)| package_path_key(package, path))
        .collect::<BTreeSet<_>>();
    let covered = BENCH_HARNESSES.iter().map(BenchHarness::key).collect();
    validate_bench_harness_inventory(&reviewed, &covered)
        .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        BENCH_HARNESSES
            .iter()
            .all(|contract| !contract.calls.is_empty() && contract.registered.len() == 1),
        "every reviewed Metal benchmark needs one complete target-list wrapper"
    );
    assert_eq!(
        BENCH_HARNESSES
            .iter()
            .map(BenchHarness::key)
            .collect::<BTreeSet<_>>()
            .len(),
        BENCH_HARNESSES.len(),
        "state harness ownership records must not repeat paths"
    );
    let mut state_actual = BTreeSet::new();
    let mut violations = Vec::new();

    for package in &packages {
        for path in package
            .target_roots(&["bench", "example", "bin"])
            .unwrap_or_else(|reason| panic!("{reason}"))
        {
            let local = package
                .relative(&path)
                .unwrap_or_else(|reason| panic!("{reason}"));
            let relative = package.key(&local);
            let sources = parsed_package_module_closure(package, &path, false)
                .unwrap_or_else(|reason| panic!("{reason}"));
            if let Some(contract) = BENCH_HARNESSES
                .iter()
                .find(|contract| contract.package == package.identity && contract.path == local)
            {
                state_actual.insert(relative.clone());
                if let Err(reason) = validate_bench_harness_sources(&sources, contract) {
                    violations.push(format!("{relative}: {reason}"));
                }
            }
            let mut has_raw_gpu_work = false;
            for source in &sources {
                for selector in RAW_GPU_SELECTORS {
                    has_raw_gpu_work |= source
                        .parsed
                        .has_selector(*selector)
                        .unwrap_or_else(|reason| panic!("{reason}"));
                }
            }
            if !has_raw_gpu_work {
                continue;
            }
            actual.insert(relative.clone());

            let protected_entrypoint = RAW_HARNESS_ENTRYPOINTS
                .iter()
                .find_map(|(owner, path, marker)| {
                    (*owner == package.identity && *path == local).then_some(*marker)
                })
                .unwrap_or_else(|| panic!("raw Metal harness {relative} is not classified"));
            let root = sources
                .iter()
                .find(|source| source.key() == relative)
                .unwrap_or_else(|| panic!("target root {relative} was not parsed"));
            root.parsed
                .validate_selector(
                    0..root.parsed.tokens.len(),
                    protected_entrypoint,
                    SHARED_LOCK_SELECTOR,
                    GuardRequirement::Lexical,
                )
                .unwrap_or_else(|reason| panic!("{reason}"));
        }
    }

    let expected = RAW_HARNESS_ENTRYPOINTS
        .iter()
        .map(|(package, path, _)| package_path_key(package, path))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual, expected,
        "raw Metal harness inventory changed; classify every added or removed path explicitly"
    );
    assert_eq!(
        state_actual,
        BENCH_HARNESSES.iter().map(BenchHarness::key).collect(),
        "state harness ownership records must name current Cargo targets"
    );
    assert!(
        violations.is_empty(),
        "state Metal measurement ownership violations:\n{}",
        violations.join("\n")
    );
}

const STATE_HARNESS_FIXTURE: BenchHarness = BenchHarness {
    package: ".",
    path: "fixtures/state_harness.rs",
    registered: &["bench_locked"],
    calls: &[
        BenchHarnessCall {
            selector: CallSelector::Path(&["measure"]),
            callers: &["bench_locked"],
            guarded: true,
        },
        BenchHarnessCall {
            selector: CallSelector::Path(&["load"]),
            callers: &["measure"],
            guarded: false,
        },
    ],
    constructions: &[("load", CallSelector::Path(&["MetalQwen35State", "new"]))],
};

const STATE_HARNESS_SOURCE: &str = r#"
fn load() -> MetalQwen35State { MetalQwen35State::new() }
fn measure() { let mut state = load(); state.forward_step(); }
fn bench_locked() {
    let guard = lattice_inference::measurement::gpu_test_lock();
    measure();
}
criterion_group!(benches, bench_locked);
criterion_main!(benches);
"#;

fn state_harness_fixture(source: &str, contract: &BenchHarness) -> Result<(), String> {
    let parsed = StructuredSource::parse(contract.path, source, false)?;
    validate_bench_harness(&parsed, contract)
}

#[test]
fn state_harness_rejects_setup_only_guard_with_registered_wrapper_intact() {
    state_harness_fixture(STATE_HARNESS_SOURCE, &STATE_HARNESS_FIXTURE)
        .expect("caller owns returned state lifetime");
    let source = STATE_HARNESS_SOURCE
        .replace("    let guard = lattice_inference::measurement::gpu_test_lock();\n", "")
        .replace("fn load() -> MetalQwen35State {", "fn load() -> MetalQwen35State { let guard = lattice_inference::measurement::gpu_test_lock();");
    let error = state_harness_fixture(&source, &STATE_HARNESS_FIXTURE)
        .expect_err("setup-only guard must fail");
    assert!(
        error.contains("no live shared-lock binding"),
        "wrong failure: {error}"
    );
}

#[test]
fn state_harness_rejects_unreviewed_callers_aliases_and_deferred_work() {
    for replacement in [
        "fn extra() { let state = load(); }",
        "fn extra() { let alias = load; alias(); }",
        "use other::function as load;",
        "fn extra(load: fn()) { load(); }",
    ] {
        let source = format!("{STATE_HARNESS_SOURCE}\n{replacement}");
        assert!(
            state_harness_fixture(&source, &STATE_HARNESS_FIXTURE).is_err(),
            "accepted {replacement}"
        );
    }
    for replacement in [
        "let delayed = || measure(); delayed();",
        "let future = async { measure(); };",
        "let measure = other; measure();",
        "fn measure() {} measure();",
        "const measure: fn() = other; measure();",
        "static measure: fn() = other; measure();",
        "struct measure(); measure();",
    ] {
        let source = STATE_HARNESS_SOURCE.replace("    measure();", replacement);
        assert!(
            state_harness_fixture(&source, &STATE_HARNESS_FIXTURE).is_err(),
            "accepted {replacement}"
        );
    }
    let asynchronous = STATE_HARNESS_SOURCE.replace("fn measure()", "async fn measure()");
    assert!(state_harness_fixture(&asynchronous, &STATE_HARNESS_FIXTURE).is_err());
}

#[test]
fn state_harness_rejects_extra_or_inactive_criterion_registration() {
    for registration in [
        "criterion_group!(benches, bench_locked, measure);",
        "#[cfg(any())] criterion_group!(benches, bench_locked);",
        "mod inactive { criterion_group!(benches, bench_locked); }",
        "criterion_group!(other, bench_locked);",
    ] {
        let source =
            STATE_HARNESS_SOURCE.replace("criterion_group!(benches, bench_locked);", registration);
        assert!(
            state_harness_fixture(&source, &STATE_HARNESS_FIXTURE).is_err(),
            "accepted {registration}"
        );
    }
}

#[test]
fn state_harness_rejects_guard_decoys_and_nested_reacquisition() {
    for guard in [
        "#[cfg(any())] let guard = lattice_inference::measurement::gpu_test_lock();",
        "let guard = lattice_inference::measurement::gpu_test_lock(); drop(guard);",
        "{ let guard = lattice_inference::measurement::gpu_test_lock(); }",
    ] {
        let source = STATE_HARNESS_SOURCE.replace(
            "let guard = lattice_inference::measurement::gpu_test_lock();",
            guard,
        );
        assert!(
            state_harness_fixture(&source, &STATE_HARNESS_FIXTURE).is_err(),
            "accepted {guard}"
        );
    }
    let nested = STATE_HARNESS_SOURCE.replace("fn load() -> MetalQwen35State {", "fn load() -> MetalQwen35State { let guard = lattice_inference::measurement::gpu_test_lock();");
    let error = state_harness_fixture(&nested, &STATE_HARNESS_FIXTURE)
        .expect_err("nested acquisition must fail");
    assert!(
        error.contains("reacquired by a callee"),
        "wrong failure: {error}"
    );
}

#[test]
fn state_harness_accepts_completed_results_but_rejects_setup_scope_escape() {
    let contract = BenchHarness {
        package: ".",
        path: "fixtures/complete_operation.rs",
        registered: &["run"],
        calls: &[],
        constructions: &[("run", CallSelector::Path(&["MetalQwen35State", "new"]))],
    };
    for output in ["", " -> Vec<Metric>"] {
        let tail = if output.is_empty() { "" } else { "vec![]" };
        let source = format!(
            r#"
fn run(){output} {{
    let guard = lattice_inference::measurement::gpu_test_lock();
    let mut state = MetalQwen35State::new();
    state.forward_step();
    {tail}
}}
criterion_group!(benches, run);
criterion_main!(benches);
"#
        );
        state_harness_fixture(&source, &contract)
            .expect("complete operation may return measurements");
    }
    let source = r#"
fn run() {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let mut state;
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))] {
        let guard = lattice_inference::measurement::gpu_test_lock();
        state = MetalQwen35State::new();
    }
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))] {
        state.chat_completion();
    }
}
criterion_group!(benches, run);
criterion_main!(benches);
"#;
    let error = state_harness_fixture(source, &contract).expect_err("state escaped setup guard");
    assert!(
        error.contains("no live shared-lock binding"),
        "wrong failure: {error}"
    );
    let complete_block = source.replace(
        "    }\n    #[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))] {\n        state.chat_completion();",
        "        state.chat_completion();",
    ).replace("    #[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]\n    let mut state;\n", "")
        .replace("state = MetalQwen35State::new();", "let mut state = MetalQwen35State::new();");
    state_harness_fixture(&complete_block, &contract)
        .expect("whole active block owns measurement lifetime");
}

#[test]
fn state_harness_inventory_rejects_omitted_stale_and_new_paths() {
    let reviewed = BTreeSet::from(["benches/known.rs".to_string()]);
    validate_bench_harness_inventory(&reviewed, &reviewed).expect("exact coverage");
    for covered in [
        BTreeSet::new(),
        BTreeSet::from(["benches/stale.rs".to_string()]),
        BTreeSet::from(["benches/known.rs".to_string(), "benches/new.rs".to_string()]),
    ] {
        assert!(validate_bench_harness_inventory(&reviewed, &covered).is_err());
    }
}

const MULTI_ARM_HARNESS_FIXTURE: BenchHarness = BenchHarness {
    package: ".",
    path: "fixtures/whole_target_list.rs",
    registered: &["bench_locked"],
    calls: &[
        BenchHarnessCall {
            selector: CallSelector::Path(&["measure"]),
            callers: &["bench_locked"],
            guarded: true,
        },
        BenchHarnessCall {
            selector: CallSelector::Path(&["cpu"]),
            callers: &["bench_locked"],
            guarded: true,
        },
        BenchHarnessCall {
            selector: CallSelector::Path(&["load"]),
            callers: &["measure"],
            guarded: false,
        },
    ],
    constructions: &[("load", CallSelector::Path(&["MetalQwen35State", "new"]))],
};

const COMPLETE_TARGET_LIST: &str =
    "let guard = lattice_inference::measurement::gpu_test_lock(); measure(); cpu();";

fn multi_arm_harness_source(body: &str) -> String {
    format!(
        r#"
fn load() -> MetalQwen35State {{ MetalQwen35State::new() }}
fn measure() {{ let mut state = load(); state.forward_step(); }}
fn cpu() {{ work(); }}
fn bench_locked() {{ {body} }}
criterion_group!(benches, bench_locked);
criterion_main!(benches);
"#
    )
}

#[test]
fn benchmark_target_list_requires_one_guard_for_every_ordered_arm() {
    let source = multi_arm_harness_source(COMPLETE_TARGET_LIST);
    state_harness_fixture(&source, &MULTI_ARM_HARNESS_FIXTURE)
        .expect("one span covers GPU and CPU arms");
    let separate_guards = multi_arm_harness_source(
        "{ let first = lattice_inference::measurement::gpu_test_lock(); measure(); }
         { let second = lattice_inference::measurement::gpu_test_lock(); cpu(); }",
    );
    let error = state_harness_fixture(&separate_guards, &MULTI_ARM_HARNESS_FIXTURE)
        .expect_err("individual arm guards cannot certify one continuous span");
    assert!(
        error.contains("immediate reviewed arms"),
        "wrong failure: {error}"
    );
    for arms in [
        "cpu(); measure();",
        "measure();",
        "measure(); cpu(); cpu();",
        "measure(); #[cfg(feature = \"extra\")] cpu();",
    ] {
        let body = format!("let guard = lattice_inference::measurement::gpu_test_lock(); {arms}");
        assert!(
            state_harness_fixture(&multi_arm_harness_source(&body), &MULTI_ARM_HARNESS_FIXTURE)
                .is_err(),
            "accepted {arms}"
        );
    }
    let cpu_outside = multi_arm_harness_source(
        "{ let guard = lattice_inference::measurement::gpu_test_lock(); measure(); } cpu();",
    );
    assert!(state_harness_fixture(&cpu_outside, &MULTI_ARM_HARNESS_FIXTURE).is_err());
    let registered_sibling = source.replace(
        "criterion_group!(benches, bench_locked);",
        "criterion_group!(benches, bench_locked, cpu);",
    );
    let error = state_harness_fixture(&registered_sibling, &MULTI_ARM_HARNESS_FIXTURE)
        .expect_err("unprotected CPU sibling registration");
    assert!(
        error.contains("Criterion must register exactly"),
        "wrong failure: {error}"
    );
    let asynchronous = source.replace("fn bench_locked()", "async fn bench_locked()");
    assert!(state_harness_fixture(&asynchronous, &MULTI_ARM_HARNESS_FIXTURE).is_err());
}

#[test]
fn benchmark_target_list_checks_every_active_wrapper_definition() {
    let source = multi_arm_harness_source(COMPLETE_TARGET_LIST);
    let wrapper = format!("fn bench_locked() {{ {COMPLETE_TARGET_LIST} }}");
    let variants = |other: &str| {
        format!(
            r#"
#[cfg(feature = "variant")]
fn bench_locked() {{ {COMPLETE_TARGET_LIST} }}
#[cfg(not(feature = "variant"))]
fn bench_locked() {{ {other} }}
"#
        )
    };
    let valid = source.replace(&wrapper, &variants(COMPLETE_TARGET_LIST));
    state_harness_fixture(&valid, &MULTI_ARM_HARNESS_FIXTURE)
        .expect("both active alternatives have complete spans");
    let invalid = source.replace(
        &wrapper,
        &variants(
            "{ let first = lattice_inference::measurement::gpu_test_lock(); measure(); }
         { let second = lattice_inference::measurement::gpu_test_lock(); cpu(); }",
        ),
    );
    assert!(
        state_harness_fixture(&invalid, &MULTI_ARM_HARNESS_FIXTURE).is_err(),
        "one correct cfg variant cannot certify the other"
    );
}

#[test]
fn benchmark_leaf_globs_cannot_shadow_reviewed_caller_edges() {
    let source = multi_arm_harness_source(COMPLETE_TARGET_LIST);
    let leaf = source.replace("fn cpu() {", "fn cpu() { use metal::*;");
    state_harness_fixture(&leaf, &MULTI_ARM_HARNESS_FIXTURE)
        .expect("leaf import does not affect the wrapper's lexical scope");
    for declaration in ["fn bench_locked() {", "fn measure() {"] {
        let changed = source.replace(declaration, &format!("{declaration} use metal::*;"));
        let error = state_harness_fixture(&changed, &MULTI_ARM_HARNESS_FIXTURE)
            .expect_err("glob in caller scope");
        assert!(
            error.contains("glob import can shadow"),
            "wrong failure: {error}"
        );
    }
    assert!(
        state_harness_fixture(
            &format!("use metal::*;\n{source}"),
            &MULTI_ARM_HARNESS_FIXTURE
        )
        .is_err()
    );
    let extra = format!("{source}\nfn extra() {{ use metal::*; load(); }}");
    let error = state_harness_fixture(&extra, &MULTI_ARM_HARNESS_FIXTURE)
        .expect_err("glob does not admit an extra caller");
    assert!(
        error.contains("unreviewed caller extra"),
        "wrong failure: {error}"
    );
}

#[test]
fn benchmark_arm_arguments_cannot_hide_control_flow_or_replace_parameters() {
    let source = multi_arm_harness_source(
        "let guard = lattice_inference::measurement::gpu_test_lock(); measure(c); cpu(c);",
    )
    .replace("fn bench_locked()", "fn bench_locked(c: &mut Criterion)")
    .replace("fn measure()", "fn measure(c: &mut Criterion)")
    .replace("fn cpu()", "fn cpu(c: &mut Criterion)");
    state_harness_fixture(&source, &MULTI_ARM_HARNESS_FIXTURE)
        .expect("the original Criterion parameter reaches both arms unchanged");
    for argument in ["{ return; }", "other", "&mut *c"] {
        let changed = source.replace("cpu(c);", &format!("cpu({argument});"));
        let error = state_harness_fixture(&changed, &MULTI_ARM_HARNESS_FIXTURE)
            .expect_err("arm argument must not change control flow or the passed value");
        assert!(
            error.contains("unchanged bare wrapper parameters"),
            "wrong failure: {error}"
        );
    }
}

/// Matches every Cargo-selected executable target against recognized source markers.
///
/// This is a pre-expansion lexical inventory. It does not classify work whose
/// protected identifiers appear only after macro expansion or name resolution.
#[test]
pub(super) fn cargo_target_lexical_metal_marker_inventory_is_explicit() {
    let packages = scanned_packages();
    let mut discovered = BTreeSet::new();
    let mut all_targets = BTreeSet::new();
    for package in &packages {
        for path in package
            .target_roots(&["bench", "example", "bin"])
            .unwrap_or_else(|reason| panic!("{reason}"))
        {
            let relative = package.key(
                &package
                    .relative(&path)
                    .unwrap_or_else(|reason| panic!("{reason}")),
            );
            all_targets.insert(relative.clone());
            let sources = parsed_package_module_closure(package, &path, false)
                .unwrap_or_else(|reason| panic!("{reason}"));
            if sources.iter().any(|source| {
                source
                    .parsed
                    .has_top_level_metal_marker()
                    .unwrap_or_else(|reason| panic!("{reason}"))
            }) {
                discovered.insert(relative);
            }
        }
    }

    let reviewed = TARGETS_WITH_RECOGNIZED_METAL_MARKERS
        .iter()
        .map(|(package, path)| package_path_key(package, path))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        discovered, reviewed,
        "top-level Metal entrypoint inventory changed; classify direct, helper-mediated, and \
         alternate-state-family paths explicitly"
    );
    let reviewed_non_metal = TARGETS_WITHOUT_RECOGNIZED_METAL_MARKERS
        .iter()
        .map(|(package, path)| package_path_key(package, path))
        .collect::<BTreeSet<_>>();
    assert!(
        reviewed.is_disjoint(&reviewed_non_metal),
        "top-level targets must have exactly one classification"
    );
    let all_reviewed = reviewed
        .union(&reviewed_non_metal)
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        all_targets, all_reviewed,
        "top-level target inventory changed; every new bench, example, or binary must be \
         classified before it can pass the contract"
    );
}

#[test]
pub(super) fn alternate_and_helper_mediated_entrypoints_hold_the_shared_lock() {
    let packages = scanned_packages();
    for (owner, relative, entrypoint) in ADDITIONAL_GUARDED_ENTRYPOINTS {
        let package = measurement_lock_helpers::require_package(&packages, owner);
        let key = package.key(relative);
        let source = std::fs::read_to_string(package.manifest_dir.join(relative))
            .unwrap_or_else(|reason| panic!("{key}: {reason}"));
        let parsed = StructuredSource::parse(&key, &source, false)
            .unwrap_or_else(|reason| panic!("{reason}"));
        parsed
            .validate_selector(
                0..parsed.tokens.len(),
                *entrypoint,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )
            .unwrap_or_else(|reason| panic!("{reason}"));
    }
}

/// Closes the `examples/`+`src/bin/` half of #1527 the same way #1524 closed
/// `benches/`: a target is admitted into scrutiny by what it actually calls
/// (`STATE_WORK_SELECTORS`), not by whether it happens to also use a raw
/// selector, so `state.forward_prefill` / `state.forward_step` can no longer
/// be silently skipped just because no raw Metal API call appears at the
/// same call site. The population is the explicit `TARGETS_WITH_STATE_DRIVEN_WORK`
/// list, asserted equal to the lexical scan below.
#[test]
pub(super) fn state_driven_metal_measurement_targets_use_live_lock_bindings() {
    let packages = scanned_packages();
    let already_owned = RAW_HARNESS_ENTRYPOINTS
        .iter()
        .chain(ADDITIONAL_GUARDED_ENTRYPOINTS)
        .map(|(package, path, _)| package_path_key(package, path))
        .collect::<BTreeSet<_>>();
    let expected = TARGETS_WITH_STATE_DRIVEN_WORK
        .iter()
        .map(|(package, path)| package_path_key(package, path))
        .collect::<BTreeSet<_>>();
    assert!(
        expected.is_disjoint(&already_owned),
        "TARGETS_WITH_STATE_DRIVEN_WORK duplicates a target RAW_HARNESS_ENTRYPOINTS or \
         ADDITIONAL_GUARDED_ENTRYPOINTS already owns"
    );
    assert_eq!(
        expected.len(),
        TARGETS_WITH_STATE_DRIVEN_WORK.len(),
        "TARGETS_WITH_STATE_DRIVEN_WORK must not repeat a path"
    );

    let mut actual = BTreeSet::new();
    let mut violations = Vec::new();
    for package in &packages {
        for path in package
            .target_roots(&["example", "bin"])
            .unwrap_or_else(|reason| panic!("{reason}"))
        {
            let local = package
                .relative(&path)
                .unwrap_or_else(|reason| panic!("{reason}"));
            let relative = package.key(&local);
            if already_owned.contains(&relative) {
                continue;
            }
            let sources = parsed_package_module_closure(package, &path, false)
                .unwrap_or_else(|reason| panic!("{reason}"));

            let mut has_state_work = false;
            for source in &sources {
                for selector in STATE_WORK_SELECTORS {
                    has_state_work |= source
                        .parsed
                        .has_selector(*selector)
                        .unwrap_or_else(|reason| panic!("{reason}"));
                }
            }
            if !has_state_work {
                continue;
            }
            actual.insert(relative.clone());

            let helper_names = STATE_WORK_HELPER_ENTRYPOINTS
                .iter()
                .filter(|(owner, path, _)| *owner == package.identity && *path == local)
                .map(|(_, _, selector)| *selector)
                .collect::<Vec<_>>();

            for source in &sources {
                let all_sites = STATE_WORK_SELECTORS
                    .iter()
                    .flat_map(|selector| {
                        source
                            .parsed
                            .call_sites(0..source.parsed.tokens.len(), *selector, true)
                            .unwrap_or_else(|reason| panic!("{reason}"))
                    })
                    .collect::<Vec<_>>();

                let mut direct_sites = Vec::new();
                let mut mediating_helpers = BTreeSet::new();
                for site in all_sites {
                    let owner = source.parsed.enclosing_function_path(site);
                    match helper_names
                        .iter()
                        .find(|selector| selector.final_name() == owner)
                    {
                        Some(_) => {
                            mediating_helpers.insert(owner);
                        }
                        None => direct_sites.push(site),
                    }
                }

                if !direct_sites.is_empty()
                    && let Err(reason) = source.parsed.validate_work_sites(
                        &direct_sites,
                        SHARED_LOCK_SELECTOR,
                        GuardRequirement::Lexical,
                    )
                {
                    violations.push(format!("{relative} ({}): {reason}", source.key()));
                }
                for helper_name in mediating_helpers {
                    let selector = *helper_names
                        .iter()
                        .find(|selector| selector.final_name() == helper_name)
                        .expect("helper name resolved from helper_names");
                    if let Err(reason) = source.parsed.validate_selector(
                        0..source.parsed.tokens.len(),
                        selector,
                        SHARED_LOCK_SELECTOR,
                        GuardRequirement::Lexical,
                    ) {
                        violations.push(format!(
                            "{relative} ({}): helper `{helper_name}` mediating STATE_WORK_SELECTORS \
                             work: {reason}",
                            source.key()
                        ));
                    }
                }
            }
        }
    }

    assert_eq!(
        actual, expected,
        "state-driven Metal measurement inventory changed; classify every added or removed \
         example/bin target explicitly in TARGETS_WITH_STATE_DRIVEN_WORK (#1527)"
    );
    assert!(
        violations.is_empty(),
        "state-driven Metal measurement targets without a live shared-lock binding:\n{}",
        violations.join("\n")
    );
}
