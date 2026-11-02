use super::*;

const BENCH_GPU_SOURCE: &str = include_str!("../../examples/bench_gpu.rs");
const GUARD: &str = "    #[cfg(target_os = \"macos\")]\n    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n";

fn validate_arm(source: &str, name: &str) -> Result<(), String> {
    let syntax = syn::parse_file(source).map_err(|error| error.to_string())?;
    let function = syntax
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .ok_or_else(|| format!("missing {name} function"))?;
    let guards = function.block.stmts.iter().filter_map(|statement| match statement {
        syn::Stmt::Local(local) if matches!(&local.pat, syn::Pat::Ident(pattern) if pattern.ident == "_gpu_lock") => Some(local),
        _ => None,
    }).collect::<Vec<_>>();
    let [guard] = guards.as_slice() else {
        return Err(format!("{name} must bind one _gpu_lock"));
    };
    let [attribute] = guard.attrs.as_slice() else {
        return Err(format!("{name} guard must have only the macOS cfg"));
    };
    if !attribute.path().is_ident("cfg") {
        return Err(format!("{name} guard must have only the macOS cfg"));
    }
    let cfg = attribute
        .parse_args::<syn::Meta>()
        .map_err(|error| error.to_string())?;
    let syn::Meta::NameValue(cfg) = cfg else {
        return Err(format!("{name} guard must have only the macOS cfg"));
    };
    if !cfg.path.is_ident("target_os")
        || !matches!(&cfg.value, syn::Expr::Lit(value) if matches!(&value.lit, syn::Lit::Str(os) if os.value() == "macos"))
    {
        return Err(format!("{name} guard must have only the macOS cfg"));
    }

    let parsed = StructuredSource::parse("examples/bench_gpu.rs", source, false)?;
    let functions = parsed
        .functions
        .iter()
        .filter(|function| function.name == name)
        .collect::<Vec<_>>();
    let [function] = functions.as_slice() else {
        return Err(format!("expected exactly one {name} function"));
    };
    let guards = parsed.guard_bindings(function.body.clone(), SHARED_LOCK_SELECTOR)?;
    if guards.len() != 1 {
        return Err(format!("{name} must hold exactly one shared lock"));
    }
    let mut selectors = vec![CallSelector::Path(&["GpuModelState", "new"])];
    if name == "run_bench" {
        selectors.push(CallSelector::Path(&["QwenModel", "from_directory"]));
    }
    for selector in selectors {
        let sites = parsed.call_sites(function.body.clone(), selector, true)?;
        if sites.len() != 1 {
            return Err(format!("{name}: expected one {} call", selector.label()));
        }
        parsed.validate_work_sites(
            &sites,
            SHARED_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        )?;
    }
    Ok(())
}

#[test]
fn both_wgpu_benchmark_arms_hold_the_shared_lock() {
    for arm in ["run_bench", "run_gpu_only"] {
        validate_arm(BENCH_GPU_SOURCE, arm).unwrap_or_else(|reason| panic!("{arm}: {reason}"));
    }
}

#[test]
fn wgpu_guard_scope_fixture_is_valid() {
    let source = r#"
#[cfg(feature = "wgpu-gpu")]
fn run_gpu_only() {
    #[cfg(target_os = "macos")]
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();
    let model = GpuModelState::new();
}
"#;
    validate_arm(source, "run_gpu_only").expect("macOS guard encloses GPU initialization");
}

#[test]
fn removing_either_wgpu_guard_fails_only_that_arm() {
    for (arm, control) in [("run_bench", "run_gpu_only"), ("run_gpu_only", "run_bench")] {
        let signature = format!("fn {arm}() {{");
        assert_eq!(BENCH_GPU_SOURCE.matches(&signature).count(), 1);
        let start = BENCH_GPU_SOURCE.find(&signature).expect("arm exists");
        let offset = BENCH_GPU_SOURCE[start..].find(GUARD).expect("guard exists");
        let mut mutant = BENCH_GPU_SOURCE.to_owned();
        mutant.replace_range(start + offset..start + offset + GUARD.len(), "");
        let error = validate_arm(&mutant, arm).expect_err("guard removal must fail");
        assert!(
            error.contains("must bind one _gpu_lock"),
            "wrong failure: {error}"
        );
        validate_arm(&mutant, control)
            .unwrap_or_else(|reason| panic!("positive control {control}: {reason}"));
    }
}

#[test]
fn metal_only_guard_cfg_is_rejected_for_each_wgpu_arm() {
    for (arm, control) in [("run_bench", "run_gpu_only"), ("run_gpu_only", "run_bench")] {
        let signature = format!("fn {arm}() {{");
        assert_eq!(BENCH_GPU_SOURCE.matches(&signature).count(), 1);
        let start = BENCH_GPU_SOURCE.find(&signature).expect("arm exists");
        let offset = BENCH_GPU_SOURCE[start..].find(GUARD).expect("guard exists");
        let mut mutant = BENCH_GPU_SOURCE.to_owned();
        let restricted = GUARD.replace(
            "#[cfg(target_os = \"macos\")]",
            "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]",
        );
        mutant.replace_range(start + offset..start + offset + GUARD.len(), &restricted);
        let error = validate_arm(&mutant, arm).expect_err("Metal-only guard must fail");
        assert!(
            error.contains("only the macOS cfg"),
            "wrong failure: {error}"
        );
        validate_arm(&mutant, control)
            .unwrap_or_else(|reason| panic!("positive control {control}: {reason}"));
    }
}
