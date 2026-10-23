use super::*;

struct GenerateWorkSyntax<'a> {
    receiver: &'a str,
    function_depth: usize,
    deferred: bool,
    sites: Vec<usize>,
    error: Option<String>,
}

impl<'ast> Visit<'ast> for GenerateWorkSyntax<'_> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "generate" {
            self.sites.push(call.method.span().byte_range().start);
            if self.deferred {
                self.error =
                    Some("deferred generate work is not covered by a lexical guard".into());
            } else if !matches!(&*call.receiver, syn::Expr::Path(path)
                if path.attrs.is_empty() && path.qself.is_none()
                    && path.path.is_ident(self.receiver))
            {
                self.error = Some("generate work must use the reviewed bare receiver".into());
            }
        }
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        let previous = self.deferred;
        self.deferred = true;
        syn::visit::visit_expr_closure(self, closure);
        self.deferred = previous;
    }

    fn visit_expr_async(&mut self, expression: &'ast syn::ExprAsync) {
        let previous = self.deferred;
        self.deferred = true;
        syn::visit::visit_expr_async(self, expression);
        self.deferred = previous;
    }

    fn visit_item(&mut self, item: &'ast syn::Item) {
        let previous = self.deferred;
        self.deferred |= self.function_depth > 0;
        syn::visit::visit_item(self, item);
        self.deferred = previous;
    }

    fn visit_item_fn(&mut self, function: &'ast ItemFn) {
        let previous = self.deferred;
        self.deferred |= function.sig.asyncness.is_some();
        self.function_depth += 1;
        syn::visit::visit_item_fn(self, function);
        self.function_depth -= 1;
        self.deferred = previous;
    }
}

fn validate_generate_work(
    sources: &[ParsedModuleSource],
    contract: &GenerateWorkTarget,
) -> Result<(), String> {
    let mut count = 0;
    for source in sources {
        let parsed = &source.parsed;
        let sites = parsed.call_sites(
            0..parsed.tokens.len(),
            CallSelector::Method("generate"),
            true,
        )?;
        if sites.is_empty() {
            continue;
        }
        if source.relative != contract.path {
            return Err(format!(
                "generate work moved to unreviewed source {}",
                source.relative
            ));
        }
        let syntax = syn::parse_file(&parsed.source).map_err(|reason| reason.to_string())?;
        let mut shape = GenerateWorkSyntax {
            receiver: contract.receiver,
            function_depth: 0,
            deferred: false,
            sites: Vec::new(),
            error: None,
        };
        shape.visit_file(&syntax);
        if let Some(error) = shape.error {
            return Err(error);
        }
        shape.sites.sort_unstable();
        let offsets = sites
            .iter()
            .map(|site| parsed.tokens[*site].offset)
            .collect::<Vec<_>>();
        if shape.sites != offsets {
            return Err("generate work has unsupported syntax outside parsed method calls".into());
        }
        for site in &sites {
            let owner = parsed.enclosing_function_path(*site);
            if owner != contract.owner {
                return Err(format!("generate work has unreviewed owner {owner}"));
            }
            if !parsed.formula_at(*site)?.satisfiable()? {
                return Err("generate work includes an inactive Metal call site".into());
            }
        }
        parsed.validate_work_sites(&sites, SHARED_LOCK_SELECTOR, GuardRequirement::Lexical)?;
        count += sites.len();
    }
    if count != contract.calls {
        return Err(format!(
            "generate work count changed: expected {}, found {count}",
            contract.calls
        ));
    }
    Ok(())
}

#[test]
fn reviewed_metal_generate_targets_use_live_lock_bindings() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let already_owned = RAW_HARNESS_ENTRYPOINTS
        .iter()
        .chain(ADDITIONAL_GUARDED_ENTRYPOINTS)
        .map(|(path, _)| *path)
        .chain(TARGETS_WITH_STATE_DRIVEN_WORK.iter().copied())
        .collect::<BTreeSet<_>>();
    let expected = GENERATE_WORK_TARGETS
        .iter()
        .map(|target| target.path)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        expected.len(),
        GENERATE_WORK_TARGETS.len(),
        "duplicate generate work target"
    );
    assert!(
        expected.is_disjoint(&already_owned),
        "generate work target already has an owner"
    );
    assert!(
        expected
            .iter()
            .all(|path| TARGETS_WITH_RECOGNIZED_METAL_MARKERS.contains(path))
    );
    let mut actual = BTreeSet::new();
    for path in cargo_target_roots(manifest_dir, &["example", "bin"])
        .unwrap_or_else(|reason| panic!("{reason}"))
    {
        let relative = path
            .strip_prefix(manifest_dir)
            .expect("source under manifest directory")
            .to_string_lossy();
        let Some(contract) = GENERATE_WORK_TARGETS
            .iter()
            .find(|target| target.path == relative.as_ref())
        else {
            continue;
        };
        actual.insert(contract.path);
        let sources = parsed_module_closure(manifest_dir, &path, false)
            .unwrap_or_else(|reason| panic!("{reason}"));
        validate_generate_work(&sources, contract)
            .unwrap_or_else(|reason| panic!("{}: {reason}", contract.path));
    }
    assert_eq!(
        actual, expected,
        "reviewed generate work target is no longer a Cargo root"
    );
}

const GENERATE_WORK_FIXTURE: GenerateWorkTarget = GenerateWorkTarget {
    path: "examples/generate_fixture.rs",
    owner: "run",
    receiver: "metal",
    calls: 2,
};
const GENERATE_WORK_SOURCE: &str = r#"
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();
    let mut metal = MetalQwen35State::from_q4_dir();
    metal.generate("warmup");
    for prompt in prompts {
        metal.generate(prompt);
    }
}
"#;

fn generate_work_fixture(source: &str) -> Result<(), String> {
    validate_generate_work(
        &[ParsedModuleSource {
            relative: GENERATE_WORK_FIXTURE.path.into(),
            parsed: StructuredSource::parse(GENERATE_WORK_FIXTURE.path, source, false)?,
        }],
        &GENERATE_WORK_FIXTURE,
    )
}

#[test]
fn generate_wrapper_accepts_live_lock_around_all_calls() {
    generate_work_fixture(GENERATE_WORK_SOURCE).expect("live guard covers warmup and loop work");
    let scoped = GENERATE_WORK_SOURCE
        .replace("    let _gpu_lock", "    { let _gpu_lock")
        .replace("    }\n}", "    } }\n}");
    generate_work_fixture(&scoped).expect("a lexical work scope need not span the whole function");
}

#[test]
fn generate_wrapper_rejects_constructor_only_lock_scope() {
    let source = GENERATE_WORK_SOURCE.replace(
        "    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n    let mut metal = MetalQwen35State::from_q4_dir();",
        "    let mut metal = {\n        let _gpu_lock = lattice_inference::measurement::gpu_test_lock();\n        MetalQwen35State::from_q4_dir()\n    };",
    );
    let parsed = StructuredSource::parse("constructor-only fixture", &source, false)
        .expect("valid source fixture");
    let construction = CallSelector::Path(&["MetalQwen35State", "from_q4_dir"]);
    assert_eq!(
        parsed
            .call_sites(0..parsed.tokens.len(), construction, true)
            .unwrap()
            .len(),
        1
    );
    parsed
        .validate_selector(
            0..parsed.tokens.len(),
            construction,
            SHARED_LOCK_SELECTOR,
            GuardRequirement::Lexical,
        )
        .expect("existing constructor-only check accepts a returned state");
    let error = generate_work_fixture(&source)
        .expect_err("the setup guard is gone before generation starts");
    assert!(
        error.contains("no live shared-lock binding"),
        "wrong failure: {error}"
    );
}

#[test]
fn generate_wrapper_rejects_unreviewed_or_deferred_work() {
    generate_work_fixture(GENERATE_WORK_SOURCE).expect("unchanged positive fixture");
    let cases = [
        (
            "closure",
            "metal.generate(\"warmup\");",
            "let later = || metal.generate(\"warmup\");",
            "deferred generate",
        ),
        (
            "async block",
            "metal.generate(\"warmup\");",
            "let later = async { metal.generate(\"warmup\"); };",
            "deferred generate",
        ),
        (
            "nested function",
            "metal.generate(\"warmup\");",
            "fn later(metal: &mut State) { metal.generate(\"warmup\"); }",
            "deferred generate",
        ),
        (
            "nested impl",
            "metal.generate(\"warmup\");",
            "struct Later; impl Later { fn call(metal: &mut State) { metal.generate(\"warmup\"); } }",
            "deferred generate",
        ),
        (
            "nested trait",
            "metal.generate(\"warmup\");",
            "trait Later { fn call(metal: &mut State) { metal.generate(\"warmup\"); } }",
            "deferred generate",
        ),
        (
            "async owner",
            "fn run()",
            "async fn run()",
            "deferred generate",
        ),
        (
            "receiver",
            "metal.generate(\"warmup\");",
            "other.generate(\"warmup\");",
            "reviewed bare receiver",
        ),
        (
            "indirect receiver",
            "metal.generate(\"warmup\");",
            "(&mut metal).generate(\"warmup\");",
            "reviewed bare receiver",
        ),
        ("owner", "fn run()", "fn other()", "unreviewed owner"),
        (
            "missing call",
            "metal.generate(\"warmup\");",
            "",
            "count changed",
        ),
        (
            "extra call",
            "metal.generate(\"warmup\");",
            "metal.generate(\"warmup\"); metal.generate(\"extra\");",
            "count changed",
        ),
        (
            "inactive call",
            "metal.generate(\"warmup\");",
            "#[cfg(any())] { metal.generate(\"warmup\"); }",
            "inactive Metal call site",
        ),
        (
            "opaque macro",
            "metal.generate(\"warmup\");",
            "opaque!(metal.generate(\"warmup\"));",
            "unclassifiable protected-work syntax",
        ),
        (
            "associated function",
            "metal.generate(\"warmup\");",
            "State::generate(&mut metal, \"warmup\");",
            "unclassifiable protected-work syntax",
        ),
        (
            "method value",
            "metal.generate(\"warmup\");",
            "let later = metal.generate;",
            "unclassifiable protected-work syntax",
        ),
    ];
    for (label, before, after, expected) in cases {
        assert_eq!(GENERATE_WORK_SOURCE.matches(before).count(), 1, "{label}");
        let source = GENERATE_WORK_SOURCE.replace(before, after);
        let error = generate_work_fixture(&source).expect_err(label);
        assert!(error.contains(expected), "{label}: wrong failure: {error}");
    }
    let source = ParsedModuleSource {
        relative: "examples/other.rs".into(),
        parsed: StructuredSource::parse("moved fixture", GENERATE_WORK_SOURCE, false).unwrap(),
    };
    let error = validate_generate_work(&[source], &GENERATE_WORK_FIXTURE)
        .expect_err("work moved to an unreviewed module");
    assert!(
        error.contains("unreviewed source"),
        "wrong failure: {error}"
    );
}
