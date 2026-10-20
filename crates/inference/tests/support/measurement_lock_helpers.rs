use super::*;

const MEASUREMENT_WORKER_SOURCE: &str = "src/serving_worker.rs";
const MEASUREMENT_WORKER_COMPAT_SOURCE: &str = "src/serve/metal_worker.rs";
const MEASUREMENT_WORKER_CALLER: &str = "examples/bench_serve_prepare.rs";
const MEASUREMENT_WORKER_ENTRY: &str =
    "lattice_inference::serve::metal_worker::bench_support::spawn_with_measurement_lock";

const WORKER_LOCK_FIXTURE: &str = r#"
#[cfg(feature = "bench-internals")]
mod bench_support {
    fn spawn_with_measurement_lock(factory: Factory, max_pending: usize, residency_limits: Limits) {
        let worker = crate::serving_worker::AffineWorker::spawn_with_guard(
            factory, max_pending, residency_limits,
            move || {
                let gpu_guard = crate::measurement::gpu_test_lock();
                let _ = started_tx.send(started);
                gpu_guard
            },
        )?;
    }
}
fn run_guarded_worker<G>(acquire_guard: impl FnOnce() -> G, work: impl FnOnce()) {
    let worker_guard = acquire_guard();
    work();
}
impl AffineWorker {
    fn spawn_with_guard<G>(factory: Factory, max_pending: usize, residency_limits: Limits,
        acquire_guard: impl FnOnce() -> G) {
        let join_handle = std::thread::spawn(move || {
            run_guarded_worker(acquire_guard, move || {
                match factory.build(worker_index, residency_limits) {
                    Ok(runtime) => run_worker_loop(runtime),
                    Err(error) => report(error),
                }
            });
        });
    }
}
"#;

const WORKER_CALLER_FIXTURE: &str = r#"
fn run_metal() {
    let worker = lattice_inference::serve::metal_worker::bench_support::spawn_with_measurement_lock(
        ServingFactory::qwen_metal(move || {
            let state = match format {
                Format::Safetensors => MetalQwen35State::new(weights, config, context),
                Format::Q4 => MetalQwen35State::from_q4_dir(dir, tokenizer, config, context),
            };
            Ok(state)
        }, VisionRuntime::unsupported()), 1, ResidencyLimits::default(),
    ).map_err(error)?;
}
"#;

fn combined_measurement_worker_source() -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let worker = std::fs::read_to_string(root.join(MEASUREMENT_WORKER_SOURCE))
        .expect("read serving worker source");
    let compatibility = std::fs::read_to_string(root.join(MEASUREMENT_WORKER_COMPAT_SOURCE))
        .expect("read worker compatibility source");
    // Inner doc comments are valid only at the top of a file, so the second
    // file's module docs are dropped before the two are parsed as one.
    let compatibility: String = compatibility
        .lines()
        .filter(|line| !line.starts_with("//!"))
        .map(|line| format!("{line}\n"))
        .collect();
    format!("{worker}\n{compatibility}")
}

#[test]
fn measurement_worker_lock_covers_factory_and_runtime() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let worker = combined_measurement_worker_source();
    let caller = std::fs::read_to_string(root.join(MEASUREMENT_WORKER_CALLER))
        .expect("read measurement worker caller");
    validate_measurement_worker(&worker).expect("measurement worker must retain its shared lock");
    validate_measurement_worker_caller(&caller)
        .expect("exempt constructors must execute through the measurement worker");
}

#[test]
fn measurement_worker_chain_accepts_named_guards() {
    validate_measurement_worker(WORKER_LOCK_FIXTURE).expect("accept direct worker lock chain");
    let renamed = WORKER_LOCK_FIXTURE
        .replace("gpu_guard", "retained_lock")
        .replace("worker_guard", "_held_until_return")
        .replace("work();", "/* retained across work */ work ( );");
    validate_measurement_worker(&renamed).expect("guard names and formatting are immaterial");
    validate_measurement_worker_caller(WORKER_CALLER_FIXTURE)
        .expect("accept constructors inside the guarded loader");
    let portable_stub = format!(
        "{WORKER_CALLER_FIXTURE}\n#[cfg(not(all(target_os = \"macos\", feature = \"metal-gpu\", feature = \"bench-internals\")))] fn run_metal() {{ skip(); }}"
    );
    validate_measurement_worker_caller(&portable_stub)
        .expect("inactive portable definitions do not replace the Metal caller");
}

fn reject_worker_fixture_mutations(mutations: &[(&str, &str)]) {
    for (before, after) in mutations {
        assert_eq!(WORKER_LOCK_FIXTURE.matches(before).count(), 1, "{before}");
        let mutant = WORKER_LOCK_FIXTURE.replacen(before, after, 1);
        syn::parse_file(&mutant).expect("worker mutation must remain valid Rust syntax");
        assert!(
            validate_measurement_worker(&mutant).is_err(),
            "accepted worker mutation: {after}"
        );
    }
}

#[test]
fn measurement_worker_chain_rejects_acquisition_mutations() {
    reject_worker_fixture_mutations(&[
        ("crate::measurement::gpu_test_lock()", "()"),
        ("let gpu_guard =", "let _ ="),
        ("                gpu_guard\n", "                ()\n"),
        (
            "                gpu_guard\n",
            "                drop(gpu_guard); ()\n",
        ),
        ("let gpu_guard =", "#[cfg(test)] let gpu_guard ="),
        (
            "crate::measurement::gpu_test_lock()",
            "(|| crate::measurement::gpu_test_lock())()",
        ),
        (
            "crate::measurement::gpu_test_lock()",
            "acquire!(crate::measurement::gpu_test_lock())",
        ),
        ("#[cfg(feature = \"bench-internals\")]", "#[cfg(test)]"),
    ]);
}

#[test]
fn measurement_worker_chain_rejects_thread_handoff_mutations() {
    reject_worker_fixture_mutations(&[
        (
            "run_guarded_worker(acquire_guard, move ||",
            "run_guarded_worker(|| (), move ||",
        ),
        (
            "run_guarded_worker(acquire_guard, move ||",
            "run_guarded_worker(acquire_guard(), move ||",
        ),
        (
            "let join_handle = std::thread::spawn(move || {",
            "let ignored = acquire_guard(); let join_handle = std::thread::spawn(move || {",
        ),
        (
            "let join_handle = std::thread::spawn(move || {",
            "let factory = factory.build(worker_index, residency_limits); let join_handle = std::thread::spawn(move || {",
        ),
        (
            "let join_handle = std::thread::spawn(move || {",
            "let join_handle = (move || {",
        ),
        (
            "run_guarded_worker(acquire_guard, move || {",
            "let deferred = move || run_guarded_worker(acquire_guard, move || {",
        ),
        (
            "match factory.build(worker_index, residency_limits)",
            "match factory",
        ),
        (
            "acquire_guard: impl FnOnce() -> G)",
            "ignored_guard: impl FnOnce() -> G)",
        ),
    ]);
}

#[test]
fn measurement_worker_chain_rejects_guard_lifetime_mutations() {
    reject_worker_fixture_mutations(&[
        (
            "let worker_guard = acquire_guard();",
            "let _ = acquire_guard();",
        ),
        ("    work();", "    drop(worker_guard); work();"),
        (
            "    let worker_guard = acquire_guard();\n    work();",
            "    work();\n    let worker_guard = acquire_guard();",
        ),
        (
            "let worker_guard = acquire_guard();",
            "{ let worker_guard = acquire_guard(); }",
        ),
        ("    work();", "    let worker_guard = (); work();"),
        ("    work();", "    let later = || work();"),
    ]);
}

#[test]
fn measurement_worker_caller_requires_guarded_loader_route() {
    for (before, after) in [
        (MEASUREMENT_WORKER_ENTRY, "MetalWorker::spawn"),
        ("ServingFactory::qwen_metal", "ServingFactory::other"),
        ("move || {", "move || { let deferred = move || {"),
        (
            "MetalQwen35State::new(weights, config, context)",
            "(|| MetalQwen35State::new(weights, config, context))()",
        ),
        (
            "MetalQwen35State::from_q4_dir(dir, tokenizer, config, context)",
            "other_state()",
        ),
        ("fn run_metal()", "#[cfg(test)] fn run_metal()"),
    ] {
        assert_eq!(WORKER_CALLER_FIXTURE.matches(before).count(), 1, "{before}");
        let mut mutant = WORKER_CALLER_FIXTURE.replacen(before, after, 1);
        if after == "move || { let deferred = move || {" {
            mutant = mutant.replace("}, VisionRuntime", "}; deferred }, VisionRuntime");
        }
        syn::parse_file(&mutant).expect("caller mutation must remain valid Rust syntax");
        assert!(
            validate_measurement_worker_caller(&mutant).is_err(),
            "accepted caller mutation: {after}"
        );
    }
}

#[test]
fn measurement_worker_contract_rejects_real_source_lock_removal() {
    use syn::spanned::Spanned;
    let source = combined_measurement_worker_source();
    validate_measurement_worker(&source).expect("unmodified worker must satisfy the contract");
    let parsed = StructuredSource::parse(MEASUREMENT_WORKER_SOURCE, &source, false).unwrap();
    let syntax = syn::parse_file(&source).unwrap();
    let entry = worker_function(
        &parsed,
        &syntax,
        "bench_support::spawn_with_measurement_lock",
    )
    .unwrap();
    let spawn = worker_top_call(
        entry,
        "crate::serving_worker::AffineWorker::spawn_with_guard",
    )
    .unwrap();
    let acquisition = worker_closure(&spawn.args[3]).unwrap();
    let syn::Stmt::Local(local) = &acquisition.stmts[0] else {
        panic!("guard binding missing")
    };
    let range = local.init.as_ref().unwrap().expr.span().byte_range();
    let mut mutant = source.clone();
    mutant.replace_range(range, "()");
    syn::parse_file(&mutant).expect("lock removal must remain valid Rust syntax");
    assert!(
        validate_measurement_worker(&mutant).is_err(),
        "removing the actual lock must fail"
    );
}

#[test]
fn measurement_worker_contract_rejects_real_source_early_drop() {
    use syn::spanned::Spanned;
    let source = combined_measurement_worker_source();
    validate_measurement_worker(&source).expect("unmodified worker must satisfy the contract");
    let parsed = StructuredSource::parse(MEASUREMENT_WORKER_SOURCE, &source, false).unwrap();
    let syntax = syn::parse_file(&source).unwrap();
    let guarded = worker_function(&parsed, &syntax, "run_guarded_worker").unwrap();
    let syn::Stmt::Local(local) = &guarded.stmts[0] else {
        panic!("guard binding missing")
    };
    let guard = worker_binding(local).unwrap();
    let offset = guarded.stmts.last().unwrap().span().byte_range().start;
    let mut mutant = source.clone();
    mutant.insert_str(offset, &format!("drop({guard});\n"));
    syn::parse_file(&mutant).expect("early drop must remain valid Rust syntax");
    assert!(
        validate_measurement_worker(&mutant).is_err(),
        "dropping the actual guard before work must fail"
    );
}

#[test]
fn measurement_worker_contract_rejects_real_source_caller_thread_execution() {
    use syn::spanned::Spanned;
    let source = combined_measurement_worker_source();
    validate_measurement_worker(&source).expect("unmodified worker must satisfy the contract");
    let parsed = StructuredSource::parse(MEASUREMENT_WORKER_SOURCE, &source, false).unwrap();
    let syntax = syn::parse_file(&source).unwrap();
    let worker = worker_function(&parsed, &syntax, "AffineWorker::spawn_with_guard").unwrap();
    let spawn = worker_top_call(worker, "std::thread::spawn").unwrap();
    let mut mutant = source.clone();
    mutant.replace_range(spawn.func.span().byte_range(), "(|work| work())");
    syn::parse_file(&mutant).expect("caller thread execution must remain valid Rust syntax");
    assert!(
        validate_measurement_worker(&mutant).is_err(),
        "executing the actual worker entry on the caller thread must fail"
    );
}

fn worker_measurement_cfg() -> CfgFormula {
    CfgFormula::all([
        CfgFormula::Atom("feature=serve".into()),
        CfgFormula::Atom("feature=bench-internals".into()),
    ])
}

fn worker_function<'a>(
    parsed: &StructuredSource,
    syntax: &'a syn::File,
    name: &str,
) -> Result<&'a syn::Block, String> {
    #[derive(Default)]
    struct Functions<'a>(Vec<(&'a syn::Signature, &'a syn::Block)>);
    impl<'ast> Visit<'ast> for Functions<'ast> {
        fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
            self.0.push((&item.sig, &item.block));
        }
        fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
            self.0.push((&item.sig, &item.block));
        }
    }
    let mut functions = Functions::default();
    functions.visit_file(syntax);
    let mut selected = Vec::new();
    for (signature, block) in functions.0 {
        let range = parsed.token_range_for_block(block, name)?;
        if parsed.enclosing_function_path(range.start) == name {
            let formula = parsed.formula_at(range.start)?;
            if !CfgFormula::all([worker_measurement_cfg(), formula.clone()]).satisfiable()? {
                continue;
            }
            if signature.asyncness.is_some() || !worker_measurement_cfg().implies(&formula)? {
                return Err(format!(
                    "{name} must be synchronous and cover measurement builds"
                ));
            }
            let parameters: &[&str] = match name {
                "bench_support::spawn_with_measurement_lock" => {
                    &["factory", "max_pending", "residency_limits"]
                }
                "AffineWorker::spawn_with_guard" => &[
                    "factory",
                    "max_pending",
                    "residency_limits",
                    "acquire_guard",
                ],
                "run_guarded_worker" => &["acquire_guard", "work"],
                _ => &[],
            };
            if !parameters.is_empty()
                && (signature.inputs.len() != parameters.len()
                    || !signature
                        .inputs
                        .iter()
                        .zip(parameters)
                        .all(|(argument, expected)| {
                            matches!(argument, syn::FnArg::Typed(argument)
                            if matches!(&*argument.pat, syn::Pat::Ident(binding)
                                if binding.ident == *expected && binding.by_ref.is_none()
                                    && binding.mutability.is_none() && binding.subpat.is_none()))
                        }))
            {
                return Err(format!("{name} must forward its named parameters"));
            }
            selected.push(block);
        }
    }
    match selected.as_slice() {
        [block] => Ok(block),
        _ => Err(format!("expected one definition of {name}")),
    }
}

fn worker_path(expression: &syn::Expr, expected: &str) -> bool {
    matches!(expression, syn::Expr::Path(path)
        if path.attrs.is_empty() && path.qself.is_none()
            && path_label(&path.path) == expected
            && path.path.segments.iter().all(|segment|
                matches!(segment.arguments, syn::PathArguments::None)))
}

fn worker_call<'a>(expression: &'a syn::Expr, path: &str) -> Option<&'a syn::ExprCall> {
    match expression {
        syn::Expr::Call(call) if call.attrs.is_empty() && worker_path(&call.func, path) => {
            Some(call)
        }
        syn::Expr::Try(expression) if expression.attrs.is_empty() => {
            worker_call(&expression.expr, path)
        }
        syn::Expr::MethodCall(call)
            if call.attrs.is_empty() && call.method == "map_err" && call.args.len() == 1 =>
        {
            worker_call(&call.receiver, path)
        }
        _ => None,
    }
}

fn worker_top_call<'a>(block: &'a syn::Block, path: &str) -> Result<&'a syn::ExprCall, String> {
    let calls = block
        .stmts
        .iter()
        .filter_map(|statement| match statement {
            syn::Stmt::Local(local) if local.attrs.is_empty() => local
                .init
                .as_ref()
                .filter(|init| init.diverge.is_none())
                .and_then(|init| worker_call(&init.expr, path)),
            syn::Stmt::Expr(expression, _) => worker_call(expression, path),
            _ => None,
        })
        .collect::<Vec<_>>();
    match calls.as_slice() {
        [call] => Ok(call),
        _ => Err(format!("expected one immediate call to {path}")),
    }
}

fn worker_closure(expression: &syn::Expr) -> Result<&syn::Block, String> {
    if let syn::Expr::Closure(closure) = expression
        && closure.attrs.is_empty()
        && closure.asyncness.is_none()
        && closure.capture.is_some()
        && closure.inputs.is_empty()
        && let syn::Expr::Block(block) = &*closure.body
        && block.attrs.is_empty()
        && block.label.is_none()
    {
        return Ok(&block.block);
    }
    Err("worker boundary must be an immediate synchronous move closure".into())
}

fn worker_binding(local: &syn::Local) -> Result<&syn::Ident, String> {
    if local.attrs.is_empty()
        && let syn::Pat::Ident(binding) = &local.pat
        && binding.by_ref.is_none()
        && binding.mutability.is_none()
        && binding.subpat.is_none()
    {
        return Ok(&binding.ident);
    }
    Err("worker guard must have a plain named binding".into())
}

fn worker_name_usage(
    block: &syn::Block,
    name: &str,
    bindings: usize,
    references: usize,
) -> Result<(), String> {
    struct Uses<'a> {
        name: &'a str,
        bindings: usize,
        references: usize,
        opaque: bool,
    }
    impl<'ast> Visit<'ast> for Uses<'_> {
        fn visit_pat_ident(&mut self, binding: &'ast syn::PatIdent) {
            self.bindings += usize::from(binding.ident == self.name);
            syn::visit::visit_pat_ident(self, binding);
        }
        fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
            self.references += usize::from(path.path.is_ident(self.name));
            syn::visit::visit_expr_path(self, path);
        }
        fn visit_item(&mut self, _: &'ast syn::Item) {
            self.opaque = true;
        }
        fn visit_macro(&mut self, _: &'ast syn::Macro) {
            self.opaque = true;
        }
    }
    let mut uses = Uses {
        name,
        bindings: 0,
        references: 0,
        opaque: false,
    };
    uses.visit_block(block);
    if uses.opaque || (uses.bindings, uses.references) != (bindings, references) {
        return Err(format!(
            "worker binding {name} is shadowed, moved, or used outside its boundary"
        ));
    }
    Ok(())
}

fn validate_measurement_worker(source: &str) -> Result<(), String> {
    let parsed = StructuredSource::parse(MEASUREMENT_WORKER_SOURCE, source, false)?;
    let syntax = syn::parse_file(source).map_err(|reason| reason.to_string())?;
    let entry = worker_function(
        &parsed,
        &syntax,
        "bench_support::spawn_with_measurement_lock",
    )?;
    let spawn = worker_top_call(
        entry,
        "crate::serving_worker::AffineWorker::spawn_with_guard",
    )?;
    if spawn.args.len() != 4
        || !spawn
            .args
            .iter()
            .take(3)
            .zip(["factory", "max_pending", "residency_limits"])
            .all(|(argument, name)| worker_path(argument, name))
    {
        return Err("measurement helper must forward its factory and worker limits".into());
    }
    worker_name_usage(entry, "factory", 0, 1)?;
    let acquisition = worker_closure(&spawn.args[3])?;
    let Some(syn::Stmt::Local(local)) = acquisition.stmts.first() else {
        return Err("measurement callback must acquire the shared lock first".into());
    };
    let guard = worker_binding(local)?.to_string();
    let init = local
        .init
        .as_ref()
        .ok_or("measurement guard has no initializer")?;
    if init.diverge.is_some()
        || !matches!(&*init.expr, syn::Expr::Call(call)
            if call.attrs.is_empty() && call.args.is_empty()
                && worker_path(&call.func, "crate::measurement::gpu_test_lock"))
        || !matches!(acquisition.stmts.last(), Some(syn::Stmt::Expr(expression, None))
            if worker_path(expression, &guard))
    {
        return Err("measurement callback must return the acquired shared lock".into());
    }
    worker_name_usage(acquisition, &guard, 1, 1)?;

    let worker = worker_function(&parsed, &syntax, "AffineWorker::spawn_with_guard")?;
    worker_name_usage(worker, "acquire_guard", 0, 1)?;
    worker_name_usage(worker, "factory", 0, 1)?;
    let thread = worker_top_call(worker, "std::thread::spawn")?;
    if thread.args.len() != 1 {
        return Err("worker thread must have one entry closure".into());
    }
    let thread_body = worker_closure(&thread.args[0])?;
    let [syn::Stmt::Expr(syn::Expr::Call(guarded), _)] = thread_body.stmts.as_slice() else {
        return Err("worker thread must enter the guarded worker directly".into());
    };
    if !guarded.attrs.is_empty()
        || !worker_path(&guarded.func, "run_guarded_worker")
        || guarded.args.len() != 2
        || !worker_path(&guarded.args[0], "acquire_guard")
    {
        return Err("worker thread must forward the acquisition callback unchanged".into());
    }
    let work = worker_closure(&guarded.args[1])?;
    let [syn::Stmt::Expr(syn::Expr::Match(construction), _)] = work.stmts.as_slice() else {
        return Err("worker factory must be built inside the guarded work closure".into());
    };
    if !construction.attrs.is_empty()
        || !matches!(&*construction.expr, syn::Expr::MethodCall(call)
            if call.attrs.is_empty() && call.method == "build"
                && worker_path(&call.receiver, "factory"))
    {
        return Err("guarded work must construct the worker factory immediately".into());
    }

    let guarded = worker_function(&parsed, &syntax, "run_guarded_worker")?;
    worker_name_usage(guarded, "acquire_guard", 0, 1)?;
    worker_name_usage(guarded, "work", 0, 1)?;
    if !matches!(guarded.stmts.last(), Some(syn::Stmt::Expr(syn::Expr::Call(call), _))
        if call.attrs.is_empty() && call.args.is_empty() && worker_path(&call.func, "work"))
    {
        return Err("guarded worker must invoke its work directly".into());
    }
    let body = parsed.token_range_for_block(guarded, "run_guarded_worker")?;
    let sites = parsed.call_sites(body.clone(), CallSelector::Path(&["work"]), true)?;
    let guards = parsed.guard_bindings(body.clone(), CallSelector::Path(&["acquire_guard"]))?;
    let ([site], [guard]) = (sites.as_slice(), guards.as_slice()) else {
        return Err("guarded worker must acquire one guard and invoke its work once".into());
    };
    if !guard.protected.contains(site)
        || guard.protected.end != body.end
        || guard.used_after_binding
        || !worker_measurement_cfg().implies(&guard.formula)?
    {
        return Err("worker guard must remain live until the guarded function returns".into());
    }
    Ok(())
}

fn validate_measurement_worker_caller(source: &str) -> Result<(), String> {
    let parsed = StructuredSource::parse(MEASUREMENT_WORKER_CALLER, source, false)?;
    let syntax = syn::parse_file(source).map_err(|reason| reason.to_string())?;
    let caller = worker_function(&parsed, &syntax, "run_metal")?;
    let spawn = worker_top_call(caller, MEASUREMENT_WORKER_ENTRY)?;
    if spawn.args.len() != 3 {
        return Err("measurement caller must pass a factory and worker limits".into());
    }
    let Some(factory) = worker_call(&spawn.args[0], "ServingFactory::qwen_metal") else {
        return Err("measurement caller must pass its factory directly".into());
    };
    if factory.args.len() != 2 {
        return Err("measurement factory must have its loader and vision runtime".into());
    }
    let loader = worker_closure(&factory.args[0])?;
    let loader_range = parsed.token_range_for_block(loader, "measurement loader")?;
    let caller_range = parsed.token_range_for_block(caller, "run_metal")?;
    struct Deferred(Vec<Range<usize>>);
    impl<'ast> Visit<'ast> for Deferred {
        fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
            use syn::spanned::Spanned;
            self.0.push(closure.span().byte_range());
        }
        fn visit_expr_async(&mut self, expression: &'ast syn::ExprAsync) {
            use syn::spanned::Spanned;
            self.0.push(expression.span().byte_range());
        }
        fn visit_item(&mut self, item: &'ast syn::Item) {
            use syn::spanned::Spanned;
            self.0.push(item.span().byte_range());
        }
    }
    let mut deferred = Deferred(Vec::new());
    deferred.visit_block(loader);
    for selector in [
        CallSelector::Path(&["MetalQwen35State", "new"]),
        CallSelector::Path(&["MetalQwen35State", "from_q4_dir"]),
    ] {
        let sites = parsed.call_sites(caller_range.clone(), selector, true)?;
        let [site] = sites.as_slice() else {
            return Err(format!(
                "measurement loader must contain one {}",
                selector.label()
            ));
        };
        if !loader_range.contains(site)
            || deferred
                .0
                .iter()
                .any(|range| range.contains(&parsed.tokens[*site].offset))
            || !worker_measurement_cfg().implies(&parsed.formula_at(*site)?)?
        {
            return Err(format!(
                "{} must execute inside the measurement loader",
                selector.label()
            ));
        }
    }
    Ok(())
}

fn assert_helper_mediated_test_holds_function_lifetime_lock(relative: &str, function_name: &str) {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(manifest_dir.join(relative))
        .expect("read helper-mediated Metal tests");
    let parsed = StructuredSource::parse(relative, &source, true)
        .unwrap_or_else(|reason| panic!("{reason}"));
    let function = parsed
        .test_functions()
        .unwrap_or_else(|reason| panic!("{reason}"))
        .into_iter()
        .find(|function| function.name == function_name)
        .unwrap_or_else(|| panic!("checked helper-mediated test `{function_name}` was not found"));
    parsed
        .validate_selector(
            function.body.clone(),
            DEVICE_SYSTEM_DEFAULT,
            LOCAL_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        )
        .unwrap_or_else(|reason| panic!("{relative}::{function_name}: {reason}"));
}

#[test]
fn mtp_draft_logit_equivalence_test_holds_a_function_lifetime_lock() {
    assert_helper_mediated_test_holds_function_lifetime_lock(
        "src/forward/metal_qwen35/inner/tests/decode_attention.rs",
        "mtp_draft_logit_equivalence_with_quarot_counter_rotation",
    );
}

#[test]
fn metal_engine_session_isolation_test_holds_a_function_lifetime_lock() {
    assert_helper_mediated_test_holds_function_lifetime_lock(
        "src/forward/metal_qwen35/inner/tests/mtp_verify.rs",
        "test_metal_qwen35_engine_session_isolation",
    );
}

/// Library modules that hold a binary's own production model loader.
///
/// The library target is scanned for test functions only, so a loader moved out
/// of a binary into the library would leave the construction inventory. Each
/// row adds the module as an extra root of that binary's scan; the binary's
/// own source stays scanned.
const BIN_LIBRARY_LOADER_ROOTS: &[(&str, &str, &str)] = &[(
    "bin",
    "lattice_serve",
    "src/serving_provider/providers/qwen/worker.rs",
)];

/// The parsed sources the construction scan reads for one Cargo target.
pub(super) fn construction_scan_sources(
    manifest_dir: &Path,
    target: &CargoTargetRoot,
) -> Vec<ParsedModuleSource> {
    let test_cfg = matches!(target.kind.as_str(), "test" | "lib");
    let mut sources = parsed_module_closure(manifest_dir, &target.path, test_cfg)
        .unwrap_or_else(|reason| panic!("{reason}"));
    for (kind, name, root) in BIN_LIBRARY_LOADER_ROOTS {
        if target.kind == *kind && target.name == *name {
            sources.extend(
                parsed_module_closure(manifest_dir, Path::new(root), false)
                    .unwrap_or_else(|reason| panic!("{reason}")),
            );
        }
    }
    sources
}
