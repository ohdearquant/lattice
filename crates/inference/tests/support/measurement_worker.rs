use super::*;
use syn::spanned::Spanned;

pub(super) const TARGET: &str = "examples/bench_serve_prepare.rs";
const WORKER: &str = "src/serve/metal_worker.rs";
const FACTORY: &str = "src/serving_factory.rs";
const SPAWN: &str =
    "lattice_inference::serve::metal_worker::bench_support::spawn_with_measurement_lock";

fn normalized(source: &str) -> Result<String, String> {
    source
        .parse::<proc_macro2::TokenStream>()
        .map(|tokens| tokens.to_string())
        .map_err(|error| error.to_string())
}

fn same_syntax(source: &str, span: proc_macro2::Span, expected: &str) -> Result<bool, String> {
    Ok(normalized(&source[span.byte_range()])? == normalized(expected)?)
}

fn body<'a>(parsed: &'a StructuredSource, name: &str) -> Result<&'a str, String> {
    let functions = parsed
        .function_paths
        .iter()
        .filter(|(_, path)| path == name)
        .collect::<Vec<_>>();
    if functions.len() != 1 {
        return Err(format!("expected exactly one worker boundary {name}"));
    }
    let range = &functions[0].0;
    let start = parsed.tokens[range.start - 1].offset;
    let end = parsed.tokens[range.end].offset + 1;
    Ok(&parsed.source[start..end])
}

fn validate_worker_source(source: &str) -> Result<(), String> {
    let parsed = StructuredSource::parse(WORKER, source, false)?;
    for (name, expected) in [
        (
            "run_guarded_worker",
            "{ let _measurement_guard = acquire_guard(); work(); }",
        ),
        (
            "bench_support::spawn_with_measurement_lock",
            "{
            let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
            let (owner, client, metadata, preparation) = super::MetalWorker::spawn_with_guard(
                factory, max_pending, residency_limits,
                move || {
                    let guard = crate::measurement::gpu_test_lock();
                    let _ = started_tx.send(std::time::Instant::now());
                    guard
                },
            )?;
            let started = started_rx.recv().map_err(|_| super::StartupError::ThreadExited)?;
            Ok((owner, client, metadata, preparation, started.elapsed()))
        }",
        ),
        (
            "MetalWorker::spawn_with_vision",
            "{
            Self::spawn_with_guard(factory, max_pending, residency_limits, || ())
        }",
        ),
    ] {
        if normalized(body(&parsed, name)?)? != normalized(expected)? {
            return Err(format!("worker lifetime boundary changed: {name}"));
        }
    }
    let spawn_body = body(&parsed, "MetalWorker::spawn_with_guard")?;
    let block = syn::parse_str::<syn::Block>(spawn_body).map_err(|error| error.to_string())?;
    let mut boundary_names = BoundaryNames { error: None };
    boundary_names.visit_block(&block);
    if let Some(error) = boundary_names.error {
        return Err(error);
    }
    let expected = "std::thread::spawn(move || {
        run_guarded_worker(acquire_guard, move || { match factory.build(worker_index, residency_limits) {
            Ok((runtime, metadata, preparation)) => {
                let vision_supported = runtime.vision_supported();
                let _ = ready_tx.send(Ok((metadata, preparation, vision_supported)));
                let runtime = RefCell::new(runtime);
                run_worker_loop_with_lora(job_rx,
                    |messages, cfg, lora, stream, on_token, should_cancel| {
                        runtime.borrow_mut().generate(messages, cfg, lora, stream, on_token, should_cancel,)
                    },
                    |command| runtime.borrow_mut().control(command),
                );
            }
            Err(error) => { let _ = ready_tx.send(Err(error)); }
        }});
    })";
    let mut thread_bindings = 0;
    for statement in &block.stmts {
        if let syn::Stmt::Local(local) = statement
            && let syn::Pat::Ident(binding) = &local.pat
            && binding.ident == "join_handle"
        {
            thread_bindings += 1;
            let init = local
                .init
                .as_ref()
                .ok_or("worker thread has no initializer")?;
            if !local.attrs.is_empty()
                || init.diverge.is_some()
                || !same_syntax(spawn_body, init.expr.span(), expected)?
            {
                return Err(
                    "worker thread no longer owns the guard around the entire runtime".into(),
                );
            }
        }
    }
    if thread_bindings != 1 {
        return Err("worker thread initializer is missing or duplicated".into());
    }
    // The capability is only transferred to the actual worker closure.
    let references = rust_tokens(spawn_body)?
        .iter()
        .filter(|token| token.ident() == Some("acquire_guard"))
        .count();
    if references != 1 {
        return Err("worker guard capability has an unreviewed use".into());
    }
    Ok(())
}

fn validate_factory_source(source: &str) -> Result<(), String> {
    let parsed = StructuredSource::parse(FACTORY, source, false)?;
    for (name, expected) in [
        ("ServingFactory::qwen_metal", "{ Self { inner: Box::new(QwenFactory { loader: Box::new(loader), vision, }), } }"),
        ("ServingFactory::build", "{
            fn require_send_sync<T: Send + Sync>(_: &T) {}
            let (runtime, metadata, preparation) = self.inner.build(index, limits)?;
            require_send_sync(&preparation);
            Ok((runtime, metadata, preparation))
        }"),
        ("QwenFactory::build", "{
            let Self { loader, vision } = *self;
            let (state, tokenizer, metadata) = loader()?;
            let tokenizer = serving_tokenizer(tokenizer, metadata.model_max_context);
            let tokenizer = Arc::new(tokenizer);
            let preparation = PreparationHandle::qwen(Arc::clone(&tokenizer), metadata.model_max_context);
            let runtime = QwenMetalRuntime::new(state, tokenizer, vision, metadata.clone(), index, limits);
            Ok((Box::new(runtime), metadata, preparation))
        }"),
    ] {
        if normalized(body(&parsed, name)?)? != normalized(expected)? {
            return Err(format!("worker factory lifetime boundary changed: {name}"));
        }
    }
    Ok(())
}

fn validate_factory_namespace(
    parsed: &StructuredSource,
    syntax: &syn::File,
    owner: &ItemFn,
    factory_path: &syn::Path,
) -> Result<(), String> {
    let mut definitions = Vec::new();
    for statement in &owner.block.stmts {
        if let syn::Stmt::Item(syn::Item::Use(import)) = statement
            && same_syntax(
                &parsed.source,
                import.span(),
                "use lattice_inference::serving_factory::ServingFactory;",
            )?
        {
            definitions.extend(
                parsed
                    .tokens
                    .iter()
                    .filter(|token| {
                        import.span().byte_range().contains(&token.offset)
                            && token.ident() == Some("ServingFactory")
                    })
                    .map(|token| token.offset),
            );
        }
    }
    if definitions.len() != 1 {
        return Err("worker factory requires its exact canonical import".into());
    }
    definitions.push(factory_path.segments[0].ident.span().byte_range().start);
    let actual = parsed
        .tokens
        .iter()
        .filter(|token| token.ident() == Some("ServingFactory"))
        .map(|token| token.offset)
        .collect::<BTreeSet<_>>();
    if actual != definitions.into_iter().collect() {
        return Err("worker factory has a shadow, alias, or unreviewed use".into());
    }
    #[derive(Default)]
    struct Imports<'a>(Vec<&'a syn::ItemUse>);
    impl<'a> Visit<'a> for Imports<'a> {
        fn visit_item_use(&mut self, import: &'a syn::ItemUse) {
            self.0.push(import);
        }
    }
    fn has_glob(tree: &syn::UseTree) -> bool {
        match tree {
            syn::UseTree::Glob(_) => true,
            syn::UseTree::Path(path) => has_glob(&path.tree),
            syn::UseTree::Group(group) => group.items.iter().any(has_glob),
            _ => false,
        }
    }
    let mut imports = Imports::default();
    imports.visit_file(syntax);
    for import in imports.0 {
        let index = parsed
            .tokens
            .iter()
            .position(|token| token.offset == import.use_token.span.byte_range().start)
            .ok_or("unmapped worker import")?;
        if has_glob(&import.tree) && parsed.formula_at(index)?.satisfiable()? {
            return Err("an active glob can replace the worker factory namespace".into());
        }
    }
    for (index, token) in parsed.tokens.iter().enumerate() {
        if token.ident() == Some("lattice_inference")
            && !(parsed
                .tokens
                .get(index + 1)
                .is_some_and(|token| token.is_punct(':'))
                && parsed
                    .tokens
                    .get(index + 2)
                    .is_some_and(|token| token.is_punct(':')))
        {
            return Err("worker crate namespace has a local binding".into());
        }
    }
    Ok(())
}

struct BoundaryNames {
    error: Option<String>,
}

impl<'ast> Visit<'ast> for BoundaryNames {
    fn visit_item(&mut self, _: &'ast syn::Item) {
        self.error = Some("nested worker items could shadow a reviewed lifetime boundary".into());
    }

    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        if matches!(
            pattern.ident.to_string().as_str(),
            "run_guarded_worker" | "acquire_guard"
        ) {
            self.error = Some("worker lifetime boundary was rebound".into());
        }
        syn::visit::visit_pat_ident(self, pattern);
    }
}

struct BenchmarkWork {
    spawns: Vec<syn::ExprCall>,
    constructions: Vec<syn::ExprCall>,
    submissions: Vec<syn::ExprMethodCall>,
    client_paths: Vec<proc_macro2::Span>,
    client_bindings: usize,
    constructor_paths: usize,
    closure_depth: usize,
    async_depth: usize,
    function_depth: usize,
    error: Option<String>,
}

impl<'ast> Visit<'ast> for BenchmarkWork {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*call.func {
            let label = path_label(&path.path);
            if label.ends_with("spawn_with_measurement_lock")
                || label.ends_with("spawn_with_vision")
            {
                self.spawns.push(call.clone());
            }
            if matches!(
                label.as_str(),
                "MetalQwen35State::new" | "MetalQwen35State::from_q4_dir"
            ) {
                if self.closure_depth != 1 || self.async_depth != 0 || self.function_depth != 1 {
                    self.error =
                        Some("Metal construction was deferred beyond its worker loader".into());
                }
                self.constructions.push(call.clone());
            }
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method.to_string().starts_with("submit") {
            if self.closure_depth != 0 || self.async_depth != 0 || self.function_depth != 1 {
                self.error =
                    Some("worker submission crosses an unreviewed deferred boundary".into());
            }
            self.submissions.push(call.clone());
        }
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.path.is_ident("client") {
            self.client_paths.push(path.span());
        }
        if matches!(
            path_label(&path.path).as_str(),
            "MetalQwen35State::new" | "MetalQwen35State::from_q4_dir"
        ) {
            self.constructor_paths += 1;
        }
        syn::visit::visit_expr_path(self, path);
    }

    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        if pattern.ident == "client" {
            self.client_bindings += 1;
        }
        if matches!(
            pattern.ident.to_string().as_str(),
            "spawn_with_measurement_lock" | "acquire_guard"
        ) {
            self.error = Some("worker entrypoint was rebound".into());
        }
        syn::visit::visit_pat_ident(self, pattern);
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        self.closure_depth += 1;
        self.async_depth += usize::from(closure.asyncness.is_some());
        syn::visit::visit_expr_closure(self, closure);
        self.async_depth -= usize::from(closure.asyncness.is_some());
        self.closure_depth -= 1;
    }

    fn visit_expr_async(&mut self, expression: &'ast syn::ExprAsync) {
        self.async_depth += 1;
        syn::visit::visit_expr_async(self, expression);
        self.async_depth -= 1;
    }

    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        self.function_depth += 1;
        syn::visit::visit_item_fn(self, function);
        self.function_depth -= 1;
    }

    fn visit_macro(&mut self, expression: &'ast syn::Macro) {
        if macro_tokens_name_protected_work(&expression.tokens) {
            self.error = Some("opaque macro hides worker construction".into());
        }
        syn::visit::visit_macro(self, expression);
    }
}

fn peel_initializer(expression: &syn::Expr) -> &syn::Expr {
    match expression {
        syn::Expr::Try(value) => peel_initializer(&value.expr),
        syn::Expr::MethodCall(value) if value.method == "map_err" => {
            peel_initializer(&value.receiver)
        }
        expression => expression,
    }
}

fn validate_benchmark(parsed: &StructuredSource) -> Result<(), String> {
    let syntax = syn::parse_file(&parsed.source).map_err(|error| error.to_string())?;
    let mut census = BenchmarkWork {
        spawns: Vec::new(),
        constructions: Vec::new(),
        submissions: Vec::new(),
        client_paths: Vec::new(),
        client_bindings: 0,
        constructor_paths: 0,
        closure_depth: 0,
        async_depth: 0,
        function_depth: 0,
        error: None,
    };
    census.visit_file(&syntax);
    if let Some(error) = census.error {
        return Err(error);
    }
    if census.spawns.len() != 1
        || census.constructions.len() != 2
        || census.submissions.len() != 1
        || census.client_paths.len() != 1
        || census.client_bindings != 1
        || census.constructor_paths != 2
    {
        return Err("worker spawn/construction/submission or client-use census changed".into());
    }
    let functions = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function)
                if function.sig.ident == "run_metal"
                    && function
                        .block
                        .span()
                        .byte_range()
                        .contains(&census.spawns[0].span().byte_range().start) =>
            {
                Some(function)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if functions.len() != 1 || functions[0].sig.asyncness.is_some() {
        return Err("worker benchmark must have one synchronous run_metal owner".into());
    }
    let function = functions[0];
    let bindings = function
        .block
        .stmts
        .iter()
        .filter_map(|statement| match statement {
            syn::Stmt::Local(local) if matches!(&local.pat, syn::Pat::Tuple(_)) => Some(local),
            _ => None,
        })
        .filter(|local| {
            same_syntax(
                &parsed.source,
                local.pat.span(),
                "(owner, client, meta, preparation, load)",
            )
            .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if bindings.len() != 1 || !bindings[0].attrs.is_empty() {
        return Err("worker must be bound directly in run_metal".into());
    }
    let init = bindings[0]
        .init
        .as_ref()
        .ok_or("missing worker initializer")?;
    let syn::Expr::Call(spawn) = peel_initializer(&init.expr) else {
        return Err("worker initializer is not a direct call".into());
    };
    let syn::Expr::Path(path) = &*spawn.func else {
        return Err("aliased worker entrypoint".into());
    };
    if path_label(&path.path) != SPAWN
        || spawn.args.len() != 3
        || spawn.span().byte_range() != census.spawns[0].span().byte_range()
    {
        return Err("benchmark bypasses the guarded worker entrypoint".into());
    }
    let syn::Expr::Call(factory) = &spawn.args[0] else {
        return Err("worker factory escaped".into());
    };
    let syn::Expr::Path(factory_path) = &*factory.func else {
        return Err("aliased factory".into());
    };
    if path_label(&factory_path.path) != "ServingFactory::qwen_metal" || factory.args.len() != 2 {
        return Err("unexpected worker factory".into());
    }
    validate_factory_namespace(parsed, &syntax, function, &factory_path.path)?;
    let syn::Expr::Closure(loader) = &factory.args[0] else {
        return Err("worker loader escaped".into());
    };
    if loader.asyncness.is_some() || loader.capture.is_none() || !loader.inputs.is_empty() {
        return Err("worker loader must be an inline synchronous move closure".into());
    }
    let range = loader.body.span().byte_range();
    if census.constructions.iter().any(|call| {
        let site = call.span().byte_range();
        site.start < range.start || site.end > range.end
    }) {
        return Err("Metal construction escaped the worker-owned loader".into());
    }
    let submit = &census.submissions[0];
    if submit.method != "submit_with_lora"
        || !same_syntax(&parsed.source, submit.receiver.span(), "client")?
        || !function
            .block
            .span()
            .byte_range()
            .contains(&submit.span().byte_range().start)
    {
        return Err("submission no longer uses the guarded worker client".into());
    }
    let spawn_token = parsed
        .tokens
        .iter()
        .position(|token| {
            token.offset
                == path
                    .path
                    .segments
                    .last()
                    .unwrap()
                    .ident
                    .span()
                    .byte_range()
                    .start
        })
        .ok_or("worker call token missing")?;
    if !parsed.formula_at(spawn_token)?.satisfiable()? {
        return Err("worker benchmark was cfg-elided".into());
    }
    if parsed
        .tokens
        .iter()
        .any(|token| token.ident() == Some("gpu_test_lock"))
    {
        return Err("caller lock would deadlock worker lock acquisition".into());
    }
    Ok(())
}

pub(super) fn validate_target(sources: &[ParsedModuleSource]) -> Result<(), String> {
    if sources.len() != 1 || sources[0].relative != TARGET {
        return Err("worker benchmark module closure changed".into());
    }
    validate_benchmark(&sources[0].parsed)?;
    let worker = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(WORKER))
        .map_err(|error| error.to_string())?;
    validate_worker_source(&worker)?;
    let factory = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(FACTORY))
        .map_err(|error| error.to_string())?;
    validate_factory_source(&factory)
}

#[test]
fn worker_measurement_contract_rejects_lifetime_and_entrypoint_mutations() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(root.join(WORKER)).unwrap();
    validate_worker_source(&source).unwrap();
    for (from, to) in [
        (
            "let _measurement_guard = acquire_guard();",
            "let _measurement_guard = ();",
        ),
        ("work();", "drop(_measurement_guard); work();"),
        (
            "let guard = crate::measurement::gpu_test_lock();",
            "let guard = ();",
        ),
        (
            "run_guarded_worker(acquire_guard,",
            "run_guarded_worker(|| (),",
        ),
        (
            "let join_handle = std::thread::spawn(move || {\n            run_guarded_worker",
            "fn run_guarded_worker<G>(_: impl FnOnce() -> G, work: impl FnOnce()) { work(); }\n        let join_handle = std::thread::spawn(move || {\n            run_guarded_worker",
        ),
    ] {
        assert!(source.contains(from), "stale worker mutation: {from}");
        assert!(
            validate_worker_source(&source.replacen(from, to, 1)).is_err(),
            "accepted {from}"
        );
    }
    let factory = std::fs::read_to_string(root.join(FACTORY)).unwrap();
    validate_factory_source(&factory).unwrap();
    let eager = factory.replacen(
        "loader: Box::new(loader),",
        "loader: Box::new({ let result = loader(); move || result }),",
        1,
    );
    assert_ne!(eager, factory);
    assert!(validate_factory_source(&eager).is_err());
    let source = std::fs::read_to_string(root.join(TARGET)).unwrap();
    validate_benchmark(&StructuredSource::parse(TARGET, &source, false).unwrap()).unwrap();
    for (from, to) in [
        (SPAWN, "MetalWorker::spawn_with_vision"),
        (".submit_with_lora(", ".submit_with_lora_mode("),
        ("drop(owner);", "drop(owner); let escaped = client.clone();"),
        ("drop(owner);", "drop(owner); let client = other_client();"),
        (
            "use lattice_inference::serving_factory::ServingFactory;",
            "use crate::eager::ServingFactory;",
        ),
        (
            "let binary = match route",
            "struct ServingFactory; let binary = match route",
        ),
        (
            "let binary = match route",
            "use other::*; let binary = match route",
        ),
        (
            "let binary = match route",
            "use other as lattice_inference; let binary = match route",
        ),
        (
            "client\n                .submit_with_lora(messages, cfg, cancel_rx, Vec::new())",
            "(|| client.submit_with_lora(messages, cfg, cancel_rx, Vec::new()))()",
        ),
        (
            "MetalQwen35State::new(model.weights(), &cfg, context)",
            "(|| MetalQwen35State::new(model.weights(), &cfg, context))()",
        ),
        (
            "let binary = match route",
            "let _guard = lattice_inference::measurement::gpu_test_lock(); let binary = match route",
        ),
    ] {
        assert!(source.contains(from), "stale benchmark mutation: {from}");
        let mutated = source.replacen(from, to, 1);
        assert!(
            validate_benchmark(&StructuredSource::parse(TARGET, &mutated, false).unwrap()).is_err(),
            "accepted {from}"
        );
    }
}
